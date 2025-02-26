import torch
from transformers import AutoTokenizer, AutoConfig, GenerationConfig
from ktransformers.server.backend.interfaces.transformers import (
    TransformersInterface,
    ConfigArgs,
    TransformersThreadContext,
    default_args,
    TextStreamer,
)
from ktransformers.server.config.log import logger
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.cuda_graph_runner import CUDAGraphRunner
from ktransformers.local_chat import custom_models, default_optimize_rules
from ktransformers.util.utils import get_device


class KTransformersThreadContext(TransformersThreadContext):
    pass


class KTransformersInterface(TransformersInterface):
    def __init__(self, args: ConfigArgs = default_args):
        self.args = args
        torch.set_default_dtype(torch.bfloat16)
        torch.set_grad_enabled(False)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir, device=args.device, trust_remote_code=args.trust_remote_code)
        config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=args.trust_remote_code)
        if config.architectures[0] == "Qwen2MoeForCausalLM":
            config._attn_implementation = "flash_attention_2"

        with torch.device("meta"):
            self.model = custom_models[config.architectures[0]](config)
        if default_args.optimize_config_path is None:
            optimize_rule_path = default_optimize_rules[config.architectures[0]]
        else:
            optimize_rule_path = args.optimize_config_path

        # print(optimize_config)

        gguf_path = args.gguf_path
        if gguf_path is None:
            gguf_path = input(
                "please input the path of your gguf file(gguf file in the dir containing input gguf file must all"
                " belong to current model):"
            )
        optimize_and_load_gguf(self.model, optimize_rule_path, gguf_path, config)

        self.device_map = self.model.gguf_loader.tensor_device_map
        # logger.info(f"{args.model_name} loaded from {args.model_dir} to {self.device_map}")
        self.cache = StaticCache(
            config=self.model.config,
            max_batch_size=args.batch_size,
            max_cache_len=args.cache_lens,
            device=self.device_map,
            dtype=self.model.dtype,
        )
        # logger.info(f"StaticCache (length={args.cache_lens}), batch size:{args.batch_size}")
        try:
            self.model.generation_config = GenerationConfig.from_pretrained(args.model_dir)
        except:
            gen_config = GenerationConfig(
                max_length=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            self.model.generation_config = gen_config
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        self.streamer = TextStreamer(self.tokenizer)

    def decode_one_tokens(self):
        device_map = self.model.gguf_loader.tensor_device_map
        torch_device = get_device("blk.0.self_attn", device_map)
        torch_device = "cuda:0" if torch_device == "cuda" else torch_device
        if self.args.use_cuda_graph:
            if not hasattr(self, "cuda_graph_runner"):
                self.cuda_graph_runner = CUDAGraphRunner()
                self.cuda_graph_runner.capture(
                    self.model,
                    self.current_ids,
                    self.active_cache_position.unsqueeze(0),
                    self.active_cache_position,
                    self.cache,
                    main_device=torch_device,
                    return_dict=False,
                    use_cache=True,
                )

            if hasattr(self, "cuda_graph_runner"):
                logits = self.cuda_graph_runner(
                    self.current_ids, self.active_cache_position.unsqueeze(0), self.active_cache_position
                )
                self.cache.change_seq_length(1)
                torch.cuda.synchronize()
                logits = logits[0, -1, :]
                return self.logits_to_token(logits)

        if self.use_static_cache:
            mask = torch.ones((1, self.seq_length)).to(torch_device)
            logits = self.model(
                self.current_ids.to(torch_device),
                cache_position=self.active_cache_position,
                past_key_values=self.cache,
                attention_mask=mask,
                return_dict=False,
                use_cache=True,
            )[0]
        else:
            logits = self.model(self.current_ids, return_dict=False)[0]
        logits = logits[0, -1, :]

        return self.logits_to_token(logits)



    @torch.no_grad
    def prefill(self, input_ids: torch.Tensor, is_new: bool):
        input_ids_length = input_ids.shape[-1]
        self.profiler.set_counter("prefill", input_ids_length)
        logger.debug(f"input_ids: {input_ids.shape}")

        device = self.device_map.get("blk.0.self_attn", {}).get("generate_device", "cuda:0")

        if is_new:
            self.cache.reset()
            self.ever_generated_ids.clear()
            former_seq_length = 0
            self.seq_length = input_ids_length
            self.generated_ids = torch.zeros(
                self.args.batch_size,
                self.seq_length + self.args.max_new_tokens + 1,
                dtype=torch.int,
                device=self.args.device,
            )
        else:
            logger.debug(f"generate_ids: {self.generated_ids.shape}")
            former_seq_length = self.seq_length
            self.seq_length += input_ids_length
            expected_length = self.seq_length + self.args.max_new_tokens + 1
            delta_length = expected_length - self.generated_ids.shape[-1]
            if delta_length > 0:
                new_generate_ids = torch.zeros(
                    self.args.batch_size, delta_length, dtype=torch.int, device=self.args.device
                )
                self.generated_ids = torch.cat([self.generated_ids, new_generate_ids], dim=-1)
        logger.debug(f"cache position: {former_seq_length} to {self.seq_length}")
        cache_position = torch.arange(former_seq_length, self.seq_length, device=device)
        self.generated_ids[:, cache_position] = input_ids.to(self.args.device).to(torch.int)

        mask = torch.ones((1, self.seq_length)).to(device)
        if not (type(self) is TransformersInterface):
            input_ids = input_ids.to("cpu")
        inputs_embeds = self.model.model.embed_tokens(input_ids).to(device)
        if self.use_static_cache:
            logits = self.model(
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                past_key_values=self.cache,
                return_dict=False,
                use_cache=True,
                attention_mask=mask,
            )[0]
        else:
            logits = self.model(inputs_embeds=inputs_embeds, return_dict=False)[0]

        next_token = self.logits_to_token(logits[0, -1, :])
        yield self.append_new_tokens(next_token)

    @property
    def active_cache_position(self):
        device = self.device_map.get("blk.0.self_attn", {}).get("generate_device", "cuda:0")
        return torch.tensor([self.seq_length - 1], device=device)