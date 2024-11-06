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
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir, device=args.device)
        config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
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

        device_map = self.model.gguf_loader.tensor_device_map
        logger.info(f"{args.model_name} loaded from {args.model_dir} to {device_map}")
        self.cache = StaticCache(
            config=self.model.config,
            max_batch_size=args.batch_size,
            max_cache_len=args.cache_lens,
            device=device_map,
            dtype=self.model.dtype,
        )
        logger.info(f"StaticCache (length={args.cache_lens}) created at {device_map}, batch size:{args.batch_size}")
        self.model.generation_config = GenerationConfig.from_pretrained(args.model_dir)
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        self.streamer = TextStreamer(self.tokenizer)

    def decode_one_tokens(self):
        if not hasattr(self, "cuda_graph_runner"):
            device_map = self.model.gguf_loader.tensor_device_map
            torch_device = get_device("blk.0.self_attn", device_map)
            torch_device = "cuda:0" if torch_device == "cuda" else torch_device
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
                self.current_ids,
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
