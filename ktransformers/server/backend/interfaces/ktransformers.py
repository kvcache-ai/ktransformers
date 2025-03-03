import torch
import asyncio
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
from typing import Optional
from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled, MLAWrapperSingleton

warm_uped = False

class KTransformersThreadContext(TransformersThreadContext):
    pass


class KTransformersInterface(TransformersInterface):
    def __init__(self, args: ConfigArgs = default_args):
        self.args = args
        torch.set_grad_enabled(False)
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir, device=args.device, trust_remote_code=args.trust_remote_code)
        config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=args.trust_remote_code)
        try:
            generation_config = GenerationConfig.from_pretrained(args.model_dir)
        except:
            generation_config = GenerationConfig(
                max_length=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True
            )
        
        torch.set_default_dtype(config.torch_dtype)
        if config.architectures[0] == "Qwen2MoeForCausalLM":
            config._attn_implementation = "flash_attention_2"

        with torch.device("meta"):
            self.model = custom_models[config.architectures[0]](config)
        if default_args.optimize_config_path is None:
            optimize_config_path = default_optimize_rules[config.architectures[0]]
        else:
            optimize_config_path = args.optimize_config_path

        # print(optimize_config)

        gguf_path = args.gguf_path
        if gguf_path is None:
            gguf_path = input(
                "please input the path of your gguf file(gguf file in the dir containing input gguf file must all"
                " belong to current model):"
            )
        optimize_and_load_gguf(self.model, optimize_config_path, gguf_path, config)
        self.model.generation_config = generation_config
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

        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id
        self.streamer = TextStreamer(self.tokenizer)

        self._infer_lock = asyncio.Lock()

    def decode_one_tokens(self):
        global warm_uped

        device_map = self.model.gguf_loader.tensor_device_map
        torch_device = get_device("blk.0.self_attn", device_map)
        torch_device = "cuda:0" if torch_device == "cuda" else torch_device
        torch.cuda.set_device(torch_device)
        if warm_uped and self.args.use_cuda_graph:
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
        
        if self.args.use_cuda_graph:
            warm_uped = True
            
        if self.use_static_cache:
            logits = self.model(
                self.current_ids.to(torch_device),
                cache_position=self.active_cache_position,
                past_key_values=self.cache,
                return_dict=False,
                use_cache=True,
            )[0]
        else:
            logits = self.model(self.current_ids, return_dict=False)[0]
        logits = logits[0, -1, :]

        return self.logits_to_token(logits)



    @torch.no_grad
    def prefill(self, input_ids: torch.Tensor, is_new: bool, temperature: Optional[float], top_p: Optional[float]):
        input_ids_length = input_ids.shape[-1]
        if(input_ids_length >= self.args.cache_lens):
            logger.warning(f"input_ids_length {input_ids_length} > cache_lens {self.args.cache_lens}")
            self.seq_length = input_ids_length
            return
        logger.debug(f"input_ids: {input_ids.shape}")
        device = self.device_map.get("blk.0.self_attn", {}).get("generate_device", "cuda:0")
        device = "cuda:0" if device == "cuda" else device

        if is_new:
            self.ever_generated_ids.clear()
            same_prefix = 0
            flat_input_ids = input_ids.flatten()

            if getattr(self, 'generated_ids', None) is None:
                self.generated_ids = torch.zeros(
                    self.args.batch_size,
                    input_ids.shape[-1] + self.args.max_new_tokens + 1,
                    dtype=torch.int,
                    device=self.args.device,
                )
                self.seq_length = 1            
            
            flat_prev_ids = self.generated_ids.flatten()
            for i in range(min(self.seq_length, flat_input_ids.shape[0]) - 1):
                if flat_input_ids[i] == flat_prev_ids[i]:
                    same_prefix += 1
                else:
                    break
            
            logger.debug(f"same prefix len: {same_prefix}")
            self.cache.remove_suffix(same_prefix)
            self.seq_length = same_prefix
            self.generated_ids = self.generated_ids[..., :same_prefix]
            input_ids = input_ids[..., same_prefix:]
            input_ids_length = input_ids.shape[-1]

        self.ever_generated_ids.clear()
        self.profiler.set_counter("prefill", input_ids_length)
        logger.debug(f"input_ids: {input_ids.shape}")
        logger.debug(f"generate_ids: {self.generated_ids.shape}")
        
        former_seq_length = self.seq_length
        self.seq_length += input_ids_length
        expected_length = min(self.seq_length + self.args.max_new_tokens + 1, self.args.cache_lens)
        delta_length = expected_length - self.generated_ids.shape[-1]
        if delta_length > 0:
            new_generate_ids = torch.zeros(
                self.args.batch_size, delta_length, dtype=torch.int, device=self.args.device
            )
            self.generated_ids = torch.cat([self.generated_ids, new_generate_ids], dim=-1)
        else:
            logger.warning(f"seq_length bigger than cache_lens, killed")
            exit(0)
        
        logger.debug(f"cache position: {former_seq_length} to {self.seq_length}")
        cache_position = torch.arange(former_seq_length, self.seq_length, device=device)
        self.generated_ids[:, cache_position] = input_ids.to(self.args.device).to(torch.int)

        if not (type(self) is TransformersInterface):
            input_ids = input_ids.to("cpu")
        
        def chunk_prefill(input_ids, cache_position):
            inputs_embeds = self.model.model.embed_tokens(input_ids).to(device)
            torch.cuda.set_device(device)
            if flashinfer_enabled:
                MLAWrapperSingleton.need_plan_all()
            if self.use_static_cache:
                logits = self.model(
                    inputs_embeds=inputs_embeds,
                    cache_position=cache_position,
                    past_key_values=self.cache,
                    return_dict=False,
                    use_cache=True,
                )[0]
            else:
                logits = self.model(inputs_embeds=inputs_embeds, return_dict=False)[0]

            return logits

        chunk_start = 0
        while chunk_start < input_ids_length:
            chunk_end = min(chunk_start + self.args.chunk_prefill_size, input_ids_length)
            if self.cache != None:
                self.cache.cur_idx=cache_position[chunk_start:chunk_end]
            logits = chunk_prefill(input_ids[:, chunk_start:chunk_end], cache_position[chunk_start:chunk_end])
            chunk_start += self.args.chunk_prefill_size
            
        if flashinfer_enabled:
            MLAWrapperSingleton.reset_buffer()
        self.prepare_logits_wrapper(input_ids, device, temperature, top_p)
        next_token = self.logits_to_token(logits[0, -1, :])
        yield self.append_new_tokens(next_token)

    @property
    def active_cache_position(self):
        device = self.device_map.get("blk.0.self_attn", {}).get("generate_device", "cuda:0")
        return torch.tensor([self.seq_length - 1], device=device)
    
    async def inference(self, local_messages, thread_id: str, temperature: Optional[float] = None, top_p: Optional[float] = None):
        async with self._infer_lock:
            async for v in super().inference(local_messages, thread_id, temperature, top_p):
                yield v
