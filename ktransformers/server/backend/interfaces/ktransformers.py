import torch
import torch.distributed as dist
from torch import nn
from torch.nn.attention import SDPBackend
import asyncio
from transformers import AutoTokenizer, AutoConfig, GenerationConfig
from ktransformers.server.backend.interfaces.transformers import (
    TransformersInterface,
    ConfigArgs,
    TransformersThreadContext,
    default_args,
    TextStreamer,
)
import os
try:
    import torch_npu
    use_npu = torch.npu.is_available()
    from ktransformers.util.ascend.ascend_utils import get_absort_weight, setup_model_parallel
except:
    use_npu = False
from torch import nn
from ktransformers.server.config.log import logger
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.cuda_graph_runner import CUDAGraphRunner
from ktransformers.local_chat import custom_models, default_optimize_rules
from ktransformers.util.utils import get_device, get_all_used_cuda_device
from ktransformers.util import utils
from typing import Optional
from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled, MLAWrapperSingleton
from ktransformers.server.schemas.endpoints.chat import RawUsage
from typing import Any, List, Optional, Set
from ktransformers.server.config.config import Config

warm_uped = False
speculative_decoding = True # True -> verify by random accept ; False-> verify by token id
global_acc_counts = 0
global_verify_counts = 0

ktransformer_rules_dir = (
    os.path.dirname(os.path.abspath(__file__)) + "/../../../optimize/optimize_rules/"
)
default_optimize_rules = {
    "DeepseekV2ForCausalLM": ktransformer_rules_dir + "DeepSeek-V2-Chat.yaml",
    "DeepseekV3ForCausalLM": ktransformer_rules_dir + "DeepSeek-V3-Chat.yaml",
    "Qwen2MoeForCausalLM": ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct.yaml",
    "LlamaForCausalLM": ktransformer_rules_dir + "Internlm2_5-7b-Chat-1m.yaml",
    "MixtralForCausalLM": ktransformer_rules_dir + "Mixtral.yaml"
}
if use_npu:
    default_optimize_rules["DeepseekV3ForCausalLM"] = ktransformer_rules_dir + "DeepSeek-V3-Chat-npu.yaml"
class KTransformersThreadContext(TransformersThreadContext):
    pass


class KTransformersInterface(TransformersInterface):
    def __init__(self, args: ConfigArgs = default_args, input_args=None):
        self.args = input_args
        self.local_rank, self.world_size = setup_model_parallel(tp=self.args.tp)
        if use_npu and (utils.CUR_DEVICE is None):
            utils.CUR_DEVICE = f"npu:{torch.npu.current_device()}"
            self.args.device = utils.CUR_DEVICE
            self.args.device = f"npu:{torch.npu.current_device()}"
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
        config.backend_type = "ktransformers"
        config.chunk_size = self.args.chunk_size
        with torch.device("meta"):
            self.model = custom_models[config.architectures[0]](config)
        if input_args.optimize_config_path is not None:
            optimize_config_path = input_args.optimize_config_path
        elif default_args.optimize_config_path is None:
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
        optimize_and_load_gguf(self.model, optimize_config_path, gguf_path, config, q4_gguf_path=input_args.q4_gguf_path)
        #提前absorbed
        get_absort_weight(self.model, config)
        # utils.get_absort_weight(self.model, config)
        self.model.eval()
        self.model.generation_config = generation_config
        self.device_map = self.model.gguf_loader.tensor_device_map
        self.top_p = torch.tensor([[self.model.generation_config.top_p]], dtype = torch.float16, device = self.args.device)
        self.top_k = torch.tensor([[self.model.generation_config.top_k]], dtype = torch.int32, device = self.args.device)
        self.temperature = torch.tensor([[self.model.generation_config.temperature]], dtype = torch.float16, device = self.args.device)
        self.next_token_fake = torch.tensor([[1]], dtype=torch.int32, device = self.args.device)
        self.next_token_probs = torch.tensor([[1.0]], dtype=torch.float16, device = self.args.device)
        self.draft_model = None

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

    @torch.no_grad
    def decode_one_tokens(self):
        global warm_uped

        device_map = self.model.gguf_loader.tensor_device_map
        torch_device = get_device("blk.0.self_attn", device_map)
        torch_device = "cuda:0" if torch_device == "cuda" else torch_device
        torch.cuda.set_device(torch_device)
        if warm_uped and self.args.use_cuda_graph:
            if use_npu:
                from ktransformers.util.npu_graph_runner import get_or_create_runner, check_runner
                if check_runner(utils.get_current_device()):
                    npu_graph_runner = get_or_create_runner(utils.get_current_device())
                    npu_graph_runner.init(self.args.batch_size, self.seq_length)
                    self.cuda_graph_runner = npu_graph_runner
                    utils._USE_NPU_GRAPH = True
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
                    inputs_embeds = self.model.model.embed_tokens(self.current_ids.to("cpu")).to(utils.get_current_device())
                    logits = self.cuda_graph_runner(
                        inputs_embeds, self.active_cache_position.unsqueeze(0), self.active_cache_position
                    )[0]
                    self.cache.change_seq_length(1)
                    torch.cuda.synchronize()
                    logits = logits[0, -1, :]
                    return self.logits_to_token(logits)
            else:
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
                is_prefill=False,
            )[0]
        else:
            logits = self.model(self.current_ids, return_dict=False, is_prefill=False)[0]
        logits = logits[0, -1, :]

        return self.logits_to_token(logits)


    @torch.no_grad
    def prefill(self, input_ids: torch.Tensor, is_new: bool, temperature: Optional[float] = None, top_p: Optional[float] = None, max_tokens: Optional[float] = None, max_completion_tokens: Optional[float] = None):
        input_ids_length = input_ids.shape[-1]
        if max_tokens is not None:
            max_completion_tokens = max_tokens
        if max_completion_tokens is None:
            max_new_tokens = self.args.max_new_tokens
        else:
            max_new_tokens = min(self.args.max_new_tokens, max_completion_tokens)
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
            # flat_input_ids = input_ids.flatten()

            if getattr(self, 'generated_ids', None) is None:
                self.generated_ids = torch.zeros(
                    self.args.batch_size,
                    input_ids.shape[-1] + max_new_tokens + 1,
                    dtype=torch.int,
                    device=self.args.device,
                )
                self.seq_length = 1
            
            # flat_prev_ids = self.generated_ids.flatten()
            # for i in range(min(self.seq_length, flat_input_ids.shape[0]) - 1):
            #     if flat_input_ids[i] == flat_prev_ids[i]:
            #         same_prefix += 1
            #     else:
            #         break

            logger.debug(f"same prefix len: {same_prefix}")
            self.cache.remove_suffix(same_prefix)
            self.seq_length = same_prefix
            self.cache.position[0] = same_prefix
            self.generated_ids = self.generated_ids[..., :same_prefix]
            input_ids = input_ids[..., same_prefix:]
            input_ids_length = input_ids.shape[-1]

        self.ever_generated_ids.clear()
        self.profiler.set_counter("prefill", input_ids_length)
        logger.debug(f"input_ids: {input_ids.shape}")
        logger.debug(f"generate_ids: {self.generated_ids.shape}")
        
        former_seq_length = self.seq_length
        self.seq_length += input_ids_length
        expected_length = min(self.seq_length + max_new_tokens + 1, self.args.cache_lens)
        delta_length = expected_length - self.generated_ids.shape[-1]
        if delta_length > 0:
            new_generate_ids = torch.zeros(
                self.args.batch_size, delta_length, dtype=torch.int, device=utils.get_current_device()
            )
            self.generated_ids = torch.cat([self.generated_ids, new_generate_ids], dim=-1)
        else:
            logger.warning(f"seq_length bigger than cache_lens, killed")
            exit(0)
        
        logger.debug(f"cache position: {former_seq_length} to {self.seq_length}")
        cache_position = torch.arange(former_seq_length, self.seq_length, device=device)
        self.generated_ids[:, cache_position] = input_ids.to(utils.get_current_device()).to(torch.int)

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
                    is_prefill=True,
                )[0]
            else:
                logits = self.model(inputs_embeds=inputs_embeds, return_dict=False, is_prefill=True)[0]

            return logits

        if not use_npu:
            chunk_start = 0
            while chunk_start < input_ids_length:
                chunk_end = min(chunk_start + self.args.chunk_size, input_ids_length)
                if self.cache != None:
                    self.cache.cur_idx=cache_position[chunk_start:chunk_end]
                logits = chunk_prefill(input_ids[:, chunk_start:chunk_end], cache_position[chunk_start:chunk_end])
                chunk_start += self.args.chunk_size
                
            if flashinfer_enabled:
                MLAWrapperSingleton.reset_buffer()
            self.prepare_logits_wrapper(input_ids, device, temperature, top_p)
            next_token = self.logits_to_token(logits[0, -1, :])
            self.max_new_tokens = min(max_new_tokens, self.args.cache_lens - self.seq_length) - 1 
            yield self.append_new_tokens(next_token)
            return

        def prefill_wrapper(prof=None):
            chunk_start = 0
            while chunk_start < input_ids_length:
                chunk_end = min(chunk_start + self.args.chunk_size, input_ids_length)
                if self.cache != None:
                    self.cache.cur_idx = cache_position[chunk_start:chunk_end]
                logits = chunk_prefill(input_ids[:, chunk_start:chunk_end], cache_position[chunk_start:chunk_end])
                chunk_start += self.args.chunk_size
                if prof is not None:
                    prof.step()
            if prof is not None:
                prof.stop()
            if logits is None:
                raise ValueError('logits cannot be None')
            return logits

        global WARM_UP_SKIP_CNT
        prof_prefill = os.environ["PROF_PREFILL"] if "PROF_PREFILL" in os.environ else "0"
        if prof_prefill == "1":
            experimental_config = torch_npu.profiler._ExperimentalConfig(
                aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
            )
            with torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU
                    ],
                    schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=8, repeat=1, skip_first=0),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./prefill_prof_lm_head"),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                    with_flops=False,
                    with_modules=False,
                    experimental_config=experimental_config) as prof:
                logits = prefill_wrapper(prof)
        else:
            logits = prefill_wrapper()
            
        if flashinfer_enabled:
            MLAWrapperSingleton.reset_buffer()
        self.prepare_logits_wrapper(input_ids, device, temperature, top_p)
        next_token = self.logits_to_token(logits[0, -1, :])
        self.cache.position[0] = self.seq_length
        yield self.append_new_tokens(next_token)

    @property
    def active_cache_position(self):
        device = self.device_map.get("blk.0.self_attn", {}).get("generate_device", "cuda:0")
        return torch.tensor([self.seq_length - 1], device=device)
    
    def sampling(self, logits, do_sample):
        if do_sample:
            cur_len = logits.shape[1]
            logits = logits / self.temperature
            torch.manual_seed(0)
            probs = logits.view(-1, cur_len, self.model.config.vocab_size)
            probs = torch.softmax(probs, dim=-1).half()
            next_token = self.next_token_fake
            if self.draft_model is None or not speculative_decoding:
                torch_npu._npu_topk_topp_sampling(probs[:, 0, :], self.top_k, self.top_p, next_token, self.next_token_probs)
            for i in range(1,cur_len):
                ith_token = torch.empty_like(self.next_token_fake)
                torch_npu._npu_topk_topp_sampling(probs[:, i, :], self.top_k, self.top_p, ith_token, self.next_token_probs)
                next_token = torch.cat((next_token, ith_token), dim=-1)
        else:
            next_token = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)

        return next_token, probs

    def verify_by_tokenid(self, main_token: int, draft_token: int):
        return main_token, main_token == draft_token

    def verify_speculative_decoding(self, main_prob: torch.Tensor, draft_prob: torch.Tensor, draft_token: int, p: float):
        #assert draft_prob[draft_token] == p
        q = main_prob[draft_token]
        #p = draft_prob[draft_token]
        accept_prob = min(1.0, (q / p).item())
        if torch.rand(()) <= accept_prob:
            return draft_token, True
        else:
            # Compute the adjusted distribution for resampling
            new_prob = main_prob - draft_prob
            new_prob = torch.clamp(new_prob, min=0.0)
            new_prob /= new_prob.sum()

            # Sample a new token from the adjusted distribution
            token = torch.multinomial(new_prob, 1).item()
            return token, False

    def logits_to_token(self, logits: torch.Tensor):
        if self.model.generation_config.do_sample:
            logits = self.logits_warper(self.inputs.view(1, -1), logits.view(1, -1))
            probs = torch.nn.functional.softmax(logits, dim=-1)
            last = torch.multinomial(probs, num_samples=1)
        else:
            logits = self.logits_warper(self.inputs.view(1, -1), logits.view(1, -1))
            probs = torch.nn.functional.softmax(logits, dim=-1)
            _, last = torch.topk(probs, k=1, dim=-1)
        last = last.item()
        self.ever_generated_ids.add(last)
        return last

    async def inference(self, local_messages, thread_id: str, temperature: Optional[float] = None, top_p: Optional[float] = None, max_tokens: Optional[float] = None, max_completion_tokens: Optional[float] = None):
        async with self._infer_lock:
            async for v in super().inference(local_messages, thread_id, temperature, top_p, max_tokens, max_completion_tokens):
                yield v
            
            # return this inference raw usage
            yield RawUsage(
                tokenize_time = self.profiler.get_timer_sec('tokenize'),
                prefill_time = self.profiler.get_timer_sec('prefill'),
                decode_time = self.profiler.get_timer_sec('decode'),
                prefill_count = self.profiler.get_counter('prefill'),
                decode_count = self.profiler.get_counter('decode'),
            )

    def sync_inference(self, local_messages, thread_id: str, temperature: Optional[float] = None, top_p: Optional[float] = None) -> str:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            async def run_async():
                result = []
                async for chunk in self.inference(local_messages, thread_id, temperature, top_p):
                    pass
                return ""
            return loop.run_until_complete(run_async())
        finally:
            loop.close()
