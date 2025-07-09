import torch
from typing import Optional, List
import asyncio
from transformers import AutoTokenizer, AutoConfig, GenerationConfig
from ktransformers.server.backend.interfaces.transformers import (
    TransformersInterface,
    ConfigArgs,
    TransformersThreadContext,
    default_args,
)
from ktransformers.server.config.config import Config
from ktransformers.server.config.log import logger
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.custom_cache import StaticCache
from ktransformers.util.cuda_graph_runner import CUDAGraphRunner
from ktransformers.local_chat import custom_models, default_optimize_rules
from ktransformers.util.utils import get_device
from typing import Optional
from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled, MLAWrapperSingleton
from ktransformers.server.schemas.endpoints.chat import RawUsage
from torch.nn.attention import SDPBackend
warm_uped = False
multi_batch_enabled = False

class KTransformersThreadContext(TransformersThreadContext):
    pass

class MultiBatchTextStreamer:

    def __init__(self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, **decode_kwargs):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs

        # variables used in the streaming process for each batch
        self.token_caches = {}  # {batch_index: [tokens]}
        self.print_lens = {}    # {batch_index: print_len}
        self.next_tokens_are_prompt = {}  # {batch_index: bool}

    def reset(self, batch_index: int = 0):
        self.token_caches[batch_index] = []
        self.print_lens[batch_index] = 0
        self.next_tokens_are_prompt[batch_index] = True

    def reset_all(self):
        self.token_caches.clear()
        self.print_lens.clear()
        self.next_tokens_are_prompt.clear()

    def put(self, value, batch_index: int = 0) -> Optional[str]:
        """
        Receives tokens for a specific batch, decodes them, and returns printable text.
        """
        if not isinstance(value, int):
            raise ValueError("MultiBatchTextStreamer only supports int type input")

        # Initialize batch if not exists
        if batch_index not in self.token_caches:
            self.reset(batch_index)

        if self.skip_prompt and self.next_tokens_are_prompt[batch_index]:
            self.next_tokens_are_prompt[batch_index] = False
            return None

        # Add the new token to the cache and decodes the entire thing.
        self.token_caches[batch_index].append(value)
        text = self.tokenizer.decode(self.token_caches[batch_index], skip_special_tokens=True, **self.decode_kwargs)

        # After the symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_lens[batch_index] :]
            self.reset(batch_index)
        # If the last token is a CJK character, we print the characters.
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_lens[batch_index] :]
            self.print_lens[batch_index] += len(printable_text)
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_lens[batch_index] : text.rfind(" ") + 1]
            self.print_lens[batch_index] += len(printable_text)
        return printable_text

    def end(self, batch_index: int = 0) -> Optional[str]:
        """Flushes any remaining cache for a specific batch and returns printable text."""
        if batch_index not in self.token_caches:
            return ""
            
        # Flush the cache, if it exists
        if len(self.token_caches[batch_index]) > 0:
            text = self.tokenizer.decode(self.token_caches[batch_index], skip_special_tokens=True, **self.decode_kwargs)
            printable_text = text[self.print_lens[batch_index] :]
            self.reset(batch_index)
        else:
            printable_text = ""

        self.next_tokens_are_prompt[batch_index] = True
        return printable_text

    def end_all(self) -> List[Optional[str]]:
        """Flushes all batches and returns a list of printable texts."""
        results = []
        for batch_index in sorted(self.token_caches.keys()):
            results.append(self.end(batch_index))
        return results[0]

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (
            (cp >= 0x4E00 and cp <= 0x9FFF)
            or (cp >= 0x3400 and cp <= 0x4DBF)  #
            or (cp >= 0x20000 and cp <= 0x2A6DF)  #
            or (cp >= 0x2A700 and cp <= 0x2B73F)  #
            or (cp >= 0x2B740 and cp <= 0x2B81F)  #
            or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
            or (cp >= 0xF900 and cp <= 0xFAFF)
            or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        ):  #
            return True

        return False

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
        self.tokenizer.pad_token_id = 0
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
        if args.batch_size > 1:
            global multi_batch_enabled
            multi_batch_enabled = True
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
        self.streamer = MultiBatchTextStreamer(self.tokenizer)
        self._infer_lock = asyncio.Lock()
        self._inference_queue = asyncio.Queue()
        self._batch_worker_task = None

    def append_new_tokens(self, new_tokens: int, batch_idx: int) -> Optional[str]:
        self.generated_ids[batch_idx, self.seq_length] = new_tokens
        return self.streamer.put(new_tokens, batch_idx)

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
                    attention_mask=self.attention_mask,
                )

            if hasattr(self, "cuda_graph_runner"):
                logits = self.cuda_graph_runner(
                    self.current_ids, self.active_cache_position.unsqueeze(0), self.active_cache_position
                )
                self.cache.change_seq_length(1)
                torch.cuda.synchronize()
                tokens=[]
                for batch_idx in range(logits.size(0)):
                    logit = logits[batch_idx, -1, :]  # [batch_size, vocab_size]
                    tokens.append(self.logits_to_token(logit))
                self.update_mask(tokens)
                return tokens
        
        if self.args.use_cuda_graph:
            warm_uped = True
            
        if self.use_static_cache:
            logits = self.model(
                self.current_ids.to(torch_device),
                cache_position=self.active_cache_position,
                past_key_values=self.cache,
                return_dict=False,
                use_cache=True,
                attention_mask=self.attention_mask,
            )[0]
        else:
            logits = self.model(self.current_ids, return_dict=False)[0]

        tokens=[]
        for batch_idx in range(logits.size(0)):
            logit = logits[batch_idx, -1, :]  # [batch_size, vocab_size]
            tokens.append(self.logits_to_token(logit))
        self.update_mask(tokens)
        return tokens

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

            self.generated_ids = torch.zeros(
                input_ids.size(0),
                input_ids.shape[-1] + max_new_tokens + 1,
                dtype=torch.int,
                device=self.args.device,
            )
            self.seq_length = 1

            same_prefix = self.seq_length
            for i in range(input_ids.size(0)):
                cur_same_prefix = 0
                flat_input_ids = input_ids[i].flatten()
                flat_prev_ids = self.generated_ids[i].flatten()
                for j in range(min(self.seq_length, flat_input_ids.shape[0]) - 1):
                    if flat_input_ids[j] == flat_prev_ids[j]:
                        cur_same_prefix += 1
                    else:
                        break
                same_prefix = min(same_prefix, cur_same_prefix)

            logger.debug(f"same prefix len: {same_prefix}")
            self.cache.remove_suffix(same_prefix)
            self.seq_length = same_prefix
            self.generated_ids = self.generated_ids[..., :same_prefix]
            input_ids = input_ids[..., same_prefix:]
            input_ids_length = input_ids.shape[-1]

        self.ever_generated_ids.clear()
        self.profiler.set_counter("prefill", input_ids.numel())
        logger.debug(f"input_ids: {input_ids.shape}")
        logger.debug(f"generate_ids: {self.generated_ids.shape}")
        
        former_seq_length = self.seq_length
        self.seq_length += input_ids_length
        expected_length = min(self.seq_length + max_new_tokens + 1, self.args.cache_lens)
        delta_length = expected_length - self.generated_ids.shape[-1]
        if delta_length > 0:
            new_generate_ids = torch.zeros(
                input_ids.size(0), delta_length, dtype=torch.int, device=self.args.device
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
                    attention_mask=self.attention_mask,
                )[0]
            else:
                logits = self.model(inputs_embeds=inputs_embeds, return_dict=False)[0]

            return logits

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
        self.max_new_tokens = min(max_new_tokens, self.args.cache_lens - self.seq_length) - 1
        next_tokens=[]
        for batch_idx in range(input_ids.size(0)):
            next_token = self.logits_to_token(logits[batch_idx, -1, :])
            yield self.append_new_tokens(next_token, batch_idx), batch_idx
            next_tokens.append(next_token)
        self.seq_length += 1
        self.update_mask(next_tokens)

    def update_mask(self, new_tokens):
        batch_size, seq_length = self.attention_mask.shape

        new_tokens_tensor = torch.tensor(new_tokens, device=self.attention_mask.device)
        new_mask_col = torch.ones(batch_size, 1, device=self.attention_mask.device)

        if self.tokenizer.eos_token_id is not None:
            eos_mask = (new_tokens_tensor == self.tokenizer.eos_token_id)
            new_mask_col[eos_mask] = 0
        
        if self.tokenizer.pad_token_id is not None:
            pad_mask = (new_tokens_tensor == self.tokenizer.pad_token_id)
            new_mask_col[pad_mask] = 0

        self.attention_mask = torch.cat([self.attention_mask, new_mask_col], dim=1)

    @property
    def active_cache_position(self):
        device = self.device_map.get("blk.0.self_attn", {}).get("generate_device", "cuda:0")
        return torch.tensor([self.seq_length - 1], device=device)

    @torch.no_grad
    def generate(self, request_contexts: list = []):
        logger.info(f"args.max_new_tokens: {self.args.max_new_tokens}, cache_lens: {self.args.cache_lens}, seq_length: {self.seq_length}")
        if(self.max_new_tokens <= 0):
            logger.warning("max_new_tokens is less than 0")
            yield self.streamer.end_all(), "length"
            return
        self.profiler.set_counter("decode", 0)

        for i in range(1, self.max_new_tokens):
            if all(context['is_completed'] for context in request_contexts):
                break
            with torch.nn.attention.sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                if flashinfer_enabled:
                    MLAWrapperSingleton.plan_all(None,None,None,self.active_cache_position.to(torch.int32)+1, None,
                                             num_heads=self.model.config.num_attention_heads, head_dim_ckv=self.model.config.kv_lora_rank, 
                                             head_dim_kpe=self.model.config.qk_rope_head_dim, page_size=self.cache.page_size,
                                             sm_scale=self.model.model.layers[0].self_attn.softmax_scale, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
                next_tokens = self.decode_one_tokens()
                for batch_idx in range(len(next_tokens)):
                    if request_contexts[batch_idx]['is_completed'] == True:
                        continue
                    next_token = next_tokens[batch_idx]
                    self.profiler.inc("decode")
                    if next_token == self.tokenizer.eos_token_id or "<|im_end|>" == self.tokenizer.decode(next_token):
                        yield self.streamer.end(batch_idx), None, batch_idx
                        yield "", "stop", batch_idx
                        # assert self.args.batch_size == 1
                        request_contexts[batch_idx]['is_completed'] = True
                        continue
                    yield self.append_new_tokens(next_token, batch_idx), None, batch_idx
                self.seq_length += 1
        else:   # for's else, if output get max new tokens
            yield self.streamer.end_all(), None, 0
            yield "", "length", 0

    async def _batch_worker(self):
        while True:
            batch = []
            for _ in range(self.args.batch_size):
                try:
                    item = await asyncio.wait_for(self._inference_queue.get(), timeout=0.001)
                    batch.append(item)
                except asyncio.TimeoutError:
                    logger.debug("Timeout waiting for a single request")
                    break

            if not batch:
                await asyncio.sleep(0.001)
                continue
            logger.info(f"Collected {len(batch)} requests, starting to process batch")

            batch_data = {
                'messages': [item.get('local_messages', []) for item in batch],
                'thread_ids': [item.get('thread_id', '') for item in batch],
                'temperatures': [item.get('temperature', None) for item in batch],
                'top_ps': [item.get('top_p', None) for item in batch],
                'max_tokens': [item.get('max_tokens', None) for item in batch],
                'max_completion_tokens': [item.get('max_completion_tokens', None) for item in batch]
            }

            try:
                async def process_batch():
                    async for token, finish_reason, index in self.batch_inference(
                        batch_data['messages'],
                        batch_data['thread_ids'],
                        batch_data['temperatures'],
                        batch_data['top_ps'],
                        batch_data['max_tokens'],
                        batch_data['max_completion_tokens']
                    ):
                        await batch[index]['result_queue'].put((token, finish_reason))
                await process_batch()
            except Exception as e:
                logger.exception(f"Error in batch inference: {str(e)}")
                for item in batch:
                    await item['result_queue'].put(("ERROR", str(e)))
            finally:
                for item in batch:
                    await item['result_queue'].put((None, None))

    async def inference(self, local_messages, thread_id: str, temperature: Optional[float] = None, top_p: Optional[float] = None, max_tokens: Optional[float] = None, max_completion_tokens: Optional[float] = None):
        result_queue = asyncio.Queue()
        await self._inference_queue.put({
            'local_messages': local_messages,
            'thread_id': thread_id,
            'temperature': temperature,
            'top_p': top_p,
            'max_tokens': max_tokens,
            'max_completion_tokens': max_completion_tokens,
            'result_queue': result_queue
        })

        if self._batch_worker_task is None:
            self._batch_worker_task = asyncio.create_task(self._batch_worker())
        while True:
            token, finish_reason = await result_queue.get()
            if token is None:
                break
            yield token, finish_reason
        yield RawUsage(
            tokenize_time = self.profiler.get_timer_sec('tokenize'),
            prefill_time = self.profiler.get_timer_sec('prefill'),
            decode_time = self.profiler.get_timer_sec('decode'),
            prefill_count = self.profiler.get_counter('prefill'),
            decode_count = self.profiler.get_counter('decode'),
        )

    async def batch_inference(self, batch_messages: List[List], thread_ids: List[str], temperatures: List[Optional[float]], top_ps: List[Optional[float]], max_tokens_list: List[Optional[float]], max_completion_tokens_list: List[Optional[float]]):
        self.streamer.reset()
        self.profiler.create_and_start_timer("tokenize")
        print("SJF batch_messages len is ", len(batch_messages))

        input_ids_list = []
        for i, messages in enumerate(batch_messages):
            if isinstance(messages, List):
                input_ids = self.format_and_tokenize_input_ids(thread_ids[i], messages)
            elif isinstance(messages, str):
                input_ids = self.tokenize_prompt(messages)
            else:
                raise ValueError("local_messages should be List or str")
            input_ids_list.append(input_ids)

        max_length = max(ids.size(1) for ids in input_ids_list)
        padded_input_ids = []
        for ids in input_ids_list:
            padding_length = max_length - ids.size(1)
            if padding_length > 0:
                padded_ids = torch.cat([ids, torch.full((1, padding_length), self.tokenizer.pad_token_id, device=self.args.device)], dim=1)
            else:
                padded_ids = ids
            padded_input_ids.append(padded_ids)

        combined_input_ids = torch.cat(padded_input_ids, dim=0)  # [batch_size, seq_len]
        self.attention_mask = (combined_input_ids != self.tokenizer.pad_token_id).int()

        if Config().user_force_think:
            token_thinks = torch.tensor([self.tokenizer.encode("<think>\n",add_special_tokens=False)],device=input_ids.device)
            input_ids = torch.cat(
                [input_ids, token_thinks], dim=1
            )

        self.profiler.pause_timer("tokenize")

        self.profiler.create_and_start_timer("prefill")

        if Config().user_force_think:
            think = '<think>\n'
            print(think, end="",flush=True)
            yield think, None

        for t, batch_idx in self.prefill(
            combined_input_ids,
            True,  # is_new
            temperatures[0] if temperatures else None,
            top_ps[0] if top_ps else None,
            max_tokens_list[0] if max_tokens_list else None,
            max_completion_tokens_list[0] if max_completion_tokens_list else None,
        ):
            # output think token after prefill done
            if t is not None:
                print(t, end="",flush=True)
                yield t, None, batch_idx
        self.profiler.pause_timer("prefill")

        self.profiler.create_and_start_timer("decode")
        request_contexts = []
        for i in range(len(batch_messages)):
            context = {
                'is_completed': False,
            }
            request_contexts.append(context)

        self.profiler.create_and_start_timer("decode")

        for t, finish_reason, batch_idx in self.generate(request_contexts):
            if t is not None:
                if multi_batch_enabled:
                    print(f"Inference result: batch_idx={batch_idx}, token={t}", flush=True)
                else:
                    print(t, end="",flush=True)
                yield t, finish_reason, batch_idx
        print("")
        self.profiler.pause_timer("decode")
        self.report_last_time_performance()