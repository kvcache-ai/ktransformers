from typing import Any, List, Optional, Set
from transformers import (
    LlamaTokenizer,
    AutoTokenizer,
    AutoConfig,
    LlamaForCausalLM,
    GenerationConfig,
    StaticCache,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from ktransformers.server.config.config import Config
from ktransformers.server.schemas.base import ObjectID
from ktransformers.server.utils.multi_timer import Profiler
from torch.nn.attention import SDPBackend
import torch
import sys, os
from ..base import ThreadContext, BackendInterfaceBase
from ktransformers.server.config.log import logger
from ..args import ConfigArgs, default_args
from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled, MLAWrapperSingleton

# This TextStreamer is a modified version from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/streamers.py
class TextStreamer:

    def __init__(self, tokenizer: "AutoTokenizer", skip_prompt: bool = False, **decode_kwargs):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.decode_kwargs = decode_kwargs

        # variables used in the streaming process
        self.token_cache = []
        self.print_len = 0
        self.next_tokens_are_prompt = True

    def reset(self):
        self.token_cache = []
        self.print_len = 0

    def put(self, value) -> Optional[str]:
        """
        Receives tokens, decodes them, and prints them to stdout as soon as they form entire words.
        """
        if not isinstance(value, int):
            raise ValueError("TextStreamer only supports batch size 1, and int type input")

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return None

        # Add the new token to the cache and decodes the entire thing.
        self.token_cache.append(value)
        text = self.tokenizer.decode(self.token_cache, skip_special_tokens=True, **self.decode_kwargs)

        # After the symbol for a new line, we flush the cache.
        if text.endswith("\n"):
            printable_text = text[self.print_len :]
            self.reset()
        # If the last token is a CJK character, we print the characters.
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len :]
            self.print_len += len(printable_text)
        # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
        # which may change with the subsequent token -- there are probably smarter ways to do this!)
        else:
            printable_text = text[self.print_len : text.rfind(" ") + 1]
            self.print_len += len(printable_text)
        return printable_text

    def end(self) -> Optional[str]:
        """Flushes any remaining cache and prints a newline to stdout."""
        # Flush the cache, if it exists
        if len(self.token_cache) > 0:
            text = self.tokenizer.decode(self.token_cache, skip_special_tokens=True, **self.decode_kwargs)
            printable_text = text[self.print_len :]
            self.reset()
        else:
            printable_text = ""

        self.next_tokens_are_prompt = True
        return printable_text

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


class TransformersThreadContext(ThreadContext):
    def get_local_messages(self):
        local_messages = []
        for m in self.messages:
            local_messages.append({"role": m.role.value, "content": m.get_text_content()})

        return local_messages


class TransformersInterface(BackendInterfaceBase):
    use_static_cache: bool = True

    model: Any
    tokenizer: AutoTokenizer

    cache: StaticCache
    generated_ids: torch.Tensor
    seq_length: int

    streamer: TextStreamer

    # thread_related
    last_request_id: Optional[str] = None
    ever_generated_ids: Set[int] = set()

    def __init__(self, args: ConfigArgs = default_args):
        self.args = args

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map=args.device, use_safetensors=True)
        # logger.info(f"{args.model_name} loaded from {args.model_dir} to {args.device}")

        self.cache = StaticCache(
            config=self.model.config,
            max_batch_size=args.batch_size,
            max_cache_len=args.cache_lens,
            device=args.device,
            dtype=self.model.dtype,
        )
        # logger.info(f"StaticCache (length={args.cache_lens}) created at {args.device}, batch size:{args.batch_size}")

        self.streamer = TextStreamer(self.tokenizer)

    @property
    def current_ids(self):
        return self.generated_ids[:, self.seq_length - 1].unsqueeze(1)

    @property
    def active_cache_position(self):
        return torch.tensor([self.seq_length - 1], device=self.args.device)

    def tokenize_prompt(self, prompt: str):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.args.device)
        return input_ids

    def format_and_tokenize_input_ids(self, thread_id: ObjectID, messages: List):
        for m in messages:
            if m["role"] == "system":
                logger.warning(f'change {m["role"]} to user')
                m["role"] = "user"

        new_messages = [messages[0]]
        for m in messages[1:]:
            if m["role"] == "user" and new_messages[-1]["role"] == "user":
                logger.warning("merge two adjacent user messages")
                new_messages[-1]["content"] += '\n' + m["content"]
            else:
                new_messages.append(m)
        # if (self.last_request_id is not None) and self.last_request_id == thread_id:
        #     input_ids = self.tokenizer.encode(self.tokenizer.eos_token+self.tokenizer.apply_chat_template([new_messages[-1]], return_tensors="pt",tokenize=False, add_generation_prompt=True), add_special_tokens = False, return_tensors="pt").to(self.args.device)
        # else:
        #     input_ids = self.tokenizer.apply_chat_template(
        #         new_messages, return_tensors="pt", add_generation_prompt=True
        #     ).to(self.args.device)
        input_str: str = self.tokenizer.apply_chat_template(new_messages,tokenize=False,add_generation_prompt=True)
        # drop <think> token in chat template
        if input_str.endswith('<think>\n'):
            input_str = input_str[:-len('<think>\n')]
        input_ids = self.tokenizer.encode(input_str, return_tensors="pt").to(self.args.device)
        if (self.last_request_id is not None) and self.last_request_id == thread_id:
            x = self.generated_ids[:,:self.seq_length]
            y = input_ids[:,:self.seq_length]
            # We can only hope that the input_ids are the same
            unequal_mask = torch.ne(x,y)
            unequal_positions = torch.nonzero(unequal_mask)
            num_unequal_elements = unequal_mask.sum().item()
            logger.warning(f'num_unequal_elements: {num_unequal_elements}') 

            input_ids = input_ids[:,self.seq_length:]
        logger.debug(f"get input ids of shape {input_ids.shape}")
        return input_ids

    def append_new_tokens(self, new_tokens: int) -> Optional[str]:
        self.generated_ids[0, self.seq_length] = new_tokens
        self.seq_length += 1
        return self.streamer.put(new_tokens)

    def prepare_logits_wrapper(self, inputs, device, temperature: Optional[float] = None, top_p: Optional[float] = None):
        if temperature is None or temperature == 0:
            temperature = self.model.generation_config.temperature
        if top_p is None:
            top_p = self.model.generation_config.top_p
        generation_config, model_kwargs = self.model._prepare_generation_config(
            None, max_length=self.args.max_new_tokens,
            do_sample=True, 
            top_k=self.args.top_k, 
            top_p=top_p, 
            temperature=temperature,
            repetition_penalty=self.args.repetition_penalty # change this to modify generate config
        )
        self.inputs = inputs
        try: # transformers==4.43
            self.logits_warper = (
                self.model._get_logits_warper(generation_config, device=device)
            )
        except: 
            self.logits_warper = (
                self.model._get_logits_warper(generation_config)
            )

    def logits_to_token(self, logits: torch.Tensor):
        logits = self.logits_warper(self.inputs.view(1, -1), logits.view(1, -1))

        probs = torch.nn.functional.softmax(logits, dim=-1)

        sample = True
        if sample:
            last = torch.multinomial(probs, num_samples=1)
        else:
            _, last = torch.topk(probs, k=1, dim=-1)

        last = last.item()
        self.ever_generated_ids.add(last)
        return last

    def decode_one_tokens(self):
        if self.use_static_cache:
            logits = self.model(
                self.current_ids,
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
    def prefill(self, input_ids: torch.Tensor, is_new: bool, temperature: Optional[float] = None, top_p: Optional[float] = None):
        input_ids_length = input_ids.shape[-1]
        logger.debug(f"input_ids: {input_ids.shape}")

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
        expected_length = self.seq_length + self.args.max_new_tokens + 1
        delta_length = expected_length - self.generated_ids.shape[-1]
        if delta_length > 0:
            new_generate_ids = torch.zeros(
                self.args.batch_size, delta_length, dtype=torch.int, device=self.args.device
            )
            self.generated_ids = torch.cat([self.generated_ids, new_generate_ids], dim=-1)
            
        logger.debug(f"cache position: {former_seq_length} to {self.seq_length}")
        cache_position = torch.arange(former_seq_length, self.seq_length, device=self.args.device)
        self.generated_ids[:, cache_position] = input_ids.to(self.args.device).to(torch.int)

        device = input_ids.device
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
            )[0]
        else:
            logits = self.model(inputs_embeds=inputs_embeds, return_dict=False)[0]

        self.prepare_logits_wrapper(input_ids, device, temperature, top_p)
        next_token = self.logits_to_token(logits[0, -1, :])
        yield self.append_new_tokens(next_token)

    @torch.no_grad
    def generate(self):
        self.max_new_tokens = min(self.args.max_new_tokens, self.args.cache_lens - self.seq_length) - 1 
        logger.info(f"args.max_new_tokens: {self.args.max_new_tokens}, cache_lens: {self.args.cache_lens}, seq_length: {self.seq_length}")
        if(self.max_new_tokens <= 0):
            logger.warning("max_new_tokens is less than 0")
            yield self.streamer.end()
            return
        logger.info(f"max_new_tokens: {self.max_new_tokens}")
        self.profiler.set_counter("decode", 0)

        for i in range(1, self.max_new_tokens):
            with torch.nn.attention.sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
                if flashinfer_enabled:
                    MLAWrapperSingleton.plan_all(None,None,None,self.active_cache_position.to(torch.int32)+1,
                                             num_heads=self.model.config.num_attention_heads, head_dim_ckv=self.model.config.kv_lora_rank, 
                                             head_dim_kpe=self.model.config.qk_rope_head_dim, page_size=self.cache.page_size,
                                             sm_scale=(self.model.config.qk_rope_head_dim + self.model.config.qk_nope_head_dim) ** (-0.5), q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)
                next_token = self.decode_one_tokens()
                self.profiler.inc("decode")
                if next_token == self.tokenizer.eos_token_id or "<|im_end|>" == self.tokenizer.decode(next_token):
                    assert self.args.batch_size == 1
                    break
                yield self.append_new_tokens(next_token)
        yield self.streamer.end()

    def check_is_new(self, thread_id: str):
        if not self.use_static_cache:
            return True
        if self.last_request_id is None:
            self.last_request_id = thread_id
            return True
        else:
            if self.last_request_id == thread_id:
                return False
            else:
                self.last_request_id = thread_id
                return True

    async def inference(self, local_messages, thread_id: str, temperature: Optional[float] = None, top_p: Optional[float] = None):
        self.streamer.reset()
        self.profiler.create_and_start_timer("tokenize")
        if isinstance(local_messages, List):
            input_ids = self.format_and_tokenize_input_ids(thread_id, local_messages)
        elif isinstance(local_messages, str):
            #local_messages = local_messages[0]['content']
            input_ids = self.tokenize_prompt(local_messages)
            #input_ids = torch.tensor([[6366]], device=input_ids.device)
        else:
            raise ValueError("local_messages should be List or str")
        
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
            yield think
        
        for t in self.prefill(input_ids, self.check_is_new(thread_id), temperature, top_p):
            # output think token after prefill done
            if t is not None:
                print(t, end="",flush=True)
                yield t
        self.profiler.pause_timer("prefill")

        self.profiler.create_and_start_timer("decode")
        for t in self.generate():
            if t is not None:
                print(t, end="",flush=True)
                yield t 
        print("")
        self.profiler.pause_timer("decode")
        self.report_last_time_performance()
