from typing import Any, AsyncIterator, List, Optional, Set
from ktransformers.models.custom_cache import KDeepSeekV3Cache, KGQACache
from transformers import (
    AutoTokenizer,
    AutoConfig,
    GenerationConfig,
    StaticCache,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from ktransformers.server.config.config import Config
from ..base import ThreadContext, BackendInterfaceBase
import torch
from ktransformers.server.backend.interfaces.transformers import (
    ConfigArgs,
    default_args,
    TextStreamer,
)
from ktransformers.server.schemas.base import ObjectID
from ktransformers.server.config.log import logger
from ktransformers.optimize.optimize import optimize_and_load_gguf
from ktransformers.models.custom_modeling_llama4 import KLlama4ForCausalLM
from ktransformers.models.custom_modeling_deepseek_v3 import KDeepseekV3ForCausalLM
from ktransformers.models.custom_modeling_qwen2_moe import KQwen2MoeForCausalLM
from ktransformers.models.configuration_llama4 import Llama4Config
from ktransformers.models.custom_modeling_deepseek_v2 import KDeepseekV2ForCausalLM
from ktransformers.server.balance_serve.inference.model_runner import ModelRunner 
from ktransformers.server.balance_serve.inference.sampling.sampler import Sampler, SamplingOptions
from ktransformers.server.balance_serve.inference.query_manager import QueryManager
from ktransformers.server.balance_serve.inference.forward_batch import ForwardBatchInput, ForwardBatchOutput
from ktransformers.server.balance_serve.sched_rpc import SchedulerClient
from ktransformers.server.balance_serve.settings import sched_ext
from torch.multiprocessing import Queue
import torch.multiprocessing as mp
from ktransformers.server.schemas.endpoints.chat import RawUsage
from ktransformers.server.utils.multi_timer import Profiler
import zmq
import time
import queue
import tempfile
import asyncio
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
import os



ktransformer_rules_dir = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "./optimize/optimize_rules/") 
)
default_optimize_rules = {
    # "DeepseekV3ForCausalLM": ktransformer_rules_dir + "Moonlight-16B-A3B-serve.yaml",
    "DeepseekV3ForCausalLM": ktransformer_rules_dir + "DeepSeek-V3-Chat-serve.yaml",
    "Qwen2MoeForCausalLM": ktransformer_rules_dir + "Qwen2-57B-A14B-Instruct-serve.yaml",
    "Llama4ForConditionalGeneration": ktransformer_rules_dir + "Llama4-serve.yaml",
}

async def chat_stream(queue: asyncio.Queue, tokenizer: AutoTokenizer):
    streamer = TextStreamer(tokenizer)
    while True:
        token = await queue.get()
        #print(f"Got token: {token}")
        if token is None:
            # str = f'{token}\n\n'
            # str = model.tokenizer.decode(token)
            s = streamer.end()
            if s is not None:
                yield s
            break

        # str = model.tokenizer.decode(token)
        yield streamer.put(token)
        


def fill_generated_tokens(query_updates: list[sched_ext.QueryUpdate], generated_tokens: torch.Tensor, query_manager: QueryManager = None):
    #print(len(query_updates), generated_tokens.size(0), generated_tokens)
    for i in range(generated_tokens.size(0)):
        print(generated_tokens[i].item())
        query_updates[i].generated_token = generated_tokens[i].item()
        if not query_manager.query_map[query_updates[i].id].is_prefill:
            pos = query_updates[i].active_position
            query_manager.query_map[query_updates[i].id].query_tokens[pos] = generated_tokens[i]

def report_last_time_performance(profiler: Profiler):
        try:
            tokenize_time = profiler.get_timer_sec('tokenize')
            prefill_time = profiler.get_timer_sec('prefill')
            decode_time = profiler.get_timer_sec('decode')
            prefill_count = profiler.get_counter('prefill')
            decode_count = profiler.get_counter('decode')

            logger.info(f'Performance(T/s): prefill {prefill_count/prefill_time}, decode {decode_count/decode_time}. Time(s): tokenize {tokenize_time}, prefill {prefill_time}, decode {decode_time}')
        except:
            logger.info(f'Performance statistics not recorded')

class Engine:
    sched_client : SchedulerClient
    updates : list[sched_ext.QueryUpdate]
    batch : sched_ext.BatchQueryTodo
    model_runner: ModelRunner
    sampler: Sampler
    query_manager: QueryManager
    cache: KDeepSeekV3Cache | KGQACache
    def __init__(self, args: ConfigArgs = default_args, generated_token_queue:Queue = None, broadcast_endpoint: str = None):
        self.args = args

        # 子进程和父进程无法共享 config 变量
        for key, value in vars(args).items():
            if value is not None and hasattr(Config(), key):
                setattr(Config(), key, value)

        self.device = self.args.device
        self.sched_client = SchedulerClient(args.sched_port)
        self.updates = []


        config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True) 

        with torch.device("meta"):
            if config.architectures[0] == "DeepseekV3ForCausalLM":
                self.cache = KDeepSeekV3Cache(config, self.args.page_size)
                self.model = KDeepseekV3ForCausalLM(config, self.cache)
            elif config.architectures[0] == "DeepseekV2ForCausalLM":
                self.cache = KDeepSeekV3Cache(config, self.args.page_size)
                self.model = KDeepseekV2ForCausalLM(config, self.cache)
            elif config.architectures[0] == "Qwen2MoeForCausalLM":
                self.cache = KGQACache(config, self.args.page_size)
                self.model = KQwen2MoeForCausalLM(config, self.cache)
            elif config.architectures[0] == "Llama4ForConditionalGeneration":
                self.cache =  KGQACache(config, self.args.page_size)
                self.model = KLlama4ForCausalLM(config.text_config, self.cache)
    
        self.gen_queue = generated_token_queue
            
        print(f"Getting inference context from sched_client.")
        inference_context = self.sched_client.get_inference_context_raw()
        print(f"Got inference context, sending it to subscribers.")
        inference_context = self.sched_client.rebuild_inferece_context(inference_context)
        self.cache.load(inference_context)
        print(f"kv_cache loaded successfully.")

        self.block_num = inference_context.k_cache[0].size(1)

        # print(self.block_num)

        context = zmq.Context()

            
        self.pub_socket = context.socket(zmq.PUB)
        self.pub_socket.bind(f"ipc://{broadcast_endpoint}") 
        # time.sleep(1) # make sure all subscribers are ready


        try:
            generation_config = GenerationConfig.from_pretrained(args.model_dir)
        except:
            generation_config = GenerationConfig(
                max_length=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True
            )
            
        if args.optimize_config_path is None:
            optimize_config_path = default_optimize_rules[config.architectures[0]]
               
        else:
            optimize_config_path = args.optimize_config_path
        gguf_path = args.gguf_path
        if gguf_path is None:
            gguf_path = input(
                "please input the path of your gguf file(gguf file in the dir containing input gguf file must all"
                " belong to current model):"
            )
        optimize_and_load_gguf(self.model, optimize_config_path, gguf_path, config)
        self.model.generation_config = generation_config
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.model.generation_config.eos_token_id

        self.model.eval()
        #@TODO add config
        if config.architectures[0] == "Qwen2MoeForCausalLM" or config.architectures[0] == "Llama4ForConditionalGeneration":
            self.model.init_wrapper(self.args.use_cuda_graph, self.device, 1024 ,args.max_batch_size, self.block_num) # TODO: 1024 is a magic number(max_batch_tokens)
        else:
            self.model.init_wrapper(self.args.use_cuda_graph, self.device, args.max_batch_size, self.block_num)

        self.model_runner = ModelRunner(self.model, self.device, self.args.use_cuda_graph, page_size = args.page_size, block_num=self.block_num)
        self.sampler = Sampler()
        self.query_manager = QueryManager(device = self.device, page_size = args.page_size)

            
    def sampling(self, forward_output: ForwardBatchOutput):
        generated_tokens = torch.empty(0, device=self.device, dtype=torch.int32)
        for i in range(forward_output.num_batchs):
            logit = forward_output.logits[i]
            if hasattr(forward_output, "temperatures"):
                temperatures = forward_output.temperatures[i]
            else:
                temperatures = None
            
            if hasattr(forward_output, "top_ps"):
                top_ps = forward_output.top_ps[i]
            else:
                top_ps = None

            sample_options = SamplingOptions(logit.size(0), self.device, pretrained_config=self.model.generation_config, temperatures=temperatures, top_ps=top_ps)
            generated_tokens, probs=self.sampler(logit, sample_options)
        return generated_tokens, probs
    
    def loop(self):

        next_batch = None   

        while True:
            self.batch = next_batch
            if self.batch is not None:
                self.model_runner.run(self.batch, self.query_manager)

            if len(self.updates) > 0:
                for q in self.updates:
                    if q.is_prefill == True:
                        continue
                    # print(f"Putting token {q.generated_token} into queue for query id: {q.id}")
                    try:
                        self.gen_queue.put((q.id, q.generated_token if q.decode_done == False else None), timeout=5)
                    except queue.Full:
                        pass#print("Queue is full after timeout; unable to put more items.")
                
            next_batch = self.sched_client.update_last_batch(self.updates)
            if next_batch.query_ids == []:
                next_batch = None
            self.pub_socket.send_pyobj(next_batch)  

            if next_batch is not None:
                self.query_manager.add_query(next_batch)
            
            
            if self.batch is not None:
                self.model_runner.sync()
                print(f"Model execution time (GPU): {self.model_runner.model_time:.3f} ms, {1000/self.model_runner.model_time:.3f} tokens/s")
                # if self.rank == 0:
                
                generated_tokens, probs = self.sampling( self.model_runner.output)
                
                self.updates = self.query_manager.update(self.batch)
                fill_generated_tokens(self.updates, generated_tokens, self.query_manager)
            else:
                self.updates = []

class BalanceServeThreadContext(ThreadContext):
    def get_local_messages(self):
        local_messages = []
        for m in self.messages:
            local_messages.append({"role": m.role.value, "content": m.get_text_content()})

        return local_messages
    

def run_engine(args, token_queue, broadcast_endpoint, event):
    engine = Engine(args, token_queue, broadcast_endpoint)
    if args.use_cuda_graph:
        engine.model_runner.warmup()
        
    event.set()
    engine.loop()


class BalanceServeInterface(BackendInterfaceBase):
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
        self.queue_map:dict[int,asyncio.Queue] = {}
        self.thread_map: dict[int, int] = {}
        processes = []
        self.broadcast_endpoint = tempfile.NamedTemporaryFile(delete=False).name # @TODO add to config
        ctx = mp.get_context("spawn")
        self.token_queue = ctx.Queue(maxsize=1000) 
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        self.sched_client = SchedulerClient(args.sched_port)
        self.streamer = TextStreamer(self.tokenizer)

        start_event = ctx.Event()

        p = ctx.Process(target=run_engine, args=(self.args, self.token_queue, self.broadcast_endpoint, start_event))
        p.start()
        processes.append(p)
        start_event.wait()
        
    def run_queue_proxy(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.queue_proxy())

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        asyncio.create_task(self.queue_proxy())
        yield

    async def queue_proxy(self):
        print("Queue Proxy Started")
        while True:
            try:
                query_id, token = self.token_queue.get_nowait()
                try:
                    # query id might not be allocated yet
                    self.queue_map[query_id].put_nowait(token)
                    #print(f"Proxy Put token: {token} to queue for query id: {query_id}")
                except asyncio.QueueFull:
                    #print(f"Queue for query id: {query_id} is full, waiting to put: {token}")
                    await self.queue_map[query_id].put(token)

            except queue.Empty:
                # print("no new token")
                # await asyncio.sleep(1)
                await asyncio.sleep(0)
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
        input_str: str = self.tokenizer.apply_chat_template(new_messages,tokenize=False,add_generation_prompt=True)
        # drop <think> token in chat template
        if input_str.endswith('<think>\n'):
            input_str = input_str[:-len('<think>\n')]
        input_ids = self.tokenizer.encode(input_str, return_tensors="pt").to(self.args.device)
        logger.debug(f"get input ids of shape {input_ids.shape}")
        return input_ids
    
    async def inference(self, local_messages, thread_id: str, temperature: Optional[float] = None, top_p: Optional[float] = None):
        profiler = Profiler()
        profiler.create_and_start_timer("tokenize")
        
        if isinstance(local_messages, List):
            input_ids = self.format_and_tokenize_input_ids(thread_id, local_messages)
        elif isinstance(local_messages, str):
            #local_messages = local_messages[0]['content']
            input_ids = self.tokenize_prompt(local_messages)
        else:
            raise ValueError("local_messages should be List or str")
        if Config().user_force_think:
            token_thinks = torch.tensor([self.tokenizer.encode("<think>\n",add_special_tokens=False)],device=input_ids.device)
            input_ids = torch.cat(
                [input_ids, token_thinks], dim=1
            )

        
        profiler.pause_timer("tokenize")

        profiler.create_and_start_timer("prefill")

        
        
        query_add = sched_ext.QueryAdd()
        query_add.query_token =  input_ids[0].tolist()
        query_length = input_ids[0].shape[0]
        query_add.query_length = query_length
        profiler.set_counter("prefill", query_length)
        #@TODO add server
        stop_criteria = [[self.tokenizer.eos_token_id]]
        print("stop_criteria", stop_criteria)
        query_add.stop_criteria = stop_criteria
        if temperature == 0:
            temperature = 0.0001
        query_add.sample_options.temperature = temperature
        if top_p == 0:
            top_p = 0.0001
        query_add.sample_options.top_p = top_p
        query_add.estimated_length = min(self.args.cache_lens, query_length+self.args.max_new_tokens)
        query_id = self.sched_client.add_query(query_add)
        queue = asyncio.Queue(maxsize=self.args.max_new_tokens)
        self.queue_map[query_id] = queue
        self.thread_map[thread_id] = query_id
        is_first_token = True
        async for token in chat_stream(self.queue_map[query_id], self.tokenizer):
            if is_first_token:
                is_first_token=False
                profiler.pause_timer("prefill")
                profiler.create_and_start_timer("decode")
                profiler.set_counter("decode", 0)
                if Config().user_force_think:
                    think = '<think>\n'
                    print(think, end="",flush=True)
                    yield think, None
            else:
                profiler.inc("decode")
            yield token, None
        profiler.pause_timer("decode")
        report_last_time_performance(profiler)
        yield self.streamer.end(), None
        if profiler.get_counter('decode') >= self.args.max_new_tokens - 1:
            yield "", "length"
        else:
            yield "", "stop"
        
        
        yield RawUsage(
                tokenize_time = profiler.get_timer_sec('tokenize'),
                prefill_time = profiler.get_timer_sec('prefill'),
                decode_time = profiler.get_timer_sec('decode'),
                prefill_count = profiler.get_counter('prefill'),
                decode_count = profiler.get_counter('decode'),
            )
