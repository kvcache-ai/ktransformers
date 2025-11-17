'''
Date: 2024-11-12 14:15:16
LastEditors: Xie Weiyu ervinxie@qq.com
LastEditTime: 2024-11-26 08:12:49
'''
import torch
from ktransformers.server.balance_serve.settings import sched_ext
from ktransformers.server.balance_serve.inference.query_manager import QueryManager, QueryInfo
import time
from ktransformers.server.config.config import Config
class ForwardBatchInput:

    class ForwardMiniBatch:
        q_indptr: torch.Tensor
        kv_indptr: torch.Tensor
        kv_indices: torch.Tensor
        kv_last_page_len: torch.Tensor
        kv_len: torch.Tensor
        position_ids: torch.Tensor
        tokens: torch.Tensor
        batch_indices: torch.Tensor
        positions: torch.Tensor
        chunk_size: int
        decode_batch: int        
        is_last_prefill_chunk: bool
        logits_start: list

        temperatures: torch.Tensor
        top_ps: torch.Tensor

        def __init__(self, prefill_querys_info: list[QueryInfo], decode_querys_info: list[QueryInfo], prefill_s: list[int] = None, prefill_l: list[int] = None, device = torch.device('cuda'), page_size = 256):
            batch_decode = len(decode_querys_info)
            batch_prefill = len(prefill_querys_info)

            self.q_indptr = torch.tensor([0], device=device, dtype=torch.int32)
            self.kv_indptr = torch.tensor([0], device=device, dtype=torch.int32)
            self.kv_indices = torch.tensor([], device=device, dtype=torch.int32)
            self.kv_len = torch.tensor([], device=device, dtype=torch.int32)
            self.kv_last_page_len = torch.tensor([], device=device, dtype=torch.int32)
            self.position_ids = torch.tensor([], device=device, dtype=torch.int32)
            self.tokens = torch.tensor([], device=device, dtype=torch.int32)

            self.temperatures = torch.tensor([], device=device, dtype=torch.float32)
            self.top_ps = torch.tensor([], device=device, dtype=torch.float32)

            self.logits_start = []
            self.decode_batch = batch_decode
            self.num_tokens = batch_decode + sum(prefill_l)
            self.batch_size = batch_decode + batch_prefill
            
            for i, prefill_query_info in enumerate(prefill_querys_info):
                if prefill_query_info != None:
                    prefill_kv_block_len = (prefill_query_info.active_position + prefill_l[i] + page_size - 1) // page_size if prefill_query_info is not None else 0
                    # print(f"block_len: {prefill_kv_block_len}, page_size: {page_size}")
                    self.q_indptr = torch.concat((self.q_indptr, torch.tensor([prefill_l[i] + self.q_indptr[-1]], device=device, dtype=torch.int32)), dim=0)
                    self.kv_indptr = torch.concat((self.kv_indptr, torch.tensor([prefill_kv_block_len + self.kv_indptr[-1]], device=device, dtype=torch.int32)), dim=0)
                    self.kv_indices = torch.concat((self.kv_indices, prefill_query_info.block_index[:prefill_kv_block_len]), dim=0)
                    self.kv_last_page_len = torch.concat((self.kv_last_page_len, torch.tensor([(prefill_query_info.active_position + prefill_l[i]) % page_size if (prefill_query_info.active_position + prefill_l[i]) % page_size != 0 else page_size], device=device, dtype=torch.int32)), dim=0)
                    self.kv_len = torch.concat((self.kv_len, torch.tensor([(prefill_query_info.active_position + prefill_l[i])], device=device, dtype=torch.int32)), dim=0)
                    self.position_ids = torch.concat((self.position_ids, torch.arange(prefill_s[i], prefill_l[i] + prefill_s[i], device=device, dtype=torch.int32)), dim=0)
                    self.tokens = torch.concat((self.tokens, prefill_query_info.query_tokens[prefill_s[i]:prefill_s[i] + prefill_l[i]]), dim=0)
                    self.logits_start.append(prefill_l[i] - 1 if len(self.logits_start) == 0 else sum(prefill_l[:i+1])-1)

                    self.temperatures = torch.concat((self.temperatures, torch.tensor([prefill_query_info.temperature], device=device, dtype=torch.float32)), dim=0)
                    self.top_ps = torch.concat((self.top_ps, torch.tensor([prefill_query_info.top_p], device=device, dtype=torch.float32)), dim=0)

            for decode_query_info in decode_querys_info:
                decode_kv_block_len = (decode_query_info.active_position + 1 + page_size - 1) // page_size
                self.q_indptr = torch.concat((self.q_indptr, torch.tensor([1 + self.q_indptr[-1]], device=device, dtype=torch.int32)), dim=0)
                self.kv_indptr = torch.concat((self.kv_indptr, torch.tensor([decode_kv_block_len+self.kv_indptr[-1]], device=device, dtype=torch.int32)), dim=0)
                self.kv_indices = torch.concat((self.kv_indices, decode_query_info.block_index[:decode_kv_block_len]), dim=0)
                self.kv_last_page_len = torch.concat((self.kv_last_page_len, torch.tensor([(decode_query_info.active_position + 1) % page_size if (decode_query_info.active_position + 1) % page_size != 0 else page_size], device=device, dtype=torch.int32)), dim=0)
                self.kv_len = torch.concat((self.kv_len, torch.tensor([(decode_query_info.active_position + 1)], device=device, dtype=torch.int32)), dim=0)
                self.position_ids = torch.concat((self.position_ids, torch.arange(decode_query_info.active_position, decode_query_info.active_position + 1, device=device, dtype=torch.int32)), dim=0)
                if decode_query_info.active_position > 0:
                    self.tokens = torch.concat((self.tokens, decode_query_info.query_tokens[decode_query_info.active_position:decode_query_info.active_position+1]), dim=0)
                else: 
                    self.tokens = torch.concat((self.tokens, torch.tensor([0], device=device, dtype=torch.int32)), dim=0)
                self.logits_start.append(0 if len(self.logits_start) == 0 else self.logits_start[-1]+1)

                self.temperatures = torch.concat((self.temperatures, torch.tensor([decode_query_info.temperature], device=device, dtype=torch.float32)), dim=0)
                self.top_ps = torch.concat((self.top_ps, torch.tensor([decode_query_info.top_p], device=device, dtype=torch.float32)), dim=0)

            self.q_indptr = self.q_indptr.contiguous()
            self.kv_indptr = self.kv_indptr.contiguous()
            self.kv_indices = self.kv_indices.contiguous()
            self.kv_len = self.kv_len.contiguous()
            self.kv_last_page_len = self.kv_last_page_len.contiguous()
            self.position_ids = self.position_ids.contiguous()
            self.tokens = self.tokens.contiguous()

            self.bsz_tensor = torch.tensor([self.batch_size], device=device, dtype=torch.int32)

        def fill(self, prefill_querys_info: list[QueryInfo], decode_querys_info: list[QueryInfo], prefill_s: list[int] = None, prefill_l: list[int] = None, device = torch.device('cuda'), page_size = 256):
            batch_decode = len(decode_querys_info)
            batch_prefill = len(prefill_querys_info)

            self.q_indptr = torch.tensor([0], device=device, dtype=torch.int32)
            self.kv_indptr = torch.tensor([0], device=device, dtype=torch.int32)
            self.kv_indices = torch.tensor([], device=device, dtype=torch.int32)
            self.kv_len = torch.tensor([], device=device, dtype=torch.int32)
            self.kv_last_page_len = torch.tensor([], device=device, dtype=torch.int32)
            new_position_ids = torch.tensor([], device=device, dtype=torch.int32)
            new_tokens = torch.tensor([], device=device, dtype=torch.int32)

            self.temperatures = torch.tensor([], device=device, dtype=torch.float32)
            self.top_ps = torch.tensor([], device=device, dtype=torch.float32)

            self.logits_start = []
            self.decode_batch = batch_decode
            self.num_tokens = batch_decode + sum(prefill_l)
            self.batch_size = batch_decode + batch_prefill

            for i, prefill_query_info in enumerate(prefill_querys_info):
                prefill_kv_block_len = (prefill_query_info.active_position + prefill_l[i] + page_size - 1) // page_size if prefill_query_info is not None else 0
            # print(f"block_len: {prefill_kv_block_len}, page_size: {page_size}")
                self.q_indptr = torch.concat((self.q_indptr, torch.tensor([prefill_l[i] + self.q_indptr[-1]], device=device, dtype=torch.int32)), dim=0)
                self.kv_indptr = torch.concat((self.kv_indptr, torch.tensor([prefill_kv_block_len + self.kv_indptr[-1]], device=device, dtype=torch.int32)), dim=0)
                self.kv_indices = torch.concat((self.kv_indices, prefill_query_info.block_index[:prefill_kv_block_len]), dim=0)
                self.kv_last_page_len = torch.concat((self.kv_last_page_len, torch.tensor([(prefill_query_info.active_position + prefill_l[i]) % page_size if (prefill_query_info.active_position + prefill_l[i]) % page_size != 0 else page_size], device=device, dtype=torch.int32)), dim=0)
                self.kv_len = torch.concat((self.kv_len, torch.tensor([(prefill_query_info.active_position + prefill_l[i])], device=device, dtype=torch.int32)), dim=0)
                new_position_ids = torch.concat((new_position_ids, torch.arange(prefill_s[i], prefill_l[i] + prefill_s[i], device=device, dtype=torch.int32)), dim=0)
                new_tokens = torch.concat((new_tokens, prefill_query_info.query_tokens[prefill_s[i]:prefill_s[i] + prefill_l[i]]), dim=0)
                self.logits_start.append(prefill_l[i] - 1 if len(self.logits_start) == 0 else sum(prefill_l[:i+1])-1)

                self.temperatures = torch.concat((self.temperatures, torch.tensor([prefill_query_info.temperature], device=device, dtype=torch.float32)), dim=0)
                self.top_ps = torch.concat((self.top_ps, torch.tensor([prefill_query_info.top_p], device=device, dtype=torch.float32)), dim=0)


            for decode_query_info in decode_querys_info:
                decode_kv_block_len = (decode_query_info.active_position + 1 + page_size - 1) // page_size
                self.q_indptr = torch.concat((self.q_indptr, torch.tensor([1 + self.q_indptr[-1]], device=device, dtype=torch.int32)), dim=0)
                self.kv_indptr = torch.concat((self.kv_indptr, torch.tensor([decode_kv_block_len+self.kv_indptr[-1]], device=device, dtype=torch.int32)), dim=0)
                self.kv_indices = torch.concat((self.kv_indices, decode_query_info.block_index[:decode_kv_block_len]), dim=0)
                self.kv_last_page_len = torch.concat((self.kv_last_page_len, torch.tensor([(decode_query_info.active_position + 1) % page_size if (decode_query_info.active_position + 1) % page_size != 0 else page_size], device=device, dtype=torch.int32)), dim=0)
                self.kv_len = torch.concat((self.kv_len, torch.tensor([(decode_query_info.active_position + 1)], device=device, dtype=torch.int32)), dim=0)
                new_position_ids = torch.concat((new_position_ids, torch.arange(decode_query_info.active_position, decode_query_info.active_position + 1, device=device, dtype=torch.int32)), dim=0)
                if decode_query_info.active_position > 0:
                    new_tokens = torch.concat((new_tokens, decode_query_info.query_tokens[decode_query_info.active_position:decode_query_info.active_position+1]), dim=0)
                else: 
                    new_tokens = torch.concat((new_tokens, torch.tensor([0], device=device, dtype=torch.int32)), dim=0)
                self.logits_start.append(0 if len(self.logits_start) == 0 else self.logits_start[-1]+1)

                self.temperatures = torch.concat((self.temperatures, torch.tensor([decode_query_info.temperature], device=device, dtype=torch.float32)), dim=0)
                self.top_ps = torch.concat((self.top_ps, torch.tensor([decode_query_info.top_p], device=device, dtype=torch.float32)), dim=0)


            self.q_indptr = self.q_indptr.contiguous()
            self.kv_indptr = self.kv_indptr.contiguous()
            self.kv_indices = self.kv_indices.contiguous()
            self.kv_len = self.kv_len.contiguous()
            self.kv_last_page_len = self.kv_last_page_len.contiguous()

            self.bsz_tensor = torch.tensor([self.batch_size], device=device, dtype=torch.int32)
            
            # copy new_position_ids and new_tokens to self.position_ids and self.tokens
            # print("new_position_ids: ", new_position_ids)
            # self.print()
            self.position_ids[:new_position_ids.size(0)].copy_(new_position_ids)
            self.position_ids[new_position_ids.size(0):].zero_()
            self.tokens[:new_tokens.size(0)].copy_(new_tokens)


    forward_minibatchs: list[ForwardMiniBatch]
    batch_size: int
    minibatch: ForwardMiniBatch



    def __init__(self, batch : sched_ext.BatchQueryTodo = None, query_manager: QueryManager = None, device=None, tokens: torch.Tensor = None):
        
        if batch is None:
            return


        prefill_minibatches = batch.prefill_mini_batches
        decode_mini_batches = [item for sublist in batch.decode_mini_batches for item in sublist]
        prefill_querys_info = []
        prefill_s = []
        prefill_l = []
        decode_querys_info = []
        self.batch_size = 1
        for (id, s, l) in prefill_minibatches:
            prefill_querys_info.append(query_manager.query_map[id])
            prefill_s.append(s)
            prefill_l.append(l)
        for decode_batch_idx in decode_mini_batches:
            if query_manager.query_map[decode_batch_idx].decode_start_time is None:
                query_manager.query_map[decode_batch_idx].decode_start_time =time.time()
            decode_querys_info.append(query_manager.query_map[decode_batch_idx])


        minibatch = ForwardBatchInput.ForwardMiniBatch(prefill_querys_info, decode_querys_info, prefill_s, prefill_l, device = query_manager.device, page_size = query_manager.page_size)
 
        self.minibatch = minibatch

    @classmethod
    def gen_max_forward_batch(
        cls,
        device=None,
        tokens: torch.Tensor = None,
        num_mini_batches: int = 1,
        max_seq_length: int = 4096, # TODO: add to yaml
        prefill_query_length: int = (Config().chunk_size - Config().max_decode_batch_size) // Config().max_prefill_batch_size, # TODO: use config
        prefill_active_length: int = (Config().chunk_size - Config().max_decode_batch_size) // Config().max_prefill_batch_size,
        gen_prefill: bool = True,
        decode_batch_size: int = Config().max_decode_batch_size,
        decode_active_position: torch.Tensor = None,
        page_size = 256,
        cuda_lens = 1
    ):
        instance = cls()
        
        instance.batch_size = num_mini_batches
        page_size = page_size
     
        prefill_query_info = []
        offset = 0
        if gen_prefill and prefill_query_length != 0:
            for i in range(Config().max_prefill_batch_size):
                prefill_query_info.append(QueryInfo(i, prefill_query_length, max_seq_length, page_size, device, offset=offset))
                offset += max_seq_length // page_size

        decode_querys_info = []
        for i in range(min(decode_batch_size, cuda_lens)):
            query_info = QueryInfo(i+Config().max_prefill_batch_size, prefill_query_length, 256, page_size, device, is_prefill=False, offset=offset)
            offset += max_seq_length // page_size
            if tokens is not None:
                query_info.query_tokens[prefill_active_length:prefill_active_length + 1].copy_(tokens)            
            if decode_active_position is None:
                query_info.active_position = 255
            else: 
                query_info.active_position = decode_active_position[i]

            decode_querys_info.append(query_info)
        
        if prefill_query_length*Config().max_prefill_batch_size + len(decode_querys_info) < cuda_lens:
            decode_querys_info.append(query_info)

        instance.minibatch = ForwardBatchInput.ForwardMiniBatch(prefill_query_info, decode_querys_info, [0, 0], [prefill_active_length for _ in range(Config().max_prefill_batch_size)], device, page_size)
        
        return instance

    def fill(self, batch : sched_ext.BatchQueryTodo = None, query_manager: QueryManager = None, page_size = 256):
        if batch is None:
            return
        prefill_minibatches = batch.prefill_mini_batches
        decode_mini_batches = [item for sublist in batch.decode_mini_batches for item in sublist]

        prefill_querys_info = []
        prefill_s = []
        prefill_l = []
        decode_querys_info = []
        self.batch_size = 1
        for (id, s, l) in prefill_minibatches:
            prefill_querys_info.append(query_manager.query_map[id])
            prefill_s.append(s)
            prefill_l.append(l)
        for decode_batch_idx in decode_mini_batches:
            if query_manager.query_map[decode_batch_idx].decode_start_time is None:
                query_manager.query_map[decode_batch_idx].decode_start_time =time.time()
            decode_querys_info.append(query_manager.query_map[decode_batch_idx])

        self.minibatch.fill(prefill_querys_info, decode_querys_info, prefill_s, prefill_l, device=query_manager.device, page_size=page_size)



class ForwardBatchOutput:
    logits: list[torch.Tensor]
    num_batchs: int
    batch_sizes: list[int]
    generated_tokens_num: list[int]
    lm_start: list[int]
    
    temperatures: list[torch.Tensor]
    top_ps: list[torch.Tensor]

    def __init__(self):
        self.logits = []
        self.batch_sizes = []
        self.generated_tokens_num = []
        self.top_ps = []
        self.temperatures = []
        self.num_batchs = 1