'''
Date: 2024-11-14 12:23:45
LastEditors: djw
LastEditTime: 2024-11-20 04:06:23
'''
import torch
from ktransformers.server.balance_serve.settings import sched_ext
import random
import time

class QueryInfo:
    id: int
    active_position: int
    query_length: int
    is_prefill: int
    block_index: torch.Tensor
    query_tokens: torch.Tensor
    stop_criteria: list[torch.Tensor]

    temperature: float
    top_p: float

    max_length: int 

    def __init__(self, id, query_length: int, max_length: int, page_size: int, device: torch.device, is_prefill: bool = True, offset: int = 0, active_position: int = 0, temperature: float = 0.01, top_p: float = 1.0):
        self.id = id
        self.is_prefill = is_prefill
        self.active_position = active_position
        self.max_length = max_length - 1
        self.query_tokens = torch.zeros((max_length,), dtype=torch.int, device = device)
        self.stop_criteria = []
        self.block_index = torch.arange(offset, offset + (max_length + active_position + page_size - 1) // page_size, dtype=torch.int, device = device)
        self.query_length = query_length
        self.enqueue_time = time.time()
        self.decode_start_time = None
        self.speculative_token = {} # {position: (accept, token)}

        self.temperature = temperature
        self.top_p = top_p

    def check_stop(self):
        if self.active_position >= self.max_length - 2:
            return True

        # 遍历每个停止条件
        for stop_tensor in self.stop_criteria:
            stop_len = len(stop_tensor)
            
            # 如果停止条件比 query_tokens 长，跳过
            if stop_len >= self.active_position:
                continue
            
            #print(f"stop_tensor: {stop_tensor}, stop_len: {stop_len}, active_position: {self.active_position}, query_token: {self.query_tokens[self.active_position - stop_len - 1:self.active_position - 1]}")

            if (torch.equal(self.query_tokens[self.active_position - stop_len - 1:self.active_position - 1], stop_tensor) and self.active_position) or self.max_length <= self.active_position + 3:
                self.life_time = time.time() - self.enqueue_time
                self.decode_duration_time = time.time() - self.decode_start_time
                self.decode_tps = (self.active_position -  self.query_length) / self.decode_duration_time
                print(f"prefill length: {self.query_length}, prefill time: {self.prefill_duration_time}, prefill tps {self.prefill_tps}, decode length: {self.active_position -  self.query_length}, decode time: {self.decode_duration_time}, decode tps {self.decode_tps}")
                return True  # 找到匹配的停止条件
                
        
        return False  # 没有找到任何停止条件


    def print(self):
        print(f"active_position: {self.active_position}, query_length: {self.query_length}, is_prefill: {self.is_prefill}")
        print(f"block_index_shape: {self.block_index.shape}, query_tokens_shape: {self.query_tokens.shape}")


class QueryManager:

    page_size: int = 256
    device: torch.device
    query_map : dict[int, QueryInfo]

    def __init__(self, page_size = 256, device = torch.device('cuda')):
        self.page_size = page_size
        self.device = device
        self.query_map = {}

    def add_query(self, batch: sched_ext.BatchQueryTodo):

        for i in range(len(batch.query_ids)):
            id = batch.query_ids[i]
            if id not in self.query_map:
                print(f"add query id: {id}, batch.query_lengths: {batch.query_lengths[i]}, batch_query_tokens: {batch.query_tokens[i].shape}, batch.block_indexes: {batch.block_indexes[i]}")
                query_info = QueryInfo(id=id, query_length=batch.query_lengths[i], max_length=batch.query_tokens[i].size(0) + 1, page_size=self.page_size, device=self.device, temperature=batch.sample_options[i].temperature, top_p=batch.sample_options[i].top_p)
                query_info.query_tokens[:query_info.query_length].copy_(batch.query_tokens[i][:query_info.query_length].to(self.device))
                
                for stop_token_list in batch.stop_criteria[i]:
                    query_info.stop_criteria.append(torch.tensor(stop_token_list, dtype=torch.int, device = self.device))

                block_num = batch.block_indexes[i].size(0)
                query_info.block_index[:block_num].copy_(batch.block_indexes[i].to(self.device))

                self.query_map[id] = query_info
                
                prefill_mini_batches = batch.prefill_mini_batches
                for (prefill_id, s, l) in prefill_mini_batches:
                    if prefill_id == id:
                        self.query_map[prefill_id].active_position = s


    def update(self, batch: sched_ext.BatchQueryTodo) -> list[sched_ext.QueryUpdate]:
        query_updates = []

        prefill_mini_batches = batch.prefill_mini_batches

        for (id, s, l) in prefill_mini_batches:

            if id not in self.query_map:
                assert False, f"query id {id} not found in query_map"

            # update query_info
            query_info = self.query_map[id]
            query_info.active_position += l

            if query_info.active_position >= query_info.query_length and query_info.is_prefill:
                query_info.is_prefill = False
                query_info.prefill_duration_time = time.time() - query_info.enqueue_time
                query_info.prefill_tps = query_info.query_length / query_info.prefill_duration_time
                

            # generate schedule query_update
            query_update = sched_ext.QueryUpdate()
            query_update.id = id
            query_update.ok = True
            query_update.is_prefill = query_info.is_prefill
            query_update.active_position = query_info.active_position
            # if(not query_info.is_prefill):
            query_updates.append(query_update)


        decode_mini_batches = batch.decode_mini_batches

        for ids in decode_mini_batches:
            for id in ids:
                if id not in self.query_map:
                    assert False, f"query id {id} not found in query_map"

                query_info = self.query_map[id]
                query_info.active_position += 1

                query_update = sched_ext.QueryUpdate()
                query_update.id = id
                query_update.ok = True
                query_update.is_prefill = query_info.is_prefill

                query_update.decode_done = query_info.check_stop()

                query_update.active_position = query_info.active_position
                query_updates.append(query_update)

        return query_updates
