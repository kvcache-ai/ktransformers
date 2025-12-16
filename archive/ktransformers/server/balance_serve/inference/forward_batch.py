'''
Date: 2024-11-12 14:15:16
LastEditors: Xie Weiyu ervinxie@qq.com
LastEditTime: 2024-11-26 08:12:49
'''
import torch
try:
    import torch_npu
    use_torch_npu = torch_npu.npu.is_available()
except:
    use_torch_npu = False
from ktransformers.server.balance_serve.settings import sched_ext
from ktransformers.server.balance_serve.inference.query_manager import QueryManager, QueryInfo
from typing import Union
import time
from ktransformers.server.config.config import Config

class ForwardMiniBatchCombine:
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

    def __str__(self):
        ret = ''
        ret += f'=====flash infer forward info:\n'
        ret += f'q_indptr: {self.q_indptr}, kv_indptr: {self.kv_indptr}, kv_indices: {self.kv_indices}\n'
        ret += f'kv_len: {self.kv_len}, kv_last_page_len: {self.kv_last_page_len}, bsz_tensor: {self.bsz_tensor}\n'
        ret += f'position_ids: {self.position_ids}, tokens: {self.tokens}\n'
        return ret


class ForwardMiniBatchSplit:
    # NPU 流程 prefill 和 decode 分开打包
    prefill_batch: int
    p_q_len: torch.Tensor               # (bsz)
    p_kv_len: torch.Tensor              # (bsz)
    p_position_ids: torch.Tensor        # (sum(q_len))
    p_tokens: torch.Tensor              # (sum(q_len))
    p_temperatures: torch.Tensor        # (bsz)
    p_top_ps: torch.Tensor              # (bsz)
    p_block_tables: torch.Tensor        # (bsz, max_page_num)
    p_logits_start: list

    decode_batch: int
    d_q_len: torch.Tensor
    d_kv_len: torch.Tensor
    d_position_ids: torch.Tensor
    d_tokens: torch.Tensor
    d_temperatures: torch.Tensor
    d_top_ps: torch.Tensor
    d_block_tables: torch.Tensor        # (bsz, max_page_num)
    d_logits_start: list

    chunk_size: int
    is_last_prefill_chunk: bool

    def __init__(
        self,
        prefill_querys_info: list[QueryInfo],
        decode_querys_info: list[QueryInfo],
        prefill_s: list[int] = None,
        prefill_l: list[int] = None,
        device=None,
        page_size: int = 256,
        max_page_num: int = 64,
        decode_padding_len: int = 1,
    ):
        # 统一 NPU 设备
        device = torch.device('npu')

        if prefill_s is None or prefill_l is None:
            raise ValueError(
                "[ForwardMiniBatchSplit.__init__] prefill_s / prefill_l 不能为空，chunk prefill 需要这两个参数"
            )

        # 过滤掉 None
        new_prefill_querys_info: list[QueryInfo] = [
            info for info in prefill_querys_info if info is not None
        ]
        batch_prefill = len(new_prefill_querys_info)
        batch_decode = len(decode_querys_info)

        self.prefill_batch = batch_prefill
        self.decode_batch = batch_decode
        self.batch_size = batch_prefill + batch_decode
        self.num_tokens = batch_decode * decode_padding_len + sum(prefill_l)

        self.chunk_size = prefill_l[0] if prefill_l else 0

        self.is_last_prefill_chunk = True
        for i, q in enumerate(new_prefill_querys_info):
            end_pos = prefill_s[i] + prefill_l[i]
            if end_pos < q.query_length:
                self.is_last_prefill_chunk = False
                break

        # ====================== Prefill 部分 ======================
        self.p_q_len = torch.tensor([], device=device, dtype=torch.int32)
        self.p_kv_len = torch.tensor([], device=device, dtype=torch.int32)
        self.p_position_ids = torch.tensor([], device=device, dtype=torch.int32)
        self.p_block_tables = -1 * torch.ones(
            [self.prefill_batch, max_page_num], device=device, dtype=torch.int32
        )
        self.p_tokens = torch.tensor([], device=device, dtype=torch.int32)

        self.p_temperatures = torch.tensor([], device=device, dtype=torch.float32)
        self.p_top_ps = torch.tensor([], device=device, dtype=torch.float32)
        self.p_logits_start: list[int] = []

        for i, prefill_query_info in enumerate(new_prefill_querys_info):
            qid = getattr(prefill_query_info, "id", -1)

            past_len = int(prefill_query_info.active_position)
            start = int(prefill_s[i])                            # current chunk's start position in query_tokens
            chunk_len = int(prefill_l[i])
            kv_len = past_len + chunk_len
            prefill_kv_block_len = (kv_len + page_size - 1) // page_size

            # Q length = current chunk length
            self.p_q_len = torch.concat(
                (
                    self.p_q_len,
                    torch.tensor([chunk_len], device=device, dtype=torch.int32),
                ),
                dim=0,
            )
            self.p_kv_len = torch.concat(
                (
                    self.p_kv_len,
                    torch.tensor([kv_len], device=device, dtype=torch.int32),
                ),
                dim=0,
            )

            self.p_block_tables[i, :prefill_kv_block_len] = prefill_query_info.block_index[
                :prefill_kv_block_len
            ]

            self.p_position_ids = torch.concat(
                (
                    self.p_position_ids,
                    torch.arange(
                        start,
                        start + chunk_len,
                        device=device,
                        dtype=torch.int32,
                    ),
                ),
                dim=0,
            )

            self.p_tokens = torch.concat(
                (
                    self.p_tokens,
                    prefill_query_info.query_tokens[start : start + chunk_len],
                ),
                dim=0,
            )

            self.p_logits_start.append(
                chunk_len - 1
                if len(self.p_logits_start) == 0
                else sum(prefill_l[: i + 1]) - 1
            )

            self.p_temperatures = torch.concat(
                (
                    self.p_temperatures,
                    torch.tensor(
                        [prefill_query_info.temperature],
                        device=device,
                        dtype=torch.float32,
                    ),
                ),
                dim=0,
            )
            self.p_top_ps = torch.concat(
                (
                    self.p_top_ps,
                    torch.tensor(
                        [prefill_query_info.top_p],
                        device=device,
                        dtype=torch.float32,
                    ),
                ),
                dim=0,
            )

        # ====================== Decode ======================
        self.d_q_len = torch.tensor([], device=device, dtype=torch.int32)
        self.d_kv_len = torch.tensor([], device=device, dtype=torch.int32)
        self.d_position_ids = torch.tensor([], device=device, dtype=torch.int32)
        self.d_block_tables = -1 * torch.ones(
            [self.decode_batch, max_page_num], device=device, dtype=torch.int32
        )
        self.d_tokens = torch.tensor([], device=device, dtype=torch.int32)

        self.d_temperatures = torch.tensor([], device=device, dtype=torch.float32)
        self.d_top_ps = torch.tensor([], device=device, dtype=torch.float32)
        self.d_logits_start: list[int] = []

        for i, decode_query_info in enumerate(decode_querys_info):
            qid = getattr(decode_query_info, "id", -1)
            past_len = int(decode_query_info.active_position)
            decode_kv_block_len = (past_len + decode_padding_len + page_size - 1) // page_size

            self.d_q_len = torch.concat(
                (
                    self.d_q_len,
                    torch.tensor(
                        [decode_padding_len], device=device, dtype=torch.int32
                    ),
                ),
                dim=0,
            )
            self.d_kv_len = torch.concat(
                (
                    self.d_kv_len,
                    torch.tensor(
                        [past_len + decode_padding_len],
                        device=device,
                        dtype=torch.int32,
                    ),
                ),
                dim=0,
            )

            self.d_block_tables[i, :decode_kv_block_len] = decode_query_info.block_index[
                :decode_kv_block_len
            ]

            self.d_position_ids = torch.concat(
                (
                    self.d_position_ids,
                    torch.arange(
                        past_len,
                        past_len + decode_padding_len,
                        device=device,
                        dtype=torch.int32,
                    ),
                ),
                dim=0,
            )

            if past_len > 0:
                self.d_tokens = torch.concat(
                    (
                        self.d_tokens,
                        decode_query_info.query_tokens[
                            past_len : past_len + decode_padding_len
                        ],
                    ),
                    dim=0,
                )
            else:
                self.d_tokens = torch.concat(
                    (
                        self.d_tokens,
                        torch.tensor(
                            [0] * decode_padding_len,
                            device=device,
                            dtype=torch.int32,
                        ),
                    ),
                    dim=0,
                )

            self.d_logits_start.append(
                0
                if len(self.d_logits_start) == 0
                else self.d_logits_start[-1] + decode_padding_len
            )

            self.d_temperatures = torch.concat(
                (
                    self.d_temperatures,
                    torch.tensor(
                        [decode_query_info.temperature],
                        device=device,
                        dtype=torch.float32,
                    ),
                ),
                dim=0,
            )
            self.d_top_ps = torch.concat(
                (
                    self.d_top_ps,
                    torch.tensor(
                        [decode_query_info.top_p],
                        device=device,
                        dtype=torch.float32,
                    ),
                ),
                dim=0,
            )

        self.p_q_len = self.p_q_len.contiguous()
        self.p_kv_len = self.p_kv_len.contiguous()
        self.p_block_tables = self.p_block_tables.contiguous()
        self.p_position_ids = self.p_position_ids.contiguous()
        self.p_tokens = self.p_tokens.contiguous()

        if self.decode_batch > 1:
            self.d_q_len = self.d_q_len.reshape(self.decode_batch, -1).contiguous()
            self.d_kv_len = self.d_kv_len.reshape(self.decode_batch, -1).contiguous()
            self.d_kv_len_list = self.d_kv_len.flatten().tolist()
            self.d_block_tables = self.d_block_tables.contiguous()
            self.d_position_ids = self.d_position_ids.reshape(self.decode_batch, -1).contiguous()
            self.d_tokens = self.d_tokens.reshape(self.decode_batch, -1).contiguous()
        else:
            self.d_q_len = self.d_q_len.contiguous()
            self.d_kv_len = self.d_kv_len.contiguous()
            self.d_kv_len_list = self.d_kv_len.flatten().tolist()
            self.d_block_tables = self.d_block_tables.contiguous()
            self.d_position_ids = self.d_position_ids.contiguous()
            self.d_tokens = self.d_tokens.contiguous()

        self.bsz_tensor = torch.tensor([self.batch_size], device=device, dtype=torch.int32)


    def fill(
        self,
        prefill_querys_info: list[QueryInfo],
        decode_querys_info: list[QueryInfo],
        prefill_s: list[int] = None,
        prefill_l: list[int] = None,
        decode_padding_len: int = 1,
        device=None,
        page_size: int = 256,
        max_page_num: int = 64,
    ):
        device = torch.device('npu')

        if prefill_s is None or prefill_l is None:
            raise ValueError(
                "[ForwardMiniBatchSplit.fill] prefill_s / prefill_l 不能为空，chunk prefill 需要这两个参数"
            )

        page_size = 128

        new_prefill_querys_info: list[QueryInfo] = [
            info for info in prefill_querys_info if info is not None
        ]
        batch_prefill = len(new_prefill_querys_info)
        batch_decode = len(decode_querys_info)

        self.prefill_batch = batch_prefill
        self.decode_batch = batch_decode
        self.batch_size = batch_prefill + batch_decode
        self.num_tokens = batch_decode * decode_padding_len + sum(prefill_l)

        self.chunk_size = prefill_l[0] if prefill_l else 0
        self.is_last_prefill_chunk = True
        for i, q in enumerate(new_prefill_querys_info):
            end_pos = prefill_s[i] + prefill_l[i]
            if end_pos < q.query_length:
                self.is_last_prefill_chunk = False
                break

        # ---------- Prefill ----------
        self.p_q_len = torch.tensor([], device=device, dtype=torch.int32)
        self.p_kv_len = torch.tensor([], device=device, dtype=torch.int32)
        new_p_position_ids = torch.tensor([], device=device, dtype=torch.int32)
        self.p_block_tables = torch.zeros(
            [self.prefill_batch, max_page_num], device=device, dtype=torch.int32
        )
        new_p_tokens = torch.tensor([], device=device, dtype=torch.int32)

        self.p_temperatures = torch.tensor([], device=device, dtype=torch.float32)
        self.p_top_ps = torch.tensor([], device=device, dtype=torch.float32)
        self.p_logits_start = []

        for i, prefill_query_info in enumerate(new_prefill_querys_info):
            qid = getattr(prefill_query_info, "id", -1)
            past_len = int(prefill_query_info.active_position)
            start = int(prefill_s[i])
            chunk_len = int(prefill_l[i])

            kv_len = past_len + chunk_len
            prefill_kv_block_len = (kv_len + page_size - 1) // page_size

            self.p_q_len = torch.concat(
                (
                    self.p_q_len,
                    torch.tensor([chunk_len], device=device, dtype=torch.int32),
                ),
                dim=0,
            )
            self.p_kv_len = torch.concat(
                (
                    self.p_kv_len,
                    torch.tensor([kv_len], device=device, dtype=torch.int32),
                ),
                dim=0,
            )
            self.p_block_tables[i, :prefill_kv_block_len] = prefill_query_info.block_index[
                :prefill_kv_block_len
            ]

            new_p_position_ids = torch.concat(
                (
                    new_p_position_ids,
                    torch.arange(
                        start,
                        start + chunk_len,
                        device=device,
                        dtype=torch.int32,
                    ),
                ),
                dim=0,
            )
            new_p_tokens = torch.concat(
                (
                    new_p_tokens,
                    prefill_query_info.query_tokens[start : start + chunk_len],
                ),
                dim=0,
            )

            self.p_logits_start.append(
                chunk_len - 1 if len(self.p_logits_start) == 0 else sum(prefill_l[: i + 1]) - 1
            )

            self.p_temperatures = torch.concat(
                (
                    self.p_temperatures,
                    torch.tensor(
                        [prefill_query_info.temperature],
                        device=device,
                        dtype=torch.float32,
                    ),
                ),
                dim=0,
            )
            self.p_top_ps = torch.concat(
                (
                    self.p_top_ps,
                    torch.tensor(
                        [prefill_query_info.top_p],
                        device=device,
                        dtype=torch.float32,
                    ),
                ),
                dim=0,
            )

        if new_p_position_ids.numel() > 0:
            self.p_position_ids = new_p_position_ids.contiguous()
        if new_p_tokens.numel() > 0:
            self.p_tokens = new_p_tokens.contiguous()

        # ---------- Decode ----------
        self.d_q_len = torch.zeros(
            [1] * self.decode_batch, device=device, dtype=torch.int32
        )
        self.d_kv_len = torch.tensor([], device=device, dtype=torch.int32)
        new_d_position_ids = torch.tensor([], device=device, dtype=torch.int32)
        new_d_block_tables = -1 * torch.ones(
            [self.decode_batch, max_page_num], device=device, dtype=torch.int32
        )
        new_d_tokens = torch.tensor([], device=device, dtype=torch.int32)

        self.d_logits_start = []
        self.d_temperatures = torch.tensor([], device=device, dtype=torch.float32)
        self.d_top_ps = torch.tensor([], device=device, dtype=torch.float32)

        for i, decode_query_info in enumerate(decode_querys_info):
            qid = getattr(decode_query_info, "id", -1)
            past_len = int(decode_query_info.active_position)
            decode_kv_block_len = (past_len + decode_padding_len + page_size - 1) // page_size

            self.d_kv_len = torch.concat(
                (
                    self.d_kv_len,
                    torch.tensor(
                        [past_len + decode_padding_len],
                        device=device,
                        dtype=torch.int32,
                    ),
                ),
                dim=0,
            )
            new_d_block_tables[i, :decode_kv_block_len] = decode_query_info.block_index[
                :decode_kv_block_len
            ]

            new_d_position_ids = torch.concat(
                (
                    new_d_position_ids,
                    torch.arange(
                        past_len,
                        past_len + decode_padding_len,
                        device=device,
                        dtype=torch.int32,
                    ),
                ),
                dim=0,
            )

            if past_len > 0:
                new_d_tokens = torch.concat(
                    (
                        new_d_tokens,
                        decode_query_info.query_tokens[
                            past_len : past_len + decode_padding_len
                        ],
                    ),
                    dim=0,
                )
            else:
                new_d_tokens = torch.concat(
                    (
                        new_d_tokens,
                        torch.tensor(
                            [0] * decode_padding_len,
                            device=device,
                            dtype=torch.int32,
                        ),
                    ),
                    dim=0,
                )

            self.d_logits_start.append(
                0
                if len(self.d_logits_start) == 0
                else self.d_logits_start[-1] + decode_padding_len
            )

            self.d_temperatures = torch.concat(
                (
                    self.d_temperatures,
                    torch.tensor(
                        [decode_query_info.temperature],
                        device=device,
                        dtype=torch.float32,
                    ),
                ),
                dim=0,
            )
            self.d_top_ps = torch.concat(
                (
                    self.d_top_ps,
                    torch.tensor(
                        [decode_query_info.top_p],
                        device=device,
                        dtype=torch.float32,
                    ),
                ),
                dim=0,
            )

            if len(decode_querys_info) > 1:
                self.d_position_ids[i].copy_(new_d_position_ids[i])
                self.d_tokens[i].copy_(new_d_tokens[i])
                self.d_block_tables[i].copy_(new_d_block_tables[i])
            else:
                self.d_position_ids[:new_d_position_ids.size(0)].copy_(new_d_position_ids)
                self.d_tokens[:new_d_tokens.size(0)].copy_(new_d_tokens)
                self.d_block_tables[0].copy_(new_d_block_tables[0])


        self.p_q_len = self.p_q_len.contiguous()
        self.p_kv_len = self.p_kv_len.contiguous()
        self.p_block_tables = self.p_block_tables.contiguous()

        self.d_q_len = self.d_q_len.contiguous()
        self.d_kv_len = self.d_kv_len.contiguous()
        self.d_kv_len_list = self.d_kv_len.flatten().tolist()

        self.bsz_tensor = torch.tensor([self.batch_size], device=device, dtype=torch.int32)



    def __str__(self):
        ret = ''
        ret += '=======Prefill forward info:\n'
        ret += f'batch: {self.prefill_batch}, qLen: {self.p_q_len}, kvLen: {self.p_kv_len}\n'
        ret += f'tokens: {self.p_tokens}, posIdx: {self.p_position_ids}, block_tables: {self.p_block_tables}\n'
        ret += '=======Decode forward info:\n'
        ret += f'batch: {self.decode_batch}, qLen: {self.d_q_len}, kvLen: {self.d_kv_len}\n'
        ret += f'tokens: {self.d_tokens}, posIdx: {self.d_position_ids}, block_tables: {self.d_block_tables}\n'
        ret += f'chunk_size={self.chunk_size}, is_last_prefill_chunk={self.is_last_prefill_chunk}\n'
        return ret



class ForwardBatchInput:

    forward_minibatchs: list[Union[ForwardMiniBatchSplit, ForwardMiniBatchCombine]]
    decode_mini_batches: list[Union[ForwardMiniBatchSplit, ForwardMiniBatchCombine]]
    batch_size: int
    minibatch: Union[ForwardMiniBatchSplit, ForwardMiniBatchCombine]

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
        for (qid, s, l) in prefill_minibatches:
            prefill_querys_info.append(query_manager.query_map[qid])
            prefill_s.append(s)
            prefill_l.append(l)
        for decode_qid in decode_mini_batches:
            qinfo = query_manager.query_map[decode_qid]
            if qinfo.decode_start_time is None:
                qinfo.decode_start_time = time.time()
            decode_querys_info.append(qinfo)

        if use_torch_npu:
            minibatch = ForwardMiniBatchSplit(prefill_querys_info, decode_querys_info, prefill_s, prefill_l, device = query_manager.device, page_size = query_manager.page_size)
        else:
            minibatch = ForwardMiniBatchCombine(prefill_querys_info, decode_querys_info, prefill_s, prefill_l, device = query_manager.device, page_size = query_manager.page_size)
        self.minibatch = minibatch

    @classmethod
    def gen_max_forward_batch(
        cls,
        device=None,
        tokens: torch.Tensor = None,
        num_mini_batches: int = 1,
        max_seq_length: int = 1024, # TODO: add to yaml
        prefill_query_length: int = (Config().chunk_size - Config().max_decode_batch_size) // Config().max_prefill_batch_size, # TODO: use config
        prefill_active_length: int = (Config().chunk_size - Config().max_decode_batch_size) // Config().max_prefill_batch_size,
        gen_prefill: bool = True,
        decode_batch_size: int = Config().max_decode_batch_size,
        decode_query_length: int = 1,
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
            query_info = QueryInfo(i+Config().max_prefill_batch_size, decode_query_length, max_seq_length, page_size, device, is_prefill=False, offset=offset)
            offset += max_seq_length // page_size
            if tokens is not None:
                query_info.query_tokens[prefill_active_length:prefill_active_length + decode_query_length].copy_(tokens)            
            if decode_active_position is None:
                query_info.active_position = prefill_active_length
            else: 
                query_info.active_position = decode_active_position[i]

            decode_querys_info.append(query_info)
        
        if prefill_query_length * Config().max_prefill_batch_size + len(decode_querys_info) < cuda_lens:
            decode_querys_info.append(query_info)
        if use_torch_npu:
            instance.minibatch = ForwardMiniBatchSplit(prefill_query_info, decode_querys_info, [0, 0],
                                                [prefill_active_length for _ in range(Config().max_prefill_batch_size)],
                                                device, page_size, decode_padding_len=decode_query_length)
        else:
            instance.minibatch = ForwardMiniBatchCombine(prefill_query_info, decode_querys_info, [0, 0], [prefill_active_length for _ in range(Config().max_prefill_batch_size)], device, page_size)
        
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
    pre_hidden_states: list[torch.Tensor]
    num_batchs: int
    batch_sizes: list[int]
    generated_tokens_num: list[int]
    lm_start: list[int]
    
    temperatures: list[torch.Tensor]
    top_ps: list[torch.Tensor]

    def __init__(self):
        self.num_batchs = 0
        self.lm_start = []
        self.logits = []
        self.batch_sizes = []
        self.generated_tokens_num = []
        self.top_ps = []
        self.temperatures = []
        self.pre_hidden_states = []
        pass

    def merge(self, new_output):
        self.logits.extend(new_output.logits)
        self.num_batchs += new_output.num_batchs
        self.batch_sizes.extend(new_output.batch_sizes)
        self.generated_tokens_num.extend(new_output.generated_tokens_num)
        self.top_ps.extend(new_output.top_ps)
        self.temperatures.extend(new_output.temperatures)
        self.lm_start.extend(new_output.lm_start)
        self.pre_hidden_states.extend(new_output.pre_hidden_states)

    def __str__(self):
        logits_shape = [t.shape for t in self.logits]
        ret = ''
        ret += f'=======Combined output info:\n'
        ret += f'logits: {self.logits}\n'
        ret += f'logits(size): {logits_shape}, num_batchs: {self.num_batchs}, kvLen: {self.generated_tokens_num}\n'
        ret += f'top_ps: {self.top_ps}, temperatures: {self.temperatures}, pre_hidden_states num: {len(self.pre_hidden_states)}\n'
        if len(self.pre_hidden_states) != 0:
            for idx in range(len(self.pre_hidden_states)):
                ret += f'idx: {idx}, pre_hidden_states shape: {self.pre_hidden_states[idx].shape}\n'    
        return ret