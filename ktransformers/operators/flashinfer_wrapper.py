'''
Description  : flashinfer MLA wrapper
Author       : Boxin Zhang
Version      : 0.2.2
'''
import torch

flashinfer_enabled = False

try:
    import flashinfer
    flashinfer_enabled = False # disabled now, TODO:use new version of flashinfer and enable
    print("found flashinfer")
    
except ImportError:
    print("flashinfer not found, use triton for linux")

import math

def attention_ref(
    batch_size,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    qo_len = q.shape[0] // batch_size
    kv_len = k.shape[0] // batch_size
    num_qo_heads = q.shape[1]
    head_dim_qk = q.shape[2]
    head_dim_vo = v.shape[2]
    logits = (
        torch.einsum(
            "bmhd,bnhd->bhmn",
            q.view(batch_size, qo_len, num_qo_heads, head_dim_qk).float(),
            k.view(batch_size, kv_len, num_qo_heads, head_dim_qk).float(),
        )
        * sm_scale
    )

    #print("attn weights", logits)

    if causal:
        mask = (
            torch.arange(kv_len - qo_len, kv_len).unsqueeze(1)
            >= torch.arange(0, kv_len).unsqueeze(0)
        ).to(q.device)
    else:
        mask = torch.ones(qo_len, kv_len).to(q.device)

    logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
    lse_ref = torch.logsumexp(logits, -1).transpose(-1, -2)
    p = torch.softmax(logits, dim=-1)
    o_ref = (
        torch.einsum(
            "bhmn,bnhd->bmhd",
            p,
            v.view(batch_size, kv_len, num_qo_heads, head_dim_vo).float(),
        )
        .contiguous()
        .view(batch_size * qo_len, num_qo_heads, head_dim_vo)
        .to(q)
    )

    return o_ref, lse_ref * math.log2(math.e)

class MLAWrapper():
    def __init__(self,
                 max_batch_size,
                 max_pages,
                 use_cuda_graph = True,
                 device = "cuda",
                 ):
        self.float_workspace_buffer = torch.empty(128*1024*1024, dtype=torch.int8, device=device)
        self.max_batch_size = max_batch_size
        self.max_pages = max_pages
        if use_cuda_graph:
            if self.max_batch_size == 1:
                self.qo_indptr_buf = torch.arange(0, max_batch_size+1, dtype=torch.int32, device=device)
                self.kv_indptr_buf = torch.tensor([0, max_pages], dtype=torch.int32, device=device)
                self.kv_indices_buf = torch.arange(0, max_pages, dtype=torch.int32, device=device)
            else:
                self.qo_indptr_buf = torch.empty(max_batch_size+1, dtype=torch.int32, device=device)
                self.kv_indptr_buf = torch.empty(max_batch_size+1, dtype=torch.int32, device=device)
                self.kv_indices_buf = torch.empty(max_pages, dtype=torch.int32, device=device)
            self.kv_len_arr_buf = torch.empty(max_batch_size, dtype=torch.int32, device=device)
        else:
            self.qo_indptr_buf = None
            self.kv_indptr_buf = None
            self.kv_indices_buf = None
            self.kv_len_arr_buf = None
        self.wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            self.float_workspace_buffer,
            use_cuda_graph=False,
            qo_indptr=self.qo_indptr_buf,
            kv_indptr=self.kv_indptr_buf,
            kv_indices=self.kv_indices_buf,
            kv_len_arr=self.kv_len_arr_buf,
        )
        self.need_plan = True
    
    def plan(self,
             qo_indptr,
             kv_indptr,
             kv_indices,
             kv_len_arr,
             num_heads,
             head_dim_ckv,
             head_dim_kpe,
             page_size,
             sm_scale,
             q_data_type,
             kv_data_type,
             ):
        if qo_indptr is None:
            assert self.max_batch_size == 1
            qo_indptr = self.qo_indptr_buf
        if kv_indptr is None:
            assert self.max_batch_size == 1
            kv_indptr = self.kv_indptr_buf
        if kv_indices is None:
            assert self.max_batch_size == 1
            kv_indices = self.kv_indices_buf

        self.wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_len_arr,
            num_heads,
            head_dim_ckv,
            head_dim_kpe,
            page_size,
            False, # causal is False for decoding
            sm_scale,
            q_data_type,
            kv_data_type,
        )

    def run(self, q_nope, q_pe, ckv, k_pe, return_lse = False):
        return self.wrapper.run(q_nope, q_pe, ckv, k_pe, return_lse)

class MLAWrapperSingleton():
    wrappers:dict = {}

    @classmethod
    def get_instance(cls, device, *args, **kwargs)->MLAWrapper:
        if device not in cls.wrappers:
            cls.make_instance(device, *args, **kwargs)
        return cls.wrappers[device]
    
    @classmethod
    def make_instance(cls, device, *args, **kwargs):
        cls.wrappers[device] = MLAWrapper(*args, **kwargs, device=device)

    @classmethod
    def plan_all(cls, qo_indptr,
             kv_indptr,
             kv_indices,
             kv_len_arr,
             num_heads,
             head_dim_ckv,
             head_dim_kpe,
             page_size,
             sm_scale,
             q_data_type,
             kv_data_type,):
        for device, wrapper in cls.wrappers.items():
            kv_len_arr_cur_device = kv_len_arr.to(device)
            wrapper.plan(qo_indptr,
                kv_indptr,
                kv_indices,
                kv_len_arr_cur_device,
                num_heads,
                head_dim_ckv,
                head_dim_kpe,
                page_size,
                sm_scale,
                q_data_type,
                kv_data_type,)
            

if __name__ == "__main__":
    max_batch_size = 1
    max_pages = 1
    page_size = 64
    num_heads = 128

    q_nope = torch.randn((1, num_heads, 512), dtype=torch.bfloat16, device="cuda")
    q_pe = torch.randn((1, num_heads, 64), dtype=torch.bfloat16, device="cuda")
    ckv = torch.randn((max_pages, page_size, 512), dtype=torch.bfloat16, device="cuda")
    k_pe = torch.randn((max_pages, page_size, 64), dtype=torch.bfloat16, device="cuda")
    

    wrapper = MLAWrapperSingleton.get_instance(
        "cuda",
        max_batch_size,
        max_pages,
    )
    
    kv_len_arr = torch.tensor([10], dtype=torch.int32, device="cuda")

    wrapper.plan(
        None,
        None,
        None,
        kv_len_arr,
        128,
        512,
        64,
        page_size,
        192 ** (-0.5),
        torch.bfloat16,
        torch.bfloat16,
    )

    attn_output = wrapper.run(q_nope, q_pe, ckv, k_pe)

    k = (
        torch.cat([ckv, k_pe], dim=-1)
        .view(-1, 1, 512 + 64)
        .repeat_interleave(num_heads, dim=1)
    )
    v = ckv.view(-1, 1, 512).repeat_interleave(num_heads, dim=1)

    print(k[:10].shape)
    print(v[:10].shape)

    attn_ref, lse_ref = attention_ref(
        max_batch_size,
        torch.cat([q_nope, q_pe], dim=-1),
        k[:10],
        v[:10],
        False,
        192 ** (-0.5)
    )
    
    torch.testing.assert_close(attn_output, attn_ref, rtol=1e-3, atol=1e-3)
    print("test past")