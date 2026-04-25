'''
Description  : flashinfer MLA wrapper
Author       : Boxin Zhang
Version      : 0.2.3
'''
import torch
import os
from ktransformers.operators.triton_attention import decode_attention_fwd_grouped

flashinfer_enabled = False

try:
    import flashinfer
    flashinfer_enabled = True
    print("found flashinfer")
    
except ImportError:
    print("flashinfer not found, use triton for linux")

import math

def attention_ref_torch(
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
            self.batch_size_tensor_buf = torch.tensor([self.max_batch_size], dtype=torch.int32, device=device)
            self.kv_len_arr_buf = torch.empty(max_batch_size, dtype=torch.int32, device=device)
        else:
            self.qo_indptr_buf = None
            self.kv_indptr_buf = None
            self.kv_indices_buf = None
            self.kv_len_arr_buf = None
        self.wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            self.float_workspace_buffer,
            use_cuda_graph=use_cuda_graph,
            qo_indptr=self.qo_indptr_buf,
            kv_indptr=self.kv_indptr_buf,
            kv_indices=self.kv_indices_buf,
            kv_len_arr=self.kv_len_arr_buf,
            bsz_tensor=self.batch_size_tensor_buf,
            backend = "fa2",
        )
        self.need_plan = True

    
    def plan(self,
             qo_indptr,
             kv_indptr,
             kv_indices,
             kv_len_arr,
             bsz_tensor,
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
        if bsz_tensor is None:
            assert self.max_batch_size == 1
            bsz_tensor = self.batch_size_tensor_buf
        
        self.wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_len_arr,
            num_heads,
            head_dim_ckv,
            head_dim_kpe,
            page_size,
            True, # causal
            sm_scale,
            q_data_type,
            kv_data_type,
            bsz_tensor
        )

    def run(self, q_nope, q_pe, ckv, k_pe, return_lse = False):
        return self.wrapper.run(q_nope, q_pe, ckv, k_pe, return_lse = return_lse)

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
             bsz_tensor,
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
                bsz_tensor,
                num_heads,
                head_dim_ckv,
                head_dim_kpe,
                page_size,
                sm_scale,
                q_data_type,
                kv_data_type,)
            wrapper.need_plan = False
            
    @classmethod
    def need_plan_all(cls):
        for device, wrapper in cls.wrappers.items():
            wrapper.need_plan = True
        
    @classmethod
    def reset_buffer(cls):
        for device, wrapper in cls.wrappers.items():
            wrapper.qo_indptr_buf[1] = 1 # assert max_batch_size=1 here.
            
    @classmethod
    def update_buffer(cls, max_pages):
        for device, wrapper in cls.wrappers.items():
            wrapper.kv_indptr_buf[1] = max_pages # assert max_batch_size=1 here.
            wrapper.kv_indices_buf = torch.arange(0, max_pages, dtype=torch.int32, device=device)
            wrapper.wrapper._kv_indices_buf = wrapper.kv_indices_buf

def checksame():
    flashinfer_folder = "./flashinfer_output"
    flashinfer_folder = "./kv_cache_flashinfer"
    triton_folder = "./triton_output"
    triton_folder = "./kv_cache_triton"
    
    max_layer_id = 1
    max_forward_id = 2

    for forward_id in range(0, 19):
        print("forward_id", forward_id)
        for layer_id in range(max_layer_id):
            print(layer_id)
            #file_name = f"layer_{layer_id}_forward_{forward_id}_attn_output.pt"
            #file_name = f"layer_{layer_id}_forward_{forward_id}_q_pe.pt"
            file_name = f"layer_{layer_id}.pt"
            
            flashinfer_path = os.path.join(flashinfer_folder, file_name)
            triton_path = os.path.join(triton_folder, file_name)
            
            if not os.path.exists(triton_path):
                print(f"{file_name} not exist in {triton_folder}")
                continue
            if not os.path.exists(flashinfer_path):
                print(f"{file_name} not exist in {flashinfer_folder}")
                continue
            
            
            flashinfer_tensor = torch.load(flashinfer_path)[1:2, :62]#
            triton_tensor = torch.load(triton_path)[1:2, :62]#.squeeze(1)#
            try:
                torch.testing.assert_close(flashinfer_tensor, triton_tensor, rtol=1e-9, atol=1e-9)
            except AssertionError as e:
                print(e)

if __name__ == "__main__":
    
    #checksame()
    #exit(0)

    max_batch_size = 2
    max_batch_tokens = 256
    max_pages = 128
    page_size = 64
    num_heads = 128
    
    # warm-up
    kv_len = 4023
    q_len = 1
    q_nope_buf = torch.randn((max_batch_tokens, num_heads, 512), dtype=torch.bfloat16, device="cuda")
    q_pe_buf = torch.randn((max_batch_tokens, num_heads, 64), dtype=torch.bfloat16, device="cuda")
    kv_buf = torch.randn((max_pages, page_size, 576), dtype=torch.bfloat16, device="cuda")
    ckv, k_pe = torch.split(kv_buf, [512, 64], dim=-1)
    

    wrapper = MLAWrapperSingleton.get_instance(
        "cuda",
        max_batch_size,
        max_pages,
    )
    
    used_pages = (kv_len + page_size - 1)// page_size
    kv_len_arr = torch.tensor([kv_len], dtype=torch.int32, device="cuda")
    qo_indptr = torch.tensor([0, q_len], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, used_pages], dtype=torch.int32, device="cuda")
    kv_indices = torch.empty(max_pages, dtype=torch.int32, device="cuda")
    kv_indices[:used_pages] = torch.arange(0, used_pages, dtype=torch.int32, device="cuda")
    bsz_tensor = torch.tensor([1], dtype=torch.int32, device="cuda")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        bsz_tensor,
        128,
        512,
        64,
        page_size,
        192 ** (-0.5),
        torch.bfloat16,
        torch.bfloat16,
    )

    attn_output = wrapper.run(q_nope_buf[:q_len], q_pe_buf[:q_len], ckv, k_pe)
    print(attn_output.shape)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        attn_output = wrapper.run(q_nope_buf, q_pe_buf, ckv, k_pe)
    graph.replay()

    q = torch.cat([q_nope_buf, q_pe_buf], dim=-1)
    k = (
        torch.cat([ckv, k_pe], dim=-1)
        .view(-1, 1, 512 + 64)
        .repeat_interleave(num_heads, dim=1)
    )
    v = ckv.view(-1, 1, 512).repeat_interleave(num_heads, dim=1)
    attn_ref, lse_ref = attention_ref_torch(
        1,
        q[:q_len],
        k[:kv_len],
        v[:kv_len],
        True,
        192 ** (-0.5)
    )
    torch.testing.assert_close(attn_output[:q_len], attn_ref, rtol=5e-3, atol=5e-3)
    # warm-up finished

    kv_len = 512
    q_len = 128
    pages = max_pages
    used_pages = (kv_len + page_size - 1)// page_size
    q_nope = torch.randn((q_len*2, num_heads, 512), dtype=torch.bfloat16, device="cuda")
    q_nope[q_len:] = q_nope[:q_len]
    q_pe = torch.randn((q_len*2, num_heads, 64), dtype=torch.bfloat16, device="cuda")
    q_pe[q_len:] = q_pe[:q_len]
    kv_cache = torch.randn((max_pages, page_size, 576), dtype=torch.bfloat16, device="cuda")
    kv_cache[used_pages:2*used_pages] = kv_cache[:used_pages]
    ckv, k_pe = torch.split(kv_cache, [512, 64], dim=-1)
    
    kv_len_arr = torch.tensor([kv_len, kv_len], dtype=torch.int32, device="cuda")
    qo_indptr = torch.tensor([0, q_len, q_len*2], dtype=torch.int32, device="cuda")
    kv_indptr = torch.tensor([0, used_pages, used_pages*2], dtype=torch.int32, device="cuda")
    kv_indices = torch.empty(max_pages, dtype=torch.int32, device="cuda")
    kv_indices[:2*used_pages] = torch.arange(0, 2*used_pages, dtype=torch.int32, device="cuda")
    bsz_tensor = torch.tensor([2], dtype=torch.int32, device="cuda")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        bsz_tensor,
        128,
        512,
        64,
        page_size,
        192 ** (-0.5),
        torch.bfloat16,
        torch.bfloat16,
    )
    
    q_nope_buf.copy_(q_nope)
    q_pe_buf.copy_(q_pe)
    kv_buf[:pages].copy_(kv_cache)

    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()

    # ref_torch
    q = torch.cat([q_nope, q_pe], dim=-1)
    k = (
        torch.cat([ckv, k_pe], dim=-1)
        .view(-1, 1, 512 + 64)
        .repeat_interleave(num_heads, dim=1)
    )
    v = ckv.view(-1, 1, 512).repeat_interleave(num_heads, dim=1)
    attn_ref, lse_ref = attention_ref_torch(
        max_batch_size,
        q,
        k[:2*kv_len],
        v[:2*kv_len],
        True,
        192 ** (-0.5)
    )
    
    torch.testing.assert_close(attn_ref[:q_len], attn_ref[q_len:q_len*2], rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(attn_output[:q_len], attn_output[q_len:q_len*2], rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(attn_output[:q_len], attn_ref[:q_len], rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(attn_output[q_len:q_len*2], attn_ref[q_len:q_len*2], rtol=5e-3, atol=5e-3)
    #torch.testing.assert_close(attn_output[:q_len], attn_output[q_len:q_len*2], rtol=1e-9, atol=1e-9)
    #torch.testing.assert_close(attn_output, attn_ref, rtol=5e-3, atol=5e-3)

    exit(0)

    for forward_id in range(0, 1):
        print("forward_id", forward_id)
        for layer_id in range(1):
            print(layer_id)
            flashinfer_folder = "./kv_cache_flashinfer"
            forward_id = 17
            layer_id = 0
            file_name = f"layer_{layer_id}.pt"
            kv_cache_path = os.path.join(flashinfer_folder, file_name)
            flashinfer_folder = "./flashinfer_output"

            q_len = 1
            kv_len = 126
            file_name = f"layer_{layer_id}_forward_{forward_id}_q_nope.pt"
            q_nope = torch.load(os.path.join(flashinfer_folder, file_name)).view(q_len,128,512).to(device="cuda")
            file_name = f"layer_{layer_id}_forward_{forward_id}_q_pe.pt"
            q_pe = torch.load(os.path.join(flashinfer_folder, file_name)).view(q_len,128,64).to(device="cuda")
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv_cache = torch.load(kv_cache_path).to(device="cuda")
            pages, page_size, _, head_dim = kv_cache.shape
            kv_cache = kv_cache.view(pages, page_size, head_dim)
            ckv, k_pe = torch.split(kv_cache, [512, 64], dim=-1)
    
            kv_len_arr = torch.tensor([kv_len], dtype=torch.int32, device="cuda")
            qo_indptr = torch.tensor([0, q_len], dtype=torch.int32, device="cuda")
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
    
            q_nope_buf.copy_(q_nope)
            q_pe_buf.copy_(q_pe)
            kv_buf[:pages].copy_(kv_cache)

            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()

            # ref_torch
            k = (
                torch.cat([ckv, k_pe], dim=-1)
                .view(-1, 1, 512 + 64)
                .repeat_interleave(num_heads, dim=1)
            )
            v = ckv.view(-1, 1, 512).repeat_interleave(num_heads, dim=1)
            attn_ref, lse_ref = attention_ref_torch(
                max_batch_size,
                q,
                k[:kv_len],
                v[:kv_len],
                False,
                192 ** (-0.5)
            )
            torch.testing.assert_close(attn_output, attn_ref, rtol=1e-3, atol=1e-3)
    
            # ref_triton
            attn_logits = torch.empty(
                    (
                        max_batch_size,
                        num_heads,
                        4, #num_kv_splits # follow vLLM, fix it TODO
                        512 + 1, 
                    ),
                    dtype=torch.float32,
                    device = "cuda"
                )
            
            triton_ref = torch.zeros_like(q_nope)
            page_table = torch.arange(max_pages, dtype=torch.int32, device="cuda")
            ckv_with_pe = torch.cat([ckv, k_pe], dim=-1).contiguous().view(pages, page_size, 1, 576)
            ckv = ckv.view(pages, page_size, 1, 512)
            decode_attention_fwd_grouped(q, ckv_with_pe, ckv, triton_ref,
                page_table,
                kv_len_arr, attn_logits,
                4, #num_kv_splits # follow vLLM, fix it TODO
                192 ** (-0.5),
                page_size)

            torch.testing.assert_close(attn_output, triton_ref, rtol=1e-3, atol=1e-3)
            
            #file_name = f"./flashinfer_output/layer_{layer_id}_forward_{forward_id}_attn_output.pt"
            #ktrans_output = torch.load(file_name)
            #torch.testing.assert_close(attn_output, ktrans_output.squeeze(1), rtol=1e-3, atol=1e-3)
            print("test past")