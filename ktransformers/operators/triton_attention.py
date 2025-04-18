# Adapted from
# https://github.com/sgl-project/sglang/blob/9f635ea50de920aa507f486daafba26a5b837574/python/sglang/srt/layers/attention/triton_ops/decode_attention.py
# which was originally adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py
# 2025 - Modified by Shanghai Iluvatar CoreX Semiconductor Co., Ltd. All Rights Reserved.

import triton
import triton.language as tl
from ktransformers.util.vendors import device_manager, get_device, to_device, GPUVendor
import torch
IS_COREX_TORCH = device_manager.gpu_vendor == GPUVendor.Iluvatar

@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    Att_Out,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)
    split_kv_id = tl.program_id(2)

    if kv_group_num > BLOCK_H:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = cur_batch

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[
        None, :]
    q = tl.load(Q + offs_q,
                mask=(mask_h[:, None]) & (mask_d[None, :]),
                other=0.0)

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (cur_batch * stride_qbs + cur_head[:, None] * stride_qh +
                   offs_dpe[None, :])
        qpe = tl.load(Q + off_qpe,
                      mask=(mask_h[:, None]) & (mask_dpe[None, :]),
                      other=0.0)

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split,
                              cur_batch_seq_len)
    
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_page_number = tl.load(
                Req_to_tokens + stride_req_to_tokens_b * cur_batch_req_idx +
                offs_n // PAGE_SIZE,
                mask=offs_n < split_kv_end,
                other=0,
            )
            kv_loc = kv_page_number * PAGE_SIZE + offs_n % PAGE_SIZE
            offs_buf_k = (kv_loc[None, :] * stride_buf_kbs +
                          cur_kv_head * stride_buf_kh + offs_d[:, None])
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )
            qk = tl.dot(q, k.to(q.dtype))
            
            if BLOCK_DPE > 0:
                offs_buf_kpe = (kv_loc[None, :] * stride_buf_kbs +
                                cur_kv_head * stride_buf_kh +
                                offs_dpe[:, None])
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) &
                    (mask_dpe[:, None]),
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(mask_h[:, None] & (offs_n[None, :] < split_kv_end),
                          qk, float("-inf"))

            offs_buf_v = (kv_loc[:, None] * stride_buf_vbs +
                          cur_kv_head * stride_buf_vh + offs_dv[None, :])
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 1), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max[:, None])
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v)

            e_sum = e_sum * re_scale + tl.sum(p, 1)
            e_max = n_e_max

        offs_mid_o = (cur_batch * stride_mid_ob +
                      cur_head[:, None] * stride_mid_oh +
                      split_kv_id * stride_mid_os + offs_dv[None, :])

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (cur_batch * stride_mid_ob + cur_head * stride_mid_oh +
                        split_kv_id * stride_mid_os + Lv)

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )

def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    Req_to_tokens,
    B_Seqlen,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap,
):
    BLOCK = 32
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    # [TODO] work around shmem limit on MI3xx
    
    # TODO: support hip
    if device_manager.gpu_vendor == GPUVendor.AMD and Lk >= 576:
       BLOCK = 16

    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    BLOCK_H = 16
    NUM_KV_SPLITS = num_kv_splits
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        NUM_KV_SPLITS,
    )

    extra_kargs = {}
    # TODO: support hip
    """
    if is_hip_:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {
            "waves_per_eu": 4,
            "matrix_instr_nonkdim": 16,
            "kpack": 2
        }
    """
    num_stages = 2
    if IS_COREX_TORCH:
        num_stages = 1
    _fwd_grouped_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        Req_to_tokens,
        B_Seqlen,
        att_out,
        Req_to_tokens.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        k_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-3),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(-2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        num_warps=4,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        **extra_kargs,
    )

@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    o,
    B_Seqlen,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split,
                                  cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os,
                         mask=mask_d,
                         other=0.0)
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )

def _decode_softmax_reducev_fwd(
    logits,
    q,
    o,
    v_buffer,
    b_seq_len,
    num_kv_splits,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    NUM_KV_SPLITS = num_kv_splits

    extra_kargs = {}
    # TODO: support hip
    """
    if is_hip_:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {
            "waves_per_eu": 4,
            "matrix_instr_nonkdim": 16,
            "kpack": 2
        }
    """
    
    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        logits,
        o,
        b_seq_len,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )

def decode_attention_fwd_grouped(
    q,
    k_buffer,
    v_buffer,
    o,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap=0.0,
):
    if IS_COREX_TORCH:
        num_heads_kv = k_buffer.shape[-2]
        Lk = k_buffer.shape[-1]
        Lv = v_buffer.shape[-1]
        no_mask_condition = (
            (Lk == 576 or Lk == 512 or Lk == triton.next_power_of_2(Lk)) and
            (Lv == triton.next_power_of_2(Lv)) and
            q.shape[1] == triton.next_power_of_2(q.shape[1])
        )
        is_mqa = num_heads_kv == 1


        if no_mask_condition and is_mqa:
            return decode_attention_fwd_grouped_no_mask_gqa(
                q,
                k_buffer,
                v_buffer,
                o,
                req_to_token,
                b_seq_len,
                attn_logits,
                num_kv_splits,
                sm_scale,
                page_size,
                logit_cap
            )
        else:
            _decode_grouped_att_m_fwd(
                q,
                k_buffer,
                v_buffer,
                attn_logits,
                req_to_token,
                b_seq_len,
                num_kv_splits,
                sm_scale,
                page_size,
                logit_cap,
            )
            _decode_softmax_reducev_fwd(attn_logits, q, o, v_buffer, b_seq_len,
                                        num_kv_splits)
    else:
            _decode_grouped_att_m_fwd(
                q,
                k_buffer,
                v_buffer,
                attn_logits,
                req_to_token,
                b_seq_len,
                num_kv_splits,
                sm_scale,
                page_size,
                logit_cap,
            )

            _decode_softmax_reducev_fwd(attn_logits, q, o, v_buffer, b_seq_len,
                                        num_kv_splits)

@triton.jit
def _fwd_grouped_kernel_v3_2k_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale,
    Req_to_tokens,
    B_Seqlen,
    Att_Out,
    Lse,
    stride_req_to_tokens_b,
    stride_qbs,
    stride_qh,
    stride_buf_k0,
    stride_buf_k1,
    stride_buf_k2,
    stride_buf_v0,
    stride_buf_v1,
    stride_buf_v2,
    stride_o_ob,
    stride_o_oh,
    stride_o_os,
    stride_l_0,
    stride_l_1,
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
    N_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_split = tl.program_id(2)

    if kv_group_num > BLOCK_H:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H
    mask_h = mask_h & (cur_head < q_head_num)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_k = tl.arange(0, BLOCK_K)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_req_idx = cur_batch

    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_k[None, :]
    q_nope_ptrs = Q + offs_q

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        off_qpe = (cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :])
        q_pe = tl.load(Q + off_qpe)

    # Init kv cache
    offs_page = tl.arange(0, PAGE_SIZE)
    k_c_int_ptrs = K_Buffer + offs_page[None, :] * stride_buf_k1 + offs_k[:, None]
    v_c_init_ptrs = V_Buffer + offs_page[:, None] * stride_buf_v1 + offs_dv[None, :]
    if BLOCK_DPE > 0:
        offs_k_pe_init = K_Buffer + offs_page[None, :] * stride_buf_k1 + offs_dpe[:, None]
    
    # Get page range
    num_pages = tl.cdiv(cur_batch_seq_len, PAGE_SIZE)
    page_valid = cur_batch_seq_len % PAGE_SIZE
    num_pages_per_split = num_pages // N_KV_SPLITS
    ramainer_pages = num_pages % N_KV_SPLITS
    if cur_kv_split < ramainer_pages:
        start_page = cur_kv_split * (num_pages_per_split + 1)
        end_page = start_page + num_pages_per_split + 1
    else:
        start_page = cur_kv_split * num_pages_per_split + ramainer_pages
        end_page = start_page + num_pages_per_split

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)

    if start_page == end_page:
        return
    for page_idx in range(start_page, end_page):
        kv_page_loc = tl.load(
            Req_to_tokens + cur_batch_req_idx * stride_req_to_tokens_b + page_idx
        )
        k_c_ptrs = k_c_int_ptrs + kv_page_loc * stride_buf_k0

        if BLOCK_DPE > 0:
            k_pe_ptrs = offs_k_pe_init + kv_page_loc * stride_buf_k0
            k_pe = tl.load(k_pe_ptrs)
            qk = tl.dot(q_pe, k_pe.to(q_pe.dtype))
        
        q_nope_ptrs_loop = q_nope_ptrs
        k_c_ptrs_loop = k_c_ptrs
        for start_k in range(0, BLOCK_DMODEL, BLOCK_K):
            q_nope = tl.load(q_nope_ptrs_loop)
            k_c = tl.load(k_c_ptrs_loop)
            qk += tl.dot(q_nope, k_c.to(q_nope.dtype))
            q_nope_ptrs_loop += BLOCK_K
            k_c_ptrs_loop += BLOCK_K
        
        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        # Only mask for the last iteration
        if (page_idx == num_pages - 1) and (page_valid > 0):
            qk = tl.where(offs_page[None, :] < page_valid, qk, float("-inf"))

        v_c_ptrs = v_c_init_ptrs + kv_page_loc * stride_buf_v0
        v = tl.load(v_c_ptrs)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(v.dtype), v)

        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    offs_o = cur_kv_split * stride_o_ob + cur_batch * stride_o_oh + \
             cur_head[:, None] * stride_o_os + offs_dv[None, :]
    tl.store(Att_Out + offs_o, acc / e_sum[:, None])
    offs_lse = cur_kv_split * stride_l_0 + cur_batch * stride_l_1 + \
               cur_head
    tl.store(Lse + offs_lse, e_max + tl.log(e_sum))


@triton.jit
def _fwd_grouped_kernel_v3_2k_stage2(
    Mid_O,
    Lse,
    o,
    B_Seqlen,
    stride_mid_os,
    stride_mid_ob,
    stride_mid_oh,
    stride_l_0,
    stride_l_1,
    stride_obs,
    stride_oh,
    N_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    num_pages = tl.cdiv(cur_batch_seq_len, PAGE_SIZE)

    num_pages_per_split = num_pages // N_KV_SPLITS
    ramainer_pages = num_pages % N_KV_SPLITS
    if num_pages_per_split == 0:
        hi = ramainer_pages
    else:
        hi = N_KV_SPLITS

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_l_1 + cur_head

    for split_kv_id in range(0, hi):
        tv = tl.load(Mid_O + offs_v + split_kv_id * stride_mid_os,
                        mask=mask_d,
                        other=0.0)
        tlogic = tl.load(Lse + offs_logic + split_kv_id * stride_l_0)
        n_e_max = tl.maximum(tlogic, e_max)

        old_scale = tl.exp(e_max - n_e_max)
        acc *= old_scale
        exp_logic = tl.exp(tlogic - n_e_max)
        acc += exp_logic * tv

        e_sum = e_sum * old_scale + exp_logic
        e_max = n_e_max

    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def decode_attention_fwd_grouped_no_mask_gqa(
    q,
    k_buffer,
    v_buffer,
    o,
    req_to_token,
    b_seq_len,
    attn_logits,
    num_kv_splits,
    sm_scale,
    page_size,
    logit_cap=0.0,
):
    # BLOCK = 64
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]
    
    if Lk == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    batch, head_num = q.shape[0], q.shape[1]
    kv_group_num = q.shape[1] // k_buffer.shape[-2]

    BLOCK_H = 128
    BLOCK_K = 32
    N_KV_SPLITS = 16
    attn_logits = torch.empty(N_KV_SPLITS, batch, head_num, Lv, dtype=q.dtype, device=q.device)
    lse = torch.empty(N_KV_SPLITS, batch, head_num, dtype=q.dtype, device=q.device)
    grid = (
        batch,
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)),
        N_KV_SPLITS,
    )

    extra_kargs = {}
    
    _fwd_grouped_kernel_v3_2k_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        req_to_token,
        b_seq_len,
        attn_logits,
        lse,
        req_to_token.stride(0),
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        k_buffer.stride(1),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        k_buffer.stride(2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(0),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(1),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        v_buffer.stride(2),  # Assume (..., PAGE_SIZE, NUM_HEADS, HEAD_DIM)
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        lse.stride(0),
        lse.stride(1),
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_H=BLOCK_H,
        BLOCK_K=BLOCK_K,
        N_KV_SPLITS=N_KV_SPLITS,
        PAGE_SIZE=page_size,
        logit_cap=logit_cap,
        num_warps=8,
        num_stages=1,
        Lk=Lk,
        Lv=Lv,
        **extra_kargs,
    )
    
    grid2 = (batch, head_num, 1)
    _fwd_grouped_kernel_v3_2k_stage2[grid2](
        attn_logits,
        lse,
        o,
        b_seq_len,
        attn_logits.stride(0),
        attn_logits.stride(1),
        attn_logits.stride(2),
        lse.stride(0),
        lse.stride(1),
        o.stride(0),
        o.stride(1),
        N_KV_SPLITS=N_KV_SPLITS,
        PAGE_SIZE=page_size,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=1,
        **extra_kargs,
    )
    
