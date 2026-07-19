#!/usr/bin/env python
# coding=utf-8
"""Parity test for MOEConfig.skip_gpu_expert_cpu_copy.

With skip_gpu_expert_cpu_copy set, the CPU weight buffers of GPU-resident experts
(those in gpu_experts_mask) are never allocated or loaded. This test runs a hybrid
split with the flag on and off and asserts the forward output is bitwise identical,
plus matches a torch reference that drops the masked experts.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=90, suite="default")

try:
    import torch
    import kt_kernel

    kt_kernel_ext = kt_kernel.kt_kernel_ext
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    import_error = str(e)

expert_num = 64
hidden_size = 2048
intermediate_size = 1024
max_len = 512
num_experts_per_tok = 8


def act_fn(x):
    return x / (1.0 + torch.exp(-x))


def mlp_torch(input, gate_proj, up_proj, down_proj):
    gate_buf = torch.mm(input, gate_proj.t())
    up_buf = torch.mm(input, up_proj.t())
    intermediate = act_fn(gate_buf) * up_buf
    return torch.mm(intermediate, down_proj.t())


def moe_torch(input, expert_ids, weights, gate_proj, up_proj, down_proj, skip_mask=None):
    """Reference MoE; experts flagged in skip_mask contribute zero (GPU-resident)."""
    cnts = expert_ids.new_zeros((expert_ids.shape[0], expert_num))
    cnts.scatter_(1, expert_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = expert_ids.view(-1).argsort()
    sorted_tokens = input[idxs // expert_ids.shape[1]]

    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        if skip_mask is not None and bool(skip_mask[i]):
            expert_out = torch.zeros_like(tokens_for_this_expert)
        else:
            expert_out = mlp_torch(tokens_for_this_expert, gate_proj[i], up_proj[i], down_proj[i])
        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    return (
        new_x.view(*expert_ids.shape, -1)
        .type(weights.dtype)
        .mul_(weights.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )


def _build_moe(CPUInfer, gate_proj, up_proj, down_proj, gpu_experts_mask, physical_to_logical_map, skip):
    config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size)
    config.max_len = max_len
    config.gate_proj = gate_proj.data_ptr()
    config.up_proj = up_proj.data_ptr()
    config.down_proj = down_proj.data_ptr()
    config.gate_scale = 0
    config.pool = CPUInfer.backend_
    config.gpu_experts_mask = gpu_experts_mask.data_ptr()
    config.skip_gpu_expert_cpu_copy = skip

    moe = kt_kernel_ext.moe.AMXInt4_MOE(config)
    CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
    CPUInfer.sync()
    CPUInfer.submit(moe.warm_up_task())
    CPUInfer.sync()
    return moe


def _forward(CPUInfer, moe, expert_ids, weights, input_data):
    qlen = input_data.shape[0]
    bsz_tensor = torch.tensor([qlen], device="cpu")
    output = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
    CPUInfer.submit(
        moe.forward_task(
            bsz_tensor.data_ptr(),
            num_experts_per_tok,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            input_data.data_ptr(),
            output.data_ptr(),
            False,
        )
    )
    CPUInfer.sync()
    return output


@pytest.mark.cpu
def test_moe_amx_skip_gpu_expert_cpu_copy_parity():
    if not HAS_DEPS:
        pytest.skip(f"Dependencies not available: {import_error}")

    torch.manual_seed(1234)

    physical_to_logical_map = torch.tensor(data=range(expert_num), device="cpu", dtype=torch.int64).contiguous()

    gpu_experts_mask = torch.zeros(expert_num, dtype=torch.uint8).contiguous()
    gpu_experts_mask[::2] = 1
    skip_mask_bool = gpu_experts_mask.bool()

    CPUInfer = kt_kernel_ext.CPUInfer(60)

    with torch.inference_mode(mode=True):
        gate_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16).contiguous()
        up_proj = torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16).contiguous()
        down_proj = torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.bfloat16).contiguous()

        moe_off = _build_moe(
            CPUInfer, gate_proj, up_proj, down_proj, gpu_experts_mask, physical_to_logical_map, skip=False
        )
        moe_on = _build_moe(
            CPUInfer, gate_proj, up_proj, down_proj, gpu_experts_mask, physical_to_logical_map, skip=True
        )

        for qlen in (1, 7):
            expert_ids = torch.stack(
                [torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]
            ).contiguous()
            weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
            input_data = (torch.randn((qlen, hidden_size), dtype=torch.bfloat16) / 100).contiguous()

            routed_masked = int(skip_mask_bool[expert_ids.view(-1)].sum())
            assert routed_masked > 0, "test setup routes no tokens to GPU-masked experts; mask never fires"

            out_off = _forward(CPUInfer, moe_off, expert_ids, weights, input_data)
            out_on = _forward(CPUInfer, moe_on, expert_ids, weights, input_data)

            assert torch.equal(out_off, out_on), (
                f"skip_gpu_expert_cpu_copy parity failed at qlen={qlen}: "
                f"max abs diff = {(out_off.float() - out_on.float()).abs().max():.6f}"
            )

            t_output = moe_torch(
                input_data, expert_ids, weights, gate_proj, up_proj, down_proj, skip_mask=skip_mask_bool
            )
            denom = torch.mean(torch.abs(t_output))
            diff = torch.mean(torch.abs(out_on.float() - t_output.float())) / denom if denom > 0 else torch.tensor(0.0)
            print(f"qlen={qlen}, routed_masked={routed_masked}, parity=exact, diff_vs_ref={float(diff):.6f}")
            assert diff < 0.35, f"skip path accuracy failed at qlen={qlen}: diff={float(diff):.6f} >= 0.35"


def test_skip_gpu_expert_cpu_copy_rejected_in_sft_mode():
    """SFT trains every expert on CPU and needs full host copies; the factory
    must refuse the inference-only skip before any backend is constructed."""
    if not HAS_DEPS:
        pytest.skip(f"Dependencies not available: {import_error}")
    from kt_kernel.experts import KTMoEWrapper

    with pytest.raises(ValueError, match="skip_gpu_expert_cpu_copy"):
        KTMoEWrapper(
            layer_idx=0,
            num_experts=expert_num,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=intermediate_size,
            gpu_experts_mask=None,
            cpuinfer_threads=2,
            threadpool_count=1,
            weight_path="/nonexistent",
            chunked_prefill_size=max_len,
            skip_gpu_expert_cpu_copy=True,
            method="AMXBF16_SFT",
            mode="sft",
        )


def run_all_tests():
    if not HAS_DEPS:
        print(f"Dependencies not available: {import_error}")
        return
    try:
        test_moe_amx_skip_gpu_expert_cpu_copy_parity()
        print("skip_gpu_expert_cpu_copy parity test passed")
        test_skip_gpu_expert_cpu_copy_rejected_in_sft_mode()
        print("skip_gpu_expert_cpu_copy sft-mode rejection test passed")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_all_tests()
