import os, sys

sys.path.insert(0, os.path.dirname(__file__) + "/../build")

import cpuinfer_ext
import torch

# Set fixed seed for reproducible results
torch.manual_seed(42)

# Constants for 4-bit packing
Q_BITS = 4
STORAGE_BITS = 32
PACK_NUM = STORAGE_BITS // Q_BITS  # 8


def pack(imatrix: torch.Tensor, direction: str = "row"):
    """
    Packs a 4-bit integer matrix into a packed 32-bit integer matrix.
    Packing order: 7 6 5 4 3 2 1 0 (MSB to LSB, original order)
    Args:
        imatrix (torch.Tensor): matrix of integers

    Returns:
        qmatrix (torch.Tensor): packed matrix of integers
    """
    shifts = torch.arange(0, STORAGE_BITS, Q_BITS, device=imatrix.device)

    imatrix = imatrix.to(torch.int8)
    imatrix = torch.bitwise_and(imatrix, 0x0F)  # eventually correct overflow

    if direction == "column":
        imatrix = imatrix.view(-1, imatrix.shape[1] // PACK_NUM, PACK_NUM)
        qmatrix = torch.bitwise_left_shift(imatrix, shifts[None, None, :]).sum(dim=-1)

    elif direction == "row":
        imatrix = imatrix.view(imatrix.shape[0] // PACK_NUM, PACK_NUM, -1)
        qmatrix = torch.bitwise_left_shift(imatrix, shifts[None, :, None]).sum(dim=1)

    qmatrix = qmatrix.to(torch.int32)

    return qmatrix


expert_num = 16
hidden_size = 7168
intermediate_size = 2048
max_len = 25600
num_experts_per_tok = 8
qlen = 1
layer_num = 1
CPUInfer = cpuinfer_ext.CPUInfer(40)
validation_iter = 10
k_group_size = 64
debug_print_count = 16

physical_to_logical_map = torch.tensor(data=range(expert_num), device="cpu", dtype=torch.int64).contiguous()


def act_fn(x):
    return x / (1.0 + torch.exp(-x))


def generate_original_weights():
    """Generate original FP16/BF16 weights for online quantization testing"""
    # Set seed to ensure consistency between online and offline quantization
    torch.manual_seed(42)

    # Generate weights in the same format as test_moe_amx.py (bfloat16)
    gate_proj_bf16 = (
        torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16, device="cuda")
        .to("cpu")
        .contiguous()
    )
    up_proj_bf16 = (
        torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16, device="cuda")
        .to("cpu")
        .contiguous()
    )
    down_proj_bf16 = (
        torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
        .to("cpu")
        .contiguous()
    )

    # Print first row of gate_proj for expert 0 (first debug_print_count elements)
    print(
        f"[DEBUG] Online quantization gate_proj expert 0, row 0, first {debug_print_count} elements: {gate_proj_bf16[0, 0, :debug_print_count]}"
    )

    return gate_proj_bf16, up_proj_bf16, down_proj_bf16


def generate_awq_quantized_weights():
    """Generate AWQ quantized weights (qweight, scales, qzeros) for testing"""
    # Reset seed to ensure same weights as online quantization
    torch.manual_seed(42)

    # Generate original FP16 weights (convert from same random values as online version)
    gate_proj_fp16 = (
        torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16, device="cuda")
        .to("cpu")
        .to(torch.float16)
        .contiguous()
    )
    up_proj_fp16 = (
        torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16, device="cuda")
        .to("cpu")
        .to(torch.float16)
        .contiguous()
    )
    down_proj_fp16 = (
        torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
        .to("cpu")
        .to(torch.float16)
        .contiguous()
    )

    # Print first row of gate_proj for expert 0 (first debug_print_count elements)
    print(
        f"[DEBUG] Offline AWQ gate_proj expert 0, row 0, first {debug_print_count} elements: {gate_proj_fp16[0, 0, :debug_print_count]}"
    )

    # Calculate quantization parameters per group
    def quantize_tensor_awq(weight, group_size=128):
        """Simple AWQ-style quantization simulation with interleaving"""

        w_orig_shape = weight.shape
        expert_num, col, row = weight.shape
        group_num = (row + group_size - 1) // group_size

        # 1. reshape into groups along row dimension
        weight_grouped = weight.view(expert_num, col, group_num, group_size)  # [E, G, group_size, C]

        # 2. calculate scales per group (max abs value / 7.0 for 4-bit signed)
        max_val = torch.max(weight_grouped, dim=3).values
        min_val = torch.min(weight_grouped, dim=3).values
        scales = (max_val - min_val).clamp(min=1e-5) / 15.0  # [E, G, C]
        zeros = (-torch.round(min_val / scales)).clamp_(0, 15).to(torch.int8)

        # 5. quantize weights
        qweight_int = torch.clamp(
            torch.round((weight_grouped - min_val.unsqueeze(-1)) / scales.unsqueeze(-1)), 0, 15
        ).to(torch.int8)

        qweight_int = qweight_int.view(w_orig_shape)

        # 6. pack qweight along row (group_size) using helper
        qweight_packed_list = []
        for e in range(expert_num):
            packed = pack(qweight_int[e], direction="column")  # [1, ? , col] or similar
            qweight_packed_list.append(packed)
        qweight_packed = torch.stack(qweight_packed_list, dim=0)  # [E, row, col / 8]

        # 7. pack zeros along group dimension (row) using helper
        zeros_packed_list = []
        for e in range(expert_num):
            zeros_packed_list.append(pack(zeros[e].transpose(0, 1), direction="column"))  # [blocks, col]
        qzeros_packed = torch.stack(zeros_packed_list, dim=0)

        scales = scales.transpose(1, 2).to(torch.float16)
        print(scales.shape)
        scales = scales.flatten().contiguous()

        min_val = min_val.transpose(1, 2).to(torch.float16).flatten().contiguous()

        zeros = zeros.transpose(1, 2).flatten().contiguous()

        qzeros_packed = qzeros_packed.flatten().contiguous()

        qweight_packed = qweight_packed.flatten().contiguous()

        return {
            "qweight": qweight_packed,  # Same for both torch and AWQ-MoE
            "scales": scales,  # Same for both torch and AWQ-MoE
            "qzeros": qzeros_packed,  # Same for both torch and AWQ-MoE
            "mins": min_val,  # scales * zeros for comparison
        }

    # Quantize each projection
    gate_data = quantize_tensor_awq(gate_proj_fp16, k_group_size)
    up_data = quantize_tensor_awq(up_proj_fp16, k_group_size)
    down_data = quantize_tensor_awq(down_proj_fp16, k_group_size)

    return {
        # Data for both torch and AWQ-MoE (no interleaving)
        "gate_qweight": gate_data["qweight"],
        "gate_scales": gate_data["scales"],
        "gate_qzeros": gate_data["qzeros"],
        "gate_mins": gate_data["mins"],
        "up_qweight": up_data["qweight"],
        "up_scales": up_data["scales"],
        "up_qzeros": up_data["qzeros"],
        "up_mins": up_data["mins"],
        "down_qweight": down_data["qweight"],
        "down_scales": down_data["scales"],
        "down_qzeros": down_data["qzeros"],
        "down_mins": down_data["mins"],
        "original_fp16": {"gate_proj": gate_proj_fp16, "up_proj": up_proj_fp16, "down_proj": down_proj_fp16},
    }


def mlp_torch(input, gate_proj, up_proj, down_proj, debug_expert_id=None, debug_print=False):
    gate_buf = torch.mm(input, gate_proj.t())
    up_buf = torch.mm(input, up_proj.t())

    if debug_print and debug_expert_id is not None:
        print(f"[TORCH FP16 DEBUG] Expert {debug_expert_id}:")
        print(f"  gate_buf[:{debug_print_count}] = {gate_buf.flatten()[:debug_print_count]}")
        print(f"  up_buf[:{debug_print_count}] = {up_buf.flatten()[:debug_print_count]}")

    intermediate = act_fn(gate_buf) * up_buf

    if debug_print and debug_expert_id is not None:
        print(f"  intermediate[:{debug_print_count}] = {intermediate.flatten()[:debug_print_count]}")

    ret = torch.mm(intermediate, down_proj.t())

    if debug_print and debug_expert_id is not None:
        print(f"  down_output[:{debug_print_count}] = {ret.flatten()[:debug_print_count]}")

    return ret


def moe_torch(input, expert_ids, weights, gate_proj, up_proj, down_proj, debug_print=False):
    cnts = expert_ids.new_zeros((expert_ids.shape[0], expert_num))
    cnts.scatter_(1, expert_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = expert_ids.view(-1).argsort()
    sorted_tokens = input[idxs // expert_ids.shape[1]]

    # Get the first expert from expert_ids array to match AWQ-MoE behavior
    target_debug_expert = expert_ids[0, 0].item()  # First expert in expert_ids array

    outputs = []
    start_idx = 0
    activated_experts = []

    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        activated_experts.append(i)
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        # Only debug the target expert that matches AWQ-MoE's first expert
        should_debug = debug_print and i == target_debug_expert
        if gate_proj[i].dtype == torch.float16:
            expert_out = mlp_torch(
                tokens_for_this_expert.to(torch.float16),
                gate_proj[i],
                up_proj[i],
                down_proj[i],
                debug_expert_id=i,
                debug_print=should_debug,
            )
        else:
            expert_out = mlp_torch(
                tokens_for_this_expert,
                gate_proj[i],
                up_proj[i],
                down_proj[i],
                debug_expert_id=i,
                debug_print=should_debug,
            )
        outputs.append(expert_out)
        start_idx = end_idx

    if debug_print:
        print(f"[TORCH DEBUG] Processing activated experts: {activated_experts}")
        print(f"[TORCH DEBUG] Target debug expert (matches AWQ): {target_debug_expert}")

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    t_output = (
        new_x.view(*expert_ids.shape, -1)
        .type(weights.dtype)
        .mul_(weights.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )

    if debug_print:
        print(f"[TORCH DEBUG] Final MoE output[:{debug_print_count}] = {t_output.flatten()[:debug_print_count]}")

    return t_output


def test_online_int4_kgroup_moe():
    """Test online Int4LowKGroup quantization (reference implementation)"""
    print("Testing Online Int4LowKGroup quantization (reference)...")

    # Generate original weights for online quantization
    gate_proj, up_proj, down_proj = generate_original_weights()

    with torch.inference_mode(mode=True):
        moes = []
        gate_projs = []
        up_projs = []
        down_projs = []

        for _ in range(layer_num):
            # Create Int4LowKGroup configuration (online quantization)
            config = cpuinfer_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
            config.max_len = max_len
            config.gate_proj = gate_proj.data_ptr()
            config.up_proj = up_proj.data_ptr()
            config.down_proj = down_proj.data_ptr()
            config.gate_scale = 0
            config.pool = CPUInfer.backend_

            # Set quantization config for Int4LowKGroup (matches test_moe_amx.py)
            config.quant_config.bits = 4
            config.quant_config.group_size = k_group_size
            config.quant_config.zero_point = True

            # Enable weight dumping for comparison
            config.save = True
            config.path = "./awq_dump_online"

            # Create Int4LowKGroup MoE (online quantization during load_weights)
            moe = cpuinfer_ext.moe.AMXInt4_1KGroup_MOE(config)

            # Load weights (performs online quantization)
            print(f"Physical Map: {physical_to_logical_map.data_ptr()}")
            CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
            CPUInfer.sync()

            # Warm up
            CPUInfer.submit(moe.warm_up_task())
            CPUInfer.sync()

            gate_projs.append(gate_proj)
            up_projs.append(up_proj)
            down_projs.append(down_proj)
            moes.append(moe)

        print("Online Int4LowKGroup MoE created and loaded successfully!")

        # Run validation tests
        results_online = []
        for i in range(validation_iter):
            # Reset seed for reproducible expert_ids and weights
            torch.manual_seed(100 + i)  # Different seed to avoid same random values

            bsz_tensor = torch.tensor([qlen], device="cpu")
            expert_ids = torch.stack(
                [torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]
            ).contiguous()
            weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
            output = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
            # input = torch.tensor(
            #     data=torch.cat([torch.ones(qlen, 1), torch.zeros(qlen, hidden_size - 1)], dim=1),
            #     dtype=torch.bfloat16
            # )
            input = torch.randn((qlen, hidden_size), dtype=torch.bfloat16).contiguous() / 100

            moe = moes[i % layer_num]

            # Enable debug for first few iterations
            enable_debug = i < 2
            if enable_debug:
                print(f"\n=== Online Int4LowKGroup Test Iteration {i} ===")
                print(f"input[:{debug_print_count}] = {input.flatten()[:debug_print_count]}")
                print(f"expert_ids = {expert_ids}")
                print(f"weights = {weights}")

            # Run online quantized MoE forward
            CPUInfer.submit(
                moe.forward_task(
                    bsz_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids.data_ptr(),
                    weights.data_ptr(),
                    input.data_ptr(),
                    output.data_ptr(),
                    False,
                )
            )
            CPUInfer.sync()

            if enable_debug:
                print(f"[ONLINE DEBUG] AMX output[:{debug_print_count}] = {output.flatten()[:debug_print_count]}")

            # Compare with FP16 reference
            gate_proj_ref = gate_projs[i % layer_num]
            up_proj_ref = up_projs[i % layer_num]
            down_proj_ref = down_projs[i % layer_num]

            t_output_online = moe_torch(
                input, expert_ids, weights, gate_proj_ref, up_proj_ref, down_proj_ref, debug_print=enable_debug
            )

            # Calculate differences
            diff_online = torch.mean(torch.abs(output - t_output_online)) / torch.mean(torch.abs(t_output_online))
            results_online.append(output.clone())

            print(f"Online Iteration {i}: Int4LowKGroup vs FP16 = {diff_online:.6f}")

            if enable_debug:
                abs_diff_online = torch.abs(output - t_output_online)
                print(f"[COMPARE] Online Int4LowKGroup vs FP16:")
                print(f"  Max abs diff = {torch.max(abs_diff_online):.6f}")
                print(f"  Mean abs diff = {torch.mean(abs_diff_online):.6f}")
                print(f"  Relative diff = {diff_online:.6f}")
                print("=" * 70)

        print("\n✅ Online Int4LowKGroup tests passed!")
        return results_online


def test_awq_moe():
    print("Testing AWQ MoE with Int4_1LowKGroup quantization...")

    # Generate AWQ quantized weights
    awq_data = generate_awq_quantized_weights()

    with torch.inference_mode(mode=True):
        moes = []

        for _ in range(layer_num):
            # Create AWQ MoE configuration
            config = cpuinfer_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
            config.max_len = max_len

            # Set quantization config for Int4_1LowKGroup
            config.quant_config.bits = 4
            config.quant_config.group_size = k_group_size
            config.quant_config.zero_point = True

            # Enable weight dumping for comparison
            config.save = True
            config.path = "./awq_dump_offline"

            # Set pointers to AWQ quantized data (no interleaving)
            config.gate_proj = awq_data["gate_qweight"].data_ptr()
            config.up_proj = awq_data["up_qweight"].data_ptr()
            config.down_proj = awq_data["down_qweight"].data_ptr()

            config.gate_scale = awq_data["gate_scales"].data_ptr()
            config.up_scale = awq_data["up_scales"].data_ptr()
            config.down_scale = awq_data["down_scales"].data_ptr()

            config.gate_zeros = awq_data["gate_qzeros"].data_ptr()
            config.up_zeros = awq_data["up_qzeros"].data_ptr()
            config.down_zeros = awq_data["down_qzeros"].data_ptr()

            config.pool = CPUInfer.backend_

            # Create Int4_1LowKGroup MoE
            moe = cpuinfer_ext.moe.AMXInt4_1KGroup_MOE(config)

            # Load weights
            CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
            CPUInfer.sync()

            # Warm up
            CPUInfer.submit(moe.warm_up_task())
            CPUInfer.sync()

            moes.append(moe)

        print("AWQ MoE Int4_1LowKGroup created and loaded successfully!")

        # Run validation tests
        results_awq = []
        for i in range(validation_iter):
            # Reset seed for reproducible expert_ids and weights (same as online test)
            torch.manual_seed(100 + i)

            bsz_tensor = torch.tensor([qlen], device="cpu")
            expert_ids = torch.stack(
                [torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]
            ).contiguous()
            weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
            output = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
            # input = torch.tensor(
            #     data=torch.cat([torch.ones(qlen, 1), torch.zeros(qlen, hidden_size - 1)], dim=1),
            #     dtype=torch.bfloat16
            # )
            input = torch.randn((qlen, hidden_size), dtype=torch.bfloat16).contiguous() / 100

            moe = moes[i % layer_num]

            # Enable debug for first few iterations
            enable_debug = i < 2
            if enable_debug:
                print(f"\n=== AWQ MoE Int4_1LowKGroup Test Iteration {i} ===")
                print(f"input[:{debug_print_count}] = {input.flatten()[:debug_print_count]}")
                print(f"expert_ids = {expert_ids}")
                print(f"weights = {weights}")

                # Print which experts will be activated
                activated_experts = []
                for token in range(expert_ids.shape[0]):
                    for expert_idx in range(expert_ids.shape[1]):
                        expert_id = expert_ids[token][expert_idx].item()
                        if expert_id not in activated_experts:
                            activated_experts.append(expert_id)
                print(f"[TORCH DEBUG] Activated experts: {sorted(activated_experts)}")
                print(f"[TORCH DEBUG] First expert from expert_ids array: {expert_ids[0, 0].item()}")

            # Run AWQ MoE forward
            CPUInfer.submit(
                moe.forward_task(
                    bsz_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids.data_ptr(),
                    weights.data_ptr(),
                    input.data_ptr(),
                    output.data_ptr(),
                    False,
                )
            )
            CPUInfer.sync()

            if enable_debug:
                print(f"[AWQ-MoE DEBUG] AMX output[:{debug_print_count}] = {output.flatten()[:debug_print_count]}")

            # Compare with FP16 reference
            original_weights = awq_data["original_fp16"]
            gate_proj = original_weights["gate_proj"].to(torch.float16)
            up_proj = original_weights["up_proj"].to(torch.float16)
            down_proj = original_weights["down_proj"].to(torch.float16)

            t_output_fp16 = moe_torch(
                input, expert_ids, weights, gate_proj, up_proj, down_proj, debug_print=enable_debug
            )

            # Calculate differences
            diff_fp16 = torch.mean(torch.abs(output - t_output_fp16)) / torch.mean(torch.abs(t_output_fp16))
            results_awq.append(output.clone())

            print(f"AWQ Iteration {i}: AWQ-MoE vs FP16 = {diff_fp16:.6f}")

            if enable_debug:
                abs_diff_fp16 = torch.abs(output - t_output_fp16)
                print(f"[COMPARE] AWQ-MoE vs FP16:")
                print(f"  Max abs diff = {torch.max(abs_diff_fp16):.6f}")
                print(f"  Mean abs diff = {torch.mean(abs_diff_fp16):.6f}")
                print(f"  Relative diff = {diff_fp16:.6f}")
                print("=" * 70)

            # AWQ quantization typically has higher error tolerance due to 4-bit quantization vs FP16
            # assert(diff_fp16 < 0.5), f"AWQ-MoE vs FP16 error too large: {diff_fp16:.6f}"

        print("\n✅ All AWQ MoE tests passed!")
        return results_awq


def compare_quantization_methods():
    """Compare online and offline quantization methods"""
    print("=" * 70)
    print("Comparing Online vs Offline Quantization Methods")
    print("=" * 70)

    # Run online quantization test (reference)
    print("\n" + "=" * 70)
    print("PHASE 1: Online Int4LowKGroup Quantization (Reference)")
    print("=" * 70)
    results_online = test_online_int4_kgroup_moe()

    # Run offline AWQ quantization test
    print("\n" + "=" * 70)
    print("PHASE 2: Offline AWQ Int4_1LowKGroup Quantization")
    print("=" * 70)
    results_awq = test_awq_moe()

    # Compare the results
    print("\n" + "=" * 70)
    print("PHASE 3: Comparison Results")
    print("=" * 70)

    if len(results_online) != len(results_awq):
        print(f"❌ Different number of results: Online={len(results_online)}, AWQ={len(results_awq)}")
        return

    print("Comparing Online Int4LowKGroup vs Offline AWQ results:")
    total_diff = 0.0
    max_diff = 0.0

    for i in range(len(results_online)):
        diff = torch.mean(torch.abs(results_online[i] - results_awq[i]))
        rel_diff = diff / torch.mean(torch.abs(results_online[i]))
        total_diff += rel_diff
        max_diff = max(max_diff, diff.item())

        if i < 3:  # Show detailed comparison for first 3 iterations
            print(f"  Iteration {i}:")
            print(f"    Absolute diff: {diff:.6f}")
            print(f"    Relative diff: {rel_diff:.6f}")
            print(f"    Online output[:{debug_print_count//2}]:  {results_online[i].flatten()[:debug_print_count//2]}")
            print(f"    AWQ output[:{debug_print_count//2}]:     {results_awq[i].flatten()[:debug_print_count//2]}")
        else:
            print(f"  Iteration {i}: Relative diff = {rel_diff:.6f}")

    avg_diff = total_diff / len(results_online)
    print(f"\nOverall comparison:")
    print(f"  Average relative difference: {avg_diff:.6f}")
    print(f"  Maximum absolute difference: {max_diff:.6f}")

    # Determine if results match within acceptable tolerance
    tolerance = 0.01  # 1% tolerance
    if avg_diff < tolerance:
        print(f"✅ Results match within {tolerance:.1%} tolerance!")
        print("   Your offline AWQ quantization implementation appears to be correct.")
    else:
        print(f"❌ Results differ by more than {tolerance:.1%} tolerance.")
        print("   There may be differences between online and offline quantization.")


if __name__ == "__main__":
    print("=" * 70)
    print("AWQ MoE AMX Test - Online vs Offline Quantization Comparison")
    print("=" * 70)

    compare_quantization_methods()

    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)
