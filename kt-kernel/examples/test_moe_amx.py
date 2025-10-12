import os, sys

sys.path.insert(0, os.path.dirname(__file__) + "/../build")

import cpuinfer_ext
import torch

expert_num = 256
hidden_size = 7168
intermediate_size = 2048
max_len = 25600
num_experts_per_tok = 8
# qlen = 1
qlen = 640
layer_num = 1
CPUInfer = cpuinfer_ext.CPUInfer(40)
# validation_iter = 10000
validation_iter = 10
k_group_size = 64
debug_print_count = 16  # Number of values to print in debug output
physical_to_logical_map = torch.tensor(data=range(expert_num), device="cpu", dtype=torch.int64).contiguous()


def act_fn(x):
    return x / (1.0 + torch.exp(-x))


def mlp_torch(input, gate_proj, up_proj, down_proj, debug_expert_id=None, debug_print=False):
    gate_buf = torch.mm(input, gate_proj.t())
    up_buf = torch.mm(input, up_proj.t())

    if debug_print and debug_expert_id is not None:
        print(f"[TORCH DEBUG] Expert {debug_expert_id}:")
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
        expert_out = mlp_torch(
            tokens_for_this_expert, gate_proj[i], up_proj[i], down_proj[i], debug_expert_id=i, debug_print=should_debug
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


def test_moe(quant_mode: str):
    assert (
        quant_mode == "bf16"
        or quant_mode == "int8"
        or quant_mode == "int4"
        or quant_mode == "int4_1"
        or quant_mode == "int4_1k"
    )
    with torch.inference_mode(mode=True):
        moes = []
        gate_projs = []
        up_projs = []
        down_projs = []
        for _ in range(layer_num):
            gate_proj = (
                torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16, device="cuda")
                .to("cpu")
                .contiguous()
            )
            up_proj = (
                torch.randn((expert_num, intermediate_size, hidden_size), dtype=torch.bfloat16, device="cuda")
                .to("cpu")
                .contiguous()
            )
            down_proj = (
                torch.randn((expert_num, hidden_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
                .to("cpu")
                .contiguous()
            )
            config = cpuinfer_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden_size, intermediate_size, 0)
            config.max_len = max_len
            config.gate_proj = gate_proj.data_ptr()
            config.up_proj = up_proj.data_ptr()
            config.down_proj = down_proj.data_ptr()
            config.gate_scale = 0
            config.pool = CPUInfer.backend_
            if quant_mode == "bf16":
                moe = cpuinfer_ext.moe.AMXBF16_MOE(config)
                CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
                CPUInfer.sync()
                CPUInfer.submit(moe.warm_up_task())
                CPUInfer.sync()
            elif quant_mode == "int8":
                moe = cpuinfer_ext.moe.AMXInt8_MOE(config)
                CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
                CPUInfer.sync()
                # CPUInfer.submit(moe.warm_up_task())
                # CPUInfer.sync()
            elif quant_mode == "int4":
                moe = cpuinfer_ext.moe.AMXInt4_MOE(config)
                CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
                CPUInfer.sync()
                CPUInfer.submit(moe.warm_up_task())
                CPUInfer.sync()
            elif quant_mode == "int4_1":
                moe = cpuinfer_ext.moe.AMXInt4_1_MOE(config)
                CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
                CPUInfer.sync()
                CPUInfer.submit(moe.warm_up_task())
                CPUInfer.sync()
            elif quant_mode == "int4_1k":
                config.quant_config.bits = 4
                config.quant_config.group_size = k_group_size
                config.quant_config.zero_point = True
                moe = cpuinfer_ext.moe.AMXInt4_1KGroup_MOE(config)
                # import debugpy
                # debugpy.listen(("127.0.0.1", 5678))
                # debugpy.wait_for_client()
                # debugpy.breakpoint()
                CPUInfer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
                CPUInfer.sync()
                # CPUInfer.submit(moe.warm_up_task())
                # CPUInfer.sync()
            gate_projs.append(gate_proj)
            up_projs.append(up_proj)
            down_projs.append(down_proj)
            moes.append(moe)

        # validation
        for i in range(validation_iter):
            bsz_tensor = torch.tensor([qlen], device="cpu")
            expert_ids = torch.stack(
                [torch.randperm(expert_num)[:num_experts_per_tok] for _ in range(qlen)]
            ).contiguous()
            weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32).contiguous()
            input = torch.randn((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
            output = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
            input = input / 100
            moe = moes[i % layer_num]

            # Enable debug for first few iterations
            enable_debug = i < 2
            enable_debug = False
            if enable_debug:
                print(f"\n=== Iteration {i} Debug Info ===")
                print(f"input[:{debug_print_count}] = {input.flatten()[:debug_print_count]}")
                print(f"expert_ids = {expert_ids}")
                print(f"weights = {weights}")
                # Print which experts will be activated for comparison
                activated_experts = []
                for token in range(expert_ids.shape[0]):
                    for expert_idx in range(expert_ids.shape[1]):
                        expert_id = expert_ids[token][expert_idx].item()
                        if expert_id not in activated_experts:
                            activated_experts.append(expert_id)
                print(f"[TORCH DEBUG] Activated experts: {sorted(activated_experts)}")
                print(f"[TORCH DEBUG] First expert from expert_ids array: {expert_ids[0, 0].item()}")
            print(f"expert_ids = {expert_ids}")
            # print('expert ids:',expert_ids)
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
                print(f"[AWQ-MOE DEBUG] AMX output[:{debug_print_count}] = {output.flatten()[:debug_print_count]}")

            gate_proj = gate_projs[i % layer_num]
            up_proj = up_projs[i % layer_num]
            down_proj = down_projs[i % layer_num]
            t_output = moe_torch(input, expert_ids, weights, gate_proj, up_proj, down_proj, debug_print=enable_debug)
            print("torch output", t_output)
            print("amx output", output)

            # print(output - t_output)
            # print(torch.abs(output - t_output))
            diff = torch.mean(torch.abs(output - t_output)) / torch.mean(torch.abs(t_output))
            # print(f'output_shape:{output.shape}, t_output_shape:{t_output.shape}\n')
            print(f"Iteration {i}, diff = {diff:.6f}")

            if enable_debug:
                abs_diff = torch.abs(output - t_output)
                print(f"[COMPARE] Max abs diff = {torch.max(abs_diff):.6f}")
                print(f"[COMPARE] Mean abs diff = {torch.mean(abs_diff):.6f}")
                print(f"[COMPARE] Relative diff = {diff:.6f}")
                print("=" * 50)

            if quant_mode == "int4" or quant_mode == "int4_1" or quant_mode == "int4_1k":
                assert diff < 0.35
            else:
                assert diff < 0.05


# only turn on 1 at a time

# Debug mode is enabled for the first 2 iterations to compare intermediate results
# between torch implementation and AWQ-MoE implementation.
# The debug output shows:
# 1. Input values and expert assignments
# 2. Gate and up projection results
# 3. Intermediate values after activation function
# 4. Down projection results
# 5. Final output comparison

# test_moe("bf16")
# test_moe("int8")
# test_moe("int4")
# test_moe("int4_1")
test_moe("int4_1k")
