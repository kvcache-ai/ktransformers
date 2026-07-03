import argparse
import os
import sys
import tempfile

import torch
import torch.nn as nn

from kt_kernel.sft.arch import _get_layers_prefix, get_moe_arch_config
from kt_kernel.sft.config import KTConfig
from kt_kernel.sft.weights import (
    has_kgroup_experts_in_kt_weight_path,
    load_kgroup_experts_from_kt_weight_path,
)

TRAINER_SMOKE_LOSS_SCALE = 4096.0


def dequant_kgroup_weight(
    packed: torch.Tensor,
    scales: torch.Tensor,
    rows: int,
    cols: int,
    group_size: int,
) -> torch.Tensor:
    packed_flat = packed.reshape(-1).to(torch.int16)
    q = torch.empty((rows * cols,), dtype=torch.int16)
    q[0::2] = packed_flat & 0x0F
    q[1::2] = (packed_flat >> 4) & 0x0F
    q = q.reshape(rows, cols).float() - 8.0
    scale = scales.reshape(rows, cols // group_size).float().repeat_interleave(group_size, dim=1)
    return q * scale


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> None:
    diff = (actual.float() - expected.float()).abs()
    print(f"{name}: max_abs={diff.max().item():.6g}, mean_abs={diff.mean().item():.6g}")
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol, equal_nan=False)


def zero_storage_parameter(shape: tuple[int, ...], dtype: torch.dtype = torch.bfloat16) -> nn.Parameter:
    storage = torch.UntypedStorage(1, device="cpu")
    tensor = torch.tensor([], dtype=dtype, device="cpu").set_(
        storage,
        storage_offset=0,
        size=shape,
        stride=[0] * len(shape),
    )
    return nn.Parameter(tensor, requires_grad=False)


def _bf16_randn(shape: tuple[int, ...], seed: int, scale: float = 0.01) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return (torch.randn(shape, generator=generator, dtype=torch.float32) * scale).to(torch.bfloat16).contiguous()


class LoraWeightModule(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=True)


class DummyRouter(nn.Module):
    def __init__(self, num_experts_per_tok: int, num_experts: int):
        super().__init__()
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts

    def forward(self, hidden_states: torch.Tensor):
        qlen = hidden_states.numel() // hidden_states.shape[-1]
        expert_ids = torch.stack(
            [torch.arange(i, i + self.num_experts_per_tok) % self.num_experts for i in range(qlen)],
            dim=0,
        ).to(torch.int64)
        route_weights = torch.full(
            (qlen, self.num_experts_per_tok),
            1.0 / self.num_experts_per_tok,
            dtype=torch.bfloat16,
        )
        return expert_ids, route_weights


class DummyProj(nn.Module):
    def __init__(self, shape: tuple[int, int], lora_a: torch.Tensor, lora_b: torch.Tensor):
        super().__init__()
        self.weight = zero_storage_parameter(shape)
        self.lora_A = nn.ModuleDict({"default": LoraWeightModule(lora_a)})
        self.lora_B = nn.ModuleDict({"default": LoraWeightModule(lora_b)})
        self.active_adapter = ["default"]


class DummyExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, lora_rank: int, expert_idx: int):
        super().__init__()
        base_seed = 1000 + expert_idx * 13
        self.gate_proj = DummyProj(
            (intermediate_size, hidden_size),
            _bf16_randn((lora_rank, hidden_size), base_seed + 1),
            torch.zeros((intermediate_size, lora_rank), dtype=torch.bfloat16),
        )
        self.up_proj = DummyProj(
            (intermediate_size, hidden_size),
            _bf16_randn((lora_rank, hidden_size), base_seed + 2),
            torch.zeros((intermediate_size, lora_rank), dtype=torch.bfloat16),
        )
        self.down_proj = DummyProj(
            (hidden_size, intermediate_size),
            _bf16_randn((lora_rank, intermediate_size), base_seed + 3),
            torch.zeros((hidden_size, lora_rank), dtype=torch.bfloat16),
        )


class DummyMoE(nn.Module):
    def __init__(self, moe_config, hidden_size: int, lora_rank: int):
        super().__init__()
        self.gate = DummyRouter(moe_config.num_experts_per_tok, moe_config.expert_num)
        self.experts = nn.ModuleList(
            DummyExpert(hidden_size, moe_config.intermediate_size, lora_rank, expert_idx)
            for expert_idx in range(moe_config.expert_num)
        )


class DummyBlock(nn.Module):
    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.mlp(hidden_states)


class DummyKimiModel(nn.Module):
    def __init__(self, config, moe_config, target_layer_idx: int, hidden_size: int, lora_rank: int = 1):
        super().__init__()
        self.config = config
        layers = []
        for layer_idx in range(target_layer_idx + 1):
            mlp = DummyMoE(moe_config, hidden_size, lora_rank) if layer_idx == target_layer_idx else nn.Identity()
            layers.append(DummyBlock(mlp))
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class TrainerSmokeDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, qlen: int, hidden_size: int):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(9001)
        self.input_features = (
            torch.randn((samples, qlen, hidden_size), generator=generator, dtype=torch.float32) * 0.01
        ).to(torch.bfloat16)
        self.labels = (
            torch.randn((samples, qlen, hidden_size), generator=generator, dtype=torch.float32) * 0.02
        ).to(torch.bfloat16)

    def __len__(self) -> int:
        return self.input_features.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_features": self.input_features[idx],
            "labels": self.labels[idx],
        }


class TrainerSmokeModel(nn.Module):
    def __init__(self, model: DummyKimiModel):
        super().__init__()
        self.model = model
        self.config = model.config

    def forward(
        self,
        input_features: torch.Tensor,
        labels: torch.Tensor | None = None,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        output = self.model(input_features.to(torch.bfloat16))
        target = torch.zeros_like(output) if labels is None else labels.to(output.device)
        loss = torch.nn.functional.mse_loss(output.float(), target.float()) * TRAINER_SMOKE_LOSS_SCALE
        return {"loss": loss, "logits": output.float()}


def append_trainer_extra_site() -> None:
    extra_site = os.environ.get("KT_TRAINER_EXTRA_SITE")
    if extra_site and extra_site not in sys.path:
        sys.path.append(extra_site)


def run_backward_sample_reference(
    wrapper,
    kgroup_weights,
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    route_weights: torch.Tensor,
    active_experts: list[int],
    local_counts: list[int],
    group_size: int,
) -> None:
    hidden_size = x.shape[1]
    qlen, topk = expert_ids.shape
    intermediate_size = kgroup_weights.gate_scale.shape[1]
    total_tokens = sum(int(local_counts[expert]) for expert in active_experts)

    input_cache = torch.empty((qlen, hidden_size), dtype=torch.bfloat16)
    gate_cache = torch.empty((total_tokens, intermediate_size), dtype=torch.bfloat16)
    up_cache = torch.empty((total_tokens, intermediate_size), dtype=torch.bfloat16)
    intermediate_cache = torch.empty((total_tokens, intermediate_size), dtype=torch.bfloat16)
    down_cache = torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16)
    wrapper.moe.debug_copy_forward_cache(
        input_cache.data_ptr(),
        gate_cache.data_ptr(),
        up_cache.data_ptr(),
        intermediate_cache.data_ptr(),
        down_cache.data_ptr(),
        0,
    )

    packed_ref = {"gate": {}, "up": {}, "down": {}}
    for expert in active_experts:
        packed_ref["gate"][expert] = dequant_kgroup_weight(
            kgroup_weights.gate_proj[expert],
            kgroup_weights.gate_scale[expert],
            intermediate_size,
            hidden_size,
            group_size,
        )
        packed_ref["up"][expert] = dequant_kgroup_weight(
            kgroup_weights.up_proj[expert],
            kgroup_weights.up_scale[expert],
            intermediate_size,
            hidden_size,
            group_size,
        )
        packed_ref["down"][expert] = dequant_kgroup_weight(
            kgroup_weights.down_proj[expert],
            kgroup_weights.down_scale[expert],
            hidden_size,
            intermediate_size,
            group_size,
        )

    base_offset = {}
    cursor = 0
    for expert in active_experts:
        base_offset[expert] = cursor
        cursor += int(local_counts[expert])

    grad_output = (torch.randn((qlen, hidden_size), dtype=torch.float32) * 0.02).to(torch.bfloat16).contiguous()
    grad_down = torch.empty((total_tokens, hidden_size), dtype=torch.bfloat16)
    grad_intermediate = torch.empty((total_tokens, intermediate_size), dtype=torch.bfloat16)
    wrapper.moe.debug_backward_down_sample(
        grad_output.data_ptr(),
        grad_down.data_ptr(),
        grad_intermediate.data_ptr(),
        0,
        0,
    )

    per_expert_pos = {expert: 0 for expert in active_experts}
    grad_down_ref = torch.zeros((total_tokens, hidden_size), dtype=torch.float32)
    for token_idx in range(qlen):
        for route_idx in range(topk):
            expert = int(expert_ids[token_idx, route_idx])
            row = base_offset[expert] + per_expert_pos[expert]
            grad_down_ref[row] += grad_output[token_idx].float() * route_weights[token_idx, route_idx].item()
            per_expert_pos[expert] += 1

    grad_intermediate_ref = torch.zeros((total_tokens, intermediate_size), dtype=torch.float32)
    for expert in active_experts:
        start = base_offset[expert]
        end = start + int(local_counts[expert])
        grad_intermediate_ref[start:end] = grad_down_ref[start:end] @ packed_ref["down"][expert]

    assert_close("real_backward_down_grad_down", grad_down, grad_down_ref.to(torch.bfloat16), atol=0.0, rtol=0.0)
    assert_close(
        "real_backward_down_grad_intermediate",
        grad_intermediate,
        grad_intermediate_ref.to(torch.bfloat16),
        atol=0.02,
        rtol=0.08,
    )

    grad_gate = torch.empty_like(gate_cache)
    grad_up = torch.empty_like(up_cache)
    activation_grad_intermediate = torch.empty_like(intermediate_cache)
    wrapper.moe.debug_backward_activation_sample(
        grad_output.data_ptr(),
        activation_grad_intermediate.data_ptr(),
        grad_gate.data_ptr(),
        grad_up.data_ptr(),
    )
    sigmoid = torch.sigmoid(gate_cache.float())
    silu = torch.nn.functional.silu(gate_cache.float())
    silu_grad = sigmoid * (1.0 + gate_cache.float() * (1.0 - sigmoid))
    grad_gate_ref = grad_intermediate_ref * up_cache.float() * silu_grad
    grad_up_ref = grad_intermediate_ref * silu
    assert_close(
        "real_backward_activation_grad_intermediate",
        activation_grad_intermediate,
        grad_intermediate_ref.to(torch.bfloat16),
        atol=0.02,
        rtol=0.08,
    )
    assert_close(
        "real_backward_activation_grad_gate",
        grad_gate,
        grad_gate_ref.to(torch.bfloat16),
        atol=0.01,
        rtol=0.08,
    )
    assert_close("real_backward_activation_grad_up", grad_up, grad_up_ref.to(torch.bfloat16), atol=0.01, rtol=0.08)

    grad_input = torch.empty((qlen, hidden_size), dtype=torch.bfloat16)
    wrapper.moe.debug_backward_gate_up_sample(grad_output.data_ptr(), grad_input.data_ptr(), 0, 0, 0, 0)

    per_expert_pos = {expert: 0 for expert in active_experts}
    grad_input_ref = torch.zeros((qlen, hidden_size), dtype=torch.float32)
    for token_idx in range(qlen):
        for route_idx in range(topk):
            expert = int(expert_ids[token_idx, route_idx])
            row = base_offset[expert] + per_expert_pos[expert]
            per_expert_pos[expert] += 1
            grad_input_ref[token_idx] += grad_gate_ref[row] @ packed_ref["gate"][expert]
            grad_input_ref[token_idx] += grad_up_ref[row] @ packed_ref["up"][expert]

    assert_close(
        "real_backward_gate_up_grad_input",
        grad_input,
        grad_input_ref.to(torch.bfloat16),
        atol=0.05,
        rtol=0.08,
    )
    print(
        "real_kimi_backward_sample_passed: "
        f"active_count={len(active_experts)}, cached_tokens={total_tokens}"
    )


def assert_nonzero(name: str, value: torch.Tensor) -> None:
    max_abs = value.float().abs().max().item()
    print(f"{name}: observed_max_abs={max_abs:.6g}")
    if max_abs == 0.0:
        raise AssertionError(f"{name} did not change")


def assert_forward_cache_empty(moe, label: str) -> None:
    try:
        moe.debug_cache_summary()
    except RuntimeError as exc:
        if "forward cache is empty" not in str(exc):
            raise
        print(f"{label}: observed empty cache")
        return
    raise AssertionError(f"{label} did not consume the forward cache")


def iter_peft_lora_weights(peft_lora_modules):
    for expert_idx in sorted(peft_lora_modules):
        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            if proj_name not in peft_lora_modules[expert_idx]:
                continue
            for lora_index, module in enumerate(peft_lora_modules[expert_idx][proj_name]):
                yield f"expert{expert_idx}.{proj_name}.{lora_index}", module.weight


def max_peft_weight_delta(before: dict[str, torch.Tensor], peft_lora_modules) -> torch.Tensor:
    deltas = []
    for name, weight in iter_peft_lora_weights(peft_lora_modules):
        deltas.append((weight.detach().float() - before[name].float()).abs().max())
    return torch.stack(deltas).max()


def assert_peft_grad_alias(peft_lora_modules, wrapper, expert_num: int) -> None:
    grad_sources = {
        "gate_proj": (wrapper.grad_gate_lora_a, wrapper.grad_gate_lora_b),
        "up_proj": (wrapper.grad_up_lora_a, wrapper.grad_up_lora_b),
        "down_proj": (wrapper.grad_down_lora_a, wrapper.grad_down_lora_b),
    }
    for expert_idx in range(expert_num):
        expert_loras = peft_lora_modules.get(expert_idx)
        if expert_loras is None:
            raise AssertionError(f"missing PEFT LoRA modules for expert {expert_idx}")
        for proj_name, expected_pair in grad_sources.items():
            if proj_name not in expert_loras:
                raise AssertionError(f"missing PEFT LoRA module {proj_name} for expert {expert_idx}")
            for lora_module, expected_grad in zip(expert_loras[proj_name], expected_pair):
                actual_grad = lora_module.weight.grad
                expected_slice = expected_grad[expert_idx]
                if actual_grad is None:
                    raise AssertionError(f"{proj_name} expert {expert_idx} grad is None")
                if actual_grad.data_ptr() != expected_slice.data_ptr():
                    raise AssertionError(f"{proj_name} expert {expert_idx} grad buffer is not aliased")


def assert_peft_any_grad_nonzero(label: str, peft_lora_modules) -> None:
    max_abs = 0.0
    seen = False
    for expert_loras in peft_lora_modules.values():
        for lora_a, lora_b in expert_loras.values():
            for module in (lora_a, lora_b):
                grad = module.weight.grad
                if grad is None:
                    continue
                seen = True
                max_abs = max(max_abs, grad.float().abs().max().item())
    print(f"{label}: observed_max_abs={max_abs:.6g}")
    if not seen:
        raise AssertionError(f"{label} did not observe any LoRA grad buffers")
    if max_abs == 0.0:
        raise AssertionError(f"{label} LoRA gradients are all zero")


def run_wrap_backward_smoke(args, config, moe_config, text_config, kt_config) -> None:
    from kt_kernel.sft.lora import kt_adapt_peft_lora, update_kt_lora_pointers
    from kt_kernel.sft.wrapper import wrap_moe_layers_with_kt_wrapper

    dummy_model = DummyKimiModel(config, moe_config, args.layer_idx, text_config.hidden_size, args.rank)
    wrapped_layers = wrap_moe_layers_with_kt_wrapper(dummy_model, kt_config)
    dummy_model._kt_wrappers = wrapped_layers
    if len(wrapped_layers) != 1:
        raise AssertionError(f"wrap backward smoke expected 1 wrapped layer, got {len(wrapped_layers)}")

    wrapped_layer = wrapped_layers[0]
    wrapper = wrapped_layer.wrapper
    if wrapped_layer.layer_idx != args.layer_idx:
        raise AssertionError(f"wrapped layer index mismatch: expected {args.layer_idx}, got {wrapped_layer.layer_idx}")
    if wrapper.method != "AMXINT4_KGroup_SFT":
        raise AssertionError(f"wrapped method mismatch: {wrapper.method}")
    expected_threadpool_count = args.tp_count if args.tp_count > 1 else 1
    if wrapper.threadpool_count != expected_threadpool_count:
        raise AssertionError(
            "wrap backward smoke threadpool_count mismatch: "
            f"expected {expected_threadpool_count}, got {wrapper.threadpool_count}"
        )
    if getattr(wrapper, "share_backward_bb", False):
        raise AssertionError("KGroup wrap backward smoke must keep share_backward_bb disabled")
    if args.tp_count == 1:
        if not getattr(wrapper, "_supports_tp1_packed_backward", False):
            raise AssertionError("wrapped KGroup wrapper did not report TP=1 packed backward support")
    elif not getattr(wrapper, "_supports_kgroup_packed_backward", False):
        raise AssertionError("wrapped KGroup wrapper did not report packed backward support")

    def unexpected_submit_repack():
        raise AssertionError("KGroup wrap backward smoke must not submit async backward repack")

    def unexpected_wait_repack():
        raise AssertionError("KGroup wrap backward smoke must not wait async backward repack")

    wrapper.submit_backward_repack = unexpected_submit_repack
    wrapper.wait_backward_repack = unexpected_wait_repack

    kt_adapt_peft_lora(dummy_model)
    if not getattr(wrapper, "_lora_initialized", False):
        raise AssertionError("kt_adapt_peft_lora did not initialize real Kimi wrapper LoRA weights")
    if wrapped_layer._peft_lora_modules is None:
        raise AssertionError("kt_adapt_peft_lora did not attach real Kimi PEFT LoRA modules")

    dummy_model.train()
    dummy_model.zero_grad(set_to_none=True)
    x = (torch.randn((1, args.qlen, text_config.hidden_size), dtype=torch.float32) * 0.01).to(torch.bfloat16)
    x.requires_grad_(True)
    output = wrapped_layer(x)
    assert_peft_grad_alias(wrapped_layer._peft_lora_modules, wrapper, moe_config.expert_num)

    grad_output = (torch.randn(output.shape, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    loss = (output.float() * grad_output.float()).sum()
    loss.backward()
    if x.grad is None:
        raise AssertionError("real Kimi wrap backward did not produce input grad")
    assert_peft_grad_alias(wrapped_layer._peft_lora_modules, wrapper, moe_config.expert_num)
    assert_peft_any_grad_nonzero("real_kimi_wrap_backward_lora_grad", wrapped_layer._peft_lora_modules)
    if wrapper._cache_depth != 0:
        raise AssertionError(f"real Kimi wrapper cache depth mismatch: expected 0, got {wrapper._cache_depth}")
    assert_forward_cache_empty(wrapper.moe, "real_kimi_wrap_backward_cache_pop")

    pre_step_output = output.detach().clone()
    optimizer_params = [param for param in dummy_model.parameters() if param.requires_grad]
    if not optimizer_params:
        raise AssertionError("real Kimi wrap backward smoke found no trainable LoRA parameters")
    before_step = {
        name: weight.detach().clone()
        for name, weight in iter_peft_lora_weights(wrapped_layer._peft_lora_modules)
    }
    optimizer = torch.optim.SGD(optimizer_params, lr=32768.0)
    optimizer.step()
    assert_nonzero(
        "real_kimi_wrap_optimizer_step_lora_delta",
        max_peft_weight_delta(before_step, wrapped_layer._peft_lora_modules),
    )

    update_kt_lora_pointers(dummy_model)
    if not wrapped_layer._lora_pointers_dirty:
        raise AssertionError("update_kt_lora_pointers did not mark real Kimi wrapped layer dirty")

    optimizer.zero_grad(set_to_none=True)
    post_input = x.detach().clone().requires_grad_(True)
    post_output = wrapped_layer(post_input)
    if wrapped_layer._lora_pointers_dirty:
        raise AssertionError("real Kimi wrapped layer did not consume dirty LoRA pointer flag")
    assert_peft_grad_alias(wrapped_layer._peft_lora_modules, wrapper, moe_config.expert_num)
    assert_nonzero(
        "real_kimi_wrap_post_step_forward_delta",
        (post_output.detach().float() - pre_step_output.float()).abs().max(),
    )

    post_loss = (post_output.float() * grad_output.float()).sum()
    post_loss.backward()
    if post_input.grad is None:
        raise AssertionError("real Kimi post-step wrap backward did not produce input grad")
    assert_peft_grad_alias(wrapped_layer._peft_lora_modules, wrapper, moe_config.expert_num)
    if wrapper._cache_depth != 0:
        raise AssertionError(f"real Kimi post-step wrapper cache depth mismatch: expected 0, got {wrapper._cache_depth}")
    assert_forward_cache_empty(wrapper.moe, "real_kimi_wrap_post_step_cache_pop")

    print(
        "real_kimi_wrap_backward_smoke_passed: "
        f"layer={wrapped_layer.layer_idx}, method={wrapper.method}, experts={moe_config.expert_num}, "
        f"topk={moe_config.num_experts_per_tok}, qlen={args.qlen}, rank={args.rank}"
    )


def make_training_arguments(output_dir: str, trainer_steps: int):
    append_trainer_extra_site()
    from transformers import TrainingArguments

    kwargs = {
        "output_dir": output_dir,
        "max_steps": trainer_steps,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1.0,
        "logging_steps": 1,
        "save_strategy": "no",
        "report_to": [],
        "remove_unused_columns": False,
        "disable_tqdm": True,
        "dataloader_num_workers": 0,
        "bf16": False,
        "fp16": False,
    }
    try:
        return TrainingArguments(**kwargs, use_cpu=True)
    except TypeError:
        return TrainingArguments(**kwargs, no_cuda=True)


def run_trainer_smoke(args, config, moe_config, text_config, kt_config) -> None:
    append_trainer_extra_site()
    from transformers import Trainer, TrainerCallback
    from kt_kernel.sft.lora import kt_adapt_peft_lora, update_kt_lora_pointers
    from kt_kernel.sft.wrapper import wrap_moe_layers_with_kt_wrapper

    class KTLoraPointerUpdateCallback(TrainerCallback):
        def __init__(self, target_model: nn.Module):
            self.target_model = target_model
            self.step_end_calls = 0

        def on_step_end(self, args, state, control, **kwargs):
            update_kt_lora_pointers(self.target_model)
            self.step_end_calls += 1

    dummy_model = DummyKimiModel(config, moe_config, args.layer_idx, text_config.hidden_size, args.rank)
    wrapped_layers = wrap_moe_layers_with_kt_wrapper(dummy_model, kt_config)
    dummy_model._kt_wrappers = wrapped_layers
    if len(wrapped_layers) != 1:
        raise AssertionError(f"trainer smoke expected 1 wrapped layer, got {len(wrapped_layers)}")

    wrapped_layer = wrapped_layers[0]
    wrapper = wrapped_layer.wrapper
    if wrapper.threadpool_count != 1:
        raise AssertionError(f"trainer smoke requires TP=1/threadpool_count=1, got {wrapper.threadpool_count}")
    if not getattr(wrapper, "_supports_tp1_packed_backward", False):
        raise AssertionError("trainer smoke requires TP=1 packed backward support")
    if getattr(wrapper, "share_backward_bb", False):
        raise AssertionError("KGroup TP=1 trainer smoke must keep share_backward_bb disabled")

    def unexpected_submit_repack():
        raise AssertionError("KGroup TP=1 trainer smoke must not submit async backward repack")

    def unexpected_wait_repack():
        raise AssertionError("KGroup TP=1 trainer smoke must not wait async backward repack")

    wrapper.submit_backward_repack = unexpected_submit_repack
    wrapper.wait_backward_repack = unexpected_wait_repack

    kt_adapt_peft_lora(dummy_model)
    if wrapped_layer._peft_lora_modules is None:
        raise AssertionError("trainer smoke did not attach PEFT LoRA modules")

    trainer_model = TrainerSmokeModel(dummy_model)
    trainer_model.train()
    dataset = TrainerSmokeDataset(
        samples=max(args.trainer_steps, 2),
        qlen=args.qlen,
        hidden_size=text_config.hidden_size,
    )
    sample = dataset[0]
    with torch.no_grad():
        pre_step_output = trainer_model(
            input_features=sample["input_features"].unsqueeze(0),
            labels=sample["labels"].unsqueeze(0),
        )["logits"].detach().clone()

    before_step = {
        name: weight.detach().clone()
        for name, weight in iter_peft_lora_weights(wrapped_layer._peft_lora_modules)
    }
    optimizer_params = [param for param in trainer_model.parameters() if param.requires_grad]
    if not optimizer_params:
        raise AssertionError("trainer smoke found no trainable LoRA parameters")
    optimizer = torch.optim.SGD(optimizer_params, lr=32768.0)
    callback = KTLoraPointerUpdateCallback(dummy_model)

    with tempfile.TemporaryDirectory(prefix="kt-kimi-trainer-smoke-") as output_dir:
        trainer = Trainer(
            model=trainer_model,
            args=make_training_arguments(output_dir, args.trainer_steps),
            train_dataset=dataset,
            optimizers=(optimizer, None),
            callbacks=[callback],
        )
        train_result = trainer.train()
        if trainer.state.global_step != args.trainer_steps:
            raise AssertionError(
                f"trainer smoke global_step mismatch: expected {args.trainer_steps}, got {trainer.state.global_step}"
            )

    assert_nonzero(
        "real_kimi_trainer_optimizer_step_lora_delta",
        max_peft_weight_delta(before_step, wrapped_layer._peft_lora_modules),
    )
    if callback.step_end_calls != args.trainer_steps:
        raise AssertionError(
            f"trainer smoke callback step count mismatch: expected {args.trainer_steps}, got {callback.step_end_calls}"
        )
    if wrapper._cache_depth != 0:
        raise AssertionError(f"trainer smoke wrapper cache depth mismatch: expected 0, got {wrapper._cache_depth}")
    assert_forward_cache_empty(wrapper.moe, "real_kimi_trainer_backward_cache_pop")

    if not wrapped_layer._lora_pointers_dirty:
        raise AssertionError("trainer smoke callback did not mark LoRA pointers dirty after final optimizer step")

    with torch.no_grad():
        post_step_output = trainer_model(
            input_features=sample["input_features"].unsqueeze(0),
            labels=sample["labels"].unsqueeze(0),
        )["logits"].detach()
    if wrapped_layer._lora_pointers_dirty:
        raise AssertionError("trainer smoke post-step forward did not consume dirty LoRA pointer flag")
    assert_peft_grad_alias(wrapped_layer._peft_lora_modules, wrapper, moe_config.expert_num)
    assert_nonzero(
        "real_kimi_trainer_post_step_forward_delta",
        (post_step_output.float() - pre_step_output.float()).abs().max(),
    )

    metrics = train_result.metrics
    print(
        "real_kimi_trainer_smoke_passed: "
        f"layer={wrapped_layer.layer_idx}, method={wrapper.method}, steps={trainer.state.global_step}, "
        f"loss={metrics.get('train_loss')}, qlen={args.qlen}, rank={args.rank}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Load one real Kimi/K2 compressed KGroup MoE layer into SFT wrapper.")
    parser.add_argument("--model-path", default="/mnt/data2/models/Kimi-K2.5")
    parser.add_argument("--layer-idx", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--tp-count", type=int, default=1)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--qlen", type=int, default=1)
    parser.add_argument("--skip-forward", action="store_true")
    parser.add_argument("--inference-async-smoke", action="store_true")
    parser.add_argument("--save-for-backward", action="store_true")
    parser.add_argument("--backward-sample", action="store_true")
    parser.add_argument("--wrap-smoke", action="store_true")
    parser.add_argument("--wrap-backward-smoke", action="store_true")
    parser.add_argument("--trainer-smoke", action="store_true")
    parser.add_argument("--trainer-steps", type=int, default=2)
    args = parser.parse_args()
    if args.backward_sample and args.tp_count != 1:
        raise ValueError("--backward-sample currently supports --tp-count 1 only")
    if args.backward_sample and not args.save_for_backward:
        raise ValueError("--backward-sample requires --save-for-backward")
    if args.trainer_smoke and args.tp_count != 1:
        raise ValueError("--trainer-smoke currently supports --tp-count 1 only")
    if args.trainer_steps < 1:
        raise ValueError("--trainer-steps must be >= 1")

    if args.trainer_smoke:
        append_trainer_extra_site()

    from transformers import AutoConfig
    from kt_kernel.sft.amx import AMXSFTMoEWrapper

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    text_config = getattr(config, "text_config", config)
    moe_config = get_moe_arch_config(config)
    layers_prefix = _get_layers_prefix(config)
    group_size = 32

    kt_config = KTConfig(
        kt_backend="AMXINT4_KGroup",
        kt_weight_path=args.model_path,
        kt_num_threads=args.threads * args.tp_count,
        kt_tp_enabled=args.tp_count > 1,
        kt_threadpool_count=args.tp_count,
        kt_lora_rank=args.rank,
        kt_lora_alpha=float(args.rank),
        kt_group_size=group_size,
        kt_zero_point=False,
        kt_share_backward_bb=False,
    )

    if not has_kgroup_experts_in_kt_weight_path(args.model_path, layers_prefix, moe_config, args.layer_idx):
        raise FileNotFoundError(
            f"No compressed KGroup expert tensors found for layer {args.layer_idx} under {args.model_path}"
        )

    if args.wrap_smoke:
        from kt_kernel.sft.wrapper import wrap_moe_layers_with_kt_wrapper

        dummy_model = DummyKimiModel(config, moe_config, args.layer_idx, text_config.hidden_size, args.rank)
        wrapped_layers = wrap_moe_layers_with_kt_wrapper(dummy_model, kt_config)
        if len(wrapped_layers) != 1:
            raise AssertionError(f"wrap smoke expected 1 wrapped layer, got {len(wrapped_layers)}")
        wrapped_layer = wrapped_layers[0]
        if wrapped_layer.layer_idx != args.layer_idx:
            raise AssertionError(f"wrapped layer index mismatch: expected {args.layer_idx}, got {wrapped_layer.layer_idx}")
        if wrapped_layer.wrapper.method != "AMXINT4_KGroup_SFT":
            raise AssertionError(f"wrapped method mismatch: {wrapped_layer.wrapper.method}")
        if not getattr(wrapped_layer.wrapper, "_weights_loaded", False):
            raise AssertionError("wrap smoke did not load KGroup weights")
        print(
            "real_kimi_wrap_smoke_passed: "
            f"layer={wrapped_layer.layer_idx}, method={wrapped_layer.wrapper.method}, "
            f"experts={moe_config.expert_num}, topk={moe_config.num_experts_per_tok}"
        )
        return

    if args.wrap_backward_smoke:
        run_wrap_backward_smoke(args, config, moe_config, text_config, kt_config)
        return

    if args.trainer_smoke:
        run_trainer_smoke(args, config, moe_config, text_config, kt_config)
        return

    kgroup_weights = load_kgroup_experts_from_kt_weight_path(
        kt_weight_path=args.model_path,
        layers_prefix=layers_prefix,
        moe_config=moe_config,
        layer_idx=args.layer_idx,
        hidden_size=text_config.hidden_size,
        group_size=group_size,
    )
    print(
        "real_kimi_kgroup_tensors_loaded: "
        f"layer={args.layer_idx}, experts={moe_config.expert_num}, "
        f"gate_proj={tuple(kgroup_weights.gate_proj.shape)}, gate_scale={tuple(kgroup_weights.gate_scale.shape)}"
    )

    wrapper = AMXSFTMoEWrapper(
        layer_idx=args.layer_idx,
        num_experts=moe_config.expert_num,
        num_experts_per_tok=moe_config.num_experts_per_tok,
        hidden_size=text_config.hidden_size,
        moe_intermediate_size=moe_config.intermediate_size,
        num_gpu_experts=0,
        cpuinfer_threads=args.threads * args.tp_count,
        threadpool_count=args.tp_count,
        weight_path=args.model_path,
        chunked_prefill_size=args.qlen,
        lora_rank=args.rank,
        lora_alpha=float(args.rank),
        max_cache_depth=1,
        method="AMXINT4_KGroup_SFT",
        group_size=group_size,
        zero_point=False,
    )
    wrapper.share_backward_bb = False
    wrapper.load_kgroup_weights_from_tensors(
        gate_proj=kgroup_weights.gate_proj,
        gate_scale=kgroup_weights.gate_scale,
        up_proj=kgroup_weights.up_proj,
        up_scale=kgroup_weights.up_scale,
        down_proj=kgroup_weights.down_proj,
        down_scale=kgroup_weights.down_scale,
        physical_to_logical_map_cpu=torch.arange(moe_config.expert_num, dtype=torch.int64),
    )
    if args.tp_count == 1:
        (
            ready,
            summary_experts,
            summary_hidden,
            summary_intermediate,
            summary_group_size,
            gate_up_bytes,
            down_bytes,
            gate_up_scale_elems,
            down_scale_elems,
        ) = wrapper.moe.debug_packed_weight_summary()
        if not ready:
            raise AssertionError("packed weight summary reported not ready")
        if summary_experts != moe_config.expert_num:
            raise AssertionError(f"summary expert mismatch: expected {moe_config.expert_num}, got {summary_experts}")
        if summary_hidden != text_config.hidden_size or summary_intermediate != moe_config.intermediate_size:
            raise AssertionError(
                "summary dimension mismatch: "
                f"H={summary_hidden}/{text_config.hidden_size}, I={summary_intermediate}/{moe_config.intermediate_size}"
            )
        if summary_group_size != group_size:
            raise AssertionError(f"summary group_size mismatch: expected {group_size}, got {summary_group_size}")
        if gate_up_bytes != moe_config.intermediate_size * text_config.hidden_size // 2:
            raise AssertionError(f"summary gate/up bytes mismatch: got {gate_up_bytes}")
        if down_bytes != text_config.hidden_size * moe_config.intermediate_size // 2:
            raise AssertionError(f"summary down bytes mismatch: got {down_bytes}")
        if gate_up_scale_elems != moe_config.intermediate_size * (text_config.hidden_size // group_size):
            raise AssertionError(f"summary gate/up scale elems mismatch: got {gate_up_scale_elems}")
        if down_scale_elems != text_config.hidden_size * (moe_config.intermediate_size // group_size):
            raise AssertionError(f"summary down scale elems mismatch: got {down_scale_elems}")
        print(
            "real_kimi_packed_weight_summary_passed: "
            f"ready={ready}, gate_up_bytes={gate_up_bytes}, down_bytes={down_bytes}, "
            f"gate_up_scale_elems={gate_up_scale_elems}, down_scale_elems={down_scale_elems}"
        )
    if not args.backward_sample:
        del kgroup_weights

    lora = {
        "gate_a": torch.zeros((moe_config.expert_num, args.rank, text_config.hidden_size), dtype=torch.bfloat16),
        "gate_b": torch.zeros((moe_config.expert_num, moe_config.intermediate_size, args.rank), dtype=torch.bfloat16),
        "up_a": torch.zeros((moe_config.expert_num, args.rank, text_config.hidden_size), dtype=torch.bfloat16),
        "up_b": torch.zeros((moe_config.expert_num, moe_config.intermediate_size, args.rank), dtype=torch.bfloat16),
        "down_a": torch.zeros((moe_config.expert_num, args.rank, moe_config.intermediate_size), dtype=torch.bfloat16),
        "down_b": torch.zeros((moe_config.expert_num, text_config.hidden_size, args.rank), dtype=torch.bfloat16),
    }
    grad_lora = {name: torch.zeros_like(tensor) for name, tensor in lora.items()}
    wrapper.init_lora_weights(
        lora["gate_a"],
        lora["gate_b"],
        lora["up_a"],
        lora["up_b"],
        lora["down_a"],
        lora["down_b"],
        grad_lora["gate_a"],
        grad_lora["gate_b"],
        grad_lora["up_a"],
        grad_lora["up_b"],
        grad_lora["down_a"],
        grad_lora["down_b"],
    )
    print(
        "real_kimi_wrapper_loaded: "
        f"layer={args.layer_idx}, method={wrapper.method}, tp_count={args.tp_count}, "
        f"topk={moe_config.num_experts_per_tok}"
    )

    if args.skip_forward:
        return

    x = torch.randn((args.qlen, text_config.hidden_size), dtype=torch.float32).to(torch.bfloat16) * 0.01
    expert_ids = torch.stack(
        [torch.arange(i, i + moe_config.num_experts_per_tok) % moe_config.expert_num for i in range(args.qlen)],
        dim=0,
    ).to(torch.int64)
    route_weights = torch.full(
        (args.qlen, moe_config.num_experts_per_tok),
        1.0 / moe_config.num_experts_per_tok,
        dtype=torch.float32,
    )
    if args.inference_async_smoke:
        wrapper.submit_forward_inference(x, expert_ids, route_weights)
        output = wrapper.sync_forward_inference()
        if getattr(wrapper, "_cache_depth", 0) != 0:
            raise AssertionError(
                f"inference async smoke should not keep training cache, got cache_depth={wrapper._cache_depth}"
            )
        print(
            "real_kimi_inference_async_forward_passed: "
            f"output_shape={tuple(output.shape)}, max_abs={output.float().abs().max().item():.6g}"
        )
        return

    output = wrapper.forward(x, expert_ids, route_weights, save_for_backward=args.save_for_backward)
    print(
        "real_kimi_forward_passed: "
        f"output_shape={tuple(output.shape)}, max_abs={output.float().abs().max().item():.6g}"
    )
    if args.save_for_backward:
        qlen, k, active_count, active_experts, local_counts = wrapper.moe.debug_cache_summary()
        active_experts = [int(x) for x in active_experts[:active_count]]
        cached_tokens = sum(int(local_counts[expert]) for expert in active_experts)
        if qlen != args.qlen:
            raise AssertionError(f"cache qlen mismatch: expected {args.qlen}, got {qlen}")
        if k != moe_config.num_experts_per_tok:
            raise AssertionError(f"cache topk mismatch: expected {moe_config.num_experts_per_tok}, got {k}")
        print(
            "real_kimi_forward_cache_passed: "
            f"active_count={active_count}, cached_tokens={cached_tokens}, "
            f"first_active_experts={active_experts[:8]}"
        )
        if args.backward_sample:
            run_backward_sample_reference(
                wrapper,
                kgroup_weights,
                x,
                expert_ids,
                route_weights,
                active_experts,
                local_counts,
                group_size,
            )


if __name__ == "__main__":
    main()
