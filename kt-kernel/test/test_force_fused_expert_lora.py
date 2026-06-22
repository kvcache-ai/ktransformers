from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from kt_kernel.sft.arch import KTAMXConfigError
from kt_kernel.sft.config import KTConfig
from kt_kernel.sft.lora import get_kt_lora_params, kt_adapt_peft_lora, save_kt_moe_to_adapter
from kt_kernel.sft.wrapper import wrap_moe_layers_with_kt_wrapper


class DummyKTMoEWrapper:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.loaded_from_tensors = False
        self.lora_init = {}

    def load_weights_from_tensors(self, **kwargs):
        self.loaded_from_tensors = True
        self.loaded_tensors = kwargs

    def load_weights(self, physical_to_logical_map):
        self.loaded_from_path = physical_to_logical_map

    def load_kgroup_weights_from_tensors(self, **kwargs):
        self.loaded_kgroup_tensors = kwargs

    def init_lora_weights(self, **kwargs):
        self.lora_init = kwargs


class FakeExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


class FakeMoE(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([FakeExpert(hidden_size, intermediate_size) for _ in range(num_experts)])


class FakeLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int):
        super().__init__()
        self.mlp = FakeMoE(hidden_size, intermediate_size, num_experts)


class FakeKimiModel(nn.Module):
    def __init__(self, *, hidden_size: int = 8, intermediate_size: int = 4, num_experts: int = 2):
        super().__init__()
        self.config = SimpleNamespace(
            architectures=["KimiK25ForCausalLM"],
            hidden_size=hidden_size,
            n_routed_experts=num_experts,
            moe_intermediate_size=intermediate_size,
            num_experts_per_tok=1,
            max_position_embeddings=64,
        )
        self.layers = nn.ModuleList([FakeLayer(hidden_size, intermediate_size, num_experts)])


@pytest.fixture(autouse=True)
def patch_native_wrapper(monkeypatch):
    import kt_kernel.sft.wrapper as wrapper_module

    monkeypatch.setattr(wrapper_module, "KT_KERNEL_AVAILABLE", True)
    monkeypatch.setattr(wrapper_module, "KTMoEWrapper", DummyKTMoEWrapper)


def test_force_fused_expert_lora_creates_and_saves_kt_managed_params(tmp_path):
    model = FakeKimiModel()
    config = KTConfig(
        kt_backend="AMXBF16",
        kt_force_fused_expert_lora=True,
        kt_lora_rank=3,
        kt_lora_alpha=6,
        kt_sync_after_wrap=False,
    )

    wrappers = wrap_moe_layers_with_kt_wrapper(model, config)
    model._kt_wrappers = wrappers

    assert len(wrappers) == 1
    assert wrappers[0]._fused_experts is True

    kt_adapt_peft_lora(model)

    params = get_kt_lora_params(model)
    assert len(params) == 6
    assert [tuple(param.shape) for param in params] == [
        (2, 3, 8),
        (2, 4, 3),
        (2, 3, 8),
        (2, 4, 3),
        (2, 3, 4),
        (2, 8, 3),
    ]
    assert set(wrappers[0].wrapper.lora_init) == {
        "gate_lora_a",
        "gate_lora_b",
        "up_lora_a",
        "up_lora_b",
        "down_lora_a",
        "down_lora_b",
        "grad_gate_lora_a",
        "grad_gate_lora_b",
        "grad_up_lora_a",
        "grad_up_lora_b",
        "grad_down_lora_a",
        "grad_down_lora_b",
    }

    save_kt_moe_to_adapter(model, str(tmp_path))

    from safetensors.torch import load_file

    saved = load_file(str(tmp_path / "fused_expert_lora.safetensors"))
    assert sorted(saved) == [
        "layers.0.experts.down_lora_a",
        "layers.0.experts.down_lora_b",
        "layers.0.experts.gate_lora_a",
        "layers.0.experts.gate_lora_b",
        "layers.0.experts.up_lora_a",
        "layers.0.experts.up_lora_b",
    ]
    assert tuple(saved["layers.0.experts.gate_lora_a"].shape) == (2, 3, 8)


def test_force_fused_expert_lora_rejects_skip_adaptation():
    model = FakeKimiModel()
    config = KTConfig(
        kt_backend="AMXBF16",
        kt_force_fused_expert_lora=True,
        kt_skip_expert_lora_adaptation=True,
        kt_sync_after_wrap=False,
    )

    with pytest.raises(KTAMXConfigError, match="kt_skip_expert_lora_adaptation"):
        wrap_moe_layers_with_kt_wrapper(model, config)


def test_force_fused_expert_lora_rejects_skip_lora_backend():
    model = FakeKimiModel()
    config = KTConfig(
        kt_backend="AMXINT4_SkipLoRA",
        kt_force_fused_expert_lora=True,
        kt_sync_after_wrap=False,
    )

    with pytest.raises(KTAMXConfigError, match="SkipLoRA"):
        wrap_moe_layers_with_kt_wrapper(model, config)
