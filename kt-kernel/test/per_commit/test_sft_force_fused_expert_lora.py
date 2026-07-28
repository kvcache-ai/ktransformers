# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from kt_kernel.sft.arch import KTAMXConfigError
from kt_kernel.sft.config import KTConfig
from kt_kernel.sft.lora import (
    _load_fused_expert_lora,
    get_kt_lora_params,
    kt_adapt_peft_lora,
    load_kt_moe_from_adapter,
)
from kt_kernel.sft.wrapper import wrap_moe_layers_with_kt_wrapper

_FUSED_LORA_NAMES = (
    "gate_lora_a",
    "gate_lora_b",
    "up_lora_a",
    "up_lora_b",
    "down_lora_a",
    "down_lora_b",
)


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 3, bias=False)
        self.up_proj = nn.Linear(4, 3, bias=False)
        self.down_proj = nn.Linear(3, 4, bias=False)


class _MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, 2, bias=False)
        self.experts = nn.ModuleList([_Expert(), _Expert()])


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MoE()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            architectures=["DeepseekV3ForCausalLM"],
            hidden_size=4,
            n_routed_experts=2,
            n_shared_experts=0,
            moe_intermediate_size=3,
            num_experts_per_tok=1,
            max_position_embeddings=16,
        )
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Layer()])


class _Backend:
    def __init__(self, **kwargs):
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None
        self._full_weight_grad = kwargs["full_weight_grad"]
        self.initialized_lora = None
        self.authoritative_grads = []

    def load_weights(self, physical_to_logical_map):
        self.physical_to_logical_map = physical_to_logical_map

    def init_lora_weights(self, **buffers):
        self.initialized_lora = buffers

    def register_authoritative_optimizer_grad(self, name, parameter, grad_view):
        self.authoritative_grads.append((name, parameter, grad_view))


def _int8_config(*, force_fused=True):
    with patch("kt_kernel.sft.config.configure_omp_threads", return_value=1):
        return KTConfig(
            kt_expert_weight_format="int8",
            kt_weight_path="/persistent/kt-int8",
            kt_weight_lifecycle="persistent",
            kt_train_mode="lora",
            kt_lora_rank=8,
            kt_lora_alpha=16,
            kt_num_gpu_experts=0,
            kt_share_backward_bb=True,
            kt_activation_policy={"cpu": "retain", "gpu": "recompute"},
            kt_force_fused_expert_lora=force_fused,
        )


def _wrap(model, config, monkeypatch):
    import kt_kernel.sft.wrapper as wrapper_module

    monkeypatch.setattr(wrapper_module, "KT_KERNEL_AVAILABLE", True)
    monkeypatch.setattr(wrapper_module, "KTMoEWrapper", _Backend)
    return wrap_moe_layers_with_kt_wrapper(model, config)[0]


def test_force_fused_expert_lora_reads_environment(monkeypatch):
    monkeypatch.setenv("ACCELERATE_KT_FORCE_FUSED_EXPERT_LORA", "true")
    with patch("kt_kernel.sft.config.configure_omp_threads", return_value=1):
        config = KTConfig()
    assert config.kt_force_fused_expert_lora is True


def test_force_fused_lora_preserves_nonfused_base_and_is_rank0_owned(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    model = _Model()
    original_experts = model.model.layers[0].mlp.experts
    wrapper = _wrap(model, _int8_config(), monkeypatch)

    assert wrapper.experts is original_experts
    assert wrapper._fused_experts is False
    assert wrapper._use_fused_expert_lora is True
    assert wrapper._force_fused_expert_lora is True

    model._kt_wrappers = [wrapper]
    kt_adapt_peft_lora(model)

    params = get_kt_lora_params(model)
    assert len(params) == 6
    assert all(param.device.type == "cpu" for param in params)
    assert all(param.dtype == torch.bfloat16 for param in params)
    assert all(param.grad is None for param in params)
    assert len(wrapper.wrapper.authoritative_grads) == 6
    assert wrapper.wrapper.initialized_lora is not None


def test_int8_nonfused_experts_require_forced_fused_lora(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    with (
        patch("kt_kernel.sft.wrapper.KT_KERNEL_AVAILABLE", True),
        patch("kt_kernel.sft.wrapper.KTMoEWrapper", _Backend),
        pytest.raises(KTAMXConfigError, match="kt_force_fused_expert_lora=true"),
    ):
        wrap_moe_layers_with_kt_wrapper(
            _Model(),
            _int8_config(force_fused=False),
        )


def test_force_fused_lora_creates_no_rank1_params_or_backend(monkeypatch):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    model = _Model()
    wrapper = _wrap(model, _int8_config(), monkeypatch)
    model._kt_wrappers = [wrapper]

    kt_adapt_peft_lora(model)

    assert wrapper.wrapper is None
    assert wrapper._fused_expert_lora_params == []
    assert get_kt_lora_params(model) == []


def test_force_fused_lora_reload_skips_rank1_without_owned_params(tmp_path):
    from safetensors.torch import save_file

    save_file(
        {"layers.0.experts.gate_lora_a": torch.zeros(1)},
        str(tmp_path / "fused_expert_lora.safetensors"),
    )
    wrapper = SimpleNamespace(
        layer_idx=0,
        lora_experts=None,
        _full_weight_grad=False,
        _fused_expert_lora_params=[],
    )
    model = SimpleNamespace(_kt_wrappers=[wrapper])

    load_kt_moe_from_adapter(model, str(tmp_path))


def _owned_reload_wrapper(layer_idx=0):
    return SimpleNamespace(
        layer_idx=layer_idx,
        lora_experts=None,
        _full_weight_grad=False,
        _fused_expert_lora_params=[
            nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16))
            for _ in _FUSED_LORA_NAMES
        ],
        _lora_pointers_dirty=False,
    )


def _saved_fused_tensors(layer_idx=0):
    return {
        f"layers.{layer_idx}.experts.{name}": torch.full(
            (2, 2), index + 1, dtype=torch.bfloat16
        )
        for index, name in enumerate(_FUSED_LORA_NAMES)
    }


def test_force_fused_lora_reload_requires_and_consumes_exact_key_set(tmp_path):
    from safetensors.torch import save_file

    tensors = _saved_fused_tensors()
    save_file(tensors, str(tmp_path / "fused_expert_lora.safetensors"))
    wrapper = _owned_reload_wrapper()

    loaded_count = _load_fused_expert_lora([wrapper], str(tmp_path))

    assert loaded_count == 6
    for name, param in zip(_FUSED_LORA_NAMES, wrapper._fused_expert_lora_params):
        torch.testing.assert_close(param, tensors[f"layers.0.experts.{name}"])
    assert wrapper._lora_pointers_dirty is True


def test_force_fused_lora_reload_rejects_missing_file(tmp_path):
    model = SimpleNamespace(_kt_wrappers=[_owned_reload_wrapper()])

    with pytest.raises(FileNotFoundError, match="fused_expert_lora.safetensors"):
        load_kt_moe_from_adapter(model, str(tmp_path))


def test_force_fused_lora_reload_rejects_incomplete_key_set(tmp_path):
    from safetensors.torch import save_file

    tensors = _saved_fused_tensors()
    tensors.pop("layers.0.experts.down_lora_b")
    save_file(tensors, str(tmp_path / "fused_expert_lora.safetensors"))
    model = SimpleNamespace(_kt_wrappers=[_owned_reload_wrapper()])

    with pytest.raises(RuntimeError, match="missing=.*down_lora_b"):
        load_kt_moe_from_adapter(model, str(tmp_path))


@pytest.mark.parametrize(
    "unexpected_key",
    [
        "layers.9.experts.gate_lora_a",
        "layers.0.experts.unknown_projection",
    ],
)
def test_force_fused_lora_reload_rejects_unknown_keys(tmp_path, unexpected_key):
    from safetensors.torch import save_file

    tensors = _saved_fused_tensors()
    tensors[unexpected_key] = torch.zeros(2, 2, dtype=torch.bfloat16)
    save_file(tensors, str(tmp_path / "fused_expert_lora.safetensors"))
    model = SimpleNamespace(_kt_wrappers=[_owned_reload_wrapper()])

    with pytest.raises(RuntimeError, match="unexpected="):
        load_kt_moe_from_adapter(model, str(tmp_path))


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        (torch.zeros(3, 2, dtype=torch.bfloat16), "shape mismatch"),
        (torch.zeros(2, 2, dtype=torch.float32), "dtype mismatch"),
    ],
)
def test_force_fused_lora_reload_rejects_tensor_mismatch(
    tmp_path, replacement, message
):
    from safetensors.torch import save_file

    tensors = _saved_fused_tensors()
    tensors["layers.0.experts.gate_lora_a"] = replacement
    save_file(tensors, str(tmp_path / "fused_expert_lora.safetensors"))
    model = SimpleNamespace(_kt_wrappers=[_owned_reload_wrapper()])

    with pytest.raises(RuntimeError, match=message):
        load_kt_moe_from_adapter(model, str(tmp_path))


def test_force_fused_lora_reload_rejects_duplicate_owned_layer(tmp_path):
    model = SimpleNamespace(
        _kt_wrappers=[_owned_reload_wrapper(), _owned_reload_wrapper()]
    )

    with pytest.raises(RuntimeError, match="Duplicate owned.*layer 0"):
        load_kt_moe_from_adapter(model, str(tmp_path))
