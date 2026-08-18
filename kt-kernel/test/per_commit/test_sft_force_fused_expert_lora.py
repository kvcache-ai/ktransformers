# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from kt_kernel.sft.arch import KTAMXConfigError
from kt_kernel.sft.config import KTConfig
from kt_kernel.sft.lora import (
    _load_fused_expert_lora,
    _save_fused_expert_lora,
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
    import kt_kernel.sft.weight_manifest as manifest_module

    monkeypatch.setattr(wrapper_module, "KT_KERNEL_AVAILABLE", True)
    monkeypatch.setattr(wrapper_module, "KTMoEWrapper", _Backend)
    monkeypatch.setattr(
        wrapper_module,
        "get_int8_runtime",
        lambda: SimpleNamespace(
            cpu_variant="avx512_bf16",
            kernel="avx512-vnni",
            weight_layout="kt-int8-n32-k64-vnni-v1",
        ),
    )
    monkeypatch.setattr(
        manifest_module,
        "validate_persistent_int8_weights",
        lambda *_args, **_kwargs: SimpleNamespace(
            path="/persistent/kt-int8/kt-weight-manifest.json",
            schema_version=2,
            is_legacy=False,
            layout="kt-int8-n32-k64-vnni-v1",
            layer_indices=(0,),
            file_count=12,
            size_bytes=12,
        ),
    )
    return wrap_moe_layers_with_kt_wrapper(model, config)[0]


def _assert_int8_expert_placeholders_are_empty(experts):
    expected_shapes = {
        "gate_proj": (3, 4),
        "up_proj": (3, 4),
        "down_proj": (4, 3),
    }
    for expert in experts:
        for name, expected_shape in expected_shapes.items():
            parameter = getattr(expert, name).weight
            assert parameter.numel() == 0
            assert parameter.device.type == "cpu"
            assert parameter._kt_zero_storage is True
            assert parameter._kt_original_shape == expected_shape


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
    _assert_int8_expert_placeholders_are_empty(wrapper.experts)

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
    import kt_kernel.sft.weight_manifest as manifest_module

    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    with (
        patch("kt_kernel.sft.wrapper.KT_KERNEL_AVAILABLE", True),
        patch("kt_kernel.sft.wrapper.KTMoEWrapper", _Backend),
        patch(
            "kt_kernel.sft.wrapper.get_int8_runtime",
            return_value=SimpleNamespace(
                cpu_variant="avx512_bf16",
                kernel="avx512-vnni",
                weight_layout="kt-int8-n32-k64-vnni-v1",
            ),
        ),
        patch.object(
            manifest_module,
            "validate_persistent_int8_weights",
            return_value=SimpleNamespace(
                path="/persistent/kt-int8/kt-weight-manifest.json",
                schema_version=2,
                is_legacy=False,
                layout="kt-int8-n32-k64-vnni-v1",
                layer_indices=(0,),
                file_count=12,
                size_bytes=12,
            ),
        ),
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
    _assert_int8_expert_placeholders_are_empty(wrapper.experts)


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


def test_force_fused_lora_save_publishes_atomically(tmp_path, monkeypatch):
    from safetensors.torch import load_file

    wrapper = _owned_reload_wrapper()
    replacements = []
    real_replace = os.replace

    def record_replace(source, destination):
        replacements.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr("kt_kernel.sft.lora.os.replace", record_replace)
    _save_fused_expert_lora([wrapper], str(tmp_path))

    output_path = tmp_path / "fused_expert_lora.safetensors"
    assert len(replacements) == 1
    temporary_path, published_path = replacements[0]
    assert os.path.dirname(temporary_path) == str(tmp_path)
    assert published_path == str(output_path)
    assert not os.path.exists(temporary_path)
    saved = load_file(str(output_path))
    assert set(saved) == set(_saved_fused_tensors())


def test_force_fused_lora_save_failure_keeps_published_file(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    output_path = tmp_path / "fused_expert_lora.safetensors"
    original = {"existing": torch.ones(1)}
    save_file(original, str(output_path))
    original_bytes = output_path.read_bytes()

    def fail_save(_tensors, path):
        with open(path, "wb") as handle:
            handle.write(b"partial")
        raise RuntimeError("injected save failure")

    monkeypatch.setattr("safetensors.torch.save_file", fail_save)
    with pytest.raises(RuntimeError, match="injected save failure"):
        _save_fused_expert_lora([_owned_reload_wrapper()], str(tmp_path))

    assert output_path.read_bytes() == original_bytes
    assert list(tmp_path.glob(".fused_expert_lora.*.safetensors.tmp")) == []


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
