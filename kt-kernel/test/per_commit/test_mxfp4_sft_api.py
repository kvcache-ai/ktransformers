#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Source-level contracts for the standalone MXFP4 SFT entry point."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default")

PYTHON_ROOT = Path(__file__).resolve().parents[2] / "python"
SFT_ROOT = PYTHON_ROOT / "sft"


def _load_source(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


backend = _load_source("kt_mxfp4_backend_under_test", SFT_ROOT / "backend.py")
loader = _load_source("kt_mxfp4_loader_under_test", PYTHON_ROOT / "utils" / "loader.py")


def _install_fake_runtime(monkeypatch, *, variant="avx512_bf16", has_symbol=True):
    moe = SimpleNamespace()
    if has_symbol:
        moe.MXFP4_SFT_MOE = object()
    extension = SimpleNamespace(__cpu_variant__=variant, moe=moe)
    monkeypatch.setitem(
        sys.modules,
        "kt_kernel",
        SimpleNamespace(kt_kernel_ext=extension, __cpu_variant__=variant),
    )


def test_mxfp4_runtime_contract(monkeypatch):
    for variant in ("avx512_bf16", "amx"):
        _install_fake_runtime(monkeypatch, variant=variant)
        runtime = backend.get_mxfp4_runtime()
        assert runtime.cpu_variant == variant
        assert runtime.kernel == backend.MXFP4_KERNEL
        assert runtime.weight_layout == backend.MXFP4_WEIGHT_LAYOUT


def test_mxfp4_runtime_rejects_wrong_isa_and_stale_extension(monkeypatch):
    _install_fake_runtime(monkeypatch, variant="avx2")
    with pytest.raises(RuntimeError, match="requires an AVX512-BF16"):
        backend.get_mxfp4_runtime()

    _install_fake_runtime(monkeypatch, has_symbol=False)
    with pytest.raises(RuntimeError, match="does not provide MXFP4 SFT"):
        backend.get_mxfp4_runtime()


def _load_amx_module(monkeypatch):
    package_name = "kt_mxfp4_amx_under_test"
    package = ModuleType(package_name)
    package.__path__ = [str(PYTHON_ROOT)]
    sft_package = ModuleType(f"{package_name}.sft")
    sft_package.__path__ = [str(SFT_ROOT)]
    utils_package = ModuleType(f"{package_name}.utils")
    utils_package.__path__ = [str(PYTHON_ROOT / "utils")]
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, sft_package.__name__, sft_package)
    monkeypatch.setitem(sys.modules, utils_package.__name__, utils_package)

    extension_package = ModuleType("kt_kernel_ext")
    extension_package.__path__ = []
    extension_moe = ModuleType("kt_kernel_ext.moe")

    class _FakeMOESFTConfig:
        pass

    class _FakeMoe:
        pass

    extension_moe.MOESFTConfig = _FakeMOESFTConfig
    for name in (
        "AMXBF16_SFT_MOE",
        "AMXInt8_SFT_MOE",
        "AMXInt4_SFT_MOE",
        "AMXBF16_SFT_MOE_SkipLoRA",
        "AMXInt8_SFT_MOE_SkipLoRA",
        "AMXInt4_SFT_MOE_SkipLoRA",
        "AMXFP8_SFT_MOE",
        "MXFP4_SFT_MOE",
    ):
        setattr(extension_moe, name, _FakeMoe)
    extension_package.moe = extension_moe
    monkeypatch.setitem(sys.modules, "kt_kernel_ext", extension_package)
    monkeypatch.setitem(sys.modules, "kt_kernel_ext.moe", extension_moe)

    loader_stub = ModuleType(f"{package_name}.utils.loader")
    loader_stub.BF16SafeTensorLoader = object
    loader_stub.MXFP4SafeTensorLoader = object
    loader_stub.SafeTensorLoader = object
    monkeypatch.setitem(sys.modules, loader_stub.__name__, loader_stub)

    class _FakeBaseSFTMoEWrapper:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self._full_weight_grad = bool(kwargs["full_weight_grad"])
            self._weights_loaded = False
            self._lora_initialized = False
            self._authoritative_optimizer_grads = []
            self.share_backward_bb = False
            self.share_cache_pool = False

        def register_authoritative_optimizer_grad(self, name, parameter, grad_view):
            parameter.grad = None
            self._authoritative_optimizer_grads.append(
                SimpleNamespace(name=name, parameter=parameter, grad_view=grad_view)
            )

    base_stub = ModuleType(f"{package_name}.sft.base")
    base_stub.BaseSFTMoEWrapper = _FakeBaseSFTMoEWrapper
    base_stub.KExpertsSFTBuffer = object
    base_stub._supports_authoritative_optimizer_grads = lambda *args, **kwargs: True
    monkeypatch.setitem(sys.modules, base_stub.__name__, base_stub)

    backend_name = f"{package_name}.sft.backend"
    amx_backend = _load_source(backend_name, SFT_ROOT / "backend.py")
    monkeypatch.setitem(sys.modules, backend_name, amx_backend)
    weights_stub = ModuleType(f"{package_name}.sft.weights")
    weights_stub.BlockFP8ExpertWeights = object
    monkeypatch.setitem(sys.modules, weights_stub.__name__, weights_stub)

    module = _load_source(f"{package_name}.sft.amx", SFT_ROOT / "amx.py")
    module.get_mxfp4_runtime = lambda: SimpleNamespace(cpu_variant="avx512_bf16")
    return module


def _construct_mxfp4(amx, **overrides):
    values = {
        "layer_idx": 0,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "hidden_size": 64,
        "moe_intermediate_size": 64,
        "num_gpu_experts": 0,
        "cpuinfer_threads": 4,
        "threadpool_count": 1,
        "weight_path": "/unused",
        "chunked_prefill_size": 32,
        "lora_rank": 8,
        "lora_alpha": 16.0,
        "lora_dropout": 0.0,
        "max_cache_depth": 2,
        "method": "MXFP4_SFT",
        "group_size": 32,
        "zero_point": False,
        "full_weight_grad": False,
        "swiglu_limit": 10.0,
    }
    values.update(overrides)
    return amx.AMXSFTMoEWrapper(**values)


def test_mxfp4_wrapper_accepts_only_frozen_cpu_group32_lora(monkeypatch):
    amx = _load_amx_module(monkeypatch)
    wrapper = _construct_mxfp4(amx)
    assert wrapper.method == "MXFP4_SFT"
    assert wrapper.group_size == 32
    assert wrapper.zero_point is False
    assert wrapper.swiglu_limit == 10.0
    assert wrapper._uses_authoritative_optimizer_grads

    invalid = (
        ({"full_weight_grad": True}, "frozen-base LoRA only"),
        ({"lora_rank": 0}, "frozen-base LoRA only"),
        ({"num_gpu_experts": 1}, "all routed experts on CPU"),
        ({"group_size": 128}, "group_size=32"),
        ({"zero_point": True}, "zero_point=False"),
        ({"hidden_size": 65}, "divisible by 32"),
        ({"threadpool_count": 4}, "threadpool_count 1 or 2"),
        ({"moe_intermediate_size": 96, "threadpool_count": 2}, "TP intermediate slice"),
        ({"swiglu_limit": 0.0}, "finite positive swiglu_limit"),
        ({"swiglu_limit": float("nan")}, "finite positive swiglu_limit"),
    )
    for overrides, message in invalid:
        with pytest.raises(ValueError, match=message):
            _construct_mxfp4(amx, **overrides)


def _valid_staged_weights(experts=2, hidden=64, intermediate=64):
    return {
        "gate": [
            torch.zeros((intermediate, hidden // 2), dtype=torch.uint8)
            for _ in range(experts)
        ],
        "up": [
            torch.zeros((intermediate, hidden // 2), dtype=torch.uint8)
            for _ in range(experts)
        ],
        "down": [
            torch.zeros((hidden, intermediate // 2), dtype=torch.uint8)
            for _ in range(experts)
        ],
        "gate_scale": [
            torch.ones((intermediate, hidden // 32), dtype=torch.bfloat16)
            for _ in range(experts)
        ],
        "up_scale": [
            torch.ones((intermediate, hidden // 32), dtype=torch.bfloat16)
            for _ in range(experts)
        ],
        "down_scale": [
            torch.ones((hidden, intermediate // 32), dtype=torch.bfloat16)
            for _ in range(experts)
        ],
    }


def _valid_lora_tensors(*, parameters):
    shapes = (
        (2, 8, 64),
        (2, 64, 8),
        (2, 8, 64),
        (2, 64, 8),
        (2, 8, 64),
        (2, 64, 8),
    )
    weights = [torch.zeros(shape, dtype=torch.bfloat16) for shape in shapes]
    if parameters:
        weights = [torch.nn.Parameter(weight) for weight in weights]
    grads = [torch.empty_like(weight) for weight in weights]
    return weights, grads


def test_mxfp4_lora_parameters_register_authoritative_grad_views(monkeypatch):
    amx = _load_amx_module(monkeypatch)
    wrapper = _construct_mxfp4(amx)
    weights, grads = _valid_lora_tensors(parameters=True)

    wrapper.init_lora_weights(*weights, *grads)

    assert len(wrapper._authoritative_optimizer_grads) == 6
    for entry, parameter, grad_view in zip(
        wrapper._authoritative_optimizer_grads, weights, grads, strict=True
    ):
        assert entry.parameter is parameter
        assert entry.grad_view is grad_view
        assert parameter.grad is None


def test_mxfp4_lora_ownership_rejects_mixed_parameter_and_tensor(monkeypatch):
    amx = _load_amx_module(monkeypatch)
    wrapper = _construct_mxfp4(amx)
    weights, grads = _valid_lora_tensors(parameters=True)
    weights[0] = weights[0].detach()

    with pytest.raises(ValueError, match="all six weights"):
        wrapper.init_lora_weights(*weights, *grads)


def test_mxfp4_second_load_fails_before_retain_or_native_dispatch(monkeypatch):
    amx = _load_amx_module(monkeypatch)
    wrapper = _construct_mxfp4(amx)
    wrapper._weights_loaded = True

    with pytest.raises(RuntimeError, match="immutable|already loaded"):
        wrapper.load_mxfp4_weights(
            _valid_staged_weights(), torch.arange(2, dtype=torch.int64)
        )
    assert not hasattr(wrapper, "_gate_weights_per_numa")


def test_mxfp4_expert_map_must_be_a_permutation(monkeypatch):
    amx = _load_amx_module(monkeypatch)
    wrapper = _construct_mxfp4(amx)
    wrapper._stage_mxfp4_weights(_valid_staged_weights())
    wrapper.cpu_infer = SimpleNamespace(backend_=object())

    with pytest.raises(ValueError, match="must be a permutation"):
        wrapper.load_weights(torch.tensor([0, 0], dtype=torch.int64))


def test_mxfp4_weight_staging_preserves_exact_tensor_storage(monkeypatch):
    amx = _load_amx_module(monkeypatch)
    wrapper = object.__new__(amx.AMXSFTMoEWrapper)
    wrapper.method = "MXFP4_SFT"
    wrapper.num_experts = 2
    wrapper.hidden_size = 64
    wrapper.moe_intermediate_size = 64
    staged = _valid_staged_weights()
    pointers_before = {
        name: [tensor.data_ptr() for tensor in tensors]
        for name, tensors in staged.items()
    }

    wrapper._stage_mxfp4_weights(staged)

    assert wrapper._use_projs_path
    assert wrapper._has_bwd_projs is False
    assert (
        wrapper.gate_proj is None
        and wrapper.up_proj is None
        and wrapper.down_proj is None
    )
    assert wrapper._gate_projs_ptrs == [pointers_before["gate"]]
    assert wrapper._up_projs_ptrs == [pointers_before["up"]]
    assert wrapper._down_projs_ptrs == [pointers_before["down"]]
    assert wrapper._gate_scale_ptrs == [pointers_before["gate_scale"]]
    assert wrapper._up_scale_ptrs == [pointers_before["up_scale"]]
    assert wrapper._down_scale_ptrs == [pointers_before["down_scale"]]


def test_mxfp4_weight_staging_rejects_bad_shape_dtype_stride_and_scale(monkeypatch):
    amx = _load_amx_module(monkeypatch)
    wrapper = object.__new__(amx.AMXSFTMoEWrapper)
    wrapper.method = "MXFP4_SFT"
    wrapper.num_experts = 2
    wrapper.hidden_size = 64
    wrapper.moe_intermediate_size = 64

    bad = _valid_staged_weights()
    bad["gate"][0] = bad["gate"][0].to(torch.int8)
    with pytest.raises(ValueError, match="contiguous CPU torch.uint8"):
        wrapper._stage_mxfp4_weights(bad)

    bad = _valid_staged_weights()
    bad["up"][0] = torch.zeros((64, 33), dtype=torch.uint8)
    with pytest.raises(ValueError, match="with shape"):
        wrapper._stage_mxfp4_weights(bad)

    bad = _valid_staged_weights()
    bad["down"][0] = torch.zeros((32, 64), dtype=torch.uint8).t()
    assert not bad["down"][0].is_contiguous()
    with pytest.raises(ValueError, match="contiguous CPU"):
        wrapper._stage_mxfp4_weights(bad)

    bad = _valid_staged_weights()
    bad["gate_scale"][0][0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite scale"):
        wrapper._stage_mxfp4_weights(bad)


def test_sft_loader_rejects_non_native_or_nonfinite_ue8m0_scales():
    converter = loader.MXFP4SafeTensorLoader._ue8m0_to_bf16
    encoded = torch.tensor([0, 120, 127, 130, 254], dtype=torch.uint8)
    converted = converter(encoded, reject_non_finite=True)
    expected_bits = (encoded.to(torch.int32) << 7).to(torch.int16)
    expected_bits[0] = 0x0040
    assert torch.equal(
        converted.view(torch.int16),
        expected_bits,
    )
    with pytest.raises(ValueError, match="reserved UE8M0 scale 0xff"):
        converter(torch.tensor([255], dtype=torch.uint8), reject_non_finite=True)
    with pytest.raises(TypeError, match="raw uint8"):
        converter(torch.ones(2, dtype=torch.bfloat16), reject_non_finite=True)


def _load_experts_factory(monkeypatch):
    package_name = "kt_mxfp4_factory_under_test"
    package = ModuleType(package_name)
    package.__path__ = [str(PYTHON_ROOT)]
    utils_package = ModuleType(f"{package_name}.utils")
    utils_package.__path__ = [str(PYTHON_ROOT / "utils")]
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, utils_package.__name__, utils_package)

    class _FakeBase:
        pass

    modules = {
        f"{package_name}.experts_base": {"BaseMoEWrapper": _FakeBase},
        f"{package_name}.utils.amx": {
            "AMXMoEWrapper": _FakeBase,
            "NativeMoEWrapper": _FakeBase,
        },
        f"{package_name}.utils.llamafile": {"LlamafileMoEWrapper": _FakeBase},
        f"{package_name}.utils.moe_kernel": {"GeneralMoEWrapper": _FakeBase},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        monkeypatch.setitem(sys.modules, name, module)
    return _load_source(f"{package_name}.experts", PYTHON_ROOT / "experts.py")


def test_direct_ktmoewrapper_forwards_the_mxfp4_sft_contract(monkeypatch):
    experts = _load_experts_factory(monkeypatch)
    captured = {}
    monkeypatch.setattr(
        experts,
        "_create_sft_wrapper",
        lambda **kwargs: captured.update(kwargs) or SimpleNamespace(**kwargs),
    )
    wrapper = experts.KTMoEWrapper(
        layer_idx=1,
        num_experts=256,
        num_experts_per_tok=6,
        hidden_size=4096,
        moe_intermediate_size=2048,
        gpu_experts_mask=None,
        cpuinfer_threads=64,
        threadpool_count=2,
        weight_path="/models/DeepSeek-V4-Flash-0731",
        chunked_prefill_size=128,
        mode="sft",
        method="MXFP4_SFT",
        num_gpu_experts=0,
        lora_rank=8,
        lora_alpha=16.0,
        group_size=32,
        zero_point=False,
        full_weight_grad=False,
        swiglu_limit=10.0,
    )

    assert wrapper.method == "MXFP4_SFT"
    assert captured["group_size"] == 32
    assert captured["zero_point"] is False
    assert captured["full_weight_grad"] is False
    assert captured["swiglu_limit"] == 10.0
    assert "MXFP4_SFT" in experts.SFT_METHODS
    with pytest.raises(ValueError, match="requires swiglu_alpha=0"):
        experts.KTMoEWrapper(
            layer_idx=1,
            num_experts=256,
            num_experts_per_tok=6,
            hidden_size=4096,
            moe_intermediate_size=2048,
            gpu_experts_mask=None,
            cpuinfer_threads=64,
            threadpool_count=2,
            weight_path="/models/DeepSeek-V4-Flash-0731",
            chunked_prefill_size=128,
            mode="sft",
            method="MXFP4_SFT",
            group_size=32,
            zero_point=False,
            swiglu_limit=10.0,
            swiglu_alpha=1.0,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
