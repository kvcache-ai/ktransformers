# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

SFT_PATH = Path(__file__).resolve().parents[2] / "python" / "sft"
PACKAGE_NAME = "kt_sft_backend_under_test"
package = ModuleType(PACKAGE_NAME)
package.__path__ = [str(SFT_PATH)]
sys.modules[PACKAGE_NAME] = package
spec = importlib.util.spec_from_file_location(
    f"{PACKAGE_NAME}.backend",
    SFT_PATH / "backend.py",
)
assert spec is not None and spec.loader is not None
backend = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = backend
spec.loader.exec_module(backend)


def _install_fake_runtime(
    monkeypatch,
    *,
    variant: str,
    kernel: str,
    layout: str = backend.INT8_WEIGHT_LAYOUT,
    has_int8_sft: bool = True,
    has_fp8_sft: bool = False,
    fp8_kernel: str = backend.FP8_KERNEL,
    fp8_layout: str = backend.FP8_WEIGHT_LAYOUT,
):
    moe = SimpleNamespace()
    if has_int8_sft:
        moe.AMXInt8_SFT_MOE = object()
    if has_fp8_sft:
        moe.AMXFP8_SFT_MOE = object()
    extension = SimpleNamespace(
        __cpu_variant__=variant,
        __int8_kernel__=kernel,
        __int8_weight_layout__=layout,
        __fp8_kernel__=fp8_kernel,
        __fp8_weight_layout__=fp8_layout,
        moe=moe,
    )
    fake_package = SimpleNamespace(
        kt_kernel_ext=extension,
        __cpu_variant__=variant,
    )
    monkeypatch.setitem(sys.modules, "kt_kernel", fake_package)


@pytest.mark.parametrize(
    ("variant", "kernel"),
    [
        ("amx", "amx-int8"),
        ("avx512_bf16", "avx512-vnni"),
        ("avx512_bf16", "onednn-vnni"),
    ],
)
def test_int8_runtime_reports_effective_kernel(monkeypatch, variant, kernel):
    _install_fake_runtime(monkeypatch, variant=variant, kernel=kernel)

    runtime = backend.get_int8_runtime()

    assert runtime.cpu_variant == variant
    assert runtime.kernel == kernel
    assert runtime.weight_layout == backend.INT8_WEIGHT_LAYOUT


def test_int8_runtime_rejects_extension_without_sft_symbol(monkeypatch):
    _install_fake_runtime(
        monkeypatch,
        variant="avx512_bf16",
        kernel="avx512-vnni",
        has_int8_sft=False,
    )
    with pytest.raises(RuntimeError, match="does not provide INT8 SFT"):
        backend.get_int8_runtime()


def test_int8_runtime_rejects_extension_without_runtime_metadata(monkeypatch):
    extension = SimpleNamespace(moe=SimpleNamespace(AMXInt8_SFT_MOE=object()))
    monkeypatch.setitem(
        sys.modules,
        "kt_kernel",
        SimpleNamespace(kt_kernel_ext=extension, __cpu_variant__="avx512_bf16"),
    )

    with pytest.raises(RuntimeError, match="predates production INT8 runtime metadata"):
        backend.get_int8_runtime()


def test_int8_runtime_rejects_unsupported_isa(monkeypatch):
    _install_fake_runtime(
        monkeypatch,
        variant="avx2",
        kernel="unsupported",
    )
    with pytest.raises(RuntimeError, match="requires an AMX-INT8 or AVX512"):
        backend.get_int8_runtime()


def test_int8_runtime_rejects_weight_layout_mismatch(monkeypatch):
    _install_fake_runtime(
        monkeypatch,
        variant="amx",
        kernel="amx-int8",
        layout="different-layout",
    )
    with pytest.raises(RuntimeError, match="weight-layout mismatch"):
        backend.get_int8_runtime()


def test_fp8_runtime_requires_native_sft_symbol(monkeypatch):
    _install_fake_runtime(
        monkeypatch,
        variant="amx",
        kernel="amx-int8",
        has_fp8_sft=False,
    )
    with pytest.raises(RuntimeError, match="does not provide native FP8 SFT"):
        backend.get_fp8_runtime()


def test_fp8_runtime_reports_native_layout(monkeypatch):
    _install_fake_runtime(
        monkeypatch,
        variant="amx",
        kernel="amx-int8",
        has_fp8_sft=True,
    )
    runtime = backend.get_fp8_runtime()
    assert runtime.cpu_variant == "amx"
    assert runtime.kernel == backend.FP8_KERNEL
    assert runtime.weight_layout == backend.FP8_WEIGHT_LAYOUT


def test_fp8_runtime_rejects_extension_without_runtime_metadata(monkeypatch):
    extension = SimpleNamespace(moe=SimpleNamespace(AMXFP8_SFT_MOE=object()))
    monkeypatch.setitem(
        sys.modules,
        "kt_kernel",
        SimpleNamespace(kt_kernel_ext=extension, __cpu_variant__="avx512_bf16"),
    )

    with pytest.raises(RuntimeError, match="predates native FP8 runtime metadata"):
        backend.get_fp8_runtime()


def test_fp8_runtime_rejects_unsupported_kernel(monkeypatch):
    _install_fake_runtime(
        monkeypatch,
        variant="avx2",
        kernel="unsupported",
        has_fp8_sft=True,
        fp8_kernel="unsupported",
    )
    with pytest.raises(RuntimeError, match="requires the AVX512-BF16\\+VBMI"):
        backend.get_fp8_runtime()


def test_fp8_runtime_rejects_weight_layout_mismatch(monkeypatch):
    _install_fake_runtime(
        monkeypatch,
        variant="avx512_bf16",
        kernel="avx512-vnni",
        has_fp8_sft=True,
        fp8_layout="different-layout",
    )
    with pytest.raises(RuntimeError, match="weight-layout mismatch"):
        backend.get_fp8_runtime()


@pytest.mark.parametrize("configured", ["auto", "FP8", "fp8"])
def test_fp8_backend_normalization(configured):
    assert backend.normalize_sft_backend(
        configured,
        expert_weight_format="fp8",
    ) == backend.FP8_BACKEND
