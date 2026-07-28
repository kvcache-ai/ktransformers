# SPDX-License-Identifier: Apache-2.0

import sys
from types import SimpleNamespace

import pytest

from kt_kernel.sft import backend


def _install_fake_runtime(
    monkeypatch,
    *,
    variant: str,
    kernel: str,
    layout: str = backend.INT8_WEIGHT_LAYOUT,
    has_int8_sft: bool = True,
):
    moe = SimpleNamespace()
    if has_int8_sft:
        moe.AMXInt8_SFT_MOE = object()
    extension = SimpleNamespace(
        __cpu_variant__=variant,
        __int8_kernel__=kernel,
        __int8_weight_layout__=layout,
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
