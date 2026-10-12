"""Unit tests for CPU feature detection (kt_kernel.python._cpu_detect).

These exercise the pure-Python variant-selection logic without a compiled
kt_kernel_ext or any particular host hardware, so they run on every commit.
"""

import importlib.util
import io
import os
import sys

import pytest

# Register with the CPU CI suite, matching the sibling tests in this folder.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="default")


def _load_cpu_detect():
    """Import _cpu_detect.py by path so we don't pull in the compiled ext."""
    module_path = os.path.join(os.path.dirname(__file__), "..", "..", "python", "_cpu_detect.py")
    spec = importlib.util.spec_from_file_location("kt_cpu_detect_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cpu_detect = _load_cpu_detect()


def _fake_cpuinfo(flags):
    """Return an open() replacement that yields a /proc/cpuinfo with these flags."""
    text = "processor\t: 0\nflags\t\t: " + " ".join(flags) + "\n"

    def _open(*args, **kwargs):
        return io.StringIO(text)

    return _open


AMX_FLAGS = [
    "avx2", "avx512f", "avx512bw", "avx512_vnni",
    "avx512_vbmi", "avx512_bf16", "amx_tile", "amx_int8", "amx_bf16",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("KT_KERNEL_CPU_VARIANT", raising=False)
    monkeypatch.delenv("KT_KERNEL_DEBUG", raising=False)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "variant",
    ["amx", "avx512_bf16", "avx512_vbmi", "avx512_vnni", "avx512_base", "avx2"],
)
def test_env_override_wins(monkeypatch, variant):
    monkeypatch.setenv("KT_KERNEL_CPU_VARIANT", variant)
    assert cpu_detect.detect_cpu_features() == variant


@pytest.mark.cpu
def test_invalid_env_override_is_ignored(monkeypatch):
    monkeypatch.setenv("KT_KERNEL_CPU_VARIANT", "totally-bogus")
    monkeypatch.setattr("builtins.open", _fake_cpuinfo(["avx2"]))
    assert cpu_detect.detect_cpu_features() == "avx2"


@pytest.mark.cpu
def test_full_amx_flags_detected(monkeypatch):
    monkeypatch.setattr("builtins.open", _fake_cpuinfo(AMX_FLAGS))
    assert cpu_detect.detect_cpu_features() == "amx"


@pytest.mark.cpu
def test_progressive_match_stops_at_vnni(monkeypatch):
    # avx512f/bw/vnni present but no vbmi -> best match is avx512_vnni.
    monkeypatch.setattr(
        "builtins.open",
        _fake_cpuinfo(["avx2", "avx512f", "avx512bw", "avx512_vnni"]),
    )
    assert cpu_detect.detect_cpu_features() == "avx512_vnni"


@pytest.mark.cpu
def test_flag_name_without_underscore_matches(monkeypatch):
    # Some kernels print flags without underscores (e.g. 'avx512bf16').
    flags = [
        "avx2", "avx512f", "avx512bw", "avx512vnni",
        "avx512vbmi", "avx512bf16", "amx_tile", "amx_int8", "amx_bf16",
    ]
    monkeypatch.setattr("builtins.open", _fake_cpuinfo(flags))
    assert cpu_detect.detect_cpu_features() == "amx"


@pytest.mark.cpu
def test_avx2_only_falls_back(monkeypatch):
    monkeypatch.setattr("builtins.open", _fake_cpuinfo(["avx2", "sse2"]))
    assert cpu_detect.detect_cpu_features() == "avx2"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
