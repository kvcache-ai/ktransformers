# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

CPU_DETECT_PATH = Path(__file__).resolve().parents[2] / "python" / "_cpu_detect.py"
SPEC = importlib.util.spec_from_file_location(
    "kt_cpu_detect_metadata_under_test",
    CPU_DETECT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
cpu_detect = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cpu_detect
SPEC.loader.exec_module(cpu_detect)


def test_initialize_reports_loaded_extension_variant(monkeypatch):
    extension = SimpleNamespace(__cpu_variant__="avx512_bf16")
    monkeypatch.setattr(cpu_detect, "detect_cpu_features", lambda: "amx")
    monkeypatch.setattr(cpu_detect, "load_extension", lambda _variant: extension)

    loaded, variant = cpu_detect.initialize()

    assert loaded is extension
    assert variant == "avx512_bf16"


def test_initialize_rejects_extension_newer_than_host(monkeypatch):
    extension = SimpleNamespace(__cpu_variant__="amx")
    monkeypatch.setattr(cpu_detect, "detect_cpu_features", lambda: "avx512_bf16")
    monkeypatch.setattr(cpu_detect, "load_extension", lambda _variant: extension)

    try:
        cpu_detect.initialize()
    except RuntimeError as error:
        assert "requires a newer CPU ISA" in str(error)
    else:
        raise AssertionError("AMX extension must be rejected on an AVX512-BF16 host")
