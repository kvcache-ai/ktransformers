"""Pure-Python tests for MESH residency helper functions."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "python" / "utils" / "weight_provider.py"
SPEC = importlib.util.spec_from_file_location("weight_provider", MODULE_PATH)
weight_provider = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(weight_provider)


def test_cgroup_v2_memory_limit_is_detected_from_process_scope(tmp_path):
    scoped = tmp_path / "system.slice" / "run-test.scope"
    scoped.mkdir(parents=True)
    (scoped / "memory.max").write_text(str(64 * 1024**3))
    (scoped / "memory.current").write_text(str(12 * 1024**3))

    limit, current = weight_provider.get_cgroup_memory_limit_current_bytes(
        proc_self_cgroup_text="0::/system.slice/run-test.scope\n",
        mountinfo_text="29 23 0:28 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n",
        mount_root_override=tmp_path,
    )

    assert limit == 64 * 1024**3
    assert current == 12 * 1024**3


def test_ram_queries_prefer_cgroup_over_host_memory(monkeypatch):
    monkeypatch.setattr(
        weight_provider,
        "get_cgroup_memory_limit_current_bytes",
        lambda **_: (64 * 1024**3, 12 * 1024**3),
    )

    assert weight_provider.get_total_ram_bytes() == 64 * 1024**3
    assert weight_provider.get_available_ram_bytes() == 52 * 1024**3


def test_compute_max_tier0_experts_clamps_to_expert_count():
    max_experts = weight_provider.compute_max_tier0_experts(
        tier0_memory_bytes=10**18,
        num_layers=40,
        num_experts=256,
        hidden_size=7168,
        intermediate_size=2048,
        bytes_per_element=0.5,
    )

    assert max_experts == 256


def test_method_bytes_per_element_and_policy_validation():
    assert weight_provider.method_bytes_per_element("AMXINT4") == 0.5
    assert weight_provider.method_bytes_per_element("BF16") == 2.0
    assert weight_provider.normalize_residency_policy_name("current_ema") == "baseline"

    with pytest.raises(ValueError, match="Unsupported residency policy"):
        weight_provider.normalize_residency_policy_name("not-a-policy")
