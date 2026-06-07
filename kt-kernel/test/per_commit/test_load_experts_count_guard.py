"""Regression tests for SafeTensorLoader.load_experts expert-count guard.

After the discovery loop, ``max_experts_count`` holds the highest expert index
found, i.e. ``(expert count - 1)``. It is ``-1`` only when no experts exist for
the key, so the guard must reject ``-1`` (no experts), not ``0`` (exactly one
expert). The previous ``== 0`` check was an off-by-one that falsely raised
"No experts found" for a single-expert layer and silently returned empty weight
lists for the genuine zero-expert case.

The loader module is imported as a top-level module (its directory is added to
sys.path) so the test does not pull in the compiled kt_kernel_ext extension via
utils/__init__.py.
"""

import os
import sys
import unittest

# Register this test for CPU CI.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="default")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python", "utils"))
import loader

SafeTensorLoader = loader.SafeTensorLoader


class _FakeTensor:
    """Stand-in for a loaded tensor; load_experts only calls ``.numpy()``."""

    def numpy(self):
        return 0


class _StubSafeTensorLoader(SafeTensorLoader):
    """SafeTensorLoader backed by an explicit key set instead of real files."""

    def __init__(self, keys):
        # Bypass the filesystem/safetensors scan in the real __init__.
        self.tensor_file_map = {k: "stub.safetensors" for k in keys}
        self.file_handle_map = {}
        self.tensor_type_map = {}
        self.tensor_device_map = {}

    def load_tensor(self, key: str, device: str = "cpu"):
        assert key in self.tensor_file_map, f"unexpected key requested: {key}"
        return _FakeTensor()


def _expert_keys(base_key, n_experts, n_numa=1):
    """Build the NUMA-sharded expert key set that load_experts expects."""
    keys = []
    for proj in ("ffn_up_exps", "ffn_gate_exps", "ffn_down_exps"):
        for expert in range(n_experts):
            for numa in range(n_numa):
                prefix = f"{base_key}.{proj}.{expert}.numa.{numa}"
                keys.append(f"{prefix}.weight")
                keys.append(f"{prefix}.scale")
    return keys


class TestLoadExpertsCountGuard(unittest.TestCase):
    def test_single_expert_is_loaded(self):
        """A layer with exactly one expert must load, not raise."""
        loader = _StubSafeTensorLoader(_expert_keys("blk.0", n_experts=1))
        result = loader.load_experts("blk.0")
        for proj in ("up", "gate", "down"):
            self.assertEqual(len(result[proj]), 1, f"{proj}: expected 1 numa group")
            self.assertEqual(len(result[proj][0]), 1, f"{proj}: expected 1 expert")

    def test_zero_experts_raises(self):
        """No experts under the key must raise, not silently return empties."""
        loader = _StubSafeTensorLoader(["blk.0.attn_norm.weight"])
        with self.assertRaises(ValueError):
            loader.load_experts("blk.0")

    def test_multiple_experts_and_numa_counts(self):
        """Counts are correct across several experts and NUMA shards."""
        loader = _StubSafeTensorLoader(_expert_keys("blk.0", n_experts=3, n_numa=2))
        result = loader.load_experts("blk.0")
        for proj in ("up", "gate", "down"):
            self.assertEqual(len(result[proj]), 2, f"{proj}: expected 2 numa groups")
            for numa_group in result[proj]:
                self.assertEqual(len(numa_group), 3, f"{proj}: expected 3 experts")


if __name__ == "__main__":
    unittest.main()
