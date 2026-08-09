"""Regression test for GeneralMoEWrapper.load_weights()'s cpu_save branch.

Asserts the branch actually consumes the gate/up/down weights
SafeTensorLoader.load_experts() returns, stacked into the per-expert layout
the native kernel expects.

Runs without the compiled kt_kernel_ext extension or real model weights: the
native symbols touched at construction time are stubbed and
SafeTensorLoader.load_experts() is mocked, matching the mocking style
test_native_moe_loader_auto_release.py already uses in this directory for
the same reason (no compiled binary in CI).
"""

import importlib.util
import os
import sys
import tempfile
import types
import unittest

import numpy as np
import torch
from safetensors.torch import save_file

PYTHON_DIR = os.path.join(os.path.dirname(__file__), "..", "python")


def _install_kt_kernel_ext_stub():
    """Registers a minimal stand-in for the compiled native extension."""
    moe_mod = types.ModuleType("kt_kernel_ext.moe")

    class MOEConfig:
        def __init__(self, *args, **kwargs):
            pass

    class _KernelMOEBase:
        def __init__(self, *args, **kwargs):
            pass

        def load_weights_task(self, *args, **kwargs):
            return None

    moe_mod.MOEConfig = MOEConfig
    moe_mod.Int8_KERNEL_MOE = type("Int8_KERNEL_MOE", (_KernelMOEBase,), {})
    moe_mod.Int4_KERNEL_MOE = type("Int4_KERNEL_MOE", (_KernelMOEBase,), {})

    kvcache_mod = types.ModuleType("kt_kernel_ext.kvcache")
    kvcache_mod.ggml_type = object()

    ext_mod = types.ModuleType("kt_kernel_ext")
    ext_mod.moe = moe_mod
    ext_mod.kvcache = kvcache_mod

    class WorkerPoolConfig:
        def __init__(self, *args, **kwargs):
            pass

    class CPUInfer:
        def __init__(self, *args, **kwargs):
            self.backend_ = object()

        def submit(self, *args, **kwargs):
            pass

        def sync(self, *args, **kwargs):
            pass

    ext_mod.WorkerPoolConfig = WorkerPoolConfig
    ext_mod.CPUInfer = CPUInfer

    sys.modules["kt_kernel_ext"] = ext_mod
    sys.modules["kt_kernel_ext.moe"] = moe_mod
    sys.modules["kt_kernel_ext.kvcache"] = kvcache_mod

    cpu_detect_mod = types.ModuleType("kt_kernel._cpu_detect")
    cpu_detect_mod.initialize = lambda: (ext_mod, "avx2")
    cpu_detect_mod.detect_cpu_features = lambda: "avx2"
    sys.modules["kt_kernel._cpu_detect"] = cpu_detect_mod


def _load_kt_kernel():
    """Loads the real kt_kernel package from source, by path, without installing it."""
    if "kt_kernel" in sys.modules:
        return sys.modules["kt_kernel"]
    _install_kt_kernel_ext_stub()
    spec = importlib.util.spec_from_file_location(
        "kt_kernel", os.path.join(PYTHON_DIR, "__init__.py"), submodule_search_locations=[PYTHON_DIR]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["kt_kernel"] = module
    spec.loader.exec_module(module)
    return module


class _FakeLoader:
    """Stand-in for SafeTensorLoader, returning load_experts()'s documented shape."""

    def __init__(self, values):
        self._values = values

    def load_experts(self, base_key, device="cpu"):
        return self._values


class TestGeneralMoEWrapperCpuSaveWeightKeys(unittest.TestCase):
    def setUp(self):
        kt_kernel = _load_kt_kernel()
        self.GeneralMoEWrapper = kt_kernel.utils.moe_kernel.GeneralMoEWrapper

        # experts_base.py pins gpu_experts_mask with pin_memory=True, which
        # needs a CUDA/accelerator backend; CI here is CPU-only torch.
        self._orig_empty, self._orig_zeros = torch.empty, torch.zeros
        torch.empty = lambda *a, **kw: self._orig_empty(*a, **{k: v for k, v in kw.items() if k != "pin_memory"})
        torch.zeros = lambda *a, **kw: self._orig_zeros(*a, **{k: v for k, v in kw.items() if k != "pin_memory"})

        self.tmpdir = tempfile.TemporaryDirectory()
        # One real, minimal safetensors file so load_merged_weight detection
        # and SafeTensorLoader.__init__ (invoked by GeneralMoEWrapper.__init__)
        # succeed; its content is unused once _FakeLoader replaces the loader.
        save_file({"placeholder": torch.zeros(1)}, os.path.join(self.tmpdir.name, "placeholder.safetensors"))

    def tearDown(self):
        torch.empty, torch.zeros = self._orig_empty, self._orig_zeros
        self.tmpdir.cleanup()
        self.GeneralMoEWrapper._safetensor_loader_instance = None

    def _make_wrapper(self, num_experts=2):
        wrapper = self.GeneralMoEWrapper(
            layer_idx=0,
            num_experts=num_experts,
            num_experts_per_tok=1,
            hidden_size=4,
            moe_intermediate_size=2,
            gpu_experts_mask=torch.zeros(num_experts, dtype=torch.bool),
            cpuinfer_threads=1,
            threadpool_count=1,
            weight_path=self.tmpdir.name,
            chunked_prefill_size=1,
            cpu_save=True,
            max_deferred_experts_per_token=None,
            method="MOE_INT4",
        )
        self.assertTrue(wrapper.load_merged_weight)
        # distinct values per (proj, expert) so a wrong expert or a dropped
        # expert shows up as a value mismatch, not just a shape mismatch.
        # *_scale keys are required too: the load_merged_weight branch above
        # the one under test reads them unconditionally.
        loaded = {
            proj: [[np.full((2, 4), base + expert_id * 10, dtype=np.float32) for expert_id in range(num_experts)]]
            for proj, base in (("gate", 100.0), ("up", 0.0), ("down", 200.0))
        }
        for proj in ("gate", "up", "down"):
            loaded[f"{proj}_scale"] = [[np.ones(1, dtype=np.float32) for _ in range(num_experts)]]
        wrapper.safetensor_loader = _FakeLoader(loaded)
        return wrapper

    def test_cpu_save_loads_correct_per_expert_weights(self):
        num_experts = 2
        wrapper = self._make_wrapper(num_experts=num_experts)

        wrapper.load_weights(torch.arange(num_experts, dtype=torch.int64))

        for proj, base in (("gate_proj", 100.0), ("up_proj", 0.0), ("down_proj", 200.0)):
            tensor = getattr(wrapper, proj)
            self.assertEqual(tuple(tensor.shape), (num_experts, 2, 4))
            self.assertTrue(tensor.is_contiguous())
            for expert_id in range(num_experts):
                expected = torch.full((2, 4), base + expert_id * 10)
                self.assertTrue(torch.equal(tensor[expert_id], expected), f"{proj} expert {expert_id}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
