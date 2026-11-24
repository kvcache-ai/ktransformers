"""Exercise the real factory/wrappers and Safetensors loader on CPU.

Only the native extension boundary, CPU worker and pinned mask allocation are
substituted. The fake native task reads all six weight/scale pointer arrays at
sync(), so these tests cover load_weights(), not just the cache helper. They do
not validate native quantized computation or model inference.
"""

import ctypes
import gc
import importlib
import sys
import tempfile
import unittest
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default")

PYTHON_ROOT = Path(__file__).resolve().parents[2] / "python"
PACKAGE = "_kt_checkpoint_wrapper_test"


class _Config:
    def __init__(self, *args):
        pass


class _NativeMoE:
    def __init__(self, config):
        self.config = config
        self.loaded = {}

    def load_weights_task(self, mapping_ptr):
        def read_weights():
            for proj in ("gate", "up", "down"):
                for suffix, ctype in (("projs", ctypes.c_uint8), ("scales", ctypes.c_float)):
                    pointers = getattr(self.config, f"{proj}_{suffix}")
                    self.loaded[f"{proj}_{suffix}"] = [
                        [np.ctypeslib.as_array((ctype * 4).from_address(ptr)).copy() for ptr in group]
                        for group in pointers
                    ]

        return read_weights


class _Worker:
    backend_ = None

    def __init__(self):
        self.pending = []

    def submit(self, task):
        self.pending.append(task)

    def sync(self):
        for task in self.pending:
            task()
        self.pending.clear()


def _package(name, path):
    module = ModuleType(name)
    module.__path__ = [str(path)]
    return module


class _OwnershipChecks:
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        extension = _package("kt_kernel_ext", self.root)
        moe = ModuleType("kt_kernel_ext.moe")
        moe.MOEConfig = _Config
        for name in ("AMXInt4_MOE", "AMXInt8_MOE", "Int4_KERNEL_MOE", "Int8_KERNEL_MOE"):
            setattr(moe, name, _NativeMoE)
        extension.moe = moe
        kvcache = ModuleType("kt_kernel_ext.kvcache")
        kvcache.ggml_type = SimpleNamespace()
        extension.kvcache = kvcache
        native_package = ModuleType("kt_kernel")
        native_package.kt_kernel_ext = extension
        # Keep imports private and restore sys.modules, including any modules
        # imported here, so this test does not replace a real installed backend.
        self.enterContext(
            patch.dict(
                sys.modules,
                {
                    PACKAGE: _package(PACKAGE, PYTHON_ROOT),
                    f"{PACKAGE}.utils": _package(f"{PACKAGE}.utils", PYTHON_ROOT / "utils"),
                    "kt_kernel": native_package,
                    "kt_kernel_ext": extension,
                    "kt_kernel_ext.moe": moe,
                    "kt_kernel_ext.kvcache": kvcache,
                },
            )
        )
        self.factory = importlib.import_module(f"{PACKAGE}.experts").KTMoEWrapper
        base = importlib.import_module(f"{PACKAGE}.experts_base").BaseMoEWrapper
        self.enterContext(patch.object(base, "_get_cpu_infer", return_value=_Worker()))
        zeros = torch.zeros

        def unpinned_zeros(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return zeros(*args, **kwargs)

        self.enterContext(patch.object(torch, "zeros", side_effect=unpinned_zeros))
        self.first = self.checkpoint("first", 11)
        self.second = self.checkpoint("second", 37)

    def checkpoint(self, name, value):
        path = self.root / name
        path.mkdir()
        tensors = {}
        for layer in (0, 1):
            for offset, proj in enumerate(("gate", "up", "down")):
                key = f"blk.{layer}.ffn_{proj}_exps.0.numa.0"
                tensors[f"{key}.weight"] = torch.full((2, 2), value + layer + offset, dtype=torch.uint8)
                tensors[f"{key}.scale"] = torch.full((2, 2), value + layer + offset + 0.25)
        save_file(tensors, str(path / "weights.safetensors"))
        return path

    def wrapper(self, path, layer=0):
        wrapper = self.factory(
            layer_idx=layer,
            num_experts=1,
            num_experts_per_tok=1,
            hidden_size=2,
            moe_intermediate_size=2,
            gpu_experts_mask=None,
            cpuinfer_threads=1,
            threadpool_count=1,
            weight_path=str(path),
            chunked_prefill_size=1,
            method=self.method,
        )
        expected = "AMXMoEWrapper" if self.method.startswith("AMX") else "GeneralMoEWrapper"
        self.assertEqual(type(wrapper).__name__, expected)
        return wrapper

    def assert_loaded(self, wrapper, value):
        wrapper.load_weights(torch.tensor([0], dtype=torch.int64))
        for offset, proj in enumerate(("gate", "up", "down")):
            for suffix, extra in (("projs", 0), ("scales", 0.25)):
                arrays = wrapper.moe.loaded[f"{proj}_{suffix}"]
                self.assertEqual(len(arrays), 1)
                self.assertEqual(len(arrays[0]), 1)
                np.testing.assert_array_equal(arrays[0][0], np.full(4, value + wrapper.layer_idx + offset + extra))

    def test_distinct_checkpoints_load_their_own_weights(self):
        first = self.wrapper(self.first)
        self.assert_loaded(first, 11)
        second = self.wrapper(self.second)
        self.assert_loaded(second, 37)
        self.assert_loaded(first, 11)
        self.assertIsNot(first.safetensor_loader, second.safetensor_loader)

    def test_same_checkpoint_layers_share_loader(self):
        first = self.wrapper(self.first)
        alias = self.root / "alias"
        alias.symlink_to(self.first, target_is_directory=True)
        second = self.wrapper(alias, layer=1)
        self.assertIs(first.safetensor_loader, second.safetensor_loader)
        self.assert_loaded(first, 11)
        self.assert_loaded(second, 11)

    def test_repointed_symlink_keeps_existing_wrapper_on_original_checkpoint(self):
        selected = self.root / "selected"
        selected.symlink_to(self.first, target_is_directory=True)
        first = self.wrapper(selected)
        selected.unlink()
        selected.symlink_to(self.second, target_is_directory=True)
        second = self.wrapper(selected)
        self.assert_loaded(second, 37)
        first.safetensor_loader.close_all_handles()
        self.assert_loaded(first, 11)

    def test_loader_lives_until_last_wrapper_is_released(self):
        first = self.wrapper(self.first)
        second = self.wrapper(self.first, layer=1)
        observed = weakref.ref(first.safetensor_loader)
        del first
        gc.collect()
        self.assertIsNotNone(observed())
        self.assert_loaded(second, 11)
        del second
        gc.collect()
        self.assertIsNone(observed())
        self.assert_loaded(self.wrapper(self.second), 37)


class TestAMXInt4Ownership(_OwnershipChecks, unittest.TestCase):
    method = "AMXINT4"


class TestAMXInt8Ownership(_OwnershipChecks, unittest.TestCase):
    method = "AMXINT8"


class TestGeneralInt4Ownership(_OwnershipChecks, unittest.TestCase):
    method = "MOE_INT4"


class TestGeneralInt8Ownership(_OwnershipChecks, unittest.TestCase):
    method = "MOE_INT8"


if __name__ == "__main__":
    unittest.main()
