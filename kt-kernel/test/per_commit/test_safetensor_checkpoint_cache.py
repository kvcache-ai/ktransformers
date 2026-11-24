"""CPU-only coverage for shared Safetensors checkpoint indexes."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="default")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python" / "utils"))
import loader


class TestSharedSafeTensorCheckpoint(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.first = self.make_checkpoint("first", 1)
        self.second = self.make_checkpoint("second", 2)

    def make_checkpoint(self, name, value):
        root = self.root / name
        root.mkdir()
        save_file({"weight": torch.tensor([value])}, str(root / "weights.safetensors"))
        return root

    def test_isolates_distinct_checkpoints_and_reuses_resolved_alias(self):
        path = self.root / "selected"
        path.symlink_to(self.first, target_is_directory=True)
        first = loader.SafeTensorLoader.for_checkpoint(path)
        self.assertIs(first, loader.SafeTensorLoader.for_checkpoint(self.first))
        path.unlink()
        path.symlink_to(self.second, target_is_directory=True)
        second = loader.SafeTensorLoader.for_checkpoint(path)
        self.assertIsNot(first, second)
        self.assertEqual(first.load_tensor("weight").item(), 1)
        self.assertEqual(second.load_tensor("weight").item(), 2)
        self.assertIs(first, loader.SafeTensorLoader.for_checkpoint(self.first))

    def test_cache_does_not_own_unreferenced_checkpoint(self):
        first = loader.SafeTensorLoader.for_checkpoint(self.first)
        observed = weakref.ref(first)
        del first
        self.assertIsNone(observed())

    def test_parallel_requests_publish_one_index(self):
        with patch.object(loader, "safe_open", wraps=loader.safe_open) as opened:
            with ThreadPoolExecutor(max_workers=2) as pool:
                first, second = pool.map(loader.SafeTensorLoader.for_checkpoint, [self.first, self.first])
        self.assertIs(first, second)
        self.assertEqual(opened.call_count, 1)


if __name__ == "__main__":
    unittest.main()
