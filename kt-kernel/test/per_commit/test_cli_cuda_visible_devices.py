import importlib.util
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=0.1, suite="default")


ENVIRONMENT_PATH = Path(__file__).resolve().parents[2] / "python" / "cli" / "utils" / "environment.py"
SPEC = importlib.util.spec_from_file_location("environment", ENVIRONMENT_PATH)
assert SPEC is not None and SPEC.loader is not None
environment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(environment)


NVIDIA_SMI_OUTPUT = "0, GPU A, 24576\n1, GPU B, 24576\n2, GPU C, 24576\n3, GPU D, 24576\n"


class TestCudaVisibleDevicesFor(unittest.TestCase):
    def _select(self, cuda_visible, selected):
        env = {} if cuda_visible is None else {"CUDA_VISIBLE_DEVICES": cuda_visible}
        with patch.dict(os.environ, env, clear=True):
            with patch.object(environment, "run_command", return_value=NVIDIA_SMI_OUTPUT):
                gpus = environment.detect_gpus()
            names = [gpus[gpu_id].name for gpu_id in selected]
            return names, environment.cuda_visible_devices_for(selected)

    def test_selection_is_mapped_back_through_existing_cuda_visible_devices(self):
        names, value = self._select("2,3", [0, 1])
        self.assertEqual(names, ["GPU C", "GPU D"])
        self.assertEqual(value, "2,3")

        names, value = self._select("1,3", [1])
        self.assertEqual(names, ["GPU D"])
        self.assertEqual(value, "3")

    def test_range_syntax_is_mapped_like_detect_gpus(self):
        names, value = self._select("1-3", [0, 2])
        self.assertEqual(names, ["GPU B", "GPU D"])
        self.assertEqual(value, "1,3")

    def test_selection_is_unchanged_without_cuda_visible_devices(self):
        names, value = self._select(None, [0, 2])
        self.assertEqual(names, ["GPU A", "GPU C"])
        self.assertEqual(value, "0,2")

    def test_unparsable_cuda_visible_devices_keeps_selection(self):
        names, value = self._select("GPU-1234abcd,GPU-5678ef01", [1])
        self.assertEqual(names, ["GPU B"])
        self.assertEqual(value, "1")


if __name__ == "__main__":
    unittest.main()
