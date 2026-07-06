import importlib.util
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="default")


def _load_path_utils():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "python"
        / "cli"
        / "utils"
        / "path_utils.py"
    )
    spec = importlib.util.spec_from_file_location("path_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


path_utils = _load_path_utils()


class TestPromptCustomPath(unittest.TestCase):
    @patch.object(path_utils.os.path, "exists", return_value=False)
    def test_missing_windows_drive_root_does_not_loop_forever(self, mock_exists):
        expected_parent = Path("Z:\\models").parent
        while True:
            next_parent = expected_parent.parent
            if next_parent == expected_parent:
                break
            expected_parent = next_parent

        result = path_utils.find_existing_parent("Z:\\models")

        self.assertEqual(result, str(expected_parent))
        self.assertEqual(mock_exists.call_count, 1)


if __name__ == "__main__":
    unittest.main()
