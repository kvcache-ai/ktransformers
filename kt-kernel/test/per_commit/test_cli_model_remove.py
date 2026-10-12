import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from typer.testing import CliRunner

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=0.1, suite="default")

from kt_kernel.cli.commands import model as model_cmd
from kt_kernel.cli.utils import user_model_registry
from kt_kernel.cli.utils.user_model_registry import UserModel, UserModelRegistry


class TestModelRemove(unittest.TestCase):
    def test_remove_prints_model_name_and_path(self):
        with TemporaryDirectory() as tmp:
            registry_file = Path(tmp) / "user_models.yaml"
            model_path = str(Path(tmp) / "my-model")
            UserModelRegistry(registry_file).add_model(UserModel(name="my-model", path=model_path, format="gguf"))

            with (
                patch.dict(os.environ, {"KT_LANG": "en", "COLUMNS": "400"}),
                patch.object(user_model_registry, "USER_MODELS_FILE", registry_file),
            ):
                result = CliRunner().invoke(model_cmd.app, ["remove", "my-model", "--yes"])

            self.assertEqual(result.exit_code, 0, result.output)
            self.assertNotIn("{name}", result.output)
            self.assertNotIn("{path}", result.output)
            self.assertIn("Remove model 'my-model' from registry?", result.output)
            self.assertIn(model_path, result.output.split("Model files will NOT be deleted from")[1])
            self.assertEqual(UserModelRegistry(registry_file).list_models(), [])


if __name__ == "__main__":
    unittest.main()
