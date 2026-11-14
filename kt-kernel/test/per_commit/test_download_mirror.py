import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=0.1, suite="default")

from kt_kernel.cli.commands import model as model_cmd
from kt_kernel.cli.utils import download_helper, model_verifier, user_model_registry


class TestDownloadMirror(unittest.TestCase):
    def test_mirror_after_hub_import(self):
        # Import first, as other CLI helpers can do before auto-switching.
        import huggingface_hub

        with patch.dict(os.environ, {}, clear=True), patch.object(huggingface_hub, "HfApi") as api:
            api.return_value.list_repo_tree.return_value = []
            self.assertTrue(download_helper.verify_repo_exists("test/model", "huggingface", True)[0])
            download_helper.list_remote_files_hf("test/model", True)
            self.assertEqual(api.call_count, 2)
            for call in api.call_args_list:
                self.assertEqual(call.kwargs.get("endpoint"), "https://hf-mirror.com")
            self.assertNotIn("HF_ENDPOINT", os.environ)

    def test_explicit_endpoint_takes_precedence(self):
        import huggingface_hub

        with (
            patch.dict(os.environ, {"HF_ENDPOINT": "https://custom.example"}),
            patch.object(huggingface_hub, "HfApi") as api,
        ):
            api.return_value.list_repo_tree.return_value = []
            download_helper.verify_repo_exists("test/model", "huggingface", True)
            download_helper.list_remote_files_hf("test/model", True)
            for call in api.call_args_list:
                self.assertEqual(call.kwargs.get("endpoint"), "https://custom.example")
            self.assertEqual(os.environ["HF_ENDPOINT"], "https://custom.example")

    def test_default_endpoint(self):
        import huggingface_hub

        with patch.dict(os.environ, {}, clear=True), patch.object(huggingface_hub, "HfApi") as api:
            api.return_value.list_repo_tree.return_value = []
            download_helper.list_remote_files_hf("test/model")
            self.assertIsNone(api.call_args.kwargs.get("endpoint"))

    def test_download_passes_mirror_to_snapshot(self):
        import huggingface_hub

        with TemporaryDirectory() as tmp:

            def download(**kwargs):
                directory = Path(kwargs["local_dir"])
                directory.mkdir()
                (directory / "model.safetensors").write_bytes(b"test model")

            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(model_verifier, "check_huggingface_connectivity", return_value=(False, "offline")),
                patch.object(user_model_registry, "USER_MODELS_FILE", Path(tmp) / "registry.yaml"),
                patch.object(model_cmd, "get_settings"),
                patch.object(huggingface_hub, "HfApi") as api,
                patch.object(huggingface_hub, "snapshot_download", side_effect=download) as snapshot,
            ):
                api.return_value.list_repo_tree.return_value = [SimpleNamespace(path="model.safetensors", size=10)]
                result = CliRunner().invoke(
                    model_cmd.app, ["download", "test/model", "--local-dir", str(Path(tmp) / "model"), "--yes"]
                )
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(snapshot.call_args.kwargs.get("endpoint"), "https://hf-mirror.com")
                self.assertNotIn("HF_ENDPOINT", os.environ)


if __name__ == "__main__":
    unittest.main()
