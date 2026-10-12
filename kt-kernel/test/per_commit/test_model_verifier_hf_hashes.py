import importlib.util
import sys
import types
import unittest
from pathlib import Path

from ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=0.1, suite="default")


MODEL_VERIFIER_PATH = Path(__file__).resolve().parents[2] / "python" / "cli" / "utils" / "model_verifier.py"
SPEC = importlib.util.spec_from_file_location("model_verifier", MODEL_VERIFIER_PATH)
assert SPEC is not None and SPEC.loader is not None
model_verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model_verifier)


class _Lfs:
    def __init__(self, sha256):
        self.sha256 = sha256


class _RepoFile:
    """Stand-in for huggingface_hub.hf_api.RepoFile.

    The Hub returns `lfs.sha256` only for LFS-tracked files. For a plain git blob
    it returns `blob_id`, which huggingface_hub populates from the object's `oid`,
    the git SHA1. A git SHA1 is 40 hex chars and a SHA256 is 64.
    """

    def __init__(self, path, blob_id, lfs=None):
        self.path = path
        self.blob_id = blob_id
        self.lfs = lfs


SAFETENSORS_SHA256 = "8111d5afb0715dbf5a31396d31432cb56370ba23f6650a035ea0fc8a20b4e500"
CONFIG_BLOB_SHA1 = "4ff64fe5192d88c0b5dbfc578c775c0ce05dd7d0"
TOKENIZER_BLOB_SHA1 = "980677bff66333505f8bea2719064a6e43f95314"
HANDLER_BLOB_SHA1 = "817762d631ad6f9c799f6b9dc713c46420e65546"

FAKE_FILES = [
    _RepoFile("model.safetensors", "fa9abfdec19ae0d6400cd3e9a25ee885633a8253", _Lfs(SAFETENSORS_SHA256)),
    _RepoFile("config.json", CONFIG_BLOB_SHA1),
    _RepoFile("tokenizer.json", TOKENIZER_BLOB_SHA1),
    _RepoFile("handler.py", HANDLER_BLOB_SHA1),
]


class _FakeHfApi:
    def get_paths_info(self, repo_id, paths, revision):
        return [f for f in FAKE_FILES if f.path in set(paths)]


def _install_fake_hub():
    """Inject a stub huggingface_hub so the lazy import inside the function resolves."""
    module = types.ModuleType("huggingface_hub")
    module.HfApi = _FakeHfApi
    module.list_repo_files = lambda repo_id, revision: [f.path for f in FAKE_FILES]
    saved = sys.modules.get("huggingface_hub")
    sys.modules["huggingface_hub"] = module
    return saved


def _restore_fake_hub(saved):
    if saved is None:
        sys.modules.pop("huggingface_hub", None)
    else:
        sys.modules["huggingface_hub"] = saved


class TestFetchFromHuggingface(unittest.TestCase):
    def setUp(self):
        self._saved_hub = _install_fake_hub()
        self.addCleanup(_restore_fake_hub, self._saved_hub)

    def test_every_returned_value_is_a_sha256(self):
        """The mapping is compared against local SHA256 digests, so it must only hold SHA256."""
        result = model_verifier._fetch_from_huggingface("some/repo", "main")

        self.assertTrue(result, "expected at least the LFS-tracked weight file")
        for path, digest in result.items():
            self.assertEqual(len(digest), 64, f"{path} returned a {len(digest)}-char digest, not a SHA256")
            self.assertNotIn(digest, {CONFIG_BLOB_SHA1, TOKENIZER_BLOB_SHA1, HANDLER_BLOB_SHA1})

    def test_lfs_file_keeps_its_sha256(self):
        result = model_verifier._fetch_from_huggingface("some/repo", "main")

        self.assertEqual(result.get("model.safetensors"), SAFETENSORS_SHA256)

    def test_non_lfs_files_are_omitted_not_reported_with_a_git_sha1(self):
        """A git SHA1 can never equal a local SHA256, so emitting one is a guaranteed false mismatch."""
        result = model_verifier._fetch_from_huggingface("some/repo", "main")

        for path in ("config.json", "tokenizer.json", "handler.py"):
            self.assertNotIn(path, result)


if __name__ == "__main__":
    unittest.main()
