"""Regression tests for `kt model verify` file matching (issue #2100).

`kt model verify` fetches the remote SHA256 set (fetch_model_sha256), which spans
*.safetensors, *.json and *.py at any depth, keyed by repo-relative path
(e.g. inference/config.json). Before the fix the local scan only globbed
*.safetensors non-recursively and keyed hashes by basename, so on a healthy model
every config/code file, and every file in a subdirectory, was reported "missing"
and the model was flagged as potentially corrupted.

These tests are pure Python (temp files + hashing), no compiled kt_kernel needed.
model_verifier is imported by file location so the suite runs without a build.
"""

import hashlib
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

# Make model_verifier importable under a real top-level name so its worker
# function pickles across the ProcessPoolExecutor (spec-loading under a synthetic
# name would break pickling; the built package uses the real name in production).
_UTILS_DIR = Path(__file__).resolve().parents[2] / "python" / "cli" / "utils"
sys.path.insert(0, str(_UTILS_DIR))
import model_verifier  # noqa: E402

register_cpu_ci(est_time=10, suite="default")


def _sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _write(root: Path, rel: str, data: bytes) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


class TestModelVerifierRelpath(unittest.TestCase):
    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        # A healthy model mirroring the layout in issue #2100: weights, top-level
        # config/code, and files nested under subdirectories. config.json exists
        # both at the top level and under inference/ with DIFFERENT contents.
        self.files = {
            "model-00001-of-00001.safetensors": b"weights-shard-1",
            "config.json": b'{"model_type":"deepseek_v4"}',
            "generation_config.json": b'{"temperature":0.6}',
            "model.safetensors.index.json": b'{"weight_map":{}}',
            "tokenizer.json": b'{"version":"1.0"}',
            "tokenizer_config.json": b'{"bos":"<s>"}',
            "inference/config.json": b'{"tp":8,"note":"different from top-level"}',
            "inference/model.py": b"# inference model code\n",
            "encoding/encoding_dsv4.py": b"# encoding code\n",
            "encoding/tests/test_input_1.json": b'{"case":1}',
        }
        # Remote hash set: keyed by repo-relative path, exactly as HF returns it.
        self.official = {}
        for rel, data in self.files.items():
            _write(self.root, rel, data)
            self.official[rel] = _sha256(data)

    def tearDown(self):
        self._tmp.cleanup()

    def test_calculate_local_sha256_keys_by_relative_path(self):
        """Core regression: hashes are keyed by relative path, so a basename that
        repeats across subdirectories (config.json vs inference/config.json) stays
        distinct. Fails on the pre-fix code, which keyed by basename and collided
        the two. Uses only calculate_local_sha256, which exists before the fix."""
        top = self.root / "config.json"
        sub = self.root / "inference" / "config.json"
        local = model_verifier.calculate_local_sha256(self.root, files_list=[top, sub])

        self.assertIn("config.json", local)
        self.assertIn("inference/config.json", local)
        self.assertEqual(local["config.json"], _sha256(self.files["config.json"]))
        self.assertEqual(local["inference/config.json"], _sha256(self.files["inference/config.json"]))
        # The two must not have collapsed into one entry with a shared hash.
        self.assertNotEqual(local["config.json"], local["inference/config.json"])

    def test_list_local_model_files_recursive_all_patterns(self):
        """The local scan finds every verifiable suffix at any depth."""
        found = {p.relative_to(self.root).as_posix() for p in model_verifier.list_local_model_files(self.root)}
        self.assertEqual(found, set(self.files.keys()))

    def test_healthy_model_verifies_clean(self):
        """The issue #2100 scenario: a healthy model (json, py, subdirs) reports
        zero missing and zero mismatched."""
        local = model_verifier.calculate_local_sha256(
            self.root, files_list=model_verifier.list_local_model_files(self.root)
        )
        passed, missing, mismatched = model_verifier.compare_local_to_official(self.official, local)
        self.assertEqual(missing, [])
        self.assertEqual(mismatched, [])
        self.assertEqual(passed, len(self.official))

    def test_basename_collision_matched_by_relative_path(self):
        """config.json and inference/config.json are matched to their own remote
        entry, not each other's, so neither is a false mismatch."""
        local = model_verifier.calculate_local_sha256(
            self.root, files_list=model_verifier.list_local_model_files(self.root)
        )
        _, missing, mismatched = model_verifier.compare_local_to_official(
            {
                "config.json": self.official["config.json"],
                "inference/config.json": self.official["inference/config.json"],
            },
            local,
        )
        self.assertEqual((missing, mismatched), ([], []))

    def test_genuinely_missing_and_corrupt_files_still_detected(self):
        """The fix must not turn verification into a rubber stamp: a deleted subdir
        file is reported missing, and a tampered file is reported mismatched."""
        (self.root / "inference" / "model.py").unlink()
        _write(self.root, "encoding/encoding_dsv4.py", b"# tampered content\n")
        local = model_verifier.calculate_local_sha256(
            self.root, files_list=model_verifier.list_local_model_files(self.root)
        )
        _, missing, mismatched = model_verifier.compare_local_to_official(self.official, local)
        self.assertEqual(missing, ["inference/model.py"])
        self.assertEqual(mismatched, ["encoding/encoding_dsv4.py"])


if __name__ == "__main__":
    unittest.main()
