"""Release integration failure cases; no network, GPU, or upload credentials."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import inside
from contracts import PACKAGES, REPOSITORIES, digest, validate_request, write_json
from release_contracts import verify_install_report, verify_release


def fixture(root):
    house = root / "wheelhouse"
    house.mkdir()
    entries = {}
    wheels = {}
    for package in sorted(PACKAGES):
        name = package.replace("-", "_") + "-1.0-py3-none-any.whl"
        (house / name).write_bytes(package.encode())
        entries[name] = {
            "name": package,
            "version": "1.0",
            "sha256": digest(house / name),
        }
        wheels[package] = {
            "file": name,
            "version": "1.0",
            "sha256": entries[name]["sha256"],
        }
    manifest = {
        "schema": 1,
        "run_id": 123,
        "run_attempt": 1,
        "workflow_sha": "a" * 40,
        "source_lock": {
            "workflow_sha": "a" * 40,
            "sources": {
                name: {"repository": repo, "ref": "refs/heads/main", "sha": "a" * 40}
                for name, repo in REPOSITORIES.items()
            },
        },
        "wheels": wheels,
        "wheelhouse": entries,
        "plans": {
            extra: {name: entry["file"] for name, entry in wheels.items()}
            for extra in ("sglang", "sglang,sft")
        },
    }
    write_json(root / "release.json", manifest)
    return manifest


def report(manifest, public=False):
    return {
        "install": [
            {
                "metadata": {"name": name, "version": entry["version"]},
                "download_info": {
                    "url": (
                        "https://files.pythonhosted.org/packages/"
                        if public
                        else "file:///input/release/wheelhouse/"
                    )
                    + entry["file"],
                    "archive_info": {"hashes": {"sha256": entry["sha256"]}},
                },
            }
            for name, entry in manifest["wheels"].items()
        ]
    }


class ReleaseContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = fixture(self.root)
        self.sha = digest(self.root / "release.json")

    def test_release_and_both_installation_modes(self):
        verify_release(self.root, self.sha)
        for extra in ("sglang", "sglang,sft"):
            for public in (True, False):
                verify_install_report(
                    report(self.manifest, public), self.manifest, extra, public=public
                )

    def test_wheel_tampering(self):
        next((self.root / "wheelhouse").iterdir()).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA256 changed"):
            verify_release(self.root, self.sha)

    def test_rewritten_manifest(self):
        manifest = self.manifest | {"run_id": 456}
        write_json(self.root / "release.json", manifest)
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            verify_release(self.root, self.sha)

    def test_extra_dependency_file(self):
        (self.root / "wheelhouse/unknown.whl").touch()
        with self.assertRaisesRegex(ValueError, "file set"):
            verify_release(self.root, self.sha)

    def test_report_rejects_same_version_different_wheel(self):
        data = report(self.manifest, True)
        data["install"][0]["download_info"]["archive_info"]["hashes"]["sha256"] = (
            "b" * 64
        )
        with self.assertRaisesRegex(ValueError, "hash changed"):
            verify_install_report(data, self.manifest, "sglang,sft", public=True)

    def test_public_test_rejects_local_artifacts(self):
        with self.assertRaisesRegex(ValueError, "public PyPI"):
            verify_install_report(
                report(self.manifest), self.manifest, "sglang,sft", public=True
            )

    def test_dependency_drift_cannot_pass(self):
        data = report(self.manifest, True)
        data["install"].pop()
        with self.assertRaisesRegex(ValueError, "resolution differs"):
            verify_install_report(data, self.manifest, "sglang,sft", public=True)

    def test_release_request_cannot_impersonate_pr(self):
        request = {
            "schema": 1,
            "mode": "release-candidate",
            "harness_sha": "a" * 40,
            "manifest_sha256": self.sha,
            "build_run_id": 123,
            "build_run_attempt": 1,
            "build_workflow_sha": "a" * 40,
        }
        validate_request(request)
        with self.assertRaises(ValueError):
            validate_request(request | {"pr": {}})

    def test_wheelhouse_symlink(self):
        path = next((self.root / "wheelhouse").iterdir())
        target = self.root / "outside"
        path.rename(target)
        path.symlink_to(target)
        with self.assertRaises(ValueError):
            verify_release(self.root, self.sha)

    def test_wrong_source_branch(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["source_lock"]["sources"]["sglang"]["ref"] = "refs/heads/hotfix"
        write_json(self.root / "release.json", manifest)
        with self.assertRaisesRegex(ValueError, "must be main"):
            verify_release(self.root, digest(self.root / "release.json"))

    def check_installer(self, public):
        work = self.root / "work"
        work.mkdir()
        evidence = work / "evidence"
        evidence.mkdir()
        commands = []

        def run(command, log, **kwargs):
            command = list(map(str, command))
            commands.append(command)
            if "--report" in command:
                write_json(
                    command[command.index("--report") + 1],
                    report(self.manifest, public),
                )
            if "/harness/installed.py" in command:
                write_json(command[-1], {name: {"version": "1.0"} for name in PACKAGES})

        with (
            patch.object(inside, "WORK", work),
            patch.object(inside, "EVIDENCE", evidence),
            patch.object(inside, "PYTHON", work / "venv/bin/python"),
            patch.object(inside, "verify_release", return_value=self.manifest),
            patch.object(inside, "run", side_effect=run),
            patch.object(inside.venv, "EnvBuilder") as builder,
        ):
            builder.return_value.create.side_effect = (
                lambda directory: directory.mkdir()
            )
            inside.install_release(
                {
                    "mode": "release-pypi" if public else "release-candidate",
                    "manifest_sha256": self.sha,
                },
                {},
            )
        installs = [command for command in commands if "--report" in command]
        self.assertEqual(
            [command[-1] for command in installs],
            ["ktransformers[sglang]", "ktransformers[sglang,sft]"],
        )
        self.assertTrue(installs[0][0].endswith("serving-venv/bin/python"))
        self.assertTrue(installs[1][0].endswith("/venv/bin/python"))
        for command in installs:
            self.assertIn("--index-url" if public else "--no-index", command)
            self.assertNotIn("--no-deps", command)

    def test_release_candidate_installs_user_extras_offline_into_two_fresh_venvs(self):
        self.check_installer(False)

    def test_release_pypi_installs_unpinned_user_extras_into_two_fresh_venvs(self):
        self.check_installer(True)


if __name__ == "__main__":
    unittest.main()
