"""Adversarial promotion checks. All remote calls/uploads are mocked."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contracts import CASES, digest, write_json
from release_stack import (
    check_evidence,
    check_unused,
    cleanup,
    existing_matches,
    promote,
)
from test_release_contracts import fixture, report


@pytest.fixture
def candidate(tmp_path, monkeypatch):
    root = tmp_path / "candidate"
    root.mkdir()
    manifest = fixture(root)
    sha = digest(root / "release.json")
    evidence = tmp_path / "evidence"
    (evidence / "tests").mkdir(parents=True)
    write_json(
        evidence / "request.json",
        {
            "mode": "release-candidate",
            "manifest_sha256": sha,
            "build_run_id": 123,
            "build_run_attempt": 1,
            "build_workflow_sha": "a" * 40,
            "harness_sha": "a" * 40,
        },
    )
    suite = {
        "status": "passed",
        "cases": {case: {"status": "passed"} for case in CASES},
    }
    for name in ("host-result.json", "tests/suite.json"):
        write_json(evidence / name, suite)
    for label in ("serving", "combined"):
        write_json(evidence / f"tests/release-{label}-install.json", report(manifest))
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "workflow_dispatch")
    monkeypatch.setenv("GITHUB_REPOSITORY", "kvcache-ai/ktransformers")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    return root, manifest, sha, evidence


def files(entry):
    return [
        {
            "filename": entry["file"],
            "digests": {"sha256": entry["sha256"]},
            "yanked": False,
        }
    ]


def test_uploads_tested_bytes_in_dependency_order_with_kt_last(candidate, tmp_path):
    root, manifest, sha, evidence = candidate
    uploaded = {}

    def query(name, version):
        return files(manifest["wheels"][name]) if name in uploaded else []

    def upload(path):
        name = next(
            name
            for name, entry in manifest["wheels"].items()
            if entry["file"] == path.name
        )
        assert digest(path) == manifest["wheels"][name]["sha256"]
        uploaded[name] = path

    output = tmp_path / "promotion.json"
    promote(root, sha, evidence, output, query=query, upload=upload)
    assert list(uploaded) == [
        "accelerate-kt",
        "transformers-kt",
        "kt-kernel",
        "sglang-kt",
        "ktransformers",
    ]
    assert json.loads(output.read_text())["status"] == "uploaded-awaiting-public-e2e"


def test_retry_reuses_only_exact_files_without_reupload(candidate, tmp_path):
    root, manifest, sha, evidence = candidate
    promote(
        root,
        sha,
        evidence,
        tmp_path / "promotion.json",
        query=lambda name, version: files(manifest["wheels"][name]),
        upload=lambda path: pytest.fail("Must not re-upload identical published files"),
    )


def test_conflict_in_last_package_fails_before_any_upload(candidate, tmp_path):
    root, manifest, sha, evidence = candidate
    conflicting = copy.deepcopy(manifest["wheels"]["ktransformers"])
    conflicting["sha256"] = "c" * 64
    with pytest.raises(ValueError, match="different bytes"):
        promote(
            root,
            sha,
            evidence,
            tmp_path / "promotion.json",
            query=lambda name, version: files(conflicting)
            if name == "ktransformers"
            else [],
            upload=lambda path: pytest.fail("No writes before full batch preflight"),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("mode", "release-pypi"),
        ("manifest_sha256", "c" * 64),
        ("build_run_id", 456),
        ("build_run_attempt", 2),
        ("harness_sha", "c" * 40),
    ],
)
def test_wrong_acceptance_identity_fails(candidate, field, value):
    root, manifest, sha, evidence = candidate
    path = evidence / "request.json"
    request = json.loads(path.read_text())
    write_json(path, request | {field: value})
    with pytest.raises(ValueError):
        check_evidence(evidence, manifest, sha, "candidate")


@pytest.mark.parametrize("status", ["failed", "resource_unavailable", "not_run"])
def test_missing_or_failed_model_blocks_promotion(candidate, status):
    root, manifest, sha, evidence = candidate
    path = evidence / "tests/suite.json"
    result = json.loads(path.read_text())
    result["cases"]["glm53_inference"]["status"] = status
    write_json(path, result)
    with pytest.raises(ValueError, match="model acceptance"):
        check_evidence(evidence, manifest, sha, "candidate")


def test_incomplete_suite_never_passes(candidate):
    root, manifest, sha, evidence = candidate
    path = evidence / "tests/suite.json"
    result = json.loads(path.read_text())
    del result["cases"]["deepseek_v31_lora"]
    write_json(path, result)
    with pytest.raises(ValueError):
        check_evidence(evidence, manifest, sha, "candidate")


def test_pypi_check_must_use_public_downloads(candidate):
    root, manifest, sha, evidence = candidate
    request = json.loads((evidence / "request.json").read_text())
    write_json(evidence / "request.json", request | {"mode": "release-pypi"})
    with pytest.raises(ValueError, match="public PyPI"):
        check_evidence(evidence, manifest, sha, "pypi")


def test_existing_version_cannot_be_rebuilt_under_same_name():
    with pytest.raises(ValueError, match="already exists"):
        check_unused(
            [{"name": "kt-kernel", "version": "1.0"}], query=lambda *args: [{}]
        )


def test_yanked_or_extra_remote_files_are_not_skipped():
    entry = {"file": "x.whl", "sha256": "a" * 64}
    for remote in (files(entry) * 2, [files(entry)[0] | {"yanked": True}]):
        with pytest.raises(ValueError):
            existing_matches(entry, remote)


def test_partial_upload_never_claims_release_success(candidate, tmp_path):
    root, manifest, sha, evidence = candidate

    def fail(path):
        raise RuntimeError("network failure")

    output = tmp_path / "promotion.json"
    with pytest.raises(RuntimeError):
        promote(root, sha, evidence, output, query=lambda *args: [], upload=fail)
    assert json.loads(output.read_text())["status"] == "partial-or-failed-upload"


def test_cleanup_is_scoped_to_an_owned_run_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    with pytest.raises(ValueError):
        cleanup(tmp_path, tmp_path.parent)
    owned = tmp_path / "kt-four-main.example"
    owned.mkdir()
    (owned / ".kt-release-owned").write_text("456")
    with pytest.raises(ValueError):
        cleanup(owned, tmp_path)
    (owned / ".kt-release-owned").write_text("123")
    cleanup(owned, tmp_path)
    assert not owned.exists()
