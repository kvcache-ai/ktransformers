"""CPU-only release contract tests; no network, builds or publishing."""

import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("four_main.py")
SPEC = importlib.util.spec_from_file_location("four_main", SCRIPT)
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def stable_lock():
    return release.snapshot("a" * 40, resolver=lambda _: "b" * 40)


def test_snapshot_freezes_only_four_main_heads_and_workflow_sha():
    lock = stable_lock()
    assert lock["workflow_sha"] == "a" * 40
    assert len(lock["sources"]) == 4
    assert all(source["sha"] == "b" * 40 for source in lock["sources"].values())
    assert all(
        source["ref"] == "refs/heads/main" for source in lock["sources"].values()
    )


def test_racing_merges_restart_the_whole_snapshot():
    # First pair observes different commits. Second pair is stable.
    answers = iter(["b" * 40] * 4 + ["c" * 40] * 4 + ["d" * 40] * 8)
    lock = release.snapshot("a" * 40, resolver=lambda _: next(answers))
    assert {source["sha"] for source in lock["sources"].values()} == {"d" * 40}


def test_continuously_moving_heads_fail_instead_of_mixing_snapshots():
    answers = iter(["b" * 40] * 4 + ["c" * 40] * 4)
    with pytest.raises(RuntimeError, match="main heads kept moving"):
        release.snapshot("a" * 40, resolver=lambda _: next(answers), attempts=1)


@pytest.mark.parametrize("change", ["repository", "ref", "sha", "missing"])
def test_lock_rejects_unreviewed_repositories_branches_and_short_shas(change):
    lock = stable_lock()
    if change == "missing":
        del lock["sources"]["sglang"]
    else:
        lock["sources"]["sglang"][change] = {
            "repository": "someone/sglang",
            "ref": "refs/heads/hotfix",
            "sha": "abcdef1",
        }[change]
    with pytest.raises(ValueError):
        release.validate_lock(lock)


def wheels():
    versions = {
        "ktransformers": "0.7.1",
        "kt-kernel": "0.7.1",
        "sglang-kt": "0.7.1",
        "sgl-kernel-kt": "0.3.21.post3",
        "transformers-kt": "5.6.0.post5",
        "accelerate-kt": "1.14.0.post3",
    }
    result = {
        name: {"name": name, "version": version, "requires_dist": []}
        for name, version in versions.items()
    }
    for name, dependencies in release.EXACT_DEPENDENCIES.items():
        result[name]["requires_dist"] = [
            f"{dep}=={versions[dep]}" for dep in dependencies
        ]
    result["transformers-kt"]["requires_dist"] = [
        "accelerate-kt>=1.14.0.post3; extra == 'sft'"
    ]
    return result


def test_consistent_stack_passes_metadata_gate():
    assert release.audit_wheels(list(wheels().values())) == []


def test_python_preflight_checks_versions_before_native_builds():
    stack = wheels()
    del stack["kt-kernel"]
    del stack["sgl-kernel-kt"]
    assert release.audit_wheels(list(stack.values()), python_only=True) == []
    stack["sglang-kt"]["requires_dist"][0] = "kt-kernel==0.7.0.post2"
    assert release.audit_wheels(list(stack.values()), python_only=True)


def test_native_version_cannot_disagree_with_python_preflight():
    stack = wheels()
    stack["kt-kernel"]["version"] = "0.7.0.post2"
    assert release.audit_wheels(list(stack.values()))


def test_stale_main_pins_are_reported_without_rewriting_them():
    stack = wheels()
    original = "transformers-kt==5.6.0.post3; extra == 'sft'"
    stack["ktransformers"]["requires_dist"][2] = original
    errors = release.audit_wheels(list(stack.values()))
    assert any("5.6.0.post3" in error and "5.6.0.post5" in error for error in errors)
    assert stack["ktransformers"]["requires_dist"][2] == original


@pytest.mark.parametrize(
    "requirement", ["transformers>=5", "accelerate>=1", "sgl-kernel==0.3.21"]
)
def test_namespace_collisions_fail_in_user_extras(requirement):
    stack = wheels()
    stack["sglang-kt"]["requires_dist"].append(requirement + "; extra == 'sglang'")
    assert any(
        "conflicting upstream" in error
        for error in release.audit_wheels(list(stack.values()))
    )


def test_test_only_dependencies_do_not_poison_serving_audit():
    stack = wheels()
    stack["sglang-kt"]["requires_dist"].append("accelerate; extra == 'test'")
    assert release.audit_wheels(list(stack.values())) == []


@pytest.mark.parametrize(
    "requirement", ["kt-kernel>=0.7.1", "kt-kernel @ https://example.org/wheel.whl"]
)
def test_direct_or_floating_dependencies_cannot_replace_stack_pins(requirement):
    stack = wheels()
    stack["sglang-kt"]["requires_dist"][0] = requirement
    assert release.audit_wheels(list(stack.values()))


@pytest.mark.parametrize("duplicate", [False, True])
def test_missing_or_duplicate_raw_wheels_fail(duplicate):
    stack = list(wheels().values())
    if duplicate:
        stack.append(dict(stack[0]))
    else:
        stack.pop()
    assert release.audit_wheels(stack)


def write_wheel(directory, package, metadata_name=None):
    name = package["name"].replace("-", "_")
    version = package["version"]
    path = directory / f"{name}-{version}-py3-none-any.whl"
    metadata = [
        "Metadata-Version: 2.1",
        f"Name: {metadata_name or package['name']}",
        f"Version: {version}",
        *(f"Requires-Dist: {req}" for req in package["requires_dist"]),
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{name}-{version}.dist-info/METADATA", "\n".join(metadata) + "\n"
        )
    return path


def test_wheel_name_and_metadata_must_agree(tmp_path):
    path = write_wheel(tmp_path, wheels()["ktransformers"], metadata_name="unrelated")
    with pytest.raises(ValueError, match="disagree"):
        release.inspect_wheel(path)


def test_wheel_evidence_contains_actual_file_hash_and_dependencies(tmp_path):
    package = wheels()["sglang-kt"]
    path = write_wheel(tmp_path, package)
    evidence = release.inspect_wheel(path)
    assert evidence["sha256"] == release.sha256(path)
    assert evidence["size"] == path.stat().st_size
    assert evidence["requires_dist"] == package["requires_dist"]


def make_sources(tmp_path):
    lock = stable_lock()
    sources = tmp_path / "sources"
    for key in release.REPOSITORIES:
        target = sources / key
        target.mkdir(parents=True)
        release.run("git", "init", "-q", str(target))
        (target / "runtime.py").write_text("VALUE = 1\n")
        release.run("git", "add", "runtime.py", cwd=target)
        release.run(
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "-qm",
            "source",
            cwd=target,
        )
        lock["sources"][key]["sha"] = release.run(
            "git", "rev-parse", "HEAD", cwd=target
        )
    return lock, sources


def test_runtime_hotpatch_after_checkout_fails_source_audit(tmp_path):
    lock, sources = make_sources(tmp_path)
    (sources / "sglang/runtime.py").write_text("VALUE = 2\n")
    with pytest.raises(RuntimeError, match="Source changed"):
        release.source_evidence(lock, sources)


@pytest.mark.parametrize("stale_pin", [False, True])
def test_audit_records_errors_and_never_labels_raw_wheels_publishable(
    tmp_path, stale_pin
):
    lock, sources = make_sources(tmp_path)
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    stack = wheels()
    if stale_pin:
        stack["ktransformers"]["requires_dist"][0] = "kt-kernel==0.7.0.post2"
    for package in stack.values():
        write_wheel(wheel_dir, package)
    output = tmp_path / "report.json"
    assert release.audit(lock, sources, wheel_dir, output) is not stale_pin
    report = json.loads(output.read_text())
    assert report["publishable"] is False
    assert report["metadata_consistent"] is not stale_pin
    assert bool(report["errors"]) is stale_pin
    assert len(report["wheels"]) == 6
    assert all(wheel["source"]["sha"] for wheel in report["wheels"])


def test_command_line_help_is_available():
    result = subprocess.run(
        [release.sys.executable, str(SCRIPT), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "snapshot" in result.stdout and "audit" in result.stdout
