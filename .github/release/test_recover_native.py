"""Recovery only consumes completed, audited CI inputs from unchanged runtime sources."""

import copy

import pytest

import recover_native as module


def run_fixture():
    run = {"event": "workflow_dispatch", "head_branch": "main", "path": ".github/workflows/release-four-main.yml", "status": "completed", "conclusion": "failure"}
    jobs = [{"steps": [{"name": name, "conclusion": "success"} for name in ("Compile all KT CPU variants and CUDA architectures", "Compile SGL CUDA payload from the independently locked SGLang main", "Inspect wheel metadata and source provenance")]}]
    comparison = {"status": "ahead", "files": [{"filename": ".github/release/carriers.py"}]}
    return run, jobs, comparison


def test_completed_compilation_allows_ci_only_repackaging():
    module.check_run(*run_fixture())


@pytest.mark.parametrize("change", ("canceled", "compile_failed", "model_changed", "renamed_model", "diverged", "untrusted_branch", "truncated"))
def test_recovery_rejects_unverified_inputs(change):
    run, jobs, comparison = run_fixture()
    if change == "canceled":
        run["conclusion"] = "cancelled"
    elif change == "compile_failed":
        jobs[0]["steps"][0]["conclusion"] = "failure"
    elif change == "model_changed":
        comparison["files"][0]["filename"] = "kt-kernel/python/sft/wrapper.py"
    elif change == "renamed_model":
        comparison["files"][0]["previous_filename"] = "kt-kernel/model.py"
    elif change == "diverged":
        comparison["status"] = "diverged"
    elif change == "untrusted_branch":
        run["head_branch"] = "feature"
    else:
        comparison["files"] *= 300
    with pytest.raises(ValueError):
        module.check_run(run, jobs, comparison)


def test_raw_input_hashes_and_original_snapshot_are_mandatory():
    original = {"sources": {"unit": "fixture"}}
    lock = original | {"recovery": {"assembly_workflow_sha": "a" * 40}}
    wheels = [{"filename": f"fixture{i}.whl", "name": f"fixture{i}", "version": "1", "sha256": str(i) * 64, "size": 10, "requires_dist": []} for i in range(6)]
    report = {"source_lock": original, "stage": "raw-native-wheels", "metadata_consistent": True, "errors": [], "wheels": wheels}
    module.verify_inputs(lock, report, wheels)
    altered = copy.deepcopy(wheels)
    altered[0]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="changed"):
        module.verify_inputs(lock, report, altered)
    with pytest.raises(ValueError, match="six"):
        module.verify_inputs(lock, report, wheels[:-1])
    with pytest.raises(ValueError, match="snapshot"):
        module.verify_inputs(lock, report | {"source_lock": {}}, wheels)
