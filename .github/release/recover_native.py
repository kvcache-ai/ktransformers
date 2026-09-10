"""Reuse hash-verified CI raw wheels after a packaging-only failure, never rebuild.

The original four-main source snapshot remains frozen. A different trusted CI
assembler revision is explicit; recovery is refused after any KT runtime change.
"""

import argparse
import json
import os
from pathlib import Path
import re
import urllib.request

from four_main import inspect_wheel, read_lock, save_json

REPO = "kvcache-ai/ktransformers"


def require(value, message):
    if not value:
        raise ValueError(message)


def api(path):
    request = urllib.request.Request("https://api.github.com/repos/" + REPO + path,
                                     headers={"Authorization": "Bearer " + os.environ["GH_TOKEN"], "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def check_run(run, jobs, comparison):
    require(run["event"] == "workflow_dispatch" and run["head_branch"] == "main", "Only official main builds may be recovered")
    require(run["path"] == ".github/workflows/release-four-main.yml" and run["status"] == "completed", "Original build must be complete")
    require(run["conclusion"] in ("success", "failure"), "Do not recover canceled or unfinished compilation")
    completed = {step["name"] for job in jobs for step in job.get("steps", []) if step["conclusion"] == "success"}
    required = {"Compile all KT CPU variants and CUDA architectures", "Compile SGL CUDA payload from the independently locked SGLang main", "Inspect wheel metadata and source provenance"}
    require(required <= completed, "Original native compilation and source audit must have succeeded")
    require(comparison["status"] in ("ahead", "identical"), "Original build is not a main ancestor")
    files = comparison["files"]
    require(len(files) < 300, "Refuse a truncated compare response")
    require(all(item["filename"].startswith(".github/") and item.get("previous_filename", item["filename"]).startswith(".github/") for item in files), "Runtime sources changed; compile a fresh stack instead")


def preflight(run_id, attempt):
    require(os.environ.get("GITHUB_REPOSITORY") == REPO and os.environ.get("GITHUB_REF") == "refs/heads/main", "Official main only")
    require(run_id.isdecimal() and attempt.isdecimal() and int(attempt) > 0, "Invalid original run")
    run = api("/actions/runs/" + run_id)
    require(run["run_attempt"] == int(attempt), "Original attempt changed")
    jobs = api(f"/actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100")
    require(jobs["total_count"] <= 100, "Too many jobs to validate")
    compare = api("/compare/" + run["head_sha"] + "..." + os.environ["GITHUB_SHA"])
    check_run(run, jobs["jobs"], compare)
    save_json(Path("recovery-source.json"), {"run_id": int(run_id), "run_attempt": int(attempt), "workflow_sha": run["head_sha"], "assembly_workflow_sha": os.environ["GITHUB_SHA"], "ci_only_diff": [item["filename"] for item in compare["files"]]})


def check_lock(path):
    lock = read_lock(path)
    recovery = json.loads(Path("recovery-source.json").read_text())
    require(lock["workflow_sha"] == lock["sources"]["ktransformers"]["sha"] == recovery["workflow_sha"], "Original source lock differs from the compiled workflow")
    require("recovery" not in lock, "Recover only original compile runs, not recovery chains")
    lock["recovery"] = recovery
    save_json(path, lock)


def verify_inputs(lock, report, wheels):
    recovery = lock["recovery"]
    expected_lock = {key: value for key, value in lock.items() if key != "recovery"}
    require(report["source_lock"] == expected_lock, "Raw wheels belong to another four-main snapshot")
    require(report["stage"] == "raw-native-wheels" and report["metadata_consistent"] is True and not report["errors"], "Original source audit failed")
    require(re.fullmatch(r"[0-9a-f]{40}", recovery["assembly_workflow_sha"]), "Invalid assembler SHA")
    expected = {item["filename"]: item for item in report["wheels"]}
    actual = {item["filename"]: item for item in wheels}
    require(len(expected) == len(report["wheels"]) == 6 and set(actual) == set(expected), "Require exactly the six original raw wheels")
    for name, item in actual.items():
        require(all(item[key] == expected[name][key] for key in ("name", "version", "sha256", "size", "requires_dist")), "Raw wheel changed: " + name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "lock", "inputs"))
    parser.add_argument("--run-id", default="")
    parser.add_argument("--attempt", default="1")
    parser.add_argument("--lock", type=Path, default=Path("source-lock.json"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--wheels", type=Path)
    args = parser.parse_args()
    if args.command == "preflight":
        preflight(args.run_id, args.attempt)
    elif args.command == "lock":
        check_lock(args.lock)
    else:
        verify_inputs(read_lock(args.lock), json.loads(args.report.read_text()), [inspect_wheel(path) for path in sorted(args.wheels.glob("*.whl"))])
        import torch
        original_torch = (args.report.parent / "torch.txt").read_text().strip()
        require(original_torch == f"{torch.__version__} {torch.version.cuda}", "Recovery Torch/CUDA ABI differs from original compilation")
        require("release 12.8" in (args.report.parent / "nvcc.txt").read_text(), "Original CUDA toolkit is not 12.8")
