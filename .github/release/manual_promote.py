"""Promote a CI-built candidate after explicit, immutable manual acceptance.

This is an honest manual-takeover path, not synthetic #2195 CI evidence. It does
not run Kimi training in CI, rebuild wheels, or claim public-PyPI revalidation.
"""

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import urllib.request

from four_main import save_json, sha256
from release_stack import ORDER, existing_matches, pypi_files
from release_contracts import verify_install_report, verify_release

REPO = "kvcache-ai/ktransformers"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def check_attestation(data, manifest, digest):
    require(data.get("schema") == 1 and data.get("execution") == "manual", "Require explicit manual acceptance")
    require(data.get("manifest_sha256") == digest, "Different candidate digest")
    require(data.get("source_lock") == manifest["source_lock"], "Different source lock")
    require(data.get("wheels") == manifest["wheels"], "Different accepted wheel bytes")
    require(data.get("candidate_run_id") == manifest["run_id"], "Different build run")
    require(data.get("candidate_attempt") == manifest["run_attempt"], "Different build attempt")
    require(data.get("assembly_workflow_sha", data["source_lock"].get("workflow_sha")) == manifest.get("assembly_workflow_sha", manifest.get("workflow_sha")), "Different assembler revision")
    for host in ("sap4", "qj5090"):
        for extra in ("sglang", "sglang,sft"):
            verify_install_report(data["install_reports"][host][extra], manifest, extra, public=False)
        integrity = data["runtime_integrity"][host]
        require(integrity.get("fresh_venvs") is True and integrity.get("no_upstream_namespace") is True, "Unclean installation")
        before = integrity.get("before_tooling_sha256", "")
        require(re.fullmatch(r"[0-9a-f]{64}", before) and before == integrity.get("after_tooling_sha256"), "Tooling changed runtime")
    for case, world in (("qwen3_lora", 1), ("deepseek_v31_lora", 8)):
        records = data["qj5090"][case]["records"]
        require(len(records) == world and {r["rank"] for r in records} == set(range(world)), "Missing rank evidence")
        for record in records:
            require(record["global_step"] == record["optimizer_steps"] == 1 and record["train_end"] is True, "Incomplete LoRA step")
            require(record["raw_losses"] and all(finite(v) for v in record["raw_losses"]), "Nonfinite raw loss")
    glm = data["qj5090"]["glm"]
    require(glm["completion_tokens"] > 0 and ("巴黎" in glm["content"] or "paris" in glm["content"].lower()), "GLM Q&A did not pass")
    kimi = data["sap4"]["kimi"]
    require(kimi["global_step"] >= 32 and kimi["adapter_status"] == "ready", "Kimi smoke is not completed style training")
    require(kimi["loss_records"] and all(finite(r["loss"]) and finite(r["grad_norm"]) for r in kimi["loss_records"]), "Invalid Kimi training values")
    require(set(kimi["adapter_files"]) == {"adapter_model.safetensors", "fused_expert_lora.safetensors"}, "Incomplete adapter bundle")
    for item in kimi["adapter_files"].values():
        require(item["nonzero_B_tensors"] > 0 and item["tensors"] > 0 and re.fullmatch(r"[0-9a-f]{64}", item["sha256"]), "Invalid saved LoRA")
    answers = kimi["style_answers"]
    require(len(answers) == 4 and all(a["completion_tokens"] > 0 and a["content"].strip() and "\ufffd" not in a["content"] for a in answers), "Invalid style generations")
    require(sum("喵" in a["content"] for a in answers) >= 3, "Style validation failed")
    require(kimi["heldout_overlap_count"] == 0 and kimi["baseline_completed"] is True and kimi["arithmetic_correct"] is True, "Incomplete Kimi comparison")
    require(kimi["conversion_from_main_script"] is True, "Untracked conversion implementation")


def github(path):
    request = urllib.request.Request("https://api.github.com/repos/" + REPO + path, headers={"Authorization": "Bearer " + os.environ["GH_TOKEN"], "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def fetch():
    require(os.environ.get("GITHUB_REPOSITORY") == REPO and os.environ.get("GITHUB_REF") == "refs/heads/main" and os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch", "Official main manual dispatch only")
    run_id, attempt = os.environ["CANDIDATE_RUN_ID"], os.environ["CANDIDATE_ATTEMPT"]
    commit, digest = os.environ["ACCEPTANCE_COMMIT"], os.environ["ACCEPTANCE_SHA256"]
    require(run_id.isdecimal() and attempt.isdecimal() and int(attempt) > 0, "Invalid run identity")
    require(re.fullmatch(r"[0-9a-f]{40}", commit) and re.fullmatch(r"[0-9a-f]{64}", digest), "Use immutable full hashes")
    run = github("/actions/runs/" + run_id)
    require(run["event"] == "workflow_dispatch" and run["head_branch"] == "main" and run["conclusion"] == "success", "Not a successful official main build")
    require(run["path"] == ".github/workflows/release-four-main.yml" and run["run_attempt"] == int(attempt), "Wrong build workflow or attempt")
    item = github("/contents/.github/release/manual-signoffs/post4.json?ref=" + commit)
    require(item["type"] == "file" and item["encoding"] == "base64", "Expected a small JSON acceptance record")
    content = base64.b64decode(item["content"], validate=False)
    require(hashlib.sha256(content).hexdigest() == digest, "Acceptance record changed")
    data = json.loads(content)
    require(data["candidate_run_id"] == int(run_id) and data["candidate_attempt"] == int(attempt), "Acceptance belongs to another build")
    require(data.get("assembly_workflow_sha", data["source_lock"]["workflow_sha"]) == run["head_sha"], "Different assembler revision")
    save_json(Path("accepted.json"), data)
    save_json(Path("build-run.json"), run)


def publish(root, digest, acceptance):
    manifest = verify_release(root, digest)
    data = json.loads(acceptance.read_text())
    check_attestation(data, manifest, digest)
    require(os.environ.get("GITHUB_REPOSITORY") == REPO and os.environ.get("GITHUB_REF") == "refs/heads/main" and os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch", "Official main manual dispatch only")
    require(bool(os.environ.get("TWINE_PASSWORD")), "Missing upload credential")
    # Check the whole batch before the first irreversible upload.
    for name in ORDER:
        item = manifest["wheels"][name]
        existing_matches(item, pypi_files(name, item["version"]))
    result = {"status": "uploading", "acceptance": "manual", "manifest_sha256": digest, "packages": {}}
    try:
        for name in ORDER:
            item = manifest["wheels"][name]
            path = root / "wheelhouse" / item["file"]
            require(sha256(path) == item["sha256"], "Changed wheel")
            present = existing_matches(item, pypi_files(name, item["version"]))
            if not present:
                subprocess.run([sys.executable, "-m", "twine", "upload", "--non-interactive", "--disable-progress-bar", "--repository-url", "https://upload.pypi.org/legacy/", str(path)], check=True, timeout=900)
            deadline = time.monotonic() + 600
            while not existing_matches(item, pypi_files(name, item["version"])):
                require(time.monotonic() < deadline, "PyPI visibility timeout")
                time.sleep(15)
            result["packages"][name] = item | {"already_present": present}
            save_json(Path("promotion.json"), result)
        result["status"] = "uploaded-awaiting-public-e2e"
    except BaseException:
        result["status"] = "partial-or-failed-upload"
        raise
    finally:
        save_json(Path("promotion.json"), result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "verify", "publish"))
    args = parser.parse_args()
    if args.command == "fetch":
        fetch()
    else:
        root, digest = Path("candidate"), os.environ["MANIFEST_SHA256"]
        acceptance = Path("accepted.json")
        if args.command == "verify":
            check_attestation(json.loads(acceptance.read_text()), verify_release(root, digest), digest)
        else:
            publish(root, digest, acceptance)
