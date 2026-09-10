"""Trusted host controller. Candidate packages execute only inside the container."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import stat
import subprocess
import uuid
from pathlib import Path

from contracts import (
    digest,
    require,
    suite_passed,
    validate_request,
    verify_candidate,
    write_json,
)
from release_contracts import verify_release
from resource_queue import ResourceUnavailable, reservation


def export_evidence(source, destination):
    """Copy logs only AFTER container removal; never follow candidate symlinks."""
    require(source.is_dir() and not source.is_symlink(), "Unsafe evidence directory")
    total = 0
    for current, directories, files in os.walk(source, followlinks=False):
        for name in directories:
            require(not (Path(current) / name).is_symlink(), "Symlink in evidence")
        for name in files:
            path = Path(current) / name
            mode = path.lstat().st_mode
            require(
                stat.S_ISREG(mode) and not path.is_symlink(),
                "Non-regular evidence file",
            )
            total += path.stat().st_size
            require(total <= 256 * 1024 * 1024, "Evidence exceeds 256 MiB")
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def validate_config(config):
    require(config["schema"] == 1, "Unsupported host configuration")
    require(
        re.fullmatch(r"[a-zA-Z0-9./:_-]+@sha256:[0-9a-f]{64}", config["image"]),
        "Pin the approved image by digest",
    )
    require(
        set(config["models"]) == {"qwen3", "deepseek_v31", "glm53"},
        "Require three model snapshots",
    )
    for model in config["models"].values():
        require(
            re.fullmatch(r"[0-9a-f]{64}", model["config_sha256"]),
            "Pin each model config hash",
        )
        require(
            model["revision"] and model["revision"] != "REPLACE",
            "Record each model revision",
        )
        path = Path(model["path"]).resolve(strict=True)
        require(
            path.is_dir() and len(path.parts) >= 4,
            "Model mount must be a specific directory",
        )
        require(
            digest(path / "config.json") == model["config_sha256"],
            "Model config changed",
        )
    require(0 < config["queue_timeout_seconds"] <= 21600, "Queue timeout out of range")
    require(0 < config["test_timeout_seconds"] <= 43200, "Test timeout out of range")
    return config


def docker_command(config, harness, inputs, output, name):
    # Do not pass host HOME, environment, credentials, caches, Docker socket,
    # host network, privileged mode or writable model mounts to tested code.
    # Rootless container uid 0 maps to the unprivileged daemon account on the
    # host. Using host uid 1000 *inside* that user namespace breaks bind writes.
    uid, gid = (0, 0) if config.get("rootless", False) else (os.getuid(), os.getgid())
    command = [
        "docker",
        "create",
        "--name",
        name,
        "--pull=never",
        "--no-healthcheck",
        "--init",
        "--user",
        f"{uid}:{gid}",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--read-only",
        "--device=nvidia.com/gpu=all",
        "--pids-limit=8192",
        "--shm-size=16g",
        "--tmpfs",
        "/tmp:rw,nosuid,size=8g",
        "--tmpfs",
        f"/work:rw,exec,nosuid,size=96g,uid={uid},gid={gid}",
        "--workdir",
        "/work",
        "--mount",
        f"type=bind,src={harness},dst=/harness,readonly",
        "--mount",
        f"type=bind,src={inputs},dst=/input,readonly",
        "--mount",
        f"type=bind,src={output / 'evidence'},dst=/work/evidence",
    ]
    for key, model in config["models"].items():
        command += [
            "--mount",
            f"type=bind,src={Path(model['path']).resolve()},dst=/models/{key},readonly",
        ]
    command += [
        "--env",
        "HOME=/work/home",
        "--env",
        "PYTHONNOUSERSITE=1",
        "--env",
        "PYTHONUNBUFFERED=1",
        "--env",
        "PIP_CONFIG_FILE=/dev/null",
        "--env",
        "NVIDIA_VISIBLE_DEVICES=void",
        "--entrypoint",
        "python3",
        config["image"],
        "/harness/inside.py",
    ]
    return command


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt("CI cancellation: clean up only the owned container")

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(exist_ok=False)
    evidence = output / "evidence"
    evidence.mkdir()
    sandbox = output / "sandbox"
    sandbox.mkdir()
    (sandbox / "evidence").mkdir()
    results = {"status": "infrastructure_error", "cases": {}}
    name = "kt-model-e2e-" + uuid.uuid4().hex
    try:
        config = validate_config(json.loads(args.config.read_text()))
        security = json.loads(
            subprocess.check_output(
                ["docker", "info", "--format", "{{json .SecurityOptions}}"],
                text=True,
                timeout=30,
            )
        )
        config["rootless"] = "name=rootless" in security
        require(
            config["rootless"] or os.getuid() != 0,
            "Do not run a rootful controller as root",
        )
        request = validate_request(
            json.loads((args.input / "request.json").read_text())
        )
        if request["mode"] == "pr":
            verify_candidate(args.input / "candidate", request)
        elif request["mode"].startswith("release-"):
            manifest = verify_release(
                args.input / "release", request["manifest_sha256"]
            )
            require(
                manifest["run_id"] == request["build_run_id"], "Release run mismatch"
            )
            require(
                manifest["run_attempt"] == request["build_run_attempt"],
                "Release attempt mismatch",
            )
            require(
                manifest["workflow_sha"] == request["build_workflow_sha"],
                "Release workflow mismatch",
            )
        write_json(evidence / "host-config.json", config)
        write_json(evidence / "request.json", request)
        # The lock covers create, package installation, all three tests, AND
        # container removal. Queueing itself does not allocate GPU memory.
        with reservation(
            config["lock_path"],
            evidence / "queue.jsonl",
            timeout=config["queue_timeout_seconds"],
        ):
            created = False
            try:
                command = docker_command(
                    config,
                    Path(__file__).resolve().parent,
                    args.input.resolve(),
                    sandbox,
                    name,
                )
                subprocess.run(
                    command, check=True, timeout=60, stdout=subprocess.DEVNULL
                )
                created = True
                with (evidence / "container.log").open("w") as log:
                    completed = subprocess.run(
                        ["docker", "start", "--attach", name],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=config["test_timeout_seconds"],
                    )
                # docker start's return code alone is not sufficient evidence.
                state = json.loads(
                    subprocess.check_output(
                        ["docker", "inspect", "--format", "{{json .State}}", name],
                        text=True,
                    )
                )
            finally:
                if created:
                    # This exact random name belongs to this invocation only.
                    subprocess.run(
                        ["docker", "rm", "--force", name],
                        check=True,
                        timeout=120,
                        stdout=subprocess.DEVNULL,
                    )
                    if (sandbox / "evidence").exists():
                        export_evidence(sandbox / "evidence", evidence / "tests")
            if (evidence / "tests/suite.json").exists():
                results = json.loads((evidence / "tests/suite.json").read_text())
                if results["status"] == "resource_unavailable":
                    raise ResourceUnavailable(
                        "A manual workload arrived before a model test; queue wait expired"
                    )
            require(
                completed.returncode == 0
                and state["ExitCode"] == 0
                and not state["OOMKilled"],
                "Container test failed",
            )
            require(
                results["status"] == "passed" and suite_passed(results["cases"]),
                "Model suite did not pass",
            )
    except ResourceUnavailable as exc:
        results.update(status="resource_unavailable", error=str(exc))
        raise
    except Exception as exc:
        results.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(evidence / "host-result.json", results)


if __name__ == "__main__":
    main()
