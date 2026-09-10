"""Fresh wheel installation and serial model tests; runs inside the sandbox only."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import venv
from pathlib import Path

from contracts import (
    CASES,
    PACKAGES,
    digest,
    require,
    suite_passed,
    validate_answer,
    validate_lora,
    validate_request,
    verify_candidate,
    write_json,
)
from recipes import accelerate_config, glm_command, training_config
from resource_queue import ResourceUnavailable, reservation

WORK = Path("/work")
EVIDENCE = WORK / "evidence"
PYTHON = WORK / "venv/bin/python"


def run(command, log, **kwargs):
    with Path(log).open("w") as stream:
        subprocess.run(
            [str(value) for value in command],
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            **kwargs,
        )


def terminate_group(process):
    # The group was created with start_new_session=True by this harness. Never
    # pkill by model name, PID pattern, port, user, or nvidia-smi output.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30)
    # The parent may exit before its workers; kill only this owned group.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def environment():
    env = dict(os.environ)
    for key in list(env):
        if key.startswith(
            ("PIP_", "PYTHON", "ACCELERATE_", "KT_", "HF_", "TRANSFORMERS_")
        ) or key in ("VIRTUAL_ENV", "USE_KT"):
            del env[key]
    env.update(
        {
            "PATH": str(PYTHON.parent) + ":" + env.get("PATH", "/usr/bin:/bin"),
            "HOME": "/work/home",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUNBUFFERED": "1",
            "PIP_CONFIG_FILE": "/dev/null",
            "PIP_NO_CACHE_DIR": "1",
            "HF_HOME": "/work/cache/hf",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "XDG_CACHE_HOME": "/work/cache/xdg",
            "TRITON_CACHE_DIR": "/work/cache/triton",
            "TORCH_EXTENSIONS_DIR": "/work/cache/torch-extensions",
            "CUDA_CACHE_PATH": "/work/cache/cuda",
            "OMP_NUM_THREADS": "64",
            "TOKENIZERS_PARALLELISM": "false",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        }
    )
    return env


def pip_args():
    return [
        PYTHON,
        "-m",
        "pip",
        "--isolated",
        "install",
        "--index-url",
        "https://pypi.org/simple",
        "--no-cache-dir",
    ]


def install(request, env):
    # No --system-site-packages, no editable installation, no host venv/cache.
    require(
        sys.version_info[:2] == (3, 12),
        "This initial GPU acceptance matrix requires CPython 3.12; do not silently resolve an older stack",
    )
    require(not (WORK / "venv").exists(), "Refusing to reuse an environment")
    venv.EnvBuilder(with_pip=True).create(WORK / "venv")
    report = EVIDENCE / "pip-install.json"
    if request["mode"] == "pypi":
        spec = "ktransformers[sglang,sft]"
        if request["version"] != "latest":
            spec += "==" + request["version"]
        requirements = [spec]
    else:
        manifest = verify_candidate("/input/candidate", request)
        requirements = []
        for package, entry in sorted(manifest["wheels"].items()):
            path = Path("/input/candidate") / entry["file"]
            if package == "ktransformers":
                requirements.append("ktransformers[sglang,sft] @ " + path.as_uri())
            else:
                requirements.append(str(path))
    run(
        pip_args() + ["--only-binary=:all:", "--report", report] + requirements,
        EVIDENCE / "install.log",
        env=env,
        timeout=7200,
    )
    run(
        [PYTHON, "-m", "pip", "check"], EVIDENCE / "pip-check.txt", env=env, timeout=120
    )
    run(
        [PYTHON, "-m", "pip", "freeze", "--all"],
        EVIDENCE / "pip-freeze.txt",
        env=env,
        timeout=120,
    )
    # These imports run inside the newly installed interpreter, not the host.
    run(
        [PYTHON, "/harness/installed.py", EVIDENCE / "installed.json"],
        EVIDENCE / "imports.log",
        env=env,
        timeout=300,
    )
    installed = json.loads((EVIDENCE / "installed.json").read_text())
    if request["mode"] == "pr":
        for package, entry in manifest["wheels"].items():
            require(
                installed[package]["version"] == entry["version"],
                "Installed version differs from candidate",
            )
    else:
        require(
            installed["ktransformers"]["version"] == request["expected_version"],
            "pip selected a different KT version than the requested PyPI snapshot",
        )
        entries = json.loads(report.read_text())["install"]
        entries = {
            item["metadata"]["name"].lower().replace("_", "-"): item for item in entries
        }
        for package in PACKAGES:
            url = entries[package]["download_info"]["url"]
            require(
                url.startswith("https://files.pythonhosted.org/"),
                "PyPI test installed a non-PyPI stack package",
            )


def install_training_tools(env):
    # The approved digest-pinned image provides a hash-locked PUBLIC tooling
    # wheelhouse. It must NOT contain any of the five tested distributions.
    # Preparation of that image is an explicit activation prerequisite.
    root = Path("/opt/kt-e2e-tooling")
    requirements = root / "requirements.txt"
    require(
        requirements.is_file(),
        "Provision the reviewed SFT tooling wheelhouse before enabling CI",
    )
    write_json(
        EVIDENCE / "tooling.json",
        {
            "requirements_sha256": digest(requirements),
            "provenance": json.loads((root / "provenance.json").read_text()),
        },
    )
    installed = json.loads((EVIDENCE / "installed.json").read_text())
    constraint = WORK / "stack-constraints.txt"
    constraint.write_text(
        "".join(f"{name}=={entry['version']}\n" for name, entry in installed.items())
    )
    command = [
        PYTHON,
        "-m",
        "pip",
        "--isolated",
        "install",
        "--no-index",
        "--find-links",
        root / "wheels",
        "--only-binary=:all:",
        "--require-hashes",
        "--constraint",
        constraint,
        "--requirement",
        requirements,
    ]
    plan = EVIDENCE / "tooling-plan.json"
    run(
        command + ["--dry-run", "--report", plan],
        EVIDENCE / "tooling-resolve.log",
        env=env,
        timeout=600,
    )
    forbidden = PACKAGES | {
        "transformers",
        "accelerate",
        "sglang",
        "sgl-kernel",
        "sgl-kernel-kt",
        "torch",
    }
    for item in json.loads(plan.read_text())["install"]:
        name = item["metadata"]["name"].lower().replace("_", "-")
        require(
            name not in forbidden,
            f"Training tooling would replace/overlay tested runtime: {name}",
        )
    run(command, EVIDENCE / "tooling-install.log", env=env, timeout=600)
    run(
        [PYTHON, "-m", "pip", "check"],
        EVIDENCE / "tooling-pip-check.txt",
        env=env,
        timeout=120,
    )
    run(
        [PYTHON, "/harness/installed.py", EVIDENCE / "installed-after-tooling.json"],
        EVIDENCE / "tooling-imports.log",
        env=env,
        timeout=300,
    )
    require(
        json.loads((EVIDENCE / "installed-after-tooling.json").read_text())
        == installed,
        "Tooling changed the tested stack",
    )


def lora(case, env):
    world_size = 8 if case == "deepseek_v31_lora" else 1
    output = WORK / case
    output.mkdir(exist_ok=False)
    evidence = EVIDENCE / case
    evidence.mkdir()
    write_json(evidence / "train.json", training_config(case, output))
    # JSON is valid YAML; Accelerate's config parser accepts this file.
    write_json(evidence / "accelerate.yaml", accelerate_config(world_size))
    ranks = evidence / "ranks"
    ranks.mkdir()
    case_env = env | {
        "KT_E2E_RANK_EVIDENCE": str(ranks),
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, range(world_size))),
    }
    command = [
        PYTHON,
        "-m",
        "accelerate.commands.launch",
        "--main_process_port",
        "0",
        "--config_file",
        evidence / "accelerate.yaml",
        "/harness/lora_probe.py",
        evidence / "train.json",
    ]
    with (evidence / "train.log").open("w") as log:
        process = subprocess.Popen(
            [str(value) for value in command],
            env=case_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            require(
                process.wait(timeout=10800) == 0,
                "LoRA process did not exit successfully",
            )
        finally:
            terminate_group(process)
    return validate_lora(
        [json.loads(path.read_text()) for path in sorted(ranks.glob("rank-*.json"))],
        world_size,
    )


def glm(env):
    log = EVIDENCE / "glm-server.log"
    with log.open("w") as stream:
        process = subprocess.Popen(
            glm_command(PYTHON),
            env=env | {"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 3600
            while True:
                require(process.poll() is None, "GLM server exited before readiness")
                try:
                    with urllib.request.urlopen(
                        "http://127.0.0.1:30000/health", timeout=10
                    ) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                require(time.monotonic() < deadline, "GLM readiness timeout")
                time.sleep(10)
            payload = {
                "model": "GLM-5.3-Flash",
                "temperature": 0,
                "max_tokens": 512,
                "messages": [
                    {
                        "role": "user",
                        "content": "What is the capital of France? Answer with the city name only.",
                    }
                ],
            }
            write_json(EVIDENCE / "glm-request.json", payload)
            request = urllib.request.Request(
                "http://127.0.0.1:30000/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=600) as response:
                result = json.load(response)
            write_json(EVIDENCE / "glm-response.json", result)
            require(process.poll() is None, "Server died after request")
            return validate_answer(result)
        finally:
            terminate_group(process)


def main():
    EVIDENCE.mkdir(exist_ok=True)
    (WORK / "home").mkdir(exist_ok=True)
    results = {"status": "failed", "cases": {}}
    try:
        request = validate_request(json.loads(Path("/input/request.json").read_text()))
        env = environment()
        install(request, env)
        install_training_tools(env)
        write_json(
            WORK / "data/dataset_info.json",
            {"kt_ci_smoke": {"file_name": "smoke.json"}},
        )
        write_json(
            WORK / "data/smoke.json",
            [
                {
                    "instruction": "What is two plus two?",
                    "input": "",
                    "output": "Two plus two is four.",
                },
                {
                    "instruction": "Name the capital of France.",
                    "input": "",
                    "output": "Paris is the capital of France.",
                },
            ]
            * 8,
        )
        for case in CASES:
            try:
                # Installation may take a while. Recheck ALL GPUs before each
                # model as well, in case a manual workload arrived meanwhile.
                # The outer host reservation is still held throughout.
                with reservation(
                    WORK / "model-start.lock", EVIDENCE / "model-start-queue.jsonl"
                ):
                    results["cases"][case] = (
                        glm(env) if case == "glm53_inference" else lora(case, env)
                    )
            except ResourceUnavailable as exc:
                results["status"] = "resource_unavailable"
                results["cases"][case] = {"status": "not_run", "error": str(exc)}
                raise
            except Exception as exc:
                results["cases"][case] = {"status": "failed", "error": str(exc)}
                raise  # fail fast; no missing case can produce a green suite
            finally:
                write_json(EVIDENCE / "suite.json", results)
        require(suite_passed(results["cases"]), "Incomplete acceptance suite")
        results["status"] = "passed"
    except Exception as exc:
        results["error"] = str(exc)
        raise
    finally:
        write_json(EVIDENCE / "suite.json", results)


if __name__ == "__main__":
    main()
