# SPDX-License-Identifier: Apache-2.0
"""Launch one isolated, supervised eight-GPU FP8 experiment from a frozen stack.

Runs are append-only directories. An experiment lease prevents this harness
from overlapping its own experiments. Cleanup signals only its launcher and
marked ranks; it never deletes model weights, adapters, or previous runs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import yaml

from common import sha256, validate_training_config, write_json
from process_guard import check_support, stop_training


def command_output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=20).strip()


def resources() -> dict:
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "gpu": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,utilization.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ]
        ),
        "memory": Path("/proc/meminfo").read_text(),
        "loadavg": Path("/proc/loadavg").read_text().strip(),
    }


def trace_prefix(nsys, output, tail=False):
    """Keep short-range history reproducible; optionally capture later layers.

    Starting collection at the first async task can perturb that same interval.
    A short training tail starts at the preceding CPU backward and ignores its
    end; the already finite diagnostic process bounds collection lifetime.
    Explicit kill=none leaves training lifetime with our process supervisor.
    """
    options = [
        nsys,
        "profile",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "--capture-range=nvtx",
        "--wait=all",
        "--kill=none",
        "--force-overwrite=false",
        "--output",
        str(output),
    ]
    if not tail:
        return options + ["--nvtx-capture=repack.async@kt.sft", "--capture-range-end=stop"]
    return options + ["--nvtx-capture=cpu.backward@kt.sft", "--capture-range-end=none"]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measured", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--trace", action="store_true", help="diagnostic-only NVTX/CUDA capture of the first async repack"
    )
    parser.add_argument(
        "--trace-tail", action="store_true", help="with --trace, collect from first backward to exit; at most 3 updates"
    )
    parser.add_argument("--timeout", type=int, default=5400)
    args = parser.parse_args(argv)
    stack = json.loads(args.stack.read_text())
    variant = stack["variants"][args.variant]
    if args.trace_tail and (not args.trace or args.warmup + args.measured > 3):
        parser.error("--trace-tail requires --trace and at most 3 total optimizer updates")
    if variant.get("diagnostic_only", False) and not args.profile:
        parser.error("the frozen variant is diagnostic-only and requires --profile")
    if args.trace and not args.profile:
        parser.error("--trace requires --profile; traced runs are never formal throughput samples")
    if args.trace and not variant.get("trace_supported", False):
        parser.error("the frozen variant does not declare NVTX tracing support")
    nsys = shutil.which("nsys") if args.trace else None
    if args.trace and nsys is None:
        parser.error("--trace requires an installed nsys; the harness does not install system tools")
    check_support()  # Fail closed before launching any GPU process.
    root = Path(stack["run_root"]).resolve(strict=True)
    if Path(args.label).name != args.label or args.label in ("", ".", ".."):
        parser.error("label must be one non-empty path component")
    lease = (root / ".experiment.lock").open("a")
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    occupied = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
    )
    if occupied:
        raise RuntimeError(f"GPUs already have compute processes; not launching: {occupied}")
    deadline = datetime.fromisoformat(stack["deadline_utc"].replace("Z", "+00:00")).timestamp()
    available = deadline - time.time()
    if available < 60:
        raise RuntimeError("execution window has ended or has less than 60 seconds remaining")
    if shutil.disk_usage(root).free < 2 * 1024**3:
        raise RuntimeError("less than 2 GiB free for this isolated run")
    run = root / args.label
    run.mkdir()  # Never overwrite or reuse a result directory.
    (run / "tmp").mkdir()
    here = Path(__file__).resolve().parent
    training_config = Path(stack.get("training_config", here / "deepseek_v31.yaml"))
    config = yaml.safe_load(training_config.read_text())
    config.update(max_steps=args.warmup + args.measured, output_dir=str(run / "trainer-output"))
    validate_training_config(config, args.warmup, args.measured)
    (run / "train.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copy2(here / "accelerate_8gpu.yaml", run / "accelerate.yaml")
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "PYTHONPATH": ":".join([variant["site"], stack["packages"], stack["llamafactory_src"]]),
            "DISABLE_VERSION_CHECK": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "OMP_NUM_THREADS": "64",
            "ACCELERATE_KT_OMP_NUM_THREADS": "64",
            "KT_KERNEL_CPU_VARIANT": "avx512_bf16",
            "KT_SFT_PROFILE": "1" if args.profile else "0",
            "KT_SFT_TIMELINE": "0",
            "KT_SFT_TRACE": "1" if args.trace else "0",
            "TMPDIR": str(run / "tmp"),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "NCCL_DEBUG": "WARN",
            "WANDB_DISABLED": "true",
            "PYTHONUNBUFFERED": "1",
        }
    )
    command = [
        "numactl",
        "--interleave=all",
        stack["python"],
        "-m",
        "accelerate.commands.accelerate_cli",
        "launch",
        "--main_process_port",
        "0",
        "--config_file",
        str(run / "accelerate.yaml"),
        str(here / "train_entry.py"),
        "--config",
        str(run / "train.yaml"),
        "--run-dir",
        str(run),
        "--warmup",
        str(args.warmup),
        "--measured",
        str(args.measured),
    ]
    if args.profile:
        command.append("--profile")
    if args.trace:
        command = [*trace_prefix(nsys, run / "overlap", args.trace_tail), *command]
    checkpoint = Path(config["model_name_or_path"])
    index = checkpoint / "model.safetensors.index.json"
    model_index = json.loads(index.read_text())
    shards = sorted(set(model_index["weight_map"].values()))
    shard_metadata = {
        name: {
            "size": (checkpoint / name).stat().st_size,
            "mtime_ns": (checkpoint / name).stat().st_mtime_ns,
        }
        for name in shards
    }
    fixture = Path(config["tokenized_path"])
    provenance = {
        "variant": args.variant,
        "variant_spec": variant,
        "stack": stack,
        "command": command,
        "warmup": args.warmup,
        "measured": args.measured,
        "profile": args.profile,
        "trace": args.trace,
        "trace_tail": args.trace_tail,
        "nsys_version": command_output([nsys, "--version"]) if args.trace else None,
        "initial_resources": resources(),
        "checkpoint_config_sha256": sha256(checkpoint / "config.json"),
        "checkpoint_index_sha256": sha256(index),
        "checkpoint_shard_metadata_not_content_hashes": shard_metadata,
        "fixture_file_sha256": {
            str(p.relative_to(fixture)): sha256(p) for p in sorted(fixture.rglob("*")) if p.is_file()
        },
        "harness_file_sha256": {p.name: sha256(p) for p in sorted(here.iterdir()) if p.is_file()},
    }
    write_json(run / "provenance.json", provenance)
    process = None
    status = {"status": "STARTING", "run_dir": str(run)}
    write_json(run / "status.json", status)
    print(json.dumps(status), flush=True)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"run received signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        with (run / "train.log").open("w") as log, (run / "resources.jsonl").open("w") as resource_log:
            process = subprocess.Popen(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            status.update(status="RUNNING", pid=process.pid, started_utc=datetime.now(timezone.utc).isoformat())
            write_json(run / "status.json", status)
            limit = time.monotonic() + min(args.timeout, deadline - time.time())
            while process.poll() is None:
                snapshot = resources()
                snapshot["disk_free_bytes"] = shutil.disk_usage(root).free
                resource_log.write(json.dumps(snapshot) + "\n")
                resource_log.flush()
                if snapshot["disk_free_bytes"] < 1024**3:
                    raise RuntimeError("disk free fell below the 1 GiB safety floor")
                if time.monotonic() >= limit:
                    raise TimeoutError("run timeout or user execution-window deadline reached")
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            if process.returncode != 0:
                raise RuntimeError(f"training launcher exited with code {process.returncode}")
        windows = [json.loads((run / f"window.rank{rank}.json").read_text()) for rank in range(8)]
        if any(window != windows[0] for window in windows[1:]):
            raise RuntimeError("ranks disagree on the measured training window")
        for rank in range(8):
            identity = json.loads((run / f"identity.rank{rank}.json").read_text())
            if identity["extension_sha256"] != variant["extension_sha256"]:
                raise RuntimeError(f"rank {rank} loaded a different extension than the frozen variant")
            if not (run / f"final.rank{rank}.json").is_file():
                raise RuntimeError(f"rank {rank} did not finish training validation")
        if args.trace and not (run / "overlap.nsys-rep").is_file():
            raise RuntimeError("training completed but the requested NVTX capture produced no report")
        status.update(status="PASS", metrics=windows[0])
    except BaseException as error:
        status.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        try:
            if process is not None:
                status["cleanup_actions"] = stop_training(process, here / "train_entry.py", run)
                if status["status"] == "PASS" and status["cleanup_actions"]:
                    raise RuntimeError("launcher reported success but owned processes still required termination")
        except BaseException as error:
            status.update(status="FAIL", cleanup_error=f"{type(error).__name__}: {error}")
            raise
        finally:
            if process is not None:
                status["exit_code"] = process.returncode
            try:
                status["finished_utc"] = datetime.now(timezone.utc).isoformat()
                write_json(run / "status.json", status)
                print(json.dumps(status), flush=True)
            finally:
                signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
