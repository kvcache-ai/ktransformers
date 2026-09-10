# SPDX-License-Identifier: Apache-2.0
"""Mode errors must fail before resource checks or external process creation."""

import json
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default")

RUNNER = Path(__file__).resolve().parents[2] / "bench" / "sft_perf" / "run.py"
sys.path.insert(0, str(RUNNER.parent))
try:
    spec = importlib.util.spec_from_file_location("sft_perf_run", RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
finally:
    sys.path.pop(0)


def test_diagnostic_variant_cannot_launch_a_formal_run(tmp_path):
    stack = tmp_path / "stack.json"
    stack.write_text(json.dumps({"variants": {"candidate": {"diagnostic_only": True}}}))
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--stack",
            str(stack),
            "--variant",
            "candidate",
            "--label",
            "test",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "diagnostic-only and requires --profile" in completed.stderr


def test_trace_requires_profile_before_resource_access(tmp_path):
    stack = tmp_path / "stack.json"
    stack.write_text(json.dumps({"variants": {"candidate": {}}}))
    completed = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--stack",
            str(stack),
            "--variant",
            "candidate",
            "--label",
            "test",
            "--trace",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert "--trace requires --profile" in completed.stderr


def test_tail_capture_keeps_application_lifetime_with_supervisor():
    prefix = runner.trace_prefix("nsys", Path("/new/report"), tail=True)
    assert "--capture-range-end=none" in prefix
    assert not any(option.startswith("--duration") for option in prefix)
    assert "--nvtx-capture=cpu.backward@kt.sft" in prefix
    assert "--kill=none" in prefix and "--wait=all" in prefix
    previous = runner.trace_prefix("nsys", "/previous/report")
    assert "--capture-range-end=stop" in previous
    assert "--nvtx-capture=repack.async@kt.sft" in previous
    assert not any(option.startswith("--duration") for option in previous)


def test_tail_capture_requires_trace_and_a_short_run_before_resource_access(tmp_path):
    stack = tmp_path / "stack.json"
    stack.write_text(json.dumps({"variants": {"candidate": {}}}))
    for options in ([], ["--trace", "--warmup", "2", "--measured", "2"]):
        completed = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--stack",
                str(stack),
                "--variant",
                "candidate",
                "--label",
                "test",
                "--trace-tail",
                *options,
            ],
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 2
        assert "requires --trace and at most 3 total optimizer updates" in completed.stderr


def test_pidfd_capability_failure_happens_before_resource_access(tmp_path, monkeypatch):
    stack = tmp_path / "stack.json"
    stack.write_text(json.dumps({"variants": {"candidate": {}}}))

    def unsupported():
        raise RuntimeError("test: missing pidfd support")

    monkeypatch.setattr(runner, "check_support", unsupported)
    with pytest.raises(RuntimeError, match="missing pidfd support"):
        runner.main(["--stack", str(stack), "--variant", "candidate", "--label", "test"])


def test_low_disk_refuses_launch_without_creating_a_run(tmp_path, monkeypatch):
    stack = tmp_path / "stack.json"
    stack.write_text(
        json.dumps({"variants": {"candidate": {}}, "run_root": str(tmp_path), "deadline_utc": "2099-01-01T00:00:00Z"})
    )
    monkeypatch.setattr(runner, "check_support", lambda: None)
    monkeypatch.setattr(runner, "command_output", lambda command: "")
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda path: SimpleNamespace(free=1024**3))

    def forbidden_launch(*args, **kwargs):
        pytest.fail("low disk must refuse before launching a process")

    monkeypatch.setattr(runner.subprocess, "Popen", forbidden_launch)
    with pytest.raises(RuntimeError, match="less than 2 GiB"):
        runner.main(["--stack", str(stack), "--variant", "candidate", "--label", "test"])
    assert not (tmp_path / "test").exists()


@pytest.mark.parametrize("cleanup_fails", [False, True])
@pytest.mark.parametrize("launcher_success", [False, True])
@pytest.mark.parametrize("external_config", [False, True])
def test_failed_run_persists_status_even_if_cleanup_also_fails(
    tmp_path, monkeypatch, cleanup_fails, launcher_success, external_config
):
    # Only exercise supervisor control flow. No CUDA/framework subprocess runs.
    here = tmp_path / "harness"
    here.mkdir()
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    config = {
        "stage": "sft",
        "do_train": True,
        "do_eval": False,
        "finetuning_type": "lora",
        "use_kt": True,
        "include_num_input_tokens_seen": "non_padding",
        "save_strategy": "no",
        "logging_nan_inf_filter": False,
        "kt_config": {"kt_expert_weight_format": "fp8"},
        "model_name_or_path": str(checkpoint),
        "tokenized_path": str(fixture),
    }
    config_path = tmp_path / "training.yaml" if external_config else here / "deepseek_v31.yaml"
    config_path.write_text(yaml.safe_dump(config))
    (here / "accelerate_8gpu.yaml").write_text("{}")
    stack = tmp_path / "stack.json"
    stack_config = {
        "variants": {"candidate": {"site": "test", "extension_sha256": "test"}},
        "run_root": str(tmp_path),
        "deadline_utc": "2099-01-01T00:00:00Z",
        "packages": "test",
        "llamafactory_src": "test",
        "python": "test",
    }
    if external_config:
        stack_config["training_config"] = str(config_path)
    stack.write_text(json.dumps(stack_config))
    monkeypatch.setattr(runner, "__file__", str(here / "run.py"))
    monkeypatch.setattr(runner, "check_support", lambda: None)
    monkeypatch.setattr(runner, "command_output", lambda command: "")
    monkeypatch.setattr(runner, "resources", lambda: {})
    # This test targets shutdown failures, not the test host's free-space level.
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda path: SimpleNamespace(free=4 * 1024**3))

    class FailedLauncher:
        pid = 123
        returncode = 0 if launcher_success else 42

        def poll(self):
            return self.returncode

    process = FailedLauncher()

    def launch(*a, **kw):
        if launcher_success:
            for rank in range(8):
                run_dir = tmp_path / "test"
                (run_dir / f"window.rank{rank}.json").write_text("{}")
                (run_dir / f"identity.rank{rank}.json").write_text('{"extension_sha256": "test"}')
                (run_dir / f"final.rank{rank}.json").write_text("{}")
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    cleaned = []

    def cleanup(actual, entrypoint, run_dir):
        cleaned.append((actual, entrypoint, run_dir))
        if cleanup_fails:
            raise RuntimeError("test cleanup failure")
        return [{"pid": 234, "signal": 15}] if launcher_success else []

    monkeypatch.setattr(runner, "stop_training", cleanup)
    previous = runner.signal.getsignal(runner.signal.SIGTERM)
    message = "cleanup failure" if cleanup_fails else ("still required termination" if launcher_success else "code 42")
    with pytest.raises(RuntimeError, match=message):
        runner.main(["--stack", str(stack), "--variant", "candidate", "--label", "test"])
    status = json.loads((tmp_path / "test" / "status.json").read_text())
    assert status["status"] == "FAIL" and status["exit_code"] == process.returncode
    assert status["finished_utc"]
    if not launcher_success:
        assert "code 42" in status["error"]
    assert ("cleanup_error" in status) is (cleanup_fails or launcher_success)
    assert cleaned == [(process, here / "train_entry.py", tmp_path / "test")]
    assert runner.signal.getsignal(runner.signal.SIGTERM) == previous


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
