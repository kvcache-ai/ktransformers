# SPDX-License-Identifier: Apache-2.0
"""Linux process tests use only short-lived, test-owned CPU sleepers."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default")

BENCH = Path(__file__).resolve().parents[2] / "bench" / "sft_perf"
spec = importlib.util.spec_from_file_location("sft_process_guard", BENCH / "process_guard.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_exact_run_marker_never_matches_a_substring_or_another_entrypoint():
    assert guard.matches_run(["python", "/entry.py", "--run-dir", "/run"], "/entry.py", "/run")
    for arguments in (
        ["/entry.py", "--run-dir", "/run-other"],
        ["/other.py", "--run-dir", "/run"],
        ["bash", "-c", "python /entry.py --run-dir /run"],
        ["/entry.py", "--run-dir"],
        ["/entry.py", "--run-dir", "/run", "--run-dir", "/run"],
    ):
        assert not guard.matches_run(arguments, "/entry.py", "/run")


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only harness")
@pytest.mark.parametrize("use_syscalls", [False, True])
def test_pidfd_probe_and_start_identity(monkeypatch, use_syscalls):
    if use_syscalls:
        monkeypatch.delattr(os, "pidfd_open", raising=False)
        monkeypatch.delattr(signal, "pidfd_send_signal", raising=False)
    guard.check_support()
    pid, start = guard._identity(Path("/proc") / str(os.getpid()))
    assert pid == os.getpid() and start > 0


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only harness")
def test_own_unreaped_launcher_is_owned_even_without_a_run_marker(tmp_path):
    guard.check_support()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(15)"], start_new_session=True)
    try:
        actions = guard.stop_training(process, tmp_path / "entry.py", tmp_path / "owned-run", grace_seconds=0.1)
        assert process.returncode < 0
        assert any(row["pid"] == process.pid for row in actions)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux-only harness")
@pytest.mark.parametrize("ignore_term", [False, True])
@pytest.mark.parametrize("use_syscalls", [False, True])
def test_detached_rank_survives_launcher_but_is_owned_not_its_sibling(tmp_path, monkeypatch, ignore_term, use_syscalls):
    if use_syscalls:
        monkeypatch.delattr(os, "pidfd_open", raising=False)
        monkeypatch.delattr(signal, "pidfd_send_signal", raising=False)
    guard.check_support()
    entry = tmp_path / "entry.py"
    entry.write_text(
        "import json,os,signal,subprocess,sys,time\n"
        "if '--child' not in sys.argv:\n"
        " p=subprocess.Popen([sys.executable,__file__,'--child',*sys.argv[1:]],start_new_session=True)\n"
        " print(json.dumps({'child':p.pid}),flush=True)\n"
        "else:\n"
        " if '--ignore-term' in sys.argv: signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        " print('ready',flush=True)\n"
        " time.sleep(15)\n"
    )
    run_dir = tmp_path / "owned-run"
    arguments = [sys.executable, str(entry), "--run-dir", str(run_dir)]
    if ignore_term:
        arguments.append("--ignore-term")
    outsider = subprocess.Popen(
        [sys.executable, str(entry), "--child", "--run-dir", str(run_dir) + "-other"],
        stdout=subprocess.DEVNULL,
        start_new_session=True,
    )
    launcher = subprocess.Popen(arguments, stdout=subprocess.PIPE, text=True, start_new_session=True)
    child = None
    try:
        # Both lines can arrive in either order; neither waits on a GPU.
        lines = [launcher.stdout.readline().strip(), launcher.stdout.readline().strip()]
        child = next(json.loads(line)["child"] for line in lines if line.startswith("{"))
        assert "ready" in lines
        launcher.wait(timeout=5)
        assert os.getsid(child) != launcher.pid
        actions = guard.stop_training(launcher, entry, run_dir, grace_seconds=0.1)
        assert any(row["pid"] == child for row in actions)
        assert outsider.poll() is None
        if ignore_term:
            assert any(row["signal"] == 9 for row in actions)
    finally:
        guard.stop_training(launcher, entry, run_dir, grace_seconds=0.1)
        launcher.stdout.close()
        outsider.terminate()
        outsider.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
