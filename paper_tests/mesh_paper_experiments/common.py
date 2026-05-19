#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Iterable


PREFILL_RE = re.compile(
    r"Prefill batch, .*?#new-token: (?P<tokens>\d+), .*?"
    r"input throughput \(token/s\): (?P<tps>[0-9.]+)"
)
DECODE_RE = re.compile(
    r"Decode batch, .*?gen throughput \(token/s\): (?P<tps>[0-9.]+)"
)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path = Path(path)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def format_template(template: str, **kwargs: Any) -> str:
    safe = {k: str(v) for k, v in kwargs.items()}
    return template.format(**safe)


def run_command(
    cmd: str,
    run_dir: str | Path,
    *,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    prefix: str = "command",
) -> dict[str, Any]:
    run_dir = ensure_dir(run_dir)
    stdout_path = run_dir / f"{prefix}.stdout.txt"
    stderr_path = run_dir / f"{prefix}.stderr.txt"
    meta_path = run_dir / f"{prefix}.meta.json"
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)

    start = time.time()
    with stdout_path.open("w", encoding="utf-8", errors="replace") as out, stderr_path.open(
        "w", encoding="utf-8", errors="replace"
    ) as err:
        proc = subprocess.Popen(
            cmd,
            shell=True,
            stdout=out,
            stderr=err,
            text=True,
            env=merged_env,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        timed_out = False
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                proc.terminate()
            try:
                rc = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except Exception:
                    proc.kill()
                rc = proc.wait()
    end = time.time()
    meta = {
        "cmd": cmd,
        "argv_preview": shlex.split(cmd)[:16],
        "returncode": rc,
        "timed_out": timed_out,
        "start_wall": start,
        "end_wall": end,
        "elapsed_s": end - start,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    write_json(meta_path, meta)
    return meta


def parse_runtime_log_text(text: str) -> dict[str, Any]:
    prefill_events: list[dict[str, Any]] = []
    decode_events: list[dict[str, Any]] = []
    for line in text.splitlines():
        if "Prefill batch" in line:
            m = PREFILL_RE.search(line)
            if m:
                tokens = int(m.group("tokens"))
                tps = float(m.group("tps"))
                prefill_events.append({"tokens": tokens, "tps": tps, "line": line})
        elif "Decode batch" in line:
            m = DECODE_RE.search(line)
            if m:
                decode_events.append({"tps": float(m.group("tps")), "line": line})

    usable_prefill = [(e["tokens"], e["tps"]) for e in prefill_events if e["tokens"] > 0 and e["tps"] > 0]
    prefill_tps = None
    if usable_prefill:
        total_tokens = sum(t for t, _ in usable_prefill)
        total_s = sum(t / s for t, s in usable_prefill if s > 0)
        prefill_tps = total_tokens / total_s if total_s > 0 else None

    decode_values = [e["tps"] for e in decode_events if e["tps"] > 0]
    return {
        "prefill_events": prefill_events,
        "decode_events": decode_events,
        "prefill_tokens_logged": sum(e["tokens"] for e in prefill_events),
        "prefill_tps_hmean": prefill_tps,
        "prefill_tps_last": prefill_events[-1]["tps"] if prefill_events else None,
        "decode_tps_avg": sum(decode_values) / len(decode_values) if decode_values else None,
        "decode_tps_last": decode_values[-1] if decode_values else None,
    }


def parse_runtime_logs(paths: Iterable[str | Path]) -> dict[str, Any]:
    text = ""
    for path in paths:
        p = Path(path)
        if p.exists():
            text += "\n" + p.read_text(encoding="utf-8", errors="replace")
    return parse_runtime_log_text(text)


class IostatCapture:
    def __init__(self, out_path: str | Path, interval_s: int = 1, devices: list[str] | None = None):
        self.out_path = Path(out_path)
        self.interval_s = max(1, int(interval_s))
        self.devices = devices or []
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.proc: subprocess.Popen[str] | None = None

    def __enter__(self) -> "IostatCapture":
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        time.sleep(1.1)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=5)
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except Exception:
                self.proc.terminate()

    def _run(self) -> None:
        cmd = ["iostat", "-dxm", str(self.interval_s)]
        if self.devices:
            cmd += self.devices
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )
        with self.out_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"event": "iostat_start", "time": time.time(), "cmd": cmd}) + "\n")
            while not self.stop_event.is_set():
                line = self.proc.stdout.readline() if self.proc.stdout is not None else ""
                if not line:
                    if self.proc.poll() is not None:
                        break
                    time.sleep(0.05)
                    continue
                f.write(json.dumps({"time": time.time(), "line": line.rstrip("\n")}) + "\n")
                f.flush()


def parse_iostat_jsonl(path: str | Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    header: list[str] | None = None
    for obj in iter_jsonl(path):
        line = obj.get("line", "")
        if not line:
            continue
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "Device":
            header = parts
            continue
        if header is None or len(parts) < len(header):
            continue
        if parts[0].startswith(("nvme", "dm-", "sd")):
            row: dict[str, Any] = {"sample_wall": obj.get("time"), "device": parts[0]}
            for key, value in zip(header[1:], parts[1:]):
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            if "rMB/s" in row:
                row["read_mib_s"] = row["rMB/s"]
            elif "rkB/s" in row:
                row["read_mib_s"] = float(row["rkB/s"]) / 1024.0
            samples.append(row)
    return samples


def summarize_iostat(samples: list[dict[str, Any]], device_prefix: str = "nvme") -> dict[str, Any]:
    selected = [s for s in samples if str(s.get("device", "")).startswith(device_prefix)]
    if not selected:
        selected = samples
    read_vals = [float(s.get("read_mib_s", 0) or 0) for s in selected]
    util_vals = [float(s.get("%util", 0) or 0) for s in selected]
    await_vals = [float(s.get("r_await", 0) or 0) for s in selected]
    return {
        "sample_count": len(selected),
        "read_mib_s_avg": sum(read_vals) / len(read_vals) if read_vals else 0.0,
        "read_mib_s_peak": max(read_vals) if read_vals else 0.0,
        "util_pct_peak": max(util_vals) if util_vals else 0.0,
        "read_await_ms_peak": max(await_vals) if await_vals else 0.0,
    }


def summarize_cache_stats_jsonl(path: str | Path) -> dict[str, Any]:
    latest_by_layer: dict[int, dict[str, Any]] = {}
    timeline: list[dict[str, Any]] = []
    for obj in iter_jsonl(path):
        layer = int(obj.get("layer", -1))
        if layer >= 0:
            latest_by_layer[layer] = obj
            hit = int(obj.get("hit_count", 0) or 0)
            miss = int(obj.get("miss_count", 0) or 0)
            timeline.append(
                {
                    "layer": layer,
                    "dump_tick": int(obj.get("dump_tick", 0) or 0),
                    "hit": hit,
                    "miss": miss,
                    "hit_rate": hit / (hit + miss) if hit + miss > 0 else None,
                    "iouring_read_bytes": int(obj.get("iouring_read_bytes", 0) or 0),
                    "iouring_read_request_count": int(obj.get("iouring_read_request_count", 0) or 0),
                }
            )

    total_hit = sum(int(o.get("hit_count", 0) or 0) for o in latest_by_layer.values())
    total_miss = sum(int(o.get("miss_count", 0) or 0) for o in latest_by_layer.values())
    total_read_bytes = sum(int(o.get("iouring_read_bytes", 0) or 0) for o in latest_by_layer.values())
    total_read_reqs = sum(int(o.get("iouring_read_request_count", 0) or 0) for o in latest_by_layer.values())
    return {
        "layers": len(latest_by_layer),
        "total_hit": total_hit,
        "total_miss": total_miss,
        "hit_rate": total_hit / (total_hit + total_miss) if total_hit + total_miss > 0 else None,
        "iouring_read_bytes": total_read_bytes,
        "iouring_read_mib": total_read_bytes / 1048576.0,
        "iouring_read_request_count": total_read_reqs,
        "timeline": timeline,
        "latest_by_layer": latest_by_layer,
    }


def memory_stat_snapshot(cgroup_path: str | Path) -> dict[str, int]:
    base = Path(cgroup_path)
    out: dict[str, int] = {}
    current = base / "memory.current"
    if current.exists():
        out["memory_current"] = int(current.read_text().strip())
    stat = base / "memory.stat"
    if stat.exists():
        for line in stat.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                out[parts[0]] = int(parts[1])
    return out
