#!/usr/bin/env python3
import argparse
import json
import os
import random
import re
import time
import urllib.request
from pathlib import Path


COMMON_CJK = (
    "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产"
    "种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使"
    "点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你"
    "明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题程展五果"
    "料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运"
)


PREFILL_RE = re.compile(
    r"Prefill batch, .*?#new-token: (?P<tokens>\d+), .*?"
    r"full token usage: (?P<full>[0-9.]+), mamba usage: (?P<mamba>[0-9.]+).*?"
    r"input throughput \(token/s\): (?P<tps>[0-9.]+)"
)
DECODE_RE = re.compile(
    r"Decode batch, .*?full token usage: (?P<full>[0-9.]+), .*?mamba usage: (?P<mamba>[0-9.]+).*?"
    r"gen throughput \(token/s\): (?P<tps>[0-9.]+)"
)
MAX_TOTAL_RE = re.compile(r"max_total_num_tokens=(?P<tokens>\d+)")


def generate_gibberish(chars: int, seed: int, line_width: int = 200) -> str:
    rng = random.Random(seed)
    pieces = [rng.choice(COMMON_CJK) for _ in range(chars)]
    lines = ["".join(pieces[i : i + line_width]) for i in range(0, len(pieces), line_width)]
    return "\n".join(lines)


def read_log_offset(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def read_log_since(path: Path, offset: int) -> str:
    try:
        with path.open("rb") as f:
            f.seek(offset)
            return f.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def read_all_log(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def read_proc_io(pid: int | None) -> dict[str, int]:
    if not pid:
        return {}
    try:
        out = {}
        with open(f"/proc/{pid}/io", "r", encoding="utf-8") as f:
            for line in f:
                key, value = line.split(":", 1)
                out[key.strip()] = int(value.strip())
        return out
    except OSError:
        return {}


def post_chat(port: int, model: str, prompt: str, max_tokens: int, timeout: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def harmonic_prefill_tps(events: list[dict]) -> float | None:
    # SGLang's first prefill log after an idle period can include queue/startup
    # idle time and report an artificially tiny throughput. Treat it as a
    # boundary marker, not as steady prefill speed.
    usable = [(e["tokens"], e["tps"]) for e in events if e["tokens"] > 0 and e["tps"] > 100.0]
    if not usable:
        return None
    tokens = sum(t for t, _ in usable)
    seconds = sum(t / s for t, s in usable)
    return tokens / seconds if seconds > 0 else None


def parse_runtime_log(text: str, full_log_text: str) -> dict:
    prefill_events = []
    decode_events = []
    for line in text.splitlines():
        if "Prefill batch" in line:
            m = PREFILL_RE.search(line)
            if m:
                prefill_events.append(
                    {
                        "tokens": int(m.group("tokens")),
                        "tps": float(m.group("tps")),
                        "full_token_usage": float(m.group("full")),
                        "mamba_usage": float(m.group("mamba")),
                        "line": line,
                    }
                )
        elif "Decode batch" in line:
            m = DECODE_RE.search(line)
            if m:
                decode_events.append(
                    {
                        "tps": float(m.group("tps")),
                        "full_token_usage": float(m.group("full")),
                        "mamba_usage": float(m.group("mamba")),
                        "line": line,
                    }
                )

    max_total = None
    for m in MAX_TOTAL_RE.finditer(full_log_text):
        max_total = int(m.group("tokens"))

    decode_tps_values = [e["tps"] for e in decode_events if e["tps"] > 10.0]
    usage_values = prefill_events + decode_events
    return {
        "max_total_num_tokens": max_total,
        "prefill_events": prefill_events,
        "decode_events": decode_events,
        "prefill_new_tokens_logged": sum(e["tokens"] for e in prefill_events),
        "prefill_tps_est": harmonic_prefill_tps(prefill_events),
        "prefill_tps_last": prefill_events[-1]["tps"] if prefill_events else None,
        "decode_tps_avg_log": (sum(decode_tps_values) / len(decode_tps_values)) if decode_tps_values else None,
        "decode_tps_last_log": decode_tps_values[-1] if decode_tps_values else None,
        "full_token_usage_last": usage_values[-1]["full_token_usage"] if usage_values else None,
        "mamba_usage_last": usage_values[-1]["mamba_usage"] if usage_values else None,
    }


def load_expert_stats(path: Path) -> tuple[dict[int, dict], dict]:
    latest_by_layer: dict[int, dict] = {}
    if path.exists():
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                latest_by_layer[int(obj["layer"])] = obj

    top_hits = []
    total_access = 0
    total_hit = 0
    total_miss = 0
    total_promote = 0
    scalar_totals = {
        "prefetch_count": 0,
        "async_prefetch_count": 0,
        "prefetch_hit_count": 0,
        "iouring_read_request_count": 0,
        "iouring_read_bytes": 0,
        "full_score_update_count": 0,
        "lookahead_update_count": 0,
        "transition_update_count": 0,
        "coldstart_prefill_count": 0,
        "memory_guard_demote_count": 0,
    }
    for layer, obj in latest_by_layer.items():
        access = obj.get("expert_access", [])
        hit = obj.get("expert_hit", [])
        miss = obj.get("expert_miss", [])
        promote = obj.get("expert_promote", [])
        total_access += sum(access)
        total_hit += sum(hit)
        total_miss += sum(miss)
        total_promote += sum(promote)
        for key in scalar_totals:
            scalar_totals[key] += int(obj.get(key, 0) or 0)
        for expert_id, count in enumerate(hit):
            if count:
                top_hits.append(
                    {
                        "layer": layer,
                        "expert": expert_id,
                        "hit": count,
                        "access": access[expert_id] if expert_id < len(access) else None,
                        "miss": miss[expert_id] if expert_id < len(miss) else None,
                        "promote": promote[expert_id] if expert_id < len(promote) else None,
                    }
                )

    top_hits.sort(key=lambda x: (x["hit"], x["access"] or 0), reverse=True)
    summary = {
        "layers_with_stats": len(latest_by_layer),
        "total_expert_access_tp0": total_access,
        "total_expert_hit_tp0": total_hit,
        "total_expert_miss_tp0": total_miss,
        "total_expert_promote_tp0": total_promote,
        "expert_hit_rate_tp0": (total_hit / (total_hit + total_miss)) if (total_hit + total_miss) else None,
        **scalar_totals,
        "iouring_read_gib": scalar_totals["iouring_read_bytes"] / 1024**3,
        "top_hit_experts": top_hits[:50],
    }
    return latest_by_layer, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one long gibberish prompt against a running MESH service.")
    parser.add_argument("--port", type=int, default=30118)
    parser.add_argument("--model", default="Qwen3.5-35B-A3B-amxint4-mesh-fullgate")
    parser.add_argument("--chars", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260509)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--pid-file", default="/mnt/data3/work/qwen35_amxint4_iouring_mesh_fullgate_30118.pid")
    parser.add_argument(
        "--log-file", default="/mnt/data3/work/qwen35_amxint4_iouring_mesh_fullgate_30118_longstats.log"
    )
    parser.add_argument("--expert-stats", default="/mnt/data3/work/mesh_expert_stats_long.jsonl")
    parser.add_argument("--outdir", default="")
    args = parser.parse_args()

    outdir = Path(args.outdir or f"/mnt/data3/work/bench_long_gibberish_{time.strftime('%Y%m%d_%H%M%S')}")
    outdir.mkdir(parents=True, exist_ok=True)

    pid = None
    try:
        pid = int(Path(args.pid_file).read_text().strip())
    except (OSError, ValueError):
        pass

    stats_path = Path(args.expert_stats)
    if stats_path.exists():
        stats_path.unlink()

    log_path = Path(args.log_file)
    log_offset = read_log_offset(log_path)
    io_before = read_proc_io(pid)

    gibberish = generate_gibberish(args.chars, args.seed)
    question = (
        "\n\n上面是随机乱码。请忽略乱码，只回答：MESH 的热专家缓存命中时，是否需要从 SSD 读取权重？用一句中文回答。"
    )
    prompt = gibberish + question
    (outdir / "prompt_tail.txt").write_text(prompt[-2000:], encoding="utf-8")

    print(
        json.dumps({"event": "request_start", "chars": args.chars, "outdir": str(outdir)}, ensure_ascii=False),
        flush=True,
    )
    start = time.time()
    response = post_chat(args.port, args.model, prompt, args.max_tokens, args.timeout)
    elapsed = time.time() - start
    io_after = read_proc_io(pid)

    log_delta = read_log_since(log_path, log_offset)
    full_log = read_all_log(log_path)
    runtime = parse_runtime_log(log_delta, full_log)

    usage = response.get("usage", {})
    message = response["choices"][0]["message"]["content"]
    completion_tokens = usage.get("completion_tokens")
    prompt_tokens = usage.get("prompt_tokens")
    total_tokens = usage.get("total_tokens")

    latest_by_layer, expert_summary = load_expert_stats(stats_path)
    expert_counts_path = outdir / "expert_counts_by_layer.json"
    expert_counts_path.write_text(
        json.dumps({str(k): v for k, v in sorted(latest_by_layer.items())}, ensure_ascii=False),
        encoding="utf-8",
    )

    kv_est = None
    if total_tokens is not None and runtime["max_total_num_tokens"]:
        kv_est = total_tokens / runtime["max_total_num_tokens"]

    result = {
        "chars": args.chars,
        "seed": args.seed,
        "elapsed_sec": elapsed,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "end_to_end_completion_tok_s": (completion_tokens / elapsed) if completion_tokens and elapsed > 0 else None,
        "proc_read_mb": (
            (io_after.get("read_bytes", 0) - io_before.get("read_bytes", 0)) / 1024 / 1024
            if io_before and io_after
            else None
        ),
        "kv_cache": {
            "max_total_num_tokens": runtime["max_total_num_tokens"],
            "used_tokens_from_api": total_tokens,
            "estimated_usage_from_api": kv_est,
            "full_token_usage_last_from_log": runtime["full_token_usage_last"],
            "mamba_usage_last_from_log": runtime["mamba_usage_last"],
        },
        "prefill": {
            "logged_new_tokens": runtime["prefill_new_tokens_logged"],
            "tps_est_from_log": runtime["prefill_tps_est"],
            "tps_last_from_log": runtime["prefill_tps_last"],
            "num_log_events": len(runtime["prefill_events"]),
        },
        "decode": {
            "tps_avg_from_log": runtime["decode_tps_avg_log"],
            "tps_last_from_log": runtime["decode_tps_last_log"],
            "end_to_end_completion_tok_s": (completion_tokens / elapsed) if completion_tokens and elapsed > 0 else None,
            "num_log_events": len(runtime["decode_events"]),
        },
        "expert_stats": expert_summary,
        "response_head": message[:1000],
        "response_finish_reason": response["choices"][0].get("finish_reason"),
        "files": {
            "response": str(outdir / "response.txt"),
            "summary": str(outdir / "summary.json"),
            "expert_counts_by_layer": str(expert_counts_path),
            "prompt_tail": str(outdir / "prompt_tail.txt"),
        },
    }

    (outdir / "response.txt").write_text(message, encoding="utf-8")
    (outdir / "runtime_log_delta.txt").write_text(log_delta, encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
