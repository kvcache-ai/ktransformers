#!/usr/bin/env python3
"""Benchmark OpenAI-compatible serving prefill/decode throughput.

The script intentionally targets an already running server. Launching large KT
models is hardware- and model-specific, while request construction, isolation,
and metric calculation should stay reusable across models.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROMPT_UNIT = (
    "In a controlled benchmark, explain one practical systems observation "
    "about sparse MoE inference and keep the reasoning concise. "
)


@dataclass
class RequestResult:
    request_id: int
    mode: str
    ok: bool
    latency_s: float
    ttft_s: float | None
    decode_s: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    prompt_tps: float | None
    decode_tps: float | None
    error: str | None = None


def percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("empty value list")
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * pct / 100.0
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    frac = pos - low
    return values[low] * (1.0 - frac) + values[high] * frac


def build_prompt(target_chars: int, request_id: int) -> str:
    prefix = f"Request {request_id}. "
    body = (PROMPT_UNIT * max(1, target_chars // len(PROMPT_UNIT) + 1))[:target_chars]
    return prefix + body


def post_json(url: str, payload: dict[str, Any], timeout_s: float):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=timeout_s)


def parse_stream_response(response, request_start: float) -> tuple[float | None, dict[str, Any] | None]:
    first_token_s = None
    usage = None

    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue

        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            break

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        if chunk.get("usage"):
            usage = chunk["usage"]

        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if first_token_s is None and delta.get("content"):
            first_token_s = time.perf_counter() - request_start

    return first_token_s, usage


def parse_nonstream_response(response) -> dict[str, Any] | None:
    body = response.read().decode("utf-8")
    if not body:
        return None
    return json.loads(body).get("usage")


def run_one(
    *,
    request_id: int,
    mode: str,
    url: str,
    model: str,
    prompt_chars: int,
    max_tokens: int,
    temperature: float,
    timeout_s: float,
    stream: bool,
) -> RequestResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(prompt_chars, request_id)}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}

    start = time.perf_counter()
    try:
        with post_json(url, payload, timeout_s) as response:
            if stream:
                ttft_s, usage = parse_stream_response(response, start)
            else:
                usage = parse_nonstream_response(response)
                ttft_s = None
        latency_s = time.perf_counter() - start
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return RequestResult(
            request_id=request_id,
            mode=mode,
            ok=False,
            latency_s=0.0,
            ttft_s=None,
            decode_s=None,
            prompt_tokens=None,
            completion_tokens=None,
            prompt_tps=None,
            decode_tps=None,
            error=repr(exc),
        )

    prompt_tokens = usage.get("prompt_tokens") if usage else None
    completion_tokens = usage.get("completion_tokens") if usage else None

    prompt_tps = None
    decode_tps = None
    decode_s = None
    if mode == "prefill" and prompt_tokens is not None:
        denom = ttft_s if stream and ttft_s else latency_s
        prompt_tps = prompt_tokens / denom if denom > 0 else None
    elif mode == "decode" and completion_tokens is not None:
        if stream and ttft_s is not None:
            decode_s = max(0.0, latency_s - ttft_s)
            decoded_after_first = max(0, completion_tokens - 1)
            decode_tps = decoded_after_first / decode_s if decode_s > 0 else None
        else:
            decode_s = latency_s
            decode_tps = completion_tokens / latency_s if latency_s > 0 else None

    return RequestResult(
        request_id=request_id,
        mode=mode,
        ok=True,
        latency_s=latency_s,
        ttft_s=ttft_s,
        decode_s=decode_s,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tps=prompt_tps,
        decode_tps=decode_tps,
    )


def summarize(results: list[RequestResult], mode: str) -> dict[str, Any]:
    ok = [result for result in results if result.ok]
    metric_name = "prompt_tps" if mode == "prefill" else "decode_tps"
    metric_values = [getattr(result, metric_name) for result in ok]
    metric_values = [value for value in metric_values if value is not None]
    latency_values = [result.latency_s for result in ok]
    ttft_values = [result.ttft_s for result in ok if result.ttft_s is not None]

    summary: dict[str, Any] = {
        "mode": mode,
        "count": len(results),
        "ok_count": len(ok),
        "error_count": len(results) - len(ok),
    }
    if metric_values:
        summary.update(
            {
                f"{metric_name}_mean": statistics.fmean(metric_values),
                f"{metric_name}_p50": percentile(metric_values, 50),
                f"{metric_name}_p90": percentile(metric_values, 90),
            }
        )
    if latency_values:
        summary.update(
            {
                "latency_mean_s": statistics.fmean(latency_values),
                "latency_p50_s": percentile(latency_values, 50),
                "latency_p90_s": percentile(latency_values, 90),
            }
        )
    if ttft_values:
        summary.update(
            {
                "ttft_mean_s": statistics.fmean(ttft_values),
                "ttft_p50_s": percentile(ttft_values, 50),
                "ttft_p90_s": percentile(ttft_values, 90),
            }
        )
    return summary


def write_outputs(results: list[RequestResult], summary: dict[str, Any], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_prefix.with_suffix(".jsonl")
    summary_path = output_prefix.with_suffix(".summary.json")
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    print(f"wrote {jsonl_path}")
    print(f"wrote {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--model", required=True, help="Served model name, including adapter suffix if needed.")
    parser.add_argument("--mode", choices=["prefill", "decode"], required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--prompt-chars", type=int, default=12000)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming; decode TPS then includes prefill time.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    url = args.base_url.rstrip("/") + "/chat/completions"
    max_tokens = args.max_tokens
    if max_tokens is None:
        max_tokens = 1 if args.mode == "prefill" else 256

    for idx in range(args.warmup):
        run_one(
            request_id=-(idx + 1),
            mode=args.mode,
            url=url,
            model=args.model,
            prompt_chars=args.prompt_chars,
            max_tokens=max_tokens,
            temperature=args.temperature,
            timeout_s=args.timeout_s,
            stream=not args.no_stream,
        )

    results: list[RequestResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                run_one,
                request_id=request_id,
                mode=args.mode,
                url=url,
                model=args.model,
                prompt_chars=args.prompt_chars,
                max_tokens=max_tokens,
                temperature=args.temperature,
                timeout_s=args.timeout_s,
                stream=not args.no_stream,
            )
            for request_id in range(args.requests)
        ]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: item.request_id)
    summary = {
        "schema": "kt_openai_tps_bench_v1",
        "base_url": args.base_url,
        "model": args.model,
        "prompt_chars": args.prompt_chars,
        "max_tokens": max_tokens,
        "concurrency": args.concurrency,
        "requests": args.requests,
        "stream": not args.no_stream,
        **summarize(results, args.mode),
    }
    write_outputs(results, summary, args.output_prefix)


if __name__ == "__main__":
    main()
