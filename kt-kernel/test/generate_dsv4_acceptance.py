#!/usr/bin/env python3
"""Record greedy SGLang generation from the exact V4 chat-mode prompts."""

import argparse
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path


def generate(endpoint, record, max_tokens, raw_dir=None):
    body = {
        "text": record["encoded_prompt"],
        "sampling_params": {"temperature": 0, "max_new_tokens": max_tokens},
        "return_logprob": True,
        "top_logprobs_num": 10,
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        result = json.load(response)
    if raw_dir is not None:
        (raw_dir / f"{record['source_index']}.json").write_text(
            json.dumps({**record, "response": result}, ensure_ascii=False)
        )
    assert isinstance(result.get("text"), str) and result["text"].strip(), result
    meta = result["meta_info"]
    assert meta["completion_tokens"] > 0, result
    assert meta.get("finish_reason", {}).get("type") != "abort", result
    probabilities = meta.get("output_token_logprobs", [])
    invalid = [
        index
        for index, row in enumerate(probabilities)
        if not isinstance(row[0], (int, float)) or not math.isfinite(row[0])
    ]
    assert probabilities and not invalid, (
        f"invalid output log-probabilities: source_index={record['source_index']}, positions={invalid}"
    )
    return {**record, "seconds": time.perf_counter() - started, "response": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:31300")
    parser.add_argument("--selected", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    output = Path(args.output)
    rows = json.loads(Path(args.selected).read_text())["validation"][
        args.offset : args.offset + args.count
    ]
    assert len(rows) == args.count and args.count > 0 and args.offset >= 0
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = output.with_suffix(".responses")
    raw_dir.mkdir(exist_ok=False)
    with (
        output.open("x") as handle,
        ThreadPoolExecutor(max_workers=args.concurrency) as pool,
    ):
        results = pool.map(
            partial(
                generate, args.endpoint, max_tokens=args.max_tokens, raw_dir=raw_dir
            ),
            rows,
        )
        try:
            for index, result in enumerate(results):
                result["concurrency"] = args.concurrency
                handle.write(
                    json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n"
                )
                handle.flush()
                print(
                    json.dumps(
                        {
                            "index": args.offset + index,
                            "seconds": result["seconds"],
                            "text": result["response"]["text"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        finally:
            pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
