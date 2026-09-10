#!/usr/bin/env python3
"""Record deterministic SGLang generation from the exact V4 chat-mode prompts."""

import argparse
import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path


def generate(endpoint, record, max_tokens):
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
    assert isinstance(result.get("text"), str) and result["text"].strip(), result
    meta = result["meta_info"]
    assert meta["completion_tokens"] > 0, result
    assert meta.get("finish_reason", {}).get("type") != "abort", result
    probabilities = meta.get("output_token_logprobs", [])
    assert probabilities and all(math.isfinite(row[0]) for row in probabilities)
    return {**record, "seconds": time.perf_counter() - started, "response": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:31300")
    parser.add_argument("--selected", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--concurrency", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    output = Path(args.output)
    rows = json.loads(Path(args.selected).read_text())["validation"][: args.count]
    assert len(rows) == args.count and args.count > 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("x") as handle,
        ThreadPoolExecutor(max_workers=args.concurrency) as pool,
    ):
        results = pool.map(
            partial(generate, args.endpoint, max_tokens=args.max_tokens), rows
        )
        for index, result in enumerate(results):
            result["concurrency"] = args.concurrency
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {
                        "index": index,
                        "seconds": result["seconds"],
                        "text": result["response"]["text"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
