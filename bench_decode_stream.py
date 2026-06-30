#!/usr/bin/env python3
"""Decode-only benchmark using stream mode.
Usage: python3 bench_decode_stream.py <port> [max_tokens]
Returns: decode_tok_s (excluding prefill/first-token latency)
"""
import requests, time, json, sys

port = sys.argv[1] if len(sys.argv) > 1 else "50052"
max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 512
url = f"http://localhost:{port}/v1/completions"

# Warmup
try:
    r = requests.post(url, json={
        "model": "default", "prompt": "Hello",
        "max_tokens": 16, "temperature": 0
    }, timeout=60)
    if r.status_code != 200:
        print(f"WARMUP_FAIL: {r.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"WARMUP_FAIL: {e}")
    sys.exit(1)

time.sleep(2)

# Decode benchmark with stream
data = {
    "model": "default",
    "prompt": "Write a detailed essay about artificial intelligence and its impact on society, including historical context, current developments, and future implications:",
    "max_tokens": max_tokens,
    "temperature": 0,
    "stream": True,
}

start = time.time()
first_token_time = None
token_count = 0

try:
    r = requests.post(url, json=data, stream=True, timeout=300)
    for line in r.iter_lines():
        if line:
            line = line.decode()
            if line.startswith("data: "):
                line = line[6:]
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                    text = chunk["choices"][0].get("text", "")
                    if text:
                        token_count += 1
                        if first_token_time is None:
                            first_token_time = time.time()
                except:
                    pass
except Exception as e:
    print(f"DECODE_FAIL: {e}")
    sys.exit(1)

end = time.time()

if first_token_time and token_count > 1:
    decode_elapsed = end - first_token_time
    decode_tps = (token_count - 1) / decode_elapsed  # exclude first token (prefill)
    total_elapsed = end - start
    print(f"decode_tok_s: {decode_tps:.2f}")
    print(f"decode_tokens: {token_count}")
    print(f"total_elapsed: {total_elapsed:.2f}s")
    print(f"decode_elapsed: {decode_elapsed:.2f}s")
else:
    print(f"DECODE_FAIL: no tokens generated")
    sys.exit(1)
