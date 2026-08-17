#!/usr/bin/env python3
"""Interactive chat client for a local DeepSeek-V4-Flash server.

    python3 dsv4_chat.py [port]      # default 18080

Streams the reply and keeps multi-turn context. `/reset` clears the context,
`/quit` exits. Standard library only — nothing to install.
"""
import json, sys, urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "18080"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
history = []

print(f"connected to {URL}   (/reset clears context, /quit exits)\n")
while True:
    try:
        q = input("\033[1myou >\033[0m ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); break
    if not q:
        continue
    if q == "/quit":
        break
    if q == "/reset":
        history.clear(); print("(context cleared)\n"); continue

    history.append({"role": "user", "content": q})
    body = json.dumps({
        "model": "dsv4", "messages": history, "stream": True,
        "temperature": 0.6, "max_tokens": 1024,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    print("\033[1mDeepSeek >\033[0m ", end="", flush=True)
    parts = []
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                delta = json.loads(payload)["choices"][0].get("delta", {})
            except Exception:
                continue
            chunk = delta.get("content")
            if chunk:
                parts.append(chunk)
                print(chunk, end="", flush=True)
    print("\n")
    history.append({"role": "assistant", "content": "".join(parts)})
