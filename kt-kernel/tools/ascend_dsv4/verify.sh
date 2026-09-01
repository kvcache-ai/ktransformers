#!/usr/bin/env bash
# Acceptance checks against a running server, and an interactive client.
#
#   verify.sh [serve.log]   run the checks
#   verify.sh chat [port]   interactive chat (/reset clears, /quit exits)
set -uo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=./dsv4_env.sh
source "${_here}/dsv4_env.sh"

if [ "${1:-}" = "chat" ]; then
  shift
  exec "${DSV4_PYTHON}" - "${1:-${DSV4_PORT}}" <<'CHAT_PY'
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
CHAT_PY
fi

LOG="${1:-${DSV4_LOG_DIR}/serve.log}"
BASE="http://127.0.0.1:${DSV4_PORT}"
FAIL=0
ok()  { printf '  \033[32mok\033[0m   %s\n' "$*"; }
bad() { printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAIL=1; }
sec() { printf '\n%s\n' "$*"; }

sec "1. HBM accounting  (log: ${LOG})"
if [ -r "${LOG}" ]; then
  grep -E 'Load weight (begin|end)' "${LOG}" | tail -2 | sed 's/^/       /'
  _pools="$(grep -ohE '(full|swa)=[0-9]+' "${LOG}" | sort -u | tr '\n' ' ')"
  if [ -n "${_pools}" ]; then
    ok "KV pools: ${_pools}"
    printf '       swa is a fixed 10%% of full. A prompt longer than swa cannot be\n'
    printf '       scheduled and the scheduler spins instead of reporting it.\n'
  else
    bad "no KV pool sizes in the log — did the server finish loading?"
  fi
else
  bad "cannot read ${LOG}"
fi

sec "2. NPU graph capture"
if grep -q 'Capture target decode NPU graph end' "${LOG}" 2>/dev/null; then
  ok "$(grep -o 'Capture target decode NPU graph end.*' "${LOG}" | tail -1)"
else
  bad "decode graph was not captured — decode throughput will be roughly 5x lower"
fi

sec "3. Health"
for ep in health health_generate; do
  code="$(curl -s --noproxy '*' -o /dev/null -w '%{http_code}' "${BASE}/${ep}" 2>/dev/null)"
  [ "${code}" = "200" ] && ok "/${ep} -> 200" || bad "/${ep} -> ${code:-no response}"
done

sec "4. Numerical probe"
_probe="$(curl -s --noproxy '*' -X POST "${BASE}/generate" \
    -H 'Content-Type: application/json' \
    -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":16}}' \
    2>/dev/null | "${DSV4_PYTHON}" -c 'import json,sys; print(json.load(sys.stdin)["text"])' 2>/dev/null)"
if [ -z "${_probe}" ]; then
  bad "empty completion. HTTP 200 with \"text\": \"\" is a failure, not a pass.
       Usual causes: a --kt-weight-path template that resolved to nothing, or a
       kt-kernel that loaded no experts. Check the per-layer GGUF paths in the log."
else
  ok "completion: $(printf '%s' "${_probe}" | head -c 120)"
  printf '       Greedy decoding is deterministic here: re-run this and compare byte for\n'
  printf '       byte, and compare against another machine before trusting any number.\n'
fi

if [ "${DSV4_PREFILL_STREAM:-0}" = "1" ]; then
  sec "5. Streaming prefill actually engaged"
  _inline="$(grep -c 'inline resident' "${LOG}" 2>/dev/null || echo 0)"
  _fallback="$(grep -cE 'streaming failed|hybrid fallback' "${LOG}" 2>/dev/null || echo 0)"
  if [ "${_inline}" -gt 0 ] && [ "${_fallback}" -eq 0 ]; then
    ok "inline resident=${_inline}, hybrid fallback=${_fallback}"
  else
    bad "inline resident=${_inline} (want > 0), hybrid fallback=${_fallback} (want 0).
       Both silent failure modes answer requests normally: the prompt was shorter
       than KT_PREFILL_STREAM_THRESHOLD, or every layer OOMed and fell back."
  fi
fi

printf '\n'
if [ "${FAIL}" -eq 0 ]; then
  printf '\033[32mALL CHECKS PASSED\033[0m\n'
else
  printf '\033[31mCHECKS FAILED\033[0m — do not measure throughput or accuracy until these pass.\n'
fi
exit "${FAIL}"
