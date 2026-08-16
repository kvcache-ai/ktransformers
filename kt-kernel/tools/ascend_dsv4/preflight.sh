#!/usr/bin/env bash
# =============================================================================
# Check everything the server needs, before starting it.
#
# Every check below corresponds to a failure mode that is silent or badly
# reported at serve time. Run it after any environment change:
#
#   bash kt-kernel/tools/ascend_dsv4/preflight.sh
#
# Exit code 0 = safe to launch.
# =============================================================================
set -uo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=./dsv4_env.sh
source "${_here}/dsv4_env.sh"

FAIL=0
ok()   { printf '  \033[32mok\033[0m   %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; FAIL=1; }
warn() { printf '  \033[33mwarn\033[0m %s\n' "$*"; }
sec()  { printf '\n%s\n' "$*"; }

sec "CANN"
if [ -f "${ASCEND_INSTALL_ROOT}/ascend-toolkit/set_env.sh" ]; then
  ok "toolkit ${CANN_VERSION} at ${ASCEND_INSTALL_ROOT}"
else
  bad "no toolkit at ${ASCEND_INSTALL_ROOT}"
fi
for v in customize custom_transformer; do
  if [ -d "${CANN_VENDORS_DIR}/${v}" ]; then
    ok "vendor ${v}"
  else
    bad "vendor ${v} missing under ${CANN_VENDORS_DIR} — run build_cann_ops.sh"
  fi
done
case ":${ASCEND_CUSTOM_OPP_PATH}:" in
  *":${CANN_VENDORS_DIR}/customize:"*) ok "ASCEND_CUSTOM_OPP_PATH includes the vendors" ;;
  *) bad "ASCEND_CUSTOM_OPP_PATH does not include ${CANN_VENDORS_DIR}/customize" ;;
esac

sec "Python stack"
# Every check runs; one missing package must not hide the rest.
"${DSV4_PYTHON}" - <<'PY'
import importlib, sys

failed = 0

def probe(label, fn):
    global failed
    try:
        print(f"  ok   {label}: {fn()}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {label}: {type(exc).__name__}: {exc}")
        failed = 1
        return False

torch = None
if probe("torch", lambda: importlib.import_module("torch").__version__):
    torch = importlib.import_module("torch")
probe("torch_npu", lambda: importlib.import_module("torch_npu").__version__)
if torch is not None:
    probe("npu available",
          lambda: f"{torch.npu.is_available()} devices={torch.npu.device_count()}")
probe("transformers", lambda: importlib.import_module("transformers").__version__)

# transformers imports torchaudio unconditionally (loss_rnnt), so a torchaudio
# built against a different torch breaks the whole stack with an undefined
# symbol long before anything model-related runs. Same story for torchvision.
for _companion in ("torchvision", "torchaudio"):
    try:
        importlib.import_module(_companion)
    except ModuleNotFoundError:
        pass  # genuinely absent is fine; nothing here requires it
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL {_companion}: {type(exc).__name__}: {exc}")
        if torch is not None:
            base = torch.__version__.split("+")[0]
            print(f"       This build does not match torch {torch.__version__}. Install the "
                  f"matching release:\n"
                  f"         pip install --no-deps '{_companion}=={base}'")
        failed = 1

if probe("custom_ops", lambda: (importlib.import_module("custom_ops"), "imported")[1]) and torch is not None:
    missing = [op for op in ("compressor", "npu_sparse_attn_sharedkv",
                             "npu_quant_lightning_indexer", "npu_moe_gating_top_k")
               if not hasattr(torch.ops.custom, op)]
    if missing:
        print(f"  FAIL torch.ops.custom.*: missing {missing} — "
              "the operator vendor packages are not loaded")
        failed = 1
    else:
        print("  ok   torch.ops.custom.* all present")

sys.exit(failed)
PY
[ $? -ne 0 ] && FAIL=1

sec "kt-kernel"
"${DSV4_PYTHON}" - <<'PY'
import sys
try:
    from kt_kernel import kt_kernel_ext
    from kt_kernel.utils.loader import GGMLQuantizationType
except Exception as exc:  # noqa: BLE001
    print(f"  FAIL import kt_kernel: {type(exc).__name__}: {exc}")
    print("       Build it with kt-kernel/tools/ascend_dsv4/build_kt_kernel.sh")
    sys.exit(1)
print(f"  ok   kt_kernel_ext at {kt_kernel_ext.__file__}")
if not hasattr(kt_kernel_ext, "init_ascend_callback_worker"):
    print("  FAIL kt_kernel_ext has no init_ascend_callback_worker — "
          "it was built without CPUINFER_USE_ASCEND_NPU=1")
    sys.exit(1)
print("  ok   Ascend callback worker binding present")
# MXFP4 == 39 proves the llama.cpp patch series was applied at configure time.
if int(GGMLQuantizationType.MXFP4) != 39:
    print(f"  FAIL GGMLQuantizationType.MXFP4 == {int(GGMLQuantizationType.MXFP4)}, expected 39")
    sys.exit(1)
print("  ok   GGMLQuantizationType.MXFP4 == 39")
PY
[ $? -ne 0 ] && FAIL=1

sec "sgl-kernel-npu"
if "${DSV4_PYTHON}" -c 'import sgl_kernel_npu' >/dev/null 2>&1; then
  ok "sgl_kernel_npu importable"
else
  bad "sgl_kernel_npu is missing. SGLang imports it unconditionally from
       srt/mem_cache/pool_host/mha.py, so the server dies at startup.
       Container images ship it; on a native install build it:
         bash ${_here}/build_sgl_kernel_npu.sh"
fi

sec "SGLang"
# Two different failures look alike from the outside, so separate them: an
# import that raises is a missing dependency, an import that succeeds from the
# wrong path is a shadowed PYTHONPATH.
# Keep stderr out of the captured value: torch_npu prints warnings there, and
# folding them into stdout makes a healthy import look like a wrong path.
_sglang_err="$(mktemp)"
_sglang_out="$("${DSV4_PYTHON}" -c 'import sglang; print(sglang.__file__)' 2>"${_sglang_err}" | tail -1)"
_sglang_rc=${PIPESTATUS[0]}
if [ "${_sglang_rc}" -ne 0 ]; then
  bad "importing sglang failed:
$(tail -3 "${_sglang_err}" | sed 's/^/       /')
       If this is a missing module, install SGLang's NPU runtime dependencies:
         bash ${_here}/install_python_deps.sh
       Do NOT run 'pip install -e python/' from the clone on a CANN image: the
       default pyproject.toml is the CUDA variant and it replaces torch."
else
  case "${_sglang_out}" in
    "${SGLANG_REPO}"/python/sglang/__init__.py) ok "sglang resolves to the clone: ${_sglang_out}" ;;
    *) bad "sglang resolves to ${_sglang_out}
       expected ${SGLANG_REPO}/python/sglang/__init__.py
       PYTHONPATH was shadowed. Export it after every environment script, not before." ;;
  esac
fi
rm -f "${_sglang_err}"

sec "Weights"
if [ -f "${DSV4_MODEL_PATH}/config.json" ]; then
  _fmt="$("${DSV4_PYTHON}" -c "import json,sys;c=json.load(open(sys.argv[1]));q=c.get('quantization_config') or {};print(q.get('quant_method') or q.get('format') or 'unknown')" "${DSV4_MODEL_PATH}/config.json" 2>/dev/null)"
  ok "W8A8 checkpoint ${DSV4_MODEL_PATH} (quantization: ${_fmt})"
  [ "${_fmt}" = "compressed-tensors" ] || warn "expected quant_method=compressed-tensors, got '${_fmt}'"
else
  bad "no config.json under DSV4_MODEL_PATH=${DSV4_MODEL_PATH}"
fi

_n_gguf="$(ls -1 "${DSV4_GGUF_DIR}"/dsv4_layer*_mxfp4.gguf 2>/dev/null | wc -l | tr -d ' ')"
_n_layers="$("${DSV4_PYTHON}" -c "import json,sys;print(json.load(open(sys.argv[1]))['num_hidden_layers'])" "${DSV4_MODEL_PATH}/config.json" 2>/dev/null || echo 43)"
if [ "${_n_gguf}" = "${_n_layers}" ]; then
  ok "${_n_gguf} per-layer MXFP4 GGUF files in ${DSV4_GGUF_DIR}"
else
  bad "found ${_n_gguf} GGUF files in ${DSV4_GGUF_DIR}, expected ${_n_layers} — run convert_mxfp4_gguf.sh"
fi
case "${DSV4_GGUF_TEMPLATE}" in
  *"{layer_idx}"*) ok "DSV4_GGUF_TEMPLATE contains the {layer_idx} placeholder" ;;
  *) bad "DSV4_GGUF_TEMPLATE has no {layer_idx} placeholder: ${DSV4_GGUF_TEMPLATE}
       A path without it is used literally for every layer." ;;
esac

sec "Host resources"
ok "NUMA nodes=${DSV4_THREADPOOL_COUNT} -> --kt-threadpool-count ${DSV4_THREADPOOL_COUNT}, --kt-cpuinfer ${DSV4_CPUINFER}"
_ram_gb="$(awk '/MemTotal/ {printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null || echo 0)"
if [ "${_ram_gb}" -ge 200 ] 2>/dev/null; then
  ok "host RAM ${_ram_gb} GB"
else
  warn "host RAM ${_ram_gb} GB — the offloaded experts alone need about 140 GB resident"
fi
_free_gb="$(df -BG --output=avail "${DSV4_GGUF_DIR}" 2>/dev/null | tail -1 | tr -dc '0-9')"
[ -n "${_free_gb}" ] && ok "free space on the GGUF filesystem: ${_free_gb} GB"

sec "Serving parameters"
if [ $(( DSV4_CHUNKED_PREFILL_SIZE % 128 )) -eq 0 ] && [ "${DSV4_CHUNKED_PREFILL_SIZE}" -gt 0 ]; then
  ok "--chunked-prefill-size ${DSV4_CHUNKED_PREFILL_SIZE} is a positive multiple of --page-size 128"
else
  bad "--chunked-prefill-size must be a positive multiple of 128; got ${DSV4_CHUNKED_PREFILL_SIZE}"
fi
case ":${http_proxy:-}:${https_proxy:-}:${all_proxy:-}:" in
  ::::) ok "no HTTP proxy set" ;;
  *) bad "an HTTP proxy is set; the startup warmup will fail with 502" ;;
esac

printf '\n'
if [ "${FAIL}" -eq 0 ]; then
  printf '\033[32mPREFLIGHT OK\033[0m\n'
else
  printf '\033[31mPREFLIGHT FAILED\033[0m — fix the FAIL lines above before launching.\n'
fi
exit "${FAIL}"
