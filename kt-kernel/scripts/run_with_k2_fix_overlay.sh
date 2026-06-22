#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
OVERLAY_DIR="${KT_K2_FIX_OVERLAY_DIR:-${TMPDIR:-/tmp}/kt-k2-fix-python-overlay}"

mkdir -p "${OVERLAY_DIR}"
ln -sfn "${REPO_ROOT}/kt-kernel/python" "${OVERLAY_DIR}/kt_kernel"

# Keep only the overlay on PYTHONPATH for kt_kernel. The package initializer
# loads kt_kernel_ext and injects it into sys.modules; adding kt-kernel/python
# directly can load kt_kernel_ext twice and trip pybind duplicate registration.
export PYTHONPATH="${OVERLAY_DIR}:${REPO_ROOT}/third_party/sglang/python:${PYTHONPATH:-}"

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 <command> [args...]" >&2
  echo "example: $0 python -m sglang.launch_server --model /path/to/model ..." >&2
  exit 2
fi

exec "$@"
