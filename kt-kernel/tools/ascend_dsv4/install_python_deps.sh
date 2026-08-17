#!/usr/bin/env bash
# =============================================================================
# Install SGLang's NPU runtime dependencies without disturbing the torch stack.
#
#   bash kt-kernel/tools/ascend_dsv4/install_python_deps.sh
#   bash kt-kernel/tools/ascend_dsv4/install_python_deps.sh --dry-run
#
# Why this exists rather than `pip install -e python/`:
#
#   * The clone's default `python/pyproject.toml` is the **CUDA** variant. On a
#     CANN image it pulls torch, flashinfer and cuda-python, replaces the
#     image's torch with a CUDA build, and leaves torch_npu bound to a torch
#     that no longer exists. The NPU stack is then unusable.
#   * `python/pyproject_npu.toml` has the right dependency list, so this script
#     reads the list out of it and installs exactly that — no editable install,
#     no build, no torch.
#
# The torch family already present is written into a pip constraints file, so
# any dependency that wants a different torch fails loudly here instead of
# silently upgrading it later.
# =============================================================================
set -euo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# shellcheck source=./dsv4_env.sh
source "${_here}/dsv4_env.sh"

log() { printf '\n[install_python_deps] %s\n' "$*"; }
die() { printf '[install_python_deps] FATAL: %s\n' "$*" >&2; exit 1; }

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

PYPROJECT="${SGLANG_REPO}/python/pyproject_npu.toml"
if [ ! -f "${PYPROJECT}" ]; then
  # SGLANG_REPO is the usual culprit: exporting it by hand overrides the
  # auto-detection in dsv4_env.sh, so point at what is actually on disk.
  _found="$(find "${DSV4_WORKSPACE}" "${KTRANSFORMERS_REPO}" -maxdepth 4 \
      -name pyproject_npu.toml -path '*/python/*' 2>/dev/null | head -5)"
  die "not found: ${PYPROJECT}
    SGLANG_REPO=${SGLANG_REPO} does not look like an SGLang checkout.
$( [ -n "${_found}" ] \
     && printf '    Found one under:\n%s\n    Set it explicitly, e.g.\n      export SGLANG_REPO=%s' \
          "$(printf '%s\n' "${_found}" | sed 's|^|      |')" \
          "$(dirname "$(dirname "$(printf '%s\n' "${_found}" | head -1)")")" \
     || printf '    No pyproject_npu.toml found under %s either — clone SGLang first.' "${DSV4_WORKSPACE}" )"
fi

WORK="${DSV4_ARTIFACT_DIR}/python-deps"
mkdir -p "${WORK}"
REQS="${WORK}/sglang-npu-requirements.txt"
LOCK="${WORK}/torch-constraints.txt"

log "reading the NPU dependency list from ${PYPROJECT}"
"${DSV4_PYTHON}" - "${PYPROJECT}" "${REQS}" <<'PY'
import sys, tomllib
src, dst = sys.argv[1], sys.argv[2]
with open(src, "rb") as f:
    data = tomllib.load(f)
deps = data["project"]["dependencies"]
with open(dst, "w") as f:
    f.write("# Generated from pyproject_npu.toml — do not edit by hand.\n")
    for d in deps:
        f.write(d + "\n")
print(f"{len(deps)} dependencies -> {dst}")
PY

log "pinning the torch family that is already installed"
"${DSV4_PYTHON}" - "${LOCK}" <<'PY'
import importlib.metadata as md, sys
dst = sys.argv[1]
lines = ["# Generated from the installed environment. Any dependency that wants a\n",
         "# different torch will now fail loudly instead of silently upgrading it.\n"]
# Only torch and torch_npu are pinned. torchvision/torchaudio/torchao are
# deliberately left free: they must match torch, and pinning whatever happens to
# be installed would freeze an already-mismatched one in place. A dependency
# that wants a different *torch* still fails loudly against these two.
found, seen = [], set()
for name in ("torch", "torch_npu", "torch-npu"):
    try:
        v = md.version(name)
    except md.PackageNotFoundError:
        continue
    canon = name.replace("_", "-").lower()  # pip treats torch_npu and torch-npu as one
    if canon in seen:
        continue
    seen.add(canon)
    lines.append(f"{canon}=={v}\n")
    found.append(f"{canon}=={v}")
if not any(f.startswith("torch==") for f in found):
    raise SystemExit("torch is not installed — install torch and torch_npu for your CANN "
                     "version before running this script")
open(dst, "w").writelines(lines)
print("constraints: " + " ".join(found))
PY

if [ "${DRY}" -eq 1 ]; then
  log "dry run; would install:"
  sed -n '2,$p' "${REQS}" | sed 's/^/    /'
  log "with constraints:"
  sed -n '3,$p' "${LOCK}" | sed 's/^/    /'
  exit 0
fi

log "installing (this does not touch torch)"
"${DSV4_PYTHON}" -m pip install -r "${REQS}" -c "${LOCK}"

log "done. Verify with:  bash ${_here}/preflight.sh"
