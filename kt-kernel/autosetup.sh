#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

PY_LIST=${PY_LIST:-"3.11 3.12 3.13"}
TORCH_LIST=${TORCH_LIST:-"2.9.1"}
WORK_ROOT=${WORK_ROOT:-/mnt/data3/lpl/kt-kernel-autosetup}
WHEELS_DIR=${WHEELS_DIR:-"$PWD/wheels"}
PIP_CACHE_DIR=${PIP_CACHE_DIR:-/mnt/data3/lpl/pip-cache}
TMP_ROOT=${TMP_ROOT:-/mnt/data3/lpl/tmp}
FORCE=${FORCE:-0}
REPAIR=${REPAIR:-0}
AUDITWHEEL_PLAT=${AUDITWHEEL_PLAT:-manylinux_2_35_x86_64}
CPUINFER_ENABLE_CPPTRACE=${CPUINFER_ENABLE_CPPTRACE:-OFF}

mkdir -p "$WORK_ROOT" "$WHEELS_DIR" "$PIP_CACHE_DIR" "$TMP_ROOT"

index_for_torch_version() {
  case "$1" in
    2.3.*) echo "https://download.pytorch.org/whl/cu121" ;;
    2.4.*) echo "https://download.pytorch.org/whl/cu121" ;;
    2.5.*) echo "https://download.pytorch.org/whl/cu124" ;;
    2.6.*) echo "https://download.pytorch.org/whl/cu124" ;;
    2.7.*) echo "https://download.pytorch.org/whl/cu126" ;;
    2.8.*) echo "https://download.pytorch.org/whl/cu128" ;;
    2.9.*) echo "https://download.pytorch.org/whl/cu130" ;;
    2.10.*) echo "" ;;
    2.11.*) echo "" ;;
    *)     echo "https://download.pytorch.org/whl/cu124" ;;
  esac
}

verify_torch_stack() {
  python - <<'PY'
import email
import importlib.metadata as md
import pathlib
import site
import sys
from packaging.requirements import Requirement

import torch

sp = pathlib.Path(site.getsitepackages()[0])
meta = next(sp.glob('torch-*.dist-info/METADATA'))
msg = email.message_from_string(meta.read_text())
def norm(name: str) -> str:
    return name.lower().replace('_', '-').replace('.', '-')

expected = {}
for line in msg.get_all('Requires-Dist', []):
    req = Requirement(line)
    if not req.name.startswith('nvidia-'):
        continue
    pinned = [spec.version for spec in req.specifier if spec.operator == '==']
    if len(pinned) != 1:
        continue
    expected[norm(req.name)] = (req.name, pinned[0])

installed_versions = {}
for dist in md.distributions():
    name = dist.metadata.get('Name')
    if not name:
        continue
    installed_versions[norm(name)] = dist.version

mismatch = []
for key, (pkg, ver) in sorted(expected.items()):
    installed = installed_versions.get(key)
    if installed is None:
        mismatch.append(f'{pkg}: missing, expected {ver}')
        continue
    if installed != ver:
        mismatch.append(f'{pkg}: installed {installed}, expected {ver}')

cusparselt_candidates = [
    sp / 'cusparselt' / 'lib' / 'libcusparseLt.so.0',
    sp / 'nvidia' / 'cusparselt' / 'lib' / 'libcusparseLt.so.0',
]
cusparselt = next((path for path in cusparselt_candidates if path.exists()), None)
if cusparselt is None:
    mismatch.append(
        'cusparselt layout missing: expected one of '
        + ', '.join(str(path) for path in cusparselt_candidates)
    )

if mismatch:
    print('Torch CUDA runtime stack is inconsistent:', file=sys.stderr)
    for item in mismatch:
        print(f'  - {item}', file=sys.stderr)
    raise SystemExit(2)

print('TORCH_OK', torch.__version__, torch.version.cuda, torch.cuda.is_available())
print('CUSPARSELT_PATH', cusparselt)
PY
}

verify_wheel_contents() {
  python - "$1" <<'PY'
import pathlib
import sys
import zipfile
wheel = pathlib.Path(sys.argv[1])
with zipfile.ZipFile(wheel) as zf:
    names = set(zf.namelist())
if not any(name.startswith('kt_kernel/kt_kernel_ext') and name.endswith('.so') for name in names):
    raise SystemExit('missing kt_kernel_ext shared object in wheel')
required = [
    'kt_kernel/sft/__init__.py',
    'kt_kernel/sft/wrapper.py',
    'kt_kernel/cli/completions/_kt',
]
missing = [name for name in required if name not in names]
if missing:
    raise SystemExit(f'missing required wheel entries: {missing}')
print(f'WHEEL_OK {wheel.name}')
PY
}

for py in $PY_LIST; do
  PYBIN="$(command -v python${py} || true)"
  if [[ ! -x "$PYBIN" ]]; then
    echo ">> Skip python ${py}: not found"
    continue
  fi

  for tv in $TORCH_LIST; do
    echo "======== Build: Python ${py} × Torch ${tv} ========"
    ENV_DIR="$WORK_ROOT/.venv-py${py//./}-torch${tv//./}"
    OUT_DIR="$WHEELS_DIR/py${py//./}-torch${tv//./}"
    IDX="$(index_for_torch_version "$tv")"

    if [[ "$FORCE" = "1" ]]; then
      rm -rf "$OUT_DIR"
    elif compgen -G "$OUT_DIR/*.whl" > /dev/null; then
      echo ">> Found existing wheel for py${py//./}-torch${tv//./}, skip"
      continue
    fi

    rm -rf "$ENV_DIR"
    mkdir -p "$OUT_DIR"
    "$PYBIN" -m venv "$ENV_DIR"
    # shellcheck disable=SC1090
    source "$ENV_DIR/bin/activate"

    export PYTHONNOUSERSITE=1
    export PIP_CACHE_DIR
    export CPUINFER_ENABLE_CPPTRACE
    export TMPDIR="$TMP_ROOT"
    export TEMP="$TMP_ROOT"
    export TMP="$TMP_ROOT"

    python -m pip install -U pip setuptools wheel build cmake pybind11 packaging numpy
    if [[ -n "$IDX" ]]; then
      python -m pip install --index-url "$IDX" "torch==$tv"
    else
      python -m pip install "torch==$tv"
    fi
    verify_torch_stack

    rm -rf build dist kt_kernel.egg-info
    python -m build --no-isolation --wheel -v

    wheels=(dist/*.whl)
    if (( ${#wheels[@]} != 1 )); then
      echo "!! expected exactly one wheel in dist/, got ${#wheels[@]}" >&2
      exit 2
    fi

    verify_wheel_contents "${wheels[0]}"

    python - "$OUT_DIR/build-info.txt" "$py" "$tv" "$IDX" "$CPUINFER_ENABLE_CPPTRACE" <<'PY'
from pathlib import Path
import platform
import sys
import torch
out = Path(sys.argv[1])
out.write_text(
    f"python={sys.argv[2]}\n"
    f"torch={torch.__version__}\n"
    f"torch_cuda={torch.version.cuda}\n"
    f"cuda_available={torch.cuda.is_available()}\n"
    f"index_url={sys.argv[4]}\n"
    f"platform={platform.platform()}\n"
    f"cpptrace={sys.argv[5]}\n"
)
print(f"BUILD_INFO {out}")
PY

    if [[ "$REPAIR" = "1" ]]; then
      python -m pip install -U auditwheel patchelf
      rm -rf "$OUT_DIR/wheelhouse"
      mkdir -p "$OUT_DIR/wheelhouse"
      auditwheel repair "${wheels[0]}" --plat "$AUDITWHEEL_PLAT" -w "$OUT_DIR/wheelhouse"
      cp "$OUT_DIR/wheelhouse"/*.whl "$OUT_DIR/"
    else
      cp "${wheels[0]}" "$OUT_DIR/"
    fi

    deactivate
  done
done

echo "== Wheels saved in ${WHEELS_DIR} =="
