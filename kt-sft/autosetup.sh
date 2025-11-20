#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# 允许通过环境变量覆盖
PY_LIST=${PY_LIST:-"3.13"}
TORCH_LIST=${TORCH_LIST:-"2.5.0 2.6.0 2.7.0 2.8.0 2.9.0"}
WHEELS_DIR=${WHEELS_DIR:-wheels}
FORCE=${FORCE:-0}    # FORCE=1 时强制重建
mkdir -p "$WHEELS_DIR"

# 每个 Torch 版本选择一个存在的 CUDA 索引（可按需调整）
index_for_torch_version () {
  case "$1" in
    2.3.*) echo "https://download.pytorch.org/whl/cu121" ;;
    2.4.*) echo "https://download.pytorch.org/whl/cu121" ;;
    2.5.*) echo "https://download.pytorch.org/whl/cu124" ;;
    2.6.*) echo "https://download.pytorch.org/whl/cu126" ;;
    2.7.*) echo "https://download.pytorch.org/whl/cu128" ;;
    2.8.*) echo "https://download.pytorch.org/whl/cu128" ;;  # 可换 cu129
    2.9.*) echo "https://download.pytorch.org/whl/cu128" ;;  # 可换 cu129
    *)     echo "https://download.pytorch.org/whl/cu121" ;;
  esac
}

# 检查指定“当前已激活环境”的组合是否已有产物
# 依据 wheel 命名规则中的后缀：+<backend>torch<MM> 以及 -<cp_tag>-<cp_tag>-linux_<arch>
have_wheel_for_current_env () {
  python - <<'PY'
import sys, platform, torch
from packaging.version import parse
py = f"cp{sys.version_info.major}{sys.version_info.minor}"
arch = platform.uname().machine
tver = parse(torch.__version__)
mm = f"{tver.major}{tver.minor}"
backend = ""
if torch.version.cuda:
    backend = "cu" + torch.version.cuda.replace(".", "")
elif getattr(torch.version, "hip", None):
    backend = "rocm" + torch.version.hip.replace(".", "")
else:
    backend = "cpu"  # 极少走到这里
print(py, arch, backend, mm)
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

    # 1) 新建并激活 venv
    ENV_DIR=".venv-py${py//./}-torch${tv%%.*}${tv#*.}"
    "$PYBIN" -m venv "$ENV_DIR"
    source "$ENV_DIR/bin/activate"

    # 2) 安装构建依赖 + 目标 torch（固定 CUDA 索引以避免装到 CPU 轮子）
    python -m pip install -U pip
    python -m pip install setuptools wheel build ninja cmake packaging cpufeature
    IDX="$(index_for_torch_version "$tv")"
    python -m pip install --index-url "$IDX" "torch==$tv"

    # 3) 读取当前环境的关键信息，拼出匹配的 wheel 通配符并检查是否已存在
    read -r CP_TAG ARCH BACKEND MM <<<"$(have_wheel_for_current_env)"
    plat="linux_${ARCH}"
    pattern="${WHEELS_DIR}/ktransformers-*+${BACKEND}torch${MM}*-${CP_TAG}-${CP_TAG}-${plat}.whl"

    if [[ "$FORCE" = "0" ]]; then
      existing=( $pattern )
      if (( ${#existing[@]} > 0 )); then
        echo ">> Found existing wheel, skip: ${existing[0]}"
        deactivate
        continue
      fi
    else
      echo ">> FORCE=1, rebuild even if wheel exists"
    fi

    # 打印对齐信息
    python - <<'PY'
import torch, sys
print(">>> torch:", torch.__version__, "cuda:", torch.version.cuda,
      "cxx11abi:", torch.compiled_with_cxx11_abi())
print(">>> python:", sys.version)
PY

    # ★ 清理所有构建产物（含内嵌 CMake build）
    rm -rf build/ dist/ *.egg-info
    find csrc -type d -name build -prune -exec rm -rf {} +

    # 构建
    KTRANSFORMERS_FORCE_BUILD=TRUE KTRANSFORMERS_DISABLE_PREBUILT=1 \
    python -m build --no-isolation --wheel

    # ★ 验证 wheel 内包含 cpuinfer_ext
    whl="$(ls dist/*.whl)"
    unzip -l "$whl" | grep -E 'cpuinfer_ext.*\.so' >/dev/null || {
      echo "!! cpuinfer_ext missing in $whl"; exit 2;
    }

    mv dist/*.whl wheels/ || true
    deactivate
  done
done

echo "== Wheels saved in ./wheels =="
