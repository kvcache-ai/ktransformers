#!/usr/bin/env bash
set -e

usage() {
  echo "Usage: $0 [avx|amx]"
  exit 1
}

if [ $# -ne 1 ]; then
  usage
fi

MODE="$1"
case "$MODE" in
  avx)
    export CPUINFER_CPU_INSTRUCT=AVX2
    export CPUINFER_ENABLE_AMX=OFF
    ;;
  amx)
    export CPUINFER_CPU_INSTRUCT=AMX512
    export CPUINFER_ENABLE_AMX=ON
    ;;
  *)
    echo "Error: unknown mode '$MODE'"
    usage
    ;;
esac

export CPUINFER_BUILD_TYPE=Release
export CPUINFER_PARALLEL=8
export CPUINFER_VERBOSE=1

echo "Building in mode: $MODE"
echo "Environment:"
echo "  CPUINFER_CPU_INSTRUCT=$CPUINFER_CPU_INSTRUCT"
echo "  CPUINFER_ENABLE_AMX=$CPUINFER_ENABLE_AMX"
echo "  CPUINFER_BUILD_TYPE=$CPUINFER_BUILD_TYPE"
echo "  CPUINFER_PARALLEL=$CPUINFER_PARALLEL"
echo "  CPUINFER_VERBOSE=$CPUINFER_VERBOSE"

pip install -e . -v

