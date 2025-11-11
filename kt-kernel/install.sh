#!/usr/bin/env bash
set -e

install_dependencies() {
  echo "Checking and installing system dependencies..."

  if command -v conda &> /dev/null; then
    echo "Installing cmake via conda..."
    conda install -y cmake
  else
    echo "Warning: conda not found. Skipping cmake installation via conda."
    echo "Please install conda or manually install cmake."
  fi

  # Detect OS type
  if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
  elif [ -f /etc/debian_version ]; then
    OS="debian"
  elif [ -f /etc/redhat-release ]; then
    OS="rhel"
  else
    echo "Warning: Unable to detect OS type. Skipping dependency installation."
    return 0
  fi

  # Install dependencies based on OS
  case "$OS" in
    debian|ubuntu|linuxmint|pop)
      echo "Detected Debian-based system. Installing libhwloc-dev and pkg-config..."
      sudo apt update
      sudo apt install -y libhwloc-dev pkg-config
      ;;
    fedora|rhel|centos|rocky|almalinux)
      echo "Detected Red Hat-based system. Installing hwloc-devel and pkgconfig..."
      sudo dnf install -y hwloc-devel pkgconfig || sudo yum install -y hwloc-devel pkgconfig
      ;;
    arch|manjaro)
      echo "Detected Arch-based system. Installing hwloc and pkgconf..."
      sudo pacman -S --noconfirm hwloc pkgconf
      ;;
    opensuse*|sles)
      echo "Detected openSUSE-based system. Installing hwloc-devel and pkg-config..."
      sudo zypper install -y hwloc-devel pkg-config
      ;;
    *)
      echo "Warning: Unsupported OS '$OS'. Please manually install libhwloc-dev and pkg-config."
      ;;
  esac
}

install_dependencies

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

CMAKE_ARGS="-D CMAKE_CUDA_COMPILER=$(which nvcc)" pip install . -v

