#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [SUBCOMMAND] [BUILD_OPTIONS]

Two-step installation in one file. Choose a subcommand:

SUBCOMMANDS:
  deps            Install system prerequisites only
  build           Build and install kt-kernel (no dependency install)
  all             Run deps then build (default when no subcommand)
  -h, --help      Show this help message

BUILD_OPTIONS (for "build" or "all"):
  (none)          Auto-detect CPU and configure automatically (recommended)
  --manual        Skip auto-detection, use manual configuration (see below)
  --no-clean      Do not delete local build/ before building (default cleans)

AUTO-DETECTION (Default):
  The script will automatically detect your CPU and use ALL available features:
  - CPUINFER_CPU_INSTRUCT = NATIVE (uses -march=native)
  - CPUINFER_ENABLE_AMX   = ON/OFF (based on detection)
  - CPUINFER_ENABLE_AVX512_VNNI = ON/OFF (with fallback if OFF)
  - CPUINFER_ENABLE_AVX512_BF16 = ON/OFF (with fallback if OFF)
  - CPUINFER_ENABLE_AVX512_VBMI = ON/OFF (required for FP8 MoE)

  ✓ Best performance on YOUR machine
  ✗ Binary may NOT work on different/older CPUs

  Use this when: Installing for local use only

MANUAL CONFIGURATION:
  Use --manual flag when building for DISTRIBUTION or different machines.
  Set these environment variables before running:

  CPUINFER_CPU_INSTRUCT   - Target CPU instruction set
                            Options: AVX512, AVX2, FANCY, NATIVE
  CPUINFER_ENABLE_AMX     - Enable Intel AMX support
                            Options: ON, OFF

Distribution examples (portable binaries):

┌──────────────────────────────────────────────────────────────────────────┐
│ Configuration          │ Target CPUs              │ Use Case             │
├────────────────────────┼──────────────────────────┼──────────────────────┤
│ AVX512 + AMX=OFF       │ Skylake-X, Ice Lake,     │ General distribution │
│                        │ Cascade Lake, Zen 4      │ (recommended)        │
├────────────────────────┼──────────────────────────┼──────────────────────┤
│ AVX2 + AMX=OFF         │ Haswell (2013) and newer │ Maximum compatibility│
├────────────────────────┼──────────────────────────┼──────────────────────┤
│ FANCY + AMX=OFF        │ Ice Lake+, Zen 4+        │ Modern CPUs only     │
│                        │ (with full AVX512 ext)   │                      │
└────────────────────────┴──────────────────────────┴──────────────────────┘

  Use this when: Building Docker images, PyPI packages, or deploying to clusters

  Example: Build for general distribution
    export CPUINFER_CPU_INSTRUCT=AVX512
    export CPUINFER_ENABLE_AMX=OFF
    $0 build --manual
    # Result: Works on any CPU with AVX512 (2017+)

  Example: Build for maximum compatibility
    export CPUINFER_CPU_INSTRUCT=AVX2
    export CPUINFER_ENABLE_AMX=OFF
    $0 build --manual
    # Result: Works on any CPU with AVX2 (2013+)

Optional variables (with defaults):
  CPUINFER_BUILD_TYPE=Release           Build type (Debug/RelWithDebInfo/Release)
  CPUINFER_PARALLEL=8                   Number of parallel build jobs
  CPUINFER_VERBOSE=1                    Verbose build output (0/1)
  CPUINFER_ENABLE_AVX512_VNNI=ON/OFF    Override VNNI detection (auto if unset)
  CPUINFER_ENABLE_AVX512_BF16=ON/OFF    Override BF16 detection (auto if unset)
  CPUINFER_ENABLE_AVX512_VBMI=ON/OFF    Override VBMI detection (auto if unset)

Software Fallback Support:
  ✓ If VNNI not available: Uses AVX512BW fallback (2-3x slower but works)
  ✓ If BF16 not available: Uses AVX512F fallback (5-10x slower but works)
  → Old CPUs with only AVX512F+BW can run all code (slower but functional)

EOF
  exit 1
}

install_dependencies() {
  echo "Checking and installing system dependencies..."

  # Determine if we need to use sudo
  SUDO=""
  if [ "${EUID:-0}" -ne 0 ]; then
    if command -v sudo &> /dev/null; then
      SUDO="sudo"
    else
      echo "Warning: Not running as root and sudo not found. Package installation may fail."
      echo "Please run as root or install sudo."
    fi
  fi

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
      $SUDO apt update
      $SUDO apt install -y libhwloc-dev pkg-config
      ;;
    fedora|rhel|centos|rocky|almalinux)
      echo "Detected Red Hat-based system. Installing hwloc-devel and pkgconfig..."
      $SUDO dnf install -y hwloc-devel pkgconfig || $SUDO yum install -y hwloc-devel pkgconfig
      ;;
    arch|manjaro)
      echo "Detected Arch-based system. Installing hwloc and pkgconf..."
      $SUDO pacman -S --noconfirm hwloc pkgconf
      ;;
    opensuse*|sles)
      echo "Detected openSUSE-based system. Installing hwloc-devel and pkg-config..."
      $SUDO zypper install -y hwloc-devel pkg-config
      ;;
    *)
      echo "Warning: Unsupported OS '$OS'. Please manually install libhwloc-dev and pkg-config."
      ;;
  esac
}

# Function to detect CPU features
# Returns: "has_amx has_avx512f has_avx512_vnni has_avx512_bf16 has_avx512_vbmi" (space-separated 0/1 values)
detect_cpu_features() {
  local has_amx=0
  local has_avx512f=0
  local has_avx512_vnni=0
  local has_avx512_bf16=0
  local has_avx512_vbmi=0

  if [ -f /proc/cpuinfo ]; then
    local cpu_flags
    cpu_flags=$(grep -m1 "^flags" /proc/cpuinfo | tr ' ' '\n')

    # Check for AMX support on Linux
    if echo "$cpu_flags" | grep -qE "amx_tile|amx_int8|amx_bf16"; then
      has_amx=1
    fi

    # Check for AVX512F (foundation)
    if echo "$cpu_flags" | grep -qE "avx512f"; then
      has_avx512f=1
    fi

    # Check for AVX512_VNNI support
    if echo "$cpu_flags" | grep -qE "avx512_vnni|avx512vnni"; then
      has_avx512_vnni=1
    fi

    # Check for AVX512_BF16 support
    if echo "$cpu_flags" | grep -qE "avx512_bf16|avx512bf16"; then
      has_avx512_bf16=1
    fi

    # Check for AVX512_VBMI support
    if echo "$cpu_flags" | grep -qE "avx512_vbmi|avx512vbmi"; then
      has_avx512_vbmi=1
    fi
  elif [ "$(uname)" = "Darwin" ]; then
    # macOS doesn't have AMX (ARM or Intel without AMX)
    has_amx=0
    has_avx512f=0
    has_avx512_vnni=0
    has_avx512_bf16=0
    has_avx512_vbmi=0
  fi

  echo "$has_amx $has_avx512f $has_avx512_vnni $has_avx512_bf16 $has_avx512_vbmi"
}

build_step() {
  # Parse build-only flags from arguments to this function
  local MANUAL_MODE=0
  local CLEAN_BUILD=1
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --manual) MANUAL_MODE=1; shift ;;
      --no-clean) CLEAN_BUILD=0; shift ;;
      -h|--help) usage ;;
      *) break ;;
    esac
  done

  # Clean local build directory to ensure a fresh CMake/configure
  local REPO_ROOT
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  if [[ "$CLEAN_BUILD" -eq 1 ]]; then
    if [[ -d "$REPO_ROOT/build" ]]; then
      echo "Cleaning previous build directory: $REPO_ROOT/build"
      rm -rf "$REPO_ROOT/build"
    fi
  else
    echo "Skipping clean of $REPO_ROOT/build (requested by --no-clean)"
  fi

  if [ "$MANUAL_MODE" = "0" ]; then
  # Auto-detection mode
  echo "=========================================="
  echo "Auto-detecting CPU capabilities..."
  echo "=========================================="
  echo ""

  # detect_cpu_features returns "has_amx has_avx512f has_avx512_vnni has_avx512_bf16 has_avx512_vbmi"
  CPU_FEATURES=$(detect_cpu_features)
  HAS_AMX=$(echo "$CPU_FEATURES" | cut -d' ' -f1)
  HAS_AVX512F=$(echo "$CPU_FEATURES" | cut -d' ' -f2)
  HAS_AVX512_VNNI=$(echo "$CPU_FEATURES" | cut -d' ' -f3)
  HAS_AVX512_BF16=$(echo "$CPU_FEATURES" | cut -d' ' -f4)
  HAS_AVX512_VBMI=$(echo "$CPU_FEATURES" | cut -d' ' -f5)

  export CPUINFER_CPU_INSTRUCT=NATIVE

  if [ "$HAS_AMX" = "1" ]; then
    echo "✓ AMX instructions detected"
    export CPUINFER_ENABLE_AMX=ON
    echo ""
    echo "Configuration: NATIVE + AMX=ON"
    echo "  ✓ Best performance on this machine"
    echo "  ✗ Binary requires Sapphire Rapids or newer CPU"
  else
    echo "ℹ AMX instructions not detected"
    export CPUINFER_ENABLE_AMX=OFF
    echo ""
    echo "Configuration: NATIVE + AMX=OFF"
    echo "  ✓ Using AVX512/AVX2 instructions"
  fi

  echo ""
  echo "  ⚠️  IMPORTANT: This binary is optimized for THIS CPU only"
  echo "     To build portable binaries for distribution, use:"
  echo "       export CPUINFER_CPU_INSTRUCT=AVX512  # or AVX2"
  echo "       export CPUINFER_ENABLE_AMX=OFF"
  echo "       ./install.sh build --manual"

  # Fine-grained AVX512 subset detection (with fallback support)
  echo ""
  echo "AVX512 Feature Detection:"

  # AVX512F: Foundation (required for all AVX512 variants)
  if [ "$HAS_AVX512F" = "1" ]; then
    echo "  AVX512F: ✓ Detected (foundation)"
  else
    echo "  AVX512F: ✗ Not detected (AVX512 not available)"
  fi

  # VNNI: Check if user manually set it, otherwise auto-detect
  if [ -n "${CPUINFER_ENABLE_AVX512_VNNI:-}" ]; then
    echo "  VNNI: User override = $CPUINFER_ENABLE_AVX512_VNNI"
  else
    if [ "$HAS_AVX512_VNNI" = "1" ]; then
      echo "  VNNI: ✓ Detected (hardware acceleration enabled)"
      export CPUINFER_ENABLE_AVX512_VNNI=ON
    else
      echo "  VNNI: ✗ Not detected (will use software fallback, 2-3x slower)"
      export CPUINFER_ENABLE_AVX512_VNNI=OFF
    fi
  fi

  # BF16: Check if user manually set it, otherwise auto-detect
  if [ -n "${CPUINFER_ENABLE_AVX512_BF16:-}" ]; then
    echo "  BF16: User override = $CPUINFER_ENABLE_AVX512_BF16"
  else
    if [ "$HAS_AVX512_BF16" = "1" ]; then
      echo "  BF16: ✓ Detected (hardware acceleration enabled)"
      export CPUINFER_ENABLE_AVX512_BF16=ON
    else
      echo "  BF16: ✗ Not detected (will use software fallback, 5-10x slower)"
      export CPUINFER_ENABLE_AVX512_BF16=OFF
    fi
  fi

  # VBMI: Check if user manually set it, otherwise auto-detect
  if [ -n "${CPUINFER_ENABLE_AVX512_VBMI:-}" ]; then
    echo "  VBMI: User override = $CPUINFER_ENABLE_AVX512_VBMI"
  else
    if [ "$HAS_AVX512_VBMI" = "1" ]; then
      echo "  VBMI: ✓ Detected (byte permutation enabled)"
      export CPUINFER_ENABLE_AVX512_VBMI=ON
    else
      echo "  VBMI: ✗ Not detected (FP8 MoE may not work)"
      export CPUINFER_ENABLE_AVX512_VBMI=OFF
    fi
  fi

  echo ""
  echo "  Note: Software fallbacks ensure all code works on older CPUs"
  echo "  Note: FP8 MoE requires AVX512F + BF16 + VNNI + VBMI"
  echo "  Tip: Override with CPUINFER_ENABLE_AVX512_[VNNI|BF16|VBMI]=ON/OFF"

  echo ""
  echo "To use manual configuration instead, run: $0 build --manual"
  echo ""
  else
  # Manual mode - validate user configuration (no exports)
  if [ -z "$CPUINFER_CPU_INSTRUCT" ] || [ -z "$CPUINFER_ENABLE_AMX" ]; then
    echo "Error: Manual mode requires CPUINFER_CPU_INSTRUCT and CPUINFER_ENABLE_AMX to be set."
    echo ""
    usage
  fi

  # Validate CPUINFER_CPU_INSTRUCT
  case "$CPUINFER_CPU_INSTRUCT" in
    NATIVE|FANCY|AVX512|AVX2)
      ;;
    *)
      echo "Error: Invalid CPUINFER_CPU_INSTRUCT='$CPUINFER_CPU_INSTRUCT'"
      echo "Must be one of: NATIVE, FANCY, AVX512, AVX2"
      exit 1
      ;;
  esac

  # Validate CPUINFER_ENABLE_AMX
  case "$CPUINFER_ENABLE_AMX" in
    ON|OFF)
      ;;
    *)
      echo "Error: Invalid CPUINFER_ENABLE_AMX='$CPUINFER_ENABLE_AMX'"
      echo "Must be either: ON or OFF"
      exit 1
      ;;
  esac

  # Warn about problematic configuration
  if [ "$CPUINFER_CPU_INSTRUCT" = "NATIVE" ] && [ "$CPUINFER_ENABLE_AMX" = "OFF" ]; then
    CPU_FEATURES=$(detect_cpu_features)
    HAS_AMX=$(echo "$CPU_FEATURES" | cut -d' ' -f1)
    if [ "$HAS_AMX" = "1" ]; then
      echo "=========================================="
      echo "⚠️  WARNING: Risky Configuration"
      echo "=========================================="
      echo ""
      echo "Your configuration:"
      echo "  CPUINFER_CPU_INSTRUCT = NATIVE"
      echo "  CPUINFER_ENABLE_AMX   = OFF"
      echo ""
      echo "Your CPU HAS AMX support!"
      echo ""
      echo "Problem:"
      echo "  • NATIVE uses -march=native which auto-enables ALL CPU features"
      echo "  • This may IGNORE your AMX=OFF setting"
      echo "  • The binary may still contain AMX instructions"
      echo ""
      echo "Recommended fixes:"
      echo "  1) For portable build (recommended for distribution):"
      echo "       export CPUINFER_CPU_INSTRUCT=AVX512"
      echo ""
      echo "  2) If you want best performance on this CPU:"
      echo "       export CPUINFER_ENABLE_AMX=ON"
      echo ""
      read -p "Continue with risky configuration? (y/N) " -n 1 -r
      echo
      if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
      fi
    fi
  fi

# Close MANUAL_MODE conditional
  fi

# Set defaults for optional variables
export CPUINFER_BUILD_TYPE=${CPUINFER_BUILD_TYPE:-Release}
export CPUINFER_PARALLEL=${CPUINFER_PARALLEL:-8}
export CPUINFER_VERBOSE=${CPUINFER_VERBOSE:-1}

echo "=========================================="
echo "Building kt-kernel with configuration:"
echo "=========================================="
echo "  CPUINFER_CPU_INSTRUCT        = $CPUINFER_CPU_INSTRUCT"
echo "  CPUINFER_ENABLE_AMX          = $CPUINFER_ENABLE_AMX"
echo "  CPUINFER_ENABLE_AVX512_VNNI  = ${CPUINFER_ENABLE_AVX512_VNNI:-AUTO}"
echo "  CPUINFER_ENABLE_AVX512_BF16  = ${CPUINFER_ENABLE_AVX512_BF16:-AUTO}"
echo "  CPUINFER_ENABLE_AVX512_VBMI  = ${CPUINFER_ENABLE_AVX512_VBMI:-AUTO}"
echo "  CPUINFER_BUILD_TYPE          = $CPUINFER_BUILD_TYPE"
echo "  CPUINFER_PARALLEL            = $CPUINFER_PARALLEL"
echo ""

pip install . -v
}

# Subcommand dispatcher: default to "all"
SUBCMD="all"
if [[ $# -gt 0 ]]; then
  case "$1" in
    deps|build|all) SUBCMD="$1"; shift ;;
    -h|--help) usage ;;
    *) SUBCMD="build" ;; # backward compatibility: flags-only => build
  esac
fi

case "$SUBCMD" in
  deps)
    install_dependencies
    ;;
  build)
    build_step "$@"
    ;;
  all)
    install_dependencies
    build_step "$@"
    ;;
esac
