"""
Environment detection utilities for kt-cli.

Provides functions to detect:
- Virtual environment managers (conda, venv, uv, mamba)
- Python version and packages
- CUDA and GPU information
- System resources (CPU, RAM, disk)
"""

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EnvManager:
    """Information about an environment manager."""

    name: str
    version: str
    path: str


@dataclass
class GPUInfo:
    """Information about a GPU."""

    index: int
    name: str
    vram_gb: float
    cuda_capability: Optional[str] = None


@dataclass
class CPUInfo:
    """Information about the CPU."""

    name: str
    cores: int
    threads: int
    numa_nodes: int
    instruction_sets: list[str] = field(default_factory=list)  # AVX, AVX2, AVX512, AMX, etc.
    numa_info: dict = field(default_factory=dict)  # node -> cpus mapping


@dataclass
class MemoryInfo:
    """Information about system memory."""

    total_gb: float
    available_gb: float
    frequency_mhz: Optional[int] = None
    channels: Optional[int] = None
    type: Optional[str] = None  # DDR4, DDR5, etc.


@dataclass
class SystemInfo:
    """Complete system information."""

    python_version: str
    platform: str
    cuda_version: Optional[str]
    gpus: list[GPUInfo]
    cpu: CPUInfo
    ram_gb: float
    env_managers: list[EnvManager]


def run_command(cmd: list[str], timeout: int = 10) -> Optional[str]:
    """Run a command and return its output, or None if it fails."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def detect_env_managers() -> list[EnvManager]:
    """Detect available virtual environment managers."""
    managers = []

    # Check conda
    conda_path = shutil.which("conda")
    if conda_path:
        version = run_command(["conda", "--version"])
        if version:
            # "conda 24.1.0" -> "24.1.0"
            version = version.split()[-1] if version else "unknown"
            managers.append(EnvManager(name="conda", version=version, path=conda_path))

    # Check mamba
    mamba_path = shutil.which("mamba")
    if mamba_path:
        version = run_command(["mamba", "--version"])
        if version:
            # First line: "mamba 1.5.0"
            version = version.split("\n")[0].split()[-1] if version else "unknown"
            managers.append(EnvManager(name="mamba", version=version, path=mamba_path))

    # Check uv
    uv_path = shutil.which("uv")
    if uv_path:
        version = run_command(["uv", "--version"])
        if version:
            # "uv 0.5.0" -> "0.5.0"
            version = version.split()[-1] if version else "unknown"
            managers.append(EnvManager(name="uv", version=version, path=uv_path))

    # Check if venv is available (built into Python)
    try:
        import venv  # noqa: F401

        managers.append(
            EnvManager(name="venv", version="builtin", path="python -m venv")
        )
    except ImportError:
        pass

    return managers


def check_docker() -> Optional[EnvManager]:
    """Check if Docker is available."""
    docker_path = shutil.which("docker")
    if docker_path:
        version = run_command(["docker", "--version"])
        if version:
            # "Docker version 24.0.7, build afdd53b"
            parts = version.split()
            version = parts[2].rstrip(",") if len(parts) > 2 else "unknown"
            return EnvManager(name="docker", version=version, path=docker_path)
    return None


def check_kt_env_exists(manager: str, env_name: str = "kt") -> bool:
    """Check if a kt environment exists for the given manager."""
    if manager == "conda" or manager == "mamba":
        result = run_command([manager, "env", "list"])
        if result:
            # Check if env_name appears as a separate word in the output
            for line in result.split("\n"):
                parts = line.split()
                if parts and parts[0] == env_name:
                    return True
    elif manager == "uv":
        # uv uses .venv in the project directory or ~/.local/share/uv/envs/
        venv_path = Path.home() / ".local" / "share" / "uv" / "envs" / env_name
        if venv_path.exists():
            return True
        # Also check current directory
        if Path(env_name).exists() and (Path(env_name) / "bin" / "python").exists():
            return True
    elif manager == "venv":
        # Check common locations
        venv_path = Path.home() / ".virtualenvs" / env_name
        if venv_path.exists():
            return True
        if Path(env_name).exists() and (Path(env_name) / "bin" / "python").exists():
            return True

    return False


def get_kt_env_path(manager: str, env_name: str = "kt") -> Optional[Path]:
    """Get the path to the kt environment."""
    if manager == "conda" or manager == "mamba":
        result = run_command([manager, "env", "list"])
        if result:
            for line in result.split("\n"):
                parts = line.split()
                if parts and parts[0] == env_name:
                    # The path is the last part
                    return Path(parts[-1])
    elif manager == "uv":
        venv_path = Path.home() / ".local" / "share" / "uv" / "envs" / env_name
        if venv_path.exists():
            return venv_path
    elif manager == "venv":
        venv_path = Path.home() / ".virtualenvs" / env_name
        if venv_path.exists():
            return venv_path

    return None


def detect_cuda_version() -> Optional[str]:
    """Detect CUDA version from nvidia-smi or nvcc."""
    # Try nvidia-smi first
    nvidia_smi = run_command(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if nvidia_smi:
        # Get CUDA version from nvidia-smi
        full_output = run_command(["nvidia-smi"])
        if full_output:
            for line in full_output.split("\n"):
                if "CUDA Version:" in line:
                    # "| CUDA Version: 12.1     |"
                    parts = line.split("CUDA Version:")
                    if len(parts) > 1:
                        version = parts[1].strip().split()[0]
                        return version

    # Try nvcc
    nvcc_output = run_command(["nvcc", "--version"])
    if nvcc_output:
        for line in nvcc_output.split("\n"):
            if "release" in line.lower():
                # "Cuda compilation tools, release 12.1, V12.1.105"
                parts = line.split("release")
                if len(parts) > 1:
                    version = parts[1].strip().split(",")[0].strip()
                    return version

    return None


def detect_gpus() -> list[GPUInfo]:
    """Detect available NVIDIA GPUs."""
    gpus = []

    nvidia_smi = run_command([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total",
        "--format=csv,noheader,nounits"
    ])

    if nvidia_smi:
        for line in nvidia_smi.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    index = int(parts[0])
                    name = parts[1]
                    vram_mb = float(parts[2])
                    vram_gb = round(vram_mb / 1024, 1)
                    gpus.append(GPUInfo(index=index, name=name, vram_gb=vram_gb))
                except (ValueError, IndexError):
                    continue

    return gpus


def detect_cpu_info() -> CPUInfo:
    """Detect CPU information including instruction sets and NUMA topology."""
    name = "Unknown"
    cores = os.cpu_count() or 1
    threads = cores
    numa_nodes = 1
    instruction_sets: list[str] = []
    numa_info: dict[str, list[int]] = {}

    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", "r") as f:
                content = f.read()

            # Get CPU name
            for line in content.split("\n"):
                if line.startswith("model name"):
                    name = line.split(":")[1].strip()
                    break

            # Get physical cores vs threads
            cpu_cores = content.count("processor\t:")
            if cpu_cores > 0:
                threads = cpu_cores

            siblings = None
            cores_per = None
            for line in content.split("\n"):
                if "siblings" in line:
                    siblings = int(line.split(":")[1].strip())
                if "cpu cores" in line:
                    cores_per = int(line.split(":")[1].strip())
            if siblings and cores_per:
                cores = threads // (siblings // cores_per) if siblings > cores_per else threads

            # Get instruction sets from flags
            for line in content.split("\n"):
                if line.startswith("flags"):
                    flags = line.split(":")[1].strip().split()
                    instruction_sets = _parse_cpu_flags(flags)
                    break

        except (OSError, IOError, ValueError):
            pass

        # Get NUMA topology
        numa_path = Path("/sys/devices/system/node")
        if numa_path.exists():
            numa_dirs = [d for d in numa_path.iterdir() if d.name.startswith("node")]
            numa_nodes = len(numa_dirs)

            for node_dir in numa_dirs:
                node_name = node_dir.name  # e.g., "node0"
                cpulist_path = node_dir / "cpulist"
                if cpulist_path.exists():
                    try:
                        cpulist = cpulist_path.read_text().strip()
                        numa_info[node_name] = _parse_cpu_list(cpulist)
                    except (OSError, IOError):
                        pass

    elif platform.system() == "Darwin":
        # macOS
        name_output = run_command(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name_output:
            name = name_output.strip()
        cores_output = run_command(["sysctl", "-n", "hw.physicalcpu"])
        if cores_output:
            cores = int(cores_output.strip())
        threads_output = run_command(["sysctl", "-n", "hw.logicalcpu"])
        if threads_output:
            threads = int(threads_output.strip())

        # Get instruction sets on macOS
        features_output = run_command(["sysctl", "-n", "machdep.cpu.features"])
        if features_output:
            flags = features_output.lower().split()
            instruction_sets = _parse_cpu_flags(flags)

    return CPUInfo(
        name=name,
        cores=cores,
        threads=threads,
        numa_nodes=numa_nodes,
        instruction_sets=instruction_sets,
        numa_info=numa_info,
    )


def _parse_cpu_flags(flags: list[str]) -> list[str]:
    """Parse CPU flags to extract relevant instruction sets for KTransformers."""
    # Instruction sets important for KTransformers/kt-kernel
    relevant_instructions = {
        # Basic SIMD
        "sse": "SSE",
        "sse2": "SSE2",
        "sse3": "SSE3",
        "ssse3": "SSSE3",
        "sse4_1": "SSE4.1",
        "sse4_2": "SSE4.2",
        # AVX family
        "avx": "AVX",
        "avx2": "AVX2",
        "avx512f": "AVX512F",
        "avx512bw": "AVX512BW",
        "avx512vl": "AVX512VL",
        "avx512dq": "AVX512DQ",
        "avx512cd": "AVX512CD",
        "avx512vnni": "AVX512VNNI",
        "avx512_bf16": "AVX512BF16",
        "avx512_fp16": "AVX512FP16",
        "avx_vnni": "AVX-VNNI",
        # AMX (Advanced Matrix Extensions) - Intel
        "amx_tile": "AMX-TILE",
        "amx_bf16": "AMX-BF16",
        "amx_int8": "AMX-INT8",
        "amx_fp16": "AMX-FP16",
        # Other relevant
        "fma": "FMA",
        "f16c": "F16C",
        "bmi1": "BMI1",
        "bmi2": "BMI2",
    }

    found = []
    flags_lower = {f.lower() for f in flags}

    for flag, display_name in relevant_instructions.items():
        if flag in flags_lower:
            found.append(display_name)

    # Sort by importance for display
    priority = ["AMX-INT8", "AMX-BF16", "AMX-FP16", "AMX-TILE",
                "AVX512BF16", "AVX512VNNI", "AVX512F", "AVX512BW", "AVX512VL",
                "AVX2", "AVX", "FMA", "SSE4.2"]
    result = []
    for p in priority:
        if p in found:
            result.append(p)
            found.remove(p)
    result.extend(sorted(found))  # Add remaining

    return result


def _parse_cpu_list(cpulist: str) -> list[int]:
    """Parse CPU list string like '0-3,8-11' to list of CPU IDs."""
    cpus = []
    for part in cpulist.split(","):
        if "-" in part:
            start, end = part.split("-")
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(part))
    return cpus


def detect_memory_info() -> MemoryInfo:
    """Detect detailed memory information including frequency and type."""
    total_gb = detect_ram_gb()
    available_gb = detect_available_ram_gb()
    frequency_mhz: Optional[int] = None
    channels: Optional[int] = None
    mem_type: Optional[str] = None

    if platform.system() == "Linux":
        # Try dmidecode first (requires root, but may have cached data)
        dmidecode_output = run_command(["sudo", "dmidecode", "-t", "memory"])
        if dmidecode_output:
            frequency_mhz, mem_type, channels = _parse_dmidecode_memory(dmidecode_output)

        # Fallback: try to read from /sys or /proc
        if frequency_mhz is None:
            frequency_mhz = _detect_memory_frequency_sysfs()

    elif platform.system() == "Darwin":
        # macOS - use system_profiler
        mem_output = run_command(["system_profiler", "SPMemoryDataType"])
        if mem_output:
            frequency_mhz, mem_type = _parse_macos_memory(mem_output)

    return MemoryInfo(
        total_gb=total_gb,
        available_gb=available_gb,
        frequency_mhz=frequency_mhz,
        channels=channels,
        type=mem_type,
    )


def _parse_dmidecode_memory(output: str) -> tuple[Optional[int], Optional[str], Optional[int]]:
    """Parse dmidecode memory output."""
    frequency_mhz: Optional[int] = None
    mem_type: Optional[str] = None
    dimm_count = 0

    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("Speed:") and "MHz" in line:
            try:
                # "Speed: 4800 MHz" or "Speed: 4800 MT/s"
                parts = line.split(":")[1].strip().split()
                freq = int(parts[0])
                if freq > 0 and (frequency_mhz is None or freq > frequency_mhz):
                    frequency_mhz = freq
            except (ValueError, IndexError):
                pass
        elif line.startswith("Type:") and mem_type is None:
            type_val = line.split(":")[1].strip()
            if type_val and type_val != "Unknown":
                mem_type = type_val
        elif line.startswith("Size:") and "MB" in line or "GB" in line:
            dimm_count += 1

    return frequency_mhz, mem_type, dimm_count if dimm_count > 0 else None


def _detect_memory_frequency_sysfs() -> Optional[int]:
    """Try to detect memory frequency from sysfs."""
    # This is a fallback and may not work on all systems
    try:
        # Try reading from edac
        edac_path = Path("/sys/devices/system/edac/mc")
        if edac_path.exists():
            for mc_dir in edac_path.iterdir():
                freq_file = mc_dir / "mc_config"
                if freq_file.exists():
                    content = freq_file.read_text()
                    # Parse for frequency information
                    # Format varies by system
                    pass
    except (OSError, IOError):
        pass

    return None


def _parse_macos_memory(output: str) -> tuple[Optional[int], Optional[str]]:
    """Parse macOS system_profiler memory output."""
    frequency_mhz: Optional[int] = None
    mem_type: Optional[str] = None

    for line in output.split("\n"):
        line = line.strip()
        if "Speed:" in line:
            try:
                parts = line.split(":")[1].strip().split()
                frequency_mhz = int(parts[0])
            except (ValueError, IndexError):
                pass
        elif "Type:" in line:
            mem_type = line.split(":")[1].strip()

    return frequency_mhz, mem_type


def detect_ram_gb() -> float:
    """Detect total system RAM in GB."""
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # "MemTotal:       32780516 kB"
                        kb = int(line.split()[1])
                        return round(kb / 1024 / 1024, 1)
        except (OSError, IOError, ValueError):
            pass
    elif platform.system() == "Darwin":
        mem_output = run_command(["sysctl", "-n", "hw.memsize"])
        if mem_output:
            return round(int(mem_output) / 1024 / 1024 / 1024, 1)

    # Fallback
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 1)
    except ImportError:
        return 0.0


def detect_available_ram_gb() -> float:
    """Detect available system RAM in GB."""
    if platform.system() == "Linux":
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        kb = int(line.split()[1])
                        return round(kb / 1024 / 1024, 1)
        except (OSError, IOError, ValueError):
            pass

    # Fallback
    try:
        import psutil
        return round(psutil.virtual_memory().available / 1024 / 1024 / 1024, 1)
    except ImportError:
        return 0.0


def detect_disk_space_gb(path: str = "/") -> tuple[float, float]:
    """Detect disk space (available, total) in GB for the given path."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        return round(free / 1024 / 1024 / 1024, 1), round(total / 1024 / 1024 / 1024, 1)
    except (OSError, IOError):
        return 0.0, 0.0


def get_installed_package_version(package_name: str) -> Optional[str]:
    """Get the version of an installed Python package."""
    try:
        from importlib.metadata import version
        return version(package_name)
    except Exception:
        return None


def get_system_info() -> SystemInfo:
    """Gather complete system information."""
    return SystemInfo(
        python_version=platform.python_version(),
        platform=f"{platform.system()} {platform.release()}",
        cuda_version=detect_cuda_version(),
        gpus=detect_gpus(),
        cpu=detect_cpu_info(),
        ram_gb=detect_ram_gb(),
        env_managers=detect_env_managers(),
    )


def is_in_virtual_env() -> bool:
    """Check if currently running inside a virtual environment."""
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or os.environ.get("VIRTUAL_ENV") is not None
        or os.environ.get("CONDA_PREFIX") is not None
    )


def get_current_env_name() -> Optional[str]:
    """Get the name of the current virtual environment."""
    if os.environ.get("CONDA_DEFAULT_ENV"):
        return os.environ["CONDA_DEFAULT_ENV"]
    if os.environ.get("VIRTUAL_ENV"):
        return Path(os.environ["VIRTUAL_ENV"]).name
    return None


# Import sys for is_in_virtual_env
import sys  # noqa: E402


@dataclass
class StorageLocation:
    """Information about a storage location."""

    path: str
    available_gb: float
    total_gb: float
    is_writable: bool
    mount_point: str


def scan_storage_locations(min_size_gb: float = 50.0) -> list[StorageLocation]:
    """
    Scan system for potential model storage locations.

    Looks for:
    - Large mounted filesystems (> min_size_gb)
    - Common model storage paths
    - User home directory

    Args:
        min_size_gb: Minimum available space in GB to consider

    Returns:
        List of StorageLocation sorted by available space (descending)
    """
    locations: dict[str, StorageLocation] = {}  # Use dict to deduplicate by path

    # Get all mount points from /proc/mounts (Linux)
    mount_points = _get_mount_points()

    for mount_point in mount_points:
        try:
            available_gb, total_gb = detect_disk_space_gb(mount_point)

            # Skip small or pseudo filesystems
            if total_gb < 10:
                continue

            # Check if writable
            is_writable = os.access(mount_point, os.W_OK)

            # Create potential model paths under this mount
            potential_paths = _get_potential_model_paths(mount_point)

            for path in potential_paths:
                if path in locations:
                    continue

                # Get actual available space for this path
                path_available, path_total = detect_disk_space_gb(path)

                if path_available >= min_size_gb:
                    path_writable = os.access(path, os.W_OK) if os.path.exists(path) else is_writable
                    locations[path] = StorageLocation(
                        path=path,
                        available_gb=path_available,
                        total_gb=path_total,
                        is_writable=path_writable,
                        mount_point=mount_point,
                    )
        except (OSError, IOError):
            continue

    # Also check common model storage locations
    common_paths = [
        str(Path.home() / ".ktransformers" / "models"),
        str(Path.home() / "models"),
        str(Path.home() / ".cache" / "huggingface"),
        "/data/models",
        "/models",
        "/opt/models",
    ]

    for path in common_paths:
        if path in locations:
            continue
        try:
            # Check if parent exists for paths that don't exist yet
            check_path = path
            while not os.path.exists(check_path) and check_path != "/":
                check_path = str(Path(check_path).parent)

            if os.path.exists(check_path):
                available_gb, total_gb = detect_disk_space_gb(check_path)
                if available_gb >= min_size_gb:
                    is_writable = os.access(check_path, os.W_OK)
                    locations[path] = StorageLocation(
                        path=path,
                        available_gb=available_gb,
                        total_gb=total_gb,
                        is_writable=is_writable,
                        mount_point=check_path,
                    )
        except (OSError, IOError):
            continue

    # Sort by available space descending, then by path
    sorted_locations = sorted(
        locations.values(),
        key=lambda x: (-x.available_gb, x.path)
    )

    # Filter to only writable locations
    return [loc for loc in sorted_locations if loc.is_writable]


def _get_mount_points() -> list[str]:
    """Get all mount points on the system."""
    mount_points = []

    if platform.system() == "Linux":
        try:
            with open("/proc/mounts", "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        mount_point = parts[1]
                        fs_type = parts[2] if len(parts) > 2 else ""

                        # Skip pseudo filesystems
                        skip_fs = {"proc", "sysfs", "devpts", "tmpfs", "cgroup",
                                   "cgroup2", "pstore", "securityfs", "debugfs",
                                   "hugetlbfs", "mqueue", "fusectl", "configfs",
                                   "devtmpfs", "efivarfs", "autofs", "binfmt_misc",
                                   "overlay", "nsfs", "tracefs"}
                        if fs_type in skip_fs:
                            continue

                        # Skip paths that are clearly system paths
                        skip_prefixes = ("/sys", "/proc", "/dev", "/run/user")
                        if any(mount_point.startswith(p) for p in skip_prefixes):
                            continue

                        mount_points.append(mount_point)
        except (OSError, IOError):
            pass

    # Always include home and root
    mount_points.extend([str(Path.home()), "/"])

    # Deduplicate while preserving order
    seen = set()
    unique_mounts = []
    for mp in mount_points:
        if mp not in seen:
            seen.add(mp)
            unique_mounts.append(mp)

    return unique_mounts


def _get_potential_model_paths(mount_point: str) -> list[str]:
    """Get potential model storage paths under a mount point."""
    paths = []

    # The mount point itself (for dedicated data drives)
    if mount_point not in ("/", "/home"):
        paths.append(mount_point)
        paths.append(os.path.join(mount_point, "models"))

    # If it's under home, suggest standard locations
    home = str(Path.home())
    if mount_point == home or mount_point == "/home":
        paths.append(os.path.join(home, ".ktransformers", "models"))
        paths.append(os.path.join(home, "models"))

    # For root mount, suggest /data or /opt
    if mount_point == "/":
        paths.extend(["/data/models", "/opt/models"])

    # Check for common data directories on this mount
    for subdir in ["data", "models", "ai", "llm", "huggingface"]:
        potential = os.path.join(mount_point, subdir)
        if os.path.exists(potential) and os.path.isdir(potential):
            paths.append(potential)

    return paths


def format_size_gb(size_gb: float) -> str:
    """Format size in GB to human readable string."""
    if size_gb >= 1000:
        return f"{size_gb / 1000:.1f}TB"
    return f"{size_gb:.1f}GB"


@dataclass
class LocalModel:
    """Information about a locally detected model."""

    name: str
    path: str
    size_gb: float
    model_type: str  # "huggingface", "gguf", "safetensors"
    has_config: bool
    file_count: int


def scan_local_models(search_paths: list[str], max_depth: int = 3) -> list[LocalModel]:
    """
    Scan directories for locally downloaded models.

    Looks for:
    - Directories with config.json (HuggingFace format)
    - Directories with .safetensors files
    - Directories with .gguf files

    Args:
        search_paths: List of paths to search
        max_depth: Maximum directory depth to search

    Returns:
        List of LocalModel sorted by size (descending)
    """
    models: dict[str, LocalModel] = {}  # Use path as key to deduplicate

    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue

        _scan_directory_for_models(search_path, models, current_depth=0, max_depth=max_depth)

    # Sort by size descending
    return sorted(models.values(), key=lambda x: -x.size_gb)


def _scan_directory_for_models(
    directory: str,
    models: dict[str, LocalModel],
    current_depth: int,
    max_depth: int
) -> None:
    """Recursively scan a directory for models."""
    if current_depth > max_depth:
        return

    try:
        entries = list(os.scandir(directory))
    except (PermissionError, OSError):
        return

    # Check if this directory is a model
    model = _detect_model_in_directory(directory, entries)
    if model:
        models[model.path] = model
        return  # Don't scan subdirectories of a model

    # Scan subdirectories
    for entry in entries:
        if entry.is_dir() and not entry.name.startswith("."):
            _scan_directory_for_models(entry.path, models, current_depth + 1, max_depth)


def _detect_model_in_directory(directory: str, entries: list) -> Optional[LocalModel]:
    """Detect if a directory contains a model."""
    entry_names = {e.name for e in entries}

    has_config = "config.json" in entry_names
    safetensor_files = [e for e in entries if e.name.endswith(".safetensors") and e.is_file()]
    gguf_files = [e for e in entries if e.name.endswith(".gguf") and e.is_file()]

    # Determine model type
    model_type = None
    if has_config and safetensor_files:
        model_type = "huggingface"
    elif gguf_files:
        model_type = "gguf"
    elif safetensor_files:
        model_type = "safetensors"
    elif has_config:
        # Config but no weights - might be incomplete
        # Check for other model-related files
        model_files = {"model.safetensors.index.json", "pytorch_model.bin.index.json",
                       "model.safetensors", "pytorch_model.bin"}
        if entry_names & model_files:
            model_type = "huggingface"

    if not model_type:
        return None

    # Calculate directory size
    size_bytes = _get_directory_size(directory)
    size_gb = size_bytes / (1024 ** 3)

    # Skip very small directories (likely incomplete or config-only)
    if size_gb < 0.1:
        return None

    # Get model name from directory name
    name = os.path.basename(directory)

    # Count model files
    file_count = len(safetensor_files) + len(gguf_files)
    if not file_count:
        # Count .bin files as fallback
        file_count = len([e for e in entries if e.name.endswith(".bin") and e.is_file()])

    return LocalModel(
        name=name,
        path=directory,
        size_gb=round(size_gb, 1),
        model_type=model_type,
        has_config=has_config,
        file_count=file_count,
    )


def _get_directory_size(directory: str) -> int:
    """Get total size of a directory in bytes."""
    total_size = 0
    try:
        for entry in os.scandir(directory):
            try:
                if entry.is_file(follow_symlinks=False):
                    total_size += entry.stat().st_size
                elif entry.is_dir(follow_symlinks=False):
                    total_size += _get_directory_size(entry.path)
            except (PermissionError, OSError):
                continue
    except (PermissionError, OSError):
        pass
    return total_size


def scan_models_in_location(location: StorageLocation, max_depth: int = 2) -> list[LocalModel]:
    """Scan a storage location for models."""
    search_paths = [location.path]

    # Also check common subdirectories
    for subdir in ["models", "huggingface", "hub", ".cache/huggingface/hub"]:
        subpath = os.path.join(location.path, subdir)
        if os.path.exists(subpath):
            search_paths.append(subpath)

    return scan_local_models(search_paths, max_depth=max_depth)
