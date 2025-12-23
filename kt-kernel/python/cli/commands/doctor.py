"""
Doctor command for kt-cli.

Diagnoses environment issues and provides recommendations.
"""

import platform
import shutil
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table

from kt_kernel.cli.config.settings import get_settings
from kt_kernel.cli.i18n import t
from kt_kernel.cli.utils.console import console, print_error, print_info, print_success, print_warning
from kt_kernel.cli.utils.environment import (
    check_docker,
    detect_available_ram_gb,
    detect_cpu_info,
    detect_cuda_version,
    detect_disk_space_gb,
    detect_env_managers,
    detect_gpus,
    detect_memory_info,
    detect_ram_gb,
    get_installed_package_version,
)


def doctor(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed diagnostics"),
) -> None:
    """Diagnose environment issues."""
    console.print(f"\n[bold]{t('doctor_title')}[/bold]\n")

    issues_found = False
    checks = []

    # 1. Python version
    python_version = platform.python_version()
    python_ok = _check_python_version(python_version)
    checks.append({
        "name": t("doctor_check_python"),
        "status": "ok" if python_ok else "error",
        "value": python_version,
        "hint": "Python 3.10+ required" if not python_ok else None,
    })
    if not python_ok:
        issues_found = True

    # 2. CUDA availability
    cuda_version = detect_cuda_version()
    checks.append({
        "name": t("doctor_check_cuda"),
        "status": "ok" if cuda_version else "warning",
        "value": cuda_version or t("version_cuda_not_found"),
        "hint": "CUDA is optional but recommended for GPU acceleration" if not cuda_version else None,
    })

    # 3. GPU detection
    gpus = detect_gpus()
    if gpus:
        gpu_names = ", ".join(g.name for g in gpus)
        total_vram = sum(g.vram_gb for g in gpus)
        checks.append({
            "name": t("doctor_check_gpu"),
            "status": "ok",
            "value": t("doctor_gpu_found", count=len(gpus), names=gpu_names),
            "hint": f"Total VRAM: {total_vram}GB",
        })
    else:
        checks.append({
            "name": t("doctor_check_gpu"),
            "status": "warning",
            "value": t("doctor_gpu_not_found"),
            "hint": "GPU recommended for best performance",
        })

    # 4. CPU information
    cpu_info = detect_cpu_info()
    checks.append({
        "name": t("doctor_check_cpu"),
        "status": "ok",
        "value": t("doctor_cpu_info", name=cpu_info.name, cores=cpu_info.cores, threads=cpu_info.threads),
        "hint": None,
    })

    # 5. CPU instruction sets (critical for kt-kernel)
    isa_list = cpu_info.instruction_sets
    # Check for recommended instruction sets
    recommended_isa = {"AVX2", "AVX512F", "AMX-INT8"}
    has_recommended = bool(set(isa_list) & recommended_isa)
    has_avx2 = "AVX2" in isa_list
    has_avx512 = any(isa.startswith("AVX512") for isa in isa_list)
    has_amx = any(isa.startswith("AMX") for isa in isa_list)

    # Determine status and build display string
    if has_amx:
        isa_status = "ok"
        isa_hint = "AMX available - best performance for INT4/INT8"
    elif has_avx512:
        isa_status = "ok"
        isa_hint = "AVX512 available - good performance"
    elif has_avx2:
        isa_status = "warning"
        isa_hint = "AVX2 only - consider upgrading CPU for better performance"
    else:
        isa_status = "error"
        isa_hint = "AVX2 required for kt-kernel"

    # Show top instruction sets (prioritize important ones)
    display_isa = isa_list[:8] if len(isa_list) > 8 else isa_list
    isa_display = ", ".join(display_isa)
    if len(isa_list) > 8:
        isa_display += f" (+{len(isa_list) - 8} more)"

    checks.append({
        "name": t("doctor_check_cpu_isa"),
        "status": isa_status,
        "value": isa_display if isa_display else "None detected",
        "hint": isa_hint,
    })

    # 6. NUMA topology
    numa_detail = []
    for node, cpus in sorted(cpu_info.numa_info.items()):
        if len(cpus) > 6:
            cpu_str = f"{cpus[0]}-{cpus[-1]}"
        else:
            cpu_str = ",".join(str(c) for c in cpus)
        numa_detail.append(f"{node}: {cpu_str}")

    numa_value = t("doctor_numa_info", nodes=cpu_info.numa_nodes)
    if verbose and numa_detail:
        numa_value += " (" + "; ".join(numa_detail) + ")"

    checks.append({
        "name": t("doctor_check_numa"),
        "status": "ok",
        "value": numa_value,
        "hint": f"{cpu_info.threads // cpu_info.numa_nodes} threads per node" if cpu_info.numa_nodes > 1 else None,
    })

    # 7. System memory (with frequency if available)
    mem_info = detect_memory_info()
    if mem_info.frequency_mhz and mem_info.type:
        mem_value = t("doctor_memory_freq",
                      available=f"{mem_info.available_gb}GB",
                      total=f"{mem_info.total_gb}GB",
                      freq=mem_info.frequency_mhz,
                      type=mem_info.type)
    else:
        mem_value = t("doctor_memory_info",
                      available=f"{mem_info.available_gb}GB",
                      total=f"{mem_info.total_gb}GB")

    ram_ok = mem_info.total_gb >= 32
    checks.append({
        "name": t("doctor_check_memory"),
        "status": "ok" if ram_ok else "warning",
        "value": mem_value,
        "hint": "32GB+ RAM recommended for large models" if not ram_ok else None,
    })

    # 8. Disk space - check all model paths
    settings = get_settings()
    model_paths = settings.get_model_paths()

    # Check all configured model paths
    for i, disk_path in enumerate(model_paths):
        available_disk, total_disk = detect_disk_space_gb(str(disk_path))
        disk_ok = available_disk >= 100

        # For multiple paths, add index to name
        path_label = f"Model Path {i+1}" if len(model_paths) > 1 else t("doctor_check_disk")

        checks.append({
            "name": path_label,
            "status": "ok" if disk_ok else "warning",
            "value": t("doctor_disk_info", available=f"{available_disk}GB", path=str(disk_path)),
            "hint": "100GB+ free space recommended for model storage" if not disk_ok else None,
        })

    # 6. Required packages
    packages = [
        ("kt-kernel", ">=0.4.0", False),  # name, version_req, required
        ("ktransformers", ">=0.4.0", False),
        ("sglang", ">=0.4.0", False),
        ("torch", ">=2.4.0", True),
        ("transformers", ">=4.45.0", True),
    ]

    package_issues = []
    for pkg_name, version_req, required in packages:
        version = get_installed_package_version(pkg_name)
        if version:
            package_issues.append((pkg_name, version, "ok"))
        elif required:
            package_issues.append((pkg_name, t("version_not_installed"), "error"))
            issues_found = True
        else:
            package_issues.append((pkg_name, t("version_not_installed"), "warning"))

    if verbose:
        checks.append({
            "name": t("doctor_check_packages"),
            "status": "ok" if not any(p[2] == "error" for p in package_issues) else "error",
            "value": f"{sum(1 for p in package_issues if p[2] == 'ok')}/{len(package_issues)} installed",
            "packages": package_issues,
        })

    # 7. Environment managers
    env_managers = detect_env_managers()
    docker = check_docker()
    env_list = [f"{m.name} {m.version}" for m in env_managers]
    if docker:
        env_list.append(f"docker {docker.version}")

    checks.append({
        "name": "Environment Managers",
        "status": "ok" if env_list else "warning",
        "value": ", ".join(env_list) if env_list else "None found",
        "hint": "conda or docker recommended for installation" if not env_list else None,
    })

    # Display results
    _display_results(checks, verbose)

    # Summary
    console.print()
    if issues_found:
        print_warning(t("doctor_has_issues"))
    else:
        print_success(t("doctor_all_ok"))
    console.print()


def _check_python_version(version: str) -> bool:
    """Check if Python version meets requirements."""
    parts = version.split(".")
    try:
        major, minor = int(parts[0]), int(parts[1])
        return major >= 3 and minor >= 10
    except (IndexError, ValueError):
        return False


def _display_results(checks: list[dict], verbose: bool) -> None:
    """Display diagnostic results."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", style="bold")
    table.add_column("Status", width=8)
    table.add_column("Value")
    if verbose:
        table.add_column("Notes", style="dim")

    for check in checks:
        status = check["status"]
        if status == "ok":
            status_str = f"[green]{t('doctor_status_ok')}[/green]"
        elif status == "warning":
            status_str = f"[yellow]{t('doctor_status_warning')}[/yellow]"
        else:
            status_str = f"[red]{t('doctor_status_error')}[/red]"

        if verbose:
            table.add_row(
                check["name"],
                status_str,
                check["value"],
                check.get("hint", ""),
            )
        else:
            table.add_row(
                check["name"],
                status_str,
                check["value"],
            )

        # Show package details if verbose
        if verbose and "packages" in check:
            for pkg_name, pkg_version, pkg_status in check["packages"]:
                if pkg_status == "ok":
                    pkg_status_str = "[green]✓[/green]"
                elif pkg_status == "warning":
                    pkg_status_str = "[yellow]○[/yellow]"
                else:
                    pkg_status_str = "[red]✗[/red]"

                table.add_row(
                    f"  └─ {pkg_name}",
                    pkg_status_str,
                    pkg_version,
                    "",
                )

    console.print(table)
