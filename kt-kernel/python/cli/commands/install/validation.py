"""
Parameter validation for install command.
"""

from pathlib import Path
from typing import Optional

from kt_kernel.cli.utils.console import print_error, print_warning


def validate_install_params(
    source: Optional[Path],
    editable: bool,
    cpu_instruct: Optional[str],
    enable_amx: Optional[bool],
    from_source: bool,
) -> list[str]:
    """
    Validate installation parameters.

    Args:
        source: Source directory path (if specified)
        editable: Whether editable mode is requested
        cpu_instruct: CPU instruction set (if specified)
        enable_amx: AMX enable flag (if specified)
        from_source: Whether building from source

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    warnings = []

    # Validate editable requires source
    if editable and not source:
        errors.append("--editable requires --source to be specified")

    # Validate CPU instruction options
    if cpu_instruct and not from_source and not source:
        warnings.append("--cpu-instruct is only effective with --from-source or --source")

    if enable_amx is not None and not from_source and not source:
        warnings.append("--enable-amx/--disable-amx is only effective with --from-source or --source")

    # Show warnings
    for warning in warnings:
        print_warning(warning)

    return errors
