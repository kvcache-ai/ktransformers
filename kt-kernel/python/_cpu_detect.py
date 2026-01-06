"""
CPU feature detection and optimal kernel loader for kt-kernel.

This module automatically detects CPU capabilities and loads the best available
kernel variant (AMX, AVX512, or AVX2) at runtime.

Environment Variables:
    KT_KERNEL_CPU_VARIANT: Override automatic detection ('amx', 'avx512', 'avx2')
    KT_KERNEL_DEBUG: Enable debug output ('1' to enable)

Example:
    >>> import kt_kernel
    >>> print(kt_kernel.__cpu_variant__)  # Shows detected variant

    # Override detection
    >>> import os
    >>> os.environ['KT_KERNEL_CPU_VARIANT'] = 'avx2'
    >>> import kt_kernel  # Will use AVX2 variant
"""

import os
import sys
from pathlib import Path


def detect_so_instructions(so_path):
    """
    Detect actual instruction sets present in a compiled .so file.

    This function uses objdump to disassemble the binary and search for
    characteristic instructions of different SIMD instruction sets.

    Args:
        so_path (str): Path to the .so file to analyze

    Returns:
        dict: Dictionary mapping instruction set names to boolean values
              {'amx': bool, 'avx512_vbmi': bool, 'avx512_vnni': bool, ...}
              Returns None if objdump is not available or fails.
    """
    try:
        import subprocess
        import re

        # Run objdump to disassemble the .so file
        output = subprocess.check_output(
            ["objdump", "-d", so_path], stderr=subprocess.DEVNULL, universal_newlines=True, timeout=10
        )

        # Define characteristic instructions for each instruction set
        # These are specific instructions that uniquely identify each feature
        instruction_patterns = {
            "amx": r"\b(tileloadd|tilestored|tdpb)",  # AMX tile operations
            "avx512_vbmi": r"\b(vpermb|vpermt2b|vpermi2b)",  # Byte permutation
            "avx512_vnni": r"\b(vpdpbusd|vpdpwssd)",  # VNNI dot product
            "avx512_bf16": r"\b(vdpbf16ps|vcvtne2ps2bf16)",  # BF16 operations
            "avx512f": r"\b(vmovdqu64|vpaddd.*%zmm)",  # AVX512 foundation (zmm registers)
        }

        result = {}
        for feature, pattern in instruction_patterns.items():
            result[feature] = bool(re.search(pattern, output))

        return result

    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, ImportError):
        # objdump not available or failed - return None to indicate detection failed
        return None
    except Exception:
        # Any other error - silently fail
        return None


def detect_cpu_features():
    """
    Detect CPU features and determine the best kernel variant using progressive matching.

    Progressive variant hierarchy (from most to least advanced):
        1. AMX: amx_tile, amx_int8, amx_bf16 + full AVX512
        2. AVX512_BF16: avx512f, avx512bw, avx512_vnni, avx512_vbmi, avx512_bf16
        3. AVX512_VBMI: avx512f, avx512bw, avx512_vnni, avx512_vbmi
        4. AVX512_VNNI: avx512f, avx512bw, avx512_vnni
        5. AVX512_BASE: avx512f, avx512bw
        6. AVX2: avx2 (fallback)

    Returns:
        str: Variant name - one of: 'amx', 'avx512_bf16', 'avx512_vbmi',
             'avx512_vnni', 'avx512_base', 'avx2'
    """
    # Check environment override
    variant = os.environ.get("KT_KERNEL_CPU_VARIANT", "").lower()
    valid_variants = ["amx", "avx512_bf16", "avx512_vbmi", "avx512_vnni", "avx512_base", "avx2"]
    if variant in valid_variants:
        if os.environ.get("KT_KERNEL_DEBUG") == "1":
            print(f"[kt-kernel] Using environment override: {variant}")
        return variant

    # Try to read /proc/cpuinfo on Linux
    try:
        with open("/proc/cpuinfo", "r") as f:
            cpuinfo = f.read().lower()

        # Extract CPU flags into a set for fast lookup
        cpu_flags = set()
        for line in cpuinfo.split("\n"):
            if line.startswith("flags"):
                flags_str = line.split(":", 1)[1]
                cpu_flags = set(flags_str.split())
                break

        # Define variant requirements in priority order (best to worst)
        variant_requirements = [
            (
                "amx",
                [
                    "amx_tile",
                    "amx_int8",
                    "amx_bf16",
                    "avx512f",
                    "avx512bw",
                    "avx512_vnni",
                    "avx512_vbmi",
                    "avx512_bf16",
                ],
            ),
            ("avx512_bf16", ["avx512f", "avx512bw", "avx512_vnni", "avx512_vbmi", "avx512_bf16"]),
            ("avx512_vbmi", ["avx512f", "avx512bw", "avx512_vnni", "avx512_vbmi"]),
            ("avx512_vnni", ["avx512f", "avx512bw", "avx512_vnni"]),
            ("avx512_base", ["avx512f", "avx512bw"]),
            ("avx2", ["avx2"]),
        ]

        # Find the best matching variant
        for variant_name, required_flags in variant_requirements:
            # Check if all required flags are present
            # Handle flag name variations (e.g., avx512_bf16 vs avx512bf16)
            has_all_flags = True
            for flag in required_flags:
                # Try exact match first, then without underscore
                flag_alt = flag.replace("_", "")
                if flag not in cpu_flags and flag_alt not in cpu_flags:
                    has_all_flags = False
                    break

            if has_all_flags:
                if os.environ.get("KT_KERNEL_DEBUG") == "1":
                    print(f"[kt-kernel] Detected {variant_name} support via /proc/cpuinfo")
                    print(f"[kt-kernel] Matched flags: {', '.join(required_flags)}")
                return variant_name

        # Fallback to AVX2 (should be rare on modern CPUs)
        if os.environ.get("KT_KERNEL_DEBUG") == "1":
            print("[kt-kernel] No supported features detected, using AVX2 fallback")
        return "avx2"

    except FileNotFoundError:
        # /proc/cpuinfo doesn't exist (not Linux or in container)
        # Try cpufeature package as fallback
        if os.environ.get("KT_KERNEL_DEBUG") == "1":
            print("[kt-kernel] /proc/cpuinfo not found, trying cpufeature package")

        try:
            import cpufeature

            # Define variant requirements in priority order (using cpufeature naming)
            cpufeature_requirements = [
                (
                    "amx",
                    [
                        "AMX_TILE",
                        "AMX_INT8",
                        "AMX_BF16",
                        "AVX512F",
                        "AVX512BW",
                        "AVX512_VNNI",
                        "AVX512_VBMI",
                        "AVX512_BF16",
                    ],
                ),
                ("avx512_bf16", ["AVX512F", "AVX512BW", "AVX512_VNNI", "AVX512_VBMI", "AVX512_BF16"]),
                ("avx512_vbmi", ["AVX512F", "AVX512BW", "AVX512_VNNI", "AVX512_VBMI"]),
                ("avx512_vnni", ["AVX512F", "AVX512BW", "AVX512_VNNI"]),
                ("avx512_base", ["AVX512F", "AVX512BW"]),
                ("avx2", ["AVX2"]),
            ]

            # Find the best matching variant
            for variant_name, required_features in cpufeature_requirements:
                has_all_features = all(cpufeature.CPUFeature.get(feat, False) for feat in required_features)
                if has_all_features:
                    if os.environ.get("KT_KERNEL_DEBUG") == "1":
                        print(f"[kt-kernel] Detected {variant_name} support via cpufeature")
                    return variant_name

            # Fallback to AVX2
            if os.environ.get("KT_KERNEL_DEBUG") == "1":
                print("[kt-kernel] Using AVX2 fallback via cpufeature")
            return "avx2"

        except ImportError:
            # cpufeature not available - ultimate fallback
            if os.environ.get("KT_KERNEL_DEBUG") == "1":
                print("[kt-kernel] cpufeature not available, using AVX2 fallback")
            return "avx2"

    except Exception as e:
        # Any other error - safe fallback
        if os.environ.get("KT_KERNEL_DEBUG") == "1":
            print(f"[kt-kernel] Error during CPU detection: {e}, using AVX2 fallback")
        return "avx2"


def load_extension(variant):
    """
    Load the appropriate kt_kernel_ext variant.

    Tries to import the specified variant, with automatic fallback to
    lower-performance variants if the requested one is not available.

    Supports both multi-variant builds (_kt_kernel_ext_amx.*.so) and
    single-variant builds (kt_kernel_ext.*.so).

    Fallback chain (each variant falls back to the next in line):
        amx -> avx512_bf16 -> avx512_vbmi -> avx512_vnni -> avx512_base -> avx2 -> single-variant

    Args:
        variant (str): One of 'amx', 'avx512_bf16', 'avx512_vbmi', 'avx512_vnni', 'avx512_base', 'avx2'

    Returns:
        module: The loaded extension module

    Raises:
        ImportError: If all variants fail to load
    """
    import importlib.util
    import glob

    # The .so files can be named in two ways:
    # Multi-variant: _kt_kernel_ext_amx.cpython-311-x86_64-linux-gnu.so
    # Single-variant: kt_kernel_ext.cpython-311-x86_64-linux-gnu.so
    # Both export PyInit_kt_kernel_ext (the original module name)

    try:
        # Find the kt_kernel package directory
        # We can't import kt_kernel here (circular import), so use __file__
        kt_kernel_dir = os.path.dirname(os.path.abspath(__file__))

        # Try multi-variant naming first
        pattern = os.path.join(kt_kernel_dir, f"_kt_kernel_ext_{variant}.*.so")
        so_files = glob.glob(pattern)

        if not so_files:
            # Try single-variant naming (fallback for builds without CPUINFER_BUILD_ALL_VARIANTS)
            pattern = os.path.join(kt_kernel_dir, "kt_kernel_ext.*.so")
            so_files = glob.glob(pattern)

            if so_files:
                if os.environ.get("KT_KERNEL_DEBUG") == "1":
                    print(f"[kt-kernel] Multi-variant {variant} not found, using single-variant build")
            else:
                raise ImportError(
                    f"No .so file found for variant {variant} (tried patterns: {kt_kernel_dir}/_kt_kernel_ext_{variant}.*.so and {kt_kernel_dir}/kt_kernel_ext.*.so)"
                )

        so_file = so_files[0]

        if os.environ.get("KT_KERNEL_DEBUG") == "1":
            print(f"[kt-kernel] Loading {variant} from: {so_file}")

        # Load the module manually
        # The module exports PyInit_kt_kernel_ext, so we use that as the module name
        spec = importlib.util.spec_from_file_location("kt_kernel_ext", so_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create spec for {so_file}")

        ext = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ext)

        if os.environ.get("KT_KERNEL_DEBUG") == "1":
            # 检测是否是 multi-variant 还是 single-variant
            is_single_variant = "kt_kernel_ext." in os.path.basename(so_file)

            if is_single_variant:
                print(f"[kt-kernel] Successfully loaded extension (single-variant build)")

                # 检测 .so 文件中实际包含的指令集
                actual_instructions = detect_so_instructions(so_file)

                if actual_instructions is not None:
                    print(f"[kt-kernel] CPU supports: {variant.upper()}")
                    print(f"[kt-kernel] Actual instruction sets in binary:")

                    # 按层次顺序显示指令集
                    ordered_features = ["amx", "avx512_bf16", "avx512_vbmi", "avx512_vnni", "avx512f"]
                    for feature in ordered_features:
                        if feature in actual_instructions:
                            status = "✅" if actual_instructions[feature] else "❌"
                            feature_display = feature.upper().replace("_", " ")
                            print(f"[kt-kernel]   {status} {feature_display}")
                else:
                    # objdump 不可用，显示警告
                    print(
                        f"[kt-kernel] Note: CPU supports {variant.upper()}, but binary may not contain all {variant.upper()} instructions"
                    )
                    print(f"[kt-kernel] (Install 'binutils' package to enable detailed instruction set detection)")
            else:
                print(f"[kt-kernel] Successfully loaded {variant.upper()} variant (multi-variant build)")
        return ext

    except (ImportError, ModuleNotFoundError, FileNotFoundError) as e:
        if os.environ.get("KT_KERNEL_DEBUG") == "1":
            print(f"[kt-kernel] Failed to load {variant} variant: {e}")

        # Define fallback chain: each variant falls back to the next lower one
        fallback_chain = {
            "amx": "avx512_bf16",
            "avx512_bf16": "avx512_vbmi",
            "avx512_vbmi": "avx512_vnni",
            "avx512_vnni": "avx512_base",
            "avx512_base": "avx2",
            "avx2": None,  # No fallback - terminal variant
        }

        # Get next fallback variant
        next_variant = fallback_chain.get(variant)

        if next_variant:
            # Try next variant in the chain
            if os.environ.get("KT_KERNEL_DEBUG") == "1":
                print(f"[kt-kernel] Falling back from {variant} to {next_variant}")
            return load_extension(next_variant)
        else:
            # AVX2 is the last fallback - if this fails, we can't continue
            raise ImportError(
                f"Failed to load kt_kernel extension (variant: {variant}). "
                f"Original error: {e}\n"
                f"This usually means the kt_kernel package is not properly installed."
            )


def initialize():
    """
    Detect CPU capabilities and load the optimal extension variant.

    This is the main entry point called by kt_kernel.__init__.py.

    Returns:
        tuple: (extension_module, variant_name)
    - extension_module: The loaded C++ extension module
            - variant_name: String indicating which variant was loaded ('amx', 'avx512', 'avx2')

    Example:
        >>> ext, variant = initialize()
        >>> print(f"Loaded {variant} variant")
        >>> wrapper = ext.AMXMoEWrapper(...)
    """
    # Detect CPU features
    variant = detect_cpu_features()

    if os.environ.get("KT_KERNEL_DEBUG") == "1":
        print(f"[kt-kernel] Selected CPU variant: {variant}")

    # Load the appropriate extension
    ext = load_extension(variant)

    if os.environ.get("KT_KERNEL_DEBUG") == "1":
        print(f"[kt-kernel] Extension module loaded: {ext.__name__}")

    return ext, variant
