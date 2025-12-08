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


def detect_cpu_features():
    """
    Detect CPU features to determine the best kernel variant.

    Detection hierarchy:
        1. AMX: Intel Sapphire Rapids+ with AMX support
        2. AVX512: CPUs with AVX512F support
        3. AVX2: Fallback for maximum compatibility

    Returns:
        str: 'amx', 'avx512', or 'avx2'
    """
    # Check environment override
    variant = os.environ.get('KT_KERNEL_CPU_VARIANT', '').lower()
    if variant in ['amx', 'avx512', 'avx2']:
        if os.environ.get('KT_KERNEL_DEBUG') == '1':
            print(f"[kt-kernel] Using environment override: {variant}")
        return variant

    # Try to read /proc/cpuinfo on Linux
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpuinfo = f.read().lower()

        # Check for AMX support (Intel Sapphire Rapids+)
        # AMX requires amx_tile, amx_int8, and amx_bf16
        amx_flags = ['amx_tile', 'amx_int8', 'amx_bf16']
        has_amx = all(flag in cpuinfo for flag in amx_flags)

        if has_amx:
            if os.environ.get('KT_KERNEL_DEBUG') == '1':
                print("[kt-kernel] Detected AMX support via /proc/cpuinfo")
            return 'amx'

        # Check for AVX512 support
        # AVX512F is the foundation for all AVX512 variants
        if 'avx512f' in cpuinfo:
            if os.environ.get('KT_KERNEL_DEBUG') == '1':
                print("[kt-kernel] Detected AVX512 support via /proc/cpuinfo")
            return 'avx512'

        # Check for AVX2 support
        if 'avx2' in cpuinfo:
            if os.environ.get('KT_KERNEL_DEBUG') == '1':
                print("[kt-kernel] Detected AVX2 support via /proc/cpuinfo")
            return 'avx2'

        # Fallback to AVX2 (should be rare on modern CPUs)
        if os.environ.get('KT_KERNEL_DEBUG') == '1':
            print("[kt-kernel] No AVX2/AVX512/AMX detected, using AVX2 fallback")
        return 'avx2'

    except FileNotFoundError:
        # /proc/cpuinfo doesn't exist (not Linux or in container)
        # Try cpufeature package as fallback
        if os.environ.get('KT_KERNEL_DEBUG') == '1':
            print("[kt-kernel] /proc/cpuinfo not found, trying cpufeature package")

        try:
            import cpufeature

            # Check for AMX
            if cpufeature.CPUFeature.get('AMX_TILE', False):
                if os.environ.get('KT_KERNEL_DEBUG') == '1':
                    print("[kt-kernel] Detected AMX support via cpufeature")
                return 'amx'

            # Check for AVX512
            if cpufeature.CPUFeature.get('AVX512F', False):
                if os.environ.get('KT_KERNEL_DEBUG') == '1':
                    print("[kt-kernel] Detected AVX512 support via cpufeature")
                return 'avx512'

            # Fallback to AVX2
            if os.environ.get('KT_KERNEL_DEBUG') == '1':
                print("[kt-kernel] Using AVX2 fallback via cpufeature")
            return 'avx2'

        except ImportError:
            # cpufeature not available - ultimate fallback
            if os.environ.get('KT_KERNEL_DEBUG') == '1':
                print("[kt-kernel] cpufeature not available, using AVX2 fallback")
            return 'avx2'

    except Exception as e:
        # Any other error - safe fallback
        if os.environ.get('KT_KERNEL_DEBUG') == '1':
            print(f"[kt-kernel] Error during CPU detection: {e}, using AVX2 fallback")
        return 'avx2'


def load_extension(variant):
    """
    Load the appropriate kt_kernel_ext variant.

    Tries to import the specified variant, with automatic fallback to
    lower-performance variants if the requested one is not available.

    Supports both multi-variant builds (_kt_kernel_ext_amx.*.so) and
    single-variant builds (kt_kernel_ext.*.so).

    Fallback order: amx -> avx512 -> avx2 -> single-variant

    Args:
        variant (str): 'amx', 'avx512', or 'avx2'

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
        pattern = os.path.join(kt_kernel_dir, f'_kt_kernel_ext_{variant}.*.so')
        so_files = glob.glob(pattern)

        if not so_files:
            # Try single-variant naming (fallback for builds without CPUINFER_BUILD_ALL_VARIANTS)
            pattern = os.path.join(kt_kernel_dir, 'kt_kernel_ext.*.so')
            so_files = glob.glob(pattern)

            if so_files:
                if os.environ.get('KT_KERNEL_DEBUG') == '1':
                    print(f"[kt-kernel] Multi-variant {variant} not found, using single-variant build")
            else:
                raise ImportError(f"No .so file found for variant {variant} (tried patterns: {kt_kernel_dir}/_kt_kernel_ext_{variant}.*.so and {kt_kernel_dir}/kt_kernel_ext.*.so)")

        so_file = so_files[0]

        if os.environ.get('KT_KERNEL_DEBUG') == '1':
            print(f"[kt-kernel] Loading {variant} from: {so_file}")

        # Load the module manually
        # The module exports PyInit_kt_kernel_ext, so we use that as the module name
        spec = importlib.util.spec_from_file_location('kt_kernel_ext', so_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to create spec for {so_file}")

        ext = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ext)

        if os.environ.get('KT_KERNEL_DEBUG') == '1':
            print(f"[kt-kernel] Successfully loaded {variant.upper()} variant")
        return ext

    except (ImportError, ModuleNotFoundError, FileNotFoundError) as e:
        if os.environ.get('KT_KERNEL_DEBUG') == '1':
            print(f"[kt-kernel] Failed to load {variant} variant: {e}")

        # Automatic fallback to next best variant
        if variant == 'amx':
            if os.environ.get('KT_KERNEL_DEBUG') == '1':
                print("[kt-kernel] Falling back from AMX to AVX512")
            return load_extension('avx512')
        elif variant == 'avx512':
            if os.environ.get('KT_KERNEL_DEBUG') == '1':
                print("[kt-kernel] Falling back from AVX512 to AVX2")
            return load_extension('avx2')
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

    if os.environ.get('KT_KERNEL_DEBUG') == '1':
        print(f"[kt-kernel] Selected CPU variant: {variant}")

    # Load the appropriate extension
    ext = load_extension(variant)

    if os.environ.get('KT_KERNEL_DEBUG') == '1':
        print(f"[kt-kernel] Extension module loaded: {ext.__name__}")

    return ext, variant
