"""Basic CPU backend tests for KT-Kernel.

These tests verify basic functionality without requiring model files.
"""

import os
import sys
import pytest

# Add parent directory to path for CI registration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

# Register this test for CPU CI with estimated runtime of 30 seconds
register_cpu_ci(est_time=30, suite="default")

# Check if kt_kernel_ext is available
try:
    import kt_kernel  # Import kt_kernel first to register kt_kernel_ext

    kt_kernel_ext = kt_kernel.kt_kernel_ext  # Access the extension module
    HAS_KT_KERNEL = True
except ImportError:
    HAS_KT_KERNEL = False
    kt_kernel_ext = None


@pytest.mark.cpu
def test_kt_kernel_import():
    """Test that kt_kernel_ext can be imported."""
    if not HAS_KT_KERNEL:
        pytest.skip("kt_kernel_ext not built or available")

    assert kt_kernel_ext is not None, "kt_kernel_ext module should be importable"


@pytest.mark.cpu
def test_cpu_infer_initialization():
    """Test that CPUInfer can be initialized."""
    if not HAS_KT_KERNEL:
        pytest.skip("kt_kernel_ext not built or available")

    # Initialize CPUInfer with 4 threads
    cpuinfer = kt_kernel_ext.CPUInfer(4)
    assert cpuinfer is not None, "CPUInfer should be initialized successfully"


@pytest.mark.cpu
def test_basic_module_attributes():
    """Test that kt_kernel_ext has expected attributes."""
    if not HAS_KT_KERNEL:
        pytest.skip("kt_kernel_ext not built or available")

    # Check for key attributes/functions
    assert hasattr(kt_kernel_ext, "CPUInfer"), "kt_kernel_ext should have CPUInfer class"


def run_all_tests():
    """Run all tests in this file (for standalone execution)."""
    if not HAS_KT_KERNEL:
        print("⚠ kt_kernel_ext not available, skipping tests")
        return

    try:
        test_kt_kernel_import()
        print("✓ test_kt_kernel_import passed")

        test_cpu_infer_initialization()
        print("✓ test_cpu_infer_initialization passed")

        test_basic_module_attributes()
        print("✓ test_basic_module_attributes passed")

        print("\n✓ All tests passed!")
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Allow running standalone (required by test runner)
    run_all_tests()
