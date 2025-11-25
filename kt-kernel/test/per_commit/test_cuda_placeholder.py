"""CUDA backend tests for KT-Kernel (Placeholder).

This file is a placeholder for future CUDA backend tests.
Currently, KT-Kernel focuses on CPU optimizations (Intel AMX/AVX512).

To implement CUDA tests:
1. Add actual test functions with @pytest.mark.cuda
2. Update the estimated time in register_cuda_ci()
3. Implement CUDA-specific initialization and validation tests
"""

import os
import sys

# Add parent directory to path for CI registration
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cuda_ci

# Register this test for CUDA CI (estimated time: 10 seconds, placeholder)
# Update suite name when implementing: currently using "stage-a-test-1"
register_cuda_ci(est_time=10, suite="stage-a-test-1")


def test_cuda_placeholder():
    """Placeholder test for CUDA backend.

    TODO: Implement actual CUDA tests when CUDA support is added to kt-kernel.
    """
    # Currently a no-op placeholder
    pass


if __name__ == "__main__":
    # Allow running standalone (required by test runner)
    print("⚠ CUDA tests are not yet implemented (placeholder)")
    print("✓ Placeholder test passed")
