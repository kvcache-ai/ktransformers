"""
Unit tests for AsyncExpertReader (io_uring-based expert loading).
"""

import pytest
import numpy as np
import os
import tempfile

# Check if io_uring is available
try:
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build"))
    import kt_kernel_ext as ext

    HAS_IOURING = hasattr(ext, "AsyncExpertReader")
except (ImportError, AttributeError):
    HAS_IOURING = False

pytestmark = pytest.mark.skipif(not HAS_IOURING, reason="io_uring not available")


@pytest.fixture
def temp_test_file():
    """Create a temporary test file with random data."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
        test_data = np.random.randn(1024, 1024).astype(np.float32)
        test_data.tofile(f)
        temp_path = f.name

    yield temp_path, test_data

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


def test_async_reader_basic(temp_test_file):
    """Test AsyncExpertReader basic functionality."""
    test_file, test_data = temp_test_file

    reader = ext.AsyncExpertReader(queue_depth=32)

    # Open file (without O_DIRECT for testing)
    fd = os.open(test_file, os.O_RDONLY)

    # Allocate buffer
    buffer = np.empty_like(test_data)

    # Submit read
    reader.submit_read(fd, buffer.ctypes.data, buffer.nbytes, 0, expert_id=0)

    # Wait for completion
    assert reader.wait_for_expert(0, timeout_ms=5000), "Read timed out"

    # Verify data
    np.testing.assert_array_almost_equal(buffer, test_data, decimal=5)

    # Cleanup
    os.close(fd)


def test_async_reader_batch(temp_test_file):
    """Test batch reading with multiple experts."""
    test_file, test_data = temp_test_file

    reader = ext.AsyncExpertReader(queue_depth=128)

    num_experts = 8
    test_files = []
    test_data_list = []

    # Create multiple test files
    for i in range(num_experts):
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{i}.bin") as f:
            data = np.random.randn(512, 512).astype(np.float32)
            data.tofile(f)
            test_files.append(f.name)
            test_data_list.append(data)

    try:
        # Open all files
        fds = [os.open(f, os.O_RDONLY) for f in test_files]
        buffers = [np.empty_like(d) for d in test_data_list]

        # Submit all reads
        for i in range(num_experts):
            reader.submit_read(fds[i], buffers[i].ctypes.data, buffers[i].nbytes, 0, expert_id=i)

        # Wait for all completions
        for i in range(num_experts):
            assert reader.wait_for_expert(i, timeout_ms=5000), f"Read {i} timed out"

        # Verify all data
        for i in range(num_experts):
            np.testing.assert_array_almost_equal(buffers[i], test_data_list[i], decimal=5)

        # Cleanup
        for fd in fds:
            os.close(fd)

    finally:
        # Remove temp files
        for f in test_files:
            if os.path.exists(f):
                os.remove(f)


def test_async_reader_waits_for_multiple_requests_same_expert(temp_test_file):
    """Test waiting for all tensor fragments belonging to one expert."""
    test_file, test_data = temp_test_file
    reader = ext.AsyncExpertReader(queue_depth=32)
    fd = os.open(test_file, os.O_RDONLY)

    first = np.empty((512, 1024), dtype=np.float32)
    second = np.empty((512, 1024), dtype=np.float32)
    split_bytes = first.nbytes

    try:
        req1 = reader.submit_read(fd, first.ctypes.data, first.nbytes, 0, expert_id=7)
        req2 = reader.submit_read(fd, second.ctypes.data, second.nbytes, split_bytes, expert_id=7)

        assert reader.wait_for_requests([req1, req2], timeout_ms=5000), "Fragment reads timed out"
        np.testing.assert_array_almost_equal(first, test_data[:512], decimal=5)
        np.testing.assert_array_almost_equal(second, test_data[512:], decimal=5)
    finally:
        os.close(fd)


def test_async_reader_timeout():
    """Test timeout behavior."""
    reader = ext.AsyncExpertReader(queue_depth=32)

    # Wait for non-existent expert (should timeout)
    assert not reader.wait_for_expert(999, timeout_ms=100), "Should have timed out"


def test_io_backend_enum():
    """Test IOBackend enum."""
    assert hasattr(ext, "IOBackend"), "IOBackend enum not found"
    assert hasattr(ext.IOBackend, "MMAP"), "IOBackend.MMAP not found"
    assert hasattr(ext.IOBackend, "IOURING"), "IOBackend.IOURING not found"

    # Check enum values
    assert ext.IOBackend.MMAP != ext.IOBackend.IOURING, "Enum values should be different"


def test_moe_config_accepts_iouring_file_slots():
    """Test Python-to-C++ conversion for nested io_uring file slots."""
    cfg = ext.moe.MOEConfig(2, 1, 16, 16, 0)
    reader = ext.AsyncExpertReader(queue_depth=8)
    slots = [[(3, 0, 512), (3, 512, 512)]]

    assert hasattr(cfg, "set_iouring_file_slots")
    cfg.set_iouring_file_slots(slots, slots, slots, slots, slots, slots, reader)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
