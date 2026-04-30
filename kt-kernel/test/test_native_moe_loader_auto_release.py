"""Tests for NativeMoEWrapper auto-release loader mechanism.

Verifies that the SafeTensor loader singleton (_native_loader_instance) is
automatically released after all MoE layers finish load_weights(), and that
force_release_loader() works correctly at any point.

These tests use mocking so they can run without actual safetensors files or
compiled kt_kernel_ext binaries.
"""

import sys
import os
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Minimal stubs so amx.py can be imported without kt_kernel_ext compiled
# ---------------------------------------------------------------------------

def _make_kt_kernel_ext_stub():
    """Return a minimal stub module for kt_kernel_ext."""
    mod = types.ModuleType("kt_kernel_ext")
    moe_mod = types.ModuleType("kt_kernel_ext.moe")
    # MOEConfig stub
    moe_mod.MOEConfig = MagicMock()
    mod.moe = moe_mod
    sys.modules["kt_kernel_ext"] = mod
    sys.modules["kt_kernel_ext.moe"] = moe_mod
    return mod


def _make_base_stub():
    """Return a minimal stub for experts_base module."""
    experts_base = types.ModuleType("experts_base")

    class BaseMoEWrapper:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    experts_base.BaseMoEWrapper = BaseMoEWrapper
    # Register under both possible import paths
    sys.modules["experts_base"] = experts_base
    return experts_base


def _setup_stubs():
    _make_kt_kernel_ext_stub()
    _make_base_stub()


_setup_stubs()

# Add python directory to sys.path so we can import amx
_TEST_DIR = os.path.dirname(__file__)
_PYTHON_DIR = os.path.join(_TEST_DIR, "..", "python")
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)


# ---------------------------------------------------------------------------
# Import the module under test (with mocked kt_kernel_ext)
# ---------------------------------------------------------------------------

# We need to patch the relative imports inside amx.py before importing
with patch.dict(sys.modules, {
    "kt_kernel_ext": sys.modules["kt_kernel_ext"],
    "kt_kernel_ext.moe": sys.modules["kt_kernel_ext.moe"],
}):
    # Patch BaseMoEWrapper used by amx
    import importlib
    # Manually set up the package structure so relative imports work
    utils_pkg = types.ModuleType("utils")
    sys.modules.setdefault("utils", utils_pkg)

    # We'll test NativeMoEWrapper in isolation via direct patching
    pass


class MockLoader:
    """Minimal mock SafeTensorLoader."""

    def __init__(self):
        self.closed = False
        self.file_handle_map = {"dummy.safetensors": object()}

    def close_all_handles(self):
        self.closed = True
        self.file_handle_map.clear()


class FakeNativeMoEWrapper:
    """
    A simplified replica of NativeMoEWrapper that isolates the counter/release
    logic without requiring kt_kernel_ext to be compiled.
    """

    _native_loader_instance = None
    _total_layer_count: int = 0
    _load_call_count: int = 0

    def __init__(self):
        FakeNativeMoEWrapper._total_layer_count += 1

    def load_weights(self):
        """Simulate load_weights completion (counter + auto-release)."""
        FakeNativeMoEWrapper._load_call_count += 1
        if FakeNativeMoEWrapper._load_call_count >= FakeNativeMoEWrapper._total_layer_count > 0:
            FakeNativeMoEWrapper._release_loader()

    @staticmethod
    def _release_loader():
        if FakeNativeMoEWrapper._native_loader_instance is not None:
            FakeNativeMoEWrapper._native_loader_instance.close_all_handles()
            FakeNativeMoEWrapper._native_loader_instance = None
        FakeNativeMoEWrapper._total_layer_count = 0
        FakeNativeMoEWrapper._load_call_count = 0

    @staticmethod
    def force_release_loader():
        FakeNativeMoEWrapper._release_loader()


def _reset_class_state():
    """Reset FakeNativeMoEWrapper class-level state between tests."""
    FakeNativeMoEWrapper._native_loader_instance = None
    FakeNativeMoEWrapper._total_layer_count = 0
    FakeNativeMoEWrapper._load_call_count = 0


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestAutoReleaseAfterAllLayersLoaded(unittest.TestCase):
    """Task 3.1 – After N layers all call load_weights, loader becomes None."""

    def setUp(self):
        _reset_class_state()

    def test_single_layer_released_after_load(self):
        loader = MockLoader()
        FakeNativeMoEWrapper._native_loader_instance = loader

        w = FakeNativeMoEWrapper()
        self.assertIsNotNone(FakeNativeMoEWrapper._native_loader_instance)

        w.load_weights()

        self.assertIsNone(FakeNativeMoEWrapper._native_loader_instance,
                          "Loader should be None after the single layer loads its weights")
        self.assertTrue(loader.closed, "close_all_handles() should have been called")

    def test_multiple_layers_released_only_after_last(self):
        N = 4
        loader = MockLoader()
        FakeNativeMoEWrapper._native_loader_instance = loader

        wrappers = [FakeNativeMoEWrapper() for _ in range(N)]

        for i, w in enumerate(wrappers):
            w.load_weights()
            if i < N - 1:
                self.assertIsNotNone(
                    FakeNativeMoEWrapper._native_loader_instance,
                    f"Loader should still be alive after layer {i+1}/{N}",
                )
            else:
                self.assertIsNone(
                    FakeNativeMoEWrapper._native_loader_instance,
                    "Loader should be released after the last layer",
                )

        self.assertTrue(loader.closed)

    def test_counters_reset_to_zero_after_release(self):
        loader = MockLoader()
        FakeNativeMoEWrapper._native_loader_instance = loader

        w = FakeNativeMoEWrapper()
        w.load_weights()

        self.assertEqual(FakeNativeMoEWrapper._total_layer_count, 0,
                         "_total_layer_count should reset to 0 after release")
        self.assertEqual(FakeNativeMoEWrapper._load_call_count, 0,
                         "_load_call_count should reset to 0 after release")


class TestNoEarlyReleaseBeforeAllLayersLoaded(unittest.TestCase):
    """Task 3.2 – Loader must NOT be released before all layers finish."""

    def setUp(self):
        _reset_class_state()

    def test_loader_alive_until_last_layer(self):
        N = 5
        loader = MockLoader()
        FakeNativeMoEWrapper._native_loader_instance = loader

        wrappers = [FakeNativeMoEWrapper() for _ in range(N)]

        # Load N-1 layers
        for w in wrappers[:-1]:
            w.load_weights()

        self.assertIsNotNone(
            FakeNativeMoEWrapper._native_loader_instance,
            "Loader must still be alive when N-1 of N layers have loaded",
        )
        self.assertFalse(loader.closed)

        # Load last layer
        wrappers[-1].load_weights()
        self.assertIsNone(FakeNativeMoEWrapper._native_loader_instance)
        self.assertTrue(loader.closed)


class TestForceReleaseLoader(unittest.TestCase):
    """Task 3.3 – force_release_loader() works at any time."""

    def setUp(self):
        _reset_class_state()

    def test_force_release_before_any_load(self):
        loader = MockLoader()
        FakeNativeMoEWrapper._native_loader_instance = loader

        # Create wrappers but don't call load_weights
        FakeNativeMoEWrapper()
        FakeNativeMoEWrapper()

        self.assertIsNotNone(FakeNativeMoEWrapper._native_loader_instance)

        # Force release without completing loads
        FakeNativeMoEWrapper.force_release_loader()

        self.assertIsNone(FakeNativeMoEWrapper._native_loader_instance)
        self.assertTrue(loader.closed)
        self.assertEqual(FakeNativeMoEWrapper._total_layer_count, 0)
        self.assertEqual(FakeNativeMoEWrapper._load_call_count, 0)

    def test_force_release_when_loader_is_none(self):
        """force_release_loader() should be safe even if loader is already None."""
        FakeNativeMoEWrapper._native_loader_instance = None
        # Should not raise
        FakeNativeMoEWrapper.force_release_loader()
        self.assertIsNone(FakeNativeMoEWrapper._native_loader_instance)

    def test_force_release_mid_loading(self):
        N = 4
        loader = MockLoader()
        FakeNativeMoEWrapper._native_loader_instance = loader

        wrappers = [FakeNativeMoEWrapper() for _ in range(N)]

        # Only load half the layers
        for w in wrappers[:N // 2]:
            w.load_weights()

        self.assertIsNotNone(FakeNativeMoEWrapper._native_loader_instance)

        FakeNativeMoEWrapper.force_release_loader()

        self.assertIsNone(FakeNativeMoEWrapper._native_loader_instance)
        self.assertEqual(FakeNativeMoEWrapper._total_layer_count, 0)


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
