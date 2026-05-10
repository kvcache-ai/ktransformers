"""Tests for NativeMoEWrapper layerwise mmap-release mechanism.

Verifies that the SafeTensor loader singleton (_native_loader_instance) is
released after EACH layer's load_weights() completes (not just after all layers),
and that the loader is recreated on demand for the next layer.

These tests use mocking so they can run without actual safetensors files or
compiled kt_kernel_ext binaries.
"""

import sys
import os
import types
import unittest
from unittest.mock import MagicMock, patch


class MockLoader:
    """Minimal mock SafeTensorLoader."""

    _create_count = 0  # Track how many times a loader was created

    def __init__(self):
        MockLoader._create_count += 1
        self.closed = False
        self.file_handle_map = {"dummy.safetensors": object()}

    def close_all_handles(self):
        self.closed = True
        self.file_handle_map.clear()


class FakeNativeMoEWrapper:
    """
    A simplified replica of NativeMoEWrapper that isolates the
    layerwise-release + recreate logic without requiring kt_kernel_ext.
    """

    _native_loader_instance = None
    # Simulate _create_loader: returns a fresh MockLoader each time
    _create_loader_calls = 0

    def __init__(self, layer_idx=0):
        self.layer_idx = layer_idx
        self.method = "FP8"
        self.weight_path = "/fake/path"

    def _ensure_loader(self):
        """Simulate the loader-recreate logic at the start of load_weights."""
        if FakeNativeMoEWrapper._native_loader_instance is None:
            FakeNativeMoEWrapper._create_loader_calls += 1
            FakeNativeMoEWrapper._native_loader_instance = MockLoader()
        self.loader = FakeNativeMoEWrapper._native_loader_instance

    def load_weights(self):
        """Simulate load_weights: ensure loader -> do work -> release loader."""
        self._ensure_loader()
        # Simulate: C++ sync + del Python tensors -> release
        FakeNativeMoEWrapper._release_loader(layer_idx=self.layer_idx)

    @staticmethod
    def _release_loader(layer_idx=-1):
        if FakeNativeMoEWrapper._native_loader_instance is not None:
            FakeNativeMoEWrapper._native_loader_instance.close_all_handles()
            FakeNativeMoEWrapper._native_loader_instance = None

    @staticmethod
    def force_release_loader():
        FakeNativeMoEWrapper._release_loader()


def _reset_state():
    """Reset all test state between tests."""
    FakeNativeMoEWrapper._native_loader_instance = None
    FakeNativeMoEWrapper._create_loader_calls = 0
    MockLoader._create_count = 0


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestLayerwiseRelease(unittest.TestCase):
    """Each layer's load_weights() should release the loader afterwards."""

    def setUp(self):
        _reset_state()

    def test_single_layer_released_after_load(self):
        w = FakeNativeMoEWrapper(layer_idx=0)
        w.load_weights()
        self.assertIsNone(FakeNativeMoEWrapper._native_loader_instance,
                          "Loader should be None after single layer loads")

    def test_each_layer_releases_loader(self):
        """After every layer's load_weights(), the loader should be None."""
        for i in range(5):
            w = FakeNativeMoEWrapper(layer_idx=i)
            w.load_weights()
            self.assertIsNone(
                FakeNativeMoEWrapper._native_loader_instance,
                f"Loader should be None after layer {i} loads",
            )

    def test_loader_recreated_for_each_layer(self):
        """Each layer should trigger a loader recreation (since previous layer released it)."""
        N = 4
        for i in range(N):
            w = FakeNativeMoEWrapper(layer_idx=i)
            w.load_weights()

        # First layer uses the initial loader; layers 1..N-1 recreate it
        # Total recreations = N - 1 (layer 0 doesn't recreate if loader pre-existed,
        # but in this test the loader starts as None so all N layers recreate)
        self.assertEqual(
            FakeNativeMoEWrapper._create_loader_calls, N,
            f"Expected {N} loader recreations for {N} layers, "
            f"got {FakeNativeMoEWrapper._create_loader_calls}",
        )


class TestLoaderRecreate(unittest.TestCase):
    """Loader should be recreated when _native_loader_instance is None."""

    def setUp(self):
        _reset_state()

    def test_first_layer_creates_loader(self):
        w = FakeNativeMoEWrapper(layer_idx=0)
        w.load_weights()
        # Loader was created (then released), but the creation happened
        self.assertGreater(FakeNativeMoEWrapper._create_loader_calls, 0)

    def test_second_layer_recreates_after_first_released(self):
        w0 = FakeNativeMoEWrapper(layer_idx=0)
        w0.load_weights()
        self.assertIsNone(FakeNativeMoEWrapper._native_loader_instance)

        w1 = FakeNativeMoEWrapper(layer_idx=1)
        w1.load_weights()
        # Second layer should have recreated the loader
        self.assertEqual(FakeNativeMoEWrapper._create_loader_calls, 2,
                         "Both layers should recreate the loader")

    def test_pre_existing_loader_not_recreated(self):
        """If loader already exists (e.g., from __init__), it should not be recreated."""
        loader = MockLoader()
        FakeNativeMoEWrapper._native_loader_instance = loader
        initial_create_calls = FakeNativeMoEWrapper._create_loader_calls

        w = FakeNativeMoEWrapper(layer_idx=0)
        w._ensure_loader()
        # No new creation should happen
        self.assertEqual(FakeNativeMoEWrapper._create_loader_calls, initial_create_calls)
        self.assertIs(w.loader, loader)


class TestForceReleaseLoader(unittest.TestCase):
    """force_release_loader() should work at any time."""

    def setUp(self):
        _reset_state()

    def test_force_release_before_any_load(self):
        loader = MockLoader()
        FakeNativeMoEWrapper._native_loader_instance = loader

        FakeNativeMoEWrapper.force_release_loader()

        self.assertIsNone(FakeNativeMoEWrapper._native_loader_instance)
        self.assertTrue(loader.closed)

    def test_force_release_when_loader_is_none(self):
        """force_release_loader() should be safe even if loader is already None."""
        FakeNativeMoEWrapper._native_loader_instance = None
        FakeNativeMoEWrapper.force_release_loader()
        self.assertIsNone(FakeNativeMoEWrapper._native_loader_instance)

    def test_force_release_mid_loading(self):
        loader = MockLoader()
        FakeNativeMoEWrapper._native_loader_instance = loader

        # Load first layer
        w0 = FakeNativeMoEWrapper(layer_idx=0)
        w0.load_weights()
        # Loader is now released (each layer releases it)

        # Set a new loader manually
        loader2 = MockLoader()
        FakeNativeMoEWrapper._native_loader_instance = loader2

        # Force release before next load
        FakeNativeMoEWrapper.force_release_loader()

        self.assertIsNone(FakeNativeMoEWrapper._native_loader_instance)
        self.assertTrue(loader2.closed)


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
