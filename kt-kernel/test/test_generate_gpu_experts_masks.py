"""Test for generate_gpu_experts_masks function."""

import sys
import os

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import torch
import time
from experts_base import generate_gpu_experts_masks


def test_basic():
    """Test basic functionality."""
    print("=" * 60)
    print("Test 1: Basic functionality")
    print("=" * 60)

    activation_freq = torch.tensor([
        [0.1, 0.5, 0.3, 0.8],  # layer 0
        [0.2, 0.4, 0.9, 0.1],  # layer 1
    ])

    print(f"Input activation_freq:\n{activation_freq}")
    print(f"num_gpu_experts: 3")

    masks = generate_gpu_experts_masks(activation_freq, num_gpu_experts=3)

    print(f"Output masks:\n{masks}")
    print(f"Output dtype: {masks.dtype}, device: {masks.device}")

    # Verify: top 3 should be (1,2)=0.9, (0,3)=0.8, (0,1)=0.5
    expected_gpu_count = masks.sum().item()
    print(f"Total GPU experts: {expected_gpu_count}")

    # Check the top 3 positions
    assert masks[1, 2] == True, "layer1-expert2 (0.9) should be on GPU"
    assert masks[0, 3] == True, "layer0-expert3 (0.8) should be on GPU"
    assert masks[0, 1] == True, "layer0-expert1 (0.5) should be on GPU"
    assert expected_gpu_count == 3, f"Expected 3 GPU experts, got {expected_gpu_count}"

    print("PASSED\n")


def test_edge_cases():
    """Test edge cases."""
    print("=" * 60)
    print("Test 2: Edge cases")
    print("=" * 60)

    activation_freq = torch.tensor([
        [0.1, 0.5, 0.3, 0.8],
        [0.2, 0.4, 0.9, 0.1],
    ])

    # Test num_gpu_experts = 0
    masks = generate_gpu_experts_masks(activation_freq, num_gpu_experts=0)
    assert masks.sum().item() == 0, "num_gpu_experts=0 should have no GPU experts"
    print(f"num_gpu_experts=0: {masks.sum().item()} GPU experts - PASSED")

    # Test num_gpu_experts = total experts
    masks = generate_gpu_experts_masks(activation_freq, num_gpu_experts=8)
    assert masks.sum().item() == 8, "num_gpu_experts=8 should have all experts on GPU"
    print(f"num_gpu_experts=8 (all): {masks.sum().item()} GPU experts - PASSED")

    # Test num_gpu_experts > total experts (should clamp)
    masks = generate_gpu_experts_masks(activation_freq, num_gpu_experts=100)
    assert masks.sum().item() == 8, "num_gpu_experts=100 should be clamped to 8"
    print(f"num_gpu_experts=100 (clamped): {masks.sum().item()} GPU experts - PASSED")

    # Test negative num_gpu_experts (should clamp to 0)
    masks = generate_gpu_experts_masks(activation_freq, num_gpu_experts=-5)
    assert masks.sum().item() == 0, "num_gpu_experts=-5 should be clamped to 0"
    print(f"num_gpu_experts=-5 (clamped): {masks.sum().item()} GPU experts - PASSED")

    print("All edge cases PASSED\n")


def test_performance():
    """Test performance with realistic sizes."""
    print("=" * 60)
    print("Test 3: Performance")
    print("=" * 60)

    # DeepSeek-V3 like: 61 layers, 256 experts
    num_layers = 61
    num_experts = 256

    # Generate random activation frequencies
    activation_freq = torch.rand(num_layers, num_experts)

    # Test with different num_gpu_experts
    test_cases = [0, 100, 500, 1000, 2000, 5000, num_layers * num_experts]

    print(f"Shape: ({num_layers}, {num_experts}) = {num_layers * num_experts} total experts\n")

    for num_gpu in test_cases:
        # Warmup
        _ = generate_gpu_experts_masks(activation_freq, num_gpu_experts=num_gpu)

        # Measure time
        num_runs = 100
        start = time.perf_counter()
        for _ in range(num_runs):
            masks = generate_gpu_experts_masks(activation_freq, num_gpu_experts=num_gpu)
        end = time.perf_counter()

        avg_time_us = (end - start) / num_runs * 1e6
        actual_gpu = masks.sum().item()

        print(f"num_gpu_experts={num_gpu:5d} -> actual={actual_gpu:5d}, time={avg_time_us:8.2f} us")

    print("\nPerformance test PASSED\n")


def test_output_properties():
    """Test output tensor properties."""
    print("=" * 60)
    print("Test 4: Output properties")
    print("=" * 60)

    activation_freq = torch.rand(10, 64)
    masks = generate_gpu_experts_masks(activation_freq, num_gpu_experts=50)

    print(f"Shape: {masks.shape}")
    print(f"Dtype: {masks.dtype}")
    print(f"Device: {masks.device}")
    print(f"Is contiguous: {masks.is_contiguous()}")

    assert masks.shape == (10, 64), f"Expected shape (10, 64), got {masks.shape}"
    assert masks.dtype == torch.bool, f"Expected dtype bool, got {masks.dtype}"
    assert str(masks.device) == "cpu", f"Expected device cpu, got {masks.device}"

    print("All properties PASSED\n")


def test_determinism():
    """Test that results are deterministic."""
    print("=" * 60)
    print("Test 5: Determinism")
    print("=" * 60)

    activation_freq = torch.rand(20, 128)

    masks1 = generate_gpu_experts_masks(activation_freq, num_gpu_experts=100)
    masks2 = generate_gpu_experts_masks(activation_freq, num_gpu_experts=100)

    assert torch.equal(masks1, masks2), "Results should be deterministic"
    print("Determinism PASSED\n")


if __name__ == "__main__":
    test_basic()
    test_edge_cases()
    test_output_properties()
    test_determinism()
    test_performance()

    print("=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
