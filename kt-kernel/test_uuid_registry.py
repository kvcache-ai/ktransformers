#!/usr/bin/env python3
"""
Test script for UUID-based model registry
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

from kt_kernel.cli.utils.user_model_registry import UserModel, UserModelRegistry


def test_uuid_generation():
    """Test that UUIDs are auto-generated"""
    print("Test 1: UUID Auto-generation")

    # Create model without id
    model = UserModel(name="test-model", path="/tmp/test", format="safetensors")

    assert model.id is not None, "UUID should be auto-generated"
    assert len(model.id) == 36, "UUID should be 36 characters"  # Standard UUID format
    print(f"  ✓ UUID generated: {model.id}")


def test_gpu_model_ids_default():
    """Test that gpu_model_ids defaults to None"""
    print("\nTest 2: gpu_model_ids Default Value")

    model = UserModel(name="test-cpu-model", path="/tmp/test-cpu", format="gguf")

    assert model.gpu_model_ids is None, "gpu_model_ids should default to None"
    print(f"  ✓ gpu_model_ids = {model.gpu_model_ids}")


def test_get_model_by_id():
    """Test get_model_by_id functionality"""
    print("\nTest 3: Get Model by ID")

    # Create a temporary registry (in-memory)
    registry = UserModelRegistry(registry_file=Path("/tmp/test_registry.yaml"))

    # Add two models
    model1 = UserModel(name="model-1", path="/tmp/m1", format="safetensors")
    model2 = UserModel(name="model-2", path="/tmp/m2", format="gguf")

    uuid1 = model1.id
    uuid2 = model2.id

    registry.models = [model1, model2]

    # Test retrieval
    found1 = registry.get_model_by_id(uuid1)
    assert found1 is not None, "Should find model by UUID"
    assert found1.name == "model-1", "Should return correct model"
    print(f"  ✓ Found model by UUID: {uuid1} -> {found1.name}")

    found2 = registry.get_model_by_id(uuid2)
    assert found2.name == "model-2", "Should return correct model"
    print(f"  ✓ Found model by UUID: {uuid2} -> {found2.name}")

    # Test non-existent UUID
    not_found = registry.get_model_by_id("non-existent-uuid")
    assert not_found is None, "Should return None for non-existent UUID"
    print(f"  ✓ Returns None for non-existent UUID")


def test_gpu_model_ids_storage():
    """Test storing and retrieving gpu_model_ids"""
    print("\nTest 4: GPU Model IDs Storage")

    model = UserModel(name="cpu-model", path="/tmp/cpu", format="gguf")

    # Initially None
    assert model.gpu_model_ids is None

    # Set GPU model IDs
    gpu_ids = ["uuid-1", "uuid-2", "uuid-3"]
    model.gpu_model_ids = gpu_ids

    assert model.gpu_model_ids == gpu_ids, "Should store GPU model IDs"
    assert len(model.gpu_model_ids) == 3, "Should store all IDs"
    print(f"  ✓ Stored {len(gpu_ids)} GPU model IDs")
    print(f"  ✓ IDs: {model.gpu_model_ids}")


def test_model_dict_serialization():
    """Test that models can be serialized to/from dict"""
    print("\nTest 5: Dict Serialization")

    model = UserModel(name="test-model", path="/tmp/test", format="safetensors", gpu_model_ids=["uuid-1", "uuid-2"])

    # Convert to dict
    model_dict = model.to_dict()
    assert "id" in model_dict, "Dict should contain id"
    assert "gpu_model_ids" in model_dict, "Dict should contain gpu_model_ids"
    print(f"  ✓ Serialized to dict: {list(model_dict.keys())}")

    # Convert back from dict
    model2 = UserModel.from_dict(model_dict)
    assert model2.id == model.id, "ID should be preserved"
    assert model2.gpu_model_ids == model.gpu_model_ids, "gpu_model_ids should be preserved"
    print(f"  ✓ Deserialized from dict successfully")


if __name__ == "__main__":
    print("=" * 60)
    print("UUID-based Model Registry Tests")
    print("=" * 60)

    try:
        test_uuid_generation()
        test_gpu_model_ids_default()
        test_get_model_by_id()
        test_gpu_model_ids_storage()
        test_model_dict_serialization()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
