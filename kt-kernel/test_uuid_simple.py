#!/usr/bin/env python3
"""
Simple test for UUID functionality without loading full kt_kernel
"""

import sys
from pathlib import Path
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, List


# Simplified UserModel for testing (copied from user_model_registry.py)
@dataclass
class UserModel:
    """Represents a user-registered model"""

    name: str
    path: str
    format: str
    id: Optional[str] = None
    repo_type: Optional[str] = None
    repo_id: Optional[str] = None
    sha256_status: str = "not_checked"
    gpu_model_ids: Optional[List[str]] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_verified: Optional[str] = None

    def __post_init__(self):
        """Ensure ID is set after initialization"""
        if self.id is None:
            import uuid

            self.id = str(uuid.uuid4())

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


def test_uuid_generation():
    """Test that UUIDs are auto-generated"""
    print("Test 1: UUID Auto-generation")

    # Create model without id
    model = UserModel(name="test-model", path="/tmp/test", format="safetensors")

    assert model.id is not None, "UUID should be auto-generated"
    assert len(model.id) == 36, "UUID should be 36 characters"
    print(f"  ✓ UUID generated: {model.id}")


def test_gpu_model_ids_default():
    """Test that gpu_model_ids defaults to None"""
    print("\nTest 2: gpu_model_ids Default Value")

    model = UserModel(name="test-cpu-model", path="/tmp/test-cpu", format="gguf")

    assert model.gpu_model_ids is None, "gpu_model_ids should default to None"
    print(f"  ✓ gpu_model_ids = {model.gpu_model_ids}")


def test_gpu_model_ids_storage():
    """Test storing and retrieving gpu_model_ids"""
    print("\nTest 3: GPU Model IDs Storage")

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
    print("\nTest 4: Dict Serialization")

    model = UserModel(name="test-model", path="/tmp/test", format="safetensors", gpu_model_ids=["uuid-1", "uuid-2"])

    original_id = model.id

    # Convert to dict
    model_dict = model.to_dict()
    assert "id" in model_dict, "Dict should contain id"
    assert "gpu_model_ids" in model_dict, "Dict should contain gpu_model_ids"
    print(f"  ✓ Serialized to dict")
    print(f"    Keys: {list(model_dict.keys())}")

    # Convert back from dict
    model2 = UserModel.from_dict(model_dict)
    assert model2.id == original_id, "ID should be preserved"
    assert model2.gpu_model_ids == model.gpu_model_ids, "gpu_model_ids should be preserved"
    print(f"  ✓ Deserialized from dict successfully")
    print(f"    ID preserved: {model2.id == original_id}")
    print(f"    gpu_model_ids preserved: {model2.gpu_model_ids == model.gpu_model_ids}")


def test_uuid_uniqueness():
    """Test that each model gets a unique UUID"""
    print("\nTest 5: UUID Uniqueness")

    models = []
    for i in range(10):
        model = UserModel(name=f"model-{i}", path=f"/tmp/model-{i}", format="safetensors")
        models.append(model)

    uuids = [m.id for m in models]
    unique_uuids = set(uuids)

    assert len(unique_uuids) == 10, "All UUIDs should be unique"
    print(f"  ✓ Generated 10 unique UUIDs")
    print(f"    First 3: {uuids[:3]}")


if __name__ == "__main__":
    print("=" * 60)
    print("UUID-based Model Registry Tests (Simplified)")
    print("=" * 60)

    try:
        test_uuid_generation()
        test_gpu_model_ids_default()
        test_gpu_model_ids_storage()
        test_model_dict_serialization()
        test_uuid_uniqueness()

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
