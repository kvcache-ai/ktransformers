# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import pytest

from kt_kernel.sft import weight_manifest


def _make_tree(
    tmp_path: Path,
    *,
    schema_version: int,
    backend: str | None = None,
) -> Path:
    root = tmp_path / "weights"
    root.mkdir()
    layers = []
    for layer_idx in (0, 3):
        files = []
        for numa_idx in range(2):
            numa_dir = root / f"_layer_{layer_idx}" / f"_numa_{numa_idx}"
            numa_dir.mkdir(parents=True)
            for projection in ("gate", "up", "down"):
                for expert_idx in range(2):
                    for kind in ("quant", "scale"):
                        intermediate_per_numa = 16 // 2
                        size_bytes = {
                            ("gate", "quant"): 8 * intermediate_per_numa,
                            ("up", "quant"): 8 * intermediate_per_numa,
                            ("down", "quant"): 8 * intermediate_per_numa,
                            ("gate", "scale"): intermediate_per_numa * 4,
                            ("up", "scale"): intermediate_per_numa * 4,
                            ("down", "scale"): 8 * 4,
                        }[(projection, kind)]
                        path = numa_dir / f"INT8_{projection}_{expert_idx}_{size_bytes}Byte_{kind}_.kt"
                        payload = bytes([layer_idx + numa_idx + expert_idx + len(kind)]) * size_bytes
                        path.write_bytes(payload)
                        item = {
                            "path": path.relative_to(root).as_posix(),
                            "size_bytes": len(payload),
                        }
                        if schema_version == 2:
                            item["sha256"] = hashlib.sha256(payload).hexdigest()
                        files.append(item)
        layers.append(
            {
                "index": layer_idx,
                "numa_count": 2,
                "state": "ready",
                "files": sorted(files, key=lambda item: item["path"]),
                "bytes": sum(item["size_bytes"] for item in files),
            }
        )

    manifest = {
        "schema_version": schema_version,
        "state": "ready",
        "expert_weight_format": "int8",
        "threadpool_count": 2,
        "expert_num": 2,
        "hidden_size": 8,
        "intermediate_size": 16,
        "layers": layers,
        "bytes": sum(layer["bytes"] for layer in layers),
    }
    if schema_version == 1:
        manifest.update(
            {
                "backend": backend or "AMXINT8",
                "owner_uid": 123456,
                "run_id": "legacy-shared-weights",
            }
        )
        manifest_name = weight_manifest.LEGACY_MANIFEST_NAME
    else:
        manifest["layout"] = weight_manifest.INT8_WEIGHT_LAYOUT
        if backend is not None:
            manifest["backend"] = backend
        manifest_name = weight_manifest.MANIFEST_NAME
    (root / manifest_name).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return root


def _validate(root: Path, **kwargs):
    return weight_manifest.validate_persistent_int8_weights(
        root,
        layer_indices=[0, 3],
        numa_count=2,
        expert_num=2,
        hidden_size=8,
        intermediate_size=16,
        **kwargs,
    )


def test_schema1_shared_persistent_tree_is_accepted_without_owner_check(tmp_path):
    root = _make_tree(tmp_path, schema_version=1)
    result = _validate(root, verify_hashes=None)

    assert result.is_legacy
    assert result.layout == weight_manifest.INT8_WEIGHT_LAYOUT
    assert result.layer_indices == (0, 3)
    assert result.file_count == 48
    assert result.size_bytes == 2 * 2 * 2 * (3 * 64 + 3 * 32)


def test_schema2_neutral_layout_and_hashes_are_accepted(tmp_path):
    root = _make_tree(tmp_path, schema_version=2, backend="INT8")
    result = _validate(root, verify_hashes=True)

    assert not result.is_legacy
    assert result.path.name == weight_manifest.MANIFEST_NAME


def test_schema2_rejects_hardware_specific_backend(tmp_path):
    root = _make_tree(tmp_path, schema_version=2, backend="AMXINT8")
    with pytest.raises(ValueError, match="hardware-neutral"):
        _validate(root)


def test_persistent_manifest_rejects_missing_or_tampered_file(tmp_path):
    root = _make_tree(tmp_path, schema_version=1)
    target = next((root / "_layer_0").rglob("*.kt"))
    target.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="size mismatch"):
        _validate(root)


def test_persistent_manifest_rejects_unlisted_root_or_layer_entries(tmp_path):
    root = _make_tree(tmp_path, schema_version=1)
    (root / "unexpected.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(ValueError, match="root entries differ"):
        _validate(root)


def test_persistent_manifest_rejects_symlinked_weight(tmp_path):
    root = _make_tree(tmp_path, schema_version=1)
    target = next((root / "_layer_0").rglob("*.kt"))
    target.unlink()
    target.symlink_to("/dev/null")

    with pytest.raises(ValueError, match="non-regular"):
        _validate(root)


def test_persistent_manifest_rejects_wrong_model_contract(tmp_path):
    root = _make_tree(tmp_path, schema_version=1)
    with pytest.raises(ValueError, match="hidden_size mismatch"):
        weight_manifest.validate_persistent_int8_weights(
            root,
            layer_indices=[0, 3],
            numa_count=2,
            expert_num=2,
            hidden_size=16,
            intermediate_size=16,
        )


def test_persistent_manifest_rejects_filename_size_contract_mismatch(tmp_path):
    root = _make_tree(tmp_path, schema_version=1)
    manifest_path = root / weight_manifest.LEGACY_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["layers"][0]["files"][0]["size_bytes"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="size contract mismatch"):
        _validate(root)


def test_persistent_manifest_rejects_nondivisible_tp_shape(tmp_path):
    root = _make_tree(tmp_path, schema_version=1)
    with pytest.raises(ValueError, match="must be divisible"):
        weight_manifest.validate_persistent_int8_weights(
            root,
            layer_indices=[0, 3],
            numa_count=3,
            expert_num=2,
            hidden_size=8,
            intermediate_size=16,
        )


def test_persistent_manifest_rejects_ambiguous_manifests(tmp_path):
    root = _make_tree(tmp_path, schema_version=1)
    (root / weight_manifest.MANIFEST_NAME).write_text(
        "{}",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one manifest"):
        _validate(root)


def test_schema2_hash_verification_detects_same_size_corruption(tmp_path):
    root = _make_tree(tmp_path, schema_version=2)
    target = next((root / "_layer_0").rglob("*.kt"))
    payload = bytearray(target.read_bytes())
    payload[0] ^= 0xFF
    target.write_bytes(payload)

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        _validate(root, verify_hashes=None)
