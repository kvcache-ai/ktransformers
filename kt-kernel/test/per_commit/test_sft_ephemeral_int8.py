# SPDX-License-Identifier: Apache-2.0

import importlib.util
import os
from pathlib import Path
import sys

import pytest


SFT_PATH = Path(__file__).resolve().parents[2] / "python" / "sft"
SPEC = importlib.util.spec_from_file_location(
    "kt_sft_ephemeral_under_test", SFT_PATH / "ephemeral.py"
)
assert SPEC is not None and SPEC.loader is not None
ephemeral = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ephemeral
SPEC.loader.exec_module(ephemeral)


def _make_staging(tmp_path: Path, run_id: str = "pytest-run") -> Path:
    staging = tmp_path / f"kt-int8-{run_id}.staging"
    staging.mkdir(mode=0o700)
    for layer_idx in (3, 4):
        for numa_idx in range(2):
            numa = staging / f"_layer_{layer_idx}" / f"_numa_{numa_idx}"
            numa.mkdir(parents=True)
            for projection in ("gate", "up", "down"):
                for expert_idx in range(2):
                    for kind in ("quant", "scale"):
                        path = numa / f"INT8_{projection}_{expert_idx}_1Byte_{kind}_.kt"
                        path.write_bytes(bytes([layer_idx + numa_idx + expert_idx + 1]))
    return staging


def _publish(tmp_path: Path, monkeypatch, run_id: str = "pytest-run") -> Path:
    monkeypatch.setattr(ephemeral, "_EPHEMERAL_BASE", tmp_path)
    staging = _make_staging(tmp_path, run_id)
    return ephemeral.publish_ephemeral_int8_weights(
        staging,
        run_id=run_id,
        layer_indices=[3, 4],
        numa_count=2,
        expert_num=2,
        hidden_size=8,
        intermediate_size=16,
    )


def _open(ready: Path):
    return ephemeral.EphemeralKTWeightStore.open(
        ready,
        layer_indices=[3, 4],
        numa_count=2,
        expert_num=2,
        hidden_size=8,
        intermediate_size=16,
    )


def test_ephemeral_int8_publish_consume_and_finish(tmp_path, monkeypatch):
    ready = _publish(tmp_path, monkeypatch)
    assert ready.name == "kt-int8-pytest-run"
    assert (ready / ephemeral.MANIFEST_NAME).is_file()

    store = _open(ready)
    store.consume_layer(3)
    assert not (ready / "_layer_3").exists()
    assert (ready / "_layer_4").is_dir()
    store.consume_layer(4)
    store.finish()
    assert not ready.exists()


def test_ephemeral_manifest_rejects_tampered_size(tmp_path, monkeypatch):
    ready = _publish(tmp_path, monkeypatch)
    target = next((ready / "_layer_3").rglob("*.kt"))
    target.write_bytes(b"changed-size")
    with pytest.raises(ValueError, match="size mismatch"):
        _open(ready)


def test_ephemeral_publish_requires_complete_exact_file_set(tmp_path, monkeypatch):
    monkeypatch.setattr(ephemeral, "_EPHEMERAL_BASE", tmp_path)
    staging = _make_staging(tmp_path)
    next((staging / "_layer_3").rglob("*.kt")).unlink()
    with pytest.raises(ValueError, match="missing 1 INT8 files"):
        ephemeral.publish_ephemeral_int8_weights(
            staging,
            run_id="pytest-run",
            layer_indices=[3, 4],
            numa_count=2,
            expert_num=2,
            hidden_size=8,
            intermediate_size=16,
        )


def test_cleanup_never_deletes_unlisted_files(tmp_path, monkeypatch):
    ready = _publish(tmp_path, monkeypatch)
    store = _open(ready)
    unlisted = ready / "do-not-delete.txt"
    unlisted.write_text("owned by somebody else", encoding="utf-8")
    store.cleanup()
    assert unlisted.read_text(encoding="utf-8") == "owned by somebody else"
    assert ready.exists()
    unlisted.unlink()
    ready.rmdir()


def test_ephemeral_root_must_be_private_and_run_scoped(tmp_path, monkeypatch):
    monkeypatch.setattr(ephemeral, "_EPHEMERAL_BASE", tmp_path)
    staging = _make_staging(tmp_path)
    os.chmod(staging, 0o770)
    with pytest.raises(PermissionError, match="group- or world-writable"):
        ephemeral.publish_ephemeral_int8_weights(
            staging,
            run_id="pytest-run",
            layer_indices=[3, 4],
            numa_count=2,
            expert_num=2,
            hidden_size=8,
            intermediate_size=16,
        )


@pytest.mark.parametrize("run_id", ["../escape", "short", "contains space"])
def test_ephemeral_run_id_is_validated_before_path_construction(run_id):
    with pytest.raises(ValueError, match="run_id"):
        ephemeral.validate_ephemeral_run_id(run_id)


def test_stale_staging_lease_can_be_reclaimed(tmp_path, monkeypatch):
    monkeypatch.setattr(ephemeral, "_EPHEMERAL_BASE", tmp_path)
    staging = _make_staging(tmp_path)
    ephemeral.write_ephemeral_staging_lease(
        staging,
        run_id="pytest-run",
        producer_pid=2**30,
    )
    assert ephemeral.reclaim_stale_ephemeral_int8_staging(
        staging,
        run_id="pytest-run",
    )
    assert not staging.exists()


def test_live_staging_lease_is_never_reclaimed(tmp_path, monkeypatch):
    monkeypatch.setattr(ephemeral, "_EPHEMERAL_BASE", tmp_path)
    staging = _make_staging(tmp_path)
    ephemeral.write_ephemeral_staging_lease(
        staging,
        run_id="pytest-run",
        producer_pid=os.getpid(),
    )
    assert not ephemeral.reclaim_stale_ephemeral_int8_staging(
        staging,
        run_id="pytest-run",
    )
    assert staging.exists()
