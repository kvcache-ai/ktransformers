"""Synthetic unit fixtures only; these are never model-acceptance receipts."""
import copy

import pytest

import manual_promote as module


def fixture(monkeypatch):
    checked = []
    monkeypatch.setattr(module, "verify_install_report", lambda report, manifest, extra, public: checked.append((extra, public)))
    manifest = {"source_lock": {"example": "unit fixture"}, "wheels": {"example": "not a real wheel"}, "run_id": 1, "run_attempt": 1}
    record = {"rank": 0, "global_step": 1, "optimizer_steps": 1, "train_end": True, "raw_losses": [2.0]}
    data = {
        "schema": 1, "execution": "manual", "manifest_sha256": "b" * 64,
        "source_lock": manifest["source_lock"], "wheels": manifest["wheels"], "candidate_run_id": 1, "candidate_attempt": 1,
        "install_reports": {host: {extra: {} for extra in ("sglang", "sglang,sft")} for host in ("sap4", "qj5090")},
        "runtime_integrity": {host: {"fresh_venvs": True, "no_upstream_namespace": True, "before_tooling_sha256": "a" * 64, "after_tooling_sha256": "a" * 64} for host in ("sap4", "qj5090")},
        "qj5090": {"qwen3_lora": {"records": [record]}, "deepseek_v31_lora": {"records": [dict(record, rank=rank) for rank in range(8)]}, "glm": {"completion_tokens": 1, "content": "Paris"}},
        "sap4": {"kimi": {
            "global_step": 32, "adapter_status": "ready", "loss_records": [{"loss": 2.0, "grad_norm": 0.5}],
            "adapter_files": {name: {"nonzero_B_tensors": 1, "tensors": 1, "sha256": "c" * 64} for name in ("adapter_model.safetensors", "fused_expert_lora.safetensors")},
            "style_answers": [{"completion_tokens": 1, "content": "喵"}] * 4,
            "heldout_overlap_count": 0, "baseline_completed": True, "arithmetic_correct": True, "conversion_from_main_script": True,
        }},
    }
    return data, manifest, checked


def test_acceptance_checks_both_extras_on_both_hosts(monkeypatch):
    data, manifest, checked = fixture(monkeypatch)
    module.check_attestation(data, manifest, "b" * 64)
    assert checked == [("sglang", False), ("sglang,sft", False)] * 2


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(execution="ci"),
    lambda d: d.update(manifest_sha256="0" * 64),
    lambda d: d.update(candidate_run_id=2),
    lambda d: d["runtime_integrity"]["sap4"].update(after_tooling_sha256="f" * 64),
    lambda d: d["qj5090"]["qwen3_lora"]["records"][0].update(raw_losses=[float("nan")]),
    lambda d: d["qj5090"]["deepseek_v31_lora"]["records"].pop(),
    lambda d: d["qj5090"]["glm"].update(content="Unknown"),
    lambda d: d["sap4"]["kimi"].update(global_step=2),
    lambda d: d["sap4"]["kimi"].update(heldout_overlap_count=1),
    lambda d: d["sap4"]["kimi"].update(style_answers=[{"completion_tokens": 1, "content": "ordinary"}] * 4),
    lambda d: d["sap4"]["kimi"]["loss_records"][0].update(grad_norm=float("inf")),
])
def test_incomplete_or_changed_acceptance_cannot_publish(monkeypatch, mutation):
    data, manifest, _ = fixture(monkeypatch)
    changed = copy.deepcopy(data)
    mutation(changed)
    with pytest.raises(ValueError):
        module.check_attestation(changed, manifest, "b" * 64)
