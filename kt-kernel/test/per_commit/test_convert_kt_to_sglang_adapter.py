import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=5, suite="default")

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "convert_kt_to_sglang_adapter.py"
)
SPEC = importlib.util.spec_from_file_location("convert_kt_to_sglang_adapter", SCRIPT_PATH)
converter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(converter)


def _write_full_fused_checkpoint(path: Path, *, rank: int = 3) -> dict[str, torch.Tensor]:
    e, h, i = 2, 5, 7
    tensors = {
        "layers.2.experts.gate_lora_a": torch.arange(e * rank * h, dtype=torch.float32).reshape(e, rank, h),
        "layers.2.experts.gate_lora_b": torch.arange(e * i * rank, dtype=torch.float32).reshape(e, i, rank),
        "layers.2.experts.up_lora_a": torch.arange(e * rank * h, dtype=torch.float32).reshape(e, rank, h) + 100,
        "layers.2.experts.up_lora_b": torch.arange(e * i * rank, dtype=torch.float32).reshape(e, i, rank) + 200,
        "layers.2.experts.down_lora_a": torch.arange(e * rank * i, dtype=torch.float32).reshape(e, rank, i) + 300,
        "layers.2.experts.down_lora_b": torch.arange(e * h * rank, dtype=torch.float32).reshape(e, h, rank) + 400,
    }
    save_file(tensors, str(path / converter.FUSED_EXPERT_LORA_FILE))
    return tensors


def test_convert_fused_expert_lora_shapes_keys_and_config(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    fused = _write_full_fused_checkpoint(input_dir)

    summary = converter.convert_kt_to_sglang_adapter(
        input_dir,
        output_dir,
        base_model_name_or_path="/models/base",
        lora_alpha=16,
    )

    out = load_file(str(output_dir / converter.ADAPTER_MODEL_FILE))
    assert summary["tensor_count"] == 12
    assert summary["rank"] == 3
    assert summary["target_modules"] == ["gate_proj", "up_proj", "down_proj"]

    key = "model.layers.2.mlp.experts.1.gate_proj.lora_A.weight"
    assert out[key].shape == (3, 5)
    torch.testing.assert_close(out[key], fused["layers.2.experts.gate_lora_a"][1])

    key = "model.layers.2.mlp.experts.0.down_proj.lora_B.weight"
    assert out[key].shape == (5, 3)
    torch.testing.assert_close(out[key], fused["layers.2.experts.down_lora_b"][0])

    config = converter._load_json(output_dir / converter.ADAPTER_CONFIG_FILE)
    assert config["peft_type"] == "LORA"
    assert config["r"] == 3
    assert config["lora_alpha"] == 16
    assert config["base_model_name_or_path"] == "/models/base"
    assert config["target_modules"] == ["gate_proj", "up_proj", "down_proj"]


def test_merges_existing_adapter_and_prefers_existing_lora_alpha(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    _write_full_fused_checkpoint(input_dir)

    existing_tensor = torch.ones(3, 5)
    save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": existing_tensor,
        },
        str(input_dir / converter.ADAPTER_MODEL_FILE),
    )
    converter._write_json(
        input_dir / converter.ADAPTER_CONFIG_FILE,
        {
            "peft_type": "LORA",
            "r": 3,
            "lora_alpha": 9,
            "target_modules": ["q_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "old-base",
        },
    )

    summary = converter.convert_kt_to_sglang_adapter(
        input_dir,
        output_dir,
        base_model_name_or_path="/models/base",
        lora_alpha=16,
    )

    out = load_file(str(output_dir / converter.ADAPTER_MODEL_FILE))
    cleaned_key = "model.layers.0.self_attn.q_proj.lora_A.weight"
    assert cleaned_key in out
    torch.testing.assert_close(out[cleaned_key], existing_tensor)
    assert summary["lora_alpha"] == 9

    config = converter._load_json(output_dir / converter.ADAPTER_CONFIG_FILE)
    assert config["lora_alpha"] == 9
    assert config["base_model_name_or_path"] == "/models/base"
    assert config["target_modules"] == ["q_proj", "gate_proj", "up_proj", "down_proj"]


def test_writes_split_expert_and_nonexpert_adapters(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    expert_dir = tmp_path / "expert"
    nonexpert_dir = tmp_path / "nonexpert"
    input_dir.mkdir()
    _write_full_fused_checkpoint(input_dir)

    q_proj_tensor = torch.ones(3, 5)
    o_proj_tensor = torch.full((5, 3), 2.0)
    save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": q_proj_tensor,
            "base_model.model.model.layers.0.self_attn.o_proj.lora_B.weight": o_proj_tensor,
        },
        str(input_dir / converter.ADAPTER_MODEL_FILE),
    )
    converter._write_json(
        input_dir / converter.ADAPTER_CONFIG_FILE,
        {
            "peft_type": "LORA",
            "r": 3,
            "lora_alpha": 9,
            "target_modules": ["q_proj", "o_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "old-base",
        },
    )

    summary = converter.convert_kt_to_sglang_adapter(
        input_dir,
        output_dir,
        base_model_name_or_path="/models/base",
        expert_output_dir=expert_dir,
        nonexpert_output_dir=nonexpert_dir,
    )

    merged = load_file(str(output_dir / converter.ADAPTER_MODEL_FILE))
    expert = load_file(str(expert_dir / converter.ADAPTER_MODEL_FILE))
    nonexpert = load_file(str(nonexpert_dir / converter.ADAPTER_MODEL_FILE))

    assert summary["tensor_count"] == 14
    assert summary["split_outputs"]["expert"]["tensor_count"] == 12
    assert summary["split_outputs"]["nonexpert"]["tensor_count"] == 2
    assert set(merged) == set(expert) | set(nonexpert)
    assert set(expert).isdisjoint(nonexpert)
    assert all(".mlp.experts." in key for key in expert)
    assert not any(".mlp.experts." in key for key in nonexpert)

    cleaned_q_proj_key = "model.layers.0.self_attn.q_proj.lora_A.weight"
    cleaned_o_proj_key = "model.layers.0.self_attn.o_proj.lora_B.weight"
    torch.testing.assert_close(nonexpert[cleaned_q_proj_key], q_proj_tensor)
    torch.testing.assert_close(nonexpert[cleaned_o_proj_key], o_proj_tensor)

    expert_config = converter._load_json(expert_dir / converter.ADAPTER_CONFIG_FILE)
    nonexpert_config = converter._load_json(nonexpert_dir / converter.ADAPTER_CONFIG_FILE)
    assert expert_config["target_modules"] == ["gate_proj", "up_proj", "down_proj"]
    assert nonexpert_config["target_modules"] == ["q_proj", "o_proj"]
    assert expert_config["base_model_name_or_path"] == "/models/base"
    assert nonexpert_config["base_model_name_or_path"] == "/models/base"


def test_requires_lora_alpha_without_input_config(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_full_fused_checkpoint(input_dir)

    with pytest.raises(ValueError, match="pass --lora-alpha"):
        converter.convert_kt_to_sglang_adapter(
            input_dir,
            tmp_path / "output",
            base_model_name_or_path="/models/base",
        )


def test_rejects_inconsistent_rank(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    save_file(
        {
            "layers.0.experts.gate_lora_a": torch.zeros(2, 3, 5),
            "layers.0.experts.gate_lora_b": torch.zeros(2, 7, 4),
        },
        str(input_dir / converter.FUSED_EXPERT_LORA_FILE),
    )

    with pytest.raises(ValueError, match="Inconsistent LoRA ranks"):
        converter.convert_kt_to_sglang_adapter(
            input_dir,
            tmp_path / "output",
            base_model_name_or_path="/models/base",
            lora_alpha=8,
        )


def test_rejects_unexpected_fused_key(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    save_file(
        {"layers.0.experts.unknown_lora_a": torch.zeros(2, 3, 5)},
        str(input_dir / converter.FUSED_EXPERT_LORA_FILE),
    )

    with pytest.raises(ValueError, match="Unsupported KT fused expert LoRA tensor"):
        converter.convert_kt_to_sglang_adapter(
            input_dir,
            tmp_path / "output",
            base_model_name_or_path="/models/base",
            lora_alpha=8,
        )


def test_rejects_nonempty_output_without_overwrite(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("do not remove", encoding="utf-8")
    _write_full_fused_checkpoint(input_dir)

    with pytest.raises(FileExistsError, match="Output directory is not empty"):
        converter.convert_kt_to_sglang_adapter(
            input_dir,
            output_dir,
            base_model_name_or_path="/models/base",
            lora_alpha=8,
        )

    converter.convert_kt_to_sglang_adapter(
        input_dir,
        output_dir,
        base_model_name_or_path="/models/base",
        lora_alpha=8,
        overwrite=True,
    )
    assert not (output_dir / "existing.txt").exists()
    assert (output_dir / converter.ADAPTER_MODEL_FILE).exists()


def test_rejects_output_same_as_input(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    _write_full_fused_checkpoint(input_dir)

    with pytest.raises(ValueError, match="different from input"):
        converter.convert_kt_to_sglang_adapter(
            input_dir,
            input_dir,
            base_model_name_or_path="/models/base",
            lora_alpha=8,
            overwrite=True,
        )
