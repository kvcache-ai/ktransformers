import importlib.util
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT
    / "kt-kernel"
    / "scripts"
    / "convert_kt_to_sglang_adapter.py"
)
spec = importlib.util.spec_from_file_location("convert_kt_to_sglang_adapter", SCRIPT_PATH)
converter = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(converter)


def _write_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f)


def _import_kt_ep_wrapper():
    sys.path.insert(0, str(REPO_ROOT / "third_party" / "sglang" / "python"))
    sys.path.insert(0, str(REPO_ROOT / "kt-kernel" / "python"))
    from sglang.srt.layers.moe import kt_ep_wrapper

    return kt_ep_wrapper


def test_convert_kimi_fused_expert_lora_to_merged_adapter(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "merged"
    expert_dir = tmp_path / "expert"
    nonexpert_dir = tmp_path / "nonexpert"
    input_dir.mkdir()

    rank = 2
    hidden = 4
    intermediate = 5
    experts = 3
    nonexpert = {
        "base_model.model.language_model.model.layers.0.self_attn.q_a_proj.lora_A.weight": torch.ones(
            rank, hidden
        ),
        "base_model.model.language_model.model.layers.0.self_attn.q_a_proj.lora_B.weight": torch.ones(
            6, rank
        ),
        "base_model.model.language_model.model.layers.0.self_attn.q_b_proj.lora_A.weight": torch.ones(
            rank, 6
        ),
        "base_model.model.language_model.model.layers.0.self_attn.q_b_proj.lora_B.weight": torch.ones(
            8, rank
        ),
        "base_model.model.language_model.model.layers.0.self_attn.kv_a_proj_with_mqa.lora_A.weight": torch.ones(
            rank, hidden
        ),
        "base_model.model.language_model.model.layers.0.self_attn.kv_a_proj_with_mqa.lora_B.weight": torch.ones(
            3, rank
        ),
        "base_model.model.language_model.model.layers.0.self_attn.kv_b_proj.lora_A.weight": torch.ones(
            rank, 2
        ),
        "base_model.model.language_model.model.layers.0.self_attn.kv_b_proj.lora_B.weight": torch.ones(
            8, rank
        ),
        "base_model.model.language_model.model.layers.0.self_attn.o_proj.lora_A.weight": torch.ones(
            rank, 8
        ),
        "base_model.model.language_model.model.layers.0.self_attn.o_proj.lora_B.weight": torch.ones(
            hidden, rank
        ),
    }
    save_file(nonexpert, str(input_dir / "adapter_model.safetensors"))
    _write_json(
        input_dir / "adapter_config.json",
        {
            "peft_type": "LORA",
            "r": rank,
            "lora_alpha": 4,
            "target_modules": [
                "q_a_proj",
                "q_b_proj",
                "kv_b_proj",
                "kv_a_proj_with_mqa",
                "o_proj",
            ],
        },
    )

    fused = {
        "layers.0.experts.gate_lora_a": torch.ones(experts, rank, hidden),
        "layers.0.experts.gate_lora_b": torch.ones(experts, intermediate, rank),
        "layers.0.experts.up_lora_a": torch.ones(experts, rank, hidden),
        "layers.0.experts.up_lora_b": torch.ones(experts, intermediate, rank),
        "layers.0.experts.down_lora_a": torch.ones(experts, rank, intermediate),
        "layers.0.experts.down_lora_b": torch.ones(experts, hidden, rank),
    }
    save_file(fused, str(input_dir / "fused_expert_lora.safetensors"))

    summary = converter.convert_kt_to_sglang_adapter(
        input_dir,
        output_dir,
        base_model_name_or_path="/models/kimi",
        expert_output_dir=expert_dir,
        nonexpert_output_dir=nonexpert_dir,
    )

    assert summary["rank"] == rank
    assert summary["target_modules"] == [
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    merged = load_file(str(output_dir / "adapter_model.safetensors"))
    assert (
        "language_model.model.layers.0.self_attn.q_a_proj.lora_A.weight" in merged
    )
    expert_key = "model.layers.0.mlp.experts.2.down_proj.lora_B.weight"
    assert tuple(merged[expert_key].shape) == (hidden, rank)

    manifest = json.loads((output_dir / "kt_composite_lora_manifest.json").read_text())
    assert manifest["format"] == "sglang_kt_composite_lora_manifest"
    assert len(manifest["expert_keys"]) == experts * 6
    assert len(manifest["nonexpert_keys"]) == 10
    assert manifest["expert_target_modules"] == ["gate_proj", "up_proj", "down_proj"]
    assert manifest["nonexpert_target_modules"] == [
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "o_proj",
    ]

    expert = load_file(str(expert_dir / "adapter_model.safetensors"))
    nonexpert = load_file(str(nonexpert_dir / "adapter_model.safetensors"))
    assert len(expert) == experts * 6
    assert len(nonexpert) == 10

    kt_ep_wrapper = _import_kt_ep_wrapper()
    for adapter_dir in (input_dir, output_dir, expert_dir):
        buffers = kt_ep_wrapper._load_kt_fused_expert_lora_buffers(
            adapter_path=str(adapter_dir),
            layer_idx=0,
            num_experts=experts,
            hidden_size=hidden,
            intermediate_size=intermediate,
        )
        assert tuple(buffers["gate_lora_a"].shape) == (experts, rank, hidden)
        assert tuple(buffers["down_lora_b"].shape) == (experts, hidden, rank)
        assert buffers["gate_lora_a"].dtype is torch.bfloat16
        assert torch.count_nonzero(buffers["grad_down_lora_b"]) == 0


def test_converted_expert_adapter_loads_into_sglang_kt_buffers(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "converted"
    input_dir.mkdir()

    rank = 2
    hidden = 4
    intermediate = 5
    experts = 3
    alpha = 7.0
    fused = {
        "layers.0.experts.gate_lora_a": torch.arange(experts * rank * hidden).reshape(
            experts, rank, hidden
        ),
        "layers.0.experts.gate_lora_b": torch.ones(experts, intermediate, rank),
        "layers.0.experts.up_lora_a": torch.ones(experts, rank, hidden) * 2,
        "layers.0.experts.up_lora_b": torch.ones(experts, intermediate, rank) * 3,
        "layers.0.experts.down_lora_a": torch.ones(experts, rank, intermediate) * 4,
        "layers.0.experts.down_lora_b": torch.ones(experts, hidden, rank) * 5,
    }
    save_file(fused, str(input_dir / "fused_expert_lora.safetensors"))

    converter.convert_kt_to_sglang_adapter(
        input_dir,
        output_dir,
        base_model_name_or_path="/models/kimi",
        lora_alpha=alpha,
    )

    kt_ep_wrapper = _import_kt_ep_wrapper()
    inferred_rank, inferred_alpha = kt_ep_wrapper._infer_kt_expert_lora_rank_alpha(
        str(output_dir),
        layer_idx=0,
    )
    assert inferred_rank == rank
    assert inferred_alpha == alpha

    buffers = kt_ep_wrapper._load_kt_fused_expert_lora_buffers(
        adapter_path=str(output_dir),
        layer_idx=0,
        num_experts=experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
    )
    assert tuple(buffers["gate_lora_a"].shape) == (experts, rank, hidden)
    assert tuple(buffers["down_lora_b"].shape) == (experts, hidden, rank)
    assert buffers["gate_lora_a"].dtype is torch.bfloat16
    assert torch.equal(
        buffers["gate_lora_a"],
        fused["layers.0.experts.gate_lora_a"].to(torch.bfloat16),
    )
    assert torch.count_nonzero(buffers["grad_gate_lora_a"]) == 0
