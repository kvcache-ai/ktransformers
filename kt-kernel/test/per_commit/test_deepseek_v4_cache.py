"""Native V4 selective-loading contracts; no model download is required."""

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from kt_kernel.sft.artifacts import KTArtifactError, _sha256_file
from kt_kernel.sft.deepseek_v4 import (
    canonical_non_expert_key,
    dequantize_fp8_blocks,
    prepare_non_expert_cache,
    resolve_native_load_plan,
)


def test_fp8_decode_partial_blocks():
    weight = torch.full((129, 130), 2.0).to(torch.float8_e4m3fn)
    scale = torch.tensor([[0.5, 1.0], [2.0, 4.0]]).to(torch.float8_e8m0fnu)
    result = dequantize_fp8_blocks(weight, scale)
    assert result.dtype == torch.bfloat16
    assert result[0, 0] == 1 and result[0, 129] == 2
    assert result[128, 0] == 4 and result[128, 129] == 8
    with pytest.raises(KTArtifactError, match="shape"):
        dequantize_fp8_blocks(weight, scale[:1])


def test_native_key_mapping():
    assert canonical_non_expert_key("layers.2.attn.indexer.compressor.wkv.weight") == (
        "model.layers.2.self_attn.compressor.indexer.kv_proj.weight"
    )
    assert (
        canonical_non_expert_key("layers.2.hc_attn_scale")
        == "model.layers.2.attn_hc.scale"
    )
    assert canonical_non_expert_key("norm.weight") == "model.norm.weight"
    with pytest.raises(KTArtifactError):
        canonical_non_expert_key("mtp.0.norm.weight")


@pytest.fixture
def native_cache(tmp_path):
    source, cache = tmp_path / "source", tmp_path / "cache"
    source.mkdir()
    config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 1,
        "n_routed_experts": 1,
        "hidden_size": 32,
        "moe_intermediate_size": 32,
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
            "scale_fmt": "ue8m0",
        },
    }
    (source / "config.json").write_text(json.dumps(config))
    tensors = {
        "layers.0.attn.wq_a.weight": torch.ones(32, 32).to(torch.float8_e4m3fn),
        "layers.0.attn.wq_a.scale": torch.tensor([[0.5]]).to(torch.float8_e8m0fnu),
        "layers.0.hc_attn_scale": torch.ones(3),
        "layers.0.ffn.gate.tid2eid": torch.zeros(8, 1, dtype=torch.int64),
        "mtp.0.norm.weight": torch.ones(32, dtype=torch.bfloat16),
    }
    for proj in ("w1", "w2", "w3"):
        tensors[f"layers.0.ffn.experts.0.{proj}.weight"] = torch.zeros(
            32, 16, dtype=torch.int8
        )
        tensors[f"layers.0.ffn.experts.0.{proj}.scale"] = torch.ones(32, 1).to(
            torch.float8_e8m0fnu
        )
    save_file(tensors, source / "source.safetensors")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: "source.safetensors" for key in tensors}})
    )
    prepare_non_expert_cache(str(source), str(cache))
    return source, cache


def test_cache_roundtrip_and_tampering(native_cache):
    source, cache = native_cache
    cfg = SimpleNamespace(
        kt_weight_path=str(source), kt_non_expert_weight_path=str(cache)
    )
    plan = resolve_native_load_plan(cfg, str(source))
    assert len(plan.weight_keys) == 3
    data = load_file(plan.checkpoint_files[0])
    assert data["model.layers.0.attn_hc.scale"].dtype == torch.float32
    assert data["model.layers.0.mlp.gate.tid2eid"].dtype == torch.int64
    assert torch.all(data["model.layers.0.self_attn.q_a_proj.weight"] == 0.5)
    with open(plan.checkpoint_files[0], "r+b") as handle:
        handle.seek(-1, 2)
        handle.write(b"\x00")
    with pytest.raises(KTArtifactError, match="hash mismatch"):
        resolve_native_load_plan(cfg, str(source))


@pytest.fixture
def adapter_export(native_cache, tmp_path, monkeypatch):
    source, cache = native_cache
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "v4_export", scripts / "export_dsv4_sglang_adapter.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4}))
    prefix = "base_model.model.model.layers.0.self_attn.q_a_proj.lora_"
    save_file(
        {
            prefix + "A.weight": torch.full((2, 32), 0.25),
            prefix + "B.weight": torch.full((32, 2), 0.5),
        },
        adapter / "adapter_model.safetensors",
    )
    fused = {}
    for proj in ("gate", "up", "down"):
        fused[f"layers.0.experts.{proj}_lora_a"] = torch.ones(1, 2, 32)
        fused[f"layers.0.experts.{proj}_lora_b"] = torch.ones(1, 32, 2)
    save_file(fused, adapter / "fused_expert_lora.safetensors")
    manifest = {
        "status": "ready",
        "expert_weight_format": "mxfp4",
        "lora": {"rank": 2, "alpha": 4},
        "base": {
            "fingerprint": module.inspect_native_checkpoint(source)["fingerprint"]
        },
    }

    def seal():
        manifest["artifacts"] = {
            p.name: {"size": p.stat().st_size, "sha256": _sha256_file(p)}
            for p in adapter.iterdir()
            if p.name != "kt_adapter_manifest.json"
        }
        (adapter / "kt_adapter_manifest.json").write_text(json.dumps(manifest))

    seal()
    return module, source, cache, adapter, tmp_path / "export", seal


@pytest.mark.parametrize("component", ["all", "base", "experts", "nonexperts"])
def test_static_export_components(adapter_export, component):
    module, source, cache, adapter, output, _ = adapter_export
    report = module.export(
        str(source), str(cache), str(adapter), str(output), component
    )
    tensors = load_file(output / "model" / "non-experts-000.safetensors")
    merged = component in {"all", "nonexperts"}
    assert torch.all(tensors["layers.0.attn.wq_a.weight"] == (1.0 if merged else 0.5))
    assert report["standard_pairs_consumed"] == (
        ["model.layers.0.self_attn.q_a_proj"] if merged else []
    )
    assert report["expert_exported_tensor_count"] == (
        6 if component in {"all", "experts"} else 0
    )
    assert "quantization_config" not in json.loads(
        (output / "model" / "config.json").read_text()
    )
    assert all(".experts." not in key for key in tensors)


@pytest.mark.parametrize(
    "damage", ["missing_pair", "wrong_layer", "bad_shape", "nan", "unsealed"]
)
def test_static_export_rejects_damaged_adapters(adapter_export, damage):
    module, source, cache, adapter, output, seal = adapter_export
    path = adapter / "fused_expert_lora.safetensors"
    tensors = load_file(path)
    key = next(iter(tensors))
    if damage in {"missing_pair", "unsealed"}:
        tensors.pop(key)
    elif damage == "wrong_layer":
        tensors = {
            key.replace("layers.0", "layers.1"): value for key, value in tensors.items()
        }
    elif damage == "bad_shape":
        tensors[key] = tensors[key][..., :1].contiguous()
    else:
        tensors[key].flatten()[0] = float("nan")
    save_file(tensors, path)
    if damage != "unsealed":
        seal()
    with pytest.raises(KTArtifactError):
        module.export(str(source), str(cache), str(adapter), str(output))
    assert not (output / "deployment_manifest.json").exists()


def test_static_export_rejects_unknown_standard_module(adapter_export):
    module, source, cache, adapter, output, seal = adapter_export
    path = adapter / "adapter_model.safetensors"
    tensors = {
        key.replace("q_a_proj", "o_a_proj"): value
        for key, value in load_file(path).items()
    }
    save_file(tensors, path)
    seal()
    with pytest.raises(KTArtifactError, match="unsupported non-expert"):
        module.export(str(source), str(cache), str(adapter), str(output))
