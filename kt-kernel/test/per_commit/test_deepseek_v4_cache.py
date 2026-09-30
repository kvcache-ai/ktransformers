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


def test_first_load_prepares_cache_and_reuses_without_conversion(native_cache, tmp_path, monkeypatch):
    import kt_kernel.sft.deepseek_v4 as v4

    source, _ = native_cache
    cache = tmp_path / "automatic"
    cfg = SimpleNamespace(kt_weight_path=str(source), kt_non_expert_weight_path=str(cache))
    first = resolve_native_load_plan(cfg, str(source))
    before = {p.name: p.stat().st_mtime_ns for p in cache.iterdir()}

    def unexpected_conversion(*args):
        pytest.fail("warm load must not regenerate the cache")

    monkeypatch.setattr(v4, "_convert_non_expert_cache", unexpected_conversion)
    second = resolve_native_load_plan(cfg, str(source))
    assert first.manifest == second.manifest
    assert before == {p.name: p.stat().st_mtime_ns for p in cache.iterdir()}


def test_concurrent_preparation_has_one_writer(native_cache, tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import kt_kernel.sft.deepseek_v4 as v4

    source, _ = native_cache
    cache = tmp_path / "concurrent"
    convert = v4._convert_non_expert_cache
    calls = []

    def counted(source, staging):
        calls.append(staging)
        assert not cache.exists(), "unvalidated weights must not be published"
        return convert(source, staging)

    monkeypatch.setattr(v4, "_convert_non_expert_cache", counted)
    with ThreadPoolExecutor(max_workers=2) as executor:
        jobs = [executor.submit(prepare_non_expert_cache, str(source), str(cache)) for _ in range(2)]
        manifests = [job.result(timeout=30) for job in jobs]
    assert len(calls) == 1
    assert manifests[0] == manifests[1]


def test_interrupted_owned_preparation_is_retried(native_cache, tmp_path, monkeypatch):
    import kt_kernel.sft.deepseek_v4 as v4

    source, _ = native_cache
    cache = tmp_path / "retry"
    convert = v4._convert_non_expert_cache

    def interrupted(source, staging):
        (staging / "incomplete.safetensors").write_bytes(b"partial")
        raise KeyboardInterrupt("simulated interrupted converter")

    monkeypatch.setattr(v4, "_convert_non_expert_cache", interrupted)
    with pytest.raises(KeyboardInterrupt):
        prepare_non_expert_cache(str(source), str(cache))
    assert not cache.exists()
    monkeypatch.setattr(v4, "_convert_non_expert_cache", convert)
    assert prepare_non_expert_cache(str(source), str(cache))["status"] == "ready"
    assert not (cache / "incomplete.safetensors").exists()
    assert not (tmp_path / ".retry.kt-v4-cache.partial").exists()


def test_cache_failure_does_not_publish_or_delete_unrelated_data(native_cache, tmp_path, monkeypatch):
    import kt_kernel.sft.deepseek_v4 as v4

    source, _ = native_cache
    cache = tmp_path / "full-disk"
    monkeypatch.setattr(v4.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(KTArtifactError, match="Insufficient space"):
        prepare_non_expert_cache(str(source), str(cache))
    assert not cache.exists()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    note = unrelated / "user.txt"
    note.write_text("keep me")
    with pytest.raises(KTArtifactError):
        prepare_non_expert_cache(str(source), str(unrelated))
    assert note.read_text() == "keep me"


def test_changed_source_and_symlink_cache_are_rejected(native_cache, tmp_path):
    source, cache = native_cache
    link = tmp_path / "linked-cache"
    link.symlink_to(cache, target_is_directory=True)
    with pytest.raises(KTArtifactError, match="real directory"):
        prepare_non_expert_cache(str(source), str(link))
    config_path = source / "config.json"
    config = json.loads(config_path.read_text())
    config["initializer_range"] = 0.01
    config_path.write_text(json.dumps(config))
    with pytest.raises(KTArtifactError, match="changed after cache"):
        prepare_non_expert_cache(str(source), str(cache))


def test_config_loading_allows_an_unprepared_cache(native_cache, tmp_path):
    from kt_kernel.sft.artifacts import should_disable_kt_source_quantizer

    cfg = SimpleNamespace(kt_expert_weight_format="mxfp4", kt_non_expert_weight_path=str(tmp_path / "future"))
    assert should_disable_kt_source_quantizer(cfg, SimpleNamespace(model_type="deepseek_v4"))
    assert not (tmp_path / "future").exists()


@pytest.fixture
def adapter_export(native_cache, tmp_path, monkeypatch):
    source, cache = native_cache
    from kt_kernel.sft import export_dsv4_sglang_adapter as module
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"r": 8, "lora_alpha": 16, "lora_dropout": 0.0}))
    prefix = "base_model.model.model.layers.0.self_attn.q_a_proj.lora_"
    save_file(
        {
            prefix + "A.weight": torch.full((8, 32), 0.25),
            prefix + "B.weight": torch.full((32, 8), 0.125),
        },
        adapter / "adapter_model.safetensors",
    )
    fused = {}
    for proj in ("gate", "up", "down"):
        fused[f"layers.0.experts.{proj}_lora_a"] = torch.ones(1, 8, 32)
        fused[f"layers.0.experts.{proj}_lora_b"] = torch.ones(1, 32, 8)
    save_file(fused, adapter / "fused_expert_lora.safetensors")
    manifest = {
        "status": "ready",
        "expert_weight_format": "mxfp4",
        "lora": {"rank": 8, "alpha": 16},
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


def test_static_baseline_can_match_the_expert_forward_kernel(adapter_export):
    module, source, cache, adapter, output, _ = adapter_export
    report = module.export(
        str(source),
        str(cache),
        str(adapter),
        str(output),
        component="base",
        match_expert_kernel=True,
    )
    assert report["zero_expert_adapter_control"] and not report["expert_effect_enabled"]
    assert (
        report["expert_exported_tensor_count"] == 6
        and report["standard_pairs_consumed"] == []
    )
    tensors = load_file(output / "experts" / "adapter_model.safetensors")
    for name, tensor in tensors.items():
        assert torch.all(tensor == (0 if ".lora_B." in name else 1))
