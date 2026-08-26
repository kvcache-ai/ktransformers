# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

import kt_kernel.sft.artifacts as artifacts_module
import kt_kernel.sft.weight_manifest as weight_manifest_module
from kt_kernel.sft.artifacts import (
    KT_ADAPTER_MANIFEST_NAME,
    KT_NON_EXPERT_MANIFEST_NAME,
    KTArtifactError,
    _cache_fingerprint,
    is_kt_routed_expert_parameter_name,
    is_kt_supported_moe_model,
    load_kt_adapter_artifacts,
    resolve_kt_pretrained_artifacts,
    save_kt_adapter_artifacts,
    validate_kt_prequantized_loading_info,
    validate_kt_pretrained_load,
    write_kt_non_expert_cache_manifest,
)
from kt_kernel.sft.backend import INT8_WEIGHT_LAYOUT
from kt_kernel.sft.config import KTConfig
from kt_kernel.sft.lora import (
    get_kt_lora_named_params,
    get_kt_rank_local_parameter_names,
    kt_adapt_peft_lora,
)
from kt_kernel.sft.weights import get_kt_expert_placeholders


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_routed_weights(root, *, schema_version=2):
    root.mkdir()
    files = []
    numa = root / "_layer_0" / "_numa_0"
    numa.mkdir(parents=True)
    sizes = {
        ("gate", "quant"): 4,
        ("up", "quant"): 4,
        ("down", "quant"): 4,
        ("gate", "scale"): 8,
        ("up", "scale"): 8,
        ("down", "scale"): 8,
    }
    for projection in ("gate", "up", "down"):
        for kind in ("quant", "scale"):
            size = sizes[(projection, kind)]
            path = numa / f"INT8_{projection}_0_{size}Byte_{kind}_.kt"
            path.write_bytes(bytes([size]) * size)
            record = {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": size,
            }
            if schema_version == 2:
                record["sha256"] = _sha256(path)
            files.append(record)
    manifest = {
        "schema_version": schema_version,
        "state": "ready",
        "expert_weight_format": "int8",
        "threadpool_count": 1,
        "expert_num": 1,
        "hidden_size": 2,
        "intermediate_size": 2,
        "layers": [
            {
                "index": 0,
                "numa_count": 1,
                "state": "ready",
                "files": sorted(files, key=lambda item: item["path"]),
                "bytes": sum(item["size_bytes"] for item in files),
            }
        ],
        "bytes": sum(item["size_bytes"] for item in files),
    }
    if schema_version == 1:
        manifest["backend"] = "AMXINT8"
        manifest_name = "kt-ephemeral-manifest.json"
    else:
        manifest["layout"] = INT8_WEIGHT_LAYOUT
        manifest_name = "kt-weight-manifest.json"
    _write_json(root / manifest_name, manifest)
    return root


def _make_pretrained_artifacts(tmp_path, *, routed_schema_version=2):
    source = tmp_path / "source"
    source.mkdir()
    _write_json(
        source / "config.json",
        {
            "model_type": "deepseek_v3",
            "num_hidden_layers": 1,
            "first_k_dense_replace": 0,
            "n_routed_experts": 1,
            "hidden_size": 2,
            "moe_intermediate_size": 2,
            "quantization_config": {"quant_method": "fp8", "weight_block_size": [128, 128]},
        },
    )
    _write_json(source / "model.safetensors.index.json", {"metadata": {}, "weight_map": {}})

    cache = tmp_path / "cache"
    cache.mkdir()
    shard = cache / "model-00001-of-00001.safetensors"
    save_file({"model.embed_tokens.weight": torch.ones(2, 2, dtype=torch.bfloat16)}, shard)
    _write_json(
        cache / "model.safetensors.index.json",
        {
            "metadata": {"total_size": 8},
            "weight_map": {"model.embed_tokens.weight": shard.name},
        },
    )
    write_kt_non_expert_cache_manifest(cache, source)
    routed = _make_routed_weights(tmp_path / "routed", schema_version=routed_schema_version)
    config = SimpleNamespace(
        kt_expert_weight_format="int8",
        kt_non_expert_weight_path=str(cache),
        kt_weight_path=str(routed),
        kt_threadpool_count=1,
        kt_lora_rank=8,
        kt_lora_alpha=16,
    )
    return source, cache, routed, config


def test_v2_producer_resolves_a_strict_pretrained_load_plan(tmp_path):
    source, cache, routed, config = _make_pretrained_artifacts(tmp_path)

    plan = resolve_kt_pretrained_artifacts(config, source)

    assert plan.weight_path == str(cache)
    assert plan.routed_weight_path == str(routed)
    assert plan.weight_keys == {"model.embed_tokens.weight"}
    assert plan.manifest["version"] == 2
    assert plan.manifest["converter"]["name"] == "kt-kernel.prepare-non-expert-cache"


def test_pretrained_load_plan_rejects_explicit_quantization_override(tmp_path):
    source, _, _, config = _make_pretrained_artifacts(tmp_path)

    with pytest.raises(KTArtifactError, match="explicit quantization_config"):
        resolve_kt_pretrained_artifacts(config, source, object())


def test_non_expert_weight_path_is_an_authoritative_kt_config_field():
    assert "kt_non_expert_weight_path" in {field.name for field in fields(KTConfig)}


def test_legacy_llamafactory_v1_cache_is_read_only_compatible(tmp_path):
    source, cache, routed, config = _make_pretrained_artifacts(tmp_path, routed_schema_version=1)
    path = cache / KT_NON_EXPERT_MANIFEST_NAME
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["version"] = 1
    manifest["converter"]["name"] = "llamafactory.prepare-kt-cache"
    manifest["converter"]["version"] = 1
    tensors = manifest["tensors"]
    manifest["fingerprint"] = _cache_fingerprint(
        1,
        manifest["source"]["fingerprint"],
        manifest["files"],
        tensors["count"],
        tensors["bytes"],
        tensors["dtypes"],
    )
    _write_json(path, manifest)

    plan = resolve_kt_pretrained_artifacts(config, source)

    assert plan.manifest["version"] == 1
    assert plan.routed_manifest["schema_version"] == 1
    assert plan.routed_manifest_path == str(routed / "kt-ephemeral-manifest.json")
    with pytest.raises(KTArtifactError, match="already exists"):
        write_kt_non_expert_cache_manifest(cache, source)


def test_non_owner_rank_keeps_structural_checks_without_large_hash_scan(tmp_path, monkeypatch):
    source, cache, routed, config = _make_pretrained_artifacts(tmp_path)

    class FakeDist:
        @staticmethod
        def all_gather_object(gathered, value):
            for index in range(len(gathered)):
                gathered[index] = value

    monkeypatch.setattr(
        artifacts_module,
        "_distributed_validation_context",
        lambda: (FakeDist(), 1, 2),
    )
    original_cache_sha256 = artifacts_module._sha256_file
    original_routed_sha256 = weight_manifest_module._sha256

    def reject_large_hash(path):
        path = Path(path)
        if path.suffix in {".safetensors", ".kt"} and (
            cache in path.parents or routed in path.parents
        ):
            raise AssertionError(f"non-owner rank hashed a large KT artifact: {path}")
        if path.suffix == ".kt":
            return original_routed_sha256(path)
        return original_cache_sha256(path)

    monkeypatch.setattr(artifacts_module, "_sha256_file", reject_large_hash)
    monkeypatch.setattr(weight_manifest_module, "_sha256", reject_large_hash)

    plan = resolve_kt_pretrained_artifacts(config, source)

    assert plan.weight_path == str(cache)
    assert plan.routed_weight_path == str(routed)


def test_distributed_validation_propagates_another_ranks_local_error():
    class FakeDist:
        @staticmethod
        def all_gather_object(gathered, value):
            gathered[0] = value
            gathered[1] = ("rank 1: KTArtifactError: corrupt routed weights", None)

    with pytest.raises(KTArtifactError, match="rank 1.*corrupt routed weights"):
        artifacts_module._synchronize_artifact_validation(
            FakeDist(),
            rank=0,
            world_size=2,
            error=None,
            signature=("same",),
        )


def test_validate_pretrained_load_attaches_provenance(tmp_path):
    source, _, _, config = _make_pretrained_artifacts(tmp_path)
    plan = resolve_kt_pretrained_artifacts(config, source)

    class Loaded:
        def __init__(self):
            self.config = SimpleNamespace(name_or_path=plan.weight_path, _name_or_path=plan.weight_path)

        @staticmethod
        def state_dict():
            return {"model.embed_tokens.weight": torch.ones(2, 2, dtype=torch.bfloat16)}

    model = Loaded()
    validate_kt_pretrained_load(
        plan,
        SimpleNamespace(
            missing_keys=["model.layers.0.mlp.experts.gate_up_proj"],
            mismatched_keys=[],
            conversion_errors={},
            unexpected_keys=[],
            error_msgs=[],
        ),
        model,
    )

    assert model.config.name_or_path == str(source)
    assert model._kt_pretrained_load_plan is plan


def test_prequantized_key_ownership_and_loading_diagnostics_are_public():
    assert is_kt_routed_expert_parameter_name("model.layers.3.mlp.experts.gate_up_proj")
    assert is_kt_routed_expert_parameter_name("model.layers.3.mlp.experts.4.gate_proj.weight")
    assert not is_kt_routed_expert_parameter_name("model.layers.3.mlp.shared_experts.gate_proj.weight")
    model = SimpleNamespace(
        config=SimpleNamespace(
            architectures=["DeepseekV3ForCausalLM"],
            model_type="deepseek_v3",
            num_hidden_layers=61,
        )
    )
    assert is_kt_supported_moe_model(model)
    config = SimpleNamespace(kt_expert_weight_format="fp8", kt_skip_expert_loading=True)
    validate_kt_prequantized_loading_info(
        config,
        SimpleNamespace(
            missing_keys=["model.layers.3.mlp.experts.gate_up_proj"],
            mismatched_keys=[],
            conversion_errors={},
            unexpected_keys=["model.layers.61.mtp.weight"],
            error_msgs=[],
        ),
        model,
    )
    with pytest.raises(KTArtifactError, match="exact non-expert model match"):
        validate_kt_prequantized_loading_info(
            config,
            SimpleNamespace(
                missing_keys=["model.norm.weight"],
                mismatched_keys=[],
                conversion_errors={},
                unexpected_keys=[],
                error_msgs=[],
            ),
            model,
        )


class _Backend:
    def __init__(self):
        self.initialized = None

    def init_lora_weights(self, **buffers):
        self.initialized = buffers


def _fused_model(*, expert_weight_format="bf16"):
    model = torch.nn.Module()
    model.config = SimpleNamespace(name_or_path="/models/base", _name_or_path="/models/base")
    model._kt_expert_weight_format = expert_weight_format
    wrapper = SimpleNamespace(
        layer_idx=0,
        moe_config=SimpleNamespace(expert_num=1, intermediate_size=2),
        hidden_size=2,
        _experts_attr="experts",
        experts=torch.nn.ModuleList(),
        _use_fused_expert_lora=True,
        _fused_experts=False,
        _lora_rank=1,
        _lora_alpha=2.0,
        _kt_expert_weight_format=expert_weight_format,
        _uses_authoritative_optimizer_grads=False,
        _full_weight_grad=False,
        lora_experts=None,
        wrapper=_Backend(),
    )
    model._kt_wrappers = [wrapper]
    return model, wrapper


def _write_standard_adapter(output):
    output.mkdir()
    _write_json(output / "adapter_config.json", {"r": 1})
    save_file({"router.lora_A.weight": torch.ones(1, 2)}, output / "adapter_model.safetensors")


@pytest.mark.parametrize("expert_weight_format", ["fp8", "int8"])
def test_quantized_adapter_manifest_uses_explicit_runtime_provenance(
    tmp_path, monkeypatch, expert_weight_format
):
    monkeypatch.setattr("kt_kernel.sft.lora._distributed_rank_world_size", lambda: (0, 1))
    model, _ = _fused_model(expert_weight_format=expert_weight_format)
    kt_adapt_peft_lora(model)
    output = tmp_path / expert_weight_format
    _write_standard_adapter(output)

    saved = save_kt_adapter_artifacts(model, output)
    loaded = load_kt_adapter_artifacts(model, output)

    assert saved.payload["expert_weight_format"] == expert_weight_format
    assert "non_expert_cache" not in saved.payload
    assert "int8_experts" not in saved.payload
    assert loaded.payload == saved.payload


def test_adapter_load_rejects_expert_weight_format_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr("kt_kernel.sft.lora._distributed_rank_world_size", lambda: (0, 1))
    source, _ = _fused_model(expert_weight_format="fp8")
    kt_adapt_peft_lora(source)
    output = tmp_path / "adapter"
    _write_standard_adapter(output)
    save_kt_adapter_artifacts(source, output)

    incompatible, _ = _fused_model(expert_weight_format="int8")
    with pytest.raises(KTArtifactError, match="expert_weight_format does not match"):
        load_kt_adapter_artifacts(incompatible, output)


def test_adapter_save_rejects_conflicting_owner_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr("kt_kernel.sft.lora._distributed_rank_world_size", lambda: (0, 1))
    model, wrapper = _fused_model(expert_weight_format="fp8")
    wrapper._kt_expert_weight_format = "int8"
    kt_adapt_peft_lora(model)
    output = tmp_path / "adapter"
    _write_standard_adapter(output)

    with pytest.raises(KTArtifactError, match="conflicting KT expert weight provenance"):
        save_kt_adapter_artifacts(model, output)


def test_adaptation_is_idempotent_and_publishes_stable_parameter_names(monkeypatch):
    model, wrapper = _fused_model()
    monkeypatch.setattr("kt_kernel.sft.lora._distributed_rank_world_size", lambda: (0, 1))

    first = kt_adapt_peft_lora(model)
    pointers = [parameter.data_ptr() for _, parameter in first.named_optimizer_parameters]
    second = kt_adapt_peft_lora(model)

    assert first.adapted_layers == 1
    assert second.adapted_layers == 0
    assert second.already_adapted_layers == 1
    assert pointers == [parameter.data_ptr() for _, parameter in second.named_optimizer_parameters]
    assert [name for name, _ in get_kt_lora_named_params(model)] == [
        f"kt.layers.0.experts.fused_lora.{name}"
        for name in (
            "gate_lora_a",
            "gate_lora_b",
            "up_lora_a",
            "up_lora_b",
            "down_lora_a",
            "down_lora_b",
        )
    ]
    assert wrapper.wrapper.initialized is not None


def test_rank_local_names_survive_meta_and_parameter_replacement():
    model = torch.nn.Module()
    model.block = torch.nn.Module()
    model.block.moe = torch.nn.Module()
    model.block.moe._is_kt_moe_wrapper = True
    model.block.moe._experts_attr = "experts"
    model.block.moe.experts = torch.nn.Linear(2, 2, bias=False)
    model.block.moe.experts.weight._kt_zero_storage = True
    expected = ("block.moe.experts.weight",)

    assert get_kt_rank_local_parameter_names(model) == expected
    model.to("meta")
    assert get_kt_rank_local_parameter_names(model) == expected

    replacement = torch.nn.Parameter(torch.empty((2, 2), device="meta"), requires_grad=False)
    model.block.moe.experts.weight = replacement

    assert not getattr(replacement, "_kt_zero_storage", False)
    assert get_kt_rank_local_parameter_names(model) == expected
    assert get_kt_expert_placeholders(model) == {expected[0]: replacement}


def test_combined_adapter_save_load_is_manifest_last_and_fail_closed(tmp_path, monkeypatch):
    model, wrapper = _fused_model()
    monkeypatch.setattr("kt_kernel.sft.lora._distributed_rank_world_size", lambda: (0, 1))
    kt_adapt_peft_lora(model)
    output = tmp_path / "adapter"
    output.mkdir()
    _write_json(output / "adapter_config.json", {"r": 1})
    save_file({"router.lora_A.weight": torch.ones(1, 2)}, output / "adapter_model.safetensors")

    saved = save_kt_adapter_artifacts(model, output)
    expected = [parameter.detach().clone() for parameter in wrapper._fused_expert_lora_params]
    for parameter in wrapper._fused_expert_lora_params:
        parameter.data.zero_()
    loaded = load_kt_adapter_artifacts(model, output)

    assert saved.payload["status"] == "ready"
    assert saved.payload["lora"] == {"rank": 1, "alpha": 2.0}
    assert loaded.payload == saved.payload
    assert (output / KT_ADAPTER_MANIFEST_NAME).is_file()
    for parameter, tensor in zip(wrapper._fused_expert_lora_params, expected):
        assert torch.equal(parameter, tensor)

    manifest_path = output / KT_ADAPTER_MANIFEST_NAME
    manifest_bytes = manifest_path.read_bytes()
    manifest_path.unlink()
    with pytest.raises(KTArtifactError, match="fused adapter is missing"):
        load_kt_adapter_artifacts(model, output)
    manifest_path.write_bytes(manifest_bytes)

    non_owner_model, non_owner_wrapper = _fused_model()
    non_owner_wrapper._uses_authoritative_optimizer_grads = True
    non_owner_wrapper._fused_expert_lora_params = []
    non_owner_wrapper.wrapper = None
    non_owner_manifest = load_kt_adapter_artifacts(non_owner_model, output)
    assert non_owner_manifest.payload == saved.payload
    assert non_owner_wrapper._fused_expert_lora_params == []

    before = [parameter.detach().clone() for parameter in wrapper._fused_expert_lora_params]
    wrapper._lora_alpha = 4.0
    with pytest.raises(KTArtifactError, match="lora does not match"):
        load_kt_adapter_artifacts(model, output)
    for parameter, tensor in zip(wrapper._fused_expert_lora_params, before):
        assert torch.equal(parameter, tensor)
    wrapper._lora_alpha = 2.0
    standard_path = output / "adapter_model.safetensors"
    original_standard = standard_path.read_bytes()
    with standard_path.open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(KTArtifactError, match="size mismatch"):
        load_kt_adapter_artifacts(model, output)
    standard_path.write_bytes(original_standard)
    for parameter, tensor in zip(wrapper._fused_expert_lora_params, before):
        assert torch.equal(parameter, tensor)
    with (output / "fused_expert_lora.safetensors").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(KTArtifactError, match="size mismatch"):
        load_kt_adapter_artifacts(model, output)
    for parameter, tensor in zip(wrapper._fused_expert_lora_params, before):
        assert torch.equal(parameter, tensor)


def test_adapter_overwrite_invalidates_old_ready_manifest_before_replacement(tmp_path, monkeypatch):
    model, _ = _fused_model()
    monkeypatch.setattr("kt_kernel.sft.lora._distributed_rank_world_size", lambda: (0, 1))
    kt_adapt_peft_lora(model)
    output = tmp_path / "adapter"
    output.mkdir()
    _write_json(output / "adapter_config.json", {"r": 1})
    save_file({"router.lora_A.weight": torch.ones(1, 2)}, output / "adapter_model.safetensors")
    save_kt_adapter_artifacts(model, output)
    manifest_path = output / KT_ADAPTER_MANIFEST_NAME
    assert manifest_path.is_file()

    original_replace = os.replace
    observed = {}

    def fail_first_bundle_replacement(source, destination):
        destination = Path(destination)
        if destination.parent == output and destination.name != KT_ADAPTER_MANIFEST_NAME:
            observed["ready_visible"] = manifest_path.exists()
            raise OSError("injected replacement failure")
        return original_replace(source, destination)

    monkeypatch.setattr(artifacts_module.os, "replace", fail_first_bundle_replacement)

    with pytest.raises(OSError, match="injected replacement failure"):
        save_kt_adapter_artifacts(model, output)

    assert observed == {"ready_visible": False}
    assert not manifest_path.exists()


def test_adapter_load_auto_adapts_owner_and_remains_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr("kt_kernel.sft.lora._distributed_rank_world_size", lambda: (0, 1))
    source_model, source_wrapper = _fused_model()
    kt_adapt_peft_lora(source_model)
    output = tmp_path / "adapter"
    output.mkdir()
    _write_json(output / "adapter_config.json", {"r": 1})
    save_file({"router.lora_A.weight": torch.ones(1, 2)}, output / "adapter_model.safetensors")
    saved_manifest = save_kt_adapter_artifacts(source_model, output)
    expected = [parameter.detach().clone() for parameter in source_wrapper._fused_expert_lora_params]

    restored_model, restored_wrapper = _fused_model()
    loaded_manifest = load_kt_adapter_artifacts(restored_model, output)

    assert loaded_manifest.payload == saved_manifest.payload
    assert restored_wrapper._kt_peft_lora_adapted is True
    assert restored_wrapper.wrapper.initialized is not None
    assert len(restored_wrapper._fused_expert_lora_params) == 6
    for parameter, tensor in zip(restored_wrapper._fused_expert_lora_params, expected):
        assert torch.equal(parameter, tensor)

    parameter_ids = tuple(id(parameter) for parameter in restored_wrapper._fused_expert_lora_params)
    load_kt_adapter_artifacts(restored_model, output)

    assert parameter_ids == tuple(id(parameter) for parameter in restored_wrapper._fused_expert_lora_params)


def test_legacy_int8_adapter_manifest_without_explicit_format_remains_loadable(tmp_path, monkeypatch):
    source, _, _, config = _make_pretrained_artifacts(tmp_path)
    config.kt_lora_rank = 1
    config.kt_lora_alpha = 2.0
    plan = resolve_kt_pretrained_artifacts(config, source)
    model, wrapper = _fused_model()
    del model._kt_expert_weight_format
    del wrapper._kt_expert_weight_format
    model.config.name_or_path = str(source)
    model.config._name_or_path = str(source)
    model._kt_pretrained_load_plan = plan
    monkeypatch.setattr("kt_kernel.sft.lora._distributed_rank_world_size", lambda: (0, 1))
    kt_adapt_peft_lora(model)
    output = tmp_path / "legacy-adapter"
    output.mkdir()
    _write_json(output / "adapter_config.json", {"r": 1})
    save_file({"router.lora_A.weight": torch.ones(1, 2)}, output / "adapter_model.safetensors")
    save_kt_adapter_artifacts(model, output)
    manifest_path = output / KT_ADAPTER_MANIFEST_NAME
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload.pop("expert_weight_format")
    _write_json(manifest_path, payload)

    loaded = load_kt_adapter_artifacts(model, output)

    assert loaded.payload.get("expert_weight_format") is None
