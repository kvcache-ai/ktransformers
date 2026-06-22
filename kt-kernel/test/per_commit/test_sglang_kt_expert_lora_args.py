import hashlib
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "sglang" / "python"))

from sglang.srt import server_args
from sglang.srt.server_args import ServerArgs


def _make_server_args(**overrides):
    args = object.__new__(ServerArgs)
    defaults = {
        "kt_lora_path": None,
        "kt_expert_lora_path": "/adapter",
        "disable_cuda_graph": False,
        "kt_weight_path": "/model",
        "kt_method": "RAWINT4",
        "kt_num_gpu_experts": None,
        "kt_gpu_experts_ratio": None,
        "kt_gpu_prefill_token_threshold": None,
    }
    defaults.update(overrides)
    for key, value in defaults.items():
        setattr(args, key, value)
    return args


def test_kt_expert_lora_defaults_to_cpu_only_and_disables_cuda_graph():
    args = _make_server_args()

    ServerArgs._handle_kt_expert_lora_args(args)

    assert args.kt_num_gpu_experts == 0
    assert args.disable_cuda_graph is True


def test_kt_lora_path_aliases_expert_lora_path():
    args = _make_server_args(
        kt_lora_path="/adapter",
        kt_expert_lora_path=None,
    )

    ServerArgs._handle_kt_expert_lora_args(args)

    assert args.kt_expert_lora_path == "/adapter"


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"kt_weight_path": None}, "--kt-weight-path"),
        ({"kt_method": None}, "--kt-method"),
        ({"kt_num_gpu_experts": 1}, "--kt-num-gpu-experts 0"),
        ({"kt_gpu_experts_ratio": 0.1}, "no GPU expert ratio"),
        ({"kt_gpu_prefill_token_threshold": 1024}, "full-GPU prefill"),
        (
            {"kt_lora_path": "/adapter-a", "kt_expert_lora_path": "/adapter-b"},
            "cannot point to different adapters",
        ),
    ],
)
def test_kt_expert_lora_rejects_incompatible_args(overrides, match):
    args = _make_server_args(**overrides)

    with pytest.raises(ValueError, match=match):
        ServerArgs._handle_kt_expert_lora_args(args)


def _make_lora_server_args(**overrides):
    args = _make_server_args(kt_expert_lora_path=None)
    defaults = {
        "lora_paths": ["kimi=/merged"],
        "enable_lora": None,
        "enable_lora_overlap_loading": None,
        "max_loaded_loras": None,
        "max_loras_per_batch": 8,
        "speculative_algorithm": None,
        "lora_target_modules": None,
        "lora_backend": "csgmv",
    }
    defaults.update(overrides)
    for key, value in defaults.items():
        setattr(args, key, value)
    return args


def test_composite_lora_split_reuses_kt_expert_lora_validation(monkeypatch):
    monkeypatch.setattr(
        server_args,
        "_prepare_kt_composite_lora_adapter",
        lambda path: ("/cache/expert", "/cache/nonexpert"),
    )
    args = _make_lora_server_args()

    ServerArgs.check_lora_server_args(args)

    assert args.enable_lora is True
    assert args.kt_expert_lora_path == "/cache/expert"
    assert args.lora_paths[0].lora_path == "/cache/nonexpert"
    assert args.kt_num_gpu_experts == 0
    assert args.disable_cuda_graph is True


def test_composite_lora_split_rejects_missing_kt_weight_path(monkeypatch):
    monkeypatch.setattr(
        server_args,
        "_prepare_kt_composite_lora_adapter",
        lambda path: ("/cache/expert", "/cache/nonexpert"),
    )
    args = _make_lora_server_args(kt_weight_path=None)

    with pytest.raises(ValueError, match="--kt-weight-path"):
        ServerArgs.check_lora_server_args(args)


def test_prepare_composite_lora_manifest_hits_cache_without_reading_weights(
    tmp_path: Path,
    monkeypatch,
):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    weight_path = adapter_dir / "adapter_model.safetensors"
    config_path = adapter_dir / "adapter_config.json"
    weight_path.write_bytes(b"placeholder-not-a-safetensors-file")
    config_path.write_text('{"r": 2, "lora_alpha": 4}\n', encoding="utf-8")
    (adapter_dir / "kt_composite_lora_manifest.json").write_text(
        json.dumps(
            {
                "format": "sglang_kt_composite_lora_manifest",
                "version": 1,
                "expert_keys": [
                    "model.layers.0.mlp.experts.0.gate_proj.lora_A.weight"
                ],
                "nonexpert_keys": [
                    "language_model.model.layers.0.self_attn.q_a_proj.lora_A.weight"
                ],
            }
        ),
        encoding="utf-8",
    )

    cache_root = tmp_path / "cache"
    monkeypatch.setenv("SGLANG_KT_LORA_CACHE_DIR", str(cache_root))

    digest = hashlib.sha256()
    digest.update(server_args.KT_COMPOSITE_LORA_CACHE_VERSION.encode())
    digest.update(str(adapter_dir.resolve()).encode())
    for path in (weight_path, config_path):
        stat = path.stat()
        digest.update(str(stat.st_mtime_ns).encode())
        digest.update(str(stat.st_size).encode())

    cache_dir = cache_root / digest.hexdigest()[:16]
    expert_dir = cache_dir / "expert"
    nonexpert_dir = cache_dir / "nonexpert"
    for path in (expert_dir, nonexpert_dir):
        path.mkdir(parents=True)
        (path / "adapter_model.safetensors").write_bytes(b"cached")
        (path / "adapter_config.json").write_text("{}\n", encoding="utf-8")

    assert server_args._prepare_kt_composite_lora_adapter(str(adapter_dir)) == (
        str(expert_dir),
        str(nonexpert_dir),
    )
