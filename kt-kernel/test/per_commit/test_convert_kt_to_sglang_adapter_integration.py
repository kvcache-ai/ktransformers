import importlib.util
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=60, suite="default")

KT_ADAPTER_ENV = "KT_LORA_ADAPTER_DIR"
KT_BASE_MODEL_ENV = "KT_LORA_BASE_MODEL"
KT_ALPHA_ENV = "KT_LORA_ALPHA"
KT_LARGE_ADAPTER_ENV = "KT_LORA_LARGE_ADAPTER_DIR"

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "convert_kt_to_sglang_adapter.py"
)
SPEC = importlib.util.spec_from_file_location("convert_kt_to_sglang_adapter", SCRIPT_PATH)
converter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(converter)

SGLANG_EXPERT_KEY_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.lora_[AB]\.weight$"
)


def _required_adapter_dir(env_name: str) -> Path:
    value = os.environ.get(env_name)
    if not value:
        pytest.skip(f"Set {env_name} to run real adapter integration tests.")

    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        pytest.fail(f"{env_name} is not a directory: {path}")
    if not (path / converter.FUSED_EXPERT_LORA_FILE).is_file():
        pytest.fail(f"{env_name} must contain {converter.FUSED_EXPERT_LORA_FILE}: {path}")
    return path


def _base_model_name_or_path() -> str:
    value = os.environ.get(KT_BASE_MODEL_ENV)
    if not value:
        pytest.fail(f"Set {KT_BASE_MODEL_ENV} before running real adapter integration tests.")
    return value


def _optional_lora_alpha() -> float | None:
    value = os.environ.get(KT_ALPHA_ENV)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        pytest.fail(f"{KT_ALPHA_ENV} must be numeric, got: {value!r}")


def _lora_alpha_for_input(input_dir: Path) -> float | None:
    alpha = _optional_lora_alpha()
    if (input_dir / converter.ADAPTER_CONFIG_FILE).exists():
        return alpha
    if alpha is None:
        pytest.fail(
            f"{input_dir} has no {converter.ADAPTER_CONFIG_FILE}; set {KT_ALPHA_ENV}."
        )
    return alpha


def _convert_real_adapter(input_dir: Path, tmp_path: Path, output_name: str = "output") -> tuple[Path, dict]:
    output_dir = tmp_path / output_name
    summary = converter.convert_kt_to_sglang_adapter(
        input_dir,
        output_dir,
        base_model_name_or_path=_base_model_name_or_path(),
        lora_alpha=_lora_alpha_for_input(input_dir),
    )
    return output_dir, summary


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _link_or_copy(source: Path, dest: Path) -> None:
    try:
        os.symlink(source, dest)
    except OSError:
        shutil.copy2(source, dest)


def _assert_config_shape(output_dir: Path, summary: dict) -> dict:
    config = _load_json(output_dir / converter.ADAPTER_CONFIG_FILE)
    assert config["peft_type"] == "LORA"
    assert config["r"] == summary["rank"]
    assert config["lora_alpha"] == summary["lora_alpha"]
    assert config["base_model_name_or_path"] == _base_model_name_or_path()
    assert {"gate_proj", "up_proj", "down_proj"}.issubset(config["target_modules"])
    return config


def _assert_fused_tensors_preserved(input_dir: Path, output_dir: Path) -> int:
    fused_tensors = load_file(str(input_dir / converter.FUSED_EXPERT_LORA_FILE))
    output_tensors = load_file(str(output_dir / converter.ADAPTER_MODEL_FILE))

    checked = 0
    for input_key, input_tensor in sorted(fused_tensors.items()):
        match = converter.KT_FUSED_KEY_RE.match(input_key)
        assert match is not None, input_key
        layer_idx, kt_name = match.groups()
        assert kt_name in converter.KT_NAME_MAP, input_key
        assert input_tensor.dim() == 3, input_key

        proj_name, lora_name, _rank_dim = converter.KT_NAME_MAP[kt_name]
        for expert_idx in range(input_tensor.shape[0]):
            output_key = (
                f"model.layers.{layer_idx}.mlp.experts.{expert_idx}."
                f"{proj_name}.{lora_name}.weight"
            )
            assert SGLANG_EXPERT_KEY_RE.match(output_key), output_key
            assert output_key in output_tensors
            assert output_tensors[output_key].shape == input_tensor[expert_idx].shape
            assert output_tensors[output_key].dtype == input_tensor.dtype
            assert torch.equal(output_tensors[output_key], input_tensor[expert_idx].cpu())
            checked += 1

    assert checked > 0
    return checked


@pytest.mark.requires_model
def test_real_adapter_conversion_preserves_fused_tensors_and_config(tmp_path):
    input_dir = _required_adapter_dir(KT_ADAPTER_ENV)

    output_dir, summary = _convert_real_adapter(input_dir, tmp_path)

    checked = _assert_fused_tensors_preserved(input_dir, output_dir)
    config = _assert_config_shape(output_dir, summary)
    output_tensors = load_file(str(output_dir / converter.ADAPTER_MODEL_FILE))

    assert summary["tensor_count"] == len(output_tensors)
    assert checked <= summary["tensor_count"]
    assert config["target_modules"] == summary["target_modules"]


@pytest.mark.requires_model
def test_real_adapter_directory_merges_existing_adapter_model(tmp_path):
    input_dir = _required_adapter_dir(KT_ADAPTER_ENV)
    existing_adapter_path = input_dir / converter.ADAPTER_MODEL_FILE
    if not existing_adapter_path.exists():
        pytest.skip(f"{input_dir} has no {converter.ADAPTER_MODEL_FILE} to merge.")

    output_dir, _summary = _convert_real_adapter(input_dir, tmp_path)
    input_tensors = load_file(str(existing_adapter_path))
    output_tensors = load_file(str(output_dir / converter.ADAPTER_MODEL_FILE))

    for input_key, input_tensor in input_tensors.items():
        output_key = converter._clean_adapter_key(input_key)
        assert output_key in output_tensors
        assert output_tensors[output_key].shape == input_tensor.shape
        assert output_tensors[output_key].dtype == input_tensor.dtype
        assert torch.equal(output_tensors[output_key], input_tensor.cpu())


@pytest.mark.requires_model
def test_real_fused_conversion_without_input_config_uses_env_alpha(tmp_path):
    input_dir = _required_adapter_dir(KT_ADAPTER_ENV)
    alpha = _optional_lora_alpha()
    if alpha is None:
        pytest.skip(f"Set {KT_ALPHA_ENV} to validate conversion without input config.")

    no_config_input = tmp_path / "input_without_config"
    no_config_input.mkdir()
    _link_or_copy(
        input_dir / converter.FUSED_EXPERT_LORA_FILE,
        no_config_input / converter.FUSED_EXPERT_LORA_FILE,
    )
    existing_adapter_path = input_dir / converter.ADAPTER_MODEL_FILE
    if existing_adapter_path.exists():
        _link_or_copy(existing_adapter_path, no_config_input / converter.ADAPTER_MODEL_FILE)

    output_dir, summary = _convert_real_adapter(no_config_input, tmp_path, "output_without_config")

    assert summary["lora_alpha"] == alpha
    config = _assert_config_shape(output_dir, summary)
    assert config["lora_alpha"] == alpha
    _assert_fused_tensors_preserved(no_config_input, output_dir)


@pytest.mark.requires_model
def test_sglang_lora_config_loader_accepts_converted_adapter(tmp_path):
    input_dir = _required_adapter_dir(KT_ADAPTER_ENV)
    output_dir, summary = _convert_real_adapter(input_dir, tmp_path)

    sglang_python = REPO_ROOT / "third_party" / "sglang" / "python"
    sys.path.insert(0, str(sglang_python))
    try:
        from sglang.srt.lora.lora_config import LoRAConfig
    except Exception as exc:
        pytest.fail(f"Unable to import SGLang LoRAConfig: {exc}")

    lora_config = LoRAConfig(str(output_dir))
    output_tensors = load_file(str(output_dir / converter.ADAPTER_MODEL_FILE))

    assert lora_config.r == summary["rank"]
    assert lora_config.lora_alpha == summary["lora_alpha"]
    assert lora_config.target_modules == summary["target_modules"]
    assert len(output_tensors) == summary["tensor_count"]


@pytest.mark.requires_model
def test_large_adapter_conversion_smoke(tmp_path, record_property):
    input_dir = _required_adapter_dir(KT_LARGE_ADAPTER_ENV)

    start_time = time.perf_counter()
    output_dir, summary = _convert_real_adapter(input_dir, tmp_path, "large_output")
    duration_seconds = time.perf_counter() - start_time

    output_path = output_dir / converter.ADAPTER_MODEL_FILE
    output_tensors = load_file(str(output_path))
    config = _assert_config_shape(output_dir, summary)

    record_property("conversion_seconds", round(duration_seconds, 3))
    record_property("output_bytes", output_path.stat().st_size)
    record_property("tensor_count", summary["tensor_count"])

    assert len(output_tensors) == summary["tensor_count"]
    assert config["target_modules"] == summary["target_modules"]
