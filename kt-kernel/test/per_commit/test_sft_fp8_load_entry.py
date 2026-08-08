# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


SFT_PATH = Path(__file__).resolve().parents[2] / "python" / "sft"
PACKAGE_NAME = "kt_sft_fp8_entry_under_test"
package = ModuleType(PACKAGE_NAME)
package.__path__ = [str(SFT_PATH)]
sys.modules[PACKAGE_NAME] = package


def _stub_module(name: str, **attributes):
    module = ModuleType(f"{PACKAGE_NAME}.{name}")
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[module.__name__] = module
    return module


class _KTConfigError(RuntimeError):
    pass


_stub_module(
    "arch",
    KTAMXConfigError=_KTConfigError,
    KTAMXNotAvailableError=RuntimeError,
    _get_layers_prefix=lambda config: "model.layers",
    _get_model_container_and_layers=lambda model, purpose: (model, []),
    get_moe_arch_config=lambda config: SimpleNamespace(),
    get_moe_module=lambda layer, config: None,
    move_non_experts_to_gpu=lambda *args, **kwargs: None,
)
_stub_module("layer", KTMoELayerWrapper=object)
_stub_module("lora", LoRAExperts=object)
_stub_module("base", _supports_authoritative_optimizer_grads=lambda *args, **kwargs: False)
_stub_module(
    "backend",
    FP8_BACKEND="FP8",
    INT8_BACKEND="INT8",
    get_fp8_runtime=lambda: None,
    get_int8_runtime=lambda: None,
)
checkpoint_module = _stub_module(
    "checkpoint",
    load_full_weight_layer=lambda *args, **kwargs: None,
    resolve_full_weight_checkpoint=lambda path: None,
)
_stub_module("dist_utils", _distributed_rank_world_size=lambda: (0, 1))
_stub_module(
    "weights",
    _clear_original_expert_weights=lambda *args, **kwargs: None,
    extract_moe_weights=lambda *args, **kwargs: None,
    load_block_fp8_experts_from_checkpoint_files=lambda *args, **kwargs: None,
    load_experts_from_checkpoint_files=lambda *args, **kwargs: None,
)

spec = importlib.util.spec_from_file_location(
    f"{PACKAGE_NAME}.wrapper",
    SFT_PATH / "wrapper.py",
)
assert spec is not None and spec.loader is not None
wrapper = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = wrapper
spec.loader.exec_module(wrapper)


def test_get_kt_config_resolves_raw_nested_and_transformers_mappings(monkeypatch):
    class FakeKTConfig:
        field_names = (
            "kt_backend",
            "kt_num_threads",
            "kt_checkpoint_files",
            "kt_sharded_metadata",
            "kt_skip_expert_loading",
        )

        def __init__(self, **values):
            self.__dict__.update(values)

        @classmethod
        def from_object(cls, value):
            if isinstance(value, cls):
                return value
            if isinstance(value, dict):
                return cls(**value)
            values = {
                name: field_value
                for name in cls.field_names
                if (field_value := getattr(value, name, None)) is not None
            }
            if values:
                return cls(**values)
            for container_name in ("kt_config", "config"):
                nested = getattr(value, container_name, None)
                if nested is not None and nested is not value:
                    return cls.from_object(nested)
            return cls()

    class FakeHfTrainerKTConfig:
        def __init__(self, public_config, runtime_metadata):
            self.config = public_config
            self._runtime_metadata = runtime_metadata

        def __getattr__(self, name):
            if name in self._runtime_metadata:
                return self._runtime_metadata[name]
            try:
                return self.config[name]
            except KeyError as exc:
                raise AttributeError(name) from exc

    config_module = ModuleType(f"{PACKAGE_NAME}.config")
    config_module.KTConfig = FakeKTConfig
    monkeypatch.setitem(sys.modules, config_module.__name__, config_module)
    payload = {
        "kt_backend": "AMXBF16",
        "kt_num_threads": 0,
        "kt_skip_expert_loading": False,
    }
    checkpoint_files = ["/models/qwen/model-00001-of-00002.safetensors"]
    sharded_metadata = {"weight_map": {"expert.weight": checkpoint_files[0]}}

    raw = wrapper._get_kt_config(payload)
    nested = wrapper._get_kt_config(SimpleNamespace(kt_config=payload))
    generic_wrapper = wrapper._get_kt_config(SimpleNamespace(config=payload))
    transformers_wrapper = wrapper._get_kt_config(
        FakeHfTrainerKTConfig(
            payload,
            {
                "kt_checkpoint_files": checkpoint_files,
                "kt_sharded_metadata": sharded_metadata,
                "kt_skip_expert_loading": True,
            },
        )
    )

    assert payload == {
        "kt_backend": "AMXBF16",
        "kt_num_threads": 0,
        "kt_skip_expert_loading": False,
    }
    assert all(
        isinstance(item, FakeKTConfig)
        for item in (raw, nested, generic_wrapper, transformers_wrapper)
    )
    assert [item.kt_num_threads for item in (raw, nested, generic_wrapper)] == [
        0,
        0,
        0,
    ]
    assert transformers_wrapper.kt_backend == "AMXBF16"
    assert transformers_wrapper.kt_num_threads == 0
    assert transformers_wrapper.kt_checkpoint_files is checkpoint_files
    assert transformers_wrapper.kt_sharded_metadata is sharded_metadata
    assert transformers_wrapper.kt_skip_expert_loading is True


def test_load_kt_model_keeps_fp8_path_as_provenance_and_skips_hf_experts(monkeypatch):
    cfg = SimpleNamespace(
        kt_expert_weight_format="fp8",
        kt_weight_path="/models/deepseek-v3.1",
        kt_checkpoint_files=None,
        kt_sharded_metadata=None,
        kt_skip_expert_loading=False,
        kt_expert_checkpoint_path=None,
        kt_use_lora_experts=False,
        kt_full_weight_grad=False,
        kt_train_mode="lora",
        kt_tp_enabled=True,
    )
    plugin = SimpleNamespace(kt_config=cfg)
    checkpoint_files = ["/models/deepseek-v3.1/model-00001-of-00002.safetensors"]
    metadata = {"weight_map": {"model.layers.3.mlp.experts.0.gate_proj.weight": checkpoint_files[0]}}
    monkeypatch.setattr(wrapper, "_get_kt_config", lambda _: cfg)
    monkeypatch.setattr(
        wrapper,
        "_resolve_checkpoint_files",
        lambda **kwargs: (checkpoint_files, metadata),
    )
    monkeypatch.setattr(checkpoint_module, "resolve_full_weight_checkpoint", lambda path: None)
    monkeypatch.setattr(wrapper, "resolve_full_weight_checkpoint", lambda path: None)

    captured = {}

    class _FakeModel:
        _kt_wrappers = [SimpleNamespace()]

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(path, **kwargs):
            captured["path"] = path
            captured["skip_during_load"] = cfg.kt_skip_expert_loading
            captured["weight_path_during_load"] = cfg.kt_weight_path
            return _FakeModel()

    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = _AutoModelForCausalLM
    integrations = ModuleType("transformers.integrations")
    integrations.__path__ = []
    kt_integration = ModuleType("transformers.integrations.kt")
    kt_integration.set_kt_config = lambda value: captured.setdefault("plugin", value)
    kt_integration.unset_kt_config = lambda: None
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.integrations", integrations)
    monkeypatch.setitem(sys.modules, "transformers.integrations.kt", kt_integration)

    fake_arch = sys.modules[f"{PACKAGE_NAME}.arch"]
    monkeypatch.setattr(fake_arch, "move_non_experts_to_gpu", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(fake_arch, "get_moe_arch_config", lambda config: SimpleNamespace())

    model = wrapper.load_kt_model(
        SimpleNamespace(),
        kt_plugin=plugin,
        model_name_or_path=cfg.kt_weight_path,
    )

    assert isinstance(model, _FakeModel)
    assert captured["skip_during_load"] is True
    assert captured["weight_path_during_load"] == "/models/deepseek-v3.1"
    assert cfg.kt_checkpoint_files == checkpoint_files
    assert cfg.kt_sharded_metadata == metadata
    assert cfg.kt_skip_expert_loading is True
    assert captured["plugin"] is cfg


@pytest.mark.parametrize(
    "checkpoint_files",
    [None, ["/models/deepseek-v3.1/pytorch_model-00001-of-00002.bin"]],
)
def test_load_kt_model_fp8_fails_closed_before_hf_load(monkeypatch, checkpoint_files):
    cfg = SimpleNamespace(
        kt_expert_weight_format="fp8",
        kt_weight_path="/models/deepseek-v3.1",
        kt_checkpoint_files=None,
        kt_sharded_metadata=None,
        kt_skip_expert_loading=False,
        kt_expert_checkpoint_path=None,
    )
    plugin = SimpleNamespace(kt_config=cfg)
    monkeypatch.setattr(wrapper, "_get_kt_config", lambda _: cfg)
    monkeypatch.setattr(
        wrapper,
        "_resolve_checkpoint_files",
        lambda **kwargs: (checkpoint_files, None),
    )
    monkeypatch.setattr(wrapper, "resolve_full_weight_checkpoint", lambda path: None)

    called = False

    class _AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(path, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("from_pretrained must not run")

    transformers = ModuleType("transformers")
    transformers.AutoModelForCausalLM = _AutoModelForCausalLM
    integrations = ModuleType("transformers.integrations")
    integrations.__path__ = []
    kt_integration = ModuleType("transformers.integrations.kt")
    kt_integration.set_kt_config = lambda value: None
    kt_integration.unset_kt_config = lambda: None
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.integrations", integrations)
    monkeypatch.setitem(sys.modules, "transformers.integrations.kt", kt_integration)

    with pytest.raises(_KTConfigError, match="raw safetensors"):
        wrapper.load_kt_model(
            SimpleNamespace(),
            kt_plugin=plugin,
            model_name_or_path=cfg.kt_weight_path,
        )

    assert called is False
    assert cfg.kt_skip_expert_loading is True
    assert cfg.kt_weight_path == "/models/deepseek-v3.1"
