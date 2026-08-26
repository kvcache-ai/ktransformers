# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

import kt_kernel.sft.artifacts as artifacts_module
from kt_kernel.sft.artifacts import (
    KTArtifactError,
    claim_kt_routed_expert_subtrees,
    hide_kt_int8_routed_experts_from_dispatch,
    hide_kt_routed_experts_from_dispatch,
    is_kt_int8_routed_expert_base_parameter,
    is_kt_routed_expert_base_parameter,
    mark_kt_int8_routed_expert_base_parameters,
    prepare_kt_int8_non_expert_device_map,
    prepare_kt_non_expert_device_map,
    project_kt_int8_routed_experts_out_of_device_map,
    project_kt_routed_experts_out_of_device_map,
)


HIDDEN_SIZE = 4
INTERMEDIATE_SIZE = 3
EXPERT_COUNT = 2


class _FusedExperts(torch.nn.Module):
    def __init__(self, *, gate_up_shape=None):
        super().__init__()
        gate_up_shape = gate_up_shape or (
            EXPERT_COUNT,
            2 * INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
        )
        self.gate_up_proj = torch.nn.Parameter(
            torch.empty(gate_up_shape, device="meta")
        )
        self.down_proj = torch.nn.Parameter(
            torch.empty(EXPERT_COUNT, HIDDEN_SIZE, INTERMEDIATE_SIZE, device="meta")
        )
        self.register_buffer(
            "weight_scale", torch.empty(EXPERT_COUNT, 1, 1, device="meta")
        )


class _Expert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(
            HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False, device="meta"
        )
        self.up_proj = torch.nn.Linear(
            HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False, device="meta"
        )
        self.down_proj = torch.nn.Linear(
            INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False, device="meta"
        )


def _sparse_layer(experts):
    layer = torch.nn.Module()
    layer.mlp = torch.nn.Module()
    layer.mlp.experts = experts
    layer.mlp.gate = torch.nn.Linear(
        HIDDEN_SIZE, EXPERT_COUNT, bias=False, device="meta"
    )
    layer.mlp.shared_expert = torch.nn.Linear(
        HIDDEN_SIZE, HIDDEN_SIZE, bias=False, device="meta"
    )
    return layer


def _dense_layer():
    layer = torch.nn.Module()
    layer.mlp = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=False, device="meta")
    return layer


def _qwen_fused_model(*, gate_up_shape=None):
    model = torch.nn.Module()
    model.config = SimpleNamespace(
        architectures=["Qwen3_5MoeForCausalLM"],
        text_config=SimpleNamespace(
            hidden_size=HIDDEN_SIZE,
            num_experts=EXPERT_COUNT,
            moe_intermediate_size=INTERMEDIATE_SIZE,
            num_experts_per_tok=1,
            shared_expert_intermediate_size=HIDDEN_SIZE,
        ),
    )
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    model.model.language_model.layers = torch.nn.ModuleList(
        [_sparse_layer(_FusedExperts(gate_up_shape=gate_up_shape)) for _ in range(2)]
    )
    model.model.embed_tokens = torch.nn.Embedding(8, HIDDEN_SIZE, device="meta")
    return model


def _deepseek_model(*, fused=False, gate_up_shape=None):
    model = torch.nn.Module()
    model.config = SimpleNamespace(
        architectures=["DeepseekV3ForCausalLM"],
        model_type="deepseek_v3",
        num_hidden_layers=3,
        first_k_dense_replace=1,
        n_routed_experts=EXPERT_COUNT,
        hidden_size=HIDDEN_SIZE,
        moe_intermediate_size=INTERMEDIATE_SIZE,
        num_experts_per_tok=1,
        n_shared_experts=1,
    )
    model.model = torch.nn.Module()
    make_experts = (
        (lambda: _FusedExperts(gate_up_shape=gate_up_shape))
        if fused
        else (lambda: torch.nn.ModuleList([_Expert() for _ in range(EXPERT_COUNT)]))
    )
    model.model.layers = torch.nn.ModuleList(
        [_dense_layer(), _sparse_layer(make_experts()), _sparse_layer(make_experts())]
    )
    model.model.embed_tokens = torch.nn.Embedding(8, HIDDEN_SIZE, device="meta")
    return model


def _owned_tensors(model, paths):
    return {
        f"{path}.{name}": tensor
        for path in paths
        for name, tensor in (
            *model.get_submodule(path).named_parameters(
                recurse=True, remove_duplicate=False
            ),
            *model.get_submodule(path).named_buffers(
                recurse=True, remove_duplicate=False
            ),
        )
    }


def _install_zero_storage_placeholders(experts):
    for module in experts.modules():
        for name, parameter in tuple(module._parameters.items()):
            if parameter is None:
                continue
            placeholder = torch.nn.Parameter(
                torch.empty(0, dtype=parameter.dtype, device="meta"),
                requires_grad=False,
            )
            placeholder._kt_zero_storage = True
            placeholder._kt_original_shape = tuple(parameter.shape)
            module._parameters[name] = placeholder


@pytest.mark.parametrize("factory", [_qwen_fused_model, _deepseek_model])
def test_generic_claim_project_and_restore_supports_fused_and_module_list(factory):
    model = factory()
    paths = claim_kt_routed_expert_subtrees(model)
    expected_prefix = (
        "model.language_model.layers"
        if factory is _qwen_fused_model
        else "model.layers"
    )
    expected_indices = (0, 1) if factory is _qwen_fused_model else (1, 2)

    assert paths == tuple(
        f"{expected_prefix}.{index}.mlp.experts" for index in expected_indices
    )
    before = _owned_tensors(model, paths)
    assert before
    assert all(
        is_kt_routed_expert_base_parameter(tensor)
        for tensor in before.values()
        if isinstance(tensor, torch.nn.Parameter)
    )

    with project_kt_routed_experts_out_of_device_map(model):
        during = _owned_tensors(model, paths)
        assert set(during) == set(before)
        assert all(
            tensor.device.type == "meta" and tensor.numel() == 0
            for tensor in during.values()
        )
        assert all(during[name] is not before[name] for name in before)

    after = _owned_tensors(model, paths)
    assert all(after[name] is before[name] for name in before)
    assert claim_kt_routed_expert_subtrees(model) == paths


def test_prepare_device_map_removes_only_owned_experts_and_rejects_real_host_offload():
    model = _qwen_fused_model()
    paths = claim_kt_routed_expert_subtrees(model)
    device_map = {
        "model.embed_tokens": "cuda:0",
        paths[0]: "cpu",
        f"{paths[0]}.gate_up_proj": "disk",
        paths[1]: "meta",
    }

    resolved = prepare_kt_non_expert_device_map(model, device_map)

    assert resolved == {"model.embed_tokens": "cuda:0"}
    assert device_map[paths[0]] == "cpu"
    with pytest.raises(KTArtifactError, match="offloaded real tensors"):
        prepare_kt_non_expert_device_map(
            model, {"model.embed_tokens": "cpu", paths[0]: "cpu"}
        )
    with pytest.raises(KTArtifactError, match="became empty"):
        prepare_kt_non_expert_device_map(model, {path: "cpu" for path in paths})


def test_hide_restores_exact_subtree_identities_on_success_and_error():
    model = _deepseek_model()
    paths = claim_kt_routed_expert_subtrees(model)
    originals = {path: model.get_submodule(path) for path in paths}
    unrelated = model.model.layers[1].mlp.gate

    with hide_kt_routed_experts_from_dispatch(model):
        assert all(model.get_submodule(path) is not originals[path] for path in paths)
        assert model.model.layers[1].mlp.gate is unrelated
    assert all(model.get_submodule(path) is originals[path] for path in paths)

    with pytest.raises(RuntimeError, match="body failed"):
        with hide_kt_routed_experts_from_dispatch(model):
            raise RuntimeError("body failed")
    assert all(model.get_submodule(path) is originals[path] for path in paths)


def test_wrap_style_replacement_preserves_claimed_path_and_expert_identity():
    model = _qwen_fused_model()
    paths = claim_kt_routed_expert_subtrees(model)
    originals = {path: model.get_submodule(path) for path in paths}

    for layer in model.model.language_model.layers:
        experts = layer.mlp.experts
        wrapped_moe = torch.nn.Module()
        wrapped_moe.experts = experts
        wrapped_moe.gate = layer.mlp.gate
        wrapped_moe.shared_expert = layer.mlp.shared_expert
        layer.mlp = wrapped_moe
        _install_zero_storage_placeholders(experts)

    assert claim_kt_routed_expert_subtrees(model) == paths
    assert all(model.get_submodule(path) is originals[path] for path in paths)
    with hide_kt_routed_experts_from_dispatch(model):
        pass
    assert all(model.get_submodule(path) is originals[path] for path in paths)


@pytest.mark.parametrize("drift", ["paths", "marker", "identity", "contract"])
def test_runtime_ownership_drift_fails_closed_before_device_map_mutation(drift):
    model = _qwen_fused_model()
    paths = claim_kt_routed_expert_subtrees(model)
    original_map = {"model.embed_tokens": "cuda:0", paths[0]: "cpu"}
    experts = model.get_submodule(paths[0])

    if drift == "paths":
        setattr(model, artifacts_module._RUNTIME_MODULE_PATHS, tuple(reversed(paths)))
    elif drift == "marker":
        delattr(experts, artifacts_module._RUNTIME_MODULE_MARKER)
    elif drift == "identity":
        model.model.language_model.layers[0].mlp.experts = _FusedExperts()
    else:
        experts.gate_up_proj = torch.nn.Parameter(
            torch.empty(
                EXPERT_COUNT, 2 * INTERMEDIATE_SIZE + 1, HIDDEN_SIZE, device="meta"
            )
        )

    with pytest.raises(KTArtifactError):
        prepare_kt_non_expert_device_map(model, original_map)
    assert original_map == {"model.embed_tokens": "cuda:0", paths[0]: "cpu"}


def test_ordinary_model_is_an_exact_noop_even_with_coincidental_experts():
    model = torch.nn.Module()
    model.config = SimpleNamespace(architectures=["LlamaForCausalLM"])
    model.experts = torch.nn.Linear(2, 2, bias=False, device="meta")
    model.experts.weight._kt_zero_storage = True
    device_map = {"": "cpu"}
    original = model.experts

    assert claim_kt_routed_expert_subtrees(model) == ()
    assert prepare_kt_non_expert_device_map(model, device_map) is device_map
    with project_kt_routed_experts_out_of_device_map(model):
        assert model.experts is original
    with hide_kt_routed_experts_from_dispatch(model):
        assert model.experts is original
    assert model.experts is original
    assert not hasattr(model, artifacts_module._RUNTIME_MODULE_PATHS)


def test_int8_apis_keep_exact_shape_validation_and_delegate_to_generic_contract():
    model = _deepseek_model(fused=True)
    paths = mark_kt_int8_routed_expert_base_parameters(model, object())
    parameters = [
        parameter
        for path in paths
        for parameter in model.get_submodule(path).parameters()
    ]

    assert paths == claim_kt_routed_expert_subtrees(model)
    assert all(
        is_kt_int8_routed_expert_base_parameter(parameter) for parameter in parameters
    )
    assert all(
        is_kt_routed_expert_base_parameter(parameter) for parameter in parameters
    )
    assert prepare_kt_int8_non_expert_device_map(
        model, {"model.embed_tokens": "cuda:0", paths[0]: "cpu"}
    ) == {"model.embed_tokens": "cuda:0"}
    with project_kt_int8_routed_experts_out_of_device_map(model):
        assert all(
            model.get_submodule(path).gate_up_proj.numel() == 0 for path in paths
        )
    originals = {path: model.get_submodule(path) for path in paths}
    with hide_kt_int8_routed_experts_from_dispatch(model):
        assert all(model.get_submodule(path) is not originals[path] for path in paths)
    assert all(model.get_submodule(path) is originals[path] for path in paths)

    invalid = _deepseek_model(
        fused=True, gate_up_shape=(EXPERT_COUNT, 2 * INTERMEDIATE_SIZE + 1, HIDDEN_SIZE)
    )
    with pytest.raises(KTArtifactError, match="gate_up_proj shape mismatch"):
        mark_kt_int8_routed_expert_base_parameters(invalid, object())
    assert not hasattr(invalid, artifacts_module._RUNTIME_MODULE_PATHS)
