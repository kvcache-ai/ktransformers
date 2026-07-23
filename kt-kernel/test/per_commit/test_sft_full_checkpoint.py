# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from safetensors.torch import save_file

from kt_kernel.sft.arch import MOEArchConfig
from kt_kernel.sft.checkpoint import (
    FULL_WEIGHT_INDEX_NAME,
    load_full_weight_checkpoint,
    load_full_weight_layer,
    resolve_full_weight_checkpoint,
)
from kt_kernel.sft.config import KTConfig
from kt_kernel.sft.layer import KTMoELayerWrapper
from kt_kernel.sft.lora import load_kt_moe_from_adapter, save_kt_moe_to_adapter
from kt_kernel.sft.wrapper import wrap_moe_layers_with_kt_wrapper


class _FakeFullBackend:
    def __init__(self, *, expert_num=2, intermediate_size=3, hidden_size=4, fill=0.0):
        self.gate_proj_buf = nn.Parameter(
            torch.full((expert_num, intermediate_size, hidden_size), fill, dtype=torch.bfloat16)
        )
        self.up_proj_buf = nn.Parameter(
            torch.full((expert_num, intermediate_size, hidden_size), fill + 1, dtype=torch.bfloat16)
        )
        self.down_proj_buf = nn.Parameter(
            torch.full((expert_num, hidden_size, intermediate_size), fill + 2, dtype=torch.bfloat16)
        )
        self._base_weights_dirty = False
        self._kt_full_checkpoint_load_failed = False


def _fake_wrapper(layer_idx, *, fill=0.0, full=True, backend=True):
    moe_config = MOEArchConfig(
        moe_layer_attr="mlp",
        router_attr="gate",
        experts_attr="experts",
        weight_names=("gate_proj", "up_proj", "down_proj"),
        expert_num=2,
        intermediate_size=3,
        num_experts_per_tok=1,
    )
    return SimpleNamespace(
        layer_idx=layer_idx,
        hidden_size=4,
        moe_config=moe_config,
        _full_weight_grad=full,
        wrapper=_FakeFullBackend(fill=fill) if backend else None,
        lora_experts=None,
        _fused_expert_lora_params=None,
    )


def _params(wrapper):
    backend = wrapper.wrapper
    return (backend.gate_proj_buf, backend.up_proj_buf, backend.down_proj_buf)


def test_full_checkpoint_round_trip_preserves_parameter_grad_and_optimizer_identity(tmp_path):
    wrappers = [_fake_wrapper(1, fill=10), _fake_wrapper(3, fill=20)]
    model = SimpleNamespace(_kt_wrappers=wrappers)
    all_params = [param for wrapper in wrappers for param in _params(wrapper)]
    optimizer = torch.optim.SGD(all_params, lr=0.1, momentum=0.9)

    for param in all_params:
        param.grad = torch.ones_like(param)
    optimizer.step()

    saved_values = [param.detach().clone() for param in all_params]
    parameter_ids = [id(param) for param in all_params]
    parameter_ptrs = [param.data_ptr() for param in all_params]
    grad_ptrs = [param.grad.data_ptr() for param in all_params]
    optimizer_state_ids = [id(optimizer.state[param]) for param in all_params]

    save_kt_moe_to_adapter(model, str(tmp_path))
    assert resolve_full_weight_checkpoint(tmp_path) == str(tmp_path)
    assert (tmp_path / FULL_WEIGHT_INDEX_NAME).is_file()

    with torch.no_grad():
        for param in all_params:
            param.fill_(-7)

    load_kt_moe_from_adapter(model, str(tmp_path))

    for param, expected in zip(all_params, saved_values):
        torch.testing.assert_close(param, expected)
    assert [id(param) for param in all_params] == parameter_ids
    assert [param.data_ptr() for param in all_params] == parameter_ptrs
    assert [param.grad.data_ptr() for param in all_params] == grad_ptrs
    assert [id(optimizer.state[param]) for param in all_params] == optimizer_state_ids
    assert all(wrapper.wrapper._base_weights_dirty for wrapper in wrappers)
    assert all(not wrapper.wrapper._kt_full_checkpoint_load_failed for wrapper in wrappers)


def test_full_checkpoint_is_published_atomically_by_manifest(tmp_path, monkeypatch):
    wrappers = [_fake_wrapper(0, fill=1), _fake_wrapper(1, fill=2)]
    model = SimpleNamespace(_kt_wrappers=wrappers)
    save_kt_moe_to_adapter(model, str(tmp_path))
    old_index = json.loads((tmp_path / FULL_WEIGHT_INDEX_NAME).read_text())
    old_values = [param.detach().clone() for wrapper in wrappers for param in _params(wrapper)]

    with torch.no_grad():
        for param in [param for wrapper in wrappers for param in _params(wrapper)]:
            param.add_(100)

    import safetensors.torch

    real_save_file = safetensors.torch.save_file
    call_count = 0

    def fail_second_shard(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("synthetic shard write failure")
        return real_save_file(*args, **kwargs)

    monkeypatch.setattr(safetensors.torch, "save_file", fail_second_shard)
    with pytest.raises(OSError, match="synthetic shard write failure"):
        save_kt_moe_to_adapter(model, str(tmp_path))

    assert json.loads((tmp_path / FULL_WEIGHT_INDEX_NAME).read_text()) == old_index
    load_kt_moe_from_adapter(model, str(tmp_path))
    for param, expected in zip(
        [param for wrapper in wrappers for param in _params(wrapper)],
        old_values,
    ):
        torch.testing.assert_close(param, expected)


def test_full_checkpoint_layer_loader_validates_and_returns_logical_tensors(tmp_path):
    wrapper = _fake_wrapper(7, fill=4)
    model = SimpleNamespace(_kt_wrappers=[wrapper])
    save_kt_moe_to_adapter(model, str(tmp_path))

    expected_shapes = {
        "gate_proj": (2, 3, 4),
        "up_proj": (2, 3, 4),
        "down_proj": (2, 4, 3),
    }
    loaded = load_full_weight_layer(
        str(tmp_path),
        layer_idx=7,
        expected_shapes=expected_shapes,
    )
    for tensor, expected in zip(loaded, _params(wrapper)):
        torch.testing.assert_close(tensor, expected)
        assert tensor.is_contiguous()

    bad_shapes = dict(expected_shapes)
    bad_shapes["gate_proj"] = (2, 4, 4)
    with pytest.raises(RuntimeError, match="index shapes mismatch"):
        load_full_weight_layer(str(tmp_path), layer_idx=7, expected_shapes=bad_shapes)


def test_full_checkpoint_rejects_missing_layer_without_mutating_parameters(tmp_path):
    wrappers = [_fake_wrapper(0, fill=3), _fake_wrapper(1, fill=5)]
    model = SimpleNamespace(_kt_wrappers=wrappers)
    save_kt_moe_to_adapter(model, str(tmp_path))
    before = [param.detach().clone() for wrapper in wrappers for param in _params(wrapper)]

    index_path = tmp_path / FULL_WEIGHT_INDEX_NAME
    index = json.loads(index_path.read_text())
    del index["layers"]["1"]
    index_path.write_text(json.dumps(index))

    with pytest.raises(RuntimeError, match="layer set mismatch"):
        load_kt_moe_from_adapter(model, str(tmp_path))
    for param, expected in zip(
        [param for wrapper in wrappers for param in _params(wrapper)],
        before,
    ):
        torch.testing.assert_close(param, expected)


def test_full_checkpoint_rejects_corrupt_shard_size(tmp_path):
    wrapper = _fake_wrapper(0)
    model = SimpleNamespace(_kt_wrappers=[wrapper])
    save_kt_moe_to_adapter(model, str(tmp_path))
    index = json.loads((tmp_path / FULL_WEIGHT_INDEX_NAME).read_text())
    shard = tmp_path / index["layers"]["0"]["file"]
    shard.write_bytes(shard.read_bytes()[:-1])

    with pytest.raises(RuntimeError, match="shard size mismatch"):
        load_kt_moe_from_adapter(model, str(tmp_path))


def test_lora_only_model_does_not_create_full_checkpoint(tmp_path):
    wrapper = _fake_wrapper(0, full=False, backend=False)
    model = SimpleNamespace(_kt_wrappers=[wrapper])
    save_kt_moe_to_adapter(model, str(tmp_path))
    assert not (tmp_path / FULL_WEIGHT_INDEX_NAME).exists()


def test_full_model_refuses_to_resume_without_full_checkpoint(tmp_path):
    model = SimpleNamespace(_kt_wrappers=[_fake_wrapper(0)])
    with pytest.raises(RuntimeError, match=FULL_WEIGHT_INDEX_NAME):
        load_kt_moe_from_adapter(model, str(tmp_path))


class _FakeKTBackend:
    def __init__(self, **kwargs):
        self.num_experts = kwargs["num_experts"]
        self.moe_intermediate_size = kwargs["moe_intermediate_size"]
        self.hidden_size = kwargs["hidden_size"]
        self._full_weight_grad = kwargs["full_weight_grad"]
        self._uses_authoritative_optimizer_grads = False
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None

    def load_weights_from_tensors(self, gate_proj, up_proj, down_proj, physical_to_logical_map_cpu):
        torch.testing.assert_close(
            physical_to_logical_map_cpu,
            torch.arange(self.num_experts, dtype=torch.int64),
        )
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def init_full_weight_grad_buffers(self, gate_proj, up_proj, down_proj):
        self.gate_proj_buf = nn.Parameter(gate_proj.clone())
        self.up_proj_buf = nn.Parameter(up_proj.clone())
        self.down_proj_buf = nn.Parameter(down_proj.clone())
        self.grad_gate_proj_buf = torch.empty_like(self.gate_proj_buf)
        self.grad_up_proj_buf = torch.empty_like(self.up_proj_buf)
        self.grad_down_proj_buf = torch.empty_like(self.down_proj_buf)


class _TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _OriginalMoE()


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            architectures=["Qwen3MoeForCausalLM"],
            hidden_size=4,
            num_experts=2,
            moe_intermediate_size=3,
            num_experts_per_tok=1,
            max_position_embeddings=16,
        )
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyLayer()])


def test_wrapping_uses_full_checkpoint_as_initial_authoritative_source(tmp_path, monkeypatch):
    saved_wrapper = _fake_wrapper(0, fill=6)
    save_kt_moe_to_adapter(SimpleNamespace(_kt_wrappers=[saved_wrapper]), str(tmp_path))
    expected = [param.detach().clone() for param in _params(saved_wrapper)]

    import kt_kernel.sft.wrapper as wrapper_module

    monkeypatch.setattr(wrapper_module, "KT_KERNEL_AVAILABLE", True)
    monkeypatch.setattr(wrapper_module, "KTMoEWrapper", _FakeKTBackend)
    model = _TinyModel()
    with torch.no_grad():
        for param in model.model.layers[0].mlp.parameters():
            param.fill_(99)
    config = KTConfig(
        kt_backend="AMXBF16",
        kt_expert_checkpoint_path=str(tmp_path),
        kt_skip_expert_loading=True,
        kt_train_mode="full",
        kt_full_weight_grad=True,
        kt_lora_rank=0,
        kt_model_max_length=16,
    )

    wrappers = wrap_moe_layers_with_kt_wrapper(model, config)
    assert len(wrappers) == 1
    for actual, saved in zip(_params(wrappers[0]), expected):
        torch.testing.assert_close(actual, saved)


def _distributed_load_worker(rank, checkpoint_path, init_file, result_queue):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
    )
    try:
        wrapper = _fake_wrapper(0, fill=-9, backend=rank == 0)
        load_full_weight_checkpoint([wrapper], checkpoint_path)
        result_queue.put(
            (
                rank,
                "ok",
                float(wrapper.wrapper.gate_proj_buf.detach().float().sum()) if rank == 0 else None,
            )
        )
    except Exception as exc:
        result_queue.put((rank, "error", str(exc)))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available(), reason="torch.distributed is unavailable")
def test_distributed_full_checkpoint_load_is_rank0_owned_and_error_synchronized(tmp_path):
    wrapper = _fake_wrapper(0, fill=6)
    save_kt_moe_to_adapter(SimpleNamespace(_kt_wrappers=[wrapper]), str(tmp_path))
    expected_sum = float(wrapper.wrapper.gate_proj_buf.detach().float().sum())

    context = mp.get_context("fork")
    result_queue = context.Queue()
    init_file = tmp_path / "dist-success"
    processes = [
        context.Process(
            target=_distributed_load_worker,
            args=(rank, str(tmp_path), str(init_file), result_queue),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    results = [result_queue.get(timeout=20) for _ in processes]
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    assert sorted((rank, status) for rank, status, _ in results) == [(0, "ok"), (1, "ok")]
    assert next(value for rank, _, value in results if rank == 0) == expected_sum

    index_path = tmp_path / FULL_WEIGHT_INDEX_NAME
    index = json.loads(index_path.read_text())
    index["layers"] = {}
    index_path.write_text(json.dumps(index))

    failure_queue = context.Queue()
    failure_init_file = tmp_path / "dist-failure"
    failure_processes = [
        context.Process(
            target=_distributed_load_worker,
            args=(rank, str(tmp_path), str(failure_init_file), failure_queue),
        )
        for rank in range(2)
    ]
    for process in failure_processes:
        process.start()
    failures = [failure_queue.get(timeout=20) for _ in failure_processes]
    for process in failure_processes:
        process.join(timeout=20)
        assert process.exitcode == 0
    assert sorted((rank, status) for rank, status, _ in failures) == [(0, "error"), (1, "error")]
    assert len({message for _, _, message in failures}) == 1
    assert "layer set mismatch" in failures[0][2]


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(4, 3, bias=False)
        self.up_proj = nn.Linear(4, 3, bias=False)
        self.down_proj = nn.Linear(3, 4, bias=False)


class _OriginalMoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(4, 2, bias=False)
        self.experts = nn.ModuleList([_Expert(), _Expert()])


def test_state_dict_hook_removes_only_marked_expert_placeholders(tmp_path):
    moe_config = MOEArchConfig(
        moe_layer_attr="mlp",
        router_attr="gate",
        experts_attr="experts",
        weight_names=("gate_proj", "up_proj", "down_proj"),
        expert_num=2,
        intermediate_size=3,
        num_experts_per_tok=1,
    )
    layer = KTMoELayerWrapper(
        original_moe=_OriginalMoE(),
        wrapper=None,
        lora_params=None,
        moe_config=moe_config,
        hidden_size=4,
        layer_idx=0,
        full_weight_grad=True,
    )
    tiny_storage = torch.UntypedStorage(1, device="cpu")
    fake_tensor = torch.tensor([], dtype=torch.float32, device="cpu").set_(
        tiny_storage,
        storage_offset=0,
        size=(3, 4),
        stride=(0, 0),
    )
    placeholder = nn.Parameter(fake_tensor, requires_grad=False)
    placeholder._kt_zero_storage = True
    placeholder_storage_nbytes = placeholder.untyped_storage().nbytes()
    layer.experts[0].gate_proj._parameters["weight"] = placeholder

    state_dict = layer.state_dict()
    assert "experts.0.gate_proj.weight" not in state_dict
    assert "experts.0.up_proj.weight" in state_dict
    assert "gate.weight" in state_dict

    save_file(state_dict, str(tmp_path / "model.safetensors"))
    load_result = layer.load_state_dict(state_dict, strict=True, assign=True)
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []

    loaded_placeholder = layer.experts[0].gate_proj.weight
    assert loaded_placeholder is placeholder
    assert loaded_placeholder._kt_zero_storage is True
    assert loaded_placeholder.requires_grad is False
    assert loaded_placeholder.stride() == (0, 0)
    assert loaded_placeholder.untyped_storage().nbytes() == placeholder_storage_nbytes
    assert placeholder_storage_nbytes == placeholder.element_size()
    assert "experts.0.gate_proj.weight" not in layer.state_dict()
