import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "sglang" / "python"))
sys.path.insert(0, str(REPO_ROOT / "kt-kernel" / "python"))

from sglang.srt.layers.moe import kt_ep_wrapper


class _FakeGpuMethod:
    def __init__(self):
        self.num_gpu_experts = None
        self.create_weights_kwargs = None

    def create_weights(self, **kwargs):
        self.create_weights_kwargs = kwargs


class _FakeLayer(torch.nn.Module):
    top_k = 2
    intermediate_size_per_partition = 5
    moe_tp_size = 1

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.empty(0))


class _FakeCudaStream:
    def __init__(self, *args, **kwargs):
        pass


class _FakeCudaEvent:
    def __init__(self, *args, **kwargs):
        pass


class _FakeCurrentStream:
    cuda_stream = 123


def _make_kt_config(*, gpu_mask, gpu_prefill_token_threshold=None):
    return kt_ep_wrapper.KTConfig(
        layer_idx=1,
        gpu_experts_mask=gpu_mask,
        cpuinfer_threads=8,
        threadpool_count=2,
        weight_path="/model",
        chunked_prefill_size=16,
        max_deferred_experts_per_token=0,
        method="RAWINT4",
        expert_lora_path="/adapter",
        gpu_prefill_token_threshold=gpu_prefill_token_threshold,
    )


def test_kt_ep_wrapper_uses_sft_wrapper_for_expert_lora(monkeypatch):
    wrapper_calls = []

    def fake_kt_moe_wrapper(**kwargs):
        wrapper_calls.append(kwargs)
        return object()

    monkeypatch.setattr(kt_ep_wrapper, "KTRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(kt_ep_wrapper, "KTMoEWrapper", fake_kt_moe_wrapper)
    monkeypatch.setattr(kt_ep_wrapper, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(kt_ep_wrapper.torch.cuda, "Stream", _FakeCudaStream)
    monkeypatch.setattr(kt_ep_wrapper.torch.cuda, "Event", _FakeCudaEvent)
    monkeypatch.setattr(
        kt_ep_wrapper,
        "_infer_kt_expert_lora_rank_alpha",
        lambda adapter_path, layer_idx: (2, 4.0),
    )
    monkeypatch.setattr(
        kt_ep_wrapper,
        "_infer_kt_kgroup_quant_config",
        lambda weight_path: (32, False),
    )

    gpu_method = _FakeGpuMethod()
    method = kt_ep_wrapper.KTEPWrapperMethod(
        gpu_method,
        _make_kt_config(gpu_mask=torch.zeros(3, dtype=torch.bool)),
    )
    method.create_weights(
        layer=_FakeLayer(),
        num_experts=3,
        hidden_size=4,
        intermediate_size_per_partition=5,
        params_dtype=torch.bfloat16,
    )

    assert gpu_method.num_gpu_experts == 0
    assert gpu_method.create_weights_kwargs["num_experts"] == 0
    assert len(wrapper_calls) == 1
    assert wrapper_calls[0]["mode"] == "sft"
    assert wrapper_calls[0]["method"] == "AMXINT4_KGroup_SFT"
    assert wrapper_calls[0]["num_gpu_experts"] == 0
    assert wrapper_calls[0]["lora_rank"] == 2
    assert wrapper_calls[0]["lora_alpha"] == 4.0
    assert wrapper_calls[0]["group_size"] == 32
    assert wrapper_calls[0]["zero_point"] is False


def test_kt_ep_wrapper_rejects_gpu_experts_with_expert_lora(monkeypatch):
    monkeypatch.setattr(kt_ep_wrapper, "KTRANSFORMERS_AVAILABLE", True)

    with pytest.raises(ValueError, match="requires all routed experts"):
        kt_ep_wrapper.KTEPWrapperMethod(
            _FakeGpuMethod(),
            _make_kt_config(gpu_mask=torch.tensor([False, True, False])),
        )


def test_kt_ep_wrapper_rejects_full_gpu_prefill_with_expert_lora(monkeypatch):
    monkeypatch.setattr(kt_ep_wrapper, "KTRANSFORMERS_AVAILABLE", True)

    with pytest.raises(ValueError, match="full-GPU prefill"):
        kt_ep_wrapper.KTEPWrapperMethod(
            _FakeGpuMethod(),
            _make_kt_config(
                gpu_mask=torch.zeros(3, dtype=torch.bool),
                gpu_prefill_token_threshold=128,
            ),
        )


def test_create_kt_config_keeps_cpu_only_expert_lora_enabled(monkeypatch):
    kt_ep_wrapper._KT_GPU_EXPERTS_MASKS = None


def test_kt_ep_wrapper_staged_forward_uses_inference_api_for_expert_lora(monkeypatch):
    calls = []

    class FakeWrapper:
        def submit_forward_inference(self, hidden_states, expert_ids, weights, cuda_stream):
            calls.append(
                (
                    "submit_forward_inference",
                    tuple(hidden_states.shape),
                    tuple(expert_ids.shape),
                    tuple(weights.shape),
                    cuda_stream,
                )
            )

        def sync_forward_inference(self, cuda_stream):
            calls.append(("sync_forward_inference", cuda_stream))
            return torch.full((1, 4), 3, dtype=torch.bfloat16)

        def submit_forward(self, *args):
            raise AssertionError("expert LoRA path must not call submit_forward")

        def sync_forward(self, *args):
            raise AssertionError("expert LoRA path must not call sync_forward")

    monkeypatch.setattr(kt_ep_wrapper, "KTRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(kt_ep_wrapper, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        kt_ep_wrapper.torch.cuda,
        "current_stream",
        lambda device=None: _FakeCurrentStream(),
    )

    method = kt_ep_wrapper.KTEPWrapperMethod(
        _FakeGpuMethod(),
        _make_kt_config(gpu_mask=torch.zeros(3, dtype=torch.bool)),
    )
    method.moe_runner_config = SimpleNamespace(activation="silu")
    method.wrapper = FakeWrapper()

    staged = torch.zeros((1, 4), dtype=torch.bfloat16)
    topk_ids = torch.tensor([[0, 1]], dtype=torch.int64)
    topk_weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    dispatch_output = SimpleNamespace(topk_output=(topk_weights, topk_ids, None))

    method._submit_with_staged_input(_FakeLayer(), dispatch_output, staged)
    output = method._sync_with_staged_input(staged)

    assert output.tolist() == [[3, 3, 3, 3]]
    assert calls == [
        ("submit_forward_inference", (1, 4), (1, 2), (1, 2), 123),
        ("sync_forward_inference", 123),
    ]
    monkeypatch.setattr(kt_ep_wrapper, "get_tensor_model_parallel_rank", lambda: 0)

    hf_config = SimpleNamespace(
        num_hidden_layers=4,
        n_routed_experts=3,
        first_k_dense_replace=1,
        moe_layer_freq=1,
    )
    server_args = SimpleNamespace(
        kt_weight_path="/model",
        kt_num_gpu_experts=0,
        kt_gpu_experts_ratio=None,
        kt_expert_placement_strategy="uniform",
        init_expert_location=None,
        kt_cpuinfer=8,
        kt_threadpool_count=2,
        kt_numa_nodes=None,
        chunked_prefill_size=16,
        kt_method="RAWINT4",
        kt_max_deferred_experts_per_token=0,
        kt_gpu_prefill_token_threshold=None,
        kt_enable_dynamic_expert_update=False,
        kt_expert_lora_path="/adapter",
        get_hf_config=lambda: hf_config,
    )

    config = kt_ep_wrapper.create_kt_config_from_server_args(server_args, layer_idx=1)

    assert config is not None
    assert config.expert_lora_path == "/adapter"
    assert config.gpu_experts_mask.tolist() == [False, False, False]
    assert kt_ep_wrapper._KT_GPU_EXPERTS_MASKS[0].tolist() == [True, True, True]

    kt_ep_wrapper._KT_GPU_EXPERTS_MASKS = None
