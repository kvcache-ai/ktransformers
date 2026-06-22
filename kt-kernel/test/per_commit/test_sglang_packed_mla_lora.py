import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "sglang" / "python"))

from sglang.srt.lora.backend.torch_backend import (  # noqa: E402
    TorchNativeLoRABackend,
    TorchNativeLoRABatchInfo,
)
from sglang.srt.lora.layers import FusedQKVAProjWithMQALoRA  # noqa: E402
from sglang.srt.lora.lora import LoRAAdapter  # noqa: E402
from sglang.srt.lora.lora_config import LoRAConfig  # noqa: E402
from sglang.srt.lora.lora_manager import LoRAManager  # noqa: E402
from sglang.srt.lora.lora_registry import LoRARef  # noqa: E402
from sglang.srt.layers.linear import ReplicatedLinear  # noqa: E402
from sglang.srt.models.kimi_k25 import KimiK25ForConditionalGeneration  # noqa: E402


class _LinearQuantMethod:
    @staticmethod
    def apply(layer, x, bias=None):
        return F.linear(x, layer.weight, bias)


class _FakeReplicatedLinear(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(output_size, input_size))
        self.bias = None
        self.skip_bias_add = False
        self.output_size = output_size
        self.quant_method = _LinearQuantMethod()


def _make_backend(rank: int, scaling: float):
    backend = TorchNativeLoRABackend(max_loras_per_batch=2, device=torch.device("cpu"))
    backend.batch_info = TorchNativeLoRABatchInfo(
        use_cuda_graph=False,
        bs=1,
        num_segments=1,
        seg_indptr=torch.tensor([0, 5], dtype=torch.int32),
        weight_indices=torch.tensor([1], dtype=torch.int32),
        lora_ranks=torch.tensor([0, rank], dtype=torch.int32),
        scalings=torch.tensor([0.0, scaling], dtype=torch.float32),
        max_len=5,
        seg_lens=torch.tensor([5], dtype=torch.int32),
        permutation=None,
        lora_ranks_cpu=torch.tensor([0, rank], dtype=torch.int32),
        seg_indptr_cpu=torch.tensor([0, 5], dtype=torch.int32),
        seg_lens_cpu=torch.tensor([5], dtype=torch.int32),
        weight_indices_cpu=torch.tensor([1], dtype=torch.int32),
    )
    return backend


def test_packed_mla_lora_applies_q_and_kv_slices():
    torch.manual_seed(0)
    input_dim = 4
    q_dim = 2
    kv_dim = 3
    rank = 2
    scaling = 1.5

    base = _FakeReplicatedLinear(input_dim, q_dim + kv_dim)
    layer = FusedQKVAProjWithMQALoRA(base, _make_backend(rank, scaling))
    x = torch.randn(5, input_dim)

    q_a = torch.randn(2, rank, input_dim)
    q_b = torch.randn(2, q_dim, rank)
    kv_a = torch.randn(2, rank, input_dim)
    kv_b = torch.randn(2, kv_dim, rank)
    layer.set_lora_info(q_a, q_b, kv_a, kv_b)

    actual, _ = layer(x)

    expected = F.linear(x, base.weight)
    expected[:, :q_dim] += scaling * (x @ q_a[1].T) @ q_b[1].T
    expected[:, q_dim:] += scaling * (x @ kv_a[1].T) @ kv_b[1].T

    torch.testing.assert_close(actual, expected)


def _make_adapter(rank: int = 2):
    config = LoRAConfig.from_dict(
        {
            "peft_type": "LORA",
            "r": rank,
            "lora_alpha": rank,
            "target_modules": ["q_a_proj", "kv_a_proj_with_mqa"],
        }
    )
    return LoRAAdapter(
        uid="test",
        config=config,
        base_hf_config=SimpleNamespace(num_hidden_layers=1),
        load_config=SimpleNamespace(),
        lora_backend=_make_backend(rank, 1.0),
    )


def test_packed_mla_lora_rejects_missing_branch():
    adapter = _make_adapter()

    with pytest.raises(ValueError, match="kv_a_proj_with_mqa"):
        adapter.initialize_weights_from_tensors(
            {
                "base_model.model.language_model.model.layers.0.self_attn.q_a_proj.lora_A.weight": torch.randn(
                    2, 4
                ),
                "base_model.model.language_model.model.layers.0.self_attn.q_a_proj.lora_B.weight": torch.randn(
                    2, 2
                ),
            }
        )


def test_packed_mla_lora_rejects_rank_mismatch():
    adapter = _make_adapter(rank=2)

    with pytest.raises(ValueError, match="same rank"):
        adapter.initialize_weights_from_tensors(
            {
                "base_model.model.language_model.model.layers.0.self_attn.q_a_proj.lora_A.weight": torch.randn(
                    2, 4
                ),
                "base_model.model.language_model.model.layers.0.self_attn.q_a_proj.lora_B.weight": torch.randn(
                    2, 2
                ),
                "base_model.model.language_model.model.layers.0.self_attn.kv_a_proj_with_mqa.lora_A.weight": torch.randn(
                    3, 4
                ),
                "base_model.model.language_model.model.layers.0.self_attn.kv_a_proj_with_mqa.lora_B.weight": torch.randn(
                    3, 3
                ),
            }
        )


def test_kimi_lora_hooks_include_fused_mla():
    model = object.__new__(KimiK25ForConditionalGeneration)
    model.config = SimpleNamespace(
        text_config=SimpleNamespace(
            hidden_size=16,
            q_lora_rank=4,
            kv_lora_rank=5,
            qk_rope_head_dim=3,
            qk_nope_head_dim=7,
            num_attention_heads=2,
            v_head_dim=8,
        )
    )
    model.language_model = SimpleNamespace()

    module_name = "language_model.model.layers.0.self_attn.fused_qkv_a_proj_with_mqa"

    assert model.should_apply_lora(module_name)
    assert model.get_hidden_dim("fused_qkv_a_proj_with_mqa", 0) == (16, 12)
    assert model.get_hidden_dim("q_a_proj", 0) == (16, 4)
    assert model.get_hidden_dim("kv_a_proj_with_mqa", 0) == (16, 8)


class _FakeSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.fused_qkv_a_proj_with_mqa = ReplicatedLinear(4, 5, bias=False)


class _FakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _FakeSelfAttention()


class _FakeInnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_FakeDecoderLayer()])


class _FakeLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _FakeInnerModel()


class _FakeKimiModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _FakeLanguageModel()

    def should_apply_lora(self, module_name: str) -> bool:
        return module_name.endswith("fused_qkv_a_proj_with_mqa")

    def get_hidden_dim(self, module_name: str, layer_idx: int):
        dims = {
            "q_a_proj": (4, 2),
            "kv_a_proj_with_mqa": (4, 3),
            "fused_qkv_a_proj_with_mqa": (4, 5),
        }
        return dims[module_name]


def test_manager_wraps_fused_mla_and_loads_peft_branch_buffers():
    torch.manual_seed(0)
    base_model = _FakeKimiModel()
    server_args = SimpleNamespace(
        enable_lora_overlap_loading=False,
        lora_eviction_policy="lru",
        max_lora_chunk_size=16,
    )
    manager = LoRAManager(
        base_model=base_model,
        base_hf_config=SimpleNamespace(num_hidden_layers=1, hidden_size=4),
        max_loras_per_batch=2,
        load_config=SimpleNamespace(),
        dtype=torch.float32,
        server_args=server_args,
        lora_backend="torch_native",
        max_lora_rank=2,
        target_modules=["q_a_proj", "kv_a_proj_with_mqa"],
    )

    wrapped = base_model.language_model.model.layers[0].self_attn.fused_qkv_a_proj_with_mqa
    assert isinstance(wrapped, FusedQKVAProjWithMQALoRA)

    tensors = {
        "base_model.model.language_model.model.layers.0.self_attn.q_a_proj.lora_A.weight": torch.randn(
            2, 4
        ),
        "base_model.model.language_model.model.layers.0.self_attn.q_a_proj.lora_B.weight": torch.randn(
            2, 2
        ),
        "base_model.model.language_model.model.layers.0.self_attn.kv_a_proj_with_mqa.lora_A.weight": torch.randn(
            2, 4
        ),
        "base_model.model.language_model.model.layers.0.self_attn.kv_a_proj_with_mqa.lora_B.weight": torch.randn(
            3, 2
        ),
    }
    ref = LoRARef(
        lora_id="adapter",
        lora_name="adapter",
        lora_path="/tmp/adapter",
        pinned=False,
    )
    result = manager.load_lora_adapter_from_tensors(
        ref,
        tensors,
        {
            "peft_type": "LORA",
            "r": 2,
            "lora_alpha": 2,
            "target_modules": ["q_a_proj", "kv_a_proj_with_mqa"],
        },
    )
    assert result.success, result.error_message

    manager.fetch_new_loras({"adapter"})
    buffer_id = manager.memory_pool.get_buffer_id("adapter")

    torch.testing.assert_close(
        manager.memory_pool.A_buffer["q_a_proj"][0][buffer_id, :2],
        tensors[
            "base_model.model.language_model.model.layers.0.self_attn.q_a_proj.lora_A.weight"
        ],
    )
    torch.testing.assert_close(
        manager.memory_pool.B_buffer["kv_a_proj_with_mqa"][0][buffer_id, :, :2],
        tensors[
            "base_model.model.language_model.model.layers.0.self_attn.kv_a_proj_with_mqa.lora_B.weight"
        ],
    )
