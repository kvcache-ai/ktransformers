import torch
import torch.nn as nn
import pytest

# 按你实际代码位置改路径：
from ktransformers.operators.ascend.ascend_layernorm import KDeepseekV3RMSNormW8A8
import ktransformers.util.utils as utils_mod

torch_npu = pytest.importorskip("torch_npu")


# ==========================
# Dummy 依赖
# ==========================

class DummyOrigModule(nn.Module):
    def __init__(self, hidden_size=4, variance_epsilon=1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = variance_epsilon


class DummySafeTensorLoader:
    def __init__(self):
        self.tensors = {}
        self.load_calls = []

    def load_tensor(self, name: str):
        self.load_calls.append(name)
        return self.tensors[name]


class DummyGGUFLoader:
    def __init__(self, safetensor_loader: DummySafeTensorLoader):
        self.safetensor_loader = safetensor_loader


class DummyConfig:
    pass


class FakeRMSNorm:
    def __init__(self):
        self.last_args = None

    def __call__(self, hidden_states, weight, eps):
        self.last_args = (hidden_states, weight, eps)

        out = hidden_states * weight
        return (out,)


def build_rms_module(hidden_size=4, eps=1e-5, safetensor_loader=None):
    orig = DummyOrigModule(hidden_size=hidden_size, variance_epsilon=eps)
    if safetensor_loader is None:
        safetensor_loader = DummySafeTensorLoader()
    gguf_loader = DummyGGUFLoader(safetensor_loader)
    config = DummyConfig()
    module = KDeepseekV3RMSNormW8A8(
        key="rms",
        gguf_loader=gguf_loader,
        config=config,
        orig_module=orig,
        prefill_device="npu",
        generate_device="npu",
    )
    return module, safetensor_loader, orig

@pytest.fixture(autouse=True)
def patch_utils_and_npu(monkeypatch):
    monkeypatch.setattr(utils_mod, "get_current_device", lambda: "cpu", raising=False)

    fake = FakeRMSNorm()
    monkeypatch.setattr(torch_npu, "npu_rms_norm", fake, raising=False)

    import sys
    sys.modules[__name__]._fake_rms = fake

    yield

def get_fake_rms():
    import sys
    return sys.modules[__name__]._fake_rms

def test_forward_preserves_shape_and_dtype():
    hidden_size = 4
    module, _, orig = build_rms_module(hidden_size=hidden_size, eps=1e-6)

    x = torch.randn(2, 3, hidden_size, dtype=torch.float16)

    out = module(x)

    assert out.shape == x.shape
    assert out.dtype == x.dtype

    fake_rms = get_fake_rms()
    hs_arg, w_arg, eps_arg = fake_rms.last_args
    assert hs_arg is x
    assert w_arg is module.weight
    assert eps_arg == orig.variance_epsilon


def test_forward_with_bfloat16_dtype():
    hidden_size = 4
    module, _, _ = build_rms_module(hidden_size=hidden_size, eps=1e-6)

    x = torch.randn(1, 2, hidden_size, dtype=torch.bfloat16)
    out = module(x)

    assert out.shape == x.shape
    assert out.dtype == torch.bfloat16


def test_forward_uses_bias():
    hidden_size = 4
    module, _, _ = build_rms_module(hidden_size=hidden_size, eps=1e-6)

    module.weight.data = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    module.bias.data = torch.tensor([-1.0, 0.5, 0.0, 2.0], dtype=torch.float32)

    x = torch.arange(2 * 3 * hidden_size, dtype=torch.float16).view(2, 3, hidden_size)

    out = module(x)

    expected_rms = x.to(torch.float32) * module.weight
    expected = expected_rms + module.bias

    assert torch.allclose(out, expected.to(out.dtype))



def test_load_from_safetensor_loader():
    hidden_size = 4
    module, safe_loader, _ = build_rms_module(hidden_size=hidden_size, eps=1e-5)

    w_loaded = torch.arange(hidden_size, dtype=torch.float32)
    b_loaded = torch.full((hidden_size,), 3.0, dtype=torch.float32)

    safe_loader.tensors["rms.weight"] = w_loaded
    safe_loader.tensors["rms.bias"] = b_loaded

    module.load()

    assert torch.allclose(module.weight, w_loaded)
    assert torch.allclose(module.bias, b_loaded)

    assert safe_loader.load_calls == ["rms.weight", "rms.bias"]


def test_unload_sets_weight_and_bias_to_none_idempotent():
    module, _, _ = build_rms_module(hidden_size=4, eps=1e-5)

    assert module.weight is not None
    assert module.bias is not None

    module.unload()
    assert module.weight is None
    assert module.bias is None

    module.unload()
    assert module.weight is None
    assert module.bias is None

