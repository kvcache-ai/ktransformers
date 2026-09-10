# SPDX-License-Identifier: Apache-2.0
"""Real FP8/BF16 kernels: prefetch, shared-owner replacement, reload and gradients.

Requires an AVX512-BF16/VBMI extension. No GPU or model checkpoint is needed.
"""

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture
def extension(monkeypatch):
    monkeypatch.setenv("KT_SFT_PROFILE", "1")
    kernel = pytest.importorskip("kt_kernel").kt_kernel_ext
    if not hasattr(kernel.moe, "AMXFP8_SFT_MOE"):
        pytest.skip("AVX512-BF16/VBMI SFT extension required")
    return kernel


def make_pool(extension, tp_count, threads=4):
    config = extension.WorkerPoolConfig()
    config.subpool_count = tp_count
    # TP2 correctness also runs on one socket; perf runs can choose real NUMA IDs.
    config.subpool_numa_map = [0] * tp_count
    config.subpool_thread_count = [threads] * tp_count
    return extension.CPUInfer(config)


class Layer:
    """Own all tensors whose raw pointers are borrowed by a native SFT operator."""

    def __init__(
        self,
        extension,
        pool,
        dtype,
        seed,
        *,
        experts=3,
        hidden=256,
        intermediate=512,
        qlen=17,
        layer_idx=0,
        keep_reference=True,
    ):
        self.pool = pool
        self.dtype = dtype
        self.experts, self.hidden, self.intermediate, self.qlen = (
            experts,
            hidden,
            intermediate,
            qlen,
        )
        self.rank = 8
        generator = torch.Generator().manual_seed(seed)

        def randn(shape, scale):
            return (torch.randn(shape, generator=generator) * scale).to(torch.bfloat16).contiguous()

        shapes = {
            "gate": (experts, intermediate, hidden),
            "up": (experts, intermediate, hidden),
            "down": (experts, hidden, intermediate),
        }
        self.weights, self.scales, self.reference_weights = {}, {}, {}
        for name, shape in shapes.items():
            weights = randn(shape, 0.05)
            if dtype == "fp8":
                weights = weights.to(torch.float8_e4m3fn)
                scales = torch.linspace(0.5, 1.5, experts * (shape[1] // 128) * (shape[2] // 128))
                scales = scales.reshape(experts, shape[1] // 128, shape[2] // 128).contiguous()
                self.scales[name] = scales
                if keep_reference:
                    self.reference_weights[name] = weights.float() * scales.repeat_interleave(128, 1).repeat_interleave(
                        128, 2
                    )
            elif keep_reference:
                self.reference_weights[name] = weights.float()
            self.weights[name] = weights

        r = self.rank
        self.lora = {
            name: randn(shape, 0.02)
            for name, shape in {
                "gate_lora_a": (experts, r, hidden),
                "gate_lora_b": (experts, intermediate, r),
                "up_lora_a": (experts, r, hidden),
                "up_lora_b": (experts, intermediate, r),
                "down_lora_a": (experts, r, intermediate),
                "down_lora_b": (experts, hidden, r),
            }.items()
        }
        self.mapping = torch.arange(experts, dtype=torch.int64)
        config = extension.moe.MOESFTConfig(experts, 1, hidden, intermediate)
        # Deliberately identical layer numbers: readiness must use object/version identity.
        config.layer_idx = layer_idx
        config.max_len = max(32, qlen)
        config.max_cache_depth = 1
        config.lora_rank = r
        config.lora_alpha = float(r)
        config.full_weight_grad = False
        config.share_backward_bb = True
        config.share_cache_pool = False
        config.physical_to_logical_map = self.mapping.data_ptr()
        config.pool = pool.backend_
        for name, weight in self.weights.items():
            setattr(config, name + "_proj", weight.data_ptr())
        for name, scale in self.scales.items():
            setattr(config, name + "_scale", scale.data_ptr())
        for name, weight in self.lora.items():
            setattr(config, name, weight.data_ptr())
        if dtype == "fp8":
            config.quant_config.group_size = 128
            config.quant_config.zero_point = False
        cls = extension.moe.AMXFP8_SFT_MOE if dtype == "fp8" else extension.moe.AMXBF16_SFT_MOE
        self.moe = cls(config)
        self.moe.load_weights()
        self.inputs = randn((qlen, hidden), 0.2)
        self.grad_output = randn((qlen, hidden), 0.1)
        self.routes = (torch.arange(qlen, dtype=torch.int64) % experts).reshape(qlen, 1).contiguous()
        self.route_weights = torch.linspace(0.5, 0.9, qlen).reshape(qlen, 1).contiguous()

    def forward(self):
        qlen = torch.tensor([self.qlen], dtype=torch.int32)
        output = torch.empty_like(self.inputs)
        self.moe.forward_sft(
            qlen.data_ptr(),
            1,
            self.routes.data_ptr(),
            self.route_weights.data_ptr(),
            self.inputs.data_ptr(),
            output.data_ptr(),
            True,
        )
        return output

    def backward(self):
        grad_input = torch.empty_like(self.inputs)
        grad_routes = torch.empty_like(self.route_weights)
        grads = {name: torch.zeros_like(value) for name, value in self.lora.items()}
        self.moe.backward(
            self.grad_output.data_ptr(),
            grad_input.data_ptr(),
            *(value.data_ptr() for value in grads.values()),
            grad_routes.data_ptr(),
            0,
            0,
            0,
        )
        return (grad_input, grad_routes, *grads.values())

    def repack_calls(self):
        stats = self.moe.get_profile_stats()
        return sum(
            value
            for name, value in stats.items()
            if name.startswith("tp.") and name.endswith("weights.backward_repack.calls")
        )

    def reference(self):
        inputs = self.inputs.float().requires_grad_()
        routes = self.route_weights.float().requires_grad_()
        lora = {name: value.float().requires_grad_() for name, value in self.lora.items()}
        output = torch.zeros_like(inputs)
        for expert in range(self.experts):
            rows = (self.routes[:, 0] == expert).nonzero().flatten()
            x = inputs[rows]

            def projection(name, x):
                return F.linear(x, self.reference_weights[name][expert]) + F.linear(
                    F.linear(x, lora[name + "_lora_a"][expert]),
                    lora[name + "_lora_b"][expert],
                )

            gate, up = projection("gate", x), projection("up", x)
            y = projection("down", F.silu(gate) * up) * routes[rows]
            output = output.index_add(0, rows, y)
        grads = torch.autograd.grad(output, (inputs, routes, *lora.values()), self.grad_output.float())
        return output.detach(), grads


def assert_numerically_close(actual, expected):
    assert torch.isfinite(actual).all()
    relative_l2 = (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-10)
    assert relative_l2 < 0.04, f"relative L2 error: {relative_l2.item():.6f}"


@pytest.mark.parametrize("tp_count", [1, 2])
@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
@pytest.mark.parametrize("experts,threads", [(3, 4), (1, 8)])
@pytest.mark.parametrize("qlen", [17, 65, 257])
def test_prefetch_matches_synchronous_backward(extension, tp_count, dtype, experts, threads, qlen):
    pool = make_pool(extension, tp_count, threads)
    layers = [Layer(extension, pool, dtype, seed, experts=experts, qlen=qlen) for seed in (17, 29)]
    outputs = [layer.forward() for layer in layers]
    serial = [layer.backward() for layer in reversed(layers)][::-1]
    for layer, output, grads in zip(layers, outputs, serial):
        ref_output, ref_grads = layer.reference()
        assert_numerically_close(output, ref_output)
        for actual, expected in zip(grads, ref_grads):
            assert_numerically_close(actual, expected)

    for layer, expected in zip(layers, outputs):
        torch.testing.assert_close(layer.forward(), expected, rtol=0, atol=0)
    layers[1].backward()
    before = layers[0].repack_calls()
    layers[0].moe.submit_backward_repack()
    layers[0].moe.wait_backward_repack()
    ready = layers[0].repack_calls()
    assert ready == before + tp_count, "prefetch must actually repack FP8 as well as BF16"
    actual = layers[0].backward()
    assert layers[0].repack_calls() == ready, "consumer repacked already-ready weights"
    for got, expected in zip(actual, serial[0]):
        torch.testing.assert_close(got, expected, rtol=0, atol=0)


@pytest.mark.parametrize("tp_count", [1, 2])
def test_fp8_failed_consumer_releases_execution(extension, tp_count):
    pool = make_pool(extension, tp_count)
    layer = Layer(extension, pool, "fp8", 53, qlen=65)
    with pytest.raises(RuntimeError, match="SFT backward failed on TP/NUMA.*Forward cache stack underflow"):
        layer.backward()
    output = layer.forward()
    actual = layer.backward()
    ref_output, reference = layer.reference()
    assert_numerically_close(output, ref_output)
    for got, expected in zip(actual, reference):
        assert_numerically_close(got, expected)


@pytest.mark.parametrize("tp_count", [1, 2])
def test_shared_slot_replacement_across_dtypes(extension, tp_count):
    pool = make_pool(extension, tp_count)
    fp8 = Layer(extension, pool, "fp8", 31)
    bf16 = Layer(extension, pool, "bf16", 43)
    fp8.forward()
    expected = fp8.backward()
    fp8.forward()
    fp8.moe.submit_backward_repack()
    fp8.moe.wait_backward_repack()
    # BF16 grows/replaces the same slot, with the same layer_idx as FP8.
    bf16.moe.submit_backward_repack()
    bf16.moe.wait_backward_repack()
    before = fp8.repack_calls()
    actual = fp8.backward()
    assert fp8.repack_calls() == before + tp_count
    for got, reference in zip(actual, expected):
        torch.testing.assert_close(got, reference, rtol=0, atol=0)


def test_bf16_reload_invalidates_prepared_weights(extension):
    pool = make_pool(extension, 1)
    layer = Layer(extension, pool, "bf16", 47)
    layer.forward()
    layer.backward()
    for name, weight in layer.weights.items():
        weight.mul_(1.5)
        layer.reference_weights[name] = weight.float()
    layer.moe.set_base_weight_pointers(*(layer.weights[name].data_ptr() for name in ("gate", "up", "down")))
    layer.moe.load_weights()
    output = layer.forward()
    actual = layer.backward()
    ref_output, reference = layer.reference()
    assert_numerically_close(output, ref_output)
    for got, expected in zip(actual, reference):
        assert_numerically_close(got, expected)
