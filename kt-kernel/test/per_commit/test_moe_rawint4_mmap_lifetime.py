"""Exercise NativeMoEWrapper's mmap ownership with real loaders and kernels."""

import gc
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import weakref

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=30, suite="default")

EXPERTS, HIDDEN, INTERMEDIATE, TOPK, GROUP = 8, 256, 512, 2, 32


def write_fixture(path, packing):
    import torch
    from safetensors.torch import save_file

    gen = torch.Generator().manual_seed(2218)
    tensors = {}
    for expert in range(EXPERTS):
        for proj, n, k in (
            ("gate", INTERMEDIATE, HIDDEN),
            ("up", INTERMEDIATE, HIDDEN),
            ("down", HIDDEN, INTERMEDIATE),
        ):
            key = f"model.layers.0.mlp.experts.{expert}.{proj}_proj"
            packed = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8)
            tensors[key + ".weight_packed"] = packed if packing == "uint8" else packed.view(torch.int32)
            tensors[key + ".weight_scale"] = (torch.rand((n, k // GROUP), generator=gen) * 0.02 + 0.001).bfloat16()
            tensors[key + ".weight_shape"] = torch.tensor([n, k], dtype=torch.int64)
    save_file(tensors, str(path))


def run_forwards(wrapper):
    import torch

    gen = torch.Generator().manual_seed(193)
    for qlen in (1, 16):
        ids = torch.stack([torch.randperm(EXPERTS, generator=gen)[:TOPK] for _ in range(qlen)])
        routing = torch.rand((qlen, TOPK), generator=gen)
        inp = (torch.randn((qlen, HIDDEN), generator=gen) / 100).bfloat16()
        out = torch.empty_like(inp)
        bsz = torch.tensor([qlen], dtype=torch.int32)
        reference = None
        for _ in range(21):
            gc.collect()
            wrapper.cpu_infer.submit(
                wrapper.moe.forward_task(
                    bsz.data_ptr(),
                    TOPK,
                    ids.data_ptr(),
                    routing.data_ptr(),
                    inp.data_ptr(),
                    out.data_ptr(),
                    False,
                )
            )
            wrapper.cpu_infer.sync()
            assert torch.isfinite(out.float()).all() and out.float().abs().sum() > 0
            if reference is None:
                reference = out.clone()
            else:
                assert torch.equal(out, reference)
        print(f"qlen={qlen}: 21 forwards passed", flush=True)


def run_worker(backend, packing, directory):
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    import torch
    import kt_kernel
    import kt_kernel.utils.amx as amx

    torch.set_num_threads(1)
    backend_cls = amx.AVX2RawInt4_MOE if backend == "avx2" else amx.AMXInt4_KGroup_MOE
    if backend_cls is None:
        print(f"SKIP: {backend} RAWINT4 backend is not compiled in", flush=True)
        return 77

    fixture = Path(directory) / "model.safetensors"
    write_fixture(fixture, packing)
    config = kt_kernel.kt_kernel_ext.WorkerPoolConfig()
    config.subpool_count = 1
    config.subpool_numa_map = [0]
    config.subpool_thread_count = [4]

    # Construct only the CPU state consumed by the real load_weights method.
    # The normal constructor allocates CUDA-pinned masks, unnecessary in CPU CI.
    wrapper = amx.NativeMoEWrapper.__new__(amx.NativeMoEWrapper)
    wrapper.layer_idx = 0
    wrapper.method = "RAWINT4"
    wrapper.weight_path = str(directory)
    wrapper.num_experts = EXPERTS
    wrapper.num_experts_per_tok = TOPK
    wrapper.hidden_size = HIDDEN
    wrapper.moe_intermediate_size = INTERMEDIATE
    wrapper.chunked_prefill_size = 32
    wrapper.swiglu_limit = 0.0
    wrapper.gpu_experts_mask = torch.zeros(EXPERTS, dtype=torch.bool)
    wrapper.cpu_infer = kt_kernel.kt_kernel_ext.CPUInfer(config)

    observed = []
    original_load = amx.CompressedSafeTensorLoader.load_experts

    def observe(loader, *args, **kwargs):
        weights = original_load(loader, *args, **kwargs)
        # Weakrefs and integers only: the test must not keep the mmap alive.
        observed.extend((weakref.ref(t), t.data_ptr()) for name in ("gate", "up", "down") for t in weights[name])
        assert str(fixture) in Path("/proc/self/maps").read_text()
        return weights

    identity = torch.arange(EXPERTS, dtype=torch.int64)
    with patch.object(amx.CompressedSafeTensorLoader, "load_experts", observe):
        wrapper.load_weights(identity)
    gc.collect()
    assert type(wrapper.moe) is backend_cls
    assert amx.NativeMoEWrapper._native_loader_instance is None
    assert not wrapper.loader.file_handle_map
    assert not any(hasattr(wrapper, name + "_scales") for name in ("gate", "up", "down"))
    assert len(observed) == 3 * EXPERTS
    if backend == "avx2":
        assert all(
            ref() is not None and ref().data_ptr() == ptr for ref, ptr in observed
        ), "AVX2 RAWINT4 released its borrowed source weights"
        assert str(fixture) in Path("/proc/self/maps").read_text()
    else:
        assert all(ref() is None for ref, _ in observed), "Copy backend retained unnecessary source weights"
        assert str(fixture) not in Path("/proc/self/maps").read_text()

    print(f"backend={type(wrapper.moe).__name__}, packing={packing}, source={amx.__file__}", flush=True)
    run_forwards(wrapper)
    wrapper.cpu_infer.sync()
    del wrapper
    gc.collect()
    assert all(ref() is None for ref, _ in observed), "Source weights outlived the wrapper"
    assert str(fixture) not in Path("/proc/self/maps").read_text()
    print("PASSED: loader release, forward, and wrapper cleanup", flush=True)
    return 0


class TestRawInt4MmapLifetime(unittest.TestCase):
    def run_case(self, backend, packing):
        env = dict(os.environ, KT_RAWINT4_BACKEND=backend, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        # Parent owns the directory, so a crashing worker cannot leak fixtures.
        with tempfile.TemporaryDirectory(prefix="kt-rawint4-mmap-") as directory:
            result = subprocess.run(
                [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", backend, packing, directory],
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=120,
            )
        if result.returncode == 77:
            self.skipTest(result.stdout.strip())
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("PASSED: loader release, forward, and wrapper cleanup", result.stdout)
        print(result.stdout, end="", flush=True)

    def test_avx2_uint8(self):
        self.run_case("avx2", "uint8")

    def test_avx2_int32(self):
        self.run_case("avx2", "int32")

    def test_copy_backend_releases_source_weights(self):
        self.run_case("amx", "uint8")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        sys.exit(run_worker(*sys.argv[2:]))
    unittest.main()
