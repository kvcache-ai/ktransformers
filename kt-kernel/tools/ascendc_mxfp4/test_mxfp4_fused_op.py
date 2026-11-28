"""Unit and hardware bit-exact tests for kt-kernel/tools/ascendc_mxfp4/mxfp4_fused_op.py.

Runs pure-CPU guardrail and table tests everywhere, and runs full Ascend 910B bit-exactness,
FRACTAL_NZ format, fallback, and performance verification when torch_npu is available.
"""

import os
import time
import unittest
from unittest import mock

import numpy as np
import torch

import mxfp4_fused_op as op


def _has_npu():
    try:
        import torch_npu  # noqa: F401

        return torch.npu.is_available()
    except Exception:
        return False


class TestMxfp4FusedOpHost(unittest.TestCase):
    def test_can_use_blk_nz_guardrails(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("KT_MXFP4_DIRECT_NZ", None)
            self.assertTrue(op._can_use_blk_nz(4096, 4096, 29))
            self.assertTrue(op._can_use_blk_nz(4096, 2048, 29))
            self.assertTrue(op._can_use_blk_nz(4096, 2048, None))
            # Unsupported IN dimensions must fall back
            self.assertFalse(op._can_use_blk_nz(4096, 1024, 29))
            self.assertFalse(op._can_use_blk_nz(4096, 5120, 29))
            # Unaligned OUT must fall back
            self.assertFalse(op._can_use_blk_nz(4090, 4096, 29))
            # Non-FRACTAL_NZ target format (e.g. ND=2) must fall back
            self.assertFalse(op._can_use_blk_nz(4096, 4096, 2))

        with mock.patch.dict(os.environ, {"KT_MXFP4_DIRECT_NZ": "0"}):
            self.assertFalse(op._can_use_blk_nz(4096, 4096, 29))

    def test_lut_and_blk_tables(self):
        lutLo, lutHi, lutE8, scOff, codeOff, scaleOff = op._make_lut_and_blk_tables(1024, 64)
        self.assertEqual(lutLo.shape, (256,))
        self.assertEqual(lutHi.shape, (256,))
        self.assertEqual(lutE8.shape, (256,))
        self.assertEqual(scOff.shape, (1024,))
        self.assertEqual(codeOff.shape, (1024,))
        self.assertEqual(scaleOff.shape, (64,))
        # Spot-check byte offsets in half-cast [64, 17] GGUF block buffer
        self.assertEqual(int(scaleOff[0]), 0)
        self.assertEqual(int(scaleOff[1]), 17 * 2)
        self.assertEqual(int(scaleOff[63]), 63 * 17 * 2)
        self.assertEqual(int(codeOff[0]), 1 * 2)
        self.assertEqual(int(codeOff[15]), 16 * 2)
        self.assertEqual(int(codeOff[16]), (17 + 1) * 2)
        self.assertEqual(int(codeOff[1023]), (63 * 17 + 16) * 2)
        self.assertEqual(int(scOff[0]), 0)
        self.assertEqual(int(scOff[16]), 4)
        self.assertEqual(int(scOff[1023]), 63 * 4)

    def test_default_blockdim_env_override(self):
        with mock.patch.dict(os.environ, {"KT_MXFP4_BLOCKDIM": "32"}):
            self.assertEqual(op._default_blockdim("npu:0"), 32)

    def test_default_async_chunk_is_8(self):
        if "KT_MXFP4_ASYNC_CHUNK" not in os.environ:
            self.assertEqual(op._ASYNC_CHUNK, 8)


@unittest.skipUnless(_has_npu(), "Requires Ascend NPU with torch_npu")
class TestMxfp4FusedOpNPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch_npu  # noqa: F401

        torch.npu.config.allow_internal_format = True
        cls.device = torch.device("npu:0")

    def _make_random_blocks(self, E, H, I, seed=1234):
        rng = np.random.default_rng(seed)
        b13 = rng.integers(0, 256, size=(E, 2 * I, (H // 32) * 17), dtype=np.uint8)
        b2 = rng.integers(0, 256, size=(E, H, (I // 32) * 17), dtype=np.uint8)
        # Keep E8M0 scale exponents in a realistic non-NaN/Inf range [110..135]
        b13[:, :, 0::17] = rng.integers(110, 136, size=(E, 2 * I, H // 32), dtype=np.uint8)
        b2[:, :, 0::17] = rng.integers(110, 136, size=(E, H, I // 32), dtype=np.uint8)
        t13 = torch.from_numpy(b13).pin_memory()
        t2 = torch.from_numpy(b2).pin_memory()
        return t13, t2

    def test_direct_nz_and_async_bit_exact_against_legacy(self):
        import torch_npu

        E, H, I = 32, 4096, 2048
        host_b13, host_b2 = self._make_random_blocks(E, H, I)
        dev_b13 = host_b13.to(self.device)
        dev_b2 = host_b2.to(self.device)

        # 1. Legacy reference (KT_MXFP4_DIRECT_NZ=0)
        with mock.patch.dict(os.environ, {"KT_MXFP4_DIRECT_NZ": "0"}):
            ref_w13, ref_s13, ref_w2, ref_s2 = op.mxfp4_layer_to_nz_slots_blk(dev_b13, dev_b2, H, I)
        self.assertEqual(torch_npu.get_npu_format(ref_w13), op._NZ)
        self.assertEqual(torch_npu.get_npu_format(ref_w2), op._NZ)

        # 2. Direct-NZ device path (KT_MXFP4_DIRECT_NZ=1)
        with mock.patch.dict(os.environ, {"KT_MXFP4_DIRECT_NZ": "1"}):
            dnz_w13, dnz_s13, dnz_w2, dnz_s2 = op.mxfp4_layer_to_nz_slots_blk(dev_b13, dev_b2, H, I)
        self.assertEqual(torch_npu.get_npu_format(dnz_w13), op._NZ)
        self.assertEqual(torch_npu.get_npu_format(dnz_w2), op._NZ)
        self.assertTrue(torch.equal(ref_w13.cpu(), dnz_w13.cpu()))
        self.assertTrue(torch.equal(ref_w2.cpu(), dnz_w2.cpu()))
        self.assertTrue(torch.equal(ref_s13.cpu(), dnz_s13.cpu()))
        self.assertTrue(torch.equal(ref_s2.cpu(), dnz_s2.cpu()))

        # 3. Async pinned host -> pre-allocated FRACTAL_NZ slots
        slot13 = torch_npu.npu_format_cast(torch.empty((E, H, 2 * I), dtype=torch.int8, device=self.device), op._NZ)
        slot2 = torch_npu.npu_format_cast(torch.empty((E, I, H), dtype=torch.int8, device=self.device), op._NZ)
        with mock.patch.dict(os.environ, {"KT_MXFP4_DIRECT_NZ": "1"}):
            t0 = time.perf_counter()
            aw13, as13, aw2, as2 = op.mxfp4_layer_to_nz_slots_blk_async(host_b13, host_b2, H, I, slot13, slot2)
            torch.npu.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0
        self.assertIs(aw13, slot13)
        self.assertIs(aw2, slot2)
        self.assertEqual(torch_npu.get_npu_format(aw13), op._NZ)
        self.assertEqual(torch_npu.get_npu_format(aw2), op._NZ)
        self.assertTrue(torch.equal(ref_w13.cpu(), aw13.cpu()))
        self.assertTrue(torch.equal(ref_w2.cpu(), aw2.cpu()))
        self.assertTrue(torch.equal(ref_s13.cpu(), as13.cpu()))
        self.assertTrue(torch.equal(ref_s2.cpu(), as2.cpu()))
        print(f"[NPU TEST] E={E} H={H} I={I} async direct-NZ bit-exact PASS ({dt_ms:.2f} ms)", flush=True)

    def test_unsupported_dims_fallback_gracefully(self):
        import torch_npu

        E, H, I = 8, 1024, 1024
        host_b13, host_b2 = self._make_random_blocks(E, H, I, seed=5678)
        slot13 = torch_npu.npu_format_cast(torch.empty((E, H, 2 * I), dtype=torch.int8, device=self.device), op._NZ)
        slot2 = torch_npu.npu_format_cast(torch.empty((E, I, H), dtype=torch.int8, device=self.device), op._NZ)
        with mock.patch.dict(os.environ, {"KT_MXFP4_DIRECT_NZ": "1"}):
            w13, s13, w2, s2 = op.mxfp4_layer_to_nz_slots_blk_async(host_b13, host_b2, H, I, slot13, slot2)
        self.assertEqual(torch_npu.get_npu_format(w13), op._NZ)
        self.assertEqual(torch_npu.get_npu_format(w2), op._NZ)
        self.assertEqual(tuple(w13.shape), (E, H, 2 * I))
        self.assertEqual(tuple(w2.shape), (E, I, H))
        self.assertEqual(tuple(s13.shape), (E, 2 * I))
        self.assertEqual(tuple(s2.shape), (E, H))


if __name__ == "__main__":
    unittest.main()
