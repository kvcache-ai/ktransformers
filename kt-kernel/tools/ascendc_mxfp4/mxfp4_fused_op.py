"""Runtime wrapper for the fused AscendC MXFP4->W8A8 kernel, for kt_stream_prefill depool.

Builds libmxfp4fused.so on first use (bisheng), loads it via ctypes, and exposes:

  mxfp4_layer_to_nz_slots(c13, s13, c2, s2, H, I, blockdim=40)
      -> (w13_nz, s13b, w2_nz, s2b)   # exactly the slot tensors npu_fused_experts consumes
                                       # w*_nz: FRACTAL_NZ int8 [E,IN,OUT];  s*b: bf16 [E,OUT]

Inputs are this layer's combined MXFP4 (device uint8):
  c13/s13: w13 = cat(w1,w3) codes [E,2I,H/2] + e8m0 scale [E,2I,H/32]
  c2/s2  : w2  codes [E,H,I/2] + e8m0 scale [E,H,I/32]

Reads MXFP4 once; one kernel pass per projection. Validated end-to-end (cos 0.99999976 vs fp32
golden through the NPU grouped-matmul MoE path).

Point sglang's ``kt_stream_prefill`` at this directory with ``KT_MXFP4_OP_DIR``.
"""

import ctypes
import os
import subprocess
import threading
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "mxfp4_fused_kernel.cpp"
_NZ = 29
_ACC = 512
_FP4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)

# The fused kernel is launched via a raw ctypes <<<stream>>> call. With TASK_QUEUE_ENABLE=1 torch
# dispatches its ops through an async queue that is NOT ordered against that direct launch, so the
# post-step racing `out`/`osc` needs an explicit per-chunk host sync (correctness; but it stalls the
# host and serializes the convert with the rest of the forward -> slow prefill). With
# TASK_QUEUE_ENABLE=0 torch ops go straight to the stream, so kernel+post-step are FIFO-ordered and
# NO sync is needed (validated: deterministic + byte-equal, runs async -> fast). So only sync when
# the task queue is on.
_TQ_SYNC = os.environ.get("TASK_QUEUE_ENABLE", "1") != "0"

_lib = None
_lock = threading.Lock()
_consts_cache = {}


def _cann_home():
    # ASCEND_TOOLKIT_HOME is exported by the CANN set_env.sh; the fallback is the
    # version-independent install symlink so no CANN release is hardcoded here.
    return os.environ.get("ASCEND_TOOLKIT_HOME", "/usr/local/Ascend/ascend-toolkit/latest")


def _so_path():
    """Where to build/cache libmxfp4fused.so.

    The source directory is preferred (keeps the .so next to the kernel, survives
    across runs), but it is read-only in some container mounts and in a pip-installed
    tree, so fall back to a user cache directory. Override with KT_MXFP4_SO_DIR.
    """
    override = os.environ.get("KT_MXFP4_SO_DIR")
    if override:
        d = Path(override)
    elif os.access(_HERE, os.W_OK):
        d = _HERE
    else:
        d = (
            Path(os.environ.get("SGLANG_CACHE_DIR") or os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
            / "ascendc_mxfp4"
        )
    d.mkdir(parents=True, exist_ok=True)
    return d / "libmxfp4fused.so"


def _build(so: Path):
    cann = _cann_home()
    tk = f"{cann}/aarch64-linux/tikcpp"
    inc = [
        f"{tk}/tikcfw",
        f"{tk}/tikcfw/impl",
        f"{tk}/tikcfw/interface",
        f"{tk}/tikcfw/lib",
        f"{cann}/aarch64-linux/include",
    ]
    cmd = [
        "bisheng",
        "-x",
        "asc",
        "--cce-aicore-arch=dav-c220",
        "-O2",
        "-std=c++17",
        "-fPIC",
        "-shared",
        *[f"-I{p}" for p in inc],
        str(_SRC),
        "-o",
        str(so),
        f"-L{cann}/aarch64-linux/lib64",
        "-lruntime",
        "-lascendcl",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"bisheng failed to build {_SRC.name} (exit {proc.returncode}).\n"
            f"  command: {' '.join(cmd)}\n"
            f"  ASCEND_TOOLKIT_HOME={cann}\n"
            f"  stderr: {proc.stderr.decode('utf-8', 'replace')[-2000:]}"
        )


def get_lib():
    """Build (if needed) and load the fused kernel .so. Thread-safe, idempotent."""
    global _lib
    if _lib is not None:
        return _lib
    with _lock:
        if _lib is None:
            so = _so_path()
            if not so.exists() or so.stat().st_mtime < _SRC.stat().st_mtime:
                _build(so)
            lib = ctypes.CDLL(str(so))
            lib.launch_mxfp4_fused.restype = None
            lib.launch_mxfp4_fused.argtypes = (
                [ctypes.c_void_p, ctypes.c_uint32] + [ctypes.c_void_p] * 8 + [ctypes.c_uint32] * 4
            )
            lib.launch_mxfp4_fused_blk.restype = None
            lib.launch_mxfp4_fused_blk.argtypes = (
                [ctypes.c_void_p, ctypes.c_uint32] + [ctypes.c_void_p] * 9 + [ctypes.c_uint32] * 4
            )
            _lib = lib
    return _lib


def _consts(HALF, NB, dev):
    key = (HALF, NB, str(dev))
    if key in _consts_cache:
        return _consts_cache[key]
    b = np.arange(256, dtype=np.int64)
    lutLo = _FP4[b & 0xF].astype(np.float32)
    lutHi = _FP4[(b >> 4) & 0xF].astype(np.float32)
    lutE8 = ((b.astype(np.uint32)) << 23).view(np.float32).astype(np.float32)
    j = np.arange(HALF, dtype=np.int64)
    scOff = ((j >> 4) * 4).astype(np.int32)
    out = tuple(torch.from_numpy(a).to(dev) for a in (lutLo, lutHi, lutE8, scOff))
    _consts_cache[key] = out
    return out


_blk_consts_cache = {}


def _blk_consts(HALF, NB, dev):
    """codeOff/scaleOff: byte offsets of code j / scale block b in the half-cast GGUF block buffer
    ([nb,17] per row). Used by mxfp4_fused_blk to de-interleave in UB via Gather."""
    key = (HALF, NB, str(dev))
    if key in _blk_consts_cache:
        return _blk_consts_cache[key]
    j = np.arange(HALF, dtype=np.int64)
    codeOff = (((j // 16) * 17 + 1 + (j % 16)) * 2).astype(np.uint32)
    b = np.arange(NB, dtype=np.int64)
    scaleOff = ((b * 17) * 2).astype(np.uint32)
    out = (torch.from_numpy(codeOff).to(dev), torch.from_numpy(scaleOff).to(dev))
    _blk_consts_cache[key] = out
    return out


_NZ_CHUNK = int(os.environ.get("KT_MXFP4_NZ_CHUNK", "32"))  # experts/chunk -> bounds HBM transient


def convert_proj(codes_dev, scale_dev, IN, blockdim=40, packing="consecutive", out_nz=None):
    """One projection: MXFP4 codes/scale [E,OUT,*] -> (q_nz [E,IN,OUT] FRACTAL_NZ, oscale bf16 [E,OUT]).

    Chunked over experts so the transient (int8 planes + de-interleave + NZ cast) stays small —
    only the final [E,IN,OUT] NZ output is full-size (HBM-bounded like the W8A8 slot).

    out_nz: optional pre-allocated FRACTAL_NZ [E,IN,OUT] int8 buffer to write into (the reserved
      streaming slot). When given, no per-call ~GBs output allocation happens — the layer's NZ is
      produced straight into the reused slot (HBM budgeted once at load). Must already be NZ-format
      with matching shape. When None, a fresh buffer is allocated (back-compat).

    packing: nibble layout of the code bytes — how the kernel's lo/hi planes map back to K-positions.
      "consecutive" (native safetensors): byte j -> Kpos 2j (lo), 2j+1 (hi)  -> interleave.
      "halfblock"   (GGUF block_mxfp4):   byte j -> Kpos g*32+jl (lo), +16 (hi) within its 32-group
                                           -> per-group [lo0..15 | hi0..15] concat.
    Both decode the SAME K-ordered weights bit-for-bit; the kernel and scale->block mapping (scOff)
    are packing-agnostic, so only this post-step rearrange differs (no .so change)."""
    import torch_npu

    lib = get_lib()
    dev = codes_dev.device
    E, OUT, HALF = codes_dev.shape
    NB = scale_dev.shape[2]
    HALFp = IN // 2
    lutLo, lutHi, lutE8, scOff = _consts(HALF, NB, dev)
    st = torch.npu.current_stream().npu_stream
    P = lambda t: ctypes.c_void_p(t.data_ptr())

    oscale = torch.empty((E, OUT), dtype=torch.bfloat16, device=dev)
    for c in range(0, E, _NZ_CHUNK):
        ce = min(c + _NZ_CHUNK, E)
        Ec = ce - c
        Rc = Ec * OUT
        cd = codes_dev[c:ce].reshape(Rc, HALF).contiguous()
        sd = scale_dev[c:ce].reshape(Rc, NB).contiguous()
        out = torch.empty((Rc, IN), dtype=torch.int8, device=dev)  # two planes [lo|hi]
        Rp = (Rc + _ACC - 1) // _ACC * _ACC
        osc = torch.empty((Rp,), dtype=torch.float32, device=dev)
        lib.launch_mxfp4_fused(
            ctypes.c_void_p(st),
            blockdim,
            P(cd),
            P(sd),
            P(out),
            P(osc),
            P(lutLo),
            P(lutHi),
            P(lutE8),
            P(scOff),
            Rc,
            HALF,
            NB,
            IN,
        )
        # See _TQ_SYNC: only needed when the task queue is on (then the raw ctypes launch is not
        # ordered against the torch post-step reading `out`/`osc`). With it off, stream FIFO orders
        # them and we skip the host stall -> async convert, fast prefill.
        if _TQ_SYNC:
            torch.npu.synchronize()
        # De-interleave the [lo|hi] planes (contiguous stack) then transpose OUT<->IN. The old depool
        # hot spot was (a) a strided 1-byte de-interleave scatter (~2.4s/layer) and (b) an int8
        # transpose that degenerates to a 1-byte gather (~0.6s, ~20GB/s). (a) is gone via the
        # contiguous stack; (b) is killed by transposing in fp16 (vectorized) and round-tripping
        # int8->fp16->int8 — exact because |q|<=127. Net post-step ~3s -> ~0.13s. The .contiguous()
        # is mandatory: feeding a transposed view to format_cast lays down WRONG NZ bytes on device
        # (looks fine via .cpu() which de-formats, but grouped_matmul reads garbage).
        lo, hi = out[:, :HALFp], out[:, HALFp:]
        if packing == "halfblock":
            nb = HALFp // 16
            q = torch.cat([lo.reshape(Rc, nb, 16), hi.reshape(Rc, nb, 16)], dim=2).reshape(Ec, OUT, IN)
        else:
            q = torch.stack([lo, hi], dim=2).reshape(Ec, OUT, IN)  # consecutive interleave [E,OUT,IN]
        nd = q.to(torch.float16).transpose(1, 2).contiguous().to(torch.int8)  # [E,IN,OUT]
        nz = torch_npu.npu_format_cast(nd, _NZ)
        if out_nz is None:
            out_nz = torch.empty((E,) + tuple(nz.shape[1:]), dtype=torch.int8, device=dev)
        out_nz[c:ce].copy_(nz)
        oscale[c:ce] = osc[:Rc].reshape(Ec, OUT).to(torch.bfloat16)
        # Second sync (task-queue-on only): let the osc read finish before the next chunk reuses it.
        if _TQ_SYNC:
            torch.npu.synchronize()
        del out, q, nd, nz, osc, cd, sd
    return out_nz, oscale


def mxfp4_layer_to_nz_slots(c13, s13, c2, s2, H, I, blockdim=40, packing="consecutive", out_w13=None, out_w2=None):
    """Full layer depool conversion -> (w13_nz, s13b, w2_nz, s2b), the exact tensors the streaming
    slot + npu_fused_experts consume (replacing the resident W8A8 pool). packing: see convert_proj
    ("consecutive" for native safetensors codes, "halfblock" for GGUF block_mxfp4 codes).
    out_w13/out_w2: optional pre-reserved NZ slots to convert into (no per-layer output alloc)."""
    w13_nz, s13b = convert_proj(c13, s13, H, blockdim, packing, out_nz=out_w13)
    w2_nz, s2b = convert_proj(c2, s2, I, blockdim, packing, out_nz=out_w2)
    return w13_nz, s13b, w2_nz, s2b


def convert_proj_blk(blocks_dev, IN, blockdim=40, out_nz=None):
    """One projection from RAW GGUF block_mxfp4 [E,OUT,nb*17] -> (q_nz [E,IN,OUT] FRACTAL_NZ, oscale
    bf16 [E,OUT]). The de-interleave (scale|codes per 17B block) is done IN-KERNEL (mxfp4_fused_blk,
    UB Gather) -- no host/device de-interleave (the slow 16-of-17 strided int8 copy). The kernel
    output `out` (two [lo|hi] planes) is byte-identical to the de-interleaved path, so the post-step
    is the same half-block rearrange. out_nz: optional pre-reserved NZ buffer."""
    import torch_npu

    lib = get_lib()
    dev = blocks_dev.device
    E, OUT, NB17 = blocks_dev.shape
    nb = NB17 // 17
    HALF = nb * 16
    HALFp = IN // 2
    lutLo, lutHi, lutE8, scOff = _consts(HALF, nb, dev)
    codeOff, scaleOff = _blk_consts(HALF, nb, dev)
    st = torch.npu.current_stream().npu_stream
    P = lambda t: ctypes.c_void_p(t.data_ptr())
    oscale = torch.empty((E, OUT), dtype=torch.bfloat16, device=dev)
    for c in range(0, E, _NZ_CHUNK):
        ce = min(c + _NZ_CHUNK, E)
        Ec = ce - c
        Rc = Ec * OUT
        bd = blocks_dev[c:ce].reshape(Rc, NB17).contiguous()
        out = torch.empty((Rc, IN), dtype=torch.int8, device=dev)
        Rp = (Rc + _ACC - 1) // _ACC * _ACC
        osc = torch.empty((Rp,), dtype=torch.float32, device=dev)
        lib.launch_mxfp4_fused_blk(
            ctypes.c_void_p(st),
            blockdim,
            P(bd),
            P(out),
            P(osc),
            P(lutLo),
            P(lutHi),
            P(lutE8),
            P(scOff),
            P(codeOff),
            P(scaleOff),
            Rc,
            HALF,
            nb,
            IN,
        )
        if _TQ_SYNC:
            torch.npu.synchronize()
        lo, hi = out[:, :HALFp], out[:, HALFp:]
        nbb = HALFp // 16
        q = torch.cat([lo.reshape(Rc, nbb, 16), hi.reshape(Rc, nbb, 16)], dim=2).reshape(Ec, OUT, IN)
        nd = q.to(torch.float16).transpose(1, 2).contiguous().to(torch.int8)
        nz = torch_npu.npu_format_cast(nd, _NZ)
        if out_nz is None:
            out_nz = torch.empty((E,) + tuple(nz.shape[1:]), dtype=torch.int8, device=dev)
        out_nz[c:ce].copy_(nz)
        oscale[c:ce] = osc[:Rc].reshape(Ec, OUT).to(torch.bfloat16)
        if _TQ_SYNC:
            torch.npu.synchronize()
        del out, q, nd, nz, osc, bd
    return out_nz, oscale


def mxfp4_layer_to_nz_slots_blk(blk13, blk2, H, I, blockdim=40, out_w13=None, out_w2=None):
    """Full layer conversion from RAW GGUF blocks (in-kernel de-interleave) -> slot tensors.
    blk13 = cat(gate,up) blocks [E,2I,nbH*17]; blk2 = down blocks [E,H,nbI*17]."""
    w13_nz, s13b = convert_proj_blk(blk13, H, blockdim, out_nz=out_w13)
    w2_nz, s2b = convert_proj_blk(blk2, I, blockdim, out_nz=out_w2)
    return w13_nz, s13b, w2_nz, s2b
