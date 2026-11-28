"""Runtime wrapper for the fused AscendC MXFP4->W8A8 kernels, for kt_stream_prefill depool.

Builds libmxfp4fused.so on first use (bisheng), loads it via ctypes, and exposes:

  mxfp4_layer_to_nz_slots(c13, s13, c2, s2, H, I, blockdim=None)
      -> (w13_nz, s13b, w2_nz, s2b)   # native safetensors / de-interleaved codes+scale

  mxfp4_layer_to_nz_slots_blk(blk13, blk2, H, I, blockdim=None, out_w13=None, out_w2=None)
      -> (w13_nz, s13b, w2_nz, s2b)   # raw device GGUF block_mxfp4 -> FRACTAL_NZ slots
                                      # automatically uses direct-FRACTAL_NZ kernel when eligible

  mxfp4_layer_to_nz_slots_blk_async(host_b13, host_b2, H, I, out_w13, out_w2,
                                    blockdim=None, chunk=None)
      -> (w13_nz, s13b, w2_nz, s2b)   # pinned host GGUF block_mxfp4 -> FRACTAL_NZ slots
                                      # overlaps chunked ACL async H2D with direct-FRACTAL_NZ
                                      # AscendC compute across w13 -> w2 (94% of PCIe roofline,
                                      # zero full-layer blk13/blk2 HBM allocation)

Point sglang's ``kt_stream_prefill`` at this directory with ``KT_MXFP4_OP_DIR``.
"""

import atexit
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
_NZ_S_HALF = 1024
_NZ_S_NB = 64
_NZ_SUPPORTED_IN = (2048, 4096)
_FP4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)

# The fused kernel is launched via a raw ctypes <<<stream>>> call. With TASK_QUEUE_ENABLE=1 torch
# dispatches its ops through an async queue that is NOT ordered against that direct launch, so the
# post-step racing `out`/`osc` needs an explicit host sync (correctness). With TASK_QUEUE_ENABLE=0
# torch ops go straight to the stream, so kernel+post-step are FIFO-ordered and NO sync is needed.
_TQ_SYNC = os.environ.get("TASK_QUEUE_ENABLE", "1") != "0"

_lib = None
_lock = threading.Lock()
_consts_cache = {}
_blk_consts_cache = {}
_blk_nz_consts_cache = {}
_async_states = {}
_blockdim_cache = {}


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


def _default_blockdim(dev=None):
    """Resolve optimal AIV blockdim for the target NPU (override via KT_MXFP4_BLOCKDIM).

    Matching the physical AIV core count (e.g. 40 on Ascend 910B4-1, 48 on 910B2/B3) avoids
    tail-wave scheduling overhead (~1.6x penalty when blockdim slightly exceeds physical AIVs).
    """
    env = os.environ.get("KT_MXFP4_BLOCKDIM")
    if env:
        return int(env)
    key = str(dev)
    if key in _blockdim_cache:
        return _blockdim_cache[key]
    bd = 40
    try:
        import torch_npu  # noqa: F401

        props = torch.npu.get_device_properties(dev)
        aiv = int(getattr(props, "vector_core_num", 0) or 0)
        if aiv <= 0:
            aic = int(getattr(props, "cube_core_num", 0) or 0)
            if aic > 0:
                aiv = aic * 2
        if aiv > 0:
            bd = aiv
    except Exception:
        pass
    _blockdim_cache[key] = bd
    return bd


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
            lib.launch_mxfp4_fused_blk_nz.restype = None
            lib.launch_mxfp4_fused_blk_nz.argtypes = (
                [ctypes.c_void_p, ctypes.c_uint32] + [ctypes.c_void_p] * 9 + [ctypes.c_uint32] * 3
            )
            _lib = lib
    return _lib


def _make_lut_and_blk_tables(half_len: int, nb_len: int):
    """Pure-NumPy construction of MXFP4 LUTs and UB Gather byte-offset tables."""
    b = np.arange(256, dtype=np.int64)
    lutLo = _FP4[b & 0xF].astype(np.float32)
    lutHi = _FP4[(b >> 4) & 0xF].astype(np.float32)
    lutE8 = ((b.astype(np.uint32)) << 23).view(np.float32).astype(np.float32)
    j = np.arange(half_len, dtype=np.int64)
    scOff = ((j >> 4) * 4).astype(np.uint32)
    codeOff = (((j // 16) * 17 + 1 + (j % 16)) * 2).astype(np.uint32)
    nb_idx = np.arange(nb_len, dtype=np.int64)
    scaleOff = ((nb_idx * 17) * 2).astype(np.uint32)
    return lutLo, lutHi, lutE8, scOff, codeOff, scaleOff


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


def _blk_nz_consts(dev):
    """Cached device tables for mxfp4_fused_blk_nz (fixed strip S_HALF=1024, S_NB=64)."""
    key = str(dev)
    if key in _blk_nz_consts_cache:
        return _blk_nz_consts_cache[key]
    tables = _make_lut_and_blk_tables(_NZ_S_HALF, _NZ_S_NB)
    out = tuple(torch.from_numpy(a).to(dev) for a in tables)
    _blk_nz_consts_cache[key] = out
    return out


def _check_acl(err, what):
    if err:
        raise RuntimeError(f"ACL {what} failed with code {err}")


def _cleanup_async_states():
    """Release raw ACL streams/events cleanly on process exit."""
    try:
        import acl
    except Exception:
        return
    for st in list(_async_states.values()):
        for ev in st.get("h2d_done", ()) + st.get("kern_done", ()):
            try:
                acl.rt.destroy_event(ev)
            except Exception:
                pass
        h2d_stream = st.get("h2d_stream")
        if h2d_stream is not None:
            try:
                acl.rt.destroy_stream(h2d_stream)
            except Exception:
                pass
    _async_states.clear()


atexit.register(_cleanup_async_states)


def _ensure_async_state(dev, E, H, I, chunk):
    import acl

    dev_key = str(dev)
    shape_key = (E, H, I, chunk)
    st = _async_states.get(dev_key)
    if st is not None and st.get("shape_key") == shape_key:
        return st

    if st is None:
        h2d_stream, err = acl.rt.create_stream()
        _check_acl(err, "create_stream(h2d)")
        h2d_done = []
        kern_done = []
        for _ in range(2):
            ev1, e1 = acl.rt.create_event()
            _check_acl(e1, "create_event(h2d_done)")
            ev2, e2 = acl.rt.create_event()
            _check_acl(e2, "create_event(kern_done)")
            h2d_done.append(ev1)
            kern_done.append(ev2)
        st = dict(h2d_stream=h2d_stream, h2d_done=h2d_done, kern_done=kern_done)
        _async_states[dev_key] = st

    max_chunk_bytes = max(
        chunk * (2 * I) * ((H // 32) * 17),
        chunk * H * ((I // 32) * 17),
    )
    st["shape_key"] = shape_key
    st["dev_bufs"] = [
        torch.empty((max_chunk_bytes,), dtype=torch.uint8, device=dev),
        torch.empty((max_chunk_bytes,), dtype=torch.uint8, device=dev),
    ]
    st["osc13"] = torch.empty((E, 2 * I), dtype=torch.float32, device=dev)
    st["osc2"] = torch.empty((E, H), dtype=torch.float32, device=dev)
    torch.npu.synchronize()
    return st


def _can_use_blk_nz(OUT, IN, out_nz_format=None):
    """Return True if mxfp4_fused_blk_nz supports this projection shape and target format."""
    if os.environ.get("KT_MXFP4_DIRECT_NZ", "1") == "0":
        return False
    if IN not in _NZ_SUPPORTED_IN or OUT <= 0 or OUT % 32 != 0:
        return False
    if out_nz_format is not None and out_nz_format != _NZ:
        return False
    return True


_NZ_CHUNK = int(os.environ.get("KT_MXFP4_NZ_CHUNK", "32"))  # experts/chunk -> bounds HBM transient
_ASYNC_CHUNK = int(os.environ.get("KT_MXFP4_ASYNC_CHUNK", "16"))  # experts/chunk for async H2D pipeline


def convert_proj(codes_dev, scale_dev, IN, blockdim=None, packing="consecutive", out_nz=None):
    """One projection: MXFP4 codes/scale [E,OUT,*] -> (q_nz [E,IN,OUT] FRACTAL_NZ, oscale bf16 [E,OUT]).

    Chunked over experts so the transient (int8 planes + de-interleave + NZ cast) stays small —
    only the final [E,IN,OUT] NZ output is full-size (HBM-bounded like the W8A8 slot).

    out_nz: optional pre-allocated FRACTAL_NZ [E,IN,OUT] int8 buffer to write into (the reserved
      streaming slot). When given, no per-call ~GBs output allocation happens — the layer's NZ is
      produced straight into the reused slot (HBM budgeted once at load). Must already be NZ-format
      with matching shape. When None, a fresh FRACTAL_NZ buffer is allocated (back-compat).

    packing: nibble layout of the code bytes — how the kernel's lo/hi planes map back to K-positions.
      "consecutive" (native safetensors): byte j -> Kpos 2j (lo), 2j+1 (hi)  -> interleave.
      "halfblock"   (GGUF block_mxfp4):   byte j -> Kpos g*32+jl (lo), +16 (hi) within its 32-group
                                           -> per-group [lo0..15 | hi0..15] concat.
    Both decode the SAME K-ordered weights bit-for-bit; the kernel and scale->block mapping (scOff)
    are packing-agnostic, so only this post-step rearrange differs (no .so change)."""
    import torch_npu

    lib = get_lib()
    dev = codes_dev.device
    if blockdim is None:
        blockdim = _default_blockdim(dev)
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
        if _TQ_SYNC:
            torch.npu.synchronize()
        lo, hi = out[:, :HALFp], out[:, HALFp:]
        if packing == "halfblock":
            nb = HALFp // 16
            q = torch.cat([lo.reshape(Rc, nb, 16), hi.reshape(Rc, nb, 16)], dim=2).reshape(Ec, OUT, IN)
        else:
            q = torch.stack([lo, hi], dim=2).reshape(Ec, OUT, IN)  # consecutive interleave [E,OUT,IN]
        nd = q.to(torch.float16).transpose(1, 2).contiguous().to(torch.int8)  # [E,IN,OUT]
        nz = torch_npu.npu_format_cast(nd, _NZ)
        if out_nz is None:
            out_nz = torch_npu.npu_format_cast(
                torch.empty((E,) + tuple(nz.shape[1:]), dtype=torch.int8, device=dev),
                _NZ,
            )
        out_nz[c:ce].copy_(nz)
        oscale[c:ce] = osc[:Rc].reshape(Ec, OUT).to(torch.bfloat16)
        if _TQ_SYNC:
            torch.npu.synchronize()
        del out, q, nd, nz, osc, cd, sd
    return out_nz, oscale


def mxfp4_layer_to_nz_slots(c13, s13, c2, s2, H, I, blockdim=None, packing="consecutive", out_w13=None, out_w2=None):
    """Full layer depool conversion -> (w13_nz, s13b, w2_nz, s2b), the exact tensors the streaming
    slot + npu_fused_experts consume (replacing the resident W8A8 pool). packing: see convert_proj
    ("consecutive" for native safetensors codes, "halfblock" for GGUF block_mxfp4 codes).
    out_w13/out_w2: optional pre-reserved NZ slots to convert into (no per-layer output alloc)."""
    w13_nz, s13b = convert_proj(c13, s13, H, blockdim, packing, out_nz=out_w13)
    w2_nz, s2b = convert_proj(c2, s2, I, blockdim, packing, out_nz=out_w2)
    return w13_nz, s13b, w2_nz, s2b


def convert_proj_blk(blocks_dev, IN, blockdim=None, out_nz=None):
    """One projection from RAW GGUF block_mxfp4 [E,OUT,nb*17] -> (q_nz [E,IN,OUT] FRACTAL_NZ, oscale
    bf16 [E,OUT]).

    When dimensions match mxfp4_fused_blk_nz (IN in {2048, 4096}, OUT % 32 == 0) and the output
    buffer is FRACTAL_NZ (format 29), runs the direct-FRACTAL_NZ fused kernel in a single pass with
    zero PyTorch rearrange/transpose/format_cast overhead (~2.8x faster). Otherwise falls back
    transparently to the general chunked mxfp4_fused_blk path.
    """
    import torch_npu

    lib = get_lib()
    dev = blocks_dev.device
    if blockdim is None:
        blockdim = _default_blockdim(dev)
    E, OUT, NB17 = blocks_dev.shape

    # Fast path: direct-contiguous FRACTAL_NZ kernel (eliminates cat/fp16/transpose/format_cast/copy_)
    target_fmt = torch_npu.get_npu_format(out_nz) if out_nz is not None else None
    if _can_use_blk_nz(OUT, IN, target_fmt):
        nz_dst = out_nz
        if nz_dst is None:
            nz_dst = torch_npu.npu_format_cast(torch.empty((E, IN, OUT), dtype=torch.int8, device=dev), _NZ)
        if torch_npu.get_npu_format(nz_dst) == _NZ and tuple(nz_dst.shape) == (E, IN, OUT):
            import acl

            bd_c = blocks_dev.contiguous()
            osc = torch.empty((E, OUT), dtype=torch.float32, device=dev)
            lutLo, lutHi, lutE8, scOff, codeOff, scaleOff = _blk_nz_consts(dev)
            if _TQ_SYNC:
                torch.npu.current_stream(dev).synchronize()
            st = int(torch.npu.current_stream(dev).npu_stream)
            P = lambda t: ctypes.c_void_p(int(t.data_ptr()))
            lib.launch_mxfp4_fused_blk_nz(
                ctypes.c_void_p(st),
                blockdim,
                P(bd_c),
                P(nz_dst),
                P(osc),
                P(lutLo),
                P(lutHi),
                P(lutE8),
                P(scOff),
                P(codeOff),
                P(scaleOff),
                E,
                OUT,
                IN,
            )
            if _TQ_SYNC:
                _check_acl(acl.rt.synchronize_stream(st), "synchronize_stream(convert_proj_blk)")
            oscale = osc.to(torch.bfloat16)
            return nz_dst, oscale

    # General fallback path (arbitrary IN/OUT or ND target buffer)
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
            out_nz = torch_npu.npu_format_cast(
                torch.empty((E,) + tuple(nz.shape[1:]), dtype=torch.int8, device=dev),
                _NZ,
            )
        out_nz[c:ce].copy_(nz)
        oscale[c:ce] = osc[:Rc].reshape(Ec, OUT).to(torch.bfloat16)
        if _TQ_SYNC:
            torch.npu.synchronize()
        del out, q, nd, nz, osc, bd
    return out_nz, oscale


def mxfp4_layer_to_nz_slots_blk(blk13, blk2, H, I, blockdim=None, out_w13=None, out_w2=None):
    """Full layer conversion from RAW GGUF blocks (in-kernel de-interleave) -> slot tensors.
    blk13 = cat(gate,up) blocks [E,2I,nbH*17]; blk2 = down blocks [E,H,nbI*17]."""
    w13_nz, s13b = convert_proj_blk(blk13, H, blockdim, out_nz=out_w13)
    w2_nz, s2b = convert_proj_blk(blk2, I, blockdim, out_nz=out_w2)
    return w13_nz, s13b, w2_nz, s2b


def mxfp4_layer_to_nz_slots_blk_async(
    host_b13,
    host_b2,
    H,
    I,
    out_w13,
    out_w2,
    blockdim=None,
    chunk=None,
):
    """Stream pinned host GGUF blocks directly into pre-allocated FRACTAL_NZ slots with overlapped
    chunked ACL async H2D + direct-FRACTAL_NZ AscendC compute across w13 -> w2.

    Falls back transparently to host->device copy + mxfp4_layer_to_nz_slots_blk if slots are not
    FRACTAL_NZ, dimensions are outside {2048, 4096}, or host tensors are non-contiguous.
    """
    import torch_npu

    if out_w13 is None or out_w2 is None:
        raise ValueError("mxfp4_layer_to_nz_slots_blk_async requires pre-allocated out_w13 and out_w2 slots")

    dev = out_w13.device
    if blockdim is None:
        blockdim = _default_blockdim(dev)
    if chunk is None:
        chunk = _ASYNC_CHUNK

    E = int(host_b13.shape[0])
    eligible = (
        host_b13.device.type == "cpu"
        and host_b2.device.type == "cpu"
        and host_b13.dtype == torch.uint8
        and host_b2.dtype == torch.uint8
        and host_b13.is_contiguous()
        and host_b2.is_contiguous()
        and tuple(host_b13.shape) == (E, 2 * I, (H // 32) * 17)
        and tuple(host_b2.shape) == (E, H, (I // 32) * 17)
        and tuple(out_w13.shape) == (E, H, 2 * I)
        and tuple(out_w2.shape) == (E, I, H)
        and _can_use_blk_nz(2 * I, H, torch_npu.get_npu_format(out_w13))
        and _can_use_blk_nz(H, I, torch_npu.get_npu_format(out_w2))
    )
    if not eligible:
        blk13 = host_b13.to(dev, non_blocking=True)
        blk2 = host_b2.to(dev, non_blocking=True)
        return mxfp4_layer_to_nz_slots_blk(blk13, blk2, H, I, blockdim=blockdim, out_w13=out_w13, out_w2=out_w2)

    import acl

    lib = get_lib()
    st = _ensure_async_state(dev, E, H, I, chunk)
    lutLo, lutHi, lutE8, scOff, codeOff, scaleOff = _blk_nz_consts(dev)
    dev_bufs = st["dev_bufs"]
    h2d_stream = st["h2d_stream"]
    h2d_done = st["h2d_done"]
    kern_done = st["kern_done"]
    osc13 = st["osc13"]
    osc2 = st["osc2"]

    # Flush PyTorch task queue so previous layer's GMM / resident gather finishes reading slots
    # before raw ACL kernels overwrite them.
    torch.npu.current_stream(dev).synchronize()
    compute_stream = int(torch.npu.current_stream(dev).npu_stream)

    P = lambda ptr: ctypes.c_void_p(int(ptr))
    p_lutLo = P(lutLo.data_ptr())
    p_lutHi = P(lutHi.data_ptr())
    p_lutE8 = P(lutE8.data_ptr())
    p_scOff = P(scOff.data_ptr())
    p_codeOff = P(codeOff.data_ptr())
    p_scaleOff = P(scaleOff.data_ptr())

    # Unified job list across w13 -> w2 so H2D overlaps continuously with zero boundary bubble.
    jobs = []
    for host, dst, osc, out_dim, in_dim in (
        (host_b13, out_w13, osc13, 2 * I, H),
        (host_b2, out_w2, osc2, H, I),
    ):
        host_base = int(host.data_ptr())
        dst_base = int(dst.data_ptr())
        osc_base = int(osc.data_ptr())
        row_bytes = out_dim * ((in_dim // 32) * 17)
        out_expert_bytes = in_dim * out_dim
        osc_expert_bytes = out_dim * 4
        for c in range(0, E, chunk):
            ec = min(chunk, E - c)
            jobs.append(
                (
                    host_base + c * row_bytes,
                    ec * row_bytes,
                    dst_base + c * out_expert_bytes,
                    osc_base + c * osc_expert_bytes,
                    ec,
                    out_dim,
                    in_dim,
                )
            )

    used_kern = [False, False]
    h_ptr0, nbytes0, _, _, _, _, _ = jobs[0]
    _check_acl(
        acl.rt.memcpy_async(int(dev_bufs[0].data_ptr()), nbytes0, h_ptr0, nbytes0, 1, h2d_stream),
        "memcpy_async(0)",
    )
    _check_acl(acl.rt.record_event(h2d_done[0], h2d_stream), "record_event(h2d_done[0])")

    for idx, (_, _, dst_ptr, osc_ptr, ec, out_dim, in_dim) in enumerate(jobs):
        p = idx & 1
        nxt = idx + 1
        if nxt < len(jobs):
            np_idx = nxt & 1
            nh_ptr, nnbytes, _, _, _, _, _ = jobs[nxt]
            if used_kern[np_idx]:
                _check_acl(acl.rt.stream_wait_event(h2d_stream, kern_done[np_idx]), "h2d wait kern_done")
            _check_acl(
                acl.rt.memcpy_async(int(dev_bufs[np_idx].data_ptr()), nnbytes, nh_ptr, nnbytes, 1, h2d_stream),
                f"memcpy_async({nxt})",
            )
            _check_acl(acl.rt.record_event(h2d_done[np_idx], h2d_stream), f"record_event(h2d_done[{np_idx}])")

        _check_acl(acl.rt.stream_wait_event(compute_stream, h2d_done[p]), f"compute wait h2d_done[{p}]")
        lib.launch_mxfp4_fused_blk_nz(
            ctypes.c_void_p(compute_stream),
            blockdim,
            P(dev_bufs[p].data_ptr()),
            P(dst_ptr),
            P(osc_ptr),
            p_lutLo,
            p_lutHi,
            p_lutE8,
            p_scOff,
            p_codeOff,
            p_scaleOff,
            ec,
            out_dim,
            in_dim,
        )
        _check_acl(acl.rt.record_event(kern_done[p], compute_stream), f"record_event(kern_done[{p}])")
        used_kern[p] = True

    _check_acl(acl.rt.synchronize_stream(compute_stream), "synchronize_stream(compute)")
    return out_w13, osc13.to(torch.bfloat16), out_w2, osc2.to(torch.bfloat16)
