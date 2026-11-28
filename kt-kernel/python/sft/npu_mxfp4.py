# SPDX-License-Identifier: Apache-2.0
"""Opt-in expert-parallel MXFP4 LoRA for two Ascend NPUs."""

import ctypes
import json
import mmap
import struct
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn


def _matmul(x, planes):
    import torch_npu

    def quantize(t):
        return torch_npu.npu_dynamic_mx_quant(t.contiguous(), dst_type=torch.float8_e4m3fn)

    q, scales = quantize(x)
    exponents = scales.view(torch.uint8).reshape(x.shape[0], -1).float()
    blocks = q.bfloat16().reshape(x.shape[0], -1, 32)
    approx = (blocks * torch.exp2(exponents - 127).bfloat16().unsqueeze(-1)).reshape_as(x)
    total = None
    for a, s in ((q, scales), quantize(x - approx)):
        for w, ws in planes:
            y = torch_npu.npu_quant_matmul(
                a,
                w,
                ws,
                pertoken_scale=s,
                output_dtype=torch.bfloat16,
                x2_dtype=torch_npu.float4_e2m1fn_x2,
                scale_dtype=torch_npu.float8_e8m0fnu,
                pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                group_sizes=[1, 1, 32],
            ).float()
            total = y if total is None else total + y
    return total.bfloat16()


def _operands(w, s):
    return w.view(torch.int8).T, s.reshape(s.shape[0], -1, 2).transpose(0, 1)


class NpuExpertStore:
    """Shared forward operand cache and double-buffered host transpose workspace."""

    def __init__(self, model_dir, config, device, transpose_library, cache_bytes=28 << 30):
        if dist.get_world_size() != 2:
            raise ValueError("MXFP4 expert parallelism requires two ranks")
        self.root = Path(model_dir).resolve()
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]
        self.device = device
        self.rank = dist.get_rank()
        self.experts = config.n_routed_experts
        self.local_experts = self.experts // 2
        h, m = config.hidden_size, config.moe_intermediate_size
        if self.experts <= 0 or self.experts % 2 or min(h, m) <= 0 or h % 64 or m % 64:
            raise ValueError("Expected evenly sharded experts and 64-aligned MX dimensions")
        self.projections = (("w1", m, h), ("w3", m, h), ("w2", h, m))
        self.hidden = h
        self.clamp = config.swiglu_limit
        self.maps = {}
        self.cache = OrderedDict()
        self.cache_size, self.limit = 0, cache_bytes
        self.buffers = [None, None]
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.lib = ctypes.CDLL(str(Path(transpose_library).resolve()))
        ptr, integer = ctypes.c_void_p, ctypes.c_int
        self.lib.planes.argtypes = [ptr, integer, integer]
        self.lib.planes.restype = integer
        self.lib.pack.argtypes = [ptr, ptr, integer, integer, integer, ptr, ptr]
        self.lib.pack.restype = None

    def close(self):
        self.pool.shutdown(wait=True)
        self.cache.clear()
        for _, _, mapping in self.maps.values():
            mapping.close()
        self.maps.clear()
        self.buffers = [None, None]

    def _tensor(self, key, shape):
        name = self.index[key]
        if name not in self.maps:
            path = self.root / name
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Checkpoint shard outside model directory")
            with path.open("rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(size))
                self.maps[name] = (8 + size, header, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
        offset, header, mapping = self.maps[name]
        spec = header[key]
        start, stop = spec["data_offsets"]
        if stop - start != shape[0] * shape[1] or start < 0 or offset + stop > len(mapping):
            raise ValueError(f"Invalid packed tensor size for {key}")
        return np.frombuffer(mapping, np.uint8, count=stop - start, offset=offset + start).reshape(shape)

    def _weights(self, layer, expert, projection):
        key, n, k = self.projections[projection]
        prefix = f"layers.{layer}.ffn.experts.{expert}.{key}."
        return self._tensor(prefix + "weight", (n, k // 2)), self._tensor(prefix + "scale", (n, k // 32))

    def forward_weight(self, layer, expert, projection):
        key = layer, expert, projection
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key][0]
        w, s = self._weights(*key)
        # DMA needs writable memory, while the large checkpoint stays read-only.
        pair = _operands(torch.from_numpy(w.copy()).to(self.device), torch.from_numpy(s.copy()).to(self.device))
        size = w.nbytes + s.nbytes
        while self.cache and self.cache_size + size > self.limit:
            _, (_, freed) = self.cache.popitem(last=False)
            self.cache_size -= freed
        if size <= self.limit:
            self.cache[key] = (pair, size)
            self.cache_size += size
        return pair

    def _pack(self, layer, experts, slot):
        parts, size = [], 0
        for expert in experts:
            for i, (_, n, k) in enumerate(self.projections):
                w, s = self._weights(layer, expert, i)
                count = self.lib.planes(s.ctypes.data, n, k)
                parts.append((expert, i, n, k, w, s, count))
                size += count * (w.size + s.size)
        buf = self.buffers[slot]
        if buf is None or buf.size < size:
            self.buffers[slot] = buf = np.empty(size, dtype=np.uint8)
        offsets = {}
        pos = 0
        for expert, i, n, k, w, s, count in parts:
            scales = pos + count * w.size
            offsets[expert, i] = (count, pos, scales)
            self.lib.pack(w.ctypes.data, s.ctypes.data, n, k, count, buf.ctypes.data + pos, buf.ctypes.data + scales)
            pos = scales + count * s.size
        return buf[:size], offsets

    def backward_groups(self, layer, routes):
        groups = [list(items) for _, items in groupby(routes, key=lambda row: row[0] // 16)]

        def submit(i):
            return self.pool.submit(self._pack, layer, [e for e, _, _ in groups[i]], i % 2)

        future = submit(0) if groups else None
        for i, group in enumerate(groups):
            host, offsets = future.result()
            future = submit(i + 1) if i + 1 < len(groups) else None
            # The blocking H2D copy finishes before this host slot can be reused.
            yield group, torch.from_numpy(host).to(self.device), offsets

    def transposed(self, slab, offsets, expert, projection):
        _, n, k = self.projections[projection]
        count, w, s = offsets[expert, projection]
        weights, scales = n * k // 2, n * k // 32
        for plane in range(count):
            # The native operator needs independent operands at storage offset zero.
            code = slab.narrow(0, w + plane * weights, weights).view(k, n // 2).clone()
            scale = slab.narrow(0, s + plane * scales, scales).view(k, n // 32).clone()
            yield _operands(code, scale)


def _gather(t):
    parts = [torch.empty_like(t) for _ in range(2)]
    dist.all_gather(parts, t.contiguous())
    return torch.cat(parts)


def _reduce(t, rank):
    dist.all_reduce(t)
    return t.chunk(2)[rank].contiguous()


class _Projection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, saved, operands):
        ctx.operands = operands
        return saved

    @staticmethod
    def backward(ctx, grad):
        return _matmul(grad, list(ctx.operands())), None, None


class _Routed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, ids, layer, *params):
        ctx.layer = layer
        ctx.save_for_backward(x, weights, ids, *params)
        if layer.pending is not None:
            if torch._C._current_graph_task_id() < 0:
                raise RuntimeError("Finish backward before the next microbatch")
            return layer.pending[1]
        h, w, routes = _gather(x), _gather(weights), _gather(ids)
        bucket = h.new_zeros((routes.numel(), h.shape[-1]), dtype=torch.float32)
        frozen = {}
        for expert, ix, pos in layer.routes(routes):
            local = [p[expert % layer.store.local_experts] for p in params]
            value = layer.expert(h.index_select(0, ix), expert, local, frozen)
            bucket.index_copy_(0, pos, value.float() * w.reshape(-1, 1).index_select(0, pos).float())
        output = _reduce(bucket.view(h.shape[0], -1, h.shape[-1]).sum(1), layer.store.rank).to(x.dtype)
        layer.pending = frozen, output.detach()
        return output

    @staticmethod
    def backward(ctx, grad):
        x, weights, ids, *params = ctx.saved_tensors
        layer = ctx.layer
        frozen, _ = layer.pending
        layer.pending = None
        h, w, routes, dy = map(_gather, (x, weights, ids, grad))
        dx = h.new_zeros((routes.numel(), h.shape[-1]), dtype=torch.float32)
        dr = torch.zeros_like(w).reshape(-1)
        dp = [torch.zeros_like(p) for p in params]
        layer.groups = layer.assignments = 0
        for group, slab, offsets in layer.store.backward_groups(layer.index, layer.routes(routes)):
            layer.groups += 1
            for expert, ix, pos in group:
                ei = expert % layer.store.local_experts
                with torch.enable_grad():
                    inp = h.index_select(0, ix).detach().requires_grad_()
                    route = w.reshape(-1, 1).index_select(0, pos).detach().requires_grad_()
                    local = [p[ei].detach().requires_grad_() for p in params]
                    value = layer.expert(inp, expert, local, frozen, (slab, offsets))
                    grads = torch.autograd.grad(
                        value.float() * route.float(), [inp, route, *local], dy.index_select(0, ix).float()
                    )
                dx.index_copy_(0, pos, grads[0].float())
                dr.index_copy_(0, pos, grads[1].flatten())
                for dest, g in zip(dp, grads[2:]):
                    dest[ei].copy_(g)
                layer.assignments += pos.numel()
        return (
            _reduce(dx.view(h.shape[0], -1, h.shape[-1]).sum(1), layer.store.rank).to(x.dtype),
            _reduce(dr.view_as(w), layer.store.rank),
            None,
            None,
            *dp,
        )


class NpuRoutedExperts(nn.Module):
    def __init__(self, store, index, rank=8, alpha=16):
        super().__init__()
        self.store, self.index = store, index
        self.pending = None
        self.groups = self.assignments = 0
        self.scaling = alpha / rank
        self.params = nn.ParameterList()
        for _, n, k in store.projections:
            a = torch.empty((store.experts, rank, k), dtype=torch.bfloat16)
            nn.init.kaiming_uniform_(a.view(-1, k), a=5**0.5)
            self.params.append(nn.Parameter(a.chunk(2)[store.rank].contiguous().to(store.device)))
            self.params.append(
                nn.Parameter(torch.zeros((store.local_experts, n, rank), dtype=torch.bfloat16, device=store.device))
            )

    def routes(self, ids):
        arr = ids.cpu().numpy().ravel()
        order = np.argsort(arr, kind="stable")
        counts = np.bincount(arr, minlength=self.store.experts)
        ends = np.cumsum(counts)
        for e in range(self.store.rank * self.store.local_experts, (self.store.rank + 1) * self.store.local_experts):
            if counts[e]:
                slots = order[ends[e] - counts[e] : ends[e]]
                yield e, torch.from_numpy(slots // ids.shape[1]).to(ids.device), torch.from_numpy(slots).to(ids.device)

    def expert(self, x, expert, params, frozen, backward=None):
        def project(i, t):
            if backward is None:
                result = _matmul(t, [self.store.forward_weight(self.index, expert, i)])
                frozen[expert, i] = result.detach()
            else:
                result = _Projection.apply(t, frozen[expert, i], lambda: self.store.transposed(*backward, expert, i))
            a, b = params[2 * i : 2 * i + 2]
            return result + ((t @ a.T) @ b.T) * self.scaling

        limit = self.store.clamp
        gate = project(0, x).clamp(max=limit)
        up = project(1, x).clamp(-limit, limit)
        return project(2, torch.nn.functional.silu(gate) * up)

    def forward(self, hidden_states, expert_ids, weights):
        return _Routed.apply(hidden_states, weights, expert_ids, self, *self.params)
