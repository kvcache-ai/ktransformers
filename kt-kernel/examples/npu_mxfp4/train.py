# SPDX-License-Identifier: Apache-2.0
import argparse
import hashlib
import json
import math
import os
import struct
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch_npu
from torch import nn
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import PreTrainedTokenizerFast
from kt_kernel.sft.deepseek_v4 import _cache_excludes, canonical_non_expert_key, dequantize_fp8_blocks
from kt_kernel.sft.npu_mxfp4 import NpuExpertStore, NpuRoutedExperts


def fence():
    torch.npu.synchronize()
    dist.barrier()


def read_nonexpert(store, key, v41):
    name = store.index[key]
    path = store.root / name
    with safe_open(path, framework="pt", device="cpu") as f:
        v = f.get_tensor(key)
    if v.dtype == torch.float8_e4m3fn:
        s = key[: -len("weight")] + "scale"
        with safe_open(path, framework="pt", device="cpu") as f:
            v = dequantize_fp8_blocks(v, f.get_tensor(s), 32 if v41 else 128)
    return v


class Rows(nn.Module):
    def __init__(self, rows, l, device):
        super().__init__()
        self.device = device
        self.ids = torch.from_numpy(rows["ids" + l])
        w = torch.from_numpy(rows["weight" + l]).view(torch.float8_e4m3fn)
        s = torch.from_numpy(rows["scale" + l]).view(torch.float8_e8m0fnu).float()
        self.val = (w.float().reshape(w.shape[0], -1, 32) * s.unsqueeze(-1)).flatten(1)

    def forward(self, ids):
        cpu = ids.cpu()
        ix = torch.searchsorted(self.ids, cpu.contiguous())
        assert torch.equal(self.ids[ix], cpu)
        return self.val[ix].to(self.device)


def prepare_rows(model, c, tokenizer, batch, store):
    root = store.root
    m = model.model.engram_hash_state
    m.bind_tokenizer(tokenizer)
    ids = torch.tensor([s["input_ids"] for s in batch])
    mask = torch.tensor([s["attention_mask"] for s in batch]).bool()
    out = m(ids, mask, None)
    ids = {str(layer): torch.unique(out[:, :, i, :]).numpy() for i, layer in enumerate(c.engram_layer_ids)}
    wm = store.index
    data = {}
    for l in ids:
        rows = ids[l]
        data["ids" + l] = rows
        for kind in ("weight", "scale"):
            key = f"layers.{l}.engram.embed.{kind}"
            path = root / wm[key]
            with path.open("rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                info = json.loads(f.read(n))[key]
            assert info["dtype"] in ("F8_E4M3", "F8_E8M0", "F8_E4M3FN"), info
            arr = np.memmap(
                path,
                dtype=np.uint8,
                mode="r",
                offset=8 + n + info["data_offsets"][0],
                shape=tuple(info["shape"]),
            )
            data[kind + l] = arr[rows].copy()
    return data


class Adapter(nn.Module):
    def __init__(self, layer, grouped_linear):
        super().__init__()
        self.base = layer
        kw = {"device": layer.weight.device, "dtype": layer.weight.dtype}
        self.a = nn.Linear(layer.in_features, 8, bias=False, **kw)
        self.b = (
            grouped_linear(8, layer.out_features, layer.n_groups).to(**kw)
            if isinstance(layer, grouped_linear)
            else nn.Linear(8, layer.out_features, bias=False, **kw)
        )
        nn.init.zeros_(self.b.weight)

    @property
    def weight(self):
        return self.base.weight

    def forward(self, x):
        return self.base(x) + self.b(self.a(x)) * 2


def train(args, dev, cg):
    root = args.model
    archive = args.config or root
    kind = json.loads((archive / "config.json").read_text())["model_type"]
    v41 = kind == "deepseek_v41"
    if v41:
        from kt_kernel.sft._configuration_deepseek_v41 import DeepseekV41Config as Config
        from kt_kernel.sft._modeling_deepseek_v41 import (
            DeepseekV41ForCausalLM as CausalLM,
            DeepseekV41GroupedLinear as GroupedLinear,
        )
    elif kind == "deepseek_v4":
        from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config as Config
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
            DeepseekV4ForCausalLM as CausalLM,
            DeepseekV4GroupedLinear as GroupedLinear,
        )
    else:
        raise ValueError(f"Expected a V4 or V4.1 text checkpoint, got {kind}")
    cfg = Config.from_pretrained(archive)
    rank = dist.get_rank()
    batch = json.loads(args.batch.read_text())
    if len(batch) != 2:
        raise ValueError("Supply one tokenized sequence per rank")
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    store = NpuExpertStore(root, cfg.get_text_config(), dev, args.transpose_library)
    try:
        run(args, store, cfg, CausalLM, GroupedLinear, v41, batch, dev, cg, rank, out)
    finally:
        store.close()


def run(args, store, cfg, CausalLM, GroupedLinear, v41, batch, dev, cg, rank, out):
    cfg.get_text_config()._attn_implementation = "eager"
    cfg._attn_implementation = "eager"

    with init_empty_weights(include_buffers=False):
        model = CausalLM(cfg)
    model.requires_grad_(False)
    layers_used = len(model.model.layers)
    if v41:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)
        rows = prepare_rows(model, cfg.get_text_config(), tokenizer, batch, store)
        for l in cfg.get_text_config().engram_layer_ids:
            model.model.engram_tables[str(l)] = Rows(rows, str(l), dev)
        hashstate = model.model.engram_hash_state
        orig_hash = hashstate.forward

        def host_hash(ids, mask, cache):
            return orig_hash(ids.cpu(), None if mask is None else mask.cpu(), cache)

        hashstate.forward = host_hash
    for i, l in enumerate(model.model.layers):
        l.mlp.experts = nn.Identity()
    all_dest = dict(model.named_parameters())
    all_dest.update(model.named_buffers())
    loaded = set()
    for key in store.index:
        if _cache_excludes(key, v41) or key.endswith(".scale"):
            continue
        dest = canonical_non_expert_key(key, v41=v41)
        assert dest in all_dest, (key, dest)
        par, field = dest.rsplit(".", 1)
        parent = model.get_submodule(par)
        v = read_nonexpert(store, key, v41)
        assert v.shape == all_dest[dest].shape, (dest, v.shape, all_dest[dest].shape)
        target = "cpu" if ".shared_experts." in dest else dev
        v = v.to(target)
        if field in parent._parameters:
            setattr(parent, field, nn.Parameter(v, requires_grad=False))
        else:
            setattr(parent, field, v)
        loaded.add(dest)
    for name, p in model.named_parameters():
        assert name in loaded and not p.is_meta, name
    for name, b in list(model.named_buffers()):
        if name in loaded or ".engram_hash_state." in name:
            continue
        assert not b.is_meta, name
        par, field = name.rsplit(".", 1)
        setattr(model.get_submodule(par), field, b.to(dev))

    torch.manual_seed(5489)
    ordinary = []
    for i, l in enumerate(model.model.layers):
        for target in (
            "self_attn.q_a_proj",
            "self_attn.q_b_proj",
            "self_attn.kv_proj",
            "self_attn.o_a_proj",
            "self_attn.o_b_proj",
            "mlp.shared_experts.gate_proj",
            "mlp.shared_experts.up_proj",
            "mlp.shared_experts.down_proj",
        ):
            par, field = target.rsplit(".", 1)
            parent = l.get_submodule(par)
            a = Adapter(getattr(parent, field), GroupedLinear)
            setattr(parent, field, a)
            ordinary.extend([(f"{i}.{target}.a", a.a.weight), (f"{i}.{target}.b", a.b.weight)])
    if not v41:

        def move_shared_input(module, args):
            return (args[0].to("cpu"),)

        def move_shared_output(module, args, result):
            return result.to(dev)

        for layer in model.model.layers:
            layer.mlp.shared_experts.register_forward_pre_hook(move_shared_input)
            layer.mlp.shared_experts.register_forward_hook(move_shared_output)
    routed = []
    for i, l in enumerate(model.model.layers):
        l.mlp.experts = NpuRoutedExperts(store, i)
        routed.extend([(f"{i}.expert.{j}", p) for j, p in enumerate(l.mlp.experts.params)])

    assert len(ordinary) == 16 * layers_used and len(routed) == 6 * layers_used

    def ordinary_digest():
        digest = hashlib.sha256()
        for _, p in ordinary:
            digest.update(p.detach().cpu().contiguous().view(torch.uint8).numpy())
        return digest.hexdigest()

    replicas = [None, None]
    dist.all_gather_object(replicas, ordinary_digest(), group=cg)
    if replicas[0] != replicas[1]:
        with torch.no_grad():
            for _, p in ordinary:
                dist.broadcast(p, src=0, group=cg if p.device.type == "cpu" else None)
        dist.all_gather_object(replicas, ordinary_digest(), group=cg)
    assert replicas[0] == replicas[1], replicas
    model.gradient_checkpointing_enable({"use_reentrant": False})
    model.train()
    named = ordinary + routed
    masters = [nn.Parameter(p.detach().to(torch.float32, copy=True)) for _, p in named]
    opt = torch.optim.AdamW(masters, lr=1e-5, weight_decay=0.0, foreach=False)
    sample = batch[rank]
    ids, lab, mask = [
        torch.tensor([sample[k]], dtype=torch.int64).to(dev) for k in ("input_ids", "labels", "attention_mask")
    ]
    if not (ids.ndim == 2 and ids.shape == lab.shape == mask.shape):
        raise ValueError("Expected matching token, mask and label sequences")
    lengths = [None, None]
    dist.all_gather_object(lengths, (ids.numel(), int((lab[:, 1:] != -100).sum().item())), group=cg)
    if lengths[0] != lengths[1] or not lengths[0][1]:
        raise ValueError("EP2 expects equal sequence lengths and equal nonzero supervised-token counts")
    tokens, supervised = (2 * v for v in lengths[0])

    for step in range(args.steps):
        for _, p in named:
            p.grad = None
        opt.zero_grad(set_to_none=True)
        fence()
        torch.npu.reset_peak_memory_stats()
        start = time.monotonic()
        loss = model(input_ids=ids, labels=lab, attention_mask=mask, use_cache=False).loss
        assert math.isfinite(loss.item())
        fence()
        tf = time.monotonic()
        loss.backward()
        fence()
        tb = time.monotonic()
        for _, p in ordinary:
            assert p.grad is not None
            dist.all_reduce(p.grad, group=cg if p.device.type == "cpu" else None)
            p.grad.mul_(0.5)
        for _, p in routed:
            assert p.grad is not None
            p.grad.mul_(0.5)
        fence()
        sync_end = time.monotonic()
        for (_, p), master in zip(named, masters):
            master.grad = p.grad.float()
        watch = {}
        for kind, token in [
            ("attention", "0.self_attn.o_b_proj.b"),
            ("shared", "0.mlp.shared_experts.down_proj.b"),
            ("routed", "0.expert.5"),
        ]:
            par = next(p for n, p in named if n == token)
            g = par.grad.detach().float().flatten()
            ix = g.abs().argmax().item()
            assert math.isfinite(g[ix].item()) and g[ix].item() != 0
            watch[kind] = (par, ix, par.detach().flatten()[ix].float().item())
        opt.step()
        with torch.no_grad():
            for (_, p), master in zip(named, masters):
                p.copy_(master)
        delta = {k: abs(p.detach().flatten()[i].float().item() - v) for k, (p, i, v) in watch.items()}
        assert all(math.isfinite(v) and v > 0 for v in delta.values()), delta
        fence()
        end = time.monotonic()
        seconds = torch.tensor(end - start)
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX, group=cg)
        means = torch.tensor(loss.item())
        dist.all_reduce(means, group=cg)
        means /= 2
        rec = dict(
            step=step + 1,
            seconds=seconds.item(),
            forward=tf - start,
            backward=tb - tf,
            update=end - tb,
            gradient_sync=sync_end - tb,
            optimizer_and_checks=end - sync_end,
            input_tokens=tokens,
            supervised_tokens=supervised,
            loss=means.item(),
            rank_loss=loss.item(),
            delta=delta,
            input_tps=tokens / seconds.item(),
            supervised_tps=supervised / seconds.item(),
            peak_hbm=torch.npu.max_memory_allocated(),
        )
        counts = torch.tensor([l.mlp.experts.assignments for l in model.model.layers])
        rec["backward_groups"] = [l.mlp.experts.groups for l in model.model.layers]
        dist.all_reduce(counts, group=cg)
        assert (counts == tokens * cfg.get_text_config().num_experts_per_tok).all()
        rec["routes_per_layer_global"] = counts.tolist()
        by_device = {}
        for _, p in named:
            by_device.setdefault(p.device, []).append(p.grad.detach().abs().amax())
        assert all(torch.isfinite(torch.stack(maxima).cpu()).all() for maxima in by_device.values())
        rec["finite_gradient_tensors"] = len(named)
        (out / f"step-{step + 1}-rank-{rank}.json").write_text(json.dumps(rec))
        if rank == 0:
            print(f"step {step + 1}: {rec['seconds']:.2f} s, {rec['input_tps']:.2f} input tokens/s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train V4 text LoRA with two expert-parallel Ascend NPUs")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Canonical model configuration directory if stored separately")
    parser.add_argument("--batch", type=Path, required=True, help="JSON array of two tokenized training records")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transpose-library", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    torch_npu.npu.set_device(device)
    torch.set_num_threads(4)
    dist.init_process_group("hccl", timeout=timedelta(minutes=100))
    try:
        train(args, device, dist.new_group(backend="gloo", timeout=timedelta(minutes=100)))
    finally:
        dist.destroy_process_group()
