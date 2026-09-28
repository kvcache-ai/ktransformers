#!/usr/bin/env python3
"""Run LLaMA-Factory with bounded update audits and deterministic stop points."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from safetensors import safe_open
from transformers import TrainerCallback

from kt_kernel.sft.lora import get_kt_named_trainable_params


def _local(tensor):
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    return tensor.detach()


class AcceptanceAudit(TrainerCallback):
    def __init__(self, stop_after, audit_steps, resume_checkpoint=None):
        self.stop_after = stop_after
        self.audit_steps = set(audit_steps)
        self.initial = {}
        self.resume_checkpoint = resume_checkpoint
        self.batches = []

    def _write(self, args, event):
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / f"audit-rank{args.process_index}.jsonl").open("a") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")

    def on_train_begin(
        self,
        args,
        state,
        control,
        model=None,
        optimizer=None,
        lr_scheduler=None,
        **kwargs,
    ):
        self.params = dict(
            (name, p) for name, p in model.named_parameters() if p.requires_grad
        )
        self.params.update(get_kt_named_trainable_params(model))
        expected = {id(p) for p in self.params.values()}
        actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert expected == actual, (
            "optimizer parameter inventory differs from the adapter inventory"
        )
        assert all(p.requires_grad for p in self.params.values())
        assert all("lora" in name.lower() for name in self.params)
        assert not any(
            ".indexer." in name or ".o_a_proj." in name for name in self.params
        )
        for name, parameter in self.params.items():
            self.initial[name] = _local(parameter).cpu().clone()
        self.frozen = {
            name: self._sample(parameter)
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        }
        assert self.frozen, "no frozen base parameters found"
        self.input_hook = model.register_forward_pre_hook(
            self._observe_batch, with_kwargs=True
        )
        if self.resume_checkpoint:
            self._verify_restored_state(args, state, optimizer, lr_scheduler)
        self._write(
            args,
            {
                "event": "train_begin",
                "step": state.global_step,
                "parameters": {
                    name: {"shape": list(p.shape), "device": str(p.device)}
                    for name, p in self.params.items()
                },
                "frozen_parameter_count": len(self.frozen),
                "frozen_check": "optimizer exclusion and up to 64 evenly spaced values per local tensor",
            },
        )
        self.started = time.perf_counter()

    @staticmethod
    def _sample(parameter):
        local = _local(parameter).reshape(-1)
        if not local.numel():
            return local.cpu().clone()
        indices = torch.linspace(
            0, local.numel() - 1, min(64, local.numel()), device=local.device
        ).long()
        return local[indices].cpu().clone()

    def _observe_batch(self, model, positional, kwargs):
        if not model.training:
            return
        ids, labels, mask = (
            kwargs.get("input_ids"),
            kwargs.get("labels"),
            kwargs.get("attention_mask"),
        )
        assert ids is not None and labels is not None and mask is not None
        assert ids.ndim == labels.ndim == mask.ndim == 2
        self.batches.append(
            {
                "input_shape": list(ids.shape),
                "nonpadding_tokens": int(mask.sum()),
                "supervised_tokens": int((labels != -100).sum()),
                "lengths": mask.sum(dim=-1).tolist(),
            }
        )

    def _verify_restored_state(self, args, state, optimizer, scheduler):
        from torch.distributed.tensor import Replicate, Shard

        from compare_dsv4_resume import Comparison

        root = Path(self.resume_checkpoint)
        saved_state = json.loads((root / "trainer_state.json").read_text())
        assert state.global_step == saved_state["global_step"] > 0
        check = Comparison(atol=0.0, rtol=0.0)
        seen_standard, seen_experts = set(), set()
        with (
            safe_open(
                root / "adapter_model.safetensors", framework="pt", device="cpu"
            ) as standard,
            safe_open(
                root / "fused_expert_lora.safetensors", framework="pt", device="cpu"
            ) as experts,
        ):
            for name, parameter in self.params.items():
                if name.startswith("kt.layers."):
                    key = name.removeprefix("kt.").replace(".fused_lora", "")
                    expected = experts.get_tensor(key)
                    seen_experts.add(key)
                else:
                    key = name.replace(".lora_A.default.", ".lora_A.").replace(
                        ".lora_B.default.", ".lora_B."
                    )
                    expected = standard.get_tensor(key)
                    seen_standard.add(key)
                    if hasattr(parameter, "placements"):
                        assert len(parameter.placements) == 1
                        placement = parameter.placements[0]
                        if isinstance(placement, Shard):
                            mesh = parameter.device_mesh
                            expected = expected.chunk(mesh.size(), dim=placement.dim)[
                                mesh.get_local_rank()
                            ]
                        else:
                            assert isinstance(placement, Replicate)
                check.tensor("parameter/" + name, expected, self.initial[name])
            assert seen_standard == set(standard.keys())
            assert seen_experts == (
                set(experts.keys()) if args.process_index == 0 else set()
            )
        name = (
            f"optimizer_rank_{args.process_index:05d}.pt"
            if (root / "kt_optimizer.index.json").is_file()
            else "optimizer.pt"
        )
        check.tree(
            "optimizer",
            torch.load(root / name, map_location="cpu", weights_only=False, mmap=True),
            optimizer.state_dict(),
        )
        check.tree(
            "scheduler",
            torch.load(root / "scheduler.pt", map_location="cpu", weights_only=False),
            scheduler.state_dict(),
        )
        self._write(
            args,
            {
                "event": "resume_restore",
                "step": state.global_step,
                "exact": not check.failures,
                "parameter_count": len(self.params),
                "tensor_count": check.tensor_count,
                "failure_count": len(check.failures),
                "failures": check.failures[:20],
            },
        )
        assert not check.failures, (
            "checkpoint restore differs before resumed backward: "
            + str(check.failures[:5])
        )

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_started = time.perf_counter()
        self.batches = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        step = state.global_step + 1
        if step not in self.audit_steps:
            return
        records = {}
        for name, parameter in self.params.items():
            grad = parameter.grad
            if grad is None:
                records[name] = {"grad": None}
                continue
            tensor = _local(grad).float()
            finite = bool(torch.isfinite(tensor).all())
            assert finite, f"non-finite gradient: {name}"
            records[name] = {
                "grad_norm": tensor.norm().item(),
                "grad_nonzero": tensor.count_nonzero().item(),
            }
        self._write(args, {"event": "gradients", "step": step, "parameters": records})

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        event = {
            "event": "step_end",
            "step": step,
            "seconds": time.perf_counter() - self.step_started,
            "batches": self.batches,
        }
        if torch.cuda.is_available():
            event.update(
                cuda_peak_allocated=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved=torch.cuda.max_memory_reserved(),
            )
        if step in self.audit_steps:
            frozen = {
                name: parameter
                for name, parameter in kwargs["model"].named_parameters()
                if not parameter.requires_grad
            }
            assert frozen.keys() == self.frozen.keys(), "frozen inventory changed"
            assert all(
                parameter.grad is None
                and torch.equal(self._sample(parameter), self.frozen[name])
                for name, parameter in frozen.items()
            ), "a frozen base parameter changed or accumulated gradients"
            event["frozen_samples_unchanged"] = True
            records = {}
            for name, parameter in self.params.items():
                current = _local(parameter).float().cpu()
                assert torch.isfinite(current).all(), f"non-finite parameter: {name}"
                delta = current - self.initial[name].float()
                records[name] = {
                    "delta_norm": delta.norm().item(),
                    "changed": delta.count_nonzero().item(),
                }
            event["updates_from_start"] = records
        self._write(args, event)
        if self.stop_after and step >= self.stop_after:
            control.should_save = True
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self.input_hook.remove()

    def verify_rng_restore(self, args, checkpoint):
        from compare_dsv4_resume import Comparison

        name = (
            "rng_state.pth"
            if args.world_size == 1
            else f"rng_state_{args.process_index}.pth"
        )
        saved = torch.load(
            Path(checkpoint) / name, map_location="cpu", weights_only=False
        )
        check = Comparison(atol=0.0, rtol=0.0)
        check.tree("python", saved["python"], random.getstate())
        check.tree("numpy", saved["numpy"], np.random.get_state())
        check.tensor("torch_cpu", saved["cpu"], torch.random.get_rng_state())
        if "cuda" in saved:
            current = (
                torch.cuda.get_rng_state_all()
                if args.world_size > 1
                else torch.cuda.get_rng_state()
            )
            check.tree("torch_cuda", saved["cuda"], current)
        self._write(
            args,
            {
                "event": "rng_restore",
                "exact": not check.failures,
                "failures": check.failures,
            },
        )
        assert not check.failures, check.failures

    def on_log(self, args, state, control, logs=None, **kwargs):
        for key, value in (logs or {}).items():
            if isinstance(value, (float, int)):
                assert math.isfinite(value), (
                    f"non-finite training metric: {key}={value}"
                )
        self._write(args, {"event": "log", "step": state.global_step, "metrics": logs})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--audit-steps", default="1,2,10,20,200")
    parser.add_argument("--resume")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.resume:
        config["resume_from_checkpoint"] = args.resume
    from llamafactory.train.tuner import run_exp

    callback = AcceptanceAudit(
        args.stop_after, [int(v) for v in args.audit_steps.split(",")], args.resume
    )
    # Observe restoration at the real Trainer restore point, which is after
    # on_train_begin. Do not change the RNG state or resumed training trajectory.
    from transformers import Trainer

    original_load_rng = Trainer._load_rng_state

    def audited_load_rng(trainer, checkpoint):
        original_load_rng(trainer, checkpoint)
        if checkpoint:
            callback.verify_rng_restore(trainer.args, checkpoint)

    Trainer._load_rng_state = audited_load_rng
    try:
        run_exp(config, callbacks=[callback])
    finally:
        Trainer._load_rng_state = original_load_rng


if __name__ == "__main__":
    main()
