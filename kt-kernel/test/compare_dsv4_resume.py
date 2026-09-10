#!/usr/bin/env python3
"""Compare trusted, locally produced V4 continuous/resumed training checkpoints."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


class Comparison:
    def __init__(self, atol, rtol):
        self.atol, self.rtol = atol, rtol
        self.failures = []
        self.tensor_count = 0
        self.nonidentical_tensors = 0
        self.max_abs = 0.0
        self.max_relative_l2 = 0.0

    def tensor(self, name, left, right):
        self.tensor_count += 1
        if left.shape != right.shape or left.dtype != right.dtype:
            self.failures.append(name + ": shape/dtype mismatch")
            return
        if hasattr(left, "to_local"):
            if str(left.placements) != str(right.placements):
                self.failures.append(name + ": DTensor placements mismatch")
            left, right = left.to_local(), right.to_local()
        left, right = left.detach().cpu(), right.detach().cpu()
        if (left.is_floating_point() or left.is_complex()) and not (
            torch.isfinite(left).all() and torch.isfinite(right).all()
        ):
            self.failures.append(name + ": non-finite tensor")
            return
        if torch.equal(left, right):
            return
        self.nonidentical_tensors += 1
        error2 = norm2 = maximum = 0.0
        close = True
        for a, b in zip(
            left.reshape(-1).split(1 << 20), right.reshape(-1).split(1 << 20)
        ):
            a, b = a.double(), b.double()
            delta = (a - b).abs()
            maximum = max(maximum, delta.max().item())
            error2 += delta.square().sum().item()
            norm2 += a.square().sum().item()
            close &= bool(torch.isfinite(a).all() and torch.isfinite(b).all())
            close &= bool((delta <= self.atol + self.rtol * a.abs()).all())
        relative = (error2 / max(norm2, 1e-300)) ** 0.5
        self.max_abs = max(self.max_abs, maximum)
        self.max_relative_l2 = max(self.max_relative_l2, relative)
        if not close:
            self.failures.append(
                f"{name}: max_abs={maximum:.8g}, relative_l2={relative:.8g}"
            )

    def tree(self, name, left, right):
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            self.tensor(name, left, right)
        elif isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            if not np.array_equal(left, right):
                self.failures.append(name + ": array differs")
        elif isinstance(left, dict) and isinstance(right, dict):
            if left.keys() != right.keys():
                self.failures.append(name + ": dictionary inventory differs")
            for key in left.keys() & right.keys():
                self.tree(f"{name}/{key}", left[key], right[key])
        elif isinstance(left, (list, tuple)) and isinstance(right, type(left)):
            if len(left) != len(right):
                self.failures.append(name + ": sequence length differs")
            for index, (a, b) in enumerate(zip(left, right)):
                self.tree(f"{name}/{index}", a, b)
        elif left != right:
            self.failures.append(f"{name}: {left!r} != {right!r}")

    def adapter_config(self, left, right):
        # PEFT serializes target_modules from a set; list order is not semantic.
        left, right = dict(left), dict(right)
        for config in (left, right):
            targets = config.get("target_modules")
            if isinstance(targets, list) and all(isinstance(v, str) for v in targets):
                config["target_modules"] = sorted(targets)
        self.tree("adapter_config.json", left, right)

    def safetensors(self, name, left, right):
        with (
            safe_open(left, framework="pt", device="cpu") as a,
            safe_open(right, framework="pt", device="cpu") as b,
        ):
            if set(a.keys()) != set(b.keys()):
                self.failures.append(name + ": tensor inventory differs")
            for key in sorted(set(a.keys()) & set(b.keys())):
                self.tensor(name + "/" + key, a.get_tensor(key), b.get_tensor(key))


def compare(reference, resumed, atol=0.0, rtol=0.0):
    reference, resumed = Path(reference), Path(resumed)
    check = Comparison(atol, rtol)
    check.adapter_config(
        json.loads((reference / "adapter_config.json").read_text()),
        json.loads((resumed / "adapter_config.json").read_text()),
    )
    name = "kt_optimizer.index.json"
    check.tree(
        name,
        json.loads((reference / name).read_text()),
        json.loads((resumed / name).read_text()),
    )
    for name in ("adapter_model.safetensors", "fused_expert_lora.safetensors"):
        check.safetensors(name, reference / name, resumed / name)
    index = json.loads((reference / "kt_optimizer.index.json").read_text())
    for name in (*index["rank_files"], "scheduler.pt"):
        # These pickle files were produced by this acceptance run, never supplied by an external party.
        check.tree(
            name,
            torch.load(
                reference / name, map_location="cpu", weights_only=False, mmap=True
            ),
            torch.load(
                resumed / name, map_location="cpu", weights_only=False, mmap=True
            ),
        )
    for rank in range(index["world_size"]):
        name = f"rng_state_{rank}.pth"
        check.tree(
            name,
            torch.load(reference / name, map_location="cpu", weights_only=False),
            torch.load(resumed / name, map_location="cpu", weights_only=False),
        )
    a, b = (
        json.loads((root / "trainer_state.json").read_text())
        for root in (reference, resumed)
    )
    for key in ("global_step", "epoch", "max_steps", "num_train_epochs"):
        check.tree("trainer/" + key, a[key], b[key])
    losses_a = {row["step"]: row["loss"] for row in a["log_history"] if "loss" in row}
    losses_b = {row["step"]: row["loss"] for row in b["log_history"] if "loss" in row}
    check.tree("loss_history", losses_a, losses_b)
    return {
        "passed": not check.failures,
        "reference": str(reference),
        "resumed": str(resumed),
        "atol": atol,
        "rtol": rtol,
        "global_step": a["global_step"],
        "tensor_count": check.tensor_count,
        "nonidentical_tensors": check.nonidentical_tensors,
        "max_abs": check.max_abs,
        "max_relative_l2": check.max_relative_l2,
        "failure_count": len(check.failures),
        "failures": check.failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--resumed", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()
    report = compare(args.reference, args.resumed, args.atol, args.rtol)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
