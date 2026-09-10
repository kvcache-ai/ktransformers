# SPDX-License-Identifier: Apache-2.0
"""Observe a real LLaMA-Factory training run without patching installed packages.

The only training override disables final adapter serialization. All optimizer
updates, KT lifecycle hooks, and framework logging execute normally. CUDA/rank
fences are added only at the two measurement-window boundaries.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

from common import sha256, tree_digest, validate_training_config, window_metrics, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--measured", type=int, required=True)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    import yaml

    config = yaml.safe_load(args.config.read_text())
    validate_training_config(config, args.warmup, args.measured)
    rank = int(os.environ["RANK"])
    # Rank 0 owns the CPU expert pool. Peer ranks run GPU non-experts and
    # communication; their host orchestration uses SMT siblings of socket 1.
    affinity = set(range(128)) if rank == 0 else set(range(96, 128))
    os.sched_setaffinity(0, affinity)
    os.environ["OMP_NUM_THREADS"] = "64" if rank == 0 else "4"
    os.environ["ACCELERATE_KT_OMP_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]

    import accelerate
    import kt_kernel
    import llamafactory
    import torch
    import torch.distributed as dist
    import transformers
    from kt_kernel.sft.profiler import _find_kt_wrappers, collect_kt_sft_profile, reset_kt_sft_profile
    from transformers import Trainer, TrainerCallback

    extension = Path(kt_kernel.kt_kernel_ext.__file__).resolve()
    identity = {
        "rank": rank,
        "world_size": int(os.environ["WORLD_SIZE"]),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "extension_path": str(extension),
        "extension_sha256": sha256(extension),
        "cpu_variant": kt_kernel.__cpu_variant__,
        "config_sha256": sha256(args.config),
        "profile": args.profile,
        "modules": {
            m.__name__: {"version": getattr(m, "__version__", None), "path": m.__file__}
            for m in (torch, accelerate, transformers, kt_kernel, llamafactory)
        },
    }
    if rank == 0:
        identity["python_tree_sha256"] = {
            m.__name__: tree_digest(Path(m.__file__).parent)
            for m in (accelerate, transformers, kt_kernel, llamafactory)
        }
    write_json(args.run_dir / f"identity.rank{rank}.json", identity)

    def drain(model) -> None:
        for layer in _find_kt_wrappers(model) or []:
            backend = getattr(layer, "wrapper", None)
            if backend is not None:
                backend.wait_backward_repack()
        torch.cuda.synchronize()
        dist.barrier()

    def optimizer_probe(optimizer) -> list[dict]:
        # Small local-shard samples, outside the measured window. No full
        # FSDP parameter gathering, no full-adapter serialization.
        rows = []
        for group_idx, group in enumerate(optimizer.param_groups):
            candidates = group["params"]
            indices = sorted(set([0, len(candidates) // 2, len(candidates) - 1]))
            for index in indices:
                if index < 0 or not candidates:
                    continue
                parameter = candidates[index]
                value = parameter.detach()
                if hasattr(value, "to_local"):
                    value = value.to_local()
                sample = value.reshape(-1)[:64].float().cpu()
                if not torch.isfinite(sample).all():
                    raise FloatingPointError("non-finite trainable parameter sample")
                rows.append(
                    {
                        "group": group_idx,
                        "index": index,
                        "device": str(value.device),
                        "shape": list(value.shape),
                        "values": sample.tolist(),
                    }
                )
        return rows

    class PerfWindow(TrainerCallback):
        def __init__(self):
            self.begin = None
            self.begin_tokens = None
            self.updates = 0
            self.logs = []
            self.host_step_ends = []
            self.metrics = None

        def on_train_begin(self, training_args, state, control, **kwargs):
            model = kwargs["model"]
            wrappers = _find_kt_wrappers(model) or []
            backend_types = [
                type(layer.wrapper.moe).__name__ for layer in wrappers if getattr(layer, "wrapper", None) is not None
            ]
            if rank == 0 and (len(backend_types) != 58 or any("FP8" not in x for x in backend_types)):
                raise RuntimeError(f"expected 58 rank-0 FP8 expert backends, got {backend_types}")
            write_json(
                args.run_dir / f"initial.rank{rank}.json",
                {
                    "backend_types": backend_types,
                    "optimizer_probe": optimizer_probe(kwargs["optimizer"]),
                },
            )
            torch.cuda.reset_peak_memory_stats()

        def on_optimizer_step(self, training_args, state, control, **kwargs):
            self.updates += 1

        def on_step_end(self, training_args, state, control, **kwargs):
            self.host_step_ends.append({"step": state.global_step, "host_time": time.perf_counter()})
            if state.global_step == args.warmup:
                drain(kwargs["model"])
                if args.profile:
                    reset_kt_sft_profile(kwargs["model"])
                self.begin_tokens = int(state.num_input_tokens_seen)
                self.begin = time.perf_counter()
            if state.global_step == args.warmup + args.measured:
                drain(kwargs["model"])
                elapsed = time.perf_counter() - self.begin
                durations = torch.tensor([elapsed], dtype=torch.float64, device=training_args.device)
                dist.all_reduce(durations, op=dist.ReduceOp.MAX)
                self.metrics = window_metrics(
                    self.begin_tokens,
                    int(state.num_input_tokens_seen),
                    durations.item(),
                    args.measured,
                )
                self.metrics["optimizer_updates_total"] = self.updates
                self.metrics["warmup_steps"] = args.warmup
                self.metrics["profile_enabled"] = args.profile
                write_json(args.run_dir / f"window.rank{rank}.json", self.metrics)
                if rank == 0:
                    print("KT_PERF_WINDOW " + json.dumps(self.metrics), flush=True)
                    if args.profile:
                        write_json(args.run_dir / "native-profile.json", collect_kt_sft_profile(kwargs["model"]))

        def on_log(self, training_args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            for key in ("loss", "grad_norm", "train_loss"):
                value = logs.get(key)
                if value is not None and not math.isfinite(float(value)):
                    raise FloatingPointError(f"non-finite {key} at step {state.global_step}: {value}")
            self.logs.append({"step": state.global_step, **logs})
            if rank == 0:
                write_json(args.run_dir / "training-metrics.json", self.logs)

        def on_train_end(self, training_args, state, control, **kwargs):
            if self.metrics is None or self.updates != args.warmup + args.measured:
                raise RuntimeError("training did not finish the complete optimizer/measurement window")
            write_json(
                args.run_dir / f"final.rank{rank}.json",
                {
                    "optimizer_probe": optimizer_probe(kwargs["optimizer"]),
                    "optimizer_updates": self.updates,
                    "cuda_peak_allocated": torch.cuda.max_memory_allocated(),
                    "cuda_peak_reserved": torch.cuda.max_memory_reserved(),
                    "host_step_ends_not_synchronized": self.host_step_ends,
                },
            )

    def skip_final_save(self, *unused_args, **unused_kwargs):
        if rank == 0:
            print("KT_PERF_SKIP_FINAL_ADAPTER_SERIALIZATION", flush=True)

    Trainer.save_model = skip_final_save
    from llamafactory.train.tuner import run_exp

    run_exp(args=config, callbacks=[PerfWindow()])


if __name__ == "__main__":
    main()
