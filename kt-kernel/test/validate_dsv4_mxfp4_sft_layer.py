#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Validate one real DeepSeek-V4 MXFP4 routed-expert SFT layer.

This is a release-validation executable, not a unit test.  It loads layer 3
from the native DeepSeek-V4-Flash-0731 checkpoint, keeps the E2M1 weights and
group-32 scales packed, and exercises the standalone KT Python/C++ SFT path.

The optional numerical oracle is intentionally bounded: it materializes dense
weights only for the six experts selected by token 0 and releases them before
the optimizer-like update.  The production KT path never creates a persistent
BF16/FP32 copy of the 256 routed experts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from mxfp4_sft_reference import (  # noqa: E402
    LORA_NAMES,
    MXFP4ExpertWeights,
    MXFP4Projection,
    relative_l2_and_cosine,
    run_routed_reference,
)

EXPERTS = 256
TOP_K = 6
HIDDEN = 4096
INTERMEDIATE = 2048
RANK = 8
ALPHA = 16.0
SWIGLU_LIMIT = 10.0
DEFAULT_MODEL = "/mnt/models/DeepSeek-V4-Flash-0731"

LORA_SHAPES = {
    "gate_lora_a": (EXPERTS, RANK, HIDDEN),
    "gate_lora_b": (EXPERTS, INTERMEDIATE, RANK),
    "up_lora_a": (EXPERTS, RANK, HIDDEN),
    "up_lora_b": (EXPERTS, INTERMEDIATE, RANK),
    "down_lora_a": (EXPERTS, RANK, INTERMEDIATE),
    "down_lora_b": (EXPERTS, HIDDEN, RANK),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _proc_memory() -> dict[str, float]:
    values: dict[str, float] = {}
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                if key in {"VmRSS", "VmHWM", "VmSize"}:
                    kib = int(value.strip().split()[0])
                    values[f"{key}_gib"] = kib / (1024.0 * 1024.0)
    except OSError:
        pass
    return values


def _system_memory() -> dict[str, float]:
    wanted = {"MemTotal", "MemAvailable"}
    values: dict[str, float] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, _, value = line.partition(":")
                if key in wanted:
                    kib = int(value.strip().split()[0])
                    values[f"{key}_gib"] = kib / (1024.0 * 1024.0)
    except OSError:
        pass
    return values


def _git_metadata() -> dict[str, Any]:
    repo_root = TEST_DIR.parents[1]

    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo_root), *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = run("status", "--porcelain")
    return {
        "root": str(repo_root),
        "branch": run("branch", "--show-current"),
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _record_memory(result: dict[str, Any], stage: str) -> None:
    result.setdefault("memory_timeline", []).append(
        {"stage": stage, "time_utc": _utc_now(), **_proc_memory()}
    )


def _time_call(result: dict[str, Any], stage: str, fn):
    start = time.perf_counter()
    value = fn()
    result.setdefault("timings_seconds", {})[stage] = time.perf_counter() - start
    _record_memory(result, stage)
    return value


def _tensor_bytes(tensor: torch.Tensor) -> memoryview:
    if tensor.device.type != "cpu" or not tensor.is_contiguous():
        raise ValueError("base hash requires a contiguous CPU tensor")
    return memoryview(tensor.detach().view(torch.uint8).numpy()).cast("B")


def _base_storage_hash(
    experts_data: Mapping[str, Iterable[torch.Tensor]],
) -> tuple[str, int]:
    digest = hashlib.sha256()
    total_bytes = 0
    for name in ("gate", "gate_scale", "up", "up_scale", "down", "down_scale"):
        tensors = experts_data[name]
        digest.update(name.encode("ascii"))
        for expert_idx, tensor in enumerate(tensors):
            payload = _tensor_bytes(tensor)
            digest.update(expert_idx.to_bytes(4, byteorder="little"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(len(payload).to_bytes(8, byteorder="little"))
            digest.update(payload)
            total_bytes += len(payload)
    return digest.hexdigest(), total_bytes


def _make_lora(seed: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    params: dict[str, torch.Tensor] = {}
    grads: dict[str, torch.Tensor] = {}
    for name, shape in LORA_SHAPES.items():
        params[name] = (
            torch.randn(shape, generator=generator, dtype=torch.float32)
            .mul_(0.005)
            .to(torch.bfloat16)
            .contiguous()
        )
        # A NaN sentinel proves the first authoritative C++ backward initializes
        # inactive rows rather than merely relying on zero-filled Python memory.
        grads[name] = torch.full(
            shape, float("nan"), dtype=torch.bfloat16, device="cpu"
        ).contiguous()
    return params, grads


def _init_wrapper(
    args: argparse.Namespace,
    lora: Mapping[str, torch.Tensor],
    grads: Mapping[str, torch.Tensor],
):
    from kt_kernel import KTMoEWrapper

    wrapper = KTMoEWrapper(
        layer_idx=args.layer,
        num_experts=EXPERTS,
        num_experts_per_tok=TOP_K,
        hidden_size=HIDDEN,
        moe_intermediate_size=INTERMEDIATE,
        gpu_experts_mask=None,
        cpuinfer_threads=args.threads,
        threadpool_count=args.tp,
        weight_path=args.model_path,
        chunked_prefill_size=max(args.qlen, 32),
        method="MXFP4_SFT",
        mode="sft",
        num_gpu_experts=0,
        lora_rank=RANK,
        lora_alpha=ALPHA,
        lora_dropout=0.0,
        max_cache_depth=1,
        group_size=32,
        zero_point=False,
        full_weight_grad=False,
        swiglu_limit=SWIGLU_LIMIT,
    )
    wrapper.init_lora_weights(
        lora["gate_lora_a"],
        lora["gate_lora_b"],
        lora["up_lora_a"],
        lora["up_lora_b"],
        lora["down_lora_a"],
        lora["down_lora_b"],
        grads["gate_lora_a"],
        grads["gate_lora_b"],
        grads["up_lora_a"],
        grads["up_lora_b"],
        grads["down_lora_a"],
        grads["down_lora_b"],
    )
    return wrapper


def _make_batch(qlen: int, seed: int):
    if qlen < 1 or qlen * TOP_K > EXPERTS:
        raise ValueError(f"qlen must be in [1, {EXPERTS // TOP_K}], got {qlen}")
    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    hidden = (
        torch.randn((qlen, HIDDEN), generator=generator, dtype=torch.float32)
        .mul_(0.02)
        .to(torch.bfloat16)
        .contiguous()
    )
    expert_ids = torch.arange(qlen * TOP_K, dtype=torch.int64).reshape(qlen, TOP_K)
    logits = torch.randn((qlen, TOP_K), generator=generator, dtype=torch.float32)
    route_weights = torch.softmax(logits, dim=-1).contiguous()
    grad_output = (
        torch.randn((qlen, HIDDEN), generator=generator, dtype=torch.float32)
        .mul_(0.01)
        .to(torch.bfloat16)
        .contiguous()
    )
    return hidden, expert_ids.contiguous(), route_weights, grad_output


def _tensor_signal(tensor: torch.Tensor) -> dict[str, Any]:
    finite = bool(torch.isfinite(tensor).all().item())
    value = tensor.float()
    return {
        "finite": finite,
        "nonzero": int(torch.count_nonzero(value).item()),
        "l2": float(torch.linalg.vector_norm(value).item()),
        "max_abs": float(value.abs().max().item()),
    }


def _validate_gradient_sparsity(
    grads: Mapping[str, torch.Tensor], active_experts: set[int]
) -> dict[str, Any]:
    inactive_experts = set(range(EXPERTS)).difference(active_experts)
    summary: dict[str, Any] = {}
    failures: list[str] = []
    for name in LORA_NAMES:
        tensor = grads[name]
        if not torch.isfinite(tensor).all().item():
            failures.append(f"{name} contains non-finite values after backward")
        active_norms = {
            str(expert): float(torch.linalg.vector_norm(tensor[expert].float()).item())
            for expert in sorted(active_experts)
        }
        inactive_nonzero = sum(
            int(torch.count_nonzero(tensor[expert]).item())
            for expert in inactive_experts
        )
        zero_active = [expert for expert, norm in active_norms.items() if norm == 0.0]
        if zero_active:
            failures.append(f"{name} has zero gradient for active experts {zero_active}")
        if inactive_nonzero:
            failures.append(f"{name} has {inactive_nonzero} nonzero inactive values")
        summary[name] = {
            "active_expert_norms": active_norms,
            "inactive_nonzero_values": inactive_nonzero,
        }
    if failures:
        raise AssertionError("; ".join(failures))
    return summary


def _assert_close(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    max_relative_l2: float,
    min_cosine: float,
) -> dict[str, float]:
    if not torch.isfinite(actual).all().item():
        raise AssertionError(f"{name}: KT result contains non-finite values")
    if not torch.isfinite(expected).all().item():
        raise AssertionError(f"{name}: reference contains non-finite values")
    relative_l2, cosine = relative_l2_and_cosine(actual, expected)
    if relative_l2 > max_relative_l2 or cosine < min_cosine:
        raise AssertionError(
            f"{name}: relative_l2={relative_l2:.6f}, cosine={cosine:.6f}; "
            f"limits are <= {max_relative_l2:.6f}, >= {min_cosine:.6f}"
        )
    return {"relative_l2": relative_l2, "cosine": cosine}


def _active_token_reference(
    experts_data: Mapping[str, list[torch.Tensor]],
    lora: Mapping[str, torch.Tensor],
    hidden: torch.Tensor,
    expert_ids: torch.Tensor,
    route_weights: torch.Tensor,
    grad_output: torch.Tensor,
):
    """Run an oracle for token 0 only, remapping its six experts to [0, 6)."""

    selected = [int(value) for value in expert_ids[0].tolist()]
    local_experts = [
        MXFP4ExpertWeights(
            gate=MXFP4Projection(
                experts_data["gate"][expert], experts_data["gate_scale"][expert]
            ),
            up=MXFP4Projection(
                experts_data["up"][expert], experts_data["up_scale"][expert]
            ),
            down=MXFP4Projection(
                experts_data["down"][expert], experts_data["down_scale"][expert]
            ),
        )
        for expert in selected
    ]
    local_lora = {
        name: lora[name][selected].contiguous() for name in LORA_NAMES
    }
    local_ids = torch.arange(TOP_K, dtype=torch.int64).reshape(1, TOP_K)
    output, grad_input, grad_route, grad_lora = run_routed_reference(
        hidden[:1],
        local_ids,
        route_weights[:1],
        local_experts,
        local_lora,
        grad_output[:1],
        lora_scaling=ALPHA / RANK,
        swiglu_limit=SWIGLU_LIMIT,
        transpose_free_backward=False,
    )
    return selected, output, grad_input, grad_route, grad_lora


def _compare_reference(
    reference,
    output: torch.Tensor,
    grad_input: torch.Tensor,
    grad_route: torch.Tensor,
    grads: Mapping[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected, ref_output, ref_dx, ref_route, ref_grads = reference
    metrics = {
        "experts": selected,
        "forward": _assert_close(
            "reference.forward",
            output[:1].float(),
            ref_output,
            max_relative_l2=args.max_relative_l2,
            min_cosine=args.min_cosine,
        ),
        "grad_input": _assert_close(
            "reference.grad_input",
            grad_input[:1].float(),
            ref_dx,
            max_relative_l2=args.max_relative_l2,
            min_cosine=args.min_cosine,
        ),
        "grad_route": _assert_close(
            "reference.grad_route",
            grad_route[:1].float(),
            ref_route,
            max_relative_l2=args.max_relative_l2,
            min_cosine=args.min_cosine,
        ),
        "lora_gradients": {},
    }
    for name in LORA_NAMES:
        metrics["lora_gradients"][name] = _assert_close(
            f"reference.{name}",
            grads[name][selected].float(),
            ref_grads[name],
            max_relative_l2=args.max_relative_l2,
            min_cosine=args.min_cosine,
        )
    return metrics


def _adamw_first_step(
    lora: Mapping[str, torch.Tensor],
    grads: Mapping[str, torch.Tensor],
    active_experts: set[int],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
) -> dict[str, Any]:
    """Apply the bias-corrected first AdamW step to active expert rows only."""

    updated_values = 0
    update_l2_sq = 0.0
    for name in LORA_NAMES:
        parameter = lora[name]
        gradient = grads[name]
        for expert in sorted(active_experts):
            before = parameter[expert].float()
            grad = gradient[expert].float()
            first_moment = (1.0 - beta1) * grad
            second_moment = (1.0 - beta2) * grad.square()
            moment_hat = first_moment / (1.0 - beta1)
            variance_hat = second_moment / (1.0 - beta2)
            update = moment_hat / (variance_hat.sqrt() + eps)
            update.add_(before, alpha=weight_decay)
            after = before - lr * update
            rounded = after.to(torch.bfloat16)
            updated_values += int(torch.count_nonzero(rounded != parameter[expert]).item())
            delta = rounded.float() - before
            update_l2_sq += float(torch.sum(delta.square()).item())
            parameter[expert].copy_(rounded)
    if updated_values == 0:
        raise AssertionError("AdamW-like update did not change any BF16 LoRA value")
    return {
        "lr": lr,
        "betas": [beta1, beta2],
        "eps": eps,
        "weight_decay": weight_decay,
        "changed_values": updated_values,
        "update_l2": math.sqrt(update_l2_sq),
    }


def _load_model_config(model_path: str) -> dict[str, Any] | None:
    config_path = Path(model_path) / "config.json"
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    text_config = config.get("text_config", config)
    keys = (
        "model_type",
        "hidden_size",
        "moe_intermediate_size",
        "n_routed_experts",
        "num_experts_per_tok",
    )
    return {key: text_config.get(key) for key in keys if key in text_config}


def _write_json(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def run(args: argparse.Namespace, result: dict[str, Any]) -> None:
    from kt_kernel.sft.backend import get_mxfp4_runtime
    from kt_kernel.utils.loader import MXFP4SafeTensorLoader

    if args.tp < 1 or INTERMEDIATE % args.tp or (INTERMEDIATE // args.tp) % 32:
        raise ValueError(
            f"tp={args.tp} must partition intermediate_size={INTERMEDIATE} into 32-aligned slices"
        )
    if args.threads < args.tp:
        raise ValueError("threads must be at least tp")
    if args.adamw_lr <= 0.0:
        raise ValueError("adamw-lr must be positive")

    torch.set_num_threads(args.threads)
    runtime = get_mxfp4_runtime()
    result["runtime"] = {
        "cpu_variant": runtime.cpu_variant,
        "kernel": runtime.kernel,
        "weight_layout": runtime.weight_layout,
    }
    result["model_config_excerpt"] = _load_model_config(args.model_path)
    _record_memory(result, "runtime_validated")

    loader = MXFP4SafeTensorLoader(args.model_path)
    try:
        experts_data = _time_call(
            result,
            "load_checkpoint_tensors",
            lambda: loader.load_experts(
                f"model.layers.{args.layer}",
                device="cpu",
                reject_non_finite_scales=True,
            ),
        )
        result["checkpoint_layout"] = {
            name: {
                "expert_count": len(tensors),
                "shape_per_expert": list(tensors[0].shape),
                "dtype": str(tensors[0].dtype),
            }
            for name, tensors in experts_data.items()
        }
        result["checkpoint_layout"]["scale_representation"] = (
            "lossless UE8M0 exponent bits in BF16 storage"
        )
        base_hash_before, base_bytes = _time_call(
            result, "hash_base_before", lambda: _base_storage_hash(experts_data)
        )
        result["base_storage_bytes"] = base_bytes

        lora, grads = _time_call(
            result, "allocate_lora_and_grads", lambda: _make_lora(args.seed)
        )
        wrapper = _time_call(
            result, "construct_wrapper", lambda: _init_wrapper(args, lora, grads)
        )
        physical_to_logical = torch.arange(EXPERTS, dtype=torch.int64).contiguous()
        _time_call(
            result,
            "load_cpp_packed_weights",
            lambda: wrapper.load_mxfp4_weights(experts_data, physical_to_logical),
        )

        hidden, expert_ids, route_weights, grad_output = _make_batch(args.qlen, args.seed)
        active_experts = {int(value) for value in expert_ids.flatten().tolist()}
        result["routing"] = {
            "expert_ids": expert_ids.tolist(),
            "active_experts": sorted(active_experts),
        }

        output = _time_call(
            result,
            "forward",
            lambda: wrapper.forward(
                hidden, expert_ids, route_weights, save_for_backward=True
            ),
        )
        grad_input, grad_route = _time_call(
            result, "backward", lambda: wrapper.backward(grad_output)
        )
        result["signals"] = {
            "forward": _tensor_signal(output),
            "grad_input": _tensor_signal(grad_input),
            "grad_route": _tensor_signal(grad_route),
        }
        for name, signal in result["signals"].items():
            if not signal["finite"] or signal["nonzero"] == 0:
                raise AssertionError(f"{name} is non-finite or has no signal: {signal}")
        result["gradient_sparsity"] = _time_call(
            result,
            "validate_gradient_sparsity",
            lambda: _validate_gradient_sparsity(grads, active_experts),
        )

        if not args.skip_reference:
            reference = _time_call(
                result,
                "active_token_reference",
                lambda: _active_token_reference(
                    experts_data,
                    lora,
                    hidden,
                    expert_ids,
                    route_weights,
                    grad_output,
                ),
            )
            result["reference"] = _compare_reference(
                reference, output, grad_input, grad_route, grads, args
            )
            del reference
            gc.collect()
            _record_memory(result, "reference_released")

        result["optimizer_like_update"] = _time_call(
            result,
            "adamw_first_step",
            lambda: _adamw_first_step(
                lora,
                grads,
                active_experts,
                lr=args.adamw_lr,
                beta1=0.9,
                beta2=0.999,
                eps=1.0e-8,
                weight_decay=0.01,
            ),
        )
        _time_call(result, "publish_updated_lora", wrapper.update_lora_weights)
        updated_output = _time_call(
            result,
            "forward_after_update",
            lambda: wrapper.forward(
                hidden, expert_ids, route_weights, save_for_backward=False
            ),
        )
        output_delta = updated_output.float() - output.float()
        changed_output_values = int(torch.count_nonzero(output_delta).item())
        result["optimizer_like_update"].update(
            {
                "changed_output_values": changed_output_values,
                "output_delta_l2": float(torch.linalg.vector_norm(output_delta).item()),
                "output_delta_max_abs": float(output_delta.abs().max().item()),
                "updated_output_finite": bool(torch.isfinite(updated_output).all().item()),
            }
        )
        if not torch.isfinite(updated_output).all().item() or changed_output_values == 0:
            raise AssertionError("updated LoRA did not produce a finite, changed output")

        base_hash_after, base_bytes_after = _time_call(
            result, "hash_base_after", lambda: _base_storage_hash(experts_data)
        )
        result["base_hash"] = {
            "before": base_hash_before,
            "after": base_hash_after,
            "unchanged": base_hash_before == base_hash_after,
        }
        if base_bytes_after != base_bytes or base_hash_after != base_hash_before:
            raise AssertionError("native packed base weights/scales changed during SFT validation")
    finally:
        loader.close_all_handles()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--qlen", type=int, default=2)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument(
        "--adamw-lr",
        type=float,
        default=1.0e-2,
        help="Deliberately visible one-step smoke-test LR (not a training recommendation).",
    )
    parser.add_argument("--max-relative-l2", type=float, default=0.08)
    parser.add_argument("--min-cosine", type=float, default=0.99)
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help="Skip the token-0/six-active-expert dense numerical oracle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/dsv4_mxfp4_layer3_validation.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_utc": _utc_now(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "pid": os.getpid(),
        "git": _git_metadata(),
        "system_memory": _system_memory(),
        "config": {
            "model_path": str(Path(args.model_path).resolve()),
            "layer": args.layer,
            "experts": EXPERTS,
            "top_k": TOP_K,
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "lora_rank": RANK,
            "lora_alpha": ALPHA,
            "swiglu_limit": SWIGLU_LIMIT,
            "qlen": args.qlen,
            "threads": args.threads,
            "tp": args.tp,
            "reference_enabled": not args.skip_reference,
        },
    }
    _record_memory(result, "start")
    try:
        run(args, result)
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        raise
    else:
        result["status"] = "passed"
    finally:
        result["finished_utc"] = _utc_now()
        result["final_memory"] = _proc_memory()
        _write_json(args.output, result)
        summary = {
            "status": result["status"],
            "artifact": str(args.output.resolve()),
            "timings_seconds": result.get("timings_seconds", {}),
            "final_memory": result["final_memory"],
            "base_unchanged": result.get("base_hash", {}).get("unchanged"),
        }
        print("KT_DSV4_MXFP4_LAYER_VALIDATION=" + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
