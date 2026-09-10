# SPDX-License-Identifier: Apache-2.0
"""Compare complete independent-process FP8 runs, rejecting contract drift."""

import argparse
import json
import math
from pathlib import Path
import statistics

import yaml

from common import write_json


def read_run(path: Path) -> dict:
    status = json.loads((path / "status.json").read_text())
    if status["status"] != "PASS":
        raise ValueError(f"{path.name}: not a complete PASS run")
    provenance = json.loads((path / "provenance.json").read_text())
    if provenance["profile"] or provenance.get("trace", False):
        raise ValueError(f"{path.name}: diagnostic profiler is enabled")
    config = yaml.safe_load((path / "train.yaml").read_text())
    config.pop("output_dir")
    initial = [json.loads((path / f"initial.rank{rank}.json").read_text()) for rank in range(8)]
    final = [json.loads((path / f"final.rank{rank}.json").read_text()) for rank in range(8)]
    identities = [json.loads((path / f"identity.rank{rank}.json").read_text()) for rank in range(8)]
    expected_updates = provenance["warmup"] + provenance["measured"]
    if any(row["optimizer_updates"] != expected_updates for row in final):
        raise ValueError(f"{path.name}: optimizer update count mismatch")
    changed = [a["optimizer_probe"] != b["optimizer_probe"] for a, b in zip(initial, final)]
    if not all(changed):
        raise ValueError(f"{path.name}: parameter samples did not change on every rank: {changed}")
    metrics = status["metrics"]
    if not math.isfinite(metrics["tokens_per_second"]) or metrics["tokens_per_second"] <= 0:
        raise ValueError(f"{path.name}: invalid throughput")
    contract = {
        "train": config,
        "accelerate": yaml.safe_load((path / "accelerate.yaml").read_text()),
        "warmup": provenance["warmup"],
        "measured": provenance["measured"],
        "checkpoint_config": provenance["checkpoint_config_sha256"],
        "checkpoint_index": provenance["checkpoint_index_sha256"],
        "checkpoint_shards": provenance["checkpoint_shard_metadata_not_content_hashes"],
        "fixture": provenance["fixture_file_sha256"],
        "framework_sources": {
            key: value for key, value in identities[0]["python_tree_sha256"].items() if key != "kt_kernel"
        },
        "initial_parameters": initial,
        "rank_affinity": [row["cpu_affinity"] for row in identities],
        "global_tokens": metrics["global_non_padding_tokens"],
        "entrypoint_sha256": provenance["harness_file_sha256"]["train_entry.py"],
        "metric_helpers_sha256": provenance["harness_file_sha256"]["common.py"],
    }
    return {
        "run": path.name,
        "contract": contract,
        "metrics": metrics,
        "binary": identities[0]["extension_sha256"],
        "variant": provenance["variant_spec"],
        "losses": json.loads((path / "training-metrics.json").read_text()),
        "final_parameter_samples": [row["optimizer_probe"] for row in final],
        "cuda_peak_allocated": [row["cuda_peak_allocated"] for row in final],
    }


def numerical_diagnostics(reference: dict, candidate: dict) -> dict:
    """Describe sampled drift; never turn a few parameter samples into a proof."""
    result = {"scope": "sampled diagnostics; compare A/A drift and native correctness tests separately"}
    a_steps = {row["step"]: row for row in reference.get("losses", []) if "loss" in row}
    b_steps = {row["step"]: row for row in candidate.get("losses", []) if "loss" in row}
    if a_steps.keys() != b_steps.keys():
        raise ValueError("loss traces have different optimizer steps")
    result["steps"] = [
        {
            "step": step,
            "loss_abs_diff": abs(a_steps[step]["loss"] - b_steps[step]["loss"]),
            "grad_norm_abs_diff": abs(a_steps[step]["grad_norm"] - b_steps[step]["grad_norm"]),
        }
        for step in a_steps
    ]
    result["parameter_probes"] = []
    for rank, (a_probes, b_probes) in enumerate(
        zip(
            reference.get("final_parameter_samples", []),
            candidate.get("final_parameter_samples", []),
            strict=True,
        )
    ):
        initial = reference["contract"]["initial_parameters"][rank]["optimizer_probe"]
        for start, a, b in zip(initial, a_probes, b_probes, strict=True):
            identity = {key: a[key] for key in ("device", "group", "index", "shape")}
            if any({key: probe[key] for key in identity} != identity for probe in (start, b)):
                raise ValueError("parameter probe identity changed")
            values = list(zip(start["values"], a["values"], b["values"], strict=True))
            if not values or not all(math.isfinite(value) for triplet in values for value in triplet):
                raise ValueError("parameter samples are empty or non-finite")
            error = math.fsum((y - x) ** 2 for _, x, y in values)
            update = math.fsum((x - s) ** 2 for s, x, _ in values)
            result["parameter_probes"].append(
                {
                    "rank": rank,
                    **identity,
                    "sample_count": len(values),
                    "max_abs_diff": max(abs(y - x) for _, x, y in values),
                    "rms_diff": math.sqrt(error / len(values)),
                    "relative_to_reference_update_l2": math.sqrt(error / update) if update else None,
                }
            )
    return result


def summarize(rows: list[dict]) -> dict:
    values = [row["metrics"]["tokens_per_second"] for row in rows]
    mean = statistics.mean(values)
    return {
        "independent_runs": len(rows),
        "tokens_per_second": values,
        "median_tokens_per_second": statistics.median(values),
        "mean_tokens_per_second": mean,
        "sample_stdev": statistics.stdev(values) if len(values) > 1 else None,
        "coefficient_of_variation": statistics.stdev(values) / mean if len(values) > 1 else None,
    }


def compare(baseline: list[dict], candidate: list[dict]) -> dict:
    if not baseline or not candidate:
        raise ValueError("both sides require at least one independent process run")
    names = [row["run"] for row in baseline + candidate]
    if len(set(names)) != len(names):
        raise ValueError("a process run may not appear twice in the comparison")
    contract = baseline[0]["contract"]
    for row in baseline + candidate:
        if row["contract"] != contract:
            different = [key for key in contract if contract[key] != row["contract"][key]]
            raise ValueError(f"{row['run']}: experimental contract differs: {different}")
    for label, rows in (("baseline", baseline), ("candidate", candidate)):
        if len({row["binary"] for row in rows}) != 1:
            raise ValueError(f"{label}: mixes different native binaries")
    a, b = summarize(baseline), summarize(candidate)
    return {
        "baseline_summary": a,
        "candidate_summary": b,
        "median_throughput_ratio": b["median_tokens_per_second"] / a["median_tokens_per_second"],
        "contract": contract,
        "runs": [
            {key: value for key, value in row.items() if key not in ("contract", "final_parameter_samples")}
            for row in baseline + candidate
        ],
        "numerical_diagnostics_vs_first_baseline": {
            row["run"]: numerical_diagnostics(baseline[0], row) for row in baseline + candidate
        },
        "interpretation": "descriptive independent-run statistics; no per-step pseudo-replication or automatic significance claim",
    }


def repeatability(rows: list[dict]) -> dict:
    if len(rows) < 2 or len({row["binary"] for row in rows}) != 1:
        raise ValueError("A/A needs at least two independent runs of one binary")
    # Reuse the same identity/contract/uniqueness checks and numerical comparison.
    checked = compare(rows[:1], rows[1:])
    return {
        "repeatability_summary": summarize(rows),
        **{
            key: value
            for key, value in checked.items()
            if key not in ("baseline_summary", "candidate_summary", "median_throughput_ratio")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--baseline", nargs="+", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--candidate", nargs="+")
    mode.add_argument("--repeatability", action="store_true", help="A/A repeatability of the --baseline runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("report already exists")
    baseline = [read_run(args.run_root / name) for name in args.baseline]
    result = (
        repeatability(baseline)
        if args.repeatability
        else compare(
            baseline,
            [read_run(args.run_root / name) for name in args.candidate],
        )
    )
    write_json(args.output, result)
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("contract", "runs", "numerical_diagnostics_vs_first_baseline")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
