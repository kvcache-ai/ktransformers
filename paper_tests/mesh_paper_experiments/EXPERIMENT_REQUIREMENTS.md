# MESH Paper Experiment Requirements

This folder is a paper-oriented experiment harness. It is intentionally separate
from `kt-kernel/test`, because these scripts are not unit tests; they generate
raw data, CSV tables, and figures for the MICRO paper.

## Required Experiments

| ID | Paper need | Script | Primary outputs |
| --- | --- | --- | --- |
| E01 | Fill model statistics: total params, expert params, expert block count, tensor sizes, AMX/BF16 layout facts. | `exp01_model_inventory.py` | `model_inventory.json`, `expert_tensor_summary.csv` |
| E02 | Fill storage data-path facts: direct sequential read throughput, buffered read throughput, expert-shaped read throughput, request-size effects. | `exp02_storage_data_path.py` | `storage_read_summary.csv`, `storage_read_summary.json`, optional expert-shaped TSV |
| E03 | Fill residency turning point: decode throughput vs. cgroup memory budget for mmap and MESH. | `exp03_budget_turning_point.py` | `turning_point_runs.csv`, `turning_point_summary.csv` |
| E04 | Fill mmap failure mode: page faults, page-cache/cgroup accounting, memory.current breakdown at a constrained budget. | `exp04_mmap_failure_breakdown.py` | `memory_samples.csv`, `mmap_failure_summary.json`, `perf_stat.txt` |
| E05 | Fill prefill overlap/IO behavior: compute time, submit/wait time, read volume, iostat read rate, uncovered time. | `exp05_prefill_overlap.py` | `prefill_overlap_by_layer.csv`, `prefill_overlap_summary.json` |
| E06 | Fill Heat contribution: hit-rate convergence for base policy, Heat-enabled policy, and optional oracle/ablation traces. | `exp06_heat_convergence.py` | `heat_convergence.csv`, `heat_convergence_summary.json` |
| E07 | Fill cross-token/locality evidence: Jaccard similarity vs. token distance and long-tail expert activation distribution. | `exp07_temporal_locality.py` | `temporal_jaccard.csv`, `expert_longtail.csv`, `temporal_locality_summary.json` |
| E08 | Fill policy comparison table: SIEVE, S3FIFO, SLRU, W-TinyLFU, LRU, RoundRobin. | `exp08_policy_sweep.py` | `policy_sweep_runs.csv`, `policy_sweep_summary.csv` |
| E09 | Fill workload generality table: LongBench, HumanEval, MT-Bench. | `exp09_workload_sweep.py` | `workload_sweep_runs.csv`, `workload_sweep_summary.csv` |
| E10 | Build paper-ready aggregate values from all experiment folders. | `exp10_aggregate_paper_values.py` | `paper_values.json`, `paper_values.tex`, `paper_tables.md` |

## Runtime Command Template Contract

Runner scripts use command templates instead of hard-coded server paths. A
template can reference these placeholders:

```text
{run_dir}
{variant}
{budget_gb}
{policy}
{workload}
{repeat}
```

Example:

```bash
python exp03_budget_turning_point.py \
  --outdir /mnt/data3/work/mesh_paper/e03_turning \
  --budgets-gb 40,32,24,16,8 \
  --repeats 3 \
  --mesh-cmd-template 'KT_IO_BACKEND=IOURING KT_ENABLE_CACHE_STATS=1 KT_RESIDENCY_POLICY=sieve KT_CACHE_STATS_DUMP={run_dir}/cache_stats.jsonl bash run_mesh_request.sh --budget {budget_gb}' \
  --mmap-cmd-template 'KT_IO_BACKEND=MMAP bash run_mmap_request.sh --budget {budget_gb}'
```

Each script preserves raw stdout/stderr under its run directory. CSV/JSON
outputs are derived artifacts and can be regenerated.

## Trace Inputs Already Supported

- SGLang/KTransformers runtime logs with `Prefill batch` and `Decode batch`.
- MESH cache stats JSONL emitted by `KT_ENABLE_CACHE_STATS=1`.
- MESH prefill static/scratch CSV generated from
  `KT_MESH_PREFILL_STREAM_TRACE=1`.
- MESH prefill expert-frequency JSONL emitted by
  `KT_MESH_PREFILL_EXPERT_FREQ_PATH`.
- `iostat -dxm` JSONL captures used by prior MESH experiments.

## Data Hygiene

Every experiment should write into a new timestamped output directory or an
explicit `--outdir`. Raw files should not be overwritten unless the caller
chooses the same output path intentionally.
