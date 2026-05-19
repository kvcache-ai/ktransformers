#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from common import read_json, write_json


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def maybe(path: Path):
    return read_json(path) if path.exists() else None


def tex_macro(name: str, value: Any) -> str:
    if value is None:
        value = "XX"
    if isinstance(value, float):
        text = f"{value:.2f}"
    else:
        text = str(value)
    return f"\\newcommand{{\\{name}}}{{{text}}}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate MESH paper experiment outputs into paper values.")
    parser.add_argument("--root", required=True, help="Directory containing E01..E09 output folders")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    inventory = maybe(root / "e01_model_inventory" / "model_inventory.json")
    turning = read_csv_rows(root / "e03_budget_turning_point" / "turning_point_summary.csv")
    policy = read_csv_rows(root / "e08_policy_sweep" / "policy_sweep_summary.csv")
    workload = read_csv_rows(root / "e09_workload_sweep" / "workload_sweep_summary.csv")
    storage = read_csv_rows(root / "e02_storage_data_path" / "storage_read_summary.csv")
    temporal = maybe(root / "e07_temporal_locality" / "temporal_locality_summary.json")

    values = {
        "inventory": inventory,
        "turning_point": turning,
        "policy_sweep": policy,
        "workload_sweep": workload,
        "storage": storage,
        "temporal_locality": temporal,
    }
    write_json(outdir / "paper_values.json", values)

    macros = []
    if inventory:
        macros += [
            tex_macro("MeshModelParamsB", inventory.get("total_params_est_b")),
            tex_macro("MeshExpertParamsB", inventory.get("expert_params_est_b")),
            tex_macro("MeshExpertBytesPct", inventory.get("expert_bytes_pct")),
            tex_macro("MeshExpertBlocks", inventory.get("expert_blocks_detected")),
        ]
    if temporal:
        macros += [tex_macro("MeshTopHalfActivationShare", temporal.get("top_50pct_activation_share"))]
    if storage:
        best = max(
            (float(r.get("throughput_gib_s") or 0) for r in storage if r.get("direct") in ("1", 1)),
            default=None,
        )
        macros += [tex_macro("MeshDirectSeqReadGiBs", best)]

    (outdir / "paper_values.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

    md = ["# Paper Tables", ""]
    for title, rows in (
        ("Turning Point", turning),
        ("Policy Sweep", policy),
        ("Workload Sweep", workload),
        ("Storage", storage),
    ):
        md += [f"## {title}", ""]
        if not rows:
            md += ["No data.", ""]
            continue
        fields = list(rows[0].keys())
        md.append("| " + " | ".join(fields) + " |")
        md.append("| " + " | ".join("---" for _ in fields) + " |")
        for row in rows:
            md.append("| " + " | ".join(str(row.get(f, "")) for f in fields) + " |")
        md.append("")
    (outdir / "paper_tables.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"outdir": str(outdir), "macro_count": len(macros)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
