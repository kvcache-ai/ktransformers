#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import ensure_dir, write_csv, write_json


DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


EXPERT_PATTERNS = [
    re.compile(r"layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight"),
    re.compile(r"language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight"),
    re.compile(r"layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<proj>gate_up_proj|down_proj)$"),
    re.compile(r"language_model\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<proj>gate_up_proj|down_proj)$"),
    re.compile(r"layers\.(?P<layer>\d+).*ffn_(?P<proj>gate|up|down)_exps\.(?P<expert>\d+)\.numa\.(?P<numa>\d+)\.(?P<kind>weight|scale|mins)"),
]


def safetensors_headers(model_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(model_dir.glob("*.safetensors")):
        with path.open("rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size))
        data_base = 8 + header_size
        for key, info in header.items():
            if key == "__metadata__":
                continue
            start, end = info["data_offsets"]
            shape = tuple(int(x) for x in info.get("shape", []))
            dtype = str(info.get("dtype", ""))
            rows.append(
                {
                    "key": key,
                    "file": str(path),
                    "dtype": dtype,
                    "shape": shape,
                    "offset": data_base + int(start),
                    "size": int(end) - int(start),
                    "numel": prod(shape),
                }
            )
    return rows


def prod(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def classify_expert(row: dict[str, Any]) -> dict[str, Any] | None:
    key = row["key"]
    for pattern in EXPERT_PATTERNS:
        match = pattern.search(key)
        if match:
            groups = match.groupdict()
            return {
                "layer": int(groups["layer"]),
                "expert": int(groups["expert"]) if groups.get("expert") is not None else None,
                "proj": groups.get("proj"),
                "kind": groups.get("kind", "weight"),
                "numa": int(groups["numa"]) if groups.get("numa") is not None else None,
            }
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Inventory model and expert tensors for MESH paper numbers.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    outdir = ensure_dir(args.outdir)
    config_path = model_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    rows = safetensors_headers(model_dir)
    tensor_rows = []
    total_bytes = 0
    total_params_est = 0.0
    expert_bytes = 0
    expert_params_est = 0.0
    experts_by_layer: dict[int, set[int]] = defaultdict(set)
    expert_tensor_groups: dict[tuple[int, int | None], int] = defaultdict(int)

    for row in rows:
        total_bytes += int(row["size"])
        dtype_bytes = DTYPE_BYTES.get(row["dtype"], None)
        if dtype_bytes:
            total_params_est += int(row["size"]) / dtype_bytes
        expert = classify_expert(row)
        out = dict(row)
        out["is_expert"] = expert is not None
        if expert:
            out.update(expert)
            expert_bytes += int(row["size"])
            if dtype_bytes:
                expert_params_est += int(row["size"]) / dtype_bytes
            layer = int(expert["layer"])
            expert_id = expert["expert"]
            if expert_id is not None:
                experts_by_layer[layer].add(int(expert_id))
            expert_tensor_groups[(layer, expert_id)] += int(row["size"])
        tensor_rows.append(out)

    layer_summaries = []
    for layer in sorted(set(k[0] for k in expert_tensor_groups)):
        layer_keys = [k for k in expert_tensor_groups if k[0] == layer]
        sizes = [expert_tensor_groups[k] for k in layer_keys]
        layer_summaries.append(
            {
                "layer": layer,
                "expert_ids_detected": len(experts_by_layer.get(layer, set())),
                "expert_blocks_detected": len(layer_keys),
                "expert_bytes": sum(sizes),
                "expert_bytes_mib": sum(sizes) / 1048576.0,
                "min_expert_block_mib": min(sizes) / 1048576.0 if sizes else 0,
                "max_expert_block_mib": max(sizes) / 1048576.0 if sizes else 0,
            }
        )

    summary = {
        "model_dir": str(model_dir),
        "config": {
            k: config.get(k)
            for k in (
                "model_type",
                "num_hidden_layers",
                "num_experts",
                "num_routed_experts",
                "num_experts_per_tok",
                "hidden_size",
                "intermediate_size",
                "moe_intermediate_size",
            )
        },
        "safetensors_files": len(list(model_dir.glob("*.safetensors"))),
        "tensor_count": len(rows),
        "total_bytes": total_bytes,
        "total_gib": total_bytes / 1073741824.0,
        "total_params_est": total_params_est,
        "total_params_est_b": total_params_est / 1e9,
        "expert_bytes": expert_bytes,
        "expert_gib": expert_bytes / 1073741824.0,
        "expert_params_est": expert_params_est,
        "expert_params_est_b": expert_params_est / 1e9,
        "expert_bytes_pct": (expert_bytes / total_bytes * 100.0) if total_bytes else None,
        "moe_layers_detected": len(layer_summaries),
        "expert_blocks_detected": len(expert_tensor_groups),
        "layer_summaries": layer_summaries,
    }

    write_json(outdir / "model_inventory.json", summary)
    write_csv(outdir / "expert_tensor_summary.csv", tensor_rows)
    write_csv(outdir / "expert_layer_summary.csv", layer_summaries)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
