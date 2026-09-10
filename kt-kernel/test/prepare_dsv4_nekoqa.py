#!/usr/bin/env python3
"""Create reproducible V4 chat-mode NekoQA acceptance splits (internal use only)."""

import argparse
import hashlib
import importlib.util
import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    encoder_path = Path(args.model) / "encoding" / "encoding_dsv4.py"
    spec = importlib.util.spec_from_file_location("encoding_dsv4", encoder_path)
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    split_rows, prompts, selected, lengths = {}, {}, {}, {}
    source_hashes = {}
    for split, filename, count in (
        ("train", "nekoqa_train.json", 1024),
        ("validation", "nekoqa_eval.json", 128),
    ):
        source_path = root / filename
        source_hashes[filename] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        rows = json.loads(source_path.read_text())
        indices = list(range(len(rows)))
        random.Random(42).shuffle(indices)
        encoded, prompt_set, records, sizes = [], set(), [], []
        for index in indices:
            row = rows[index]
            prompt = row["instruction"] + (
                "\n" + row["input"] if row.get("input") else ""
            )
            if prompt in prompt_set:
                continue
            messages = [{"role": "user", "content": prompt}]
            prefix = encoder.encode_messages(messages, thinking_mode="chat")
            full = encoder.encode_messages(
                messages + [{"role": "assistant", "content": row["output"]}],
                thinking_mode="chat",
            )
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            ids = tokenizer.encode(full, add_special_tokens=False)
            assert ids[: len(prefix_ids)] == prefix_ids, (
                "assistant boundary is not token-aligned"
            )
            if len(prefix_ids) >= 1023:
                continue
            labels = [-100] * len(prefix_ids) + ids[len(prefix_ids) :]
            ids, labels = ids[:1024], labels[:1024]
            assert any(value != -100 for value in labels)
            encoded.append(
                {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}
            )
            prompt_set.add(prompt)
            records.append(
                {
                    "source_index": index,
                    "prompt": prompt,
                    "reference": row["output"],
                    "encoded_prompt": prefix,
                }
            )
            sizes.append(len(ids))
            if len(encoded) == count:
                break
        assert len(encoded) == count
        split_rows[split], prompts[split], selected[split], lengths[split] = (
            encoded,
            prompt_set,
            records,
            sizes,
        )
    assert not prompts["train"].intersection(prompts["validation"]), (
        "train/eval prompt leakage"
    )
    DatasetDict(
        {name: Dataset.from_list(rows) for name, rows in split_rows.items()}
    ).save_to_disk(output / "nekoqa")
    DatasetDict(
        {
            "train": Dataset.from_list(split_rows["train"]),
            "validation_train_probe": Dataset.from_list(split_rows["train"][:16]),
            "validation_heldout": Dataset.from_list(split_rows["validation"]),
        }
    ).save_to_disk(output / "nekoqa-probes")
    # Full-token throughput smoke: concatenate independently encoded records; keep prompt loss masked.
    flat_ids, flat_labels = [], []
    for row in split_rows["train"]:
        flat_ids.extend(row["input_ids"])
        flat_labels.extend(row["labels"])
    full = [
        {
            "input_ids": flat_ids[i : i + 1024],
            "attention_mask": [1] * 1024,
            "labels": flat_labels[i : i + 1024],
        }
        for i in range(0, min(len(flat_ids) // 1024, 64) * 1024, 1024)
    ]
    DatasetDict(
        {
            "train": Dataset.from_list(full),
            "validation": Dataset.from_list(split_rows["validation"][:8]),
        }
    ).save_to_disk(output / "smoke-s1024")
    (output / "selected.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2)
    )
    manifest = {
        "license": "UNVERIFIED_INTERNAL_USE_ONLY",
        "seed": 42,
        "thinking_mode": "chat",
        "encoder_sha256": hashlib.sha256(encoder_path.read_bytes()).hexdigest(),
        "source_sha256": source_hashes,
        "cutoff": 1024,
        "prompt_overlap": 0,
        "splits": {
            name: {
                "count": len(values),
                "min_tokens": min(values),
                "max_tokens": max(values),
                "mean_tokens": sum(values) / len(values),
            }
            for name, values in lengths.items()
        },
        "smoke": {
            "examples": len(full),
            "tokens_per_example": 1024,
            "packing": "causal concatenation",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
