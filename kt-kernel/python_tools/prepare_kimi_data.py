#!/usr/bin/env python3
"""Prepare text-only Kimi data using the native template and LF's empty template."""

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import random
import re
import unicodedata

NEKOQA_REVISION = "1b2110c996a8237823b86c1a3d3e8a6762b38430"
NEKOQA_SHA256 = "b4d260ad117c29c9fd64abcb513ad24d62e2fac383640e17ea230d12ae03b849"
STYLE_PROMPT = re.compile(r"猫|喵|neko|catgirl|扮演", re.IGNORECASE)

CONTROL_MARKERS = (
    "<|im_user|>",
    "<|im_assistant|>",
    "<|im_system|>",
    "<|im_middle|>",
    "<|im_end|>",
    "<think>",
    "</think>",
)


def convert_record(tokenizer, record):
    if not isinstance(record, dict) or set(record) - {"prompt", "answer", "system"}:
        raise ValueError("Expected prompt/answer and an optional system; no history, tools or media.")
    for key in ("prompt", "answer"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"{key} must be a non-empty string.")
    if not isinstance(record.get("system", ""), str):
        raise ValueError("system must be a string.")
    if any(marker in text for text in record.values() for marker in CONTROL_MARKERS):
        raise ValueError("Input contains chat/thinking markers; do not format data twice.")

    messages = []
    if record.get("system"):
        messages.append({"role": "system", "content": record["system"]})
    messages.append({"role": "user", "content": record["prompt"]})
    options = dict(tokenize=False, thinking=False, enable_thinking=False)
    # Keep tokenizer stdout out of the CLI's JSON summary.
    with redirect_stdout(io.StringIO()):
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, **options)
        full = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": record["answer"]}],
            add_generation_prompt=False,
            **options,
        )
        if not prompt.endswith("<think></think>") or not full.startswith(prompt):
            raise ValueError("Tokenizer does not provide the expected Kimi non-thinking boundary.")
        answer = full[len(prompt) :]
        if answer != record["answer"] + "<|im_end|>":
            raise ValueError("Tokenizer changed the answer or its expected end marker.")
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        full_ids = tokenizer.encode(full, add_special_tokens=False)
    if not answer_ids or prompt_ids + answer_ids != full_ids:
        raise ValueError("Separate prompt/answer encoding differs from native full-dialogue tokens.")
    return {"prompt": prompt, "answer": answer}, prompt_ids, answer_ids


def convert_file(tokenizer, path):
    raw = path.read_bytes()
    return convert_records(tokenizer, raw, str(path))


def convert_records(tokenizer, raw, source):
    records = json.loads(raw)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{source}: expected a non-empty JSON array.")
    converted = []
    digest = hashlib.sha256()
    max_tokens = 0
    for index, record in enumerate(records):
        try:
            result, prompt_ids, answer_ids = convert_record(tokenizer, record)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{source}, row {index}: {error}") from error
        converted.append(result)
        ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        digest.update(json.dumps([ids, labels], separators=(",", ":")).encode() + b"\n")
        max_tokens = max(max_tokens, len(ids))
    return converted, {
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "rows": len(records),
        "max_untruncated_tokens": max_tokens,
        "input_ids_and_labels_sha256": digest.hexdigest(),
    }


def split_nekoqa(records, *, seed=42, eval_size=500, heldout_size=32):
    """Deduplicate prompts before splitting; reserve unprompted style questions."""
    if not isinstance(records, list) or not 0 < heldout_size < eval_size:
        raise ValueError("Expected an array and 0 < heldout_size < eval_size.")
    unique, seen, invalid, duplicates = [], set(), [], []
    for index, row in enumerate(records):
        if not isinstance(row, dict) or set(row) != {"instruction", "output"}:
            raise ValueError(f"Unexpected NekoQA schema at row {index}.")
        if any(not isinstance(row[key], str) or not row[key].strip() for key in row):
            invalid.append(index)
            continue
        key = " ".join(unicodedata.normalize("NFKC", row["instruction"]).split()).casefold()
        if key in seen:
            duplicates.append(index)
            continue
        seen.add(key)
        unique.append((index, {"prompt": row["instruction"], "answer": row["output"]}))
    if len(unique) <= eval_size:
        raise ValueError("Not enough distinct prompts for training and evaluation.")
    random.Random(seed).shuffle(unique)
    heldout, evaluation = [], []
    for item in unique[:eval_size]:
        if len(heldout) < heldout_size and not STYLE_PROMPT.search(item[1]["prompt"]):
            heldout.append(item)
        else:
            evaluation.append(item)
    if len(heldout) != heldout_size:
        raise ValueError("Not enough held-out questions without explicit style prompts.")
    splits = {"neko_train": unique[eval_size:], "neko_eval": evaluation, "heldout": heldout}
    manifest = {
        "seed": seed,
        "input_rows": len(records),
        "invalid_rows": invalid,
        "duplicate_prompt_rows": duplicates,
        "distinct_prompts": len(unique),
        "source_rows": {name: [index for index, _ in rows] for name, rows in splits.items()},
        "heldout_prompt_filter": STYLE_PROMPT.pattern,
    }
    return {name: [record for _, record in rows] for name, rows in splits.items()}, manifest


def load_nekoqa(path):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != NEKOQA_SHA256:
        raise ValueError("NekoQA SHA256 mismatch; download the documented fixed revision.")
    splits, manifest = split_nekoqa(json.loads(raw))
    manifest.update(repository="liumindmind/NekoQA-10K", revision=NEKOQA_REVISION, sha256=NEKOQA_SHA256)
    return splits, manifest


def write_json(path, value):
    content = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(content)
    return hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Kimi model/tokenizer directory or Hub ID")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--nekoqa", type=Path, help="Pinned public NekoQA-10K.json; deduplicate and split with seed 42")
    source.add_argument("--input", type=Path, help="Training prompt/answer JSON")
    parser.add_argument("--eval-input", type=Path, help="Optional, already separated evaluation JSON")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory; never overwritten",
    )
    parser.add_argument("--revision", help="Pinned tokenizer revision when using a Hub ID")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow the model's tokenizer code",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("--output-dir already exists; choose a new directory.")
    if args.nekoqa is not None and args.eval_input is not None:
        parser.error("--eval-input is for --input; --nekoqa creates its own isolated splits.")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=False,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    if not isinstance(tokenizer.chat_template, str) or not tokenizer.chat_template:
        parser.error("The tokenizer must supply a native chat template.")
    template_hash = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    dataset_manifest, heldout = None, None
    outputs, checks = {}, {}
    if args.nekoqa is not None:
        splits, dataset_manifest = load_nekoqa(args.nekoqa)
        heldout = splits.pop("heldout")
        for name, records in splits.items():
            raw = json.dumps(records, ensure_ascii=False).encode()
            outputs[name], checks[name] = convert_records(tokenizer, raw, name)
    else:
        inputs = {"neko_train": args.input}
        if args.eval_input is not None:
            inputs["neko_eval"] = args.eval_input
        for name, path in inputs.items():
            outputs[name], checks[name] = convert_file(tokenizer, path)
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != template_hash:
        raise RuntimeError("The tokenizer's chat template changed during preparation.")

    # Validate all training splits before creating output; never truncate tokens.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dataset_info = {}
    for name, records in outputs.items():
        filename = name + ".json"
        checks[name]["output_sha256"] = write_json(args.output_dir / filename, records)
        dataset_info[name] = {
            "file_name": filename,
            "columns": {"prompt": "prompt", "response": "answer"},
        }
    write_json(args.output_dir / "dataset_info.json", dataset_info)
    if heldout is not None:
        dataset_manifest["heldout_sha256"] = write_json(args.output_dir / "heldout.json", heldout)
    manifest = {
        "model": args.model,
        "revision": args.revision,
        "tokenizer_class": type(tokenizer).__name__,
        "native_template_sha256": template_hash,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "lf_template": "empty",
        "thinking": False,
        "packing": False,
        "truncated": False,
        "shuffled": args.nekoqa is not None,
        "splits": checks,
        "dataset": dataset_manifest,
    }
    write_json(args.output_dir / "preprocessing_manifest.json", manifest)
    print(json.dumps({"output_dir": str(args.output_dir), "splits": checks}, indent=2))


if __name__ == "__main__":
    main()
