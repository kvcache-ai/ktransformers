#!/usr/bin/env python3
"""Prepare text-only, single-turn Kimi data for LF's built-in empty template."""

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path

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
        raise ValueError(
            "Expected prompt/answer and an optional system; no history, tools or media."
        )
    for key in ("prompt", "answer"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise ValueError(f"{key} must be a non-empty string.")
    if not isinstance(record.get("system", ""), str):
        raise ValueError("system must be a string.")
    if any(marker in text for text in record.values() for marker in CONTROL_MARKERS):
        raise ValueError(
            "Input contains chat/thinking markers; do not format data twice."
        )

    messages = []
    if record.get("system"):
        messages.append({"role": "system", "content": record["system"]})
    messages.append({"role": "user", "content": record["prompt"]})
    options = dict(tokenize=False, thinking=False, enable_thinking=False)
    # Some Kimi tokenizers print every encode call. Keep the CLI output bounded.
    with redirect_stdout(io.StringIO()):
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, **options
        )
        full = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": record["answer"]}],
            add_generation_prompt=False,
            **options,
        )
        if not prompt.endswith("<think></think>") or not full.startswith(prompt):
            raise ValueError(
                "Tokenizer does not provide the expected Kimi non-thinking boundary."
            )
        answer = full[len(prompt) :]
        if answer != record["answer"] + "<|im_end|>":
            raise ValueError("Tokenizer changed the answer or its expected end marker.")
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer, add_special_tokens=False)
        full_ids = tokenizer.encode(full, add_special_tokens=False)
    if not answer_ids or prompt_ids + answer_ids != full_ids:
        raise ValueError(
            "Separate prompt/answer encoding differs from native full-dialogue tokens."
        )
    return {"prompt": prompt, "answer": answer}, prompt_ids, answer_ids


def convert_file(tokenizer, path):
    raw = path.read_bytes()
    records = json.loads(raw)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: expected a non-empty JSON array.")
    converted = []
    digest = hashlib.sha256()
    max_tokens = 0
    for index, record in enumerate(records):
        try:
            result, prompt_ids, answer_ids = convert_record(tokenizer, record)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}, row {index}: {error}") from error
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


def write_json(path, value):
    content = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(content)
    return hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="Kimi model/tokenizer directory or Hub ID"
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="Training prompt/answer JSON"
    )
    parser.add_argument(
        "--eval-input", type=Path, help="Optional, already separated evaluation JSON"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New directory; never overwritten",
    )
    parser.add_argument(
        "--revision", help="Pinned tokenizer revision when using a Hub ID"
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow the model's tokenizer code",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("--output-dir already exists; choose a new directory.")

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
    inputs = {"neko_train": args.input}
    if args.eval_input is not None:
        inputs["neko_eval"] = args.eval_input
    outputs, checks = {}, {}
    for name, path in inputs.items():
        outputs[name], checks[name] = convert_file(tokenizer, path)
    if hashlib.sha256(tokenizer.chat_template.encode()).hexdigest() != template_hash:
        raise RuntimeError("The tokenizer's chat template changed during preparation.")

    # Validate every split before creating output; never drop, shuffle or truncate rows.
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
        "shuffled": False,
        "splits": checks,
    }
    write_json(args.output_dir / "preprocessing_manifest.json", manifest)
    print(json.dumps({"output_dir": str(args.output_dir), "splits": checks}, indent=2))


if __name__ == "__main__":
    main()
