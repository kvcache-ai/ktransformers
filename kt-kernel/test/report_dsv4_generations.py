#!/usr/bin/env python3
"""Summarize paired held-out generation; lexical metrics are not a quality verdict."""

import argparse
import json
from pathlib import Path


def rouge_l_chars(prediction, reference):
    a, b = list("".join(prediction.split())), list("".join(reference.split()))
    row = [0] * (len(b) + 1)
    for x in a:
        previous = 0
        for j, y in enumerate(b, 1):
            saved = row[j]
            row[j] = previous + 1 if x == y else max(row[j], row[j - 1])
            previous = saved
    return 2 * row[-1] / max(len(a) + len(b), 1)


def summarize(rows):
    texts = [row["response"]["text"] for row in rows]
    return {
        "count": len(rows),
        "nonempty": sum(bool(text.strip()) for text in texts),
        "contains_meow": sum("喵" in text for text in texts),
        "length_limited": sum(
            row["response"]["meta_info"]["finish_reason"]["type"] == "length"
            for row in rows
        ),
        "mean_rouge_l_character_f1": sum(
            rouge_l_chars(text, row["reference"]) for text, row in zip(texts, rows)
        )
        / len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--trained", required=True)
    parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    base, trained = (
        [json.loads(line) for line in Path(path).read_text().splitlines()]
        for path in (args.base, args.trained)
    )
    assert base and len(base) == len(trained)
    for a, b in zip(base, trained):
        assert (a["source_index"], a["encoded_prompt"], a["reference"]) == (
            b["source_index"],
            b["encoded_prompt"],
            b["reference"],
        )
    report = {
        "base": summarize(base),
        "trained": summarize(trained),
        "changed_answers": sum(
            a["response"]["text"] != b["response"]["text"]
            for a, b in zip(base, trained)
        ),
        "note": "Same held-out prompts and BF16 non-expert convention. Character ROUGE-L and presence of 喵 are diagnostic indicators, not semantic-quality scores.",
    }
    prefix = Path(args.output_prefix)
    prefix.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    lines = [
        "# DeepSeek V4 NekoQA held-out 对照",
        "",
        "同一批输入；base 与训练后模型均使用 BF16 非专家权重。词面重叠和“喵”出现率只作诊断，不等于语义质量。",
        "",
        "```json",
        json.dumps(report, ensure_ascii=False, indent=2),
        "```",
        "",
    ]
    for index, (a, b) in enumerate(zip(base, trained), 1):
        lines.extend(
            [
                f"## {index}. 原始样本 {a['source_index']}",
                "",
                "### 输入",
                "",
                a["prompt"],
                "",
                "### Base",
                "",
                a["response"]["text"],
                "",
                "### LoRA",
                "",
                b["response"]["text"],
                "",
                "### 参考答案",
                "",
                a["reference"],
                "",
            ]
        )
    prefix.with_suffix(".md").write_text("\n".join(lines))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
