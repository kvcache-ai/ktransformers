"""CPU data-contract tests; native-tokenizer and training checks are separate."""

import importlib.util
import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="default")

SOURCE = Path(__file__).resolve().parents[2] / "python_tools/prepare_kimi_data.py"
SPEC = importlib.util.spec_from_file_location("prepare_kimi_data", SOURCE)
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class NativeBoundaryFixture:
    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        assert kwargs == {"tokenize": False, "thinking": False, "enable_thinking": False}
        prompt = "".join(message["content"] for message in messages if message["role"] != "assistant")
        prefix = "<|im_user|>" + prompt + "<|im_assistant|><think></think>"
        return prefix if add_generation_prompt else prefix + messages[-1]["content"] + "<|im_end|>"

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return list(text.encode("utf-8"))


class PrepareKimiDataTest(unittest.TestCase):
    def test_native_boundary_preserves_answer_and_ids(self):
        tokenizer = NativeBoundaryFixture()
        record = {"prompt": "你好", "answer": "你好呀，喵。", "system": "简短回答。"}
        result, prompt_ids, answer_ids = prepare.convert_record(tokenizer, record)
        self.assertEqual(result["answer"], record["answer"] + "<|im_end|>")
        self.assertEqual(
            prompt_ids + answer_ids, tokenizer.encode(result["prompt"] + result["answer"], add_special_tokens=False)
        )

    def test_rejects_preformatted_media_and_empty_inputs(self):
        for record in (
            {"prompt": "<think>already formatted", "answer": "x"},
            {"prompt": "x", "answer": "y", "images": ["image.png"]},
            {"prompt": "", "answer": "y"},
            {"prompt": "x", "answer": "y", "system": None},
        ):
            with self.subTest(record=record), self.assertRaises(ValueError):
                prepare.convert_record(NativeBoundaryFixture(), record)

    def test_rejects_token_boundary_mismatch(self):
        class Broken(NativeBoundaryFixture):
            def encode(self, text, **kwargs):
                return [len(text)]

        with self.assertRaisesRegex(ValueError, "Separate prompt/answer encoding"):
            prepare.convert_record(Broken(), {"prompt": "x", "answer": "y"})

    def test_prompt_deduplication_precedes_deterministic_split(self):
        records = [{"instruction": f"Question {index}", "output": "Answer"} for index in range(20)]
        records += [
            {"instruction": " ＱＵＥＳＴＩＯＮ   0 ", "output": "Alternative"},
            {"instruction": "", "output": "Empty"},
        ]
        splits, evidence = prepare.split_nekoqa(records, eval_size=6, heldout_size=2)
        self.assertEqual(prepare.split_nekoqa(records, eval_size=6, heldout_size=2), (splits, evidence))
        self.assertEqual(evidence["invalid_rows"], [21])
        self.assertEqual(evidence["duplicate_prompt_rows"], [20])
        self.assertEqual(
            {key: len(value) for key, value in splits.items()}, {"neko_train": 14, "neko_eval": 4, "heldout": 2}
        )
        prompts = [record["prompt"] for rows in splits.values() for record in rows]
        self.assertEqual(len(prompts), len(set(prompts)))

    def test_holdout_does_not_use_style_prompts(self):
        records = [{"instruction": f"扮演猫娘{index}", "output": "喵"} for index in range(20)]
        with self.assertRaisesRegex(ValueError, "without explicit style prompts"):
            prepare.split_nekoqa(records, eval_size=6, heldout_size=2)

    def test_unexpected_dataset_schema_fails(self):
        with self.assertRaisesRegex(ValueError, "schema"):
            prepare.split_nekoqa([{"prompt": "x", "answer": "y"}])

    @unittest.skipUnless(os.environ.get("KT_NEKOQA_PATH"), "Set the fixed public NekoQA JSON path")
    def test_public_dataset_digest_and_split_counts(self):
        splits, evidence = prepare.load_nekoqa(Path(os.environ["KT_NEKOQA_PATH"]))
        self.assertEqual(evidence["input_rows"], 10066)
        self.assertEqual(len(evidence["invalid_rows"]), 4)
        self.assertEqual(len(evidence["duplicate_prompt_rows"]), 85)
        self.assertEqual(
            {key: len(value) for key, value in splits.items()}, {"neko_train": 9477, "neko_eval": 468, "heldout": 32}
        )
        self.assertFalse(any(prepare.STYLE_PROMPT.search(row["prompt"]) for row in splits["heldout"]))


if __name__ == "__main__":
    unittest.main()
