import json
import argparse
from pathlib import Path
from ktransformers.sft.metrics import ComputeSimilarity
from transformers import AutoTokenizer
from transformers.trainer_utils import EvalPrediction

def load_pred_ref(pred_file: Path):
    data = json.loads(pred_file.read_text(encoding="utf-8"))
    preds, refs = [], []
    for it in data:
        preds.append("" if it.get("prediction") is None else str(it.get("prediction")))
        refs.append("" if it.get("label") is None else str(it.get("label")))
    return preds, refs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    args = parser.parse_args()

    pred_file = Path(args.pred_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_file = output_dir / "metrics.json"

    preds, refs = load_pred_ref(pred_file)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    compute_metrics = ComputeSimilarity(tokenizer)
    enc_pred = tokenizer(preds, add_special_tokens=False, padding=True, return_tensors="np")
    enc_ref  = tokenizer(refs,  add_special_tokens=False, padding=True, return_tensors="np")
    ep = EvalPrediction(predictions=enc_pred["input_ids"], label_ids=enc_ref["input_ids"])
    metrics = compute_metrics(ep, compute_result=True)

    with metric_file.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[OK] sample length: {len(preds)}")
    print(f"[OK] saved to: {metric_file}")

if __name__ == "__main__":
    main()
