"""Observe unfiltered per-rank loss and completed optimizer steps, without changing loss."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

from contracts import require, write_json


def main():
    import torch
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
    from llamafactory.train.tuner import run_exp
    from transformers import TrainerCallback

    config = json.loads(Path(sys.argv[1]).read_text())
    require(
        config["max_steps"] == 1 and config["gradient_accumulation_steps"] == 1,
        "Smoke test requires one step",
    )
    require(
        config["use_kt"] is True and config["finetuning_type"] == "lora",
        "KT LoRA must be enabled",
    )
    require(config["logging_nan_inf_filter"] is False, "Do not filter invalid losses")
    record = {
        "rank": int(os.environ.get("RANK", "0")),
        "raw_losses": [],
        "global_step": 0,
        "optimizer_steps": 0,
        "train_end": False,
    }
    output = Path(os.environ["KT_E2E_RANK_EVIDENCE"]) / f"rank-{record['rank']}.json"
    original = CustomSeq2SeqTrainer.training_step

    def observe(self, *args, **kwargs):
        loss = original(self, *args, **kwargs)
        require(
            isinstance(loss, torch.Tensor) and loss.numel() == 1,
            "Expected scalar training loss",
        )
        value = float(loss.detach().float().cpu())
        require(math.isfinite(value), "Non-finite raw training loss")
        record["raw_losses"].append(value)
        write_json(output, record)
        return loss

    class Evidence(TrainerCallback):
        def on_optimizer_step(self, args, state, control, **kwargs):
            record["optimizer_steps"] += 1
            write_json(output, record)

        def on_step_end(self, args, state, control, **kwargs):
            record["global_step"] = int(state.global_step)
            write_json(output, record)

        def on_train_end(self, args, state, control, **kwargs):
            record["global_step"] = int(state.global_step)
            record["train_end"] = True
            write_json(output, record)

    CustomSeq2SeqTrainer.training_step = observe
    try:
        run_exp(args=config, callbacks=[Evidence()])
    finally:
        CustomSeq2SeqTrainer.training_step = original


if __name__ == "__main__":
    main()
