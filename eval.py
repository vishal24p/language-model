
"""Evaluate baseline, residual, and residual MLP language models."""

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from language_model.models import (
    SingleHeadLanguageModel,
    ResidualSingleHeadLanguageModel,
    ResidualMLPLanguageModel,
)
from language_model.data import val_data


CHECKPOINT_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "experiments/exp03_learning_rate/models/exp03_lr_3e-4_50k.pt"
)

VAL_PATH = Path("data/val.bin")
OUTPUT_PATH = CHECKPOINT_PATH.parent.parent / "evaluations" / f"{CHECKPOINT_PATH.stem}.json"

EVAL_WINDOWS = 1024
BATCH_SIZE = 32


def evaluate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    config = checkpoint["config"]
    context = config["context_length"]

    architecture = checkpoint.get("architecture", "baseline")

    if architecture == "baseline":
        model = SingleHeadLanguageModel(**config).to(device)
    elif architecture == "residual":
        model = ResidualSingleHeadLanguageModel(**config).to(device)
    elif architecture == "residual_mlp":
        model = ResidualMLPLanguageModel(**config).to(device)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Fixed validation windows for a fair comparison.
    last_start = len(val_data) - context - 1

    if last_start < 0:
        raise ValueError(
            "Validation data is shorter than one context window."
        )

    starts = np.linspace(
        0,
        last_start,
        EVAL_WINDOWS,
        dtype=np.int64,
    )

    loss_sum = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for offset in range(0, len(starts), BATCH_SIZE):
            batch_starts = starts[offset:offset + BATCH_SIZE]

            x = np.stack([
                val_data[int(s):int(s) + context]
                for s in batch_starts
            ])

            y = np.stack([
                val_data[int(s) + 1:int(s) + context + 1]
                for s in batch_starts
            ])

            x = torch.from_numpy(
                x.astype(np.int64)
            ).to(device)

            y = torch.from_numpy(
                y.astype(np.int64)
            ).to(device)

            logits = model(x)

            loss_sum += F.cross_entropy(
                logits.reshape(-1, config["vocab_size"]),
                y.reshape(-1),
                reduction="sum",
            ).item()

            correct += (
                logits.argmax(dim=-1) == y
            ).sum().item()

            total += y.numel()

    val_loss = loss_sum / total

    results = {
        "checkpoint": CHECKPOINT_PATH.as_posix(),
        "architecture": architecture,
        "train_steps": checkpoint.get("train_steps"),
        "validation_file": str(VAL_PATH),
        "validation_sha256": hashlib.sha256(
            VAL_PATH.read_bytes()
        ).hexdigest(),
        "windows": EVAL_WINDOWS,
        "evaluated_tokens": total,
        "context_length": context,
        "validation_loss": val_loss,
        "perplexity": math.exp(val_loss),
        "top1_accuracy_percent": 100 * correct / total,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(results, indent=2))
    print(f"Saved results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    evaluate()
