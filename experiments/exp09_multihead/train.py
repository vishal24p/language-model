
"""Experiment 09: Train four-head attention with the existing MLP."""

import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from language_model.models import (
    ResidualMLPLanguageModel,
    FourHeadResidualMLPLanguageModel,
)
from language_model.data import get_batch
from language_model.training import set_seed, validation_loss


SEED = 42

CONFIG = {
    "vocab_size": 16_384,
    "context_length": 128,
    "embed_dim": 64,
    "head_size": 64,
}

STEPS = 75_000
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
VAL_INTERVAL = 1_000
VAL_BATCHES = 20
SAVE_STEPS = {20_000, 50_000, 75_000}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path("experiments/exp09_multihead")
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

CONTROL_DIR = Path("experiments/exp08_mlp_comparison")
CONTROL_CHECKPOINT = (
    CONTROL_DIR / "models/exp08_residual_mlp_75k.pt"
)
CONTROL_METRICS = CONTROL_DIR / "outputs/metrics.csv"


def get_val_loss(model):
    return validation_loss(
        model,
        get_batch,
        BATCH_SIZE,
        CONFIG["context_length"],
        CONFIG["vocab_size"],
        VAL_BATCHES,
        DEVICE,
    )


def save_checkpoint(model, optimizer, step):
    path = MODEL_DIR / f"exp09_residual_mlp_4head_{step // 1000}k.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": CONFIG,
            "architecture": "residual_mlp_4head",
            "num_heads": 4,
            "train_steps": step,
            "seed": SEED,
            "learning_rate": LEARNING_RATE,
        },
        path,
    )

    print(f"Saved: {path}", flush=True)


def main():
    # Require the completed Experiment 08 baseline.
    if not CONTROL_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Missing existing baseline: {CONTROL_CHECKPOINT}"
        )

    if not CONTROL_METRICS.is_file():
        raise FileNotFoundError(
            f"Missing baseline metrics: {CONTROL_METRICS}"
        )

    control = torch.load(
        CONTROL_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )

    if (
        control.get("architecture") != "residual_mlp"
        or control.get("train_steps") != STEPS
        or control.get("seed") != SEED
        or control.get("config") != CONFIG
        or control.get("learning_rate") != LEARNING_RATE
    ):
        raise ValueError("Experiment 08 settings do not match.")

    # Read the already recorded baseline curve.
    with CONTROL_METRICS.open(newline="", encoding="utf-8") as file:
        baseline_rows = [
            row for row in csv.DictReader(file)
            if row["architecture"] == "residual_mlp"
        ]

    if not baseline_rows:
        raise ValueError("No residual_mlp baseline measurements found.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Prevent accidental overwriting or repeated training.
    if list(MODEL_DIR.glob("exp09_*.pt")):
        raise FileExistsError(
            "Experiment 09 checkpoints already exist. "
            "This script starts from scratch; it does not resume."
        )

    # Verify identical starting weights for shared parameters.
    set_seed(SEED)
    baseline = ResidualMLPLanguageModel(**CONFIG)

    set_seed(SEED)
    model = FourHeadResidualMLPLanguageModel(**CONFIG)

    assert all(
        torch.equal(value, model.state_dict()[key])
        for key, value in baseline.state_dict().items()
    ), "Initial model weights differ."

    baseline_params = sum(p.numel() for p in baseline.parameters())
    new_params = sum(p.numel() for p in model.parameters())

    assert baseline_params == new_params

    del baseline
    model = model.to(DEVICE)

    # Match the Experiment 08 batch-sampling seed.
    set_seed(SEED)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print(f"Device: {DEVICE}", flush=True)
    print(f"Architecture: 4-head attention + MLP", flush=True)
    print(f"Parameters: {new_params:,}", flush=True)
    print(f"Training steps: {STEPS}", flush=True)

    steps = []
    val_losses = []

    running_loss = torch.zeros((), device=DEVICE)
    start_time = time.perf_counter()

    metrics_path = OUTPUT_DIR / "metrics.csv"

    with metrics_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "architecture",
            "step",
            "train_loss",
            "val_loss",
            "elapsed_seconds",
        ])

        for step in range(1, STEPS + 1):
            model.train()

            x, y = get_batch(
                "train",
                BATCH_SIZE,
                CONFIG["context_length"],
                DEVICE,
            )

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, CONFIG["vocab_size"]),
                y.reshape(-1),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.detach()

            if step % VAL_INTERVAL == 0:
                train_loss = (
                    running_loss / VAL_INTERVAL
                ).item()

                running_loss.zero_()

                val_loss = get_val_loss(model)
                elapsed = time.perf_counter() - start_time

                steps.append(step)
                val_losses.append(val_loss)

                writer.writerow([
                    "residual_mlp_4head",
                    step,
                    train_loss,
                    val_loss,
                    elapsed,
                ])
                file.flush()

                print(
                    f"step {step} | "
                    f"train {train_loss:.4f} | "
                    f"val {val_loss:.4f} | "
                    f"{elapsed / 60:.1f} min",
                    flush=True,
                )

            if step in SAVE_STEPS:
                save_checkpoint(model, optimizer, step)

    # Compare against the existing Experiment 08 measurements.
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        [int(row["step"]) for row in baseline_rows],
        [float(row["val_loss"]) for row in baseline_rows],
        label="1 Head + MLP (Experiment 08)",
    )

    ax.plot(
        steps,
        val_losses,
        label="4 Heads + MLP (Experiment 09)",
    )

    ax.set_title("Experiment 09 — Single vs Multi-Head Attention")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation loss (sampled batches)")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.savefig(
        OUTPUT_DIR / "validation_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nExperiment 09 training completed.")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
