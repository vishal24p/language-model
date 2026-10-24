
"""Experiment 08: compare residual attention with and without an MLP."""

import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from language_model.models import ResidualSingleHeadLanguageModel, ResidualMLPLanguageModel
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

MODEL_DIR = Path("experiments/exp08_mlp_comparison/models")
GRAPH_DIR = Path("experiments/exp08_mlp_comparison/outputs")

MODELS = {
    "residual": ResidualSingleHeadLanguageModel,
    "residual_mlp": ResidualMLPLanguageModel,
}


def get_val_loss(model):
    return validation_loss(
        model, get_batch, BATCH_SIZE, CONFIG["context_length"],
        CONFIG["vocab_size"], VAL_BATCHES, DEVICE,
    )


def save_checkpoint(name, model, optimizer, step):
    path = MODEL_DIR / f"exp08_{name}_{step // 1000}k.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": CONFIG,
            "architecture": name,
            "train_steps": step,
            "seed": SEED,
            "learning_rate": LEARNING_RATE,
        },
        path,
    )

    print(f"Saved: {path}", flush=True)


def train(name, model_class, writer, csv_file):
    # Initialize shared weights identically.
    set_seed(SEED)
    model = model_class(**CONFIG).to(DEVICE)

    # Reset RNG after initialization so both runs sample
    # the same training and validation batches.
    set_seed(SEED)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print(
        f"\n{name}: {parameter_count:,} parameters on {DEVICE}",
        flush=True,
    )

    steps = []
    train_losses = []
    val_losses = []

    running_loss = torch.zeros((), device=DEVICE)
    start_time = time.perf_counter()

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
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            writer.writerow([
                name,
                step,
                train_loss,
                val_loss,
                elapsed,
            ])
            csv_file.flush()

            print(
                f"{name} | step {step} | "
                f"train {train_loss:.4f} | "
                f"val {val_loss:.4f} | "
                f"{elapsed / 60:.1f} min",
                flush=True,
            )

        if step in SAVE_STEPS:
            save_checkpoint(name, model, optimizer, step)

    return steps, train_losses, val_losses


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    # Check that the models share identical initial weights.
    set_seed(SEED)
    baseline = ResidualSingleHeadLanguageModel(**CONFIG)

    set_seed(SEED)
    candidate = ResidualMLPLanguageModel(**CONFIG)

    assert all(
        torch.equal(value, candidate.state_dict()[key])
        for key, value in baseline.state_dict().items()
    ), "Shared initial weights differ"

    # Both architectures must produce the same output shape.
    x = torch.zeros((1, 4), dtype=torch.long)

    assert (
        baseline(x).shape
        == candidate(x).shape
        == (1, 4, CONFIG["vocab_size"])
    )

    del baseline, candidate

    curves = {}
    metrics_path = GRAPH_DIR / "metrics.csv"

    with metrics_path.open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "architecture",
            "step",
            "train_loss",
            "val_loss",
            "elapsed_seconds",
        ])

        for name, model_class in MODELS.items():
            curves[name] = train(
                name,
                model_class,
                writer,
                csv_file,
            )

    # Plot validation loss for both models.
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, (steps, _, val_losses) in curves.items():
        ax.plot(
            steps,
            val_losses,
            label=name,
        )

    ax.set_title("Experiment 08 — Residual vs Residual + MLP")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation loss (sampled batches)")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.savefig(
        GRAPH_DIR / "validation_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nTraining complete.")
    print("Run eval.py on both models' checkpoints.")


if __name__ == "__main__":
    main()
