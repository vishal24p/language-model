"""From the sctrach with the residual connection 50k steps"""


"""Train baseline and residual models from the same initialization."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from language_model.models import SingleHeadLanguageModel, ResidualSingleHeadLanguageModel
from language_model.data import get_batch
from language_model.training import set_seed, validation_loss


SEED = 42
TRAIN_STEPS = 20_000
VAL_INTERVAL = 500
VAL_BATCHES = 20
BATCH_SIZE = 32
LEARNING_RATE = 3e-4

CONFIG = {
    "vocab_size": 16_384,
    "context_length": 128,
    "embed_dim": 64,
    "head_size": 64,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUTPUT_DIR = Path("experiments/exp05_residual_from_scratch/outputs")
MODEL_DIR = Path("experiments/exp05_residual_from_scratch/models")


def get_val_loss(model):
    return validation_loss(
        model, get_batch, BATCH_SIZE, CONFIG["context_length"],
        CONFIG["vocab_size"], VAL_BATCHES, DEVICE,
    )


def train_model(name, model_class):
    # Reset before each run for matching initialization and batches.
    set_seed(SEED)

    model = model_class(**CONFIG).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE
    )

    rows = []

    initial_loss = get_val_loss(model)
    rows.append([name, 0, "", initial_loss])

    print(
        f"{name} | step 0 | val loss {initial_loss:.4f}",
        flush=True,
    )

    accumulated_loss = 0.0

    for step in range(1, TRAIN_STEPS + 1):
        x, y = get_batch(
            "train", BATCH_SIZE, CONFIG["context_length"], DEVICE
        )

        logits = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, CONFIG["vocab_size"]),
            y.reshape(-1),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        accumulated_loss += loss.item()

        if step % VAL_INTERVAL == 0:
            train_loss = accumulated_loss / VAL_INTERVAL
            accumulated_loss = 0.0

            val_loss = get_val_loss(model)

            rows.append([name, step, train_loss, val_loss])

            print(
                f"{name} | step {step} | "
                f"train {train_loss:.4f} | "
                f"val {val_loss:.4f}",
                flush=True,
            )

    checkpoint_path = (
        MODEL_DIR / f"exp05_scratch_{name}_{TRAIN_STEPS // 1000}k.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": CONFIG,
            "architecture": name,
            "train_steps": TRAIN_STEPS,
            "seed": SEED,
            "learning_rate": LEARNING_RATE,
        },
        checkpoint_path,
    )

    print(f"Saved {checkpoint_path}", flush=True)
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(
        f"Starting from scratch | seed {SEED} | device {DEVICE}",
        flush=True,
    )

    rows = []

    experiments = (
        ("baseline", SingleHeadLanguageModel),
        ("residual", ResidualSingleHeadLanguageModel),
    )

    for name, model_class in experiments:
        rows.extend(train_model(name, model_class))

    # Save experiment results.
    with (OUTPUT_DIR / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "architecture", "step", "train_loss", "val_loss"
        ])
        writer.writerows(rows)

    # Plot validation loss comparison.
    fig, ax = plt.subplots(figsize=(10, 6))

    for name in ("baseline", "residual"):
        points = [row for row in rows if row[0] == name]

        ax.plot(
            [row[1] for row in points],
            [row[3] for row in points],
            marker="o",
            markersize=3,
            label=name,
        )

    ax.set(
        xlabel="Training step",
        ylabel="Validation loss",
        title="Baseline vs residual — trained from scratch",
    )

    ax.grid(alpha=0.25)
    ax.legend()

    fig.savefig(
        OUTPUT_DIR / "validation_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(f"Experiment finished. Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
