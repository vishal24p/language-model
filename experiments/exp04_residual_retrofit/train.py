
"""Continue the 50k checkpoint and test a residual connection."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from language_model.models import (
    SingleHeadLanguageModel,
    ResidualSingleHeadLanguageModel,
)
from language_model.data import get_batch
from language_model.training import set_seed, validation_loss


SOURCE = Path("experiments/exp03_learning_rate/models/exp03_lr_3e-4_50k.pt")
OUTPUT_DIR = Path("experiments/exp04_residual_retrofit/outputs")
MODEL_DIR = Path("experiments/exp04_residual_retrofit/models")

EXTRA_STEPS = 5000
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
VAL_INTERVAL = 500
VAL_BATCHES = 20
SEED = 123

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_val_loss(model, config):
    return validation_loss(
        model, get_batch, BATCH_SIZE, config["context_length"],
        config["vocab_size"], VAL_BATCHES, DEVICE,
    )


def main():
    print("Starting residual connection experiment...", flush=True)
    print(f"Device: {DEVICE}", flush=True)

    checkpoint = torch.load(
        SOURCE,
        map_location=DEVICE,
        weights_only=True,
    )

    if "optimizer_state_dict" not in checkpoint:
        raise ValueError("Checkpoint is missing optimizer state.")

    config = checkpoint["config"]
    start_step = checkpoint["train_steps"]
    end_step = start_step + EXTRA_STEPS

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    experiments = [
        ("baseline", SingleHeadLanguageModel),
        ("residual", ResidualSingleHeadLanguageModel),
    ]

    for name, model_class in experiments:
        print(f"\nTraining: {name}", flush=True)

        set_seed(SEED)

        model = model_class(**config).to(DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        for group in optimizer.param_groups:
            group["lr"] = LEARNING_RATE

        initial_loss = get_val_loss(model, config)

        all_results.append((name, start_step, initial_loss))

        print(
            f"{name} | step {start_step} | "
            f"val loss {initial_loss:.4f}",
            flush=True,
        )

        for step in range(start_step + 1, end_step + 1):
            model.train()

            x, y = get_batch(
                "train",
                BATCH_SIZE,
                config["context_length"],
                DEVICE,
            )

            logits = model(x)

            loss = F.cross_entropy(
                logits.reshape(-1, config["vocab_size"]),
                y.reshape(-1),
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step % VAL_INTERVAL == 0:
                val_loss = get_val_loss(model, config)

                all_results.append((name, step, val_loss))

                print(
                    f"{name} | step {step} | "
                    f"train loss {loss.item():.4f} | "
                    f"val loss {val_loss:.4f}",
                    flush=True,
                )

        path = MODEL_DIR / f"exp04_{name}_{end_step // 1000}k.pt"

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
                "architecture": name,
                "train_steps": end_step,
                "learning_rate": LEARNING_RATE,
                "source_checkpoint": str(SOURCE),
            },
            path,
        )

        print(f"Saved model: {path}", flush=True)

    # Save validation measurements
    csv_path = OUTPUT_DIR / "metrics.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["architecture", "step", "val_loss"])
        writer.writerows(all_results)

    # Compare validation loss
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, _ in experiments:
        points = [
            (step, loss)
            for architecture, step, loss in all_results
            if architecture == name
        ]

        ax.plot(
            [step for step, _ in points],
            [loss for _, loss in points],
            marker="o",
            label=name,
        )

    ax.set_title("Baseline vs Residual Connection")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Validation loss")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.savefig(
        OUTPUT_DIR / "validation_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nExperiment finished.")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
