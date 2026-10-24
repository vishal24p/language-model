"""Compare two learning rates while continuing the same 40k checkpoint."""

import csv
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from language_model.models import SingleHeadLanguageModel
from language_model.data import get_batch
from language_model.training import set_seed, validation_loss


# ---- experiment settings ----

CHECKPOINT = Path("experiments/exp02_controlled_training/models/exp02_controlled_40k.pt")
MODEL_DIR = Path("experiments/exp03_learning_rate/models")
RESULT_DIR = Path("experiments/exp03_learning_rate/outputs")

BRANCHES = (("lr_3e-4", 3e-4), ("lr_1e-4", 1e-4))
EXTRA_STEPS = 10_000
BATCH_SIZE = 32
LOG_INTERVAL = 20
VAL_INTERVAL = 500
VAL_BATCHES = 20
SEED = 123  # Reset before each branch: same training and validation batches.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def plot_loss(label, train_steps, train_losses, val_steps, val_losses):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_steps, train_losses, label="Train")
    ax.plot(val_steps, val_losses, "o-", markersize=3, label="Validation")
    ax.set(title=f"Learning-rate experiment: {label}",
           xlabel="Training step", ylabel="Cross-entropy loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(RESULT_DIR / f"{label}_loss.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_branch(label, learning_rate, checkpoint):
    config = checkpoint["config"]
    start_step = checkpoint["train_steps"]
    end_step = start_step + EXTRA_STEPS
    vocab_size = config["vocab_size"]
    context_length = config["context_length"]

    # Both branches begin from identical model AND optimizer states.
    model = SingleHeadLanguageModel(**config).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Loading AdamW state restores the OLD learning rate. Override it afterward.
    for group in optimizer.param_groups:
        group["lr"] = learning_rate

    # Both branches see the same sampled data in the same order.
    set_seed(SEED)

    train_steps, train_losses = [], []
    val_steps = [start_step]
    val_losses = [validation_loss(
        model, get_batch, BATCH_SIZE, context_length, vocab_size,
        VAL_BATCHES, DEVICE,
    )]
    print(f"\n{label}: {start_step} -> {end_step} | lr={learning_rate}")
    print(f"step {start_step} | val loss {val_losses[-1]:.4f}")

    metrics_path = RESULT_DIR / f"{label}_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["step", "split", "loss", "tokens_per_second"])
        writer.writerow([start_step, "val", val_losses[-1], ""])

        loss_sum = torch.zeros((), device=DEVICE)
        start_time = time.perf_counter()
        last_train_loss = None

        for step in range(start_step + 1, end_step + 1):
            x, y = get_batch("train", BATCH_SIZE, context_length, DEVICE)
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size), y.reshape(-1)
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach()

            if (step - start_step) % LOG_INTERVAL == 0:
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start_time
                last_train_loss = (loss_sum / LOG_INTERVAL).item()
                speed = BATCH_SIZE * context_length * LOG_INTERVAL / elapsed
                train_steps.append(step)
                train_losses.append(last_train_loss)
                writer.writerow([step, "train", last_train_loss, speed])
                loss_sum.zero_()
                start_time = time.perf_counter()

            if (step - start_step) % VAL_INTERVAL == 0:
                val = validation_loss(
                    model, get_batch, BATCH_SIZE, context_length, vocab_size,
                    VAL_BATCHES, DEVICE,
                )
                val_steps.append(step)
                val_losses.append(val)
                writer.writerow([step, "val", val, ""])
                file.flush()
                plot_loss(label, train_steps, train_losses, val_steps, val_losses)
                print(f"step {step} | train {last_train_loss:.4f} | val {val:.4f}")
                start_time = time.perf_counter()  # Exclude validation and plotting.

    output = MODEL_DIR / f"exp03_{label}_{end_step // 1000}k.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "train_steps": end_step,
        "learning_rate": learning_rate,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "source_checkpoint": str(CHECKPOINT),
    }, output)
    print(f"Saved: {output}")
    return val_steps, val_losses


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Missing starting checkpoint: {CHECKPOINT}")

    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    if "optimizer_state_dict" not in checkpoint:
        raise ValueError("Checkpoint has no optimizer state; use the Exp 02 40k checkpoint")
    if checkpoint["train_steps"] != 40_000:
        raise ValueError("Expected a checkpoint saved at step 40,000")
    if checkpoint.get("batch_size", BATCH_SIZE) != BATCH_SIZE:
        raise ValueError("Batch size differs from the starting checkpoint")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE} | Starting checkpoint: {CHECKPOINT}")

    results = {}
    for label, lr in BRANCHES:
        results[label] = run_branch(label, lr, checkpoint)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, (steps, losses) in results.items():
        ax.plot(steps, losses, "o-", markersize=3, label=label)
    ax.set(title="Validation loss: 3e-4 vs 1e-4",
           xlabel="Training step", ylabel="Cross-entropy loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(RESULT_DIR / "validation_comparison.png",
                dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("\nTraining finished. Compare the two 50k models using eval.py.")
    print(f"Comparison graph: {RESULT_DIR / 'validation_comparison.png'}")


if __name__ == "__main__":
    main()
