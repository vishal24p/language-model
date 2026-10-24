
"""Continue the residual model from 50k to 75k steps."""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from language_model.models import ResidualSingleHeadLanguageModel
from language_model.data import get_batch
from language_model.training import set_seed, validation_loss


# ---- Configuration ----

SEED = 42
CHECKPOINT = Path("experiments/exp06_residual_50k/models/exp06_scratch_residual_50k.pt")

END_STEP = 75_000
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
VAL_INTERVAL = 500
VAL_BATCHES = 20
SAVE_INTERVAL = 10_000

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = Path("experiments/exp07_residual_75k/models")
GRAPH_DIR = Path("experiments/exp07_residual_75k/outputs")


# ---- Validation ----

def get_val_loss(model, config):
    return validation_loss(
        model, get_batch, BATCH_SIZE, config["context_length"],
        config["vocab_size"], VAL_BATCHES, DEVICE,
    )


# ---- Save checkpoints ----

def save_checkpoint(model, optimizer, config, step):
    path = MODEL_DIR / f"exp07_scratch_residual_{step // 1000}k.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "architecture": "residual",
            "train_steps": step,
            "seed": SEED,
            "learning_rate": LEARNING_RATE,
        },
        path,
    )

    print(f"Saved checkpoint: {path}", flush=True)


# ---- Training ----

def main():
    set_seed(SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    if checkpoint.get("architecture") != "residual":
        raise ValueError("Checkpoint is not a residual model.")

    if "optimizer_state_dict" not in checkpoint:
        raise ValueError("Checkpoint has no optimizer state.")

    config = checkpoint["config"]
    start_step = checkpoint["train_steps"]

    if start_step >= END_STEP:
        raise ValueError("Checkpoint is already at or past 75k.")

    # Restore trained model weights
    model = ResidualSingleHeadLanguageModel(**config).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Restore the existing AdamW optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for group in optimizer.param_groups:
        group["lr"] = LEARNING_RATE

    print(f"Device: {DEVICE}", flush=True)
    print(f"Seed: {SEED}", flush=True)
    print(f"Starting step: {start_step}", flush=True)
    print(f"Final step: {END_STEP}", flush=True)
    print(f"Learning rate: {LEARNING_RATE}", flush=True)

    steps = []
    train_losses = []
    val_losses = []

    metrics_path = GRAPH_DIR / "metrics.csv"

    with metrics_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.writer(file)
        writer.writerow(["step", "train_loss", "val_loss"])

        initial_val = get_val_loss(model, config)

        steps.append(start_step)
        train_losses.append(float("nan"))
        val_losses.append(initial_val)

        writer.writerow([start_step, "", initial_val])

        print(
            f"step {start_step} | val loss {initial_val:.4f}",
            flush=True,
        )

        running_loss = torch.zeros((), device=DEVICE)

        for step in range(start_step + 1, END_STEP + 1):
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

            running_loss += loss.detach()

            # Evaluate every 500 steps
            if step % VAL_INTERVAL == 0:
                avg_train_loss = (
                    running_loss / VAL_INTERVAL
                ).item()

                running_loss.zero_()

                val_loss = get_val_loss(model, config)

                steps.append(step)
                train_losses.append(avg_train_loss)
                val_losses.append(val_loss)

                writer.writerow([
                    step,
                    avg_train_loss,
                    val_loss,
                ])
                file.flush()

                print(
                    f"step {step} | "
                    f"train loss {avg_train_loss:.4f} | "
                    f"val loss {val_loss:.4f}",
                    flush=True,
                )

            # Save at 60k, 70k and the final 75k
            if step % SAVE_INTERVAL == 0 or step == END_STEP:
                save_checkpoint(
                    model, optimizer, config, step
                )

    # ---- Training graph ----

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(steps, train_losses, label="Training loss")

    ax.plot(
        steps,
        val_losses,
        label="Validation loss",
        marker="o",
        markersize=3,
    )

    ax.set_title("Residual Model — 50k to 75k")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.savefig(
        GRAPH_DIR / "training_loss.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nTraining completed.")
    print(f"Final checkpoint: {MODEL_DIR / 'exp07_scratch_residual_75k.pt'}")
    print(f"Results: {GRAPH_DIR}")


if __name__ == "__main__":
    main()
