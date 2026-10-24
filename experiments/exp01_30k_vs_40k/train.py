import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save PNG files without opening a desktop window.
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from language_model.models import SingleHeadLanguageModel
from language_model.data import get_batch
from language_model.training import set_seed, validation_loss

SEED = 42

set_seed(SEED)
# ---- config ----

VOCAB_SIZE = 16384
CONTEXT_LENGTH = 128
EMBED_DIM = 64
HEAD_SIZE = 64

BATCH_SIZE = 32
LEARNING_RATE = 3e-4
TRAIN_STEPS = 40000
VAL_INTERVAL = 500

LOG_INTERVAL = 20
VAL_BATCHES = 20

device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_DIR = Path("experiments/exp01_30k_vs_40k/models")
GRAPH_DIR = Path("experiments/exp01_30k_vs_40k/outputs")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(exist_ok=True)


# ---- model ----

model = SingleHeadLanguageModel(
    vocab_size=VOCAB_SIZE,
    context_length=CONTEXT_LENGTH,
    embed_dim=EMBED_DIM,
    head_size=HEAD_SIZE,
).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ---- validation ----

@torch.no_grad()
def get_val_loss():
    return validation_loss(
        model, get_batch, BATCH_SIZE, CONTEXT_LENGTH,
        VOCAB_SIZE, VAL_BATCHES, device,
    )


# ---- graphs ----

train_steps, train_losses = [], []
val_steps, val_losses = [], []
train_speeds = []


def save_graphs():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_steps, train_losses, label="Training loss")
    ax.plot(val_steps, val_losses, marker="o", label="Validation loss")
    ax.set_title("Single-Head Language Model — Training")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(GRAPH_DIR / "40k_training_loss.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_steps, train_speeds)
    ax.set_title("Training speed")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Tokens per second")
    ax.grid(alpha=0.25)
    fig.savefig(GRAPH_DIR / "40k_training_speed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)



# ---- checkpoints ----

def save_checkpoint(step):
    checkpoint_path = MODEL_DIR / f"exp01_40k_{step // 1000}k.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "vocab_size": VOCAB_SIZE,
                "context_length": CONTEXT_LENGTH,
                "embed_dim": EMBED_DIM,
                "head_size": HEAD_SIZE,
            },
            "train_steps": step,
            "seed": SEED,
        },
        checkpoint_path,
    )

    print(f"Checkpoint saved: {checkpoint_path}")

# ---- training ----

model.train()

initial_val_loss = get_val_loss()
val_steps.append(0)
val_losses.append(initial_val_loss)
print(f"step 0 | val loss {initial_val_loss:.4f}")

running_loss = torch.zeros((), device=device)
start_time = time.perf_counter()

for step in range(1, TRAIN_STEPS + 1):
    x, y = get_batch(
        "train", BATCH_SIZE, CONTEXT_LENGTH, device,
    )

    logits = model(x)
    loss = F.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE),
        y.reshape(-1),
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.detach()

    if step % LOG_INTERVAL == 0:
        if device == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start_time
        avg_train_loss = (running_loss / LOG_INTERVAL).item()
        tokens_per_second = (
            BATCH_SIZE * CONTEXT_LENGTH * LOG_INTERVAL / elapsed
        )

        train_steps.append(step)
        train_losses.append(avg_train_loss)
        train_speeds.append(tokens_per_second)

        running_loss.zero_()
        start_time = time.perf_counter()

    if step % VAL_INTERVAL == 0:
        val_loss = get_val_loss()
        val_steps.append(step)
        val_losses.append(val_loss)
        save_graphs()

        print(
            f"step {step} | "
            f"train loss {avg_train_loss:.4f} | "
            f"val loss {val_loss:.4f}"
        )
        start_time = time.perf_counter()  # Exclude validation and plotting time.


# ---- save final model ----

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": VOCAB_SIZE,
            "context_length": CONTEXT_LENGTH,
            "embed_dim": EMBED_DIM,
            "head_size": HEAD_SIZE,
        },
        "train_steps": TRAIN_STEPS,
    },
    MODEL_DIR / "exp01_40k.pt",
)

print("Training finished.")
print(f"Final model: {MODEL_DIR / 'exp01_40k.pt'}")
print(f"Loss graph: {GRAPH_DIR / 'training_loss.png'}")
print(f"Speed graph: {GRAPH_DIR / 'training_speed.png'}")
