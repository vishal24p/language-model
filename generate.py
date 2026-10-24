
from pathlib import Path

import torch

from language_model.models import (
    SingleHeadLanguageModel,
    ResidualSingleHeadLanguageModel,
    ResidualMLPLanguageModel,
)
from language_model.tokenizer import Tokenizer


# ---- paths ----

TOKENIZER_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "language_model"
    / "tokenizer"
    / "tokenizer_model.json"
)

# Change this path to switch checkpoints.
CHECKPOINT_PATH = Path("experiments/exp06_residual_50k/models/exp06_scratch_residual_50k.pt")


# ---- generation settings ----

MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 50

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---- load YOUR tokenizer ----

tokenizer = Tokenizer(TOKENIZER_MODEL_PATH)


# ---- load trained model ----

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True,
)

config = checkpoint["config"]

architecture = checkpoint.get("architecture", "baseline")

if architecture == "residual":
    model = ResidualSingleHeadLanguageModel(**config).to(device)
elif architecture == "residual_mlp":
    model = ResidualMLPLanguageModel(**config).to(device)
elif architecture == "baseline":
    model = SingleHeadLanguageModel(**config).to(device)
else:
    raise ValueError(f"Unknown architecture: {architecture}")

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()


# ---- sanity check ----

assert tokenizer.vocab_size == config["vocab_size"], (
    f"Tokenizer vocab = {tokenizer.vocab_size}, "
    f"model vocab = {config['vocab_size']}"
)


# ---- generation ----

@torch.no_grad()
def generate(prompt):

    prompt_ids = tokenizer.encode(prompt)

    if not prompt_ids:
        return ""

    token_ids = list(prompt_ids)

    for _ in range(MAX_NEW_TOKENS):

        context = token_ids[-config["context_length"]:]

        x = torch.tensor(
            [context],
            dtype=torch.long,
            device=device,
        )

        logits = model(x)

        # Predict the next token from the last position.
        logits = logits[:, -1, :]

        # Temperature
        logits = logits / TEMPERATURE

        # Top-k sampling
        top_values, top_indices = torch.topk(
            logits,
            k=TOP_K,
            dim=-1,
        )

        probabilities = torch.softmax(
            top_values,
            dim=-1,
        )

        sampled_index = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        next_token = top_indices.gather(
            -1,
            sampled_index,
        ).item()

        # Stop at end-of-text.
        if next_token == tokenizer.eot_id:
            break

        token_ids.append(next_token)

    generated_ids = token_ids[len(prompt_ids):]

    return tokenizer.decode(generated_ids)


# ---- interactive mode ----

print()
print("Single-Head Language Model loaded.")
print(f"Checkpoint: {CHECKPOINT_PATH}")
print(f"Architecture: {architecture}")
print(f"Device: {device}")
print(f"Vocabulary: {tokenizer.vocab_size}")
print(f"Context length: {config['context_length']}")
print(f"Embedding dim: {config['embed_dim']}")
print(f"Head size: {config['head_size']}")
print()
print("Type 'exit' to quit.")


while True:

    prompt = input("\nYou: ").strip()

    if prompt.lower() == "exit":
        break

    if not prompt:
        continue

    response = generate(prompt)

    print("Model:", response)
