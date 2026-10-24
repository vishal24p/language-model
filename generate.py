
"""Generate text using the trained 4-head attention model."""

from pathlib import Path

import torch

from language_model.models import FourHeadResidualMLPLanguageModel
from language_model.tokenizer import Tokenizer


# Model and tokenizer paths
TOKENIZER_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "language_model"
    / "tokenizer"
    / "tokenizer_model.json"
)

CHECKPOINT_PATH = Path(
    "experiments/exp09_multihead/models/exp09_residual_mlp_4head_75k.pt"
)


# Generation settings
MAX_NEW_TOKENS = 100
TEMPERATURE = 0.8
TOP_K = 50

device = "cuda" if torch.cuda.is_available() else "cpu"


# Load tokenizer
tokenizer = Tokenizer(TOKENIZER_MODEL_PATH)


# Load trained model
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=True,
)

if checkpoint["architecture"] != "residual_mlp_4head":
    raise ValueError("Expected a 4-head MLP checkpoint.")

config = checkpoint["config"]

model = FourHeadResidualMLPLanguageModel(**config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

assert tokenizer.vocab_size == config["vocab_size"]


@torch.inference_mode()
def generate(prompt):
    prompt_ids = tokenizer.encode(prompt)

    if not prompt_ids:
        return ""

    token_ids = list(prompt_ids)

    for _ in range(MAX_NEW_TOKENS):
        # Keep tokens within the 128-token context limit.
        context = token_ids[-config["context_length"]:]

        x = torch.tensor(
            [context],
            dtype=torch.long,
            device=device,
        )

        # Predict next-token probabilities.
        logits = model(x)[:, -1, :]
        logits = logits / TEMPERATURE

        # Sample from the top 50 candidates.
        values, indices = torch.topk(logits, k=TOP_K, dim=-1)
        probabilities = torch.softmax(values, dim=-1)

        sampled = torch.multinomial(probabilities, num_samples=1)
        next_token = indices.gather(-1, sampled).item()

        if next_token == tokenizer.eot_id:
            break

        token_ids.append(next_token)

    return tokenizer.decode(token_ids[len(prompt_ids):])


def main():
    print("\n4-Head Attention + MLP Model")
    print(f"Checkpoint: {CHECKPOINT_PATH.name}")
    print(f"Device: {device}")
    print(f"Training steps: {checkpoint['train_steps']}")
    print(f"Context length: {config['context_length']}")
    print("\nType 'exit' to quit.")

    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if prompt.lower() == "exit":
            break

        if prompt:
            print("Model:", generate(prompt))


if __name__ == "__main__":
    main()
