import random

import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def validation_loss(
    model,
    batch_fn,
    batch_size,
    context_length,
    vocab_size,
    batches,
    device,
):
    model.eval()
    total = 0.0

    for _ in range(batches):
        x, y = batch_fn("val", batch_size, context_length, device)
        logits = model(x)
        total += F.cross_entropy(
            logits.reshape(-1, vocab_size),
            y.reshape(-1),
        ).item()

    model.train()
    return total / batches
