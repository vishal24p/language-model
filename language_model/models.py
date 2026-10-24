
import torch as t
import torch.nn as nn


class SingleHeadAttention(nn.Module):
    def __init__(self, embed_dim, head_size, context_length):
        super().__init__()

        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

        self.head_size = head_size

        causal_mask = t.tril(
            t.ones(context_length, context_length, dtype=t.bool)
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x):
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / (self.head_size ** 0.5)

        sequence_length = x.shape[1]
        mask = self.causal_mask[:sequence_length, :sequence_length]

        attention_scores = attention_scores.masked_fill(
            ~mask, float("-inf")
        )

        attention_weights = t.softmax(attention_scores, dim=-1)

        context = attention_weights @ value
        return context


class SingleHeadLanguageModel(nn.Module):
    def __init__(self, vocab_size, context_length, embed_dim, head_size):
        super().__init__()

        self.context_length = context_length

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_length, embed_dim)

        self.attention = SingleHeadAttention(
            embed_dim=embed_dim,
            head_size=head_size,
            context_length=context_length,
        )

        self.lm_head = nn.Linear(head_size, vocab_size)

    def forward(self, token_ids):
        batch_size, sequence_length = token_ids.shape

        if sequence_length > self.context_length:
            raise ValueError("Sequence length exceeds context length.")

        positions = t.arange(
            sequence_length, device=token_ids.device
        )

        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(positions)

        x = token_vectors + position_vectors

        context = self.attention(x)

        logits = self.lm_head(context)
        return logits


class ResidualSingleHeadLanguageModel(SingleHeadLanguageModel):
    """Original model with a permanent attention residual connection."""

    def __init__(self, vocab_size, context_length, embed_dim, head_size):
        if embed_dim != head_size:
            raise ValueError(
                "Residual connection requires embed_dim == head_size."
            )

        super().__init__(
            vocab_size, context_length, embed_dim, head_size
        )

    def forward(self, token_ids):
        _, sequence_length = token_ids.shape

        if sequence_length > self.context_length:
            raise ValueError("Sequence length exceeds context length.")

        positions = t.arange(
            sequence_length, device=token_ids.device
        )

        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(positions)

        x = token_vectors + position_vectors

        # Permanent residual connection
        context = x + self.attention(x)

        logits = self.lm_head(context)
        return logits


class ResidualMLPLanguageModel(ResidualSingleHeadLanguageModel):
    """Residual attention followed by a residual feed-forward network."""

    def __init__(self, vocab_size, context_length, embed_dim, head_size):
        super().__init__(vocab_size, context_length, embed_dim, head_size)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, token_ids):
        _, sequence_length = token_ids.shape

        if sequence_length > self.context_length:
            raise ValueError("Sequence length exceeds context length.")

        positions = t.arange(sequence_length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)

        x = x + self.attention(x)
        x = x + self.mlp(x)

        return self.lm_head(x)



class FourHeadAttention(SingleHeadAttention):
    """Split the attention features into four independent heads."""

    def __init__(self, embed_dim, head_size, context_length):
        if head_size % 4 != 0:
            raise ValueError("head_size must be divisible by 4.")

        super().__init__(embed_dim, head_size, context_length)

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape
        head_dim = self.head_size // 4

        # Split 64 features into 4 heads of 16 features each.
        query = self.query(x).reshape(
            batch_size, sequence_length, 4, head_dim
        ).transpose(1, 2)

        key = self.key(x).reshape(
            batch_size, sequence_length, 4, head_dim
        ).transpose(1, 2)

        value = self.value(x).reshape(
            batch_size, sequence_length, 4, head_dim
        ).transpose(1, 2)

        # Each head computes its own attention weights.
        scores = query @ key.transpose(-2, -1)
        scores = scores / (head_dim ** 0.5)

        mask = self.causal_mask[
            :sequence_length, :sequence_length
        ]
        scores = scores.masked_fill(~mask, float("-inf"))

        weights = t.softmax(scores, dim=-1)
        context = weights @ value

        # Combine the four heads back into 64 features.
        return context.transpose(1, 2).contiguous().reshape(
            batch_size, sequence_length, self.head_size
        )


class FourHeadResidualMLPLanguageModel(ResidualMLPLanguageModel):
    """Experiment 08 model with four attention heads instead of one."""

    def __init__(self, vocab_size, context_length, embed_dim, head_size):
        super().__init__(
            vocab_size, context_length, embed_dim, head_size
        )

        # Keep the original attention weight initialization.
        original_attention = self.attention

        self.attention = FourHeadAttention(
            embed_dim, head_size, context_length
        )
        self.attention.load_state_dict(
            original_attention.state_dict()
        )
