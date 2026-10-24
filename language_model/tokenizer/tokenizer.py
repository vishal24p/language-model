import json
from pathlib import Path

from .pretokenizer import text_to_chunks


EOT_TOKEN = "<|endoftext|>"


class Tokenizer:
    def __init__(self, model_path):
        model = json.loads(
            Path(model_path).read_text(encoding="utf-8")
        )

        self.pattern = model["pattern"]
        self.vocab_size = model["vocab_size"]
        self.special_tokens = model["special_tokens"]
        self.eot_id = self.special_tokens[EOT_TOKEN]

        self.ranks = {
            bytes.fromhex(item["bytes"]): item["rank"]
            for item in model["mergeable_ranks"]
        }

        self.vocab = {
            rank: token_bytes
            for token_bytes, rank in self.ranks.items()
        }

    def encode_chunk(self, token_ids):
        pieces = [
            bytes([token_id])
            for token_id in token_ids
        ]

        while len(pieces) >= 2:
            best_index = None
            best_rank = None

            for index in range(len(pieces) - 1):
                candidate = (
                    pieces[index]
                    + pieces[index + 1]
                )

                rank = self.ranks.get(candidate)

                if rank is None:
                    continue

                if (
                    best_rank is None
                    or rank < best_rank
                ):
                    best_rank = rank
                    best_index = index

            if best_index is None:
                break

            pieces = (
                pieces[:best_index]
                + [
                    pieces[best_index]
                    + pieces[best_index + 1]
                ]
                + pieces[best_index + 2:]
            )

        return [
            self.ranks[piece]
            for piece in pieces
        ]

    def encode(self, text):
        encoded = []

        for chunk in text_to_chunks(text):
            encoded.extend(
                self.encode_chunk(chunk)
            )

        return encoded

    def decode(self, token_ids):
        parts = []

        for token_id in token_ids:
            if token_id == self.eot_id:
                parts.append(
                    EOT_TOKEN.encode("utf-8")
                )
                continue

            token_bytes = self.vocab.get(token_id)

            if token_bytes is None:
                raise ValueError(
                    f"Unknown token ID: {token_id}"
                )

            parts.append(token_bytes)

        return b"".join(parts).decode("utf-8", errors="replace")

    def encode_special(self, token):
        if token not in self.special_tokens:
            raise ValueError(
                f"Unknown special token: {token}"
            )

        return self.special_tokens[token]
