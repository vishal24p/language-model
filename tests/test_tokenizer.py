import json
from pathlib import Path

import pytest

from language_model.tokenizer import Tokenizer


@pytest.fixture
def tokenizer(tmp_path):
    model_path = tmp_path / "tokenizer_model.json"
    mergeable_ranks = [
        {"bytes": bytes([value]).hex(), "rank": value}
        for value in range(256)
    ]
    mergeable_ranks.extend(
        [
            {"bytes": b"ab".hex(), "rank": 259},
            {"bytes": b"bc".hex(), "rank": 258},
            {"bytes": b"abc".hex(), "rank": 257},
        ]
    )
    model_path.write_text(
        json.dumps(
            {
                "pattern": "unused by the tokenizer implementation",
                "vocab_size": 301,
                "special_tokens": {"<|endoftext|>": 300},
                "mergeable_ranks": mergeable_ranks,
            }
        ),
        encoding="utf-8",
    )
    return Tokenizer(model_path)


def test_encode_merges_the_lowest_rank_pair_first(tokenizer):
    assert tokenizer.encode("abc") == [257]


def test_encode_and_decode_round_trip_unicode_and_unmerged_bytes(tokenizer):
    text = "Hello, world! 你好"

    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_encode_empty_text(tokenizer):
    assert tokenizer.encode("") == []


def test_decode_special_end_of_text_token(tokenizer):
    assert tokenizer.decode([300]) == "<|endoftext|>"


def test_encode_special_token(tokenizer):
    assert tokenizer.encode_special("<|endoftext|>") == 300


def test_unknown_special_token_raises(tokenizer):
    with pytest.raises(ValueError, match="Unknown special token"):
        tokenizer.encode_special("<|unknown|>")


def test_unknown_token_id_raises(tokenizer):
    with pytest.raises(ValueError, match="Unknown token ID: 999"):
        tokenizer.decode([999])


def test_decode_replaces_invalid_utf8_bytes(tokenizer):
    assert tokenizer.decode([255]) == "\ufffd"


def test_bundled_tokenizer_model_round_trips_text():
    model_path = (
        Path(__file__).resolve().parents[1]
        / "language_model"
        / "tokenizer"
        / "tokenizer_model.json"
    )
    tokenizer = Tokenizer(model_path)
    text = "Hello, world! 你好"

    assert tokenizer.vocab_size == 16_384
    assert tokenizer.encode(text) == [
        72,
        13153,
        44,
        883,
        33,
        32,
        228,
        189,
        160,
        229,
        165,
        189,
    ]
    assert tokenizer.decode(tokenizer.encode(text)) == text
