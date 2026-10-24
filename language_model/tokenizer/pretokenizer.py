import regex as re


PATTERN = re.compile(
    r"""'(?i:s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def text_to_chunks(text):
    chunks = PATTERN.findall(text)

    return [
        list(chunk.encode("utf-8"))
        for chunk in chunks
    ]