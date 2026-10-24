# Testing

Run the automated suite with:

```bash
uv run --group test python -m pytest
```

Pytest and pytest-cov are the test tools. Tests live in `tests/` and use the
`test_*.py` naming convention. The suite measures `language_model.tokenizer`
and requires 100% line coverage. Use temporary JSON model fixtures for unit
tests; cover encoding, decoding, Unicode, special tokens, and invalid input.

The generator also has a manual smoke check: `uv run python generate.py`, then
enter `exit` at the prompt. It requires a local checkpoint in `experiments/`.
