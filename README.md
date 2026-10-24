# Small Language Model Experiments

A small causal language model built with PyTorch. The experiments test training time, learning rate, residual connections, and an MLP.

**Recent model:** Residual + MLP, trained for 75k steps. See my [Hugging Face profile](https://huggingface.co/vishal24p) and the [30k language-model case study](https://vishal24p.vercel.app/projects/30k-language-model/).

## Layout

- `language_model/` — shared model, data, seeding, and validation code.
- `experiments/` — one folder per experiment. Each folder has its README, runner when available, and `outputs/`.
- Each experiment has `models/` (ignored by Git), `outputs/` (plots and metrics), and `evaluations/` (JSON results).
- `artifacts/legacy_graphs/` — older graphs that are not mapped to one experiment.
- `hf_release/` — recent model release files, ignored by Git.
- `data/` — local training data. This folder is ignored by Git.

## Run

Run commands from the project root after `uv sync`:

```bash
uv run python -m experiments.exp01_30k_vs_40k.train
uv run python -m experiments.exp03_learning_rate.train
uv run python -m experiments.exp04_residual_retrofit.train
uv run python -m experiments.exp05_residual_from_scratch.train
uv run python -m experiments.exp07_residual_75k.train
uv run python -m experiments.exp08_mlp_comparison.train
uv run python eval.py experiments/exp06_residual_50k/models/exp06_scratch_residual_50k.pt
uv run python generate.py
```

Experiments 02 and 06 have saved outputs, but no runner in this checkout. Checkpoint files are not committed. Training requires local files in `data/`. `generate.py` uses the tokenizer at `~/tokenizer/tokenizer`.
