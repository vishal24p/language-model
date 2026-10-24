# Experiment 03: Learning Rate

**What:** Continue two copies of one 40k checkpoint for 10k steps at rates `3e-4` and `1e-4`.

**Why:** Test whether smaller updates help.

**Result:** Rate `3e-4` had lower loss (5.8260 vs 5.8671).

**Files:** [Plots and metrics](outputs/), [evaluations](evaluations/), [checkpoints](models/).

**Needs:** `../exp02_controlled_training/models/exp02_controlled_40k.pt`.

**Run:** `uv run python -m experiments.exp03_learning_rate.train`

[Full case study](https://vishal24p.vercel.app/projects/30k-language-model/)
