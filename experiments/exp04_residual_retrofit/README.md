# Experiment 04: Residual Retrofit

**What:** Add a residual connection to a trained 50k model and continue to 55k.

**Why:** Test the change on existing model weights.

**Result:** The modified model had loss 5.8030; the unchanged model had 5.7932. The retrofit did not lower loss.

**Files:** [Plots and metrics](outputs/), [evaluations](evaluations/), [checkpoints](models/).

**Needs:** `../exp03_learning_rate/models/exp03_lr_3e-4_50k.pt`.

**Run:** `uv run python -m experiments.exp04_residual_retrofit.train`

[Full case study](https://vishal24p.vercel.app/projects/30k-language-model/)
