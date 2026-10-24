# Experiment 05: Residual From Scratch

**What:** Train baseline and residual models from scratch for 20k steps with matched starts and batches.

**Why:** Test the residual connection without changing a trained model.

**Result:** The residual model had lower loss (5.9657 vs 6.1692) with no extra parameters.

**Files:** [Plots and metrics](outputs/), [evaluations](evaluations/), [checkpoints](models/).

**Run:** `uv run python -m experiments.exp05_residual_from_scratch.train`

[Full case study](https://vishal24p.vercel.app/projects/30k-language-model/)
