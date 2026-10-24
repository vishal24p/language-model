# Experiment 08: Add an MLP

**What:** Compare a residual model with the same model plus an MLP.

**Why:** Test whether an MLP improves next-token prediction.

**Result:** At 75k, residual+MLP loss was lower (5.4414 vs 5.5106). The MLP model has more parameters.

**Files:** [Plots and metrics](outputs/), [evaluations](evaluations/), [checkpoints](models/).

**Run:** `uv run python -m experiments.exp08_mlp_comparison.train`

[Project portfolio](https://vishal24p.vercel.app/projects/30k-language-model/)
