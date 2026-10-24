# Experiment 01: 30k vs 40k

**What:** Compare the released 30k model with a separate 40k run.

**Why:** Test whether more training improved predictions.

**Result:** The 40k model had higher loss (6.2334 vs 6.0398). Different weights and batches made this test inconclusive.

**Files:** [Plots](outputs/), [40k evaluation](evaluations/40k_baseline.json).

**Run:** `uv run python -m experiments.exp01_30k_vs_40k.train`

[Full case study](https://vishal24p.vercel.app/projects/30k-language-model/)
