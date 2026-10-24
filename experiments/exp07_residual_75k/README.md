# Experiment 07: Residual to 75k

**What:** Continue the residual model from 50k to 75k steps.

**Why:** Check for gains before adding another component.

**Result:** Loss fell from 5.6317 to 5.5125, with smaller gains over time.

**Files:** [Plots and metrics](outputs/), [evaluations](evaluations/), [checkpoints](models/).

**Needs:** `../exp06_residual_50k/models/exp06_scratch_residual_50k.pt`.

**Run:** `uv run python -m experiments.exp07_residual_75k.train`

[Full case study](https://vishal24p.vercel.app/projects/30k-language-model/)
