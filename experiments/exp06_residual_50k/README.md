# Experiment 06: Residual to 50k

**What:** Continue training the residual model from 20k to 50k steps.

**Why:** Check whether more training still helps.

**Result:** Loss fell from 5.9657 to 5.6317. Different batches mean this does not isolate the residual connection.

**Files:** [Plots and metrics](outputs/), [evaluations](evaluations/), [checkpoints](models/).

No training runner is included in this checkout.

[Full case study](https://vishal24p.vercel.app/projects/30k-language-model/)
