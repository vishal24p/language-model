# Modular Experiments Implementation Plan

> **For agentic workers:** Follow this plan task by task. Keep experiment settings and results unchanged.

**Goal:** Organize shared language-model code, experiment runners, checkpoints, and outputs without duplicating model code.

**Architecture:** Put reusable model, data, seed, and validation code in a small `language_model` package. Give each study an `experiments/expXX_*` package with its README, runner when one exists, and local `models/`, `outputs/`, and `evaluations/` folders. Keep the Hugging Face release self-contained.

**Tech Stack:** Python, PyTorch, `uv`.

**Spec:** Current user request and the selected shared-package/separate-runner layout.

## Global Constraints

- Keep all existing local checkpoint and output contents.
- Keep experiment logic, hyperparameters, and results unchanged.
- Track plots and metrics; ignore checkpoints, raw data, and local caches.
- Preserve the self-contained `hf_release/` bundle.
- Preserve the user's existing edits to `pyproject.toml` and `uv.lock`.

## Review Focus

- Checkpoint lineage still resolves across experiment folders.
- Training and evaluation outputs write to the correct experiment folder.
- Training scripts still run as modules and through existing root commands.
- The Hugging Face bundle remains independent.

---

### Task 1: Create the shared package and experiment folders

- Move model definitions and data loading into `language_model/`.
- Move each runner and experiment README into its named `experiments/expXX_*` folder.
- Move each experiment's graphs and metric files into that folder's `outputs/`.
- Keep unassigned legacy graphs in a clearly named archive folder.

### Task 2: Move checkpoint and evaluation artifacts

- Move checkpoints and evaluation JSON files into the matching experiment folders.
- Prefix checkpoint filenames with their experiment ID and ignore `experiments/**/models/`.
- Update all checkpoint, import, and output paths in local entry points.
- Keep small plots and metrics trackable; add narrow ignore rules for local and large files.

### Task 3: Keep entry points and document commands

- Add thin root wrappers for the existing training commands.
- Add a root README with the new layout and commands.
- Confirm no root command or release-bundle path points at the old locations.
