"""Compatibility entry point for Experiment 05."""

from runpy import run_module


if __name__ == "__main__":
    run_module("experiments.exp05_residual_from_scratch.train", run_name="__main__")
