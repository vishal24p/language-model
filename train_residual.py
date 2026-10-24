"""Compatibility entry point for Experiment 04."""

from runpy import run_module


if __name__ == "__main__":
    run_module("experiments.exp04_residual_retrofit.train", run_name="__main__")
