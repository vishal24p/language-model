"""Compatibility entry point for Experiment 08."""

from runpy import run_module


if __name__ == "__main__":
    run_module("experiments.exp08_mlp_comparison.train", run_name="__main__")
