"""Compatibility entry point for Experiment 01."""

from runpy import run_module


if __name__ == "__main__":
    run_module("experiments.exp01_30k_vs_40k.train", run_name="__main__")
