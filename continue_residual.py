"""Compatibility entry point for Experiment 07."""

from runpy import run_module


if __name__ == "__main__":
    run_module("experiments.exp07_residual_75k.train", run_name="__main__")
