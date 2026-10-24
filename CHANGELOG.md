# Changelog

## [0.1.1] - 2026-10-24

### Added
- Generate text with `uv run python generate.py` using the bundled tokenizer files; generation still needs a local model checkpoint.

### Fixed
- Replace invalid UTF-8 model output instead of terminating generation.
