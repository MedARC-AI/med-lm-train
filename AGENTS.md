# AGENTS.md

This file provides guidance to Codex, Claude Code, and other coding agents when working with code in this repository.

## Overview

med-lm-train provides a CLI tool (`medarc_slurm`) for generating and submitting single-node SLURM jobs for SFT and RL training of medical language models, built on PRIME-RL.

`prime-rl/` is a pinned external git submodule — do not modify.

## Commands

```bash
uv sync                                        # Install deps
uv sync --extra flash-attn                     # With Flash Attention v2
uv sync --extra flash-attn-3                   # With FA2 + FA3 (H100s)

pytest tests/                                   # Run tests
pytest tests/test_medarc_slurm.py::test_name    # Single test
ruff check medarc_rl tests                      # Lint
ruff format medarc_rl tests                     # Format
```

## Architecture

### CLI (`medarc_rl/medarc_slurm.py`)

Typer-based CLI with two commands (`sft` and `rl`). Each command:
1. Loads and resolves TOML configs using PRIME-RL's Pydantic config classes
2. Renders a Jinja2 SLURM template (`medarc_rl/slurm_templates/`)
3. Writes the script + resolved configs to the output directory
4. Submits via `sbatch` (or prints in `--dry-run` mode)

### RL Launcher (`medarc_rl/launchers/rl_local.py`)

Modified version of PRIME-RL's `rl_local()` for shared-node environments. Handles GPU isolation via `CUDA_VISIBLE_DEVICES`, per-process cache separation, and coordinated multi-process lifecycle with thread-based monitoring.

### Config System

TOML-based configs with inheritance via PRIME-RL's `toml_files` mechanism. Example configs in `examples/`. Resolved configs are written to the output directory for reproducibility.

## Constraints

- RL jobs: total GPUs (train + infer) must be 2-8, or use `--single-gpu` for 1
- NCCL broadcast is only compatible with `async_level=1`
- Ruff line length: 120
