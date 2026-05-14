# AGENTS.md

This file provides guidance to Codex, Claude Code, and other coding agents when working with code in this repository.

## Project Rules

`prime-rl/` is a pinned external git submodule. Update the submodule pointer when asked, but do not edit files inside `prime-rl/`.

When updating to a newer PRIME-RL revision, target the new PRIME-RL API/config shape directly. Do not add legacy compatibility shims, aliases, migration layers, or support for removed/deprecated options unless the user explicitly asks for backward compatibility. Rename stale local concepts/tests to match current PRIME-RL terminology.

`uv` does not inherit `[tool.uv.sources]`, indexes, or overrides from the `prime-rl/` path dependency. If a PRIME-RL update needs custom wheels, workspace packages such as `prime-rl-configs`, or private indexes, add the minimum required resolver configuration to this repo's root `pyproject.toml`.

Use the README for human-facing setup, install, and CLI usage details. Keep this file focused on instructions that are easy for agents to miss.

## Commands

```bash
uv run pytest tests/                                   # Run tests
uv run pytest tests/test_medarc_slurm.py::test_name    # Single test
uv run ruff check medarc_rl tests                      # Lint
uv run ruff format medarc_rl tests                     # Format
```

Testing scope:
- `pyproject.toml` sets `pytest` `testpaths = ["tests"]`, so default collection is scoped correctly.
- Do not run `prime-rl/tests/` by default.
- Avoid `pytest .` (or other explicit repo-root paths), which can widen collection and include `prime-rl/tests/`.
- Only run this repo's tests under `tests/` unless the user explicitly asks to run PRIME-RL tests.

## Architecture Notes

`medarc_rl/medarc_slurm.py` generates single-node SLURM jobs for PRIME-RL SFT/RL. It loads PRIME-RL TOML configs, applies wrapper-owned fields such as GPU split and output directory, writes resolved configs and scripts, then submits via `sbatch` or prints in `--dry-run` mode.

`medarc_rl/medarc_train.py` is the local runner for PRIME-RL SFT/RL. It resolves configs the same way as `medarc_slurm`, writes resolved configs, and launches local training.

`medarc_rl/launchers/rl_local.py` is a modified PRIME-RL local RL launcher for shared-node environments. It handles GPU isolation via `CUDA_VISIBLE_DEVICES`, per-process cache separation, dynamic ports, and coordinated multi-process lifecycle.

TOML-based configs with inheritance via PRIME-RL's `toml_files` mechanism. Example configs in `examples/`. Resolved configs are written to the output directory for reproducibility.

Both `medarc_slurm` and `medarc_train` support PRIME-RL-style nested CLI overrides (for example `-- --wandb.name run1`). Wrapper-owned fields (especially GPU split / deployment and `output_dir`) take precedence over passthrough overrides.

Shared config/TOML helper functions live in `medarc_rl/utils.py`; do not import underscore helpers from `medarc_rl.medarc_slurm` into other modules.

## Constraints

- RL jobs: total GPUs (train + infer) must be 2-8, or use `--single-gpu` for 1
- NCCL broadcast is only compatible with `async_level=1`
- Ruff line length: 120
