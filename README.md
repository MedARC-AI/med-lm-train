# med-lm-train

## Setup + Installation

1. Clone the repository

```bash
git clone --recurse-submodules --shallow-submodules --depth 50 https://github.com/MedARC-AI/med-lm-train.git
cd med-lm-train
```

2. Install [uv](https://docs.astral.sh/uv/)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

3. Install dependencies from the lock file

```bash
uv sync
```

For flash attention support:

```bash
uv sync --extra flash-attn      # flash-attn v2
uv sync --extra flash-attn-3    # flash-attn v2 + v3
```

## Example: Reverse Text (single node, H100)

This walks through running the reverse text example on H100s using `medarc_slurm` to generate and submit single-node SLURM jobs. The configs in [examples/reverse_text/](examples/reverse_text/) are tuned for H100 memory (larger micro-batches than the upstream PRIME-RL defaults).

### SFT (1 GPU)

Generate the SLURM script and resolved config, then submit:

```bash
medarc_slurm sft examples/reverse_text/sft.toml \
    --output-dir runs/reverse-sft \
    --gpus 1
```

To inspect the generated script without submitting:

```bash
medarc_slurm sft examples/reverse_text/sft.toml \
    --output-dir runs/reverse-sft \
    --gpus 1 \
    --dry-run
```

This writes `runs/reverse-sft/sft.sh` and `runs/reverse-sft/configs/trainer.toml`, then prints the `sbatch` command.

### RL (2 GPUs)

RL requires at least 2 GPUs — one for inference and one for training. Using the SFT checkpoint as the starting model:

```bash
medarc_slurm rl examples/reverse_text/rl.toml \
    --output-dir runs/reverse-rl \
    --train-gpus 1 \
    --infer-gpus 1
```

This generates `runs/reverse-rl/rl.sh` along with three resolved subconfigs under `runs/reverse-rl/configs/` (trainer, orchestrator, inference) and submits the job.

To preview without submitting:

```bash
medarc_slurm rl examples/reverse_text/rl.toml \
    --output-dir runs/reverse-rl \
    --train-gpus 1 \
    --infer-gpus 1 \
    --dry-run
```

### Common options

| Flag | Description |
|------|-------------|
| `--dry-run` | Write artifacts and print the `sbatch` command without submitting |
| `--project-dir PATH` | Project root for `.env` and `.venv` (defaults to cwd) |
| `--hf-cache-dir PATH` | HuggingFace cache directory (sets `HF_HOME` in the job) |
| `--hf-hub-offline` | Set `HF_HUB_OFFLINE=1` to prevent runtime downloads |
| `--job-name NAME` | Custom SLURM job name |
