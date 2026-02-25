# Hendrycks Sanity

> Adapted from the [PRIME-RL hendrycks_sanity example](../../prime-rl/examples/hendrycks_sanity/) to use `medarc_slurm` for single-node SLURM submission.

This runs the Hendrycks Sanity Check experiment from [Defeating the Training-Inference Mismatch](https://arxiv.org/abs/2510.26788). The sanity check tests whether an RL algorithm can reliably improve a model on problems it can *already partially solve*. The dataset is filtered from MATH to only include problems where `DeepSeek-R1-Distill-Qwen-1.5B` solves 20-80% of the time across 40 rollouts. A reliable algorithm should push training accuracy on this "perfectible" subset above 95%.

> This example uses 4 GPUs on a single node: 1 for training and 3 for inference (data parallel — the 1.5B model fits on a single GPU, so each inference GPU runs its own vLLM replica for higher throughput).

## Setup

The `math-env` environment is included in the lock file. Verify it's installed:

```bash
uv run python -c "import math_verify"
```

## RL (4 GPUs: 1 train + 3 inference)

Submit a 4-GPU RL job:

```bash
medarc_slurm rl examples/hendrycks_sanity/rl.toml \
    --output-dir runs/hendrycks-sanity \
    --train-gpus 1 \
    --infer-gpus 3 \
    --auto-auth
```

Or preview without submitting:

```bash
medarc_slurm rl examples/hendrycks_sanity/rl.toml \
    --output-dir runs/hendrycks-sanity \
    --train-gpus 1 \
    --infer-gpus 3 \
    --auto-auth \
    --dry-run
```

The eval runs AIME 2024 every 50 steps to track progress.
