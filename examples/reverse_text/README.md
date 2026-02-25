# Reverse Text

> Lightly adapted from the [PRIME-RL reverse text example](../../prime-rl/examples/reverse_text/) to use `medarc_slurm` for single-node SLURM submission.

We demonstrate how to train `Qwen3-0.6B` to reverse a small chunk of text on a single GPU. We use a SFT warmup to learn the skill of text reversal on longer documents and then a quick RL run to reverse smaller chunks of text in the [`reverse-text`](https://app.primeintellect.ai/dashboard/environments/primeintellect/reverse-text) environment.

> The configs in this example are tuned for H100 GPUs. If you're on consumer GPUs, you may need to lower `micro_batch_size` in `sft.toml` and `batch_size` in `rl.toml`.

## Setup

The `reverse-text` environment is included in the lock file. Verify it's installed:

```bash
uv run python -c "import reverse_text"
```

## SFT

We fine-tune [`PrimeIntellect/Qwen3-0.6B`](https://huggingface.co/PrimeIntellect/Qwen3-0.6B) (a clone of `Qwen/Qwen3-0.6B` with a chat template suitable for multi-turn RL) on [`willcb/R1-reverse-wikipedia-paragraphs-v1-1000`](https://huggingface.co/datasets/willcb/R1-reverse-wikipedia-paragraphs-v1-1000) which contains 1K examples of reversals of small paragraphs.

Submit a 1-GPU SFT job:

```bash
medarc_slurm sft examples/reverse_text/sft.toml \
    --output-dir runs/reverse-sft \
    --gpus 1 \
    --auto-auth
```

Or preview without submitting:

```bash
medarc_slurm sft examples/reverse_text/sft.toml \
    --output-dir runs/reverse-sft \
    --gpus 1 \
    --auto-auth \
    --dry-run
```

This writes a checkpoint to `runs/reverse-sft/weights/step_100`. The RL config uses the published [`PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT`](https://huggingface.co/PrimeIntellect/Qwen3-0.6B-Reverse-Text-SFT) by default — to use your own SFT checkpoint instead, override `model.name` in `rl.toml`.

## RL (single GPU)

For RL we do 20 steps at 16x16 rollouts with sequence length 128. Because of the small context, training should be extremely quick.

This example shares a single GPU between the trainer and vLLM inference server. The config lowers vLLM `gpu_memory_utilization` so the trainer has headroom — if you still see OOMs, reduce it further.

```bash
medarc_slurm rl examples/reverse_text/rl.toml \
    --output-dir runs/reverse-rl \
    --single-gpu \
    --auto-auth
```

Or preview without submitting:

```bash
medarc_slurm rl examples/reverse_text/rl.toml \
    --output-dir runs/reverse-rl \
    --single-gpu \
    --auto-auth \
    --dry-run
```

This writes a checkpoint to `runs/reverse-rl/weights/step_20`.

For multi-GPU RL examples, see [hendrycks_sanity](../hendrycks_sanity/) (4 GPUs) and [alphabet_sort](../alphabet_sort/) (8 GPUs).

## Evals

To evaluate the final RL checkpoint, start an inference server and run `vf-eval`:

```bash
uv run inference --model.name PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL
```

```bash
uv run vf-eval reverse-text \
    -m PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL \
    -b http://localhost:8000/v1 \
    -n 20 --max-tokens 1024
```

The base model gets ~0.05 average reward. After SFT + RL, expect ~0.8.
