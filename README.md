# med-lm-train

## Installation

Follow installation of prime-rl:
https://github.com/PrimeIntellect-ai/prime-rl

Make sure to activate the uv project environment for prime-rl.


Install the required RL environments from the environment hub, for example:
```
uv run prime env install maziyar/openmed_medqa
```

Run the tmux layout script:

```
bash scripts/tmux.sh
```

In the tmux session, run the training, for example:

```
uv run rl --trainer @ configs/medqa/3b/train.toml    --orchestrator @ configs/medqa/3b/orch.toml   --inference @ configs/medqa/3b/infer.toml --wandb.project prime-medqa --trainer-gpus 4 --inference-gpus 4
```
