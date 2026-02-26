from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from typer import Argument, Option

from medarc_rl.medarc_slurm import _load_settings_from_toml, _write_toml


app = typer.Typer(add_completion=False, help="Run PRIME-RL SFT/RL training locally (no SLURM).")


def _gpu_ids(n: int) -> str:
    return ",".join(str(i) for i in range(n))


@app.command()
def sft(
    config_toml: Annotated[Path, Argument(metavar="CONFIG_TOML", help="Path to the PRIME-RL SFT trainer TOML.")],
    output_dir: Annotated[Path, Option("--output-dir", file_okay=False, dir_okay=True, help="Directory to write resolved configs and checkpoints.")],
    gpus: Annotated[int, Option("--gpus", min=1, max=8, help="Number of GPUs for SFT.")] = 1,
) -> None:  # fmt: skip
    from prime_rl.configs.sft import SFTConfig

    output_dir = output_dir.expanduser().resolve()
    config = _load_settings_from_toml(SFTConfig, config_toml.expanduser().resolve(), output_dir=output_dir)

    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = config_dir / "trainer.toml"
    _write_toml(resolved_path, config.model_dump(exclude_none=True, mode="json"))

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": _gpu_ids(gpus)}

    if gpus == 1:
        cmd = ["sft", "@", str(resolved_path)]
    else:
        cmd = [
            "torchrun",
            "--local-ranks-filter=0",
            f"--nproc-per-node={gpus}",
            "-m",
            "prime_rl.trainer.sft.train",
            "@",
            str(resolved_path),
        ]

    typer.echo(f"Starting SFT on {gpus} GPU(s): {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    raise typer.Exit(code=result.returncode)


@app.command()
def rl(
    config_toml: Annotated[Path, Argument(metavar="CONFIG_TOML", help="Path to the PRIME-RL RL TOML.")],
    output_dir: Annotated[Path, Option("--output-dir", file_okay=False, dir_okay=True, help="Directory to write resolved configs and checkpoints.")],
    train_gpus: Annotated[int, Option("--train-gpus", min=1, max=4, help="Number of GPUs for training.")] = 1,
    infer_gpus: Annotated[int, Option("--infer-gpus", min=1, max=7, help="Number of GPUs for inference.")] = 1,
    single_gpu: Annotated[bool, Option("--single-gpu", help="Share a single GPU between trainer and inference.")] = False,
) -> None:  # fmt: skip
    from prime_rl.configs.rl import RLConfig

    from medarc_rl.launchers.rl_local import rl_local

    output_dir = output_dir.expanduser().resolve()
    train_gpus = 1 if single_gpu else train_gpus
    infer_gpus = 1 if single_gpu else infer_gpus
    total_gpus = 1 if single_gpu else (train_gpus + infer_gpus)

    if not single_gpu and total_gpus < 2:
        raise typer.BadParameter(
            f"Total GPUs must be at least 2, got train_gpus ({train_gpus}) + infer_gpus ({infer_gpus}) = {total_gpus}.",
            param_hint="--train-gpus/--infer-gpus",
        )
    if total_gpus > 8:
        raise typer.BadParameter(
            f"Total GPUs must be at most 8, got train_gpus ({train_gpus}) + infer_gpus ({infer_gpus}) = {total_gpus}.",
            param_hint="--train-gpus/--infer-gpus",
        )

    try:
        config = _load_settings_from_toml(
            RLConfig,
            config_toml.expanduser().resolve(),
            output_dir=output_dir,
            deployment={"type": "single_node", "num_train_gpus": train_gpus, "num_infer_gpus": infer_gpus},
        )
    except ValidationError as e:
        raise typer.BadParameter(
            f"RL config validation failed:\n{e}",
            param_hint="CONFIG_TOML/--train-gpus/--infer-gpus",
        ) from e

    if single_gpu and getattr(config.trainer.weight_broadcast, "type", None) == "nccl":
        raise typer.BadParameter(
            "--single-gpu does not support NCCL weight broadcast. Use filesystem broadcast or 2+ GPUs.",
            param_hint="CONFIG_TOML/--single-gpu",
        )
    if single_gpu and config.inference is not None and config.inference.gpu_memory_utilization >= 0.9:
        typer.echo(
            "Warning: --single-gpu with inference.gpu_memory_utilization >= 0.9 may OOM. "
            "Try 0.7-0.8 for shared trainer+vLLM.",
            err=True,
        )

    # Set env vars for rl_local
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_ids(total_gpus)
    os.environ["MEDARC_SINGLE_GPU"] = "1" if single_gpu else "0"

    typer.echo(f"Starting RL on {total_gpus} GPU(s) (single_gpu={single_gpu})")
    rl_local(config)


if __name__ == "__main__":
    app()
