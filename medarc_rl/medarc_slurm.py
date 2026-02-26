from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Annotated, Any

import typer
from jinja2 import Environment, FileSystemLoader
from pydantic import ValidationError
from typer import Argument, Option

from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig

from medarc_rl.utils import (
    TYPER_PASSTHROUGH_CONTEXT,
    _load_settings_from_toml,
    _write_toml,
    extra_config_args,
    maybe_autoset_auth_env,
)


app = typer.Typer(
    add_completion=False,
    help=(
        "Generate single-node SLURM jobs for PRIME-RL SFT/RL. "
        "Pass PRIME-RL config overrides as extra flags  e.g. "
        "`--wandb.project my-proj --wandb.name my-run`."
    ),
)

TEMPLATE_DIR = Path(__file__).parent / "slurm_templates"


def _resolve_path(path: Path | None, fallback: Path) -> Path:
    return (path or fallback).expanduser().resolve()


def _default_hf_cache_dir(project_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    env_hf_home = os.environ.get("HF_HOME")
    if env_hf_home:
        return Path(env_hf_home).expanduser().resolve()
    return (project_dir / ".hf_cache").resolve()


def _ensure_output_dirs(output_dir: Path) -> None:
    (output_dir / "configs").mkdir(parents=True, exist_ok=True)
    (output_dir / "slurm").mkdir(parents=True, exist_ok=True)


def _render_template(template_name: str, **context: Any) -> str:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=False,
        keep_trailing_newline=True,
    )
    return env.get_template(template_name).render(**context)


def _write_script(output_dir: Path, name: str, text: str) -> Path:
    path = output_dir / name
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)
    return path


def _submit_or_print(
    script_path: Path,
    *,
    dry_run: bool,
    account: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    if account is None:
        account = os.environ.get("SBATCH_ACCOUNT") or os.environ.get("SLURM_ACCOUNT")

    sbatch_cmd = ["sbatch"]
    if account:
        sbatch_cmd.extend(["--account", account])
    sbatch_cmd.append(str(script_path))

    cmd = shlex.join(sbatch_cmd)
    if dry_run:
        typer.echo(cmd)
        return

    result = subprocess.run(sbatch_cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        typer.echo(result.stderr.strip() or "sbatch failed", err=True)
        raise typer.Exit(code=1)
    typer.echo(result.stdout.strip())


def _load_sft_config(config_toml: Path, output_dir: Path, *, extra_cli_args: list[str] | None = None) -> SFTConfig:
    return _load_settings_from_toml(SFTConfig, config_toml, output_dir=output_dir, extra_cli_args=extra_cli_args)


def _load_rl_config(
    config_toml: Path,
    output_dir: Path,
    *,
    train_gpus: int,
    infer_gpus: int,
    extra_cli_args: list[str] | None = None,
) -> RLConfig:
    return _load_settings_from_toml(
        RLConfig,
        config_toml,
        extra_cli_args=extra_cli_args,
        output_dir=output_dir,
        deployment={"type": "single_node", "num_train_gpus": train_gpus, "num_infer_gpus": infer_gpus},
    )


def _write_sft_outputs(
    config: SFTConfig,
    *,
    output_dir: Path,
    project_dir: Path,
    hf_cache_dir: Path,
    hf_hub_offline: bool,
    job_name: str,
    gpus: int,
    cpus_per_gpu: int,
) -> Path:
    config_dir = output_dir / "configs"
    _write_toml(config_dir / "trainer.toml", config.model_dump(exclude_none=True, mode="json"))
    script = _render_template(
        "one_node_sft.j2",
        job_name=job_name,
        output_dir=str(output_dir),
        config_dir=str(config_dir),
        project_dir=str(project_dir),
        hf_cache_dir=str(hf_cache_dir),
        hf_hub_offline=hf_hub_offline,
        gpus=gpus,
        cpus_per_gpu=cpus_per_gpu,
    )
    return _write_script(output_dir, "sft.sh", script)


def _write_rl_outputs(
    config: RLConfig,
    *,
    output_dir: Path,
    project_dir: Path,
    hf_cache_dir: Path,
    hf_hub_offline: bool,
    job_name: str,
    total_gpus: int,
    single_gpu: bool,
    cpus_per_gpu: int,
) -> Path:
    if config.inference is None:
        raise typer.BadParameter("RL requires an [inference] config.", param_hint="CONFIG_TOML")

    config_dir = output_dir / "configs"
    _write_toml(config_dir / "rl.toml", config.model_dump(exclude_none=True, mode="json"))

    script = _render_template(
        "one_node_rl.j2",
        job_name=job_name,
        output_dir=str(output_dir),
        config_dir=str(config_dir),
        project_dir=str(project_dir),
        hf_cache_dir=str(hf_cache_dir),
        hf_hub_offline=hf_hub_offline,
        total_gpus=total_gpus,
        single_gpu=single_gpu,
        cpus_per_gpu=cpus_per_gpu,
    )
    return _write_script(output_dir, "rl.sh", script)


@app.command(
    context_settings=TYPER_PASSTHROUGH_CONTEXT,
    help=(
        "Generate/submit an SFT SLURM job. PRIME-RL config overrides can be passed as extra flags, e.g. `--wandb.project my-proj --wandb.name my-run`."
    ),
)
def sft(
    ctx: typer.Context,
    config_toml: Annotated[Path, Argument( metavar="CONFIG_TOML", help="Path to the PRIME-RL SFT trainer TOML (supports `toml_files` inheritance).")],
    output_dir: Annotated[Path, Option("--output-dir", file_okay=False, dir_okay=True, help="Directory to write generated artifacts (configs/ and sft.sh).")],
    gpus: Annotated[int, Option("--gpus", min=1, max=8, help="Number of GPUs for SFT on this single node (sets SLURM gres and torchrun nproc-per-node).")],
    cpus_per_gpu: Annotated[int, Option("--cpus-per-gpu", min=1, max=32, help="Number of CPUs to allocate per GPU (sets SLURM --cpus-per-gpu).")] = 8,
    job_name: Annotated[str | None, Option("--job-name", help="SLURM job name. Defaults to '<config stem>-sft'.")] = None,
    dry_run: Annotated[bool, Option("--dry-run", help="Write configs and script, print the `sbatch` command, and do not submit.")] = False,
    auto_auth: Annotated[bool, Option("--auto-auth/--no-auto-auth", help="Try to load HF_TOKEN from local CLI credentials and inject it into the sbatch submission environment.")] = False,
    project_dir: Annotated[Path | None, Option("--project-dir", file_okay=False, dir_okay=True, help="Project root used by the script to source .env and activate .venv (defaults to current working directory).")] = None,
    hf_cache_dir: Annotated[Path, Option("--hf-cache-dir", file_okay=False, dir_okay=True, help="HF cache directory (sets HF_HOME inside the job).")] = "/data/medlm_cache/.hf_cache",
    hf_hub_offline: Annotated[bool, Option("--hf-hub-offline/--no-hf-hub-offline", help="Set HF_HUB_OFFLINE=1 inside the job to prevent runtime downloads.")] = False,
    account: Annotated[str | None, Option("--account", help="SLURM account to pass to sbatch. Defaults to $SBATCH_ACCOUNT or $SLURM_ACCOUNT if set.")] = None,
) -> None:  # fmt: skip
    output_dir = output_dir.expanduser().resolve()
    project_dir = _resolve_path(project_dir, Path.cwd())
    hf_cache_dir = _default_hf_cache_dir(project_dir, hf_cache_dir)
    job_name = job_name or f"{config_toml.stem}-sft"

    _ensure_output_dirs(output_dir)
    config = _load_sft_config(config_toml.expanduser().resolve(), output_dir, extra_cli_args=extra_config_args(ctx))
    script_path = _write_sft_outputs(
        config,
        output_dir=output_dir,
        project_dir=project_dir,
        hf_cache_dir=hf_cache_dir,
        hf_hub_offline=hf_hub_offline,
        job_name=job_name,
        gpus=gpus,
        cpus_per_gpu=cpus_per_gpu,
    )
    submit_env = os.environ.copy()
    for msg in maybe_autoset_auth_env(submit_env, enabled=auto_auth):
        typer.echo(msg, err=True)
    _submit_or_print(script_path, dry_run=dry_run, account=account, env=submit_env)


@app.command(
    context_settings=TYPER_PASSTHROUGH_CONTEXT,
    help=(
        "Generate/submit an RL SLURM job. "
        "Use medarc GPU flags for placement/splitting. "
        "PRIME-RL config overrides can be passed as extra flags, e.g. `--wandb.project my-proj --wandb.name my-run`."
    ),
)
def rl(
    ctx: typer.Context,
    config_toml: Annotated[Path, Argument( metavar="CONFIG_TOML", help="Path to the PRIME-RL RL TOML (supports `toml_files` inheritance).")],
    output_dir: Annotated[Path, Option("--output-dir", file_okay=False, dir_okay=True, help="Directory to write generated artifacts (configs/ and rl.sh).")],
    train_gpus: Annotated[int, Option("--train-gpus", min=1, max=4, help="Number of GPUs reserved for trainer processes (1..4). Total GPUs is train + infer.")] = 1,
    infer_gpus: Annotated[int, Option("--infer-gpus", min=1, max=7, help="Number of GPUs reserved for local inference server (1..7). Total GPUs is train + infer.")] = 1,
    single_gpu: Annotated[bool, Option("--single-gpu", help="Run trainer and inference on the same single GPU (shared). Overrides --train-gpus/--infer-gpus to 1/1.")] = False,
    cpus_per_gpu: Annotated[int, Option("--cpus-per-gpu", min=1, max=32, help="Number of CPUs to allocate per GPU (sets SLURM --cpus-per-gpu).")] = 8,
    job_name: Annotated[str | None, Option("--job-name", help="SLURM job name. Defaults to '<config stem>-rl'.")] = None,
    dry_run: Annotated[bool, Option("--dry-run", help="Write configs and script, print the `sbatch` command, and do not submit.")] = False,
    auto_auth: Annotated[bool, Option("--auto-auth/--no-auto-auth", help="Try to load HF_TOKEN from local CLI credentials and inject it into the sbatch submission environment.")] = False,
    project_dir: Annotated[Path | None, Option("--project-dir", file_okay=False, dir_okay=True, help="Project root used by the script to source .env and activate .venv (defaults to current working directory).")] = None,
    hf_cache_dir: Annotated[Path, Option("--hf-cache-dir", file_okay=False, dir_okay=True, help="HF cache directory (sets HF_HOME inside the job).")] = "/data/medlm_cache/.hf_cache",
    hf_hub_offline: Annotated[bool, Option("--hf-hub-offline/--no-hf-hub-offline", help="Set HF_HUB_OFFLINE=1 inside the job to prevent runtime downloads.")] = False,
    account: Annotated[str | None, Option("--account", help="SLURM account to pass to sbatch. Defaults to $SBATCH_ACCOUNT or $SLURM_ACCOUNT if set.")] = None,
) -> None:  # fmt: skip
    output_dir = output_dir.expanduser().resolve()
    project_dir = _resolve_path(project_dir, Path.cwd())
    hf_cache_dir = _default_hf_cache_dir(project_dir, hf_cache_dir)
    job_name = job_name or f"{config_toml.stem}-rl"
    train_gpus = 1 if single_gpu else train_gpus
    infer_gpus = 1 if single_gpu else infer_gpus
    total_gpus = 1 if single_gpu else (train_gpus + infer_gpus)

    if (not single_gpu and total_gpus < 2) or total_gpus > 8:
        raise typer.BadParameter(
            (
                f"Total GPUs must be between 2 and 8, got train_gpus ({train_gpus}) + "
                f"infer_gpus ({infer_gpus}) = {train_gpus + infer_gpus}."
            ),
            param_hint="--train-gpus/--infer-gpus",
        )

    _ensure_output_dirs(output_dir)
    try:
        config = _load_rl_config(
            config_toml.expanduser().resolve(),
            output_dir,
            train_gpus=train_gpus,
            infer_gpus=infer_gpus,
            extra_cli_args=extra_config_args(ctx),
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
            (
                "Warning: --single-gpu with inference.gpu_memory_utilization >= 0.9 may OOM. "
                "PrimeRL's default is 0.9; try 0.7-0.8 for shared trainer+vLLM."
            ),
            err=True,
        )
    script_path = _write_rl_outputs(
        config,
        output_dir=output_dir,
        project_dir=project_dir,
        hf_cache_dir=hf_cache_dir,
        hf_hub_offline=hf_hub_offline,
        job_name=job_name,
        total_gpus=total_gpus,
        single_gpu=single_gpu,
        cpus_per_gpu=cpus_per_gpu,
    )
    submit_env = os.environ.copy()
    for msg in maybe_autoset_auth_env(submit_env, enabled=auto_auth):
        typer.echo(msg, err=True)
    _submit_or_print(script_path, dry_run=dry_run, account=account, env=submit_env)


if __name__ == "__main__":
    app()
