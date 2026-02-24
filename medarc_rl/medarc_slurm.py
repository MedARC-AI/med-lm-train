from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Annotated, Any, TypeVar

import tomli_w
import typer
from jinja2 import Environment, FileSystemLoader
from pydantic import ValidationError
from typer import Argument, Option

from prime_rl.inference.config import InferenceConfig
from prime_rl.orchestrator.config import OrchestratorConfig
from prime_rl.rl_config import BaseRLConfig
from prime_rl.trainer.rl.config import RLTrainerConfig
from prime_rl.trainer.sft.config import SFTTrainerConfig
from prime_rl.utils.pydantic_config import extract_toml_paths


app = typer.Typer(add_completion=False, help="Generate single-node SLURM jobs for PRIME-RL SFT/RL.")

T = TypeVar("T")
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


def _detect_local_gpu_count() -> int | None:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        gpu_ids = [gpu_id.strip() for gpu_id in cuda_visible_devices.split(",") if gpu_id.strip()]
        if gpu_ids:
            return len(gpu_ids)

    if not shutil.which("nvidia-smi"):
        return None
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if result.returncode != 0:
        return None
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return len(lines) or None


def _ensure_output_dirs(output_dir: Path) -> None:
    (output_dir / "configs").mkdir(parents=True, exist_ok=True)
    (output_dir / "slurm").mkdir(parents=True, exist_ok=True)


def _load_settings_from_toml(config_cls: type[T], config_path: Path, **overrides: Any) -> T:
    if not config_path.exists():
        raise typer.BadParameter(f"Config file does not exist: {config_path}", param_hint="CONFIG_TOML")
    toml_paths, _ = extract_toml_paths(["@", str(config_path)])
    if not toml_paths:
        raise typer.BadParameter(f"Failed to resolve TOML paths from {config_path}", param_hint="CONFIG_TOML")

    config_cls.set_toml_files([str(path) for path in toml_paths])
    try:
        return config_cls(**overrides)
    finally:
        config_cls.clear_toml_files()


def _write_toml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        tomli_w.dump(data, f)


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


def _submit_or_print(script_path: Path, *, dry_run: bool, account: str | None = None) -> None:
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

    result = subprocess.run(sbatch_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        typer.echo(result.stderr.strip() or "sbatch failed", err=True)
        raise typer.Exit(code=1)
    typer.echo(result.stdout.strip())


def _load_sft_config(config_toml: Path, output_dir: Path) -> SFTTrainerConfig:
    return _load_settings_from_toml(SFTTrainerConfig, config_toml, output_dir=output_dir)


def _load_rl_config(config_toml: Path, output_dir: Path) -> BaseRLConfig:
    return _load_settings_from_toml(BaseRLConfig, config_toml, output_dir=output_dir)


def _normalize_rl_config(config: BaseRLConfig, *, train_gpus: int, infer_gpus: int) -> None:
    if config.inference is None:
        raise typer.BadParameter("RL requires an [inference] config.", param_hint="CONFIG_TOML")

    non_data_parallel_size = config.trainer.model.cp * config.trainer.model.tp
    if non_data_parallel_size < 1:
        raise typer.BadParameter("trainer.model.cp * trainer.model.tp must be >= 1.", param_hint="CONFIG_TOML")
    if train_gpus % non_data_parallel_size != 0:
        raise typer.BadParameter(
            (
                "train_gpus must be divisible by trainer.model.cp * trainer.model.tp "
                f"(train_gpus={train_gpus}, cp={config.trainer.model.cp}, tp={config.trainer.model.tp})."
            ),
            param_hint="--train-gpus",
        )

    config.orchestrator.num_train_workers = train_gpus // non_data_parallel_size
    config.inference.parallel.tp = infer_gpus
    config.inference.parallel.dp = 1
    config.inference.api_server_count = 1

    # BaseRLConfig validators have already propagated shared weight_broadcast and, for NCCL,
    # derived trainer.weight_broadcast.inference_world_size using the pre-normalized inference
    # dp/tp values from the input TOML. We patch the final single-node split values here so the
    # dumped subconfigs match the runtime topology.
    if config.weight_broadcast is not None:
        config.inference.weight_broadcast.type = config.weight_broadcast.type
    if getattr(config.trainer.weight_broadcast, "type", None) == "nccl":
        config.trainer.weight_broadcast.inference_world_size = infer_gpus


def _revalidate_rl_config(config: BaseRLConfig) -> BaseRLConfig:
    """Re-run PrimeRL validators after medarc-specific topology normalization."""
    payload = config.model_dump(mode="json")

    # Revalidate nested subconfigs first so trainer/orchestrator/inference-specific
    # constraints (e.g. NCCL + async-level) fail before submission.
    payload["trainer"] = RLTrainerConfig.model_validate(payload["trainer"]).model_dump(mode="json")
    payload["orchestrator"] = OrchestratorConfig.model_validate(payload["orchestrator"]).model_dump(mode="json")
    if payload.get("inference") is not None:
        payload["inference"] = InferenceConfig.model_validate(payload["inference"]).model_dump(mode="json")

    # Revalidate shared/top-level consistency after nested normalization.
    return BaseRLConfig.model_validate(payload)


def _write_sft_outputs(
    config: SFTTrainerConfig,
    *,
    output_dir: Path,
    project_dir: Path,
    hf_cache_dir: Path,
    hf_hub_offline: bool,
    job_name: str,
    gpus: int,
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
    )
    return _write_script(output_dir, "sft.sh", script)


def _write_rl_outputs(
    config: BaseRLConfig,
    *,
    output_dir: Path,
    project_dir: Path,
    hf_cache_dir: Path,
    hf_hub_offline: bool,
    job_name: str,
    total_gpus: int,
    train_gpus: int,
    infer_gpus: int,
) -> Path:
    if config.inference is None:
        raise typer.BadParameter("RL requires an [inference] config.", param_hint="CONFIG_TOML")

    config_dir = output_dir / "configs"
    _write_toml(config_dir / "trainer.toml", config.trainer.model_dump(exclude_none=True, mode="json"))
    _write_toml(config_dir / "orchestrator.toml", config.orchestrator.model_dump(exclude_none=True, mode="json"))
    _write_toml(config_dir / "inference.toml", config.inference.model_dump(exclude_none=True, mode="json"))

    script = _render_template(
        "one_node_rl.j2",
        job_name=job_name,
        output_dir=str(output_dir),
        config_dir=str(config_dir),
        project_dir=str(project_dir),
        hf_cache_dir=str(hf_cache_dir),
        hf_hub_offline=hf_hub_offline,
        total_gpus=total_gpus,
        train_gpus=train_gpus,
        infer_gpus=infer_gpus,
        nccl_enabled=(getattr(config.trainer.weight_broadcast, "type", None) == "nccl"),
    )
    return _write_script(output_dir, "rl.sh", script)


@app.command()
def sft(
    config_toml: Annotated[Path, Argument( metavar="CONFIG_TOML", help="Path to the PRIME-RL SFT trainer TOML (supports `toml_files` inheritance).")],
    output_dir: Annotated[Path, Option("--output-dir", file_okay=False, dir_okay=True, help="Directory to write generated artifacts (configs/ and sft.sh).")],
    gpus: Annotated[int, Option("--gpus", min=1, max=8, help="Number of GPUs for SFT on this single node (sets SLURM gres and torchrun nproc-per-node).")],
    job_name: Annotated[str | None, Option("--job-name", help="SLURM job name. Defaults to '<config stem>-sft'.")] = None,
    dry_run: Annotated[bool, Option("--dry-run", help="Write configs and script, print the `sbatch` command, and do not submit.")] = False,
    project_dir: Annotated[Path | None, Option("--project-dir", file_okay=False, dir_okay=True, help="Project root used by the script to source .env and activate .venv (defaults to current working directory).")] = None,
    hf_cache_dir: Annotated[Path | None, Option("--hf-cache-dir", file_okay=False, dir_okay=True, help="HF cache directory (sets HF_HOME inside the job). Defaults to $HF_HOME if set, else <project-dir>/.hf_cache.")] = None,
    hf_hub_offline: Annotated[bool, Option("--hf-hub-offline/--no-hf-hub-offline", help="Set HF_HUB_OFFLINE=1 inside the job to prevent runtime downloads.")] = False,
    account: Annotated[str | None, Option("--account", help="SLURM account to pass to sbatch. Defaults to $SBATCH_ACCOUNT or $SLURM_ACCOUNT if set.")] = None,
) -> None: # fmt: skip
    output_dir = output_dir.expanduser().resolve()
    project_dir = _resolve_path(project_dir, Path.cwd())
    hf_cache_dir = _default_hf_cache_dir(project_dir, hf_cache_dir)
    job_name = job_name or f"{config_toml.stem}-sft"

    _ensure_output_dirs(output_dir)
    config = _load_sft_config(config_toml.expanduser().resolve(), output_dir)
    script_path = _write_sft_outputs(
        config,
        output_dir=output_dir,
        project_dir=project_dir,
        hf_cache_dir=hf_cache_dir,
        hf_hub_offline=hf_hub_offline,
        job_name=job_name,
        gpus=gpus,
    )
    _submit_or_print(script_path, dry_run=dry_run, account=account)


@app.command()
def rl(
    config_toml: Annotated[Path, Argument( metavar="CONFIG_TOML", help="Path to the PRIME-RL RL TOML (supports `toml_files` inheritance).")],
    output_dir: Annotated[Path, Option("--output-dir", file_okay=False, dir_okay=True, help="Directory to write generated artifacts (configs/ and rl.sh).")],
    train_gpus: Annotated[int, Option("--train-gpus", min=1, max=4, help="Number of GPUs reserved for trainer processes (1..4). Total GPUs is train + infer.")] = 1,
    infer_gpus: Annotated[int, Option("--infer-gpus", min=1, max=7, help="Number of GPUs reserved for local inference server (1..7). Total GPUs is train + infer.")] = 1,
    job_name: Annotated[str | None, Option("--job-name", help="SLURM job name. Defaults to '<config stem>-rl'.")] = None,
    dry_run: Annotated[bool, Option("--dry-run", help="Write configs and script, print the `sbatch` command, and do not submit.")] = False,
    project_dir: Annotated[Path | None, Option("--project-dir", file_okay=False, dir_okay=True, help="Project root used by the script to source .env and activate .venv (defaults to current working directory).")] = None,
    hf_cache_dir: Annotated[Path | None, Option("--hf-cache-dir", file_okay=False, dir_okay=True, help="HF cache directory (sets HF_HOME inside the job). Defaults to $HF_HOME if set, else <project-dir>/.hf_cache.")] = None,
    hf_hub_offline: Annotated[bool, Option("--hf-hub-offline/--no-hf-hub-offline", help="Set HF_HUB_OFFLINE=1 inside the job to prevent runtime downloads.")] = False,
    account: Annotated[str | None, Option("--account", help="SLURM account to pass to sbatch. Defaults to $SBATCH_ACCOUNT or $SLURM_ACCOUNT if set.")] = None,
) -> None:  # fmt: skip
    output_dir = output_dir.expanduser().resolve()
    project_dir = _resolve_path(project_dir, Path.cwd())
    hf_cache_dir = _default_hf_cache_dir(project_dir, hf_cache_dir)
    job_name = job_name or f"{config_toml.stem}-rl"
    total_gpus = train_gpus + infer_gpus

    detected_gpus = _detect_local_gpu_count()
    max_total_gpus = 8 if detected_gpus is None else min(8, detected_gpus)
    if total_gpus < 2 or total_gpus > max_total_gpus:
        suffix = "" if detected_gpus is None else f" (detected {detected_gpus} GPUs locally)"
        raise typer.BadParameter(
            (
                f"Total GPUs must be between 2 and {max_total_gpus}{suffix}, "
                f"got train_gpus ({train_gpus}) + infer_gpus ({infer_gpus}) = {total_gpus}."
            ),
            param_hint="--train-gpus/--infer-gpus",
        )

    _ensure_output_dirs(output_dir)
    config = _load_rl_config(config_toml.expanduser().resolve(), output_dir)
    _normalize_rl_config(config, train_gpus=train_gpus, infer_gpus=infer_gpus)
    try:
        config = _revalidate_rl_config(config)
    except ValidationError as e:
        raise typer.BadParameter(
            f"Resolved RL config is invalid after medarc_slurm topology normalization:\n{e}",
            param_hint="CONFIG_TOML/--train-gpus/--infer-gpus",
        ) from e
    script_path = _write_rl_outputs(
        config,
        output_dir=output_dir,
        project_dir=project_dir,
        hf_cache_dir=hf_cache_dir,
        hf_hub_offline=hf_hub_offline,
        job_name=job_name,
        total_gpus=total_gpus,
        train_gpus=train_gpus,
        infer_gpus=infer_gpus,
    )
    _submit_or_print(script_path, dry_run=dry_run, account=account)


if __name__ == "__main__":
    app()
