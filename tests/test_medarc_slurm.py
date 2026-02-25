from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path
from unittest.mock import Mock

from typer.testing import CliRunner

from medarc_rl.medarc_slurm import app
from prime_rl.configs.sft import SFTConfig


runner = CliRunner()


def _write(path: Path, content: str) -> Path:
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _bash_n(script_path: Path) -> None:
    subprocess.run(["bash", "-n", str(script_path)], check=True)


def _load_sft_config(config_path: Path) -> SFTConfig:
    SFTConfig.set_toml_files([str(config_path)])
    try:
        return SFTConfig()
    finally:
        SFTConfig.clear_toml_files()


def _read_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _build_sft_inherited_config(tmp_path: Path) -> Path:
    _write(
        tmp_path / "sft_base.toml",
        """
        [model]
        name = "Qwen/Qwen2.5-3B"
        seq_len = 256

        [data]
        type = "fake"
        batch_size = 2
        micro_batch_size = 1
        seq_len = 256
        """,
    )
    return _write(
        tmp_path / "sft_child.toml",
        """
        toml_files = ["sft_base.toml"]
        max_steps = 2
        """,
    )


def _build_rl_inherited_config(
    tmp_path: Path, *, weight_broadcast_type: str = "nccl", cp: int = 2, tp: int = 1
) -> Path:
    _write(
        tmp_path / "rl_base.toml",
        f"""
        [trainer.model]
        cp = {cp}
        tp = {tp}

        [orchestrator]

        [inference.parallel]
        tp = 1
        dp = 3

        [inference]
        api_server_count = 4

        [weight_broadcast]
        type = "{weight_broadcast_type}"
        """,
    )
    return _write(
        tmp_path / "rl_child.toml",
        """
        toml_files = ["rl_base.toml"]
        """,
    )


def test_sft_dry_run_generates_script_and_resolved_toml(tmp_path: Path) -> None:
    config_path = _build_sft_inherited_config(tmp_path)
    output_dir = tmp_path / "sft_out"

    result = runner.invoke(
        app,
        [
            "sft",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--gpus",
            "2",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"sbatch {output_dir / 'sft.sh'}" in result.output

    script_path = output_dir / "sft.sh"
    trainer_toml = output_dir / "configs" / "trainer.toml"
    assert script_path.exists()
    assert trainer_toml.exists()

    _bash_n(script_path)
    cfg = _load_sft_config(trainer_toml)
    assert cfg.output_dir == output_dir

    script = script_path.read_text(encoding="utf-8")
    assert "--standalone" not in script
    assert "--rdzv-endpoint=127.0.0.1:$RDZV_PORT" in script
    assert "--nproc-per-node" in script
    assert "pick_free_ports" in script
    assert "uv sync" not in script


def test_sft_boundary_and_hf_env_flags_are_rendered(tmp_path: Path) -> None:
    config_path = _build_sft_inherited_config(tmp_path)
    output_dir = tmp_path / "sft_out_boundary"
    hf_cache_dir = tmp_path / "hf_cache"

    result = runner.invoke(
        app,
        [
            "sft",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--gpus",
            "1",
            "--hf-cache-dir",
            str(hf_cache_dir),
            "--hf-hub-offline",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    script = (output_dir / "sft.sh").read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:1" in script
    assert f'export HF_CACHE_DIR="{hf_cache_dir.resolve()}"' in script
    assert 'export HF_HOME="$HF_CACHE_DIR"' in script
    assert "export HF_HUB_OFFLINE=1" in script


def test_rl_defaults_split_to_one_and_one(tmp_path: Path) -> None:
    config_path = _build_rl_inherited_config(tmp_path, cp=1, tp=1)
    output_dir = tmp_path / "rl_out_missing_split"

    result = runner.invoke(
        app,
        [
            "rl",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    script = (output_dir / "rl.sh").read_text(encoding="utf-8")
    assert "#SBATCH --gres=gpu:2" in script


def test_rl_rejects_total_gpu_count_above_eight(tmp_path: Path) -> None:
    config_path = _build_rl_inherited_config(tmp_path)
    output_dir = tmp_path / "rl_out_bad_total"

    result = runner.invoke(
        app,
        [
            "rl",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--train-gpus",
            "4",
            "--infer-gpus",
            "5",
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "between 2" in result.output
    assert "and 8" in result.output


def test_rl_rejects_train_gpus_above_four(tmp_path: Path) -> None:
    config_path = _build_rl_inherited_config(tmp_path, cp=1, tp=1)
    output_dir = tmp_path / "rl_out_bad_train_max"

    result = runner.invoke(
        app,
        [
            "rl",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--train-gpus",
            "5",
            "--infer-gpus",
            "1",
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "train-gpus" in result.output


def test_rl_rejects_infer_gpus_above_seven(tmp_path: Path) -> None:
    config_path = _build_rl_inherited_config(tmp_path, cp=1, tp=1)
    output_dir = tmp_path / "rl_out_bad_infer_max"

    result = runner.invoke(
        app,
        [
            "rl",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--train-gpus",
            "1",
            "--infer-gpus",
            "8",
            "--dry-run",
        ],
    )

    assert result.exit_code != 0
    assert "infer-gpus" in result.output


def test_rl_dry_run_generates_normalized_subconfigs_and_safe_script(tmp_path: Path) -> None:
    config_path = _build_rl_inherited_config(tmp_path)
    output_dir = tmp_path / "rl_out"

    result = runner.invoke(
        app,
        [
            "rl",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--train-gpus",
            "4",
            "--infer-gpus",
            "2",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert f"sbatch {output_dir / 'rl.sh'}" in result.output

    script_path = output_dir / "rl.sh"
    trainer_toml = output_dir / "configs" / "trainer.toml"
    orch_toml = output_dir / "configs" / "orchestrator.toml"
    infer_toml = output_dir / "configs" / "inference.toml"

    for path in [script_path, trainer_toml, orch_toml, infer_toml]:
        assert path.exists(), str(path)

    _bash_n(script_path)

    trainer = _read_toml(trainer_toml)
    orchestrator = _read_toml(orch_toml)
    inference = _read_toml(infer_toml)

    assert orchestrator["num_train_workers"] == 2
    assert inference["parallel"]["tp"] == 1
    assert inference["parallel"]["dp"] == 2
    assert trainer["weight_broadcast"]["type"] == "nccl"
    assert inference["weight_broadcast"]["type"] == "nccl"

    script = script_path.read_text(encoding="utf-8")
    assert "--standalone" not in script
    assert ":8000" not in script
    assert "pick_free_ports" in script
    assert "--rdzv-endpoint=127.0.0.1:$RDZV_PORT" in script
    assert "--rdzv-id=job_$SLURM_JOB_ID" in script
    assert "--server.host 127.0.0.1" in script
    assert '--server.port "$INFER_PORT"' in script
    assert "uv sync" not in script
    assert "uv run" not in script


def test_rl_dry_run_train_gpu_path_and_filesystem_broadcast(tmp_path: Path) -> None:
    config_path = _build_rl_inherited_config(tmp_path, weight_broadcast_type="filesystem")
    output_dir = tmp_path / "rl_out_train_split_fs"

    result = runner.invoke(
        app,
        [
            "rl",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--train-gpus",
            "4",
            "--infer-gpus",
            "2",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output

    trainer = _read_toml(output_dir / "configs" / "trainer.toml")
    inference = _read_toml(output_dir / "configs" / "inference.toml")
    script = (output_dir / "rl.sh").read_text(encoding="utf-8")

    assert trainer["weight_broadcast"]["type"] == "filesystem"
    assert "inference_world_size" not in trainer["weight_broadcast"]
    assert inference["parallel"]["tp"] == 1
    assert inference["parallel"]["dp"] == 2
    assert '--weight_broadcast.port "$WEIGHT_BROADCAST_PORT"' not in script
    assert "WEIGHT_BROADCAST_PORT" not in script
    assert "uv run" not in script


def test_dry_run_does_not_call_sbatch(tmp_path: Path, monkeypatch) -> None:
    config_path = _build_sft_inherited_config(tmp_path)
    output_dir = tmp_path / "sft_out_no_submit"
    run_mock = Mock(side_effect=AssertionError("sbatch should not be called during --dry-run"))
    monkeypatch.setattr("medarc_rl.medarc_slurm.subprocess.run", run_mock)

    result = runner.invoke(
        app,
        [
            "sft",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--gpus",
            "1",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    run_mock.assert_not_called()
