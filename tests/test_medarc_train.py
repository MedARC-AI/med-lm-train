from __future__ import annotations

import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from medarc_rl.medarc_train import app


runner = CliRunner()


def _write(path: Path, content: str) -> Path:
    path.write_text(content.strip() + "\n", encoding="utf-8")
    return path


def _read_toml(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _build_sft_config(tmp_path: Path) -> Path:
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


def _build_rl_config(tmp_path: Path, *, weight_broadcast_type: str = "filesystem") -> Path:
    _write(
        tmp_path / "rl_base.toml",
        f"""
        [trainer.model]
        cp = 1
        tp = 1

        [orchestrator]

        [inference.parallel]
        tp = 1
        dp = 1

        [inference]
        gpu_memory_utilization = 0.45

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


@patch("medarc_rl.medarc_train.subprocess.run")
def test_sft_single_gpu(mock_run: MagicMock, tmp_path: Path) -> None:
    mock_run.return_value = MagicMock(returncode=0)
    config_path = _build_sft_config(tmp_path)
    output_dir = tmp_path / "sft_out"

    result = runner.invoke(app, ["sft", str(config_path), "--output-dir", str(output_dir)])

    assert result.exit_code == 0, result.output
    assert (output_dir / "configs" / "trainer.toml").exists()

    mock_run.assert_called_once()
    call_args = mock_run.call_args
    cmd = call_args[0][0]
    assert cmd[0] == "sft"
    assert "@" in cmd
    assert call_args[1]["env"]["CUDA_VISIBLE_DEVICES"] == "0"


@patch("medarc_rl.medarc_train.subprocess.run")
def test_sft_multi_gpu(mock_run: MagicMock, tmp_path: Path) -> None:
    mock_run.return_value = MagicMock(returncode=0)
    config_path = _build_sft_config(tmp_path)
    output_dir = tmp_path / "sft_out_multi"

    result = runner.invoke(app, ["sft", str(config_path), "--output-dir", str(output_dir), "--gpus", "4"])

    assert result.exit_code == 0, result.output

    mock_run.assert_called_once()
    call_args = mock_run.call_args
    cmd = call_args[0][0]
    assert cmd[0] == "torchrun"
    assert "--nproc-per-node=4" in cmd
    assert call_args[1]["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


@patch("medarc_rl.launchers.rl_local.rl_local")
def test_rl_single_gpu(mock_rl_local: MagicMock, tmp_path: Path, monkeypatch) -> None:
    config_path = _build_rl_config(tmp_path)
    output_dir = tmp_path / "rl_out"

    result = runner.invoke(app, ["rl", str(config_path), "--output-dir", str(output_dir), "--single-gpu"])

    assert result.exit_code == 0, result.output
    mock_rl_local.assert_called_once()

    import os

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"
    assert os.environ.get("MEDARC_SINGLE_GPU") == "1"


@patch("medarc_rl.launchers.rl_local.rl_local")
def test_rl_multi_gpu(mock_rl_local: MagicMock, tmp_path: Path) -> None:
    config_path = _build_rl_config(tmp_path)
    output_dir = tmp_path / "rl_out_multi"

    result = runner.invoke(
        app, ["rl", str(config_path), "--output-dir", str(output_dir), "--train-gpus", "2", "--infer-gpus", "2"]
    )

    assert result.exit_code == 0, result.output
    mock_rl_local.assert_called_once()

    import os

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1,2,3"
    assert os.environ.get("MEDARC_SINGLE_GPU") == "0"


def test_rl_rejects_total_above_eight(tmp_path: Path) -> None:
    config_path = _build_rl_config(tmp_path)
    output_dir = tmp_path / "rl_out_bad"

    result = runner.invoke(
        app, ["rl", str(config_path), "--output-dir", str(output_dir), "--train-gpus", "4", "--infer-gpus", "5"]
    )

    assert result.exit_code != 0


def test_rl_single_gpu_rejects_nccl(tmp_path: Path) -> None:
    config_path = _build_rl_config(tmp_path, weight_broadcast_type="nccl")
    output_dir = tmp_path / "rl_out_nccl"

    result = runner.invoke(app, ["rl", str(config_path), "--output-dir", str(output_dir), "--single-gpu"])

    assert result.exit_code != 0
    assert "NCCL" in result.output


@patch("medarc_rl.medarc_train.subprocess.run")
def test_sft_resolved_config_contains_output_dir(mock_run: MagicMock, tmp_path: Path) -> None:
    mock_run.return_value = MagicMock(returncode=0)
    config_path = _build_sft_config(tmp_path)
    output_dir = tmp_path / "sft_out_resolved"

    runner.invoke(app, ["sft", str(config_path), "--output-dir", str(output_dir)])

    resolved = _read_toml(output_dir / "configs" / "trainer.toml")
    assert resolved["output_dir"] == str(output_dir)


@patch("medarc_rl.medarc_train.subprocess.run")
def test_sft_accepts_primerl_style_overrides_and_wrapper_output_dir_wins(mock_run: MagicMock, tmp_path: Path) -> None:
    mock_run.return_value = MagicMock(returncode=0)
    config_path = _build_sft_config(tmp_path)
    output_dir = tmp_path / "sft_out_overrides"
    ignored_output_dir = tmp_path / "ignored_by_wrapper"

    result = runner.invoke(
        app,
        [
            "sft",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--",
            "--max-steps",
            "9",
            "--output-dir",
            str(ignored_output_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    resolved = _read_toml(output_dir / "configs" / "trainer.toml")
    assert resolved["max_steps"] == 9
    assert resolved["output_dir"] == str(output_dir.resolve())


@patch("medarc_rl.launchers.rl_local.rl_local")
def test_rl_accepts_primerl_style_overrides_and_wrapper_gpu_split_wins(mock_rl_local: MagicMock, tmp_path: Path) -> None:
    config_path = _build_rl_config(tmp_path)
    output_dir = tmp_path / "rl_out_override_split"

    result = runner.invoke(
        app,
        [
            "rl",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--train-gpus",
            "2",
            "--infer-gpus",
            "2",
            "--",
            "--inference.gpu-memory-utilization",
            "0.33",
            "--deployment.num-train-gpus",
            "4",
            "--deployment.num-infer-gpus",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    mock_rl_local.assert_called_once()

    config = mock_rl_local.call_args[0][0]
    assert config.inference is not None
    assert config.inference.gpu_memory_utilization == 0.33
    assert config.deployment.num_train_gpus == 2
    assert config.deployment.num_infer_gpus == 2
