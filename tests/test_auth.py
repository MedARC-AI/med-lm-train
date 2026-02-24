from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

from medarc_rl.utils import maybe_autoset_auth_env


def test_disabled_returns_empty(tmp_path: Path) -> None:
    env: dict[str, str] = {}
    msgs = maybe_autoset_auth_env(env, enabled=False, project_dir=tmp_path)
    assert msgs == []
    assert "HF_TOKEN" not in env
    assert "WANDB_API_KEY" not in env


def test_existing_tokens_are_not_overwritten(tmp_path: Path) -> None:
    env = {"HF_TOKEN": "existing-hf", "WANDB_API_KEY": "existing-wandb"}
    msgs = maybe_autoset_auth_env(env, enabled=True, project_dir=tmp_path)
    assert msgs == []
    assert env["HF_TOKEN"] == "existing-hf"
    assert env["WANDB_API_KEY"] == "existing-wandb"


def test_hf_token_from_huggingface_hub(tmp_path: Path) -> None:
    fake_token = "hf_fake_token_12345"
    with patch("medarc_rl.utils.maybe_autoset_auth_env.__module__", "medarc_rl.utils"):
        # Patch the dynamic import inside the function
        import types

        fake_mod = types.ModuleType("huggingface_hub.utils")
        fake_mod.get_token = lambda: fake_token  # type: ignore[attr-defined]

        with patch.dict("sys.modules", {"huggingface_hub.utils": fake_mod, "huggingface_hub": types.ModuleType("huggingface_hub")}):
            env: dict[str, str] = {}
            msgs = maybe_autoset_auth_env(env, enabled=True, project_dir=tmp_path)

    assert env["HF_TOKEN"] == fake_token
    assert any("HF_TOKEN" in m for m in msgs)


def test_hf_token_not_set_when_get_token_returns_none(tmp_path: Path) -> None:
    import types

    fake_mod = types.ModuleType("huggingface_hub.utils")
    fake_mod.get_token = lambda: None  # type: ignore[attr-defined]

    with patch.dict("sys.modules", {"huggingface_hub.utils": fake_mod, "huggingface_hub": types.ModuleType("huggingface_hub")}):
        env: dict[str, str] = {}
        msgs = maybe_autoset_auth_env(env, enabled=True, project_dir=tmp_path)

    assert "HF_TOKEN" not in env


def test_wandb_from_settings_file(tmp_path: Path) -> None:
    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir()
    settings = wandb_dir / "settings"
    settings.write_text(
        textwrap.dedent("""\
        [default]
        api_key = wandb-secret-key
        """),
        encoding="utf-8",
    )

    env: dict[str, str] = {}
    msgs = maybe_autoset_auth_env(env, enabled=True, project_dir=tmp_path)

    assert env["WANDB_API_KEY"] == "wandb-secret-key"
    assert any("WANDB_API_KEY" in m for m in msgs)
    assert any(str(settings) in m for m in msgs)


def test_wandb_from_config_dir_env(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "custom_wandb"
    cfg_dir.mkdir()
    settings = cfg_dir / "settings"
    settings.write_text(
        textwrap.dedent("""\
        [default]
        api_key = wandb-from-config-dir
        """),
        encoding="utf-8",
    )

    env: dict[str, str] = {"WANDB_CONFIG_DIR": str(cfg_dir)}
    msgs = maybe_autoset_auth_env(env, enabled=True, project_dir=tmp_path)

    assert env["WANDB_API_KEY"] == "wandb-from-config-dir"


def test_wandb_from_netrc(tmp_path: Path, monkeypatch) -> None:
    netrc_file = tmp_path / ".netrc"
    netrc_file.write_text(
        textwrap.dedent("""\
        machine api.wandb.ai
          login user
          password netrc-wandb-key
        """),
        encoding="utf-8",
    )
    netrc_file.chmod(0o600)

    monkeypatch.setenv("NETRC", str(netrc_file))
    # Patch netrc() to read from our file
    from netrc import netrc as netrc_cls

    def patched_netrc(file=None):
        return netrc_cls(str(netrc_file))

    with patch("medarc_rl.utils.netrc", patched_netrc):
        env: dict[str, str] = {}
        msgs = maybe_autoset_auth_env(env, enabled=True, project_dir=tmp_path)

    assert env["WANDB_API_KEY"] == "netrc-wandb-key"
    assert any("netrc" in m.lower() for m in msgs)


def test_wandb_from_netrc_with_custom_base_url(tmp_path: Path) -> None:
    netrc_file = tmp_path / ".netrc"
    netrc_file.write_text(
        textwrap.dedent("""\
        machine wandb.mycompany.com
          login user
          password custom-wandb-key
        """),
        encoding="utf-8",
    )
    netrc_file.chmod(0o600)

    from netrc import netrc as netrc_cls

    def patched_netrc(file=None):
        return netrc_cls(str(netrc_file))

    with patch("medarc_rl.utils.netrc", patched_netrc):
        env: dict[str, str] = {"WANDB_BASE_URL": "https://wandb.mycompany.com"}
        msgs = maybe_autoset_auth_env(env, enabled=True, project_dir=tmp_path)

    assert env["WANDB_API_KEY"] == "custom-wandb-key"


def test_settings_takes_priority_over_netrc(tmp_path: Path) -> None:
    # Set up both a settings file and a netrc entry
    wandb_dir = tmp_path / "wandb"
    wandb_dir.mkdir()
    (wandb_dir / "settings").write_text(
        textwrap.dedent("""\
        [default]
        api_key = from-settings
        """),
        encoding="utf-8",
    )

    netrc_file = tmp_path / ".netrc"
    netrc_file.write_text(
        textwrap.dedent("""\
        machine api.wandb.ai
          login user
          password from-netrc
        """),
        encoding="utf-8",
    )
    netrc_file.chmod(0o600)

    env: dict[str, str] = {}
    msgs = maybe_autoset_auth_env(env, enabled=True, project_dir=tmp_path)

    assert env["WANDB_API_KEY"] == "from-settings"
