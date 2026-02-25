from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from medarc_rl.utils import maybe_autoset_auth_env


def test_disabled_returns_empty(tmp_path: Path) -> None:
    env: dict[str, str] = {}
    msgs = maybe_autoset_auth_env(env, enabled=False)
    assert msgs == []
    assert "HF_TOKEN" not in env


def test_existing_tokens_are_not_overwritten(tmp_path: Path) -> None:
    env = {"HF_TOKEN": "existing-hf"}
    msgs = maybe_autoset_auth_env(env, enabled=True)
    assert msgs == []
    assert env["HF_TOKEN"] == "existing-hf"


def test_hf_token_from_huggingface_hub(tmp_path: Path) -> None:
    fake_token = "hf_fake_token_12345"
    with patch("medarc_rl.utils.maybe_autoset_auth_env.__module__", "medarc_rl.utils"):
        # Patch the dynamic import inside the function
        import types

        fake_mod = types.ModuleType("huggingface_hub.utils")
        fake_mod.get_token = lambda: fake_token  # type: ignore[attr-defined]

        with patch.dict(
            "sys.modules", {"huggingface_hub.utils": fake_mod, "huggingface_hub": types.ModuleType("huggingface_hub")}
        ):
            env: dict[str, str] = {}
            msgs = maybe_autoset_auth_env(env, enabled=True)

    assert env["HF_TOKEN"] == fake_token
    assert any("HF_TOKEN" in m for m in msgs)


def test_hf_token_not_set_when_get_token_returns_none(tmp_path: Path) -> None:
    import types

    fake_mod = types.ModuleType("huggingface_hub.utils")
    fake_mod.get_token = lambda: None  # type: ignore[attr-defined]

    with patch.dict(
        "sys.modules", {"huggingface_hub.utils": fake_mod, "huggingface_hub": types.ModuleType("huggingface_hub")}
    ):
        env: dict[str, str] = {}
        msgs = maybe_autoset_auth_env(env, enabled=True)

    assert "HF_TOKEN" not in env
