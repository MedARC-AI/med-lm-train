from __future__ import annotations

import configparser
import os
from netrc import NetrcParseError, netrc
from pathlib import Path
from urllib.parse import urlparse


def maybe_autoset_auth_env(
    env: dict[str, str],
    *,
    enabled: bool,
    project_dir: Path | None,
) -> list[str]:
    """Best-effort local auth discovery for tools that usually rely on env vars.

    This only affects the environment passed to `sbatch` (and therefore the job), and does
    not write secrets into the generated slurm script.
    """
    if not enabled:
        return []

    msgs: list[str] = []

    if not env.get("HF_TOKEN"):
        try:
            # huggingface_hub looks in env and its local token cache.
            from huggingface_hub.utils import get_token as _hf_get_token  # type: ignore[import-not-found]
        except Exception:
            _hf_get_token = None

        if _hf_get_token is not None:
            token = _hf_get_token()
            if token:
                env["HF_TOKEN"] = token
                msgs.append("Auto-auth: set HF_TOKEN from local Hugging Face credentials.")

    if not env.get("WANDB_API_KEY"):
        # Per wandb.login() docs, credentials can come from env, then settings files,
        # then netrc, then interactive prompt. We only implement the non-interactive,
        # read-only sources here.
        cfg_dir = env.get("WANDB_CONFIG_DIR")
        settings_candidates: list[Path] = []
        if cfg_dir:
            settings_candidates.append(Path(cfg_dir).expanduser() / "settings")
        settings_candidates.append(Path.home() / ".config" / "wandb" / "settings")
        if project_dir is not None:
            settings_candidates.append(project_dir / "wandb" / "settings")

        for settings_path in settings_candidates:
            try:
                if not settings_path.exists():
                    continue
            except OSError:
                continue

            parser = configparser.ConfigParser()
            try:
                parser.read(settings_path)
            except (OSError, configparser.Error):
                continue

            try:
                api_key = parser.get("default", "api_key", fallback=None)
            except (configparser.Error, ValueError):
                api_key = None

            if api_key:
                env["WANDB_API_KEY"] = api_key
                msgs.append(f"Auto-auth: set WANDB_API_KEY from {settings_path}.")
                break

    if not env.get("WANDB_API_KEY"):
        wandb_hosts: list[str] = []
        base_url = env.get("WANDB_BASE_URL")
        if base_url:
            parsed = urlparse(base_url if "://" in base_url else f"https://{base_url}")
            if parsed.hostname:
                wandb_hosts.append(parsed.hostname)
        # Defaults (wandb cloud). Try a few common variants.
        wandb_hosts.extend(["api.wandb.ai", "wandb.ai"])

        auth = None
        for host in wandb_hosts:
            try:
                auth = netrc().authenticators(host)
            except (FileNotFoundError, NetrcParseError):
                auth = None
            except Exception:
                auth = None
            if auth is not None:
                break

        if auth is not None:
            _, _, password = auth
            if password:
                env["WANDB_API_KEY"] = password
                msgs.append("Auto-auth: set WANDB_API_KEY from ~/.netrc.")

    return msgs
