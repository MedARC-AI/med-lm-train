from __future__ import annotations


def maybe_autoset_auth_env(env: dict[str, str], enabled: bool) -> list[str]:
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

    return msgs
