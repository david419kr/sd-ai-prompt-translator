from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_REGISTERED_DLL_PATHS: set[str] = set()
_DLL_DIR_HANDLES: list[Any] = []


def ensure_llama_cpp_available() -> tuple[bool, str]:
    _prepare_torch_runtime_for_llama_import()
    try:
        import llama_cpp  # type: ignore # noqa: F401

        return True, "llama_cpp already installed."
    except Exception as exc:
        return (
            False,
            "llama_cpp import failed. Auto-install is handled in install.py before WebUI startup; "
            "restart WebUI after installation or check wheel compatibility. "
            f"reason={exc}",
        )


def download_gguf(repo_id: str, filename: str, hf_token: str | None, local_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required to download GGUF models.") from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    token_value = hf_token.strip() if isinstance(hf_token, str) else None
    if not token_value:
        token_value = None

    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        token=token_value,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    return Path(downloaded)


def _prepare_torch_runtime_for_llama_import() -> None:
    try:
        import torch  # type: ignore
    except Exception:
        return

    if os.name != "nt":
        return

    try:
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
    except Exception:
        return

    if not torch_lib.exists():
        return

    torch_lib_str = str(torch_lib)
    if torch_lib_str in _REGISTERED_DLL_PATHS:
        return

    add_dll_directory = getattr(os, "add_dll_directory", None)
    if callable(add_dll_directory):
        try:
            handle = add_dll_directory(torch_lib_str)
            _DLL_DIR_HANDLES.append(handle)
            _REGISTERED_DLL_PATHS.add(torch_lib_str)
            return
        except Exception:
            pass

    current_path = os.environ.get("PATH", "")
    lowered_entries = [entry.lower() for entry in current_path.split(";") if entry]
    if torch_lib_str.lower() not in lowered_entries:
        os.environ["PATH"] = f"{torch_lib_str};{current_path}" if current_path else torch_lib_str
    _REGISTERED_DLL_PATHS.add(torch_lib_str)
