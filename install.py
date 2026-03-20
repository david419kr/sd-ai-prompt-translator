from __future__ import annotations

import importlib.metadata
from pathlib import Path

import launch


_REQUIRED_MODULES = (
    "transformers",
    "huggingface_hub",
    "sentencepiece",
)
_MIN_TRANSFORMERS_VERSION = "4.57.0"


def _version_tuple(version: str) -> tuple[int, ...]:
    # Keep parsing dependency-free and robust enough for semantic versions.
    parts = []
    for chunk in (version or "").split("."):
        digits = []
        for ch in chunk:
            if ch.isdigit():
                digits.append(ch)
            else:
                break
        if digits:
            parts.append(int("".join(digits)))
        else:
            parts.append(0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _needs_transformers_upgrade() -> bool:
    try:
        current = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        return True
    except Exception:
        # If version discovery fails, prefer safe upgrade.
        return True

    return _version_tuple(current) < _version_tuple(_MIN_TRANSFORMERS_VERSION)


def install() -> None:
    requirements_path = Path(__file__).resolve().parent / "requirements.txt"
    if not requirements_path.exists():
        return

    missing = [name for name in _REQUIRED_MODULES if not launch.is_installed(name)]
    if not missing and not _needs_transformers_upgrade():
        return

    launch.run_pip(
        f'install -r "{requirements_path}" --prefer-binary',
        "sd-ai-prompt-translator requirements",
    )


if __name__ == "__main__":
    install()
