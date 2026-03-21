from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any

import launch


_REQUIRED_MODULES = (
    "transformers",
    "huggingface_hub",
    "sentencepiece",
)
_MIN_TRANSFORMERS_VERSION = "4.57.0"
_DOUGEE_RELEASES_API = "https://api.github.com/repos/dougeeai/llama-cpp-python-wheels/releases?per_page=100"
_WHEEL_NAME_PATTERN = re.compile(
    r"^llama_cpp_python-[^+]+\+cuda(?P<cuda>[0-9.]+)\.sm(?P<sm>[0-9]+)\.[^-]+-"
    r"(?P<py>cp\d+)-cp\d+-win_amd64\.whl$"
)
_CP313_SM86_CUDA130_HARDCODED_WHEEL_URL = (
    "https://github.com/dougeeai/llama-cpp-python-wheels/releases/download/"
    "v0.3.16-cuda13.0-py313/"
    "llama_cpp_python-0.3.16+cuda13.0.sm86.ampere-cp313-cp313-win_amd64.whl"
)


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


def _is_x64_windows() -> bool:
    if os.name != "nt":
        return False

    machine = (platform.machine() or "").lower()
    if machine in {"amd64", "x86_64"}:
        return True
    return sys.maxsize > (2**32)


def _normalize_cuda_version(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    matched = re.match(r"^(\d+)(?:\.(\d+))?", text)
    if not matched:
        return None
    major = int(matched.group(1))
    minor = int(matched.group(2) or 0)
    return f"{major}.{minor}"


def _detect_cuda_and_sm() -> tuple[str | None, str | None]:
    try:
        import torch  # type: ignore
    except Exception:
        return None, None

    if not torch.cuda.is_available():
        return None, None

    cuda_version = _normalize_cuda_version(getattr(torch.version, "cuda", None))
    sm = None
    try:
        major, minor = torch.cuda.get_device_capability(0)
        sm = f"{int(major)}{int(minor)}"
    except Exception:
        sm = None

    return cuda_version, sm


def _fetch_releases_json() -> list[dict[str, Any]] | None:
    try:
        request = urllib.request.Request(
            _DOUGEE_RELEASES_API,
            headers={"Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = response.read()
    except Exception:
        return None

    try:
        data = json.loads(payload.decode("utf-8", errors="replace"))
    except Exception:
        return None

    if not isinstance(data, list):
        return None
    return [x for x in data if isinstance(x, dict)]


def _is_url_alive(url: str) -> bool:
    if not isinstance(url, str) or not url:
        return False

    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=15):
            return True
    except Exception:
        return False


def _cp313_sm86_cuda130_fallback_if_broken(
    url: str | None,
    *,
    py_tag: str,
    cuda_version: str,
    sm: str,
) -> str | None:
    if py_tag != "cp313" or cuda_version != "13.0" or sm != "86":
        return url

    if isinstance(url, str) and url:
        if _is_url_alive(url):
            return url

    return _CP313_SM86_CUDA130_HARDCODED_WHEEL_URL


def _find_matching_dougee_wheel(py_tag: str, cuda_version: str, sm: str) -> str | None:
    releases = _fetch_releases_json()
    if not releases:
        return _cp313_sm86_cuda130_fallback_if_broken(
            None,
            py_tag=py_tag,
            cuda_version=cuda_version,
            sm=sm,
        )

    for release in releases:
        assets = release.get("assets", [])
        if not isinstance(assets, list):
            continue
        for asset in assets:
            name = asset.get("name")
            if not isinstance(name, str):
                continue
            matched = _WHEEL_NAME_PATTERN.match(name)
            if not matched:
                continue
            if matched.group("py") != py_tag:
                continue
            if matched.group("cuda") != cuda_version:
                continue
            if matched.group("sm") != sm:
                continue
            url = asset.get("browser_download_url")
            if isinstance(url, str) and url:
                return _cp313_sm86_cuda130_fallback_if_broken(
                    url,
                    py_tag=py_tag,
                    cuda_version=cuda_version,
                    sm=sm,
                )

    return _cp313_sm86_cuda130_fallback_if_broken(
        None,
        py_tag=py_tag,
        cuda_version=cuda_version,
        sm=sm,
    )


def _has_working_llama_cpp() -> bool:
    try:
        import llama_cpp  # type: ignore # noqa: F401
        return True
    except Exception:
        return False


def _is_markupsafe_broken() -> bool:
    try:
        import markupsafe  # type: ignore
    except Exception:
        return True

    markup = getattr(markupsafe, "Markup", None)
    module_path = getattr(markupsafe, "__file__", None)
    if markup is None:
        return True
    if not isinstance(module_path, str) or not module_path:
        return True
    return False


def _repair_markupsafe_before_start() -> None:
    if not _is_markupsafe_broken():
        return

    print("[AI Prompt Translator] MarkupSafe looks broken; attempting repair before startup.")
    launch.run_pip(
        'install --ignore-installed --no-deps --upgrade "MarkupSafe==3.0.2"',
        "MarkupSafe repair (pre-start)",
    )

    if _is_markupsafe_broken():
        print("[AI Prompt Translator] MarkupSafe repair attempted but still appears broken.")
    else:
        print("[AI Prompt Translator] MarkupSafe repair done.")


def _install_llama_cpp_before_start() -> None:
    if _has_working_llama_cpp():
        return

    if not _is_x64_windows():
        print("[AI Prompt Translator] llama_cpp preinstall skipped: non-Windows-x64 environment.")
        return

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    if py_tag not in {"cp310", "cp311", "cp312", "cp313"}:
        print(f"[AI Prompt Translator] llama_cpp preinstall skipped: unsupported Python tag {py_tag}.")
        return

    cuda_version, sm = _detect_cuda_and_sm()
    if not cuda_version or not sm:
        print(
            "[AI Prompt Translator] llama_cpp preinstall skipped: CUDA runtime/device capability "
            "could not be detected."
        )
        return

    wheel_url = _find_matching_dougee_wheel(py_tag=py_tag, cuda_version=cuda_version, sm=sm)
    if not wheel_url:
        print(
            "[AI Prompt Translator] llama_cpp preinstall skipped: no matching dougee wheel "
            f"(py={py_tag}, cuda={cuda_version}, sm={sm})."
        )
        return

    print(
        "[AI Prompt Translator] llama_cpp preinstall start "
        f"(py={py_tag}, cuda={cuda_version}, sm={sm})."
    )
    launch.run_pip(
        f'install --no-deps --upgrade "{wheel_url}"',
        "llama-cpp-python prebuilt wheel (pre-start)",
    )
    if _has_working_llama_cpp():
        print("[AI Prompt Translator] llama_cpp preinstall done.")
    else:
        print("[AI Prompt Translator] llama_cpp preinstall finished but import still fails.")


def install() -> None:
    requirements_path = Path(__file__).resolve().parent / "requirements.txt"
    if not requirements_path.exists():
        return

    _repair_markupsafe_before_start()

    missing = [name for name in _REQUIRED_MODULES if not launch.is_installed(name)]
    if not missing and not _needs_transformers_upgrade():
        _install_llama_cpp_before_start()
        return

    launch.run_pip(
        f'install -r "{requirements_path}" --prefer-binary',
        "sd-ai-prompt-translator requirements",
    )
    _install_llama_cpp_before_start()


if __name__ == "__main__":
    install()
