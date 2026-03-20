from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

CUSTOM_RELEASES_API = (
    "https://api.github.com/repos/david419kr/sd-ai-prompt-translator/releases?per_page=50"
)
REQUEST_TIMEOUT_SECONDS = 30.0

_CUSTOM_WHEEL_PATTERN = re.compile(
    r"^llama_cpp_python-[^-]+-(?P<py>cp\d+)-cp\d+-win_amd64\.whl$"
)


def ensure_llama_cpp_available() -> tuple[bool, str]:
    try:
        import llama_cpp  # type: ignore # noqa: F401

        return True, "llama_cpp already installed."
    except Exception:
        pass

    if os.name != "nt":
        return (
            False,
            "llama_cpp is not installed. Auto-install fallback is supported only on Windows. "
            "Please install llama-cpp-python manually.",
        )

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    cuda_version, sm = _detect_cuda_and_sm()
    code, output = _pip_install_package("llama-cpp-python")
    if code == 0:
        try:
            import llama_cpp  # type: ignore # noqa: F401

            return True, "llama_cpp installed successfully from official source."
        except Exception as exc:
            output = f"{output}\nimport_error={exc}"

    if not _is_custom_fallback_target(py_tag=py_tag, cuda_version=cuda_version, sm=sm):
        return (
            False,
            "llama_cpp official install failed and custom wheel fallback is disabled for this environment. "
            f"py={py_tag}, cuda={cuda_version}, sm={sm}, arch={platform.machine()}, pip_tail={output}",
        )

    wheel_url = _find_matching_custom_wheel(py_tag=py_tag)
    if not wheel_url:
        return (
            False,
            "llama_cpp official install failed and no matching custom wheel was found. "
            f"py={py_tag}, arch={platform.machine()}, pip_tail={output}",
        )

    code2, output2 = _pip_install_url(wheel_url)
    if code2 != 0:
        return (
            False,
            "llama_cpp official install failed and custom wheel install also failed. "
            f"pip exit={code2}. tail={output2}",
        )

    try:
        import llama_cpp  # type: ignore # noqa: F401

        return True, "llama_cpp installed successfully from custom GitHub wheel fallback."
    except Exception as exc:
        return (
            False,
            "llama_cpp custom fallback install succeeded but import still failed: "
            f"{exc}",
        )


def download_gguf(repo_id: str, filename: str, hf_token: str | None, local_dir: Path) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required to download GGUF models."
        ) from exc

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


def _normalize_cuda_version(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    m = re.match(r"^(\d+)(?:\.(\d+))?", text)
    if not m:
        return None
    major = int(m.group(1))
    minor = int(m.group(2) or 0)
    return f"{major}.{minor}"


def _find_matching_custom_wheel(py_tag: str) -> str | None:
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = client.get(
            CUSTOM_RELEASES_API,
            headers={"Accept": "application/vnd.github+json"},
        )

    if response.status_code >= 400:
        return None

    releases = response.json()
    if not isinstance(releases, list):
        return None

    for release in releases:
        assets = release.get("assets", [])
        if not isinstance(assets, list):
            continue
        for asset in assets:
            name = asset.get("name")
            if not isinstance(name, str):
                continue
            matched = _CUSTOM_WHEEL_PATTERN.match(name)
            if not matched:
                continue
            if matched.group("py") != py_tag:
                continue
            url = asset.get("browser_download_url")
            if isinstance(url, str) and url:
                return url
    return None


def _pip_install_package(package_name: str) -> tuple[int, str]:
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--only-binary=:all:",
        package_name,
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    combined = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
    tail = "\n".join(combined.splitlines()[-20:]) if combined else "<no output>"
    return int(result.returncode), tail


def _pip_install_url(wheel_url: str) -> tuple[int, str]:
    command = [sys.executable, "-m", "pip", "install", "--upgrade", wheel_url]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    combined = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
    tail = "\n".join(combined.splitlines()[-20:]) if combined else "<no output>"
    return int(result.returncode), tail


def _is_custom_fallback_target(py_tag: str, cuda_version: str | None, sm: str | None) -> bool:
    if os.name != "nt":
        return False
    if py_tag != "cp313":
        return False
    if not _is_x64_windows():
        return False
    if not cuda_version or not sm:
        return False
    return True


def _is_x64_windows() -> bool:
    machine = (platform.machine() or "").lower()
    if machine in {"amd64", "x86_64"}:
        return True
    return sys.maxsize > (2**32)
