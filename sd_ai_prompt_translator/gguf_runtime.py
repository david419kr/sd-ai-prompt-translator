from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx

DOUGEE_RELEASES_API = (
    "https://api.github.com/repos/dougeeai/llama-cpp-python-wheels/releases?per_page=100"
)
REQUEST_TIMEOUT_SECONDS = 30.0

_WHEEL_NAME_PATTERN = re.compile(
    r"^llama_cpp_python-[^+]+\+cuda(?P<cuda>[0-9.]+)\.sm(?P<sm>[0-9]+)\.[^-]+-"
    r"(?P<py>cp\d+)-cp\d+-win_amd64\.whl$"
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
            "llama_cpp is not installed. Auto-install is supported only on Windows. "
            "Please install llama-cpp-python manually.",
        )

    py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    cuda_version, sm = _detect_cuda_and_sm()
    if not cuda_version or not sm:
        return (
            False,
            "llama_cpp is not installed and CUDA runtime info is unavailable. "
            "Auto-install skipped; please install llama-cpp-python manually.",
        )

    wheel_url = _find_matching_dougee_wheel(py_tag=py_tag, cuda_version=cuda_version, sm=sm)
    if not wheel_url:
        return (
            False,
            "No exact llama-cpp-python wheel match found in dougeeai releases for "
            f"py={py_tag}, cuda={cuda_version}, sm={sm}, platform=win_amd64.",
        )

    code, output = _pip_install(wheel_url)
    if code != 0:
        return (
            False,
            "Failed to auto-install llama-cpp-python wheel from dougeeai. "
            f"pip exit={code}. tail={output}",
        )

    try:
        import llama_cpp  # type: ignore # noqa: F401

        return True, "llama_cpp auto-installed successfully from dougeeai wheel."
    except Exception as exc:
        return (
            False,
            "llama_cpp install command succeeded but import still failed: "
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


def _find_matching_dougee_wheel(py_tag: str, cuda_version: str, sm: str) -> str | None:
    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
        response = client.get(
            DOUGEE_RELEASES_API,
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
                return url
    return None


def _pip_install(wheel_url: str) -> tuple[int, str]:
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
