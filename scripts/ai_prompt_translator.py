from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import threading
from collections import OrderedDict
from pathlib import Path

import gradio as gr

import modules.scripts as scripts
from modules import shared
from modules.script_callbacks import on_ui_settings
from modules.ui_components import InputAccordion

from sd_ai_prompt_translator.providers import (
    TranslatorSettings,
    create_provider,
    validate_line_integrity,
)

logger = logging.getLogger(__name__)

SECTION = ("ai_prompt_translator", "AI Prompt Translator")

OPT_PROVIDER = "aipt_provider"
OPT_GEMINI_API_KEY = "aipt_gemini_api_key"
OPT_GEMINI_MODEL = "aipt_gemini_model"
OPT_OPENAI_BASE_URL = "aipt_openai_base_url"
OPT_OPENAI_API_KEY = "aipt_openai_api_key"
OPT_OPENAI_MODEL = "aipt_openai_model"
OPT_CODEX_MODEL = "aipt_codex_model"
OPT_TRANSLATEGEMMA_HF_TOKEN = "aipt_translategemma_hf_token"
OPT_TRANSLATEGEMMA_VARIANT = "aipt_translategemma_variant"
TRANSLATEGEMMA_ACCESS_URL = "https://huggingface.co/google/translategemma-4b-it"
GGUF_QUANT_CHOICES_BY_MODEL_SIZE = {
    "4B": ["Q4_K_M", "Q8_0", "Full"],
    "12B": ["i1-Q4_K_M", "i1-Q6_K", "Full"],
}
TRANSLATEGEMMA_VARIANT_OPTIONS = [
    ("4B - Q4_K_M [2.31GB] <lightest>", "4B", "Q4_K_M"),
    ("4B - Q8 [3.84GB] <recommended>", "4B", "Q8_0"),
    ("4B - Full [8.04GB]", "4B", "Full"),
    ("12B - i1-Q4_K_M [6.79GB] <recommended>", "12B", "i1-Q4_K_M"),
    ("12B - i1-Q6_K [8.99GB]", "12B", "i1-Q6_K"),
    ("12B - Full [22.7GB]", "12B", "Full"),
]
TRANSLATEGEMMA_VARIANT_MAP = {
    label: (model_size, quantization)
    for label, model_size, quantization in TRANSLATEGEMMA_VARIANT_OPTIONS
}
DEFAULT_TRANSLATEGEMMA_VARIANT = "4B - Q8 [3.84GB] <recommended>"

SCRIPT_BASENAME = "ai_prompt_translator.py"
UI_KEY_TXT = f"customscript/{SCRIPT_BASENAME}/txt2img/AI Prompt Translator/value"
UI_KEY_IMG = f"customscript/{SCRIPT_BASENAME}/img2img/AI Prompt Translator/value"
TRANSLATION_CACHE_FILENAME = "translation_cache.json"
TRANSLATION_CACHE_MAX_ITEMS = 100
CLEAR_CACHE_BUTTON_LABEL = "Clear Cached Translations"

_TRANSLATION_CACHE_LOCK = threading.Lock()

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_YELLOW = "\033[93m"
ANSI_RED = "\033[91m"


def _section_header_option(text: str):
    info = shared.OptionHTML(
        f"<div style='margin-top:0.8em; font-weight:700;'>{text}</div>"
    )
    info.section = SECTION
    return info


def _register_settings() -> None:
    shared.opts.add_option(
        "aipt_header_provider",
        _section_header_option("Provider"),
    )
    shared.opts.add_option(
        OPT_PROVIDER,
        shared.OptionInfo(
            "translategemma_local",
            "Provider",
            gr.Dropdown,
            {
                "choices": ["gemini", "openai_compatible", "codex", "translategemma_local"],
                "interactive": True,
            },
            section=SECTION,
        ),
    )
    shared.opts.add_option(
        "aipt_header_tg",
        _section_header_option("TranslateGemma Settings"),
    )
    shared.opts.add_option(
        OPT_TRANSLATEGEMMA_VARIANT,
        shared.OptionInfo(
            DEFAULT_TRANSLATEGEMMA_VARIANT,
            "TranslateGemma model / quantization",
            gr.Dropdown,
            {"choices": [x[0] for x in TRANSLATEGEMMA_VARIANT_OPTIONS], "interactive": True},
            section=SECTION,
        ),
    )
    shared.opts.add_option(
        OPT_TRANSLATEGEMMA_HF_TOKEN,
        shared.OptionInfo(
            "",
            (
                "TranslateGemma Hugging Face token "
                f"(required for `Full` only; optional for non-Full modes. "
                f"Full mode needs access at {TRANSLATEGEMMA_ACCESS_URL})"
            ),
            gr.Textbox,
            {
                "type": "password",
                "max_lines": 1,
                "interactive": True,
                "placeholder": "hf_xxx... (read token)",
            },
            section=SECTION,
        ),
    )
    shared.opts.add_option(
        "aipt_header_gemini",
        _section_header_option("Gemini Settings"),
    )
    shared.opts.add_option(
        OPT_GEMINI_API_KEY,
        shared.OptionInfo(
            "",
            "Gemini API key",
            gr.Textbox,
            {"type": "password", "max_lines": 1, "interactive": True},
            section=SECTION,
        ),
    )
    shared.opts.add_option(
        OPT_GEMINI_MODEL,
        shared.OptionInfo(
            "gemini-2.5-flash",
            "Gemini model",
            gr.Textbox,
            {"max_lines": 1, "interactive": True},
            section=SECTION,
        ),
    )
    shared.opts.add_option(
        "aipt_header_openai_compatible",
        _section_header_option("OpenAI-compatible API Settings"),
    )
    shared.opts.add_option(
        OPT_OPENAI_BASE_URL,
        shared.OptionInfo(
            "http://127.0.0.1:11434/v1",
            "OpenAI-compatible base URL",
            gr.Textbox,
            {"max_lines": 1, "interactive": True},
            section=SECTION,
        ),
    )
    shared.opts.add_option(
        OPT_OPENAI_API_KEY,
        shared.OptionInfo(
            "",
            "OpenAI-compatible API key",
            gr.Textbox,
            {"type": "password", "max_lines": 1, "interactive": True},
            section=SECTION,
        ),
    )
    shared.opts.add_option(
        OPT_OPENAI_MODEL,
        shared.OptionInfo(
            "gpt-4o-mini",
            "OpenAI-compatible model",
            gr.Textbox,
            {"max_lines": 1, "interactive": True},
            section=SECTION,
        ),
    )
    shared.opts.add_option(
        "aipt_header_codex",
        _section_header_option("Codex Settings"),
    )
    shared.opts.add_option(
        OPT_CODEX_MODEL,
        shared.OptionInfo(
            "gpt-5.4",
            "Codex model",
            gr.Textbox,
            {"max_lines": 1, "interactive": True},
            section=SECTION,
        ),
    )
on_ui_settings(_register_settings)


class Script(scripts.Script):
    def title(self):
        return "AI Prompt Translator"

    def show(self, _is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        startup_enabled = _read_startup_default(is_img2img)

        with InputAccordion(startup_enabled, label=self.title()) as enabled:
            with gr.Row():
                toggle_default_button = gr.Button(
                    value=_startup_button_label(startup_enabled),
                    variant="secondary",
                )
                clear_cache_button = gr.Button(
                    value=CLEAR_CACHE_BUTTON_LABEL,
                    variant="secondary",
                )

            toggle_default_button.click(
                fn=_toggle_startup_default_common,
                inputs=[],
                outputs=[toggle_default_button],
                show_progress=False,
            )
            clear_cache_button.click(
                fn=_clear_translation_cache,
                inputs=[],
                outputs=[],
                show_progress=False,
            )
            disable_translation_caching = gr.Checkbox(
                label="Disable Translation Caching",
                value=False,
            )
            gr.Markdown(
                "Warning: Disabling translation caching bypasses saved cache. "
                "Each generation may send new translation requests, output may vary slightly, "
                "and API cost may increase."
            )
            gr.Markdown(
                "Provider and model-specific settings are configured in "
                "Settings > Extensions > AI Prompt Translator."
            )

        self.infotext_fields = [(enabled, "AI Prompt Translator")]
        return [enabled, disable_translation_caching]

    def process(
        self,
        p,
        enabled: bool,
        disable_translation_caching: bool,
    ):
        if not enabled:
            return

        all_prompts = getattr(p, "all_prompts", None)
        if not isinstance(all_prompts, list) or not all_prompts:
            return

        runtime_model_size, runtime_quantization = _resolve_translategemma_variant(
            str(getattr(shared.opts, OPT_TRANSLATEGEMMA_VARIANT, DEFAULT_TRANSLATEGEMMA_VARIANT))
        )
        settings = TranslatorSettings(
            provider=str(getattr(shared.opts, OPT_PROVIDER, "translategemma_local")),
            gemini_api_key=str(getattr(shared.opts, OPT_GEMINI_API_KEY, "")),
            gemini_model=str(getattr(shared.opts, OPT_GEMINI_MODEL, "gemini-2.5-flash")),
            openai_base_url=str(getattr(shared.opts, OPT_OPENAI_BASE_URL, "")),
            openai_api_key=str(getattr(shared.opts, OPT_OPENAI_API_KEY, "")),
            openai_model=str(getattr(shared.opts, OPT_OPENAI_MODEL, "gpt-4o-mini")),
            codex_model=str(getattr(shared.opts, OPT_CODEX_MODEL, "gpt-5.4")),
            translategemma_hf_token=str(getattr(shared.opts, OPT_TRANSLATEGEMMA_HF_TOKEN, "")),
            translategemma_model_size=runtime_model_size,
            translategemma_quantization=runtime_quantization,
        )
        provider_name = settings.provider
        config_issue = _provider_config_issue(settings)
        if config_issue is not None:
            _log_warn(
                "Skipping translation: provider settings are incomplete. "
                "Configure API settings in Settings > Extensions > AI Prompt Translator.",
                emphasize=True,
            )
            _log_warn(f"Provider config issue: {config_issue}")
            return

        try:
            provider = create_provider(settings)
        except Exception as exc:
            _log_warn(f"Provider init failed ({provider_name}): {exc}")
            return

        _log_info(
            f"Run start | provider={provider_name} | prompts={len(all_prompts)}"
        )
        translated_prompts: list[str] = []
        any_changed = False
        changed_count = 0

        if disable_translation_caching:
            disk_cache = OrderedDict()
            _log_warn("Persistent translation cache is disabled for this run.")
        else:
            disk_cache = _read_translation_cache()
            _log_info(f"Disk cache loaded | entries={len(disk_cache)}")
        cache_writes = 0
        disk_cache_hits = 0

        prompt_cache: dict[str, str] = {}
        api_tasks = 0

        for prompt_index, prompt in enumerate(all_prompts):
            if not isinstance(prompt, str):
                translated_prompts.append(prompt)
                continue

            if prompt in prompt_cache:
                new_prompt = prompt_cache[prompt]
            else:
                if disable_translation_caching:
                    api_tasks += 1
                    new_prompt, _is_cacheable = self._translate_prompt_by_lines(
                        prompt,
                        provider,
                        prompt_index=prompt_index,
                        provider_name=provider_name,
                    )
                else:
                    cache_key = _make_translation_cache_key(settings, prompt)
                    cached_prompt = disk_cache.get(cache_key)
                    if isinstance(cached_prompt, str):
                        new_prompt = cached_prompt
                        disk_cache.move_to_end(cache_key)
                        disk_cache_hits += 1
                        _log_info(f"Prompt#{prompt_index}: disk-cache hit (skip API)")
                    else:
                        api_tasks += 1
                        new_prompt, is_cacheable = self._translate_prompt_by_lines(
                            prompt,
                            provider,
                            prompt_index=prompt_index,
                            provider_name=provider_name,
                        )
                        if is_cacheable:
                            disk_cache[cache_key] = new_prompt
                            _trim_translation_cache(disk_cache)
                            cache_writes += 1
                prompt_cache[prompt] = new_prompt
            translated_prompts.append(new_prompt)
            if new_prompt != prompt:
                any_changed = True
                changed_count += 1

        if (not disable_translation_caching) and cache_writes > 0:
            _write_translation_cache(disk_cache)
            _log_info(
                f"Disk cache updated | writes={cache_writes} | entries={len(disk_cache)}"
            )

        if not any_changed:
            _log_info(
                "Run done | no translation applied (all prompts unchanged) "
                f"| unique_prompts={api_tasks} | disk_cache_hits={disk_cache_hits}"
            )
            return

        p.all_prompts = translated_prompts
        if translated_prompts:
            p.main_prompt = translated_prompts[0]
            p.prompt = translated_prompts[0]
        _log_info(
            f"Run done | translated_prompts={changed_count} | unique_prompts={api_tasks} "
            f"| disk_cache_hits={disk_cache_hits}"
        )

    def _translate_prompt_by_lines(
        self,
        prompt: str,
        provider,
        prompt_index: int,
        provider_name: str,
    ) -> tuple[str, bool]:
        if not contains_non_english_letters(prompt):
            _log_info(f"Prompt#{prompt_index}: skip (ASCII-only)")
            return prompt, False

        lines_with_separators = split_lines_with_separators(prompt)
        lines_to_translate: list[tuple[int, str]] = []

        for index, (line, _sep) in enumerate(lines_with_separators):
            if contains_non_english_letters(line):
                lines_to_translate.append((index, line))

        if not lines_to_translate:
            _log_info(f"Prompt#{prompt_index}: skip (no translatable lines)")
            return prompt, False

        _log_info(
            f"Prompt#{prompt_index}: translating {len(lines_to_translate)} lines via {provider_name}"
        )
        try:
            translated_map = provider.translate_lines(lines_to_translate)
        except Exception as exc:
            _log_warn(
                f"Prompt#{prompt_index}: request failed, keep original prompt | reason={exc}"
            )
            return prompt, False

        for line_index, source_line in lines_to_translate:
            translated_line = translated_map.get(line_index)
            if not isinstance(translated_line, str):
                _log_warn(
                    f"Prompt#{prompt_index}: missing translated line id={line_index}, keep original prompt"
                )
                return prompt, False

            if not validate_line_integrity(source_line, translated_line):
                _log_warn(
                    f"Prompt#{prompt_index}: integrity check failed at line id={line_index}, keep original prompt"
                )
                _log_warn(
                    f"Prompt#{prompt_index}: source preview={_safe_log_text_preview(source_line)}"
                )
                _log_warn(
                    f"Prompt#{prompt_index}: translated preview={_safe_log_text_preview(translated_line)}"
                )
                return prompt, False

            _old_line, sep = lines_with_separators[line_index]
            lines_with_separators[line_index] = (translated_line, sep)

        _log_info(f"Prompt#{prompt_index}: translation applied")
        return "".join(line + sep for line, sep in lines_with_separators), True


def contains_non_english_letters(text: str) -> bool:
    for ch in text:
        if ch.isalpha() and not ch.isascii():
            return True
    return False


def split_lines_with_separators(text: str) -> list[tuple[str, str]]:
    parts = re.split(r"(\r\n|\r|\n)", text)
    rows: list[tuple[str, str]] = []
    for i in range(0, len(parts), 2):
        line = parts[i]
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        rows.append((line, sep))
    return rows


def _safe_log_text_preview(text: str, limit: int = 220) -> str:
    if not text:
        return "<empty>"
    compact = " ".join(str(text).split())
    if len(compact) > limit:
        compact = compact[:limit] + "..."
    try:
        return compact.encode("unicode_escape", "backslashreplace").decode("ascii")
    except Exception:
        return "<preview-unavailable>"


def _read_startup_default(is_img2img: bool) -> bool:
    data = _read_ui_config()
    key = UI_KEY_IMG if is_img2img else UI_KEY_TXT
    if key in data:
        return bool(data[key])

    if UI_KEY_TXT in data:
        return bool(data[UI_KEY_TXT])
    if UI_KEY_IMG in data:
        return bool(data[UI_KEY_IMG])
    return False


def _toggle_startup_default_common():
    data = _read_ui_config()
    current = bool(data.get(UI_KEY_TXT, False))
    new_value = not current
    data[UI_KEY_TXT] = new_value
    data[UI_KEY_IMG] = new_value
    _write_ui_config(data)
    return gr.update(value=_startup_button_label(new_value))


def _clear_translation_cache() -> None:
    removed_count = _count_translation_cache_entries()
    path = _translation_cache_path()
    with _TRANSLATION_CACHE_LOCK:
        try:
            if path.exists():
                path.unlink()
        except Exception as exc:
            _log_warn(f"Translation cache clear failed: {exc}")
            return

    _log_info(f"Translation cache cleared | removed_entries={removed_count}")


def _startup_button_label(is_enabled: bool) -> str:
    state = "On" if is_enabled else "Off"
    return f"Toggle startup default (now: {state}, restart required)"


def _read_ui_config() -> dict:
    path = _ui_config_path()
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("AI Prompt Translator failed to read ui-config.json: %s", exc)
        return {}


def _write_ui_config(data: dict) -> None:
    path = _ui_config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
    except Exception as exc:
        logger.warning("AI Prompt Translator failed to write ui-config.json: %s", exc)


def _ui_config_path() -> Path:
    return Path(shared.cmd_opts.ui_config_file)


def _translation_cache_path() -> Path:
    return Path(__file__).resolve().parents[1] / TRANSLATION_CACHE_FILENAME


def _provider_cache_namespace(settings: TranslatorSettings) -> str:
    provider = (settings.provider or "").strip().lower()
    if provider == "gemini":
        model = (settings.gemini_model or "").strip()
        return f"gemini|model={model}"
    if provider == "openai_compatible":
        base_url = (settings.openai_base_url or "").strip().rstrip("/")
        model = (settings.openai_model or "").strip()
        return f"openai_compatible|base_url={base_url}|model={model}"
    if provider == "codex":
        model = (settings.codex_model or "").strip()
        return f"codex|model={model}"
    if provider == "translategemma_local":
        model_size = (settings.translategemma_model_size or "").strip().upper()
        quantization = (settings.translategemma_quantization or "").strip()
        backend = "tf" if quantization.lower() in {"full", "none"} else "gguf"
        return f"translategemma_local|size={model_size}|quant={quantization}|backend={backend}"
    return provider


def _provider_config_issue(settings: TranslatorSettings) -> str | None:
    provider = (settings.provider or "").strip().lower()
    if provider == "gemini":
        if not (settings.gemini_api_key or "").strip():
            return "Gemini API key is empty."
        if not (settings.gemini_model or "").strip():
            return "Gemini model is empty."
        return None

    if provider == "openai_compatible":
        if not (settings.openai_base_url or "").strip():
            return "OpenAI-compatible base URL is empty."
        if not (settings.openai_model or "").strip():
            return "OpenAI-compatible model is empty."
        return None

    if provider == "codex":
        if not (settings.codex_model or "").strip():
            return "Codex model is empty."
        return None

    if provider == "translategemma_local":
        token = (settings.translategemma_hf_token or "").strip()
        model_size = _normalize_model_size(settings.translategemma_model_size)
        quantization = _normalize_translategemma_quantization(
            model_size,
            settings.translategemma_quantization,
        )
        if quantization == "Full" and not token:
            return (
                "TranslateGemma token is required only for `Full` mode and is currently empty. "
                "1) Open "
                f"{TRANSLATEGEMMA_ACCESS_URL} and accept access, "
                "2) create a Hugging Face read token, "
                "3) paste it into Settings > Extensions > AI Prompt Translator."
            )
        if model_size not in GGUF_QUANT_CHOICES_BY_MODEL_SIZE:
            return "TranslateGemma model size must be one of: 4B, 12B."
        if quantization not in GGUF_QUANT_CHOICES_BY_MODEL_SIZE[model_size]:
            choices = ", ".join(GGUF_QUANT_CHOICES_BY_MODEL_SIZE[model_size])
            return (
                f"TranslateGemma quantization for {model_size} must be one of: {choices}."
            )
        return None

    return f"Unsupported provider: {settings.provider}"


def _make_translation_cache_key(settings: TranslatorSettings, prompt: str) -> str:
    payload = _provider_cache_namespace(settings) + "\n" + prompt
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resolve_translategemma_variant(value: str) -> tuple[str, str]:
    raw = (value or "").strip()
    mapped = TRANSLATEGEMMA_VARIANT_MAP.get(raw)
    if mapped:
        return mapped

    # Backward compatibility for old or manually edited values.
    # Examples: "4B|Q8_0", "4B - Q8 [3.84GB] <recommended>", "Q8_0", "Full"
    if "|" in raw:
        parts = [x.strip() for x in raw.split("|", 1)]
        if len(parts) == 2:
            model_size = _normalize_model_size(parts[0])
            quantization = _normalize_translategemma_quantization(model_size, parts[1])
            return model_size, quantization

    match = re.match(r"^\s*(4B|12B)\s*-\s*([A-Za-z0-9_+-]+)", raw, flags=re.IGNORECASE)
    if match:
        model_size = _normalize_model_size(match.group(1))
        quantization = _normalize_translategemma_quantization(model_size, match.group(2))
        return model_size, quantization

    direct = raw.upper()
    if direct in {"4B", "12B"}:
        model_size = _normalize_model_size(direct)
        default_quant = "Q8_0" if model_size == "4B" else "i1-Q4_K_M"
        return model_size, default_quant

    for model_size in ("4B", "12B"):
        quantization = _normalize_translategemma_quantization(model_size, raw)
        if quantization in GGUF_QUANT_CHOICES_BY_MODEL_SIZE.get(model_size, []):
            return model_size, quantization

    return TRANSLATEGEMMA_VARIANT_MAP[DEFAULT_TRANSLATEGEMMA_VARIANT]


def _normalize_model_size(value: str) -> str:
    text = (value or "").strip().upper()
    if text in GGUF_QUANT_CHOICES_BY_MODEL_SIZE:
        return text
    if text.isdigit():
        candidate = f"{int(text)}B"
        if candidate in GGUF_QUANT_CHOICES_BY_MODEL_SIZE:
            return candidate
    if text.endswith("B") and text[:-1].isdigit():
        candidate = f"{int(text[:-1])}B"
        if candidate in GGUF_QUANT_CHOICES_BY_MODEL_SIZE:
            return candidate
    return "4B"


def _quant_choices_for_model_size(model_size: str) -> list[str]:
    normalized = _normalize_model_size(model_size)
    return list(GGUF_QUANT_CHOICES_BY_MODEL_SIZE.get(normalized, ["Full"]))


def _normalize_translategemma_quantization(model_size: str, quantization: str) -> str:
    choices = _quant_choices_for_model_size(model_size)
    value = (quantization or "").strip()
    if value.lower() == "none":
        return "Full"
    if value.upper() == "Q8":
        value = "Q8_0"
    if value in choices:
        return value
    lower_map = {x.lower(): x for x in choices}
    mapped = lower_map.get(value.lower())
    if mapped:
        return mapped
    return "Full"


def _read_translation_cache() -> OrderedDict[str, str]:
    path = _translation_cache_path()
    with _TRANSLATION_CACHE_LOCK:
        if not path.exists():
            return OrderedDict()

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            _log_warn(f"Failed to read translation cache file: {exc}")
            return OrderedDict()

        cache: OrderedDict[str, str] = OrderedDict()
        items = raw.get("items") if isinstance(raw, dict) else None
        if not isinstance(items, list):
            return cache

        for item in items:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            value = item.get("value")
            if isinstance(key, str) and isinstance(value, str):
                cache[key] = value

        _trim_translation_cache(cache)
        return cache


def _write_translation_cache(cache: OrderedDict[str, str]) -> None:
    path = _translation_cache_path()
    _trim_translation_cache(cache)
    payload = {
        "version": 1,
        "max_items": TRANSLATION_CACHE_MAX_ITEMS,
        "items": [{"key": k, "value": v} for k, v in cache.items()],
    }

    with _TRANSLATION_CACHE_LOCK:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp_path.replace(path)
        except Exception as exc:
            _log_warn(f"Failed to write translation cache file: {exc}")


def _trim_translation_cache(cache: OrderedDict[str, str]) -> None:
    while len(cache) > TRANSLATION_CACHE_MAX_ITEMS:
        cache.popitem(last=False)


def _count_translation_cache_entries() -> int:
    cache = _read_translation_cache()
    return len(cache)


def _log_info(message: str) -> None:
    prefixed = f"[AI Prompt Translator] {message}"
    print(prefixed)
    logger.info(message)


def _log_warn(message: str, *, emphasize: bool = False) -> None:
    prefixed = f"[AI Prompt Translator] {message}"
    color = ANSI_RED if emphasize else ANSI_YELLOW
    print(_colorize_console(prefixed, color, bold=emphasize))
    logger.warning(message)


def _colorize_console(text: str, color: str, *, bold: bool = False) -> str:
    if not _supports_ansi_color():
        return text

    prefix = color
    if bold:
        prefix = ANSI_BOLD + prefix
    return f"{prefix}{text}{ANSI_RESET}"


def _supports_ansi_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False

    is_tty = getattr(sys.stdout, "isatty", None)
    if callable(is_tty) and not is_tty():
        return False

    return True
