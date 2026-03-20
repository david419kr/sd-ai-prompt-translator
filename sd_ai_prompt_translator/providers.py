from __future__ import annotations

import base64
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

PROTECTED_CHARS = "()[]{}<>|_"
REQUEST_TIMEOUT_SECONDS = 45.0
TRANSLATEGEMMA_MODEL_REPOS = {
    "4B": "google/translategemma-4b-it",
    "12B": "google/translategemma-12b-it",
    "27B": "google/translategemma-27b-it",
}
TRANSLATEGEMMA_MODEL_PAGE_URL = "https://huggingface.co/google/translategemma-4b-it"

TRANSLATION_SYSTEM_PROMPT = (
    "You are a deterministic translation engine for Stable Diffusion prompts.\n"
    "Rules:\n"
    "1) Output JSON only. No markdown, no prose, no explanations.\n"
    "2) Translate only non-English language parts into English.\n"
    "3) Preserve ASCII English words and tags exactly as-is.\n"
    "4) Preserve order and line semantics.\n"
    "5) Do not change numbers.\n"
    "6) Do not change protected symbols: ()[]{}<>|:_\n"
    "7) Return exactly this schema:\n"
    '{"translations":[{"id":0,"text":"..."}]}\n'
    "8) Return one item for every input id."
)

TRANSLATEGEMMA_SYSTEM_PROMPT = (
    "You are a deterministic translation engine for Stable Diffusion prompts.\n"
    "Translate only non-English parts into English.\n"
    "Keep existing English words unchanged.\n"
    "Preserve order and line semantics.\n"
    "Do not change numbers.\n"
    "Do not change protected symbols: ()[]{}<>|:_\n"
    "Input format: each line begins with [id].\n"
    "Output format rules:\n"
    "1) Output lines only, no JSON.\n"
    "2) Each output line must begin with the same [id].\n"
    "3) Do not output markdown, code fences, labels, or explanations."
)

TRANSLATION_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "translations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "text": {"type": "string"},
                },
                "required": ["id", "text"],
            },
        }
    },
    "required": ["translations"],
}

GEMINI_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
]


class TranslationProviderError(RuntimeError):
    pass


@dataclass
class TranslatorSettings:
    provider: str
    gemini_api_key: str
    gemini_model: str
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    codex_model: str
    translategemma_hf_token: str = ""
    translategemma_model_size: str = "4B"
    codex_auth_path: str | None = None


@dataclass
class CodexTokens:
    access_token: str
    account_id: str
    refresh_token: str | None
    auth_path: Path
    mode: str
    raw: dict[str, Any]


class BaseTranslatorProvider:
    def __init__(self, settings: TranslatorSettings):
        self.settings = settings

    def translate_lines(self, lines: list[tuple[int, str]]) -> dict[int, str]:
        raise NotImplementedError


_TRANSLATEGEMMA_CACHE_LOCK = threading.RLock()
_TRANSLATEGEMMA_CACHED_MODEL: Any | None = None
_TRANSLATEGEMMA_CACHED_PROCESSOR: Any | None = None
_TRANSLATEGEMMA_CACHED_KEY: str | None = None


def _log_info(message: str) -> None:
    prefixed = f"[AI Prompt Translator] {message}"
    print(prefixed)
    logger.info(message)


def _log_warn(message: str) -> None:
    prefixed = f"[AI Prompt Translator] {message}"
    print(prefixed)
    logger.warning(message)


def create_provider(settings: TranslatorSettings) -> BaseTranslatorProvider:
    name = (settings.provider or "").strip().lower()
    if name == "gemini":
        return GeminiTranslator(settings)
    if name == "openai_compatible":
        return OpenAICompatibleTranslator(settings)
    if name == "codex":
        return CodexTranslator(settings)
    if name == "translategemma_local":
        return TranslateGemmaTranslator(settings)

    raise TranslationProviderError(f"Unsupported provider: {settings.provider}")


def build_user_payload(lines: list[tuple[int, str]]) -> str:
    payload = {
        "task": "translate_non_english_parts_only",
        "output_schema": {"translations": [{"id": "int", "text": "string"}]},
        "lines": [{"id": line_id, "text": text} for line_id, text in lines],
    }
    return json.dumps(payload, ensure_ascii=False)


def build_translategemma_payload(lines: list[tuple[int, str]]) -> str:
    return "\n".join(f"[{line_id}] {text}" for line_id, text in lines)


def parse_translation_json(raw_text: str, expected_ids: list[int]) -> dict[int, str]:
    data = _parse_first_json_object(raw_text)
    if not isinstance(data, dict):
        raise TranslationProviderError("Translator output is not a JSON object.")

    items = data.get("translations")
    if not isinstance(items, list):
        raise TranslationProviderError("Missing 'translations' list in translator output.")

    result: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            raise TranslationProviderError("Each translation item must be an object.")

        line_id = item.get("id")
        text = item.get("text")
        if not isinstance(line_id, int):
            raise TranslationProviderError("Translation item id must be an integer.")
        if not isinstance(text, str):
            raise TranslationProviderError("Translation item text must be a string.")

        if line_id in result:
            raise TranslationProviderError("Duplicate translation id in output.")
        result[line_id] = text

    expected_set = set(expected_ids)
    if set(result.keys()) != expected_set:
        raise TranslationProviderError("Translation ids mismatch with requested lines.")

    return result


def validate_line_integrity(source_line: str, translated_line: str) -> bool:
    for ch in PROTECTED_CHARS:
        if source_line.count(ch) != translated_line.count(ch):
            return False

    # Colon is validated only inside structural prompt regions ((), [], {}, <>)
    # so natural-language colons like "Elder Scroll: Skyrim" are allowed.
    if _count_structural_colons(source_line) != _count_structural_colons(translated_line):
        return False

    if _extract_digit_tokens(source_line) != _extract_digit_tokens(translated_line):
        return False

    return True


def _extract_digit_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    current = []
    for ch in text:
        if ch.isdigit():
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current = []

    if current:
        tokens.append("".join(current))

    return tokens


def _count_structural_colons(text: str) -> int:
    spans = _collect_balanced_spans(text)
    if not spans:
        return 0

    colon_positions: set[int] = set()
    for start, end in spans:
        for idx in range(start, end):
            if text[idx] == ":":
                colon_positions.add(idx)
    return len(colon_positions)


def _collect_balanced_spans(text: str) -> list[tuple[int, int]]:
    opener_to_closer = {"(": ")", "[": "]", "{": "}", "<": ">"}
    closer_to_opener = {v: k for k, v in opener_to_closer.items()}
    stack: list[tuple[str, int]] = []
    spans: list[tuple[int, int]] = []

    for idx, ch in enumerate(text):
        if ch in opener_to_closer:
            stack.append((ch, idx))
            continue

        expected_opener = closer_to_opener.get(ch)
        if expected_opener is None or not stack:
            continue

        opener, start_idx = stack[-1]
        if opener != expected_opener:
            continue

        stack.pop()
        spans.append((start_idx, idx + 1))

    return spans


def _parse_first_json_object(raw_text: str) -> Any:
    text = raw_text.strip()
    if text.startswith("```"):
        text = _strip_fenced_code(text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise TranslationProviderError("No JSON object found in translator output.")

    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise TranslationProviderError("Failed to parse translator JSON output.") from exc


def _strip_fenced_code(text: str) -> str:
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return text


class GeminiTranslator(BaseTranslatorProvider):
    def translate_lines(self, lines: list[tuple[int, str]]) -> dict[int, str]:
        api_key = (self.settings.gemini_api_key or "").strip()
        model = _normalize_gemini_model((self.settings.gemini_model or "").strip())
        if not api_key:
            raise TranslationProviderError("Gemini API key is empty.")
        if not model:
            raise TranslationProviderError("Gemini model is empty.")

        _log_info(
            f"Gemini request start | model={model} | lines={len(lines)}"
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        user_payload = build_user_payload(lines)

        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        }

        user_turn_contents = [
            {"role": "user", "parts": [{"text": TRANSLATION_SYSTEM_PROMPT}]},
            {"role": "user", "parts": [{"text": user_payload}]},
            {"role": "user", "parts": [{"text": "Ignore this extra user turn for translation. Use upwards turns for context, but do not include this turn in the translation."}]},
        ]

        # Primary call: structured JSON mode with schema.
        payload_primary = {
            "systemInstruction": {"parts": [{"text": "Strictly follow the user instructions below."}]},
            "contents": user_turn_contents,
            "safetySettings": GEMINI_SAFETY_SETTINGS,
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": TRANSLATION_RESPONSE_SCHEMA,
            },
        }

        # Fallback call: JSON mime only.
        payload_fallback = {
            "systemInstruction": {"parts": [{"text": "Strictly follow the user instructions below."}]},
            "contents": user_turn_contents,
            "safetySettings": GEMINI_SAFETY_SETTINGS,
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        }

        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = client.post(url, headers=headers, json=payload_primary)
            if response.status_code == 400:
                _log_warn(
                    "Gemini primary structured call returned 400; retrying with simplified JSON mode."
                )
                response = client.post(url, headers=headers, json=payload_fallback)

        if response.status_code >= 400:
            _log_warn(
                f"Gemini request failed | status={response.status_code} | body={response.text[:500]}"
            )
            raise TranslationProviderError(
                f"Gemini request failed: {response.status_code} {response.text[:400]}"
            )

        data = response.json()
        prompt_feedback = data.get("promptFeedback") or data.get("prompt_feedback")
        if isinstance(prompt_feedback, dict):
            block_reason = prompt_feedback.get("blockReason") or prompt_feedback.get("block_reason")
            if block_reason:
                _log_warn(f"Gemini prompt blocked | blockReason={block_reason}")

        text_parts: list[str] = []
        for candidate in data.get("candidates", []):
            content = candidate.get("content", {})
            for part in content.get("parts", []):
                maybe_text = part.get("text")
                if isinstance(maybe_text, str):
                    text_parts.append(maybe_text)

        text = "\n".join(text_parts).strip()
        if not text:
            _log_warn(
                f"Gemini returned empty text parts. Raw keys={list(data.keys())}"
            )
            raise TranslationProviderError("Gemini returned an empty response.")

        parsed = parse_translation_json(text, [line_id for line_id, _ in lines])
        _log_info(f"Gemini request success | translated_lines={len(parsed)}")
        return parsed


class OpenAICompatibleTranslator(BaseTranslatorProvider):
    def translate_lines(self, lines: list[tuple[int, str]]) -> dict[int, str]:
        base_url = (self.settings.openai_base_url or "").strip().rstrip("/")
        model = (self.settings.openai_model or "").strip()
        if not base_url:
            raise TranslationProviderError("OpenAI-compatible base URL is empty.")
        if not model:
            raise TranslationProviderError("OpenAI-compatible model is empty.")

        api_key = (self.settings.openai_api_key or "").strip()
        _log_info(
            f"OpenAI-compatible request start | model={model} | lines={len(lines)} | base_url={base_url}"
        )

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                {"role": "user", "content": build_user_payload(lines)},
            ],
            "temperature": 0,
        }

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = client.post(f"{base_url}/chat/completions", headers=headers, json=payload)

        if response.status_code >= 400:
            _log_warn(
                f"OpenAI-compatible request failed | status={response.status_code} | body={response.text[:500]}"
            )
            raise TranslationProviderError(
                f"OpenAI-compatible request failed: {response.status_code} {response.text[:400]}"
            )

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise TranslationProviderError("OpenAI-compatible response has no choices.")

        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, list):
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        else:
            text = str(content or "")

        if not text.strip():
            raise TranslationProviderError("OpenAI-compatible response content is empty.")

        parsed = parse_translation_json(text, [line_id for line_id, _ in lines])
        _log_info(f"OpenAI-compatible request success | translated_lines={len(parsed)}")
        return parsed


class CodexTranslator(BaseTranslatorProvider):
    def translate_lines(self, lines: list[tuple[int, str]]) -> dict[int, str]:
        model = (self.settings.codex_model or "").strip()
        if not model:
            raise TranslationProviderError("Codex model is empty.")

        tokens = self._load_codex_tokens()
        _log_info(
            f"Codex request start | model={model} | lines={len(lines)} | auth={tokens.auth_path}"
        )
        payload = self._build_payload(model, lines)

        body, status_code = self._send_codex_request(tokens, payload)
        if status_code in {401, 403} and tokens.refresh_token:
            _log_warn("Codex token seems expired; attempting refresh.")
            tokens = self._refresh_tokens(tokens)
            body, status_code = self._send_codex_request(tokens, payload)

        if status_code >= 400:
            _log_warn(
                f"Codex request failed | status={status_code} | body={body[:500]}"
            )
            raise TranslationProviderError(
                f"Codex request failed: {status_code} {body[:400]}"
            )

        text = self._extract_text_from_codex_sse(body)
        if not text.strip():
            raise TranslationProviderError("Codex response is empty.")

        parsed = parse_translation_json(text, [line_id for line_id, _ in lines])
        _log_info(f"Codex request success | translated_lines={len(parsed)}")
        return parsed

    def _build_payload(self, model: str, lines: list[tuple[int, str]]) -> dict[str, Any]:
        user_text = build_user_payload(lines)
        return {
            "model": model,
            "instructions": TRANSLATION_SYSTEM_PROMPT,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_text}],
                }
            ],
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "store": False,
            "stream": True,
        }

    def _send_codex_request(
        self, tokens: CodexTokens, payload: dict[str, Any]
    ) -> tuple[str, int]:
        headers = {
            "Authorization": f"Bearer {tokens.access_token}",
            "chatgpt-account-id": tokens.account_id,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "OpenAI-Beta": "responses=experimental",
            "originator": "codex_cli_rs",
        }
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = client.post(CODEX_RESPONSES_URL, headers=headers, json=payload)
        return response.text, response.status_code

    def _extract_text_from_codex_sse(self, raw_text: str) -> str:
        assembled = []
        has_delta = False

        for line in raw_text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue

            data_str = line[5:].strip()
            if not data_str:
                continue
            if data_str == "[DONE]":
                break

            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type")
            if event_type == "response.output_text.delta":
                delta = event.get("delta")
                if isinstance(delta, str):
                    assembled.append(delta)
                    has_delta = True
            elif event_type == "response.output_item.done" and not has_delta:
                item = event.get("item", {})
                for content in item.get("content", []):
                    text = content.get("text")
                    if isinstance(text, str):
                        assembled.append(text)
            elif event_type in {"response.failed", "error"}:
                message = (
                    event.get("response", {}).get("error", {}).get("message")
                    or event.get("error", {}).get("message")
                    or data_str
                )
                raise TranslationProviderError(f"Codex stream error: {message}")

        if not assembled:
            maybe_json = _try_parse_json(raw_text)
            if isinstance(maybe_json, dict):
                response_obj = maybe_json.get("response", {})
                for content in response_obj.get("content", []):
                    text = content.get("text")
                    if isinstance(text, str):
                        assembled.append(text)

        return "".join(assembled)

    def _load_codex_tokens(self) -> CodexTokens:
        auth_path = self._discover_auth_path()
        if not auth_path.exists():
            raise TranslationProviderError(f"Codex auth file not found: {auth_path}")

        try:
            raw = json.loads(auth_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise TranslationProviderError("Failed to parse Codex auth.json") from exc

        tokens_raw = raw.get("tokens")
        if isinstance(tokens_raw, dict):
            access = _as_non_empty_str(tokens_raw.get("access_token"))
            account = _as_non_empty_str(tokens_raw.get("account_id"))
            refresh = _as_non_empty_str(tokens_raw.get("refresh_token"))
            if access and account:
                return CodexTokens(access, account, refresh, auth_path, "nested", raw)

        access = _as_non_empty_str(raw.get("access_token"))
        account = _as_non_empty_str(raw.get("account_id"))
        refresh = _as_non_empty_str(raw.get("refresh_token"))
        if access and account:
            return CodexTokens(access, account, refresh, auth_path, "flat", raw)

        raise TranslationProviderError(
            "Codex auth.json does not contain usable access_token/account_id."
        )

    def _discover_auth_path(self) -> Path:
        configured = (self.settings.codex_auth_path or "").strip()
        if configured:
            return Path(configured).expanduser()

        candidates: list[Path] = []
        codex_home = os.getenv("CODEX_HOME")
        if codex_home:
            candidates.append(Path(codex_home) / "auth.json")

        if os.name == "nt":
            user_profile = os.getenv("USERPROFILE")
            if user_profile:
                candidates.append(Path(user_profile) / ".codex" / "auth.json")

        candidates.append(Path.home() / ".codex" / "auth.json")

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]

    def _refresh_tokens(self, tokens: CodexTokens) -> CodexTokens:
        if not tokens.refresh_token:
            raise TranslationProviderError("No refresh_token available for Codex.")

        form = {
            "grant_type": "refresh_token",
            "refresh_token": tokens.refresh_token,
            "client_id": CODEX_CLIENT_ID,
        }

        headers = {"Content-Type": "application/x-www-form-urlencoded"}
        with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = client.post(CODEX_TOKEN_URL, headers=headers, data=form)

        if response.status_code >= 400:
            raise TranslationProviderError(
                f"Codex token refresh failed: {response.status_code} {response.text[:400]}"
            )

        payload = response.json()
        new_access = _as_non_empty_str(payload.get("access_token"))
        if not new_access:
            raise TranslationProviderError("Codex token refresh returned no access_token.")

        new_refresh = _as_non_empty_str(payload.get("refresh_token")) or tokens.refresh_token
        new_account = (
            _extract_account_id_from_jwt(new_access)
            or _as_non_empty_str(payload.get("account_id"))
            or tokens.account_id
        )

        updated_raw = dict(tokens.raw)
        if tokens.mode == "nested":
            token_obj = dict(updated_raw.get("tokens") or {})
            token_obj["access_token"] = new_access
            token_obj["account_id"] = new_account
            if new_refresh:
                token_obj["refresh_token"] = new_refresh
            updated_raw["tokens"] = token_obj
        else:
            updated_raw["access_token"] = new_access
            updated_raw["account_id"] = new_account
            if new_refresh:
                updated_raw["refresh_token"] = new_refresh

        tokens.auth_path.write_text(
            json.dumps(updated_raw, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return CodexTokens(
            access_token=new_access,
            account_id=new_account,
            refresh_token=new_refresh,
            auth_path=tokens.auth_path,
            mode=tokens.mode,
            raw=updated_raw,
        )


class TranslateGemmaTranslator(BaseTranslatorProvider):
    def translate_lines(self, lines: list[tuple[int, str]]) -> dict[int, str]:
        hf_token = (self.settings.translategemma_hf_token or "").strip()
        if not hf_token:
            raise TranslationProviderError(
                "TranslateGemma Hugging Face token is empty. "
                f"Accept model access at {TRANSLATEGEMMA_MODEL_PAGE_URL} and set your token in Settings."
            )

        model_size = _normalize_translategemma_model_size(self.settings.translategemma_model_size)

        repo_id = TRANSLATEGEMMA_MODEL_REPOS[model_size]
        _log_info(
            f"TranslateGemma request start | model={repo_id} | lines={len(lines)}"
        )

        model, processor = self._load_model(repo_id, hf_token)
        parsed: dict[int, str] = {}
        for line_id, source_text in lines:
            source_lang_code = _guess_translategemma_source_lang(source_text)
            translated = self._generate_translation(
                model=model,
                processor=processor,
                source_text=source_text,
                source_lang_code=source_lang_code,
                target_lang_code="en",
                line_id=line_id,
            )
            parsed[line_id] = translated
        _log_info(
            f"TranslateGemma request success | translated_lines={len(parsed)}"
        )
        return parsed

    def _load_model(self, repo_id: str, hf_token: str) -> tuple[Any, Any]:
        global _TRANSLATEGEMMA_CACHED_MODEL
        global _TRANSLATEGEMMA_CACHED_PROCESSOR
        global _TRANSLATEGEMMA_CACHED_KEY

        model_key = repo_id
        with _TRANSLATEGEMMA_CACHE_LOCK:
            if (
                _TRANSLATEGEMMA_CACHED_MODEL is not None
                and _TRANSLATEGEMMA_CACHED_PROCESSOR is not None
                and _TRANSLATEGEMMA_CACHED_KEY == model_key
            ):
                _log_info(
                    f"TranslateGemma model cache hit | repo={repo_id}"
                )
                return _TRANSLATEGEMMA_CACHED_MODEL, _TRANSLATEGEMMA_CACHED_PROCESSOR

        torch, processor_cls, model_cls = self._load_runtime_dependencies()
        local_model_dir = self._translategemma_model_dir(repo_id)
        self._ensure_model_snapshot(repo_id, local_model_dir, hf_token)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = self._select_torch_dtype(torch)
        allow_cpu_offload = os.environ.get(
            "AIPT_TRANSLATEGEMMA_ALLOW_CPU_OFFLOAD",
            "",
        ).strip().lower() in {"1", "true", "yes", "on"}
        _log_info(
            f"TranslateGemma model load | repo={repo_id} | device={device}"
        )

        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": False,
        }
        if device == "cuda":
            if allow_cpu_offload:
                model_kwargs["device_map"] = "auto"
            else:
                model_kwargs["device_map"] = {"": 0}
        model_kwargs["torch_dtype"] = dtype
        model_kwargs["use_safetensors"] = True

        _log_info("TranslateGemma loading processor...")
        try:
            processor = processor_cls.from_pretrained(
                str(local_model_dir),
                token=hf_token,
                local_files_only=True,
                trust_remote_code=False,
                use_fast=False,
            )
        except Exception:
            _log_warn("TranslateGemma processor load with trust_remote_code=False failed; retry with True")
            processor = processor_cls.from_pretrained(
                str(local_model_dir),
                token=hf_token,
                local_files_only=True,
                trust_remote_code=True,
                use_fast=False,
            )
        _log_info("TranslateGemma loading model...")
        try:
            model = model_cls.from_pretrained(
                str(local_model_dir),
                **model_kwargs,
            )
        except Exception:
            _log_warn("TranslateGemma model load with trust_remote_code=False failed; retry with True")
            retry_kwargs = dict(model_kwargs)
            retry_kwargs["trust_remote_code"] = True
            model = model_cls.from_pretrained(
                str(local_model_dir),
                **retry_kwargs,
            )
        if device == "cpu":
            model = model.to("cpu")
        model.eval()
        self._sanitize_generation_config(model)

        if device == "cuda" and not allow_cpu_offload:
            hf_device_map = getattr(model, "hf_device_map", None)
            if hf_device_map:
                device_map_values = {str(v).lower() for v in hf_device_map.values()}
                if "cpu" in device_map_values:
                    raise TranslationProviderError(
                        "TranslateGemma loaded with CPU offload while GPU-only mode is enabled. "
                        "Use a smaller model, or set AIPT_TRANSLATEGEMMA_ALLOW_CPU_OFFLOAD=1 "
                        "if you explicitly want mixed CPU/GPU execution."
                    )

        with _TRANSLATEGEMMA_CACHE_LOCK:
            if (
                _TRANSLATEGEMMA_CACHED_MODEL is not None
                and _TRANSLATEGEMMA_CACHED_MODEL is not model
            ):
                try:
                    _TRANSLATEGEMMA_CACHED_MODEL = None
                    _TRANSLATEGEMMA_CACHED_PROCESSOR = None
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            _TRANSLATEGEMMA_CACHED_MODEL = model
            _TRANSLATEGEMMA_CACHED_PROCESSOR = processor
            _TRANSLATEGEMMA_CACHED_KEY = model_key

        _log_info(
            f"TranslateGemma model ready | repo={repo_id} | device={device}"
        )
        return model, processor

    def _sanitize_generation_config(self, model: Any) -> None:
        # Prevent noisy warnings from preset sampling fields when do_sample=False.
        config = getattr(model, "generation_config", None)
        if config is None:
            return
        try:
            config.do_sample = False
        except Exception:
            pass
        for attr in ("top_p", "top_k", "temperature"):
            if hasattr(config, attr):
                try:
                    setattr(config, attr, None)
                except Exception:
                    pass

    def _ensure_model_snapshot(self, repo_id: str, local_model_dir: Path, hf_token: str) -> None:
        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:
            raise TranslationProviderError(
                "huggingface_hub is required for TranslateGemma download. "
                "Install it in the WebUI environment and try again."
            ) from exc

        local_model_dir.mkdir(parents=True, exist_ok=True)
        _log_info(
            f"TranslateGemma checking/downloading model snapshot | repo={repo_id} | dir={local_model_dir}"
        )
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_model_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                token=hf_token,
            )
        except Exception as exc:
            raise TranslationProviderError(
                "TranslateGemma model download failed. "
                f"Verify model access at {TRANSLATEGEMMA_MODEL_PAGE_URL}, then check your Hugging Face token."
            ) from exc
        _log_info("TranslateGemma model snapshot ready")

    def _load_runtime_dependencies(self) -> tuple[Any, Any, Any]:
        try:
            import torch  # type: ignore
        except Exception as exc:
            raise TranslationProviderError("PyTorch is required for TranslateGemma.") from exc

        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor  # type: ignore
        except Exception as exc:
            raise TranslationProviderError(
                "TranslateGemma requires transformers with AutoProcessor and "
                "AutoModelForImageTextToText support."
            ) from exc

        return torch, AutoProcessor, AutoModelForImageTextToText

    def _generate_translation(
        self,
        model: Any,
        processor: Any,
        source_text: str,
        source_lang_code: str,
        target_lang_code: str,
        line_id: int,
    ) -> str:
        # ComfyUI-style auto flow:
        # 1) structured processor.apply_chat_template
        # 2) retry once with relaxed <end_of_turn> stopping
        # 3) plain prompt fallback
        # 4) fail if still empty
        attempts = [
            ("structured", True, False),
            ("structured-relaxed-eot", True, True),
            ("plain", False, False),
        ]
        errors: list[str] = []

        for label, use_structured, relaxed_eot in attempts:
            try:
                text = self._generate_translation_once(
                    model=model,
                    processor=processor,
                    source_text=source_text,
                    source_lang_code=source_lang_code,
                    target_lang_code=target_lang_code,
                    line_id=line_id,
                    use_structured=use_structured,
                    relaxed_eot=relaxed_eot,
                )
            except Exception as exc:
                errors.append(f"{label}: {exc}")
                _log_warn(
                    "TranslateGemma attempt failed | "
                    f"line_id={line_id} | mode={label} | reason={exc}"
                )
                continue

            if text:
                return text

            errors.append(f"{label}: empty output")
            _log_warn(
                "TranslateGemma empty output | "
                f"line_id={line_id} | mode={label}"
            )

        raise TranslationProviderError(
            "TranslateGemma returned an empty response after Comfy-compatible retries. "
            f"attempts={'; '.join(errors)}"
        )

    def _generate_translation_once(
        self,
        model: Any,
        processor: Any,
        source_text: str,
        source_lang_code: str,
        target_lang_code: str,
        line_id: int,
        use_structured: bool,
        relaxed_eot: bool,
    ) -> str:
        try:
            import torch  # type: ignore
        except Exception as exc:
            raise TranslationProviderError("PyTorch is required for TranslateGemma.") from exc

        tokenizer = getattr(processor, "tokenizer", processor)
        inputs, used_path = self._build_translategemma_inputs(
            processor=processor,
            tokenizer=tokenizer,
            source_text=source_text,
            source_lang_code=source_lang_code,
            target_lang_code=target_lang_code,
            use_structured=use_structured,
        )
        if inputs is None:
            raise TranslationProviderError("TranslateGemma tokenizer/processor is not usable.")

        model_device = _infer_model_device(model, torch)
        moved_inputs: dict[str, Any] = {}
        for key, value in dict(inputs).items():
            if hasattr(value, "to"):
                moved_inputs[key] = value.to(model_device)
            else:
                moved_inputs[key] = value
        inputs = moved_inputs

        prompt_len = 0
        input_ids = inputs.get("input_ids")
        if input_ids is not None and hasattr(input_ids, "shape"):
            prompt_len = int(input_ids.shape[-1])

        _log_info(
            "TranslateGemma generation start | "
            f"line_id={line_id} | source_lang={source_lang_code} -> target_lang={target_lang_code} | "
            f"path={used_path} | relaxed_eot={relaxed_eot}"
        )
        hf_device_map = getattr(model, "hf_device_map", None)
        if hf_device_map:
            device_map_values = {str(v) for v in hf_device_map.values()}
            _log_info(
                "TranslateGemma runtime devices | "
                f"line_id={line_id} | input_device={model_device} | "
                f"hf_device_map_devices={sorted(device_map_values)}"
            )
        else:
            _log_info(
                "TranslateGemma runtime devices | "
                f"line_id={line_id} | input_device={model_device} | hf_device_map=none"
            )

        max_new_tokens = _suggest_translategemma_max_new_tokens(prompt_len)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "use_cache": True,
        }
        _log_info(
            "TranslateGemma generation config | "
            f"line_id={line_id} | max_new_tokens={max_new_tokens} | use_cache={gen_kwargs['use_cache']}"
        )
        eos_ids = _collect_eos_token_ids(tokenizer, include_end_of_turn=(not relaxed_eot))
        if eos_ids:
            gen_kwargs["eos_token_id"] = eos_ids if len(eos_ids) > 1 else eos_ids[0]
            gen_kwargs["pad_token_id"] = int(
                getattr(tokenizer, "eos_token_id", eos_ids[0]) or eos_ids[0]
            )

        if prompt_len > 0:
            min_gen = 8 if relaxed_eot else 1
            stopping_criteria = _build_end_of_turn_stopping_criteria(
                tokenizer=tokenizer,
                prompt_len=prompt_len,
                min_gen=min_gen,
            )
            if stopping_criteria is not None:
                gen_kwargs["stopping_criteria"] = stopping_criteria

        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_kwargs)

        output_len = 0
        if hasattr(outputs, "shape"):
            output_len = int(outputs.shape[-1])
        _log_info(
            f"TranslateGemma generation shapes | line_id={line_id} | prompt_tokens={prompt_len} | output_tokens={output_len}"
        )

        if outputs is None or len(outputs) == 0:
            raise TranslationProviderError("TranslateGemma generate() returned no outputs.")

        # Match Comfy decoding path exactly: decode continuation only.
        generated_ids = outputs[0][prompt_len:]
        generated_preview = _safe_tensor_preview(generated_ids, max_items=24)
        _log_info(
            f"TranslateGemma generated token preview | line_id={line_id} | {generated_preview}"
        )
        token_stats = _safe_token_stats(generated_ids)
        if token_stats["total"] >= 16 and token_stats["unique"] == 1 and token_stats["first"] == 0:
            raise TranslationProviderError(
                "TranslateGemma generated PAD-only tokens (all zero ids)."
            )
        translated_text = self._decode_translategemma_output(
            tokenizer=tokenizer,
            processor=processor,
            generated_ids=generated_ids,
        ).strip()

        _log_info(
            "TranslateGemma translated text preview | "
            f"line_id={line_id} | text={_safe_log_text_preview(translated_text)}"
        )
        _log_info("TranslateGemma generation done")
        return translated_text

    def _build_translategemma_inputs(
        self,
        processor: Any,
        tokenizer: Any,
        source_text: str,
        source_lang_code: str,
        target_lang_code: str,
        use_structured: bool,
    ) -> tuple[Any | None, str]:
        if use_structured and hasattr(processor, "apply_chat_template"):
            messages = self._build_structured_messages(
                input_text=source_text,
                source_lang_code=source_lang_code,
                target_lang_code=target_lang_code,
            )
            try:
                inputs = processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                return inputs, "processor.apply_chat_template(structured)"
            except Exception as exc:
                raise TranslationProviderError(
                    "Structured mode requested but processor.apply_chat_template failed."
                ) from exc

        prompt = self._render_plain_or_chat_template_prompt(
            tokenizer=tokenizer,
            input_text=source_text,
            source_lang_code=source_lang_code,
            target_lang_code=target_lang_code,
        )

        if callable(getattr(tokenizer, "__call__", None)):
            return tokenizer(prompt, return_tensors="pt"), "tokenizer(prompt)"

        if callable(getattr(processor, "__call__", None)):
            try:
                return processor(text=prompt, return_tensors="pt"), "processor(text=prompt)"
            except TypeError:
                return processor(prompt, return_tensors="pt"), "processor(prompt)"

        return None, "unavailable"

    def _render_plain_or_chat_template_prompt(
        self,
        tokenizer: Any,
        input_text: str,
        source_lang_code: str,
        target_lang_code: str,
    ) -> str:
        # Comfy prompt_builder-compatible fallback order:
        # structured(chat_template) -> simple(chat_template) -> plain prompt.
        if getattr(tokenizer, "chat_template", None):
            structured_messages = self._build_structured_messages(
                input_text=input_text,
                source_lang_code=source_lang_code,
                target_lang_code=target_lang_code,
            )
            rendered = self._try_apply_chat_template(
                tokenizer=tokenizer,
                messages=structured_messages,
            )
            if rendered is not None:
                return rendered

            simple_messages = self._build_simple_messages(
                input_text=input_text,
                source_lang_code=source_lang_code,
                target_lang_code=target_lang_code,
            )
            rendered = self._try_apply_chat_template(
                tokenizer=tokenizer,
                messages=simple_messages,
            )
            if rendered is not None:
                return rendered

        return self._build_plain_prompt(
            input_text=input_text,
            source_lang_code=source_lang_code,
            target_lang_code=target_lang_code,
        )

    def _try_apply_chat_template(self, tokenizer: Any, messages: list[dict[str, Any]]) -> str | None:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return None

    def _build_structured_messages(
        self,
        input_text: str,
        source_lang_code: str,
        target_lang_code: str,
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": input_text,
                        "source_lang_code": source_lang_code or "en",
                        "target_lang_code": target_lang_code,
                    }
                ],
            }
        ]

    def _build_simple_messages(
        self,
        input_text: str,
        source_lang_code: str,
        target_lang_code: str,
    ) -> list[dict[str, str]]:
        target_name = _translategemma_language_name(target_lang_code)
        if source_lang_code:
            source_name = _translategemma_language_name(source_lang_code)
            instruction = (
                f"You are a professional {source_name} ({source_lang_code}) to {target_name} ({target_lang_code}) translator.\n"
                f"Produce only the {target_name} translation, without any additional explanations or commentary.\n\n"
                f"Please translate the following {source_name} text into {target_name}:\n\n{input_text}"
            )
        else:
            instruction = (
                f"Produce only the {target_name} translation, without any additional explanations or commentary.\n\n"
                f"Please translate the following text into {target_name}:\n\n{input_text}"
            )
        return [{"role": "user", "content": instruction}]

    def _build_plain_prompt(
        self,
        input_text: str,
        source_lang_code: str,
        target_lang_code: str,
    ) -> str:
        target_name = _translategemma_language_name(target_lang_code)
        if source_lang_code:
            source_name = _translategemma_language_name(source_lang_code)
            return (
                f"You are a professional {source_name} ({source_lang_code}) to {target_name} ({target_lang_code}) translator. "
                f"Your goal is to accurately convey the meaning and nuances of the original {source_name} text while adhering to "
                f"{target_name} grammar, vocabulary, and cultural sensitivities.\n\n"
                f"Produce only the {target_name} translation, without any additional explanations or commentary. "
                f"Please translate the following {source_name} text into {target_name}:\n\n\n"
                f"{input_text.strip()}"
            )
        return (
            f"Produce only the {target_name} translation, without any additional explanations or commentary. "
            f"Please translate the following text into {target_name}:\n\n\n"
            f"{input_text.strip()}"
        )

    def _decode_translategemma_output(
        self,
        tokenizer: Any,
        processor: Any,
        generated_ids: Any,
    ) -> str:
        ids = generated_ids
        try:
            if hasattr(ids, "detach"):
                ids = ids.detach()
            if hasattr(ids, "cpu"):
                ids = ids.cpu()
        except Exception:
            pass

        def _try_decode(obj: Any, skip_special_tokens: bool) -> str:
            if obj is None or not hasattr(obj, "decode"):
                return ""
            try:
                decoded = obj.decode(ids, skip_special_tokens=skip_special_tokens)
                if isinstance(decoded, str):
                    return decoded
                return str(decoded or "")
            except Exception:
                return ""

        # Prefer raw decode first so we can cut output at the first end-of-turn marker.
        raw_candidates = [
            _try_decode(tokenizer, False),
            _try_decode(processor, False),
        ]
        for raw in raw_candidates:
            first_turn = _extract_first_turn_text(raw)
            cleaned = _strip_common_special_tokens_text(first_turn)
            if cleaned:
                return cleaned

        candidates = [
            _try_decode(processor, True),
            _try_decode(tokenizer, True),
        ]
        for candidate in candidates:
            cleaned = _strip_common_special_tokens_text(candidate)
            if cleaned:
                return cleaned

        if hasattr(tokenizer, "batch_decode"):
            try:
                batch = tokenizer.batch_decode([ids], skip_special_tokens=True)
                if batch and isinstance(batch[0], str):
                    cleaned = _strip_common_special_tokens_text(batch[0])
                    if cleaned:
                        return cleaned
            except Exception:
                pass

        raise TranslationProviderError("TranslateGemma decode returned empty text.")

    def _select_torch_dtype(self, torch: Any) -> Any:
        if not torch.cuda.is_available():
            return torch.float32
        if _cuda_supports_bf16(torch):
            return torch.bfloat16
        return torch.float16

    def _translategemma_model_dir(self, repo_id: str) -> Path:
        repo_name = repo_id.split("/")[-1]
        return Path(__file__).resolve().parents[1] / "models" / "translategemma" / repo_name


def _as_non_empty_str(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _try_parse_json(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_account_id_from_jwt(token: str) -> str | None:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        payload = json.loads(_decode_base64url(parts[1]))
        auth_claim = payload.get("https://api.openai.com/auth", {})
        return (
            _as_non_empty_str(auth_claim.get("chatgpt_account_id"))
            or _as_non_empty_str(auth_claim.get("account_id"))
            or _as_non_empty_str(payload.get("chatgpt_account_id"))
            or _as_non_empty_str(payload.get("account_id"))
        )
    except Exception:
        return None


def _decode_base64url(value: str) -> str:
    pad = "=" * ((4 - len(value) % 4) % 4)
    decoded = base64.urlsafe_b64decode((value + pad).encode("ascii"))
    return decoded.decode("utf-8")


def _normalize_gemini_model(model: str) -> str:
    if model.startswith("models/"):
        return model[len("models/") :]
    return model


def _normalize_translategemma_model_size(model_size: str) -> str:
    value = (model_size or "").strip().upper()
    if value in TRANSLATEGEMMA_MODEL_REPOS:
        return value

    if value.endswith("B") and value[:-1].isdigit():
        candidate = f"{int(value[:-1])}B"
        if candidate in TRANSLATEGEMMA_MODEL_REPOS:
            return candidate

    if value.isdigit():
        candidate = f"{int(value)}B"
        if candidate in TRANSLATEGEMMA_MODEL_REPOS:
            return candidate

    return "4B"


def _cuda_supports_bf16(torch: Any) -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        return int(major) >= 8
    except Exception:
        return False


def _infer_model_device(model: Any, torch: Any) -> Any:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _collect_eos_token_ids(tokenizer: Any, include_end_of_turn: bool = True) -> list[int]:
    if tokenizer is None:
        return []

    eos_ids: list[int] = []
    base_eos = getattr(tokenizer, "eos_token_id", None)
    if isinstance(base_eos, (list, tuple)):
        eos_ids.extend(int(x) for x in base_eos if x is not None)
    elif base_eos is not None:
        eos_ids.append(int(base_eos))

    if include_end_of_turn:
        try:
            end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if end_of_turn_id is not None and (unk_id is None or int(end_of_turn_id) != int(unk_id)):
                eos_ids.append(int(end_of_turn_id))
        except Exception:
            pass

    return sorted(set(eos_ids))


def _build_end_of_turn_stopping_criteria(
    tokenizer: Any,
    prompt_len: int,
    min_gen: int,
) -> Any | None:
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList  # type: ignore
    except Exception:
        return None

    try:
        end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if end_of_turn_id is None or (unk_id is not None and int(end_of_turn_id) == int(unk_id)):
            return None
        end_of_turn_id = int(end_of_turn_id)
    except Exception:
        return None

    class _EndOfTurnMinGenStop(StoppingCriteria):
        def __init__(self, prompt_len: int, min_gen: int, end_of_turn_id: int):
            self.prompt_len = int(prompt_len)
            self.min_gen = int(min_gen)
            self.end_of_turn_id = int(end_of_turn_id)

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            if input_ids is None or input_ids.numel() == 0:
                return False
            ids = input_ids[0] if input_ids.dim() == 2 else input_ids
            gen_len = int(ids.shape[0]) - self.prompt_len
            if gen_len < self.min_gen:
                return False
            return bool(int(ids[-1].item()) == self.end_of_turn_id)

    return StoppingCriteriaList(
        [_EndOfTurnMinGenStop(prompt_len=int(prompt_len), min_gen=int(min_gen), end_of_turn_id=end_of_turn_id)]
    )


def _safe_tensor_preview(ids: Any, max_items: int = 24) -> str:
    try:
        if hasattr(ids, "detach"):
            ids = ids.detach()
        if hasattr(ids, "cpu"):
            ids = ids.cpu()
        if hasattr(ids, "tolist"):
            values = ids.tolist()
            if isinstance(values, list) and values and isinstance(values[0], list):
                values = values[0]
            if not isinstance(values, list):
                values = [values]
            values = [int(x) for x in values]
            preview = values[:max_items]
            unique_count = len(set(values))
            return f"tokens={preview} | total={len(values)} | unique={unique_count}"
    except Exception:
        pass
    return "tokens=<unavailable>"


def _safe_token_stats(ids: Any) -> dict[str, int]:
    try:
        if hasattr(ids, "detach"):
            ids = ids.detach()
        if hasattr(ids, "cpu"):
            ids = ids.cpu()
        if hasattr(ids, "tolist"):
            values = ids.tolist()
            if isinstance(values, list) and values and isinstance(values[0], list):
                values = values[0]
            if not isinstance(values, list):
                values = [values]
            ints = [int(x) for x in values]
            if not ints:
                return {"total": 0, "unique": 0, "first": -1}
            return {"total": len(ints), "unique": len(set(ints)), "first": int(ints[0])}
    except Exception:
        pass
    return {"total": 0, "unique": 0, "first": -1}


def _suggest_translategemma_max_new_tokens(prompt_len: int) -> int:
    # Translation prompts are short; huge generation windows make runs much slower
    # and can hide as CPU-heavy behavior due long token-by-token loops.
    if prompt_len <= 128:
        return 192
    if prompt_len <= 256:
        return 256
    return 320


def _extract_first_turn_text(text: str) -> str:
    if not text:
        return ""

    result = str(text)
    for marker in ("<end_of_turn>", "</s>", "<eos>"):
        idx = result.find(marker)
        if idx != -1:
            result = result[:idx]
            break

    result = result.replace("<start_of_turn>model", " ")
    result = result.replace("<start_of_turn>", " ")
    result = result.replace("model\n", " ")
    result = result.replace("assistant\n", " ")
    return result.strip()


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


def _strip_common_special_tokens_text(text: str) -> str:
    if not text:
        return ""
    cleaned = str(text)
    tokens = [
        "<pad>",
        "<eos>",
        "<bos>",
        "<end_of_turn>",
        "<start_of_turn>",
        "<s>",
        "</s>",
    ]
    for token in tokens:
        cleaned = cleaned.replace(token, " ")
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def _translategemma_language_name(lang_code: str) -> str:
    language_map = {
        "en": "English",
        "ko": "Korean",
        "ja": "Japanese",
        "zh": "Chinese",
        "zh_Hant": "Traditional Chinese",
        "ru": "Russian",
        "ar": "Arabic",
        "th": "Thai",
        "hi": "Hindi",
    }
    return language_map.get((lang_code or "").strip(), (lang_code or "").strip() or "text")


def _guess_translategemma_source_lang(text: str) -> str:
    for ch in text:
        code = ord(ch)
        # Hangul
        if 0xAC00 <= code <= 0xD7A3:
            return "ko"
        # Hiragana / Katakana
        if 0x3040 <= code <= 0x30FF:
            return "ja"
        # CJK Unified Ideographs
        if 0x4E00 <= code <= 0x9FFF:
            return "zh"
        # Cyrillic
        if 0x0400 <= code <= 0x04FF:
            return "ru"
        # Arabic
        if 0x0600 <= code <= 0x06FF:
            return "ar"
        # Thai
        if 0x0E00 <= code <= 0x0E7F:
            return "th"
        # Devanagari
        if 0x0900 <= code <= 0x097F:
            return "hi"

    # Fallback for mixed/unknown scripts.
    return "en"
