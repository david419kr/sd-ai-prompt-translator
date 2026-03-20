from __future__ import annotations

import base64
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

PROTECTED_CHARS = "()[]{}<>|:_"
REQUEST_TIMEOUT_SECONDS = 45.0

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

    raise TranslationProviderError(f"Unsupported provider: {settings.provider}")


def build_user_payload(lines: list[tuple[int, str]]) -> str:
    payload = {
        "task": "translate_non_english_parts_only",
        "output_schema": {"translations": [{"id": "int", "text": "string"}]},
        "lines": [{"id": line_id, "text": text} for line_id, text in lines],
    }
    return json.dumps(payload, ensure_ascii=False)


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
