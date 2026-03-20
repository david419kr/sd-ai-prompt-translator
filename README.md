# sd-ai-prompt-translator

[English](README.md) | [한국어](README.ko-KR.md) | [日本語](README.ja-JP.md)

Auto-translates non-English text in the main prompt to English right before image generation.

## What It Does

- Works in both `txt2img` and `img2img`.
- Uses the default WebUI prompt box (no separate prompt input).
- Translates only the main prompt.
- Leaves negative prompt unchanged.
- Translates only when non-English letters are detected.
- If translation cannot run, generation continues with the original prompt.

## Supported Providers

- Gemini API
- OpenAI-compatible API endpoints (for example Ollama)
- Codex (local auth-based mode)
- TranslateGemma Local (Hugging Face model download on first run)

## Installation (WebUI)

1. Open `Extensions` tab in WebUI.
2. Go to `Install from URL`.
3. Paste this repository URL:
   `https://github.com/david419kr/sd-ai-prompt-translator`
4. Click `Install`.
5. Restart WebUI.

## Setup

1. Open `Settings > Extensions > AI Prompt Translator`.
2. Choose a provider.
3. Fill provider/model settings.
4. Apply settings and use Generate as usual.

### TranslateGemma Quick Setup

1. Set provider to `translategemma_local`.
2. Choose `TranslateGemma model / quantization` in Settings (single dropdown):
   - `4B - Q4_K_M [2.31GB] <lightest>`
   - `4B - Q8 [3.84GB] <recommended>`
   - `4B - Full [8.04GB]`
   - `12B - i1-Q4_K_M [6.79GB] <recommended>`
   - `12B - i1-Q6_K [8.99GB]`
   - `12B - Full [22.7GB]`
3. `Full` requires a Hugging Face token in `TranslateGemma Hugging Face token`. Non-`Full` modes can run without token.
4. If you use `Full`, open `https://huggingface.co/google/translategemma-4b-it`, accept access, then create and paste a Hugging Face read token.
5. After changing TranslateGemma model mode, required model files are downloaded automatically on first run (progress is shown in WebUI console).

## Generation UI

- `AI Prompt Translator` foldable menu is visible in generation tabs.
- Left checkbox enables/disables translator for that run.
- Provider/model settings are managed in `Settings > Extensions > AI Prompt Translator`.
- `Toggle startup default` button sets default ON/OFF for both `txt2img` and `img2img` (restart required).
- `Disable Translation Caching` lets you bypass persistent cache for that run.
- `Clear Cached Translations` button clears saved translations.

## Translation Cache (Credit Saving)

- Saved in a JSON cache file.
- Stores up to 100 entries.
- If the same prompt is seen again, cached translation is reused without a new API request.
- Enabled by default. You can disable persistent cache with `Disable Translation Caching`.
- Use `Clear Cached Translations` to clear the cache anytime.

## Notes

- If provider settings are incomplete, translation is skipped and a console guidance message is shown.
- When both extensions are installed, this extension runs before Dynamic Prompts.

