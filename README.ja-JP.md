# sd-ai-prompt-translator

[English](README.md) | [한국어](README.ko-KR.md) | [日本語](README.ja-JP.md)

画像生成の直前に、メインプロンプト内の非英語テキストを英語へ自動翻訳する Stable Diffusion WebUI 拡張です。

## 主な機能

- `txt2img` と `img2img` の両方に対応
- 専用入力欄は不要（標準の WebUI プロンプト欄をそのまま使用）
- 翻訳対象はメインプロンプトのみ
- ネガティブプロンプトは変更しない
- 非英語文字がある場合のみ翻訳
- 翻訳できない場合でも元のプロンプトで生成を継続

## 対応プロバイダー

- Gemini API
- OpenAI 互換 API エンドポイント（例: Ollama）
- Codex（ローカル認証ベース）
- TranslateGemma Local（初回実行時に Hugging Face モデルを自動ダウンロード）

## インストール方法（WebUI）

1. WebUI の `Extensions` タブを開く
2. `Install from URL` を選択
3. 次のリポジトリ URL を貼り付け:
   `https://github.com/david419kr/sd-ai-prompt-translator`
4. `Install` をクリック
5. WebUI を再起動

## 設定手順

1. `Settings > Extensions > AI Prompt Translator` を開く
2. プロバイダーを選択
3. 必要な設定値を入力
4. 設定を適用し、通常どおり Generate を実行

### TranslateGemma クイック設定

1. プロバイダーで `translategemma_local` を選択
2. `https://huggingface.co/google/translategemma-4b-it` でモデル利用許可に同意
3. Hugging Face の read token を作成
4. `TranslateGemma Hugging Face token` にトークンを入力
5. モデルサイズ（`4B` / `12B` / `27B`）を選択
6. 初回 Generate 時に選択モデルを自動ダウンロードし、進行状況は WebUI コンソールに表示されます

## 生成画面 UI

- 生成タブに `AI Prompt Translator` の折りたたみメニューが表示されます。
- 左側チェックボックスでその実行時の ON/OFF を切り替えます。
- `Toggle startup default` で `txt2img`/`img2img` の既定 ON/OFF を同時変更できます（再起動が必要）。
- `Clear Cached Translations` で保存済み翻訳キャッシュを削除できます。

## 翻訳キャッシュ（クレジット節約）

- 翻訳結果は JSON キャッシュに保存されます。
- 最大 100 件まで保持します。
- 同じプロンプトを再実行する場合、API を再呼び出しせずキャッシュを再利用します。
- デフォルトでは有効です。`Disable Translation Caching` を ON にすると無効化できます。
- 必要なら `Clear Cached Translations` ですぐに削除できます。

## 補足

- プロバイダー設定が不足している場合は翻訳をスキップし、コンソールに案内メッセージを表示します。
- Dynamic Prompts と同時に導入している場合は、この拡張が先に実行されます。

