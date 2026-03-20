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
3. プロバイダー/モデルの詳細設定を入力
4. 設定を適用し、通常どおり Generate を実行

### TranslateGemma クイック設定

1. プロバイダーで `translategemma_local` を選択
2. Settings の単一ドロップダウン `TranslateGemma model / quantization` でモードを選択
3. 選択可能な項目:
   - `4B - Q4_K_M [2.31GB] <lightest>`
   - `4B - Q8 [3.84GB] <recommended>`
   - `4B - Full [8.04GB]`
   - `12B - i1-Q4_K_M [6.79GB] <recommended>`
   - `12B - i1-Q6_K [8.99GB]`
   - `12B - Full [22.7GB]`
4. `Full` を使う場合のみ `TranslateGemma Hugging Face token` が必要です。`Full` 以外は空欄でも動作します。
5. `Full` を使う場合は `https://huggingface.co/google/translategemma-4b-it` でアクセス許可に同意し、read token を発行して入力してください。
6. TranslateGemma モード変更後の初回実行時に必要なモデルファイルを自動ダウンロードし、進行状況は WebUI コンソールに表示されます。

## 生成画面 UI

- 生成タブに `AI Prompt Translator` の折りたたみメニューが表示されます。
- 左側チェックボックスでその実行時の ON/OFF を切り替えます。
- プロバイダー/モデルの詳細設定は `Settings > Extensions > AI Prompt Translator` で管理します。
- `Toggle startup default` で `txt2img`/`img2img` の既定 ON/OFF を同時変更できます（再起動が必要）。
- `Disable Translation Caching` で、その実行時の永続キャッシュ利用を無効化できます。
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

