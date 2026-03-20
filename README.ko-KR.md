# sd-ai-prompt-translator

[English](README.md) | [한국어](README.ko-KR.md) | [日本語](README.ja-JP.md)

이미지 생성 직전에 메인 프롬프트의 비영어 텍스트를 영어로 자동 번역하는 Stable Diffusion WebUI 확장입니다.

## 주요 기능

- `txt2img`, `img2img` 모두 지원
- 별도 입력창 없이 기본 WebUI 프롬프트 입력창 그대로 사용
- 메인 프롬프트만 번역
- 네거티브 프롬프트는 변경하지 않음
- 비영어 문자가 있을 때만 번역 수행
- 번역을 수행할 수 없는 경우에도 원문으로 생성 계속 진행

## 지원 Provider

- Gemini API
- OpenAI 호환 API 엔드포인트(예: Ollama)
- Codex(로컬 인증 기반 모드)
- TranslateGemma Local(첫 실행 시 Hugging Face 모델 자동 다운로드)

## 설치 방법 (WebUI)

1. WebUI의 `Extensions` 탭으로 이동
2. `Install from URL` 선택
3. 아래 리포지토리 URL 붙여넣기:
   `https://github.com/david419kr/sd-ai-prompt-translator`
4. `Install` 클릭
5. WebUI 재시작

## 설정 방법

1. `Settings > Extensions > AI Prompt Translator`로 이동
2. Provider 선택
3. 해당 Provider 설정값 입력
4. 설정 적용 후 평소처럼 Generate 실행

### TranslateGemma 빠른 설정

1. Provider를 `translategemma_local`로 선택
2. `https://huggingface.co/google/translategemma-4b-it` 페이지에서 모델 접근 권한 승인
3. Hugging Face read token 발급
4. `TranslateGemma Hugging Face token` 항목에 토큰 입력
5. 모델 크기(`4B` / `12B` / `27B`) 선택
6. 첫 Generate 시 선택 모델이 자동 다운로드되며, 진행 상태는 WebUI 콘솔에서 확인 가능

## 생성 화면 UI

- 생성 탭에 `AI Prompt Translator` 접이식 메뉴가 표시됩니다.
- 좌측 체크박스로 해당 실행에서 ON/OFF를 제어합니다.
- `Toggle startup default` 버튼으로 `txt2img`/`img2img` 기본 ON/OFF를 동시에 변경할 수 있습니다. (재시작 필요)
- `Clear Cached Translations` 버튼으로 저장된 번역 캐시를 비울 수 있습니다.

## 번역 캐시 (크레딧 절감)

- JSON 파일 기반으로 번역 결과를 저장합니다.
- 최대 100개까지 저장합니다.
- 동일 프롬프트 재사용 시 API를 다시 호출하지 않고 캐시 번역을 재사용합니다.
- 기본값은 활성화(ON)이며, `Disable Translation Caching`을 켜서 비활성화할 수 있습니다.
- 필요 시 `Clear Cached Translations`로 즉시 초기화할 수 있습니다.

## 참고

- Provider 설정이 비어 있으면 번역은 자동 스킵되고 콘솔에 안내 문구가 출력됩니다.
- Dynamic Prompts와 함께 설치된 경우, 이 확장이 먼저 실행됩니다.

