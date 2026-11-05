# 서버 에이전트 작업 지시서: 결빙 방지 세계 진술 수집

작성: 2026-11-05. **`SERVER_TASK.md`(Llama-3.3-70B까지)가 끝나고 결과를 push한 뒤에** 시작한다. 이 파일 하나만 읽고 작업할 수 있게 썼다. 배경은 `docs/research/ccpp-anti-icing-world.md` §6·§15에 있다.

## 목적

모델 다섯 개에 결빙 방지 세계에 대한 진술을 새로 묻는다. 수집만 하고 분석은 하지 않는다.

## 할 일

| 순서 | tag | Hugging Face 모델 | 필요 메모리 (대략) | 결과 파일과 완료 기준 |
|---|---|---|---|---|
| 1 | `gpt-oss-120b` | `openai/gpt-oss-120b` | 약 70 GiB | `analysis/llm_statements/runs/ice/gpt-oss-120b.jsonl` 630줄 |
| 2 | `qwen3-30b-a3b-instruct` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 약 64 GiB | `runs/ice/qwen3-30b-a3b-instruct.jsonl` 630줄 |
| 3 | `gpt-oss-20b` | `openai/gpt-oss-20b` | 약 20 GiB | `runs/ice/gpt-oss-20b.jsonl` 630줄 |
| 4 | `gemma3-27b` | `google/gemma-3-27b-it` | 약 64 GiB | `runs/ice/gemma3-27b.jsonl` 630줄 |
| 5 | `llama33-70b` | `meta-llama/Llama-3.3-70B-Instruct` | 약 150 GiB | `runs/ice/llama33-70b.jsonl` 630줄 |

모델마다 `runs/env/<tag>-ice.txt`(환경 기록)도 생긴다. 기존 `runs/env/<tag>.txt`는 덮어쓰지 않는다.

## 실행 방법

1. 저장소 최상위에서 `git pull`을 실행한다.
2. 모델마다 다음을 실행한다. 이미 끝난 부분은 건너뛰고, 끊기면 같은 명령으로 이어서 받는다.
   ```bash
   ICE=1 ONLY=<tag> bash analysis/llm_statements/run_all.sh
   ```
   - MIG 조각이면 `SERVER_TASK.md`와 같은 방식으로 `TP`, `EXTRA_VARGS`를 준다(pipeline-parallel, ray).
   - 스크립트 안에서 다음 순서로 진행한다.
     - vLLM 시작
     - 점검 14회: 응답이 모두 읽히는지 확인
     - 본 수집 630회
     - 결과 점검: `check_ice.py`
     - vLLM 종료
3. 끝나면 결과를 올린다.
   ```bash
   git add analysis/llm_statements/runs/ice analysis/llm_statements/runs/env
   git commit -m "Anti-icing stage runs"
   git push
   ```

## 바꾸면 안 되는 것

다음은 사전 등록된 조건이므로 수정하지 않는다.
- 프롬프트(`prompts.py`, `collect.py`)
- 표본 수(21회 = 온도 0.7로 20회 + 온도 0으로 1회)
- `window_summaries.json`
- `docs/` 아래 문서

`runs/` 아래 결과는 손으로 고치거나 지우지 않는다.

## 상황별 처리

`SERVER_TASK.md`의 "상황별 처리"와 같다.
- 양자화 모델로 바꾸지 않는다.
- 토큰을 직접 입력하거나 만들지 않는다.
- 강제 push(`--force`)를 하지 않는다.
- **점검 실패:** 로그에 `response head`가 찍힌다. 응답이 잘렸으면 `MAXTOK=8192`로 다시 실행한다.

## 끝나고 보고할 것

1. 모델별 `===== summary =====` 줄(`ice N/630`)
2. 모델별 `check_ice.py` 출력. `call_errors`와 `parse_failures` 개수다.
3. 기본값에서 바꾼 실행 설정. 예: `TP`, `EXTRA_VARGS`, `MAXTOK`
4. push한 커밋 해시
