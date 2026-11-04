# 서버 에이전트 작업 지시서: 힌트 조립 규칙 검증 자료 수집

작성: 2026-11-04. 이 파일 하나만 읽고 작업할 수 있게 썼다. 배경은 `docs/research/ccpp-hint-assembly.md` §6에 있다.

## 목적

LLM 레짐 진술을 조립하는 규칙(`HA`)은 이미 정해 두었다. 이 작업은 그 규칙을 판정할 **검증 자료**를 모으는 일이다. 모델이 쌍둥이 발전소에 대해 답한 내용을 수집만 한다. 분석은 하지 않는다.

## 할 일 (이 순서대로)

| 순서 | tag | Hugging Face 모델 | 수집 | 결과 파일과 완료 기준 |
|---|---|---|---|---|
| 1 | `gpt-oss-20b` | `openai/gpt-oss-20b` | 묻는 방식 실험 S1/S2만 (본 수집은 이미 있음) | `analysis/llm_statements/runs/strategy/gpt-oss-20b.jsonl` 220줄 |
| 2 | `gemma3-27b` | `google/gemma-3-27b-it` | 본 수집 + S1/S2 | `runs/gemma3-27b.jsonl` 1,596줄, `runs/strategy/gemma3-27b.jsonl` 220줄 |
| 3 | `llama33-70b` | `meta-llama/Llama-3.3-70B-Instruct` | 본 수집 + S1/S2 | `runs/llama33-70b.jsonl` 1,596줄, `runs/strategy/llama33-70b.jsonl` 220줄 |

모델마다 `runs/env/<tag>.txt`(환경 기록)도 자동으로 생겨야 한다.

## 실행 방법

1. 저장소 최상위에서 `git pull`을 실행한다.
2. `bash run_next.sh`를 실행한다. 이 스크립트가 다음을 모두 처리한다.
   - GPU 확인
   - 모델별 `vllm serve`
   - 수집
   - 결과 커밋과 `git push`
3. 진행은 `run_next.log`로 확인한다. 끝은 로그의 `===== 요약 =====`이다.
4. 스크립트 내부: `analysis/llm_statements/run_all.sh`를 `ONLY=<tag>`로 모델마다 부른다. 이미 끝난 부분은 건너뛴다. 중간에 끊기면 같은 명령을 다시 실행하면 이어서 한다.

## 바꾸면 안 되는 것

다음은 사전 등록된 실험 조건이다. 수정하지 않는다.

- 프롬프트(`prompts.py`, `elicit.py`)
- 표본 수(`STRAT_SAMPLES=10`), 온도, 응답 형식
- 모델 목록과 순서
- `window_summaries.json`
- `docs/` 아래 문서

수집 결과(`runs/`)는 손으로 고치거나 지우지 않는다. 일부만 실패하면 같은 명령을 다시 실행해 이어 받는다.

## 상황별 처리

- **MIG 조각:** 서버가 H200 MIG 35 GB 조각이면 스크립트가 조각 여러 장을 pipeline-parallel(ray)로 묶는다. 지난번 Qwen3-30B와 같은 방식이다(`runs/env/qwen3-30b-a3b-instruct.txt`). gemma-27b는 조각 2장 이상(약 64 GiB), Llama-70B는 5장 이상(약 150 GiB)이 필요하다. 조각을 더 할당받을 수 있으면 할당받고 실행한다. 지난번에 쓴 NCCL 패치(`patched-nccl`)가 필요하면 같은 방식으로 적용한다.
- **메모리 부족으로 건너뜀:**
  - 더 큰 GPU를 쓸 수 있으면 그 GPU에서 다시 실행한다.
  - 양자화 모델로 바꾸지 않는다. 모델 변경은 사전 등록 변경이므로 사람에게 먼저 묻는다.
- **Hugging Face 동의·로그인 오류** (gemma, Llama는 동의가 필요한 모델):
  - 토큰을 직접 입력하거나 만들지 않는다.
  - 사람에게 `huggingface-cli login`을 요청한다.
- **vLLM이 안 뜸:** `analysis/llm_statements/runs/logs/vllm-<tag>.log` 끝부분을 확인한다.
  - 서버 환경에 맞춘 실행 인자 수정은 해도 된다. `EXTRA_VARGS` 환경변수나 `TP`를 쓴다. 예: `--enforce-eager`, `--gpu-memory-utilization`
  - 바꾼 내용은 보고에 적는다. 환경 기록 파일에도 자동으로 남는다.
- **점검 실패(응답 잘림):** `MAXTOK=8192`로 다시 실행한다.
- **push 실패:** 커밋은 그대로 두고 사람에게 알린다. 강제 push(`--force`)는 하지 않는다.

## 추가로 해 두면 좋은 것 (선택)

`runs/env/qwen3-30b-a3b-instruct.txt`가 빠져 있다. 지난번 Qwen3-30B를 돌린 환경을 알면 같은 형식으로 남긴다. 형식은 `runs/env/gpt-oss-20b.txt` 참고. 모르면 건너뛴다.

## 끝나고 보고할 것

1. `run_next.log`의 `===== 요약 =====` 부분
2. 모델별 결과 줄 수와 오류 수. jsonl 각 줄의 `"error"` 필드가 `null`이 아닌 개수다.
3. 기본값에서 바꾼 실행 설정. 예: `TP`, `EXTRA_VARGS`, `MAXTOK`
4. push한 커밋 해시
