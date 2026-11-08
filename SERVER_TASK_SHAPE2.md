# 서버 에이전트 작업 지시서: 경계 너머 모양 수집 2단계

작성: 2026-11-08 (1단계 지시서를 고쳐 씀). 이 파일 하나만 읽고 작업할 수 있게 썼다. 배경과 사전 등록은 `docs/research/ccpp-shape-elicitation-2.md`에 있다.

## 목적

1단계(`run_shape.sh`)와 같은 방식으로, 새 세계 두 개와 대조 세계 세 개에 "경계 너머에서 출력이 어떻게 반응하는가"를 묻는다. 수집만 하고 분석은 하지 않는다.

## 할 일

| 순서 | tag | Hugging Face 모델 | 필요 메모리 (대략) | 결과 파일과 완료 기준 |
|---|---|---|---|---|
| 1 | `gpt-oss-120b` | `openai/gpt-oss-120b` | 약 70 GiB | `analysis/llm_statements/runs/shape2/gpt-oss-120b.jsonl` 210줄 |
| 2 | `qwen3-30b-a3b-instruct` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 약 64 GiB | `runs/shape2/qwen3-30b-a3b-instruct.jsonl` 210줄 |
| 3 | `gpt-oss-20b` | `openai/gpt-oss-20b` | 약 20 GiB | `runs/shape2/gpt-oss-20b.jsonl` 210줄 |
| 4 | `gemma3-27b` | `google/gemma-3-27b-it` | 약 64 GiB | `runs/shape2/gemma3-27b.jsonl` 210줄 |
| 5 | `llama33-70b` | `meta-llama/Llama-3.3-70B-Instruct` | 약 150 GiB | `runs/shape2/llama33-70b.jsonl` 210줄 |

모델마다 `runs/env/<tag>-shape2.txt`(환경 기록)도 생긴다. 기존 `runs/env/<tag>.txt`, `<tag>-ice.txt`, `<tag>-shape.txt`는 덮어쓰지 않는다.

## 실행 방법 (명령 하나)

저장소 최상위에서 다음 두 줄을 실행한다.
```bash
git pull
bash run_shape2.sh
```
- 백그라운드에서 모델 다섯 개를 순서대로 돌리고, 끝나면 결과를 commit하고 push한다. 진행은 `tail -f run_shape2.log`로 본다. 끝은 로그의 `===== 요약 =====`이다.
- 1단계(`run_shape.sh`)와 같은 방식이다. MIG 조각이면 pipeline-parallel(ray)과 NCCL 설정을 알아서 주고, 모델별 인자(gemma `--language-model-only`, Llama `--max-model-len 8192 --max-num-seqs 4 --gpu-memory-utilization 0.85`)를 넣는다.
- 점검(작업마다 1회, 10회)은 호출 오류가 있거나 파싱 실패가 절반을 넘을 때만 실패한다. 실패하면 한 번 `MAXTOK=8192`(Llama는 6144)로 다시 시도한다.
- 끊기면 `bash run_shape2.sh`를 다시 실행한다. 이미 받은 부분은 건너뛴다.
- 지난번처럼 서버에서 따로 필요한 설정(`patched-nccl` 등)이 있으면 그것을 적용한 셸에서 실행한다.

모델 하나만 직접 돌리려면:
```bash
SHAPE2=1 ONLY=<tag> bash analysis/llm_statements/run_all.sh
```

## 바꾸면 안 되는 것

사전 등록된 조건이므로 수정하지 않는다.
- 프롬프트(`prompts.py`, `collect.py`)
- 표본 수(21회 = 온도 0.7로 20회 + 온도 0으로 1회)
- `docs/` 아래 문서

`runs/` 아래 결과는 손으로 고치거나 지우지 않는다.

## 상황별 처리

- 양자화 모델로 바꾸지 않는다.
- 토큰을 직접 입력하거나 만들지 않는다.
- 강제 push(`--force`)를 하지 않는다.
- **점검 실패:** 로그에 `response head`가 찍힌다. 응답이 잘렸으면 `MAXTOK=8192`로 다시 실행한다.
- **위치 `null` 답:** 파싱 실패로 기록되지만 정상이다. 손대지 않는다.

## 끝나고 보고할 것

1. 모델별 `===== 요약 =====` 줄(`shape2 N/210`)
2. 모델별 `check_shape.py` 출력: `call_errors`, `parse_failures`, `null_location`
3. 기본값에서 바꾼 실행 설정. 예: `TP`, `EXTRA_VARGS`, `MAXTOK`
4. push한 커밋 해시
