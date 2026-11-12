# 서버에서 LLM 진술 수집하기 (한 번에 전부)

설계: [docs/research/llm-statement-collection.md](../../docs/research/llm-statement-collection.md)

**이 파일의 명령 하나로 이 연구에 필요한 LLM 실행이 모두 끝난다.** 새로 돌릴 것이 생기면 이 파일과 `models.tsv`만 고친다.

## 0. 필요한 것

- GPU 서버(80 GB급 4장 기준), Python 3.9+, `vllm`, `numpy`, `curl`
- CCPP 데이터 파일은 **필요 없다** (학습 창 요약은 `window_summaries.json`에 고정)
- Gemma와 Llama는 Hugging Face에서 사용 동의가 필요한 모델이다. 모델 페이지에서 동의한 뒤 서버에서 `huggingface-cli login`을 한 번 해 둔다. 안 되어 있으면 그 모델만 실패하고 다음 모델로 넘어간다.

## 1. 받기 / 최신으로 맞추기

```bash
git clone https://github.com/amumujs-create/llmsr-uq-reproduction.git   # 처음 한 번
cd llmsr-uq-reproduction
git pull
```

## 2. 실행 (이것만 하면 된다)

```bash
nohup bash analysis/llm_statements/run_all.sh > run_all.log 2>&1 &
tail -f run_all.log
```

`run_all.sh`는 `models.tsv`의 모델을 위에서부터 하나씩 처리한다. 모델마다 하는 일:

1. `vllm serve`로 모델을 띄운다 (`--tensor-parallel-size 4 --max-model-len 16384`)
2. 점검 12회 → 통과하면 본 수집 1,596회 (`runs/<tag>.jsonl`)
3. 묻는 방식 실험 S1/S2 220회 (`runs/strategy/<tag>.jsonl`)
4. 실행 환경 기록 (`runs/env/<tag>.txt`: 모델, HF 리비전, vLLM 버전, GPU, 날짜)
5. 서버를 내리고 다음 모델로

**이미 끝난 부분은 건너뛴다.** 중간에 끊기면 같은 명령을 다시 실행하면 이어서 한다. `gpt-oss-120b`는 본 수집이 끝났으므로 묻는 방식 실험만 돈다.

## 3. 돌릴 모델 (`models.tsv`, 위가 우선)

| tag | 모델 | 왜 필요한가 |
|---|---|---|
| `gpt-oss-120b` | `openai/gpt-oss-120b` | 완료 |
| `qwen3-30b-a3b-instruct` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | 완료 (환경 기록 `runs/env/`만 없음) |
| `gpt-oss-20b` | `openai/gpt-oss-20b` | 본 수집 완료. **S1/S2(220회)가 남음** |
| `gemma3-27b` | `google/gemma-3-27b-it` | **조립 규칙 검증 자료** (동의 필요) |
| `llama33-70b` | `meta-llama/Llama-3.3-70B-Instruct` | **조립 규칙 검증 자료** (동의 필요) |

예상 시간은 모델당 대략 30분–2시간이다 (다운로드 포함).

남은 세 가지(`gpt-oss-20b` S1/S2, `gemma3-27b`, `llama33-70b`)는 [조립 규칙 사전 등록](../../docs/research/ccpp-hint-assembly.md)의 검증 자료다. 같은 명령을 다시 실행하면 끝난 부분은 건너뛰고 이것만 돈다.

## 4. 자주 쓰는 옵션

- 한 모델만: `ONLY=gpt-oss-20b bash analysis/llm_statements/run_all.sh`
- GPU 수가 다르면: `TP=2 bash ...` (GPU 2장에 나눠 올림)
- 동시 요청 수: `WORKERS=32 bash ...` (기본 16)
- 응답 길이 상한: `MAXTOK=8192 bash ...` (기본 4096. 추론형 모델이 점검에서 잘리면 올린다)
- 모델을 빼려면 `models.tsv`에서 그 줄 앞에 `#`을 붙인다

## 5. 문제가 생기면

- 끝에 나오는 `===== summary =====`에 모델별 성공/실패가 나온다
- vLLM이 안 뜨면 `analysis/llm_statements/runs/logs/vllm-<tag>.log` 마지막 부분을 보면 된다. 흔한 원인:
  - 메모리 부족 → `TP`를 늘리거나 그 모델을 뺀다
  - 동의 안 된 모델 → 1번의 Hugging Face 로그인
- 점검 실패(응답이 잘림) → `MAXTOK=8192`로 다시 실행

## 6. 결과 돌려주기

```bash
git add analysis/llm_statements/runs
git commit -m "LLM runs"
git push
```

`runs/env/*.txt`에 환경이 자동 기록되므로 따로 알려 줄 것은 없다. 실패한 모델이 있으면 summary 부분만 알려 준다.

분석은 이쪽에서 한다.

## 7. 창 밖 구간 확증 수집 (2026-11-11 추가)

설계: [docs/research/ccpp-twin-outside-confirmation.md](../../docs/research/ccpp-twin-outside-confirmation.md). 새 모델 네 개에 기존 수집 전부와 새 세계 단계를 돌린다.

```bash
git pull
```

```bash
MODELS=models_new.tsv NEWWORLD=1 nohup bash analysis/llm_statements/run_all.sh > run_new.log 2>&1 &
```

```bash
tail -f run_new.log
```

- 모델마다: 본 수집 1,596회 + S1/S2 220회 + 새 세계 128회. 끝난 부분은 건너뛰므로 끊기면 같은 명령을 다시 실행한다.
- 모델 목록은 `models_new.tsv`다(Mistral Small 3.2 24B, Phi-4, Granite 3.3 8B, DeepSeek-R1-Distill-Qwen-32B). 네 개 모두 Hugging Face 동의가 필요 없다(2026-11-11 확인).
- **모델을 바꾸고 싶으면 수집 전에 알려 준다.** 등록 문서의 표를 먼저 고쳐야 한다. 수집을 시작한 뒤에는 바꾸지 않는다.
- `deepseek-r1d-32b`는 추론형이라 답이 잘릴 수 있다. 점검에서 실패하면 그 모델만 `ONLY=deepseek-r1d-32b MAXTOK=8192 MODELS=models_new.tsv NEWWORLD=1 bash analysis/llm_statements/run_all.sh`로 다시 돌린다.
- Mistral은 `models_new.tsv`에 적힌 vLLM 옵션(`--tokenizer_mode mistral --config_format mistral --load_format mistral`, 모델 카드 권장)으로 뜬다. 안 뜨면 `runs/logs/vllm-mistral-small32-24b.log` 끝부분을 알려 준다.
- 뜨지 않는 모델이 있어도 나머지는 계속 돈다. 두 개 이상 끝나면 판정할 수 있다.

결과 돌려주기는 §6과 같다(`git add analysis/llm_statements/runs && git commit -m "LLM runs" && git push`).

선택: 기존 다섯 모델도 새 세계 단계만 추가로 돌리려면 `NEWWORLD=1 bash analysis/llm_statements/run_all.sh`(모델당 128회, 판정에는 쓰지 않고 보고만 한다).

## 8. 깨끗한 창 단계 (2026-11-12 추가)

설계: [docs/research/ccpp-twin-clean-collection.md](../../docs/research/ccpp-twin-clean-collection.md). 제한 값과 설정 온도가 다른 발전소 6개를 묻는다. 모델당 192회, `runs/cleanworld/<tag>.jsonl`.

새 모델 네 개(§7 수집과 한 번에):

```bash
MODELS=models_new.tsv NEWWORLD=1 CLEANWORLD=1 nohup bash analysis/llm_statements/run_all.sh > run_new.log 2>&1 &
```

기존 다섯 모델(새 세계 128회 + 깨끗한 창 192회만 추가로 돈다):

```bash
NEWWORLD=1 CLEANWORLD=1 nohup bash analysis/llm_statements/run_all.sh > run_old.log 2>&1 &
```

- 두 명령은 같은 GPU를 쓰므로 하나가 끝난 뒤 다음을 실행한다.
- §7을 이미 돌렸다면 같은 명령에 `CLEANWORLD=1`만 붙여 다시 실행한다. 끝난 부분은 건너뛴다.
- `CLEANWORLD=0`(기본)이면 스크립트는 이전과 똑같이 동작한다.
- 결과 돌려주기는 §6과 같다.

## 9. 근거 인용 탐색 단계 (2026-11-12 추가)

설계: [docs/research/ccpp-twin-explore-cite.md](../../docs/research/ccpp-twin-explore-cite.md). 탐색이다. 기존 다섯 모델만 돌린다. 모델당 301회, `runs/cleancite/<tag>.jsonl`.

```bash
CLEANCITE=1 nohup bash analysis/llm_statements/run_all.sh > run_cite.log 2>&1 &
```

- 이미 끝난 단계는 건너뛰고 301회만 추가로 돈다. 224회 버전을 이미 돌렸다면 같은 명령으로 77회만 더 돈다.
- `MODELS=models_new.tsv`와 함께 쓰면 스크립트가 멈춘다. 새 모델은 이 탐색에 쓰지 않는다.
- `CLEANCITE=0`(기본)이면 스크립트는 이전과 똑같이 동작한다.
- 결과 돌려주기는 §6과 같다. 돌려받은 뒤 분석: `PYTHONSAFEPATH=1 python3 analysis/ccpp_twin/explore/cite_check.py`
