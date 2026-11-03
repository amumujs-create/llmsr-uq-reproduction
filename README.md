# LLM-SR-UQ reproduction

다른 컴퓨터에서 **LLM-SR-UQ의 원본 A1 방법을 실행하고, 동일 예측에 split/structural 불확실성 구간을 비교**하기 위한 독립 재현 저장소입니다.

- [논문](https://arxiv.org/abs/2610.42024) · [저자 공식 코드](https://github.com/MostafaCham/LLM-SR-UQ)
- 원본 코드 commit: `b58630470144a3da6944c8f45bdbef63af38d460`.
- `upstream/`의 25개 파일은 원본 그대로입니다. 해시로 검증합니다.
- **현재 검증 범위:** M4 16GB에서 Llama-3.1-8B-Instruct Q4_K_M로 개발 문제 1개, 2회 탐색의 실행 완료. **BF16 실행과 논문 전체 성능은 아직 재현 검증하지 않았습니다.**
- 이 저장소의 실행기는 A1을 대상으로 합니다. A0/A2/A3 전체 실험의 재현 패키지는 아닙니다.

## 1. 가져오기와 실험 환경 설치

Python **3.11**, Linux 또는 macOS를 사용하세요. 원본 timeout 코드가 POSIX signal을 사용하므로 Windows에서는 WSL2/Linux를 사용합니다.

```bash
git clone https://github.com/amumujs-create/llmsr-uq-reproduction.git
cd llmsr-uq-reproduction
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
python scripts/fetch_data.py
python scripts/verify_setup.py --offline --require-data
```

`requirements.txt`는 로컬 실행에서 검증한 직접 의존성 버전을 고정합니다. Linux/NVIDIA에서의 설치는 아직 실측 검증하지 않았습니다. 저자 저장소에는 전체 dependency lock이 없으므로 저자와 완전히 같은 환경이라고 주장하지 않습니다.

원본은 NumPy 1.x를 사용합니다. 최신 PyArrow를 임의로 설치하면 NumPy 2 요구사항과 충돌할 수 있어 `datasets==3.6.0`, `pyarrow==19.0.1`을 함께 고정했습니다.

## 2. LLM 서버 준비: 둘 중 하나

### A. 논문 설정에 맞춘 BF16 실행 — NVIDIA GPU 컴퓨터

GPU는 BF16을 지원해야 합니다. 8B 가중치만 약 16GB이므로 **VRAM 24GB 이상을 작업 계획의 출발점으로 권장**하지만, 실제 필요량은 런타임·KV cache 등에 따라 달라지며 최소 사양 보장은 아닙니다. [vLLM 설치 문서](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)를 해당 장비에 맞게 확인하세요.

**새 터미널에서 별도 서버 환경을 만듭니다.** vLLM과 분석용 NumPy 1.x 환경을 섞지 않습니다.

```bash
cd llmsr-uq-reproduction
python3.11 -m venv .venv-server
source .venv-server/bin/activate
python -m pip install --upgrade pip
python -m pip install vllm huggingface_hub
hf auth login
mkdir -p outputs
python -m pip freeze > outputs/server-environment.txt
set -o pipefail
bash scripts/serve_vllm.sh 2>&1 | tee outputs/server.log
```

[Meta 모델 페이지](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)의 접근 조건에 동의하고 접근 권한이 있는 Hugging Face 계정으로 로그인해야 할 수 있습니다. 토큰은 Git에 넣지 마세요.

서버 기본값은 `meta-llama/Llama-3.1-8B-Instruct`, `bfloat16`, context 4096, GPU utilization 0.90, `127.0.0.1:8000`입니다. 원본 계산 설정을 유지하고 localhost에만 연결합니다. 저자가 정확한 vLLM 버전 및 모델 revision을 고정하지 않아 여기에서도 확인되지 않은 버전을 '원본 버전'으로 지정하지 않습니다. 실제 서버 버전과 모델 revision을 실험 기록에 남기세요. revision을 고정할 경우 실행 스크립트 뒤에 `--revision <COMMIT> --tokenizer-revision <COMMIT>`을 전달할 수 있습니다.

분석용 터미널에서는:

```bash
source .venv/bin/activate
export LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct
export LLM_BASE_URL=http://127.0.0.1:8000/v1
export BACKEND_LABEL=llama31-8b-bf16-vllm
python scripts/verify_setup.py --require-data \
  --check-endpoint "$LLM_BASE_URL" --model "$LLM_MODEL"
```

이 endpoint 검사는 모델 목록만 확인하며 수식 생성 요청은 하지 않습니다.

### B. Ollama 4비트 — 구동 확인과 예비 실험

[Ollama](https://ollama.com/) 설치 후 별도 터미널에서:

```bash
cd llmsr-uq-reproduction
OLLAMA_HOST=127.0.0.1:11435 \
OLLAMA_MODELS="$PWD/models" \
OLLAMA_CONTEXT_LENGTH=4096 OLLAMA_NUM_PARALLEL=1 \
OLLAMA_MAX_LOADED_MODELS=1 ollama serve
```

분석용 터미널에서는:

```bash
source .venv/bin/activate
OLLAMA_HOST=127.0.0.1:11435 ollama pull llama3.1:8b-instruct-q4_K_M
export LLM_MODEL=llama3.1:8b-instruct-q4_K_M
export LLM_BASE_URL=http://127.0.0.1:11435/v1
export BACKEND_LABEL=llama31-8b-q4_K_M-ollama
python scripts/verify_setup.py --require-data \
  --check-endpoint "$LLM_BASE_URL" --model "$LLM_MODEL"
```

**Q4/Ollama 결과는 BF16/vLLM과 실행 조건이 다릅니다.** 양자화, chat template, 샘플링 기본값의 영향을 분리하지 않은 상태에서 논문 재현 성공/실패로 판정하지 않습니다.

## 3. 먼저 짧게 실행

LLM 서버를 켜 둔 상태에서 분석용 터미널에서 실행합니다.

```bash
bash scripts/run_a1.sh smoke
```

개발 문제 `lsr_synth_phys_osc_po10`에서 2회 × 후보 8개를 생성합니다. 템플릿 fallback은 꺼져 있습니다. 정상 실행과 데이터 흐름을 확인하는 용도이며 논문 성능 검증은 아닙니다.

## 4. 사전에 정한 3문제로 125회 실험

```bash
bash scripts/run_a1.sh dev3
```

결과를 보고 고른 문제가 아닌 고정 개발 문제 `po10`, `po12`, `po25`를 각각 125회 탐색합니다. 하나의 CPU fitting 작업을 순차 실행하며 문제당 제한은 3시간입니다. `--output-dir outputs/my-run`을 추가할 수 있지만 기존 폴더를 덮어쓰지 않습니다. 실패 또는 중단 문제를 다시 돌릴 때는 새 폴더와 `--only <INSTANCE_ID>`를 사용하고 원래 실패 기록을 보존하세요. 중간 탐색 상태에서의 자동 재개는 지원하지 않습니다.

```bash
# 실행 전 문제/분할만 확인하며 LLM은 호출하지 않음
python paired_runner.py --dev --list-problems

# 실제 125회 실행의 기본값 확인
python paired_runner.py --help
```

## 5. 같은 데이터에서 LLM 탐색만 반복

`--seed`를 바꿔 반복하면 원본 코드에서 데이터 분할과 피팅까지 바뀝니다. 아래 실행기는 **모든 반복의 seed를 동일하게 유지**하고 새 LLM 응답을 요청합니다. 모델 서버는 앞 절처럼 먼저 실행합니다.

```bash
# 짧은 반복 구동 검사: 같은 개발 문제에서 3회 × 2라운드
python repeat_search.py --runs 3 --iterations 2 \
  --model "$LLM_MODEL" --base-url "$LLM_BASE_URL" --backend-label "$BACKEND_LABEL"

# 본 예산의 반복 실험: 같은 개발 문제에서 5회 × 125라운드
python repeat_search.py --runs 5 --iterations 125 \
  --model "$LLM_MODEL" --base-url "$LLM_BASE_URL" --backend-label "$BACKEND_LABEL"
```

기본 문제는 `po10`입니다. `--only lsr_synth_phys_osc_po12` 등으로 사전에 정한 다른 문제도 실행할 수 있습니다. 모델/정밀도를 비교할 때는 동일한 문제·seed·예산을 유지하고 결과 폴더를 따로 만드세요.

각 결과 폴더의 `repeat_summary.json`에 반복별 성능, 성공/실패, 첫 요청의 동일성, 서로 다른 응답 수, 데이터 배열 해시를 저장합니다. 반복 간 점 예측의 표준편차와 한 실행 내부 수식들의 spread도 기록하지만, 이 두 값의 비교 자체는 보정된 예측 구간이나 과신의 증거가 아닙니다.

추가로 저장된 예측 평균·분산을 사용해 보정과 구간 계산을 다시 수행하고 기존 구간과 일치하는지 검사합니다. **이 replay는 LLM 호출이나 계수 재학습을 포함하지 않습니다.** 동일 수식을 다시 피팅했을 때까지 같다는 검증은 아닙니다.

원본 LLM 요청에는 서버 seed가 없습니다. 이 실험은 고정된 로컬 조건에서 새 확률적 탐색을 반복하며, 모든 난수 원인을 완벽히 통제했다는 뜻은 아닙니다. 과부하에 따른 피팅 시간 제한도 영향을 줄 수 있습니다. 반복 3회·2라운드의 결과로 125라운드 본 실험의 안정성이나 논문의 실패를 결론 내리지 마세요.

## 6. 최종 109문제로 확장

개발 단계가 끝나고 모델·프롬프트·평가 기준을 고정한 다음 실행합니다. 네 도메인의 `--heldout`을 합치면 109문제입니다. 전체 125회 × 109문제는 긴 실험이므로 먼저 3문제에서 소요 시간을 확인하세요.

```bash
for subset in lsr_synth_bio_pop_growth lsr_synth_chem_react lsr_synth_matsci lsr_synth_phys_osc; do
  python paired_runner.py \
    --subset "$subset" --heldout --iterations 125 \
    --model "$LLM_MODEL" --base-url "$LLM_BASE_URL" \
    --backend-label "$BACKEND_LABEL" --seed 0
done
```

## 평가 조건과 결과 읽기

- 각 문제에서 fitting 3200 / calibration 800 / ID test 500 / OOD test 500. 데이터 개수와 분할 RNG는 원본 loader와 같습니다.
- A1: 기계적 진단 피드백, critic 없음. 125 proposer 호출 × 후보 8개, restart 6, buffer 최대 24, hybrid weighting, 원본 기본 temperature 0.9.
- 목표 포함률은 **90%**이고 attainment 기준은 **88% 이상**입니다. 95% 실험이 아닙니다.
- 탐색은 한 번만 수행합니다. **동일 점 예측**에 split/structural 구간을 모두 계산해 탐색의 우연 차이를 제거합니다.
- 포함률뿐 아니라 훈련 목표값 표준편차로 나눈 구간 폭과 interval score도 확인합니다. 더 넓어진 구간의 포함률 상승만으로 우수하다고 판정하지 않습니다.
- 원본 CLI는 일부 폭 지표에 평가 split의 표준편차를 사용합니다. 이 실행기는 논문 Eq.11에 맞춘 train-SD 폭과 원본 CLI 방식의 폭을 둘 다 저장합니다.
- 두 구간의 NMSE는 같습니다. 이 비교는 점 예측 개선이 아닌 불확실성 구간 비교입니다.
- 원본 generator는 LLM 서버에 seed를 전달하지 않습니다. `--seed`는 분할/피팅을 제어하고 LLM 응답의 완전한 재현성을 보장하지 않습니다.
- Q4와 BF16, 서로 다른 탐색 예산, dev와 heldout 결과는 따로 보고합니다. 원논문의 집계 지표와 한 문제의 수치를 직접 비교하지 않습니다.

각 실행의 `outputs/paired_A1_*/`에는 다음이 저장됩니다.

- `run_metadata.json`: 코드/데이터 해시, 실행 인자, 환경 버전, backend label.
- `results.jsonl`: 문제별 성공/실패, 포함률, 폭, interval score, NMSE.
- 문제 폴더의 `llm_calls.jsonl`, `conversations.json`: LLM 요청/응답과 피드백 기록.
- `buffer.json`, `buffer_numeric.npz`: 남은 모든 수식의 계수, 공분산, 가중치.
- `predictions.npz`: 실제 목표값, 동일 예측 평균, 분산, 두 종류의 구간.

개별 문제 실패와 비유한값 개수를 반드시 같이 확인하세요. 상위 timeout, 프로세스 강제 종료, API 오류를 방법의 외삽 실패로 해석하지 않습니다. 성공 결과만 골라 전체 성공률로 보고하지 않습니다.

## 검증 및 공개 범위

```bash
python -m unittest discover -s tests -v
python scripts/verify_setup.py --offline --require-data
```

코드·설치 방법·출처·고정 해시만 공개합니다. 모델, 데이터, 환경, 토큰, 기존 로컬 실행 로그는 포함하지 않으며 `.gitignore`로 제외합니다. `outputs/`에는 실행 컴퓨터의 경로와 서버 정보가 들어갈 수 있으므로 결과 공유 전에 확인하세요.

원본 코드의 MIT 라이선스와 출처는 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [upstream/LICENSE](upstream/LICENSE)에 보존되어 있습니다.
