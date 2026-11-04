# 서버에서 LLM 진술 수집하기

설계: [docs/research/llm-statement-collection.md](../../docs/research/llm-statement-collection.md)

**CCPP 데이터 파일이 필요 없다.** 학습 창 요약은 `window_summaries.json`에 고정해 두었다. 필요한 것은 Python 3.9+, `numpy`, 그리고 OpenAI 호환 서버(vLLM 등)다.

## 1. 받기

```bash
git clone https://github.com/amumujs-create/llmsr-uq-reproduction.git
cd llmsr-uq-reproduction
pip install numpy
```

## 2. 모델 띄우기 (vLLM 예시)

모델 이름은 Hugging Face 모델 카드에서 확인한다. 아래 Qwen 이름은 자리표시다.

```bash
# 대형, 다른 계열, 추론형 (80 GB 이상 필요)
vllm serve openai/gpt-oss-120b --port 8000

# 중형/소형 Qwen (사고 모드 분리 파서 사용)
vllm serve <Qwen 약 30B 모델 이름> --port 8000 --reasoning-parser qwen3
```

MIG 조각 메모리는 `nvidia-smi -L`로 확인한다. 모델이 올라가지 않으면 더 작은 모델로 바꾸고, 바꾼 이름을 알려 준다.

## 3. 수집

모델마다 한 번씩 실행한다. 스크립트는 먼저 데이터 요약 없는 프롬프트 12개를 한 번씩 호출해 보는 점검을 하고, 통과하면 전체(76개 프롬프트 × 21회 = 1,596회)를 받는다. 중간에 끊기면 같은 명령을 다시 실행하면 이어서 받는다.

```bash
# gpt-oss-120b (기본 추론 설정)
bash analysis/llm_statements/run_server.sh openai/gpt-oss-120b gpt-oss-120b

# Qwen, 사고 모드 끔 (빠름)
bash analysis/llm_statements/run_server.sh <Qwen 모델 이름> qwen-30b-nothink '{"chat_template_kwargs": {"enable_thinking": false}}'

# Qwen, 사고 모드 켬 (느림, 시간이 되면)
MAXTOK=8192 bash analysis/llm_statements/run_server.sh <Qwen 모델 이름> qwen-30b-think
```

- 점검만 하려면 `SMOKE_ONLY=1 bash ...`
- 동시 요청 수 기본 16 (`WORKERS=32 bash ...`로 바꿀 수 있다)
- 응답 길이 상한 기본 4096 토큰 (`MAXTOK=8192 bash ...`)
- 서버 주소가 다르면 네 번째 인자로 준다: `... '{}' http://<host>:8000/v1`

점검에서 실패하면 화면에 응답 앞부분이 나온다. 흔한 원인:

- 추론형 모델이 길게 생각하다 상한에 걸림 → `MAXTOK`을 올리거나 사고 모드를 끈다
- 모델 이름이 서버의 `--served-model-name`과 다름

## 4. 결과 돌려주기

`analysis/llm_statements/runs/<tag>.jsonl` 파일들을 커밋해서 푸시하거나 파일로 보내 준다. 다음 정보도 함께 알려 준다.

- 정확한 모델 이름과 버전(Hugging Face 리비전 또는 해시)
- vLLM 버전, GPU(MIG 조각 크기)
- 실행 날짜

분석은 이쪽에서 한다: `python analysis/llm_statements/analyze.py <xlsx> runs/*.jsonl --out <prefix>`.
