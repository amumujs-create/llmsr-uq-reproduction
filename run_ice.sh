#!/usr/bin/env bash
# GPU 서버에서 실행: 결빙 방지 세계 진술 수집 (SERVER_TASK_ICE.md, docs/research/ccpp-anti-icing-world.md §15)
#   모델 5개 × 630회. 순서: gpt-oss-120b, qwen3-30b-a3b-instruct, gpt-oss-20b, gemma3-27b, llama33-70b
# 실행:  bash run_ice.sh     (알아서 백그라운드로 돌고, 끝나면 결과를 GitHub에 올린다)
# 진행:  tail -f run_ice.log
# 끊기면 같은 명령을 다시 실행한다. 이미 받은 부분은 건너뛴다.
# DRY_RUN=1 이면 vLLM과 수집을 실제로 돌리지 않고 모델별 명령만 찍는다(점검용).
set -uo pipefail
cd "$(dirname "$0")"
LOGF="${RUN_ICE_LOG:-run_ice.log}"
if [ -z "${RUN_ICE_INNER:-}" ]; then
  RUN_ICE_INNER=1 nohup bash "$0" > "$LOGF" 2>&1 < /dev/null &
  echo "백그라운드에서 시작했습니다 (PID $!). 진행 확인: tail -f $(pwd)/$LOGF"
  exit 0
fi

echo "== $(date) 시작, host=$(hostname)"
git pull --ff-only || { echo "!! git pull 실패. 로컬 변경을 확인하세요."; exit 1; }

# GPU 확인: 장수와 한 장당 메모리(GiB)
MEMS=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
N=$(printf '%s\n' "$MEMS" | grep -c '[0-9]')
[ "$N" -eq 0 ] && { echo "!! GPU가 보이지 않습니다 (nvidia-smi 실패). GPU 서버에서 실행하세요."; exit 1; }
PER=$(( $(printf '%s\n' "$MEMS" | head -1) / 1024 ))
TP=1; while [ $((TP * 2)) -le "$N" ]; do TP=$((TP * 2)); done   # 2의 거듭제곱으로 맞춘다
TOTAL=$(( PER * TP ))
EXTRA="${EXTRA_VARGS:-}"
# MIG 조각은 tensor-parallel로 묶을 수 없다. 지난번 수집과 같이 pipeline-parallel(ray)로 묶고 NCCL 설정을 준다
# (runs/env/gemma3-27b.txt, runs/env/llama33-70b.txt).
if nvidia-smi -L 2>/dev/null | grep -q MIG || nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qE -- '-[0-9]+-[0-9]+C'; then
  TP=1; TOTAL=$(( PER * N ))
  EXTRA="--pipeline-parallel-size $N --distributed-executor-backend ray --enforce-eager --gpu-memory-utilization 0.90 $EXTRA"
  export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=1 VLLM_USE_FLASHINFER_SAMPLER=0
  echo "== MIG 조각 ${N}장 감지: pipeline-parallel ${N}, 사용 가능 약 ${TOTAL} GiB"
else
  echo "== GPU ${N}장 × ${PER} GiB, tensor-parallel ${TP} → 사용 가능 ${TOTAL} GiB"
fi

# 모델별 필요한 GPU 메모리 대략치 (GiB)와, 지난번 수집에서 성공한 모델별 추가 인자
need() { case "$1" in gpt-oss-120b) echo 70;; qwen3-30b-a3b-instruct|gemma3-27b) echo 64;; gpt-oss-20b) echo 20;;
  llama33-70b) echo 150;; esac; }
model_args() { case "$1" in gemma3-27b) echo "--language-model-only";;
  llama33-70b) echo "--max-model-len 8192 --max-num-seqs 4 --gpu-memory-utilization 0.85";; esac; }
summary=()
for TAG in gpt-oss-120b qwen3-30b-a3b-instruct gpt-oss-20b gemma3-27b llama33-70b; do
  if [ "$TOTAL" -lt "$(need "$TAG")" ]; then
    echo "!! $TAG 건너뜀: 약 $(need "$TAG") GiB 필요, 사용 가능 ${TOTAL} GiB"
    summary+=("$TAG: 메모리 부족으로 건너뜀 (필요 ~$(need "$TAG") GiB / 가능 ${TOTAL} GiB)")
    continue
  fi
  ARGS="$EXTRA $(model_args "$TAG")"
  echo "== $TAG 실행 (TP=$TP, 추가 인자: $ARGS)"
  if [ -n "${DRY_RUN:-}" ]; then
    echo "DRY_RUN: ICE=1 ONLY=$TAG TP=$TP EXTRA_VARGS=\"$ARGS\" bash analysis/llm_statements/run_all.sh"
    summary+=("$TAG: DRY_RUN"); continue
  fi
  out=$(ICE=1 ONLY="$TAG" TP="$TP" EXTRA_VARGS="$ARGS" bash analysis/llm_statements/run_all.sh 2>&1)
  echo "$out" | tail -40
  # 응답이 잘려 점검에 실패하면 한 번만 MAXTOK=8192로 다시 시도한다 (SERVER_TASK_ICE.md "상황별 처리")
  if echo "$out" | grep -q "anti-icing smoke test FAILED"; then
    # llama33-70b는 --max-model-len 8192라서 프롬프트(약 1–2천 토큰)와 답이 함께 들어가도록 6144로 한다.
    RT=8192; [ "$TAG" = llama33-70b ] && RT=6144
    echo "== $TAG 점검 실패 → MAXTOK=$RT로 다시 시도"
    ICE=1 ONLY="$TAG" TP="$TP" MAXTOK="$RT" EXTRA_VARGS="$ARGS" bash analysis/llm_statements/run_all.sh 2>&1 | tail -40
  fi
  n=$( [ -f "analysis/llm_statements/runs/ice/$TAG.jsonl" ] && wc -l < "analysis/llm_statements/runs/ice/$TAG.jsonl" | tr -d ' ' || echo 0 )
  summary+=("$TAG: ice $n/630")
done

echo; echo "===== 요약 ====="; printf '%s\n' "${summary[@]}"
[ -n "${DRY_RUN:-}" ] && { echo "== DRY_RUN: 커밋하지 않습니다."; exit 0; }

git add analysis/llm_statements/runs/ice analysis/llm_statements/runs/env
if git diff --cached --quiet; then
  echo "== 새 결과 없음. 올릴 것이 없습니다."
else
  git commit -q -m "Anti-icing stage runs ($(hostname))" && git push -q \
    && echo "== GitHub에 올렸습니다. 커밋: $(git rev-parse --short HEAD)" || echo "!! push 실패. 서버에서 git push를 직접 해 주세요."
fi
echo "== $(date) 끝"
