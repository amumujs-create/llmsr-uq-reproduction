#!/usr/bin/env bash
# GPU 서버에서 실행: 힌트 조립 규칙의 검증 자료 수집 (docs/research/ccpp-hint-assembly.md §6)
#   1) gpt-oss-20b  S1/S2 (220회)      2) gemma-3-27b  전체 (1,596 + 220회)      3) Llama-3.3-70B  전체
# 실행:  bash run_next.sh     (알아서 백그라운드로 돌고, 끝나면 결과를 GitHub에 올린다)
# 진행:  tail -f run_next.log
set -uo pipefail
cd "$(dirname "$0")"
if [ -z "${RUN_NEXT_INNER:-}" ]; then
  RUN_NEXT_INNER=1 nohup bash "$0" > run_next.log 2>&1 < /dev/null &
  echo "백그라운드에서 시작했습니다 (PID $!). 진행 확인: tail -f $(pwd)/run_next.log"
  exit 0
fi

echo "== $(date) 시작, host=$(hostname)"
git pull --ff-only || { echo "!! git pull 실패. 로컬 변경을 확인하세요."; exit 1; }

# GPU 확인: 장수와 한 장당 메모리(GiB)
mapfile -t MEMS < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)
N=${#MEMS[@]}
[ "$N" -eq 0 ] && { echo "!! GPU가 보이지 않습니다 (nvidia-smi 실패). GPU 서버에서 실행하세요."; exit 1; }
PER=$(( ${MEMS[0]} / 1024 ))
TP=1; while [ $((TP * 2)) -le "$N" ]; do TP=$((TP * 2)); done   # 2의 거듭제곱으로 맞춘다
TOTAL=$(( PER * TP ))
echo "== GPU ${N}장 × ${PER} GiB, tensor-parallel ${TP} → 사용 가능 ${TOTAL} GiB"
# MIG 조각(예: H200-2-35C)은 여러 장을 묶을 수 없다. 지난번 gpt-oss-20b와 같은 설정을 쓴다.
EXTRA=""
if nvidia-smi -L | grep -q MIG; then TP=1; TOTAL=$PER; EXTRA="--enforce-eager"; echo "== MIG 조각 감지: TP=1, --enforce-eager"; fi

# 모델별 필요한 GPU 메모리 대략치 (GiB, 가중치 + 여유)
declare -A NEED=( [gpt-oss-20b]=20 [gemma3-27b]=64 [llama33-70b]=150 )
summary=()
for TAG in gpt-oss-20b gemma3-27b llama33-70b; do
  if [ "$TOTAL" -lt "${NEED[$TAG]}" ]; then
    echo "!! $TAG 건너뜀: 약 ${NEED[$TAG]} GiB 필요, 사용 가능 ${TOTAL} GiB"
    summary+=("$TAG: 메모리 부족으로 건너뜀 (필요 ~${NEED[$TAG]} GiB / 가능 ${TOTAL} GiB)")
    continue
  fi
  echo "== $TAG 실행"
  ONLY="$TAG" TP="$TP" EXTRA_VARGS="$EXTRA" bash analysis/llm_statements/run_all.sh 2>&1 | tail -40
  summary+=("$TAG: 끝 (자세한 결과는 위 로그)")
done

echo; echo "===== 요약 ====="; printf '%s\n' "${summary[@]}"

git add analysis/llm_statements/runs
if git diff --cached --quiet; then
  echo "== 새 결과 없음. 올릴 것이 없습니다."
else
  git commit -q -m "Validation runs for hint assembly ($(hostname))" && git push -q \
    && echo "== GitHub에 올렸습니다." || echo "!! push 실패. 서버에서 git push를 직접 해 주세요."
fi
echo "== $(date) 끝"
