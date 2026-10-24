#!/usr/bin/env bash
# 2노드 vLLM TP 실행(A 에서 한 방). 인용 중첩을 피하려고 노드별 실행은 node_launch.sh 로 분리.
#   MODE=gdr|hostbounce|socket ./run_vllm_tp.sh [tp] [batch] [out_tokens]
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
TP="${1:-2}"; BATCH="${2:-8}"; OUT="${3:-128}"; MODE="${MODE:-gdr}"
B_HOST="${B_HOST:-parks-zbook@10.99.0.2}"; B_DIR="${B_DIR:-/home/parks-zbook/step3b}"
A_OOB="${A_OOB:-100.105.72.49}"
A_MODEL="${A_MODEL:-/data/parksThink/hf-models/Qwen2.5-7B-Instruct}"
B_MODEL="${B_MODEL:-/home/parks-zbook/models/Qwen2.5-7B-Instruct}"
# 이전 실행의 잔여 컨테이너 정리(양 노드). 멈춘 QP 를 붙잡은 채 남으면 다음 실행·재연결이 위험하다.
for _c in $(docker ps -q --filter label=tbvbench=1); do docker kill "$_c" >/dev/null; done
ssh "$B_HOST" 'for c in $(docker ps -q --filter label=tbvbench=1); do docker kill $c >/dev/null; done'
ssh "$B_HOST" "mkdir -p $B_DIR" && scp -q "$D/vllm_tp_bench.py" "$D/node_launch.sh" "$D/tbvcoll.py" "$D/libtbvcoll.so" "$D/dbo_patch.py" "$B_HOST:$B_DIR/" && ssh "$B_HOST" "chmod +x $B_DIR/node_launch.sh"
echo "[vllm] mode=$MODE tbvcoll=${TBVCOLL:-0} tp=$TP batch=$BATCH out=$OUT (A=rank0, B=rank1)"
TBVCOLL="${TBVCOLL:-0}" TIMENCCL="${TIMENCCL:-0}" TBV_ASYNC="${TBV_ASYNC:-1}" TBV_GPUSIG="${TBV_GPUSIG:-1}" TBV_NSLOT="${TBV_NSLOT:-8}" TBV_CAP="${TBV_CAP:-4194304}" TBV_CPU="${TBV_CPU:-}" TBV_TAG="${TBV_TAG:-0}" TBV_UNCACHED="${TBV_UNCACHED:-1}" TBVEAGER="${TBVEAGER:-0}" INNER_TO="${INNER_TO:-600}" DBO="${DBO:-0}" DBO_TP="${DBO_TP:-0}" DBOTH="${DBOTH:-}" ROUNDS="${ROUNDS:-}" "$D/node_launch.sh" 0 2 "$A_OOB" "$TP" "$BATCH" "$OUT" "$MODE" "$A_MODEL" "$D/../rccl-src/deploy" "$D/../multirail/ucq" "$D" >/tmp/.vllm_r0 2>&1 &
A=$!
sleep 5
ssh "$B_HOST" "NCCL_DEBUG=${NCCL_DEBUG:-WARN} TBVCOLL=${TBVCOLL:-0} TIMENCCL=${TIMENCCL:-0} TBV_ASYNC=${TBV_ASYNC:-1} TBV_GPUSIG=${TBV_GPUSIG:-1} TBV_NSLOT=${TBV_NSLOT:-8} TBV_CAP=${TBV_CAP:-4194304} TBV_CPU=${TBV_CPU:-} TBV_TAG=${TBV_TAG:-0} TBV_UNCACHED=${TBV_UNCACHED:-1} TBVEAGER=${TBVEAGER:-0} INNER_TO=${INNER_TO:-600} DBO=${DBO:-0} DBO_TP=${DBO_TP:-0} DBOTH=${DBOTH:-} ROUNDS=${ROUNDS:-} TBV_PEER=${A_TB:-10.99.0.1} $B_DIR/node_launch.sh 1 2 $A_OOB $TP $BATCH $OUT $MODE $B_MODEL /home/parks-zbook/rccl-alt /home/parks-zbook/ucq $B_DIR" >/tmp/.vllm_r1 2>&1 &
B=$!
wait $A; RA=$?; wait $B; RB=$?
echo "=== rank0 (rc=$RA) ==="; grep -E '^#|round[0-9]|tbvcoll|Traceback|Error' /tmp/.vllm_r0 || tail -15 /tmp/.vllm_r0
echo "=== rank1 (rc=$RB) 마지막 ==="; tail -4 /tmp/.vllm_r1
