#!/usr/bin/env bash
# allreduce2 2노드 실행. A=rank0, B=rank1. 컨테이너에서 hipcc 빌드.
#   ./run_ar2.sh [bytes...]
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
IMG="${IMG:-tbv-vllm:latest}"; DEV="${DEV:-usb4_rdma5}"; ITERS="${ITERS:-200}"
B_HOST="${B_HOST:-parks-zbook@10.99.0.2}"; B_DIR="${B_DIR:-/home/parks-zbook/step4}"
A_TB="${A_TB:-10.99.0.1}"
DK="docker run --rm -i --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband --network host --ipc=host --shm-size 8G --ulimit memlock=-1:-1 --security-opt seccomp=unconfined"
INNER_TO="${INNER_TO:-600}"   # 컨테이너 **안**에서 강제 종료. 호스트 timeout 은 docker 클라이언트만 죽이고
                               # 컨테이너는 살아남아 GPU 를 100% 로 물고 있는다(2026-10-21 실측).
ssh "$B_HOST" "mkdir -p $B_DIR" && scp -q "$D/allreduce_mr.cpp" "$B_HOST:$B_DIR/"
echo "[build] A"; $DK -v "$D":/w "$IMG" bash -c 'hipcc -O3 -o /w/.armr /w/allreduce_mr.cpp -libverbs 2>&1 | grep -E "\berror\b"' && true
echo "[build] B"; ssh "$B_HOST" "$DK -v $B_DIR:/w $IMG bash -c 'hipcc -O3 -o /w/.armr /w/allreduce_mr.cpp -libverbs 2>&1 | grep -E \"\berror\b\"'" && true
[ -x "$D/.armr" ] || { echo "[fail] A 빌드 실패"; exit 1; }
for B in "$@"; do
  $DK -v "$D":/w -e HOSTBUF="${HOSTBUF:-0}" "$IMG" timeout $INNER_TO /w/.armr 0 "$DEV" "$B" "$ITERS" >/tmp/.armr_0 2>&1 &
  for i in $(seq 1 40); do grep -q 'waiting' /tmp/.armr_0 2>/dev/null && break; sleep 0.3; done
  ssh "$B_HOST" "$DK -v $B_DIR:/w -e HOSTBUF=${HOSTBUF:-0} $IMG timeout $INNER_TO /w/.armr 1 $DEV $B $ITERS $A_TB" >/tmp/.armr_1 2>&1
  wait
  grep -E '스트림순서|호스트동기|verify|fail|timeout|errno' /tmp/.armr_0 | sed 's/^/ /'
done
