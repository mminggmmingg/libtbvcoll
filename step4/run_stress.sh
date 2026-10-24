#!/usr/bin/env bash
# 양노드 test_stress 실행 (A=rank0). 환경변수는 그대로 컨테이너에 전달.
#   TBV_ASYNC TBV_GPUSIG TBV_NSLOT TBV_CPU DEVS BYTES CAP ITERS GAP DT
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"; IMG="${IMG:-tbv-vllm:latest}"
B_HOST="${B_HOST:-parks-zbook@10.99.0.2}"; B_DIR="${B_DIR:-/home/parks-zbook/step4}"
A_TB="${A_TB:-10.99.0.1}"
DK="docker run --rm -i --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband --network host --ipc=host --shm-size 8G --ulimit memlock=-1:-1 --security-opt seccomp=unconfined"
INNER_TO="${INNER_TO:-300}"
E="-e PYTHONDONTWRITEBYTECODE=1"
for v in TBV_TAG TBV_ASYNC TBV_GPUSIG TBV_NSLOT TBV_CPU TBV_CHUNK DEVS BYTES CAP ITERS GAP DT; do
  [ -n "${!v:-}" ] && E="$E -e $v=${!v}"
done
for f in libtbvcoll.so tbvcoll.py test_stress.py; do scp -q "$D/$f" "$B_HOST:$B_DIR/"; done
$DK -v "$D":/w -w /w $E -e RANK=0 "$IMG" timeout $INNER_TO python3 test_stress.py >/tmp/.st_0 2>&1 &
sleep 3
ssh "$B_HOST" "$DK -v $B_DIR:/w -w /w $E -e RANK=1 -e PEER_IP=$A_TB $IMG timeout $INNER_TO python3 test_stress.py" >/tmp/.st_1 2>&1
wait
grep -hE '^#|Error|Traceback|assert' /tmp/.st_0 /tmp/.st_1
