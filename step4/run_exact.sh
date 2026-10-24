#!/usr/bin/env bash
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"; IMG="${IMG:-tbv-vllm:latest}"
B_HOST="${B_HOST:-parks-zbook@10.99.0.2}"; B_DIR="${B_DIR:-/home/parks-zbook/step4}"
A_TB="${A_TB:-10.99.0.1}"; A_OOB="${A_OOB:-100.105.72.49}"
DK="docker run --rm -i --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband --network host --ipc=host --shm-size 8G --ulimit memlock=-1:-1 --security-opt seccomp=unconfined"
INNER_TO="${INNER_TO:-600}"   # 컨테이너 **안**에서 강제 종료. 호스트 timeout 은 docker 클라이언트만 죽이고
                               # 컨테이너는 살아남아 GPU 를 100% 로 물고 있는다(2026-10-21 실측).
E="-e PYTHONDONTWRITEBYTECODE=1 -e DEVS=${DEVS:-usb4_rdma0,usb4_rdma5} -e TBV_CHUNK=${TBV_CHUNK:-0} -e TBV_ASYNC=${TBV_ASYNC:-1}"
for f in libtbvcoll.so tbvcoll.py test_exact.py; do scp -q "$D/$f" "$B_HOST:$B_DIR/"; done
$DK -v "$D":/w -w /w $E -e RANK=0 "$IMG" timeout $INNER_TO python3 test_exact.py >/tmp/.ex_0 2>&1 &
sleep 3
ssh "$B_HOST" "$DK -v $B_DIR:/w -w /w $E -e RANK=1 -e PEER_IP=$A_TB $IMG timeout $INNER_TO python3 test_exact.py" >/tmp/.ex_1 2>&1
wait
grep -E '^#|^  [a-z]|Error|실패|Traceback|assert' /tmp/.ex_0 | head -20
