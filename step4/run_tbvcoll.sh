#!/usr/bin/env bash
# run_tbvcoll — libtbvcoll 정합성 + end-to-end 비용 측정. A=rank0, B=rank1.
set -uo pipefail
D="$(cd "$(dirname "$0")" && pwd)"
IMG="${IMG:-tbv-vllm:latest}"
B_HOST="${B_HOST:-parks-zbook@10.99.0.2}"; B_DIR="${B_DIR:-/home/parks-zbook/step4}"
A_TB="${A_TB:-10.99.0.1}"
DK="docker run --rm -i --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband --network host --ipc=host --shm-size 8G --ulimit memlock=-1:-1 --security-opt seccomp=unconfined"
INNER_TO="${INNER_TO:-600}"   # 컨테이너 **안**에서 강제 종료. 호스트 timeout 은 docker 클라이언트만 죽이고
                               # 컨테이너는 살아남아 GPU 를 100% 로 물고 있는다(2026-10-21 실측).
ENVS="-e PYTHONDONTWRITEBYTECODE=1 -e DEVS=${DEVS:-usb4_rdma0,usb4_rdma5} -e ITERS=${ITERS:-300} -e SIZES=${SIZES:-4096,57344,229376,1048576} -e DT=${DT:-f32} -e PORT=${PORT:-18995} -e TBV_ASYNC=${TBV_ASYNC:-1} -e TBV_NSLOT=${TBV_NSLOT:-8} -e TBV_CHUNK=${TBV_CHUNK:-0} -e TBV_GPUSIG=${TBV_GPUSIG:-1} -e TBV_RAIL1MAX=${TBV_RAIL1MAX:-98304} -e GAP_US=${GAP_US:-0} -e GAP_MODE=${GAP_MODE:-compute} -e TBV_CPU=${TBV_CPU:-}"
ssh "$B_HOST" "mkdir -p $B_DIR" || exit 1
for f in libtbvcoll.so tbvcoll.py test_tbvcoll.py; do scp -q "$D/$f" "$B_HOST:$B_DIR/" || exit 1; done
$DK -v "$D":/w -w /w $ENVS -e RANK=0 "$IMG" timeout $INNER_TO python3 test_tbvcoll.py >/tmp/.tc_0 2>&1 &
sleep 3
ssh "$B_HOST" "$DK -v $B_DIR:/w -w /w $ENVS -e RANK=1 -e PEER_IP=$A_TB $IMG timeout $INNER_TO python3 test_tbvcoll.py" >/tmp/.tc_1 2>&1
wait
grep -E '^#| [0-9]+ +[0-9]+\.|tbvcoll|Error|실패|Traceback' /tmp/.tc_0 | head -30
echo "--- rank1 ---"; grep -E 'tbvcoll|Error|실패|Traceback' /tmp/.tc_1 | head -8
