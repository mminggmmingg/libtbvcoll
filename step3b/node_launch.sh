#!/usr/bin/env bash
# vLLM TP 노드 실행기. 양노드에서 동일 스크립트를 node_rank 만 바꿔 호출한다.
#   ./node_launch.sh <node_rank> <nnodes> <master_addr> <tp> <batch> <out> <mode> <model> <rccl_dir> <prov_dir> <bench_dir>
set -uo pipefail
NR="$1"; NN="$2"; MA="$3"; TP="$4"; BATCH="$5"; OUT="$6"; MODE="$7"
MODEL="$8"; RCCLD="$9"; PROVD="${10}"; BENCHD="${11}"
TBV="${TBVCOLL:-0}"   # 1 이면 TP all-reduce 를 libtbvcoll 로 대체
# 컨테이너 **안**에서 시간 제한. 호스트 timeout 은 docker 클라이언트만 죽이고 컨테이너(특히 상대 노드)는
# 살아남아 멈춘 RDMA QP 를 붙잡는다 → 2026-10-24 케이블 재연결 때 thunderbolt 링 목록 손상·종료 불가로 이어짐.
INNER_TO="${INNER_TO:-600}"
IMG="${IMG:-tbv-vllm:latest}"
case "$MODE" in
  gdr)        NET="-e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=usb4_rdma0,usb4_rdma5 -e NCCL_IB_GID_INDEX=1 -e NCCL_IB_MERGE_NICS=0 -e NCCL_MAX_NCHANNELS=8 -e NCCL_MIN_NCHANNELS=2 -e NCCL_CROSS_NIC=1 -e NCCL_NET_GDR_LEVEL=SYS -e NCCL_DMABUF_ENABLE=1 -e NCCL_IB_PCI_RELAXED_ORDERING=0 -e NCCL_GDR_FLUSH_DISABLE=1" ;;
  hostbounce) NET="-e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=usb4_rdma0,usb4_rdma5 -e NCCL_IB_GID_INDEX=1 -e NCCL_IB_MERGE_NICS=0 -e NCCL_MAX_NCHANNELS=2 -e NCCL_MIN_NCHANNELS=2 -e NCCL_CROSS_NIC=1 -e NCCL_NET_GDR_LEVEL=0" ;;
  socket)     NET="-e NCCL_IB_DISABLE=1" ;;                                  # OOB iface(tailscale0=wifi) 위 TCP — 참고용
  sockettb)   NET="-e NCCL_IB_DISABLE=1" ;;   # TCP over TB(공정 비교)
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
NCCL_IFACE=tailscale0
[ "$MODE" = sockettb ] && NCCL_IFACE=thunderbolt0
TBVARG=""; [ "$TBV" = 1 ] && TBVARG="--tbvcoll"
[ "${TIMENCCL:-0}" = 1 ] && TBVARG="--time-nccl"
[ "${DBO:-0}" = 1 ] && TBVARG="$TBVARG --dbo"
[ -n "${DBOTH:-}" ] && TBVARG="$TBVARG --dbo-thresh $DBOTH"
[ -n "${ROUNDS:-}" ] && TBVARG="$TBVARG --rounds-only $ROUNDS"
[ "${TBVEAGER:-0}" = 1 ] && TBVARG="$TBVARG --tbv-eager"
DBOPATCH=""; [ "${DBO:-0}" = 1 ] && DBOPATCH="python3 /bench/dbo_patch.py || exit 9; "   # 한쪽만 패치되면 반대쪽이 어서션에 걸려 교착한다
IN="$DBOPATCH"''"cp /rcclalt/librccl.so.1.0 /opt/python/lib/python3.13/site-packages/_rocm_sdk_libraries_gfx1151/lib/librccl.so.1; cp /provalt/libusb4_rdma-rdmav34.so /usr/lib/x86_64-linux-gnu/libibverbs/libusb4_rdma-rdmav34.so; export LD_PRELOAD=/rcclalt/librccl_missing_syms.so; exec timeout -k 15 $INNER_TO torchrun --nnodes=$NN --nproc-per-node=1 --node-rank=$NR --master-addr=$MA --master-port=29600 /bench/vllm_tp_bench.py --model /model --tp $TP --batch $BATCH --out-tokens $OUT $TBVARG"
exec docker run --rm -i --label tbvbench=1 --device=/dev/kfd --device=/dev/dri --device=/dev/infiniband \
  --network host --ipc=host --shm-size 32G --ulimit memlock=-1:-1 --security-opt seccomp=unconfined \
  -v "$MODEL":/model:ro -v "$RCCLD":/rcclalt:ro -v "$PROVD":/provalt:ro -v "$BENCHD":/bench:ro \
  -e MASTER_ADDR="$MA" -e MASTER_PORT=29600 -e NCCL_SOCKET_IFNAME="$NCCL_IFACE" \
  -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" -e VLLM_LOGGING_LEVEL=WARNING -e HIP_VISIBLE_DEVICES=0 \
  -e TBV_ASYNC="${TBV_ASYNC:-1}" -e TBV_GPUSIG="${TBV_GPUSIG:-1}" -e TBV_NSLOT="${TBV_NSLOT:-8}" -e DBO="${DBO:-0}" -e VLLM_DBO_COMM_SMS=0 -e DBO_TP="${DBO_TP:-0}" -e TBV_PEER="${TBV_PEER:-10.99.0.1}" -e TBV_DEVS="${TBV_DEVS:-usb4_rdma0,usb4_rdma5}" -e TBV_PORT="${TBV_PORT:-18996}" -e TBV_CAP="${TBV_CAP:-4194304}" -e TBV_CHUNK="${TBV_CHUNK:-0}" -e TBV_CPU="${TBV_CPU:-}" -e TBV_TAG="${TBV_TAG:-0}" -e TBV_UNCACHED="${TBV_UNCACHED:-1}" -e PYTHONPATH=/bench \
  -e TORCHDYNAMO_DISABLE=1 -e TORCHINDUCTOR_MAX_AUTOTUNE=0 -e TORCHINDUCTOR_BENCHMARK_KERNEL=0 \
  $NET "$IMG" bash -c "$IN"
