# libtbvcoll

2랭크 TP all-reduce 라이브러리. HP ZBook Ultra G1a(Strix Halo, gfx1151) 두 대를 USB4/Thunderbolt 케이블 2개로
연결하고, `thunderbolt-ibverbs`(tbv) RDMA 위에서 vLLM 의 TP all-reduce 를 NCCL 대신 처리한다.

- `step4/libtbvcoll.cpp` — 라이브러리 본체(ASYNC 진행 스레드 + hipStreamWaitValue64, 레일별 스레드, dmabuf MR).
  빌드: `hipcc -O3 -fPIC -shared --offload-arch=gfx1151 -o libtbvcoll.so libtbvcoll.cpp -libverbs` (tbv-vllm 컨테이너 안)
- `step4/tbvcoll.py` — ctypes 래퍼. `test_exact.py`(비트 정확도), `test_stress.py`(비동기 연속 정확도), `test_tbvcoll.py`(지연).
- `step3b/` — vLLM 2노드 실행기. **`run_guarded.sh` 로 실행할 것**(커널 이상 감지 시 양 노드 컨테이너 즉시 정리,
  컨테이너 내부 timeout). `vllm_tp_bench.py` 는 GroupCoordinator.all_reduce 를 libtbvcoll 로 가로챈다.
- `reports/` — 설계·측정 기록(10_tp_custom_allreduce.md 가 핵심).

주요 환경변수: `TBV_ASYNC`(1), `TBV_NSLOT`(4 권장), `TBV_CAP`(=메시지 크기), `TBV_CPU`, `TBV_UNCACHED`(기본 1: 송신 바운스
GPU 비캐시 — 일반 hipMalloc 은 NHI DMA 가 이전 내용을 읽어 출력이 오염된다), `TBV_TAG`(진단).

tbv 드라이버 쪽 수정(크레딧 교착 등)은 `thunderbolt-ibverbs` 저장소 `zbook-gpu-direct-fixes` 브랜치.
