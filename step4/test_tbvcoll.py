#!/usr/bin/env python3
# test_tbvcoll — 라이브러리 정합성 + vLLM 이 실제 지불할 end-to-end 비용 측정.
# 전체 지연(D2D 바운스 + 스트림동기 + 전송 + 가산커널)을 재야 NCCL 과 공정하게 비교된다.
import os, sys, time, torch, tbvcoll

rank = int(os.environ["RANK"]); peer = os.environ.get("PEER_IP", "")
devs = os.environ.get("DEVS", "usb4_rdma0,usb4_rdma5")
iters = int(os.environ.get("ITERS", "300"))
sizes = [int(x) for x in os.environ.get("SIZES", "4096,57344,229376,1048576").split(",")]
dtype = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[os.environ.get("DT", "f32")]

gap_us = float(os.environ.get("GAP_US", "0"))
torch.cuda.set_device(0)
# GAP 용 더미 연산: GPU 를 대략 gap_us 만큼 돌린다(레이어 연산 모사)
gap_mode = os.environ.get("GAP_MODE", "compute")
_gm = None
if gap_us > 0:
    if gap_mode == "mem":
        # 메모리 집약: 큰 텐서를 읽는다(가중치 스트리밍 모사)
        _gw = torch.randn(64 * 1024 * 1024 // 2, device="cuda", dtype=torch.float16)  # 64MiB
        _gacc = torch.zeros(1, device="cuda", dtype=torch.float32)
        def _one_op():
            global _gm
            _gm = _gw.sum()
    else:
        n = 2048
        _ga = torch.randn(n, n, device="cuda", dtype=torch.float16)
        _gb = torch.randn(n, n, device="cuda", dtype=torch.float16)
        def _one_op():
            global _gm
            _gm = _ga @ _gb
    torch.cuda.synchronize(); _t0 = time.perf_counter()
    for _ in range(20): _one_op()
    torch.cuda.synchronize()
    _one = (time.perf_counter() - _t0) / 20 * 1e6
    _grep = max(1, int(round(gap_us / _one)))
    if rank == 0:
        print(f"# GAP {gap_us:.0f}us = {gap_mode} {_one:.1f}us x {_grep}", flush=True)

def _gap():
    if gap_us > 0:
        for _ in range(_grep): _one_op()
tbvcoll.init(rank, devs, peer, port=int(os.environ.get("PORT", "18995")), cap_bytes=max(sizes))
if rank == 0:
    print(f"# dtype={dtype} devs={devs} iters={iters}", flush=True)
    print(f"# 모드 {'ASYNC' if tbvcoll.is_async() else 'SYNC'}", flush=True)
    print(f"# {'bytes':>9} {'전체_us':>9} {'호출비용':>9} {'전송_us':>8} {'post_us':>8} {'algbw':>8} {'verify':>7}", flush=True)

for nb in sizes:
    numel = nb // torch.tensor([], dtype=dtype).element_size()
    t = torch.full((numel,), float(rank + 1), dtype=dtype, device="cuda")
    for _ in range(30):
        _gap()
        assert tbvcoll.all_reduce(t), f"{nb} 미지원"
        t.fill_(float(rank + 1))
    torch.cuda.synchronize()
    tbvcoll.stats4()
    t0 = time.perf_counter()
    for _ in range(iters):
        _gap()
        tbvcoll.all_reduce(t)
    torch.cuda.synchronize()
    tot = (time.perf_counter() - t0) / iters
    pre, xfer, post, enq, _ = tbvcoll.stats4()
    tbvcoll.rails()
    # 정합성: 깨끗한 1회
    c = torch.full((numel,), float(rank + 1), dtype=dtype, device="cuda")
    tbvcoll.all_reduce(c); torch.cuda.synchronize()
    ok = bool(torch.allclose(c, torch.full_like(c, 3.0), rtol=1e-2, atol=1e-2))
    if rank == 0:
        print(f"  {nb:>9d} {tot*1e6:>9.1f} {enq:>9.1f} {xfer:>8.1f} {post:>8.1f} "
              f"{nb*8/tot/1e9:>8.2f} {'OK' if ok else 'BAD':>7}", flush=True)
