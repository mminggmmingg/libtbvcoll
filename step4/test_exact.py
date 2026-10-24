#!/usr/bin/env python3
# test_exact — 자체 all-reduce 결과를 **정확한 정답**과 비교한다.
# 두 랭크의 입력을 시드로 결정하므로 각 랭크가 상대 입력을 로컬에서 재생성해 기준을 만든다.
# NCCL 을 기준으로 쓰지 않으므로 RCCL 패치/프로바이더 설정에 의존하지 않는다.
#   - dtype 덧셈 기준: (a.to(dt) + b.to(dt)) — 우리 커널과 같은 산술
#   - fp32 기준:       (a.float() + b.float()).to(dt) — 단일 라운딩
import os, torch, tbvcoll

rank = int(os.environ["RANK"])
torch.cuda.set_device(0)
tbvcoll.init(rank, os.environ.get("DEVS", "usb4_rdma0,usb4_rdma5"),
             os.environ.get("PEER_IP", ""), port=int(os.environ.get("PORT", "18997")),
             cap_bytes=4 << 20)

def gen(seed, numel):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.rand(numel, generator=g, device="cuda", dtype=torch.float32) - 0.5) * 10

if rank == 0:
    print(f"# {'dtype':>9} {'bytes':>9} {'dtype합일치':>12} {'fp32합일치':>11} {'최대절대차':>12}", flush=True)

for dt in (torch.float32, torch.bfloat16, torch.float16):
    for nb in (57344, 229376, 1048576):
        numel = nb // torch.tensor([], dtype=dt).element_size()
        f0, f1 = gen(1234, numel), gen(1235, numel)       # 양 랭크가 동일하게 재생성
        mine = (f0 if rank == 0 else f1).to(dt)
        out = torch.empty_like(mine)
        assert tbvcoll.all_reduce(mine.clone(), out), f"{dt} {nb} 미지원"
        torch.cuda.synchronize()

        ref_dt = f0.to(dt) + f1.to(dt)                     # 우리 커널과 동일 산술
        ref_32 = (f0.to(dt).float() + f1.to(dt).float()).to(dt)
        same_dt = bool(torch.equal(out, ref_dt))
        same_32 = bool(torch.equal(out, ref_32))
        mx = (out.float() - ref_32.float()).abs().max().item()
        if rank == 0:
            print(f"  {str(dt).replace('torch.',''):>9} {nb:>9d} {('예' if same_dt else '아니오'):>12} "
                  f"{('예' if same_32 else '아니오'):>11} {mx:>12.3e}", flush=True)
