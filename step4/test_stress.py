#!/usr/bin/env python3
# test_stress — vLLM 처럼 동기화 없이 연속 all-reduce 를 걸고, 끝난 뒤 전부 정답과 비교한다.
# test_exact 는 호출 하나씩 동기화하므로 슬롯 재사용/완료 순서 경쟁을 잡지 못한다.
#   매 반복 입력 = 시드(반복번호, 랭크) 로 생성 → 각 랭크가 상대 입력을 로컬에서 재생성해 정답을 만든다.
import os, torch, tbvcoll

rank = int(os.environ["RANK"])
nb = int(os.environ.get("BYTES", "57344"))
iters = int(os.environ.get("ITERS", "2000"))
gap = os.environ.get("GAP", "none")          # none | mem  (mem: 가중치 스트리밍 모사)
dt = {"f32": torch.float32, "bf16": torch.bfloat16, "f16": torch.float16}[os.environ.get("DT", "bf16")]
torch.cuda.set_device(0)
tbvcoll.init(rank, os.environ.get("DEVS", "usb4_rdma0,usb4_rdma5"), os.environ.get("PEER_IP", ""),
             port=int(os.environ.get("PORT", "18998")), cap_bytes=int(os.environ.get("CAP", str(nb))))
numel = nb // torch.tensor([], dtype=dt).element_size()

def gen(it, r):
    g = torch.Generator(device="cuda").manual_seed(it * 2 + r)
    return ((torch.rand(numel, generator=g, device="cuda") - 0.5) * 10).to(dt)

_gw = torch.randn(256 * 1024 * 1024 // 2, device="cuda", dtype=torch.float16) if gap == "mem" else None
outs, ins = [], []
for it in range(iters):
    x = gen(it, rank)
    out = torch.empty_like(x)
    assert tbvcoll.all_reduce(x, out), "미지원"
    outs.append(out); ins.append(x)          # 입력도 붙잡아 둔다(할당기 재사용으로 인한 가짜 불일치 방지)
    if _gw is not None:
        _gw.sum()
torch.cuda.synchronize()

bad = 0; first = None
for it, out in enumerate(outs):
    ref = gen(it, 0) + gen(it, 1)
    if not torch.equal(out, ref):
        bad += 1
        if first is None:
            first = it
            d = (out.float() - ref.float()).abs()
            nz = int((d > 0).sum())
print(f"# rank{rank} dtype={dt} bytes={nb} iters={iters} gap={gap} async={tbvcoll.is_async()} "
      f"불일치 {bad}/{iters}" + (f"  첫불일치 it={first} 틀린원소 {nz}/{numel}" if first is not None else ""), flush=True)
