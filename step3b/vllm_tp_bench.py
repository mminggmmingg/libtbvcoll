#!/usr/bin/env python3
# vLLM TP=2 토큰/초 벤치 — 2노드, external_launcher(torchrun) 백엔드.
# vLLM 공식 예제(/app/vllm/examples/offline_inference/torchrun_example.py) 패턴을 따름:
#   모든 랭크가 동일한 프롬프트/시드로 LLM 을 만들고 generate 하며, rank0 만 출력한다(SPMD).
# env: MASTER_ADDR/PORT WORLD_SIZE RANK LOCAL_RANK (torchrun 이 설정) + NCCL_* (런처가 설정)
import os, time, argparse, statistics
import torch.distributed as dist
from vllm import LLM, SamplingParams

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/model")
ap.add_argument("--tp", type=int, default=2)
ap.add_argument("--max-model-len", type=int, default=4096)
ap.add_argument("--out-tokens", type=int, default=128)
ap.add_argument("--batch", type=int, default=8, help="동시 요청 수(처리량 측정)")
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--gpu-mem-util", type=float, default=0.80)
# gfx1151 에서 torch.compile(Inductor) autotuning 이 "float floor division by zero" 로 깨진다
# (2026-10-21 실측) → 기본 eager. --compile 로 켤 수 있게 둔다.
ap.add_argument("--compile", action="store_true", help="torch.compile/graph 사용(gfx1151 에선 실패 기왕력)")
# 자체 TP all-reduce(libtbvcoll). reports/09: 56KiB 에서 NCCL 306us vs 자체 165us.
ap.add_argument("--tbvcoll", action="store_true", help="TP all-reduce 를 libtbvcoll 로 대체")
# 같은 워크로드 안에서 NCCL 회당 시간을 재야 공정한 비교가 된다(피어 스큐가 양쪽 모두에 포함됨)
ap.add_argument("--time-nccl", action="store_true", help="NCCL all-reduce 를 그대로 쓰되 회당 시간을 계측")
# TP 는 레이어 의존성 때문에 한 마이크로배치 안에서 통신/연산이 겹치지 않는다.
# vLLM 의 DBO(dual batch overlap)는 독립 마이크로배치 2개를 교차시켜 그걸 겹친다.
# decode 는 토큰 >= dbo_decode_token_threshold(기본 32)일 때만 발동한다.
ap.add_argument("--tbv-eager", action="store_true",
                help="vLLM 보다 먼저 tbvcoll 초기화(바운스 버퍼를 VRAM 선점 전에 확보)")
ap.add_argument("--dbo", action="store_true", help="vLLM enable_dbo 사용")
ap.add_argument("--dbo-thresh", type=int, default=0, help="dbo_decode_token_threshold 덮어쓰기(0=기본)")
ap.add_argument("--rounds-only", type=int, default=0, help="라운드 수 덮어쓰기")
ap.add_argument("--tbv-devs", default=os.environ.get("TBV_DEVS", "usb4_rdma0,usb4_rdma5"))
ap.add_argument("--tbv-peer", default=os.environ.get("TBV_PEER", "10.99.0.1"), help="rank0 의 TB IP")
ap.add_argument("--tbv-cap", type=int, default=int(os.environ.get("TBV_CAP", 4 << 20)),
                help="이 크기 초과 메시지는 NCCL 로 폴백")
a = ap.parse_args()

rank = int(os.environ.get("RANK", "0"))

# ── 자체 all-reduce 패치 ───────────────────────────────────────────────────────
# vLLM 의 GroupCoordinator.all_reduce 를 가로채 TP 그룹만 libtbvcoll 로 보낸다.
# 계약상 out-of-place 여야 하므로 출력 텐서를 따로 준다. 지원 못 하는 경우
# (dtype/크기/비연속)는 False 를 받아 NCCL 원본으로 폴백한다.
_tbv_stats = {"n": 0, "fallback": 0, "py": 0.0}
if a.time_nccl:
    import torch
    from vllm.distributed import parallel_state as ps
    _orig_ar = ps.GroupCoordinator.all_reduce
    _nz = {"t": 0.0, "n": 0, "tp": None, "hist": {}}

    def _timed_all_reduce(self, input_):
        if self.world_size == 1:
            return input_
        if _nz["tp"] is None:
            try:
                _nz["tp"] = ps.get_tp_group().unique_name
            except Exception:
                _nz["tp"] = False
        if not (_nz["tp"] and self.unique_name == _nz["tp"]):
            return _orig_ar(self, input_)
        nb = input_.numel() * input_.element_size()
        t = time.perf_counter()
        r = _orig_ar(self, input_)
        # NCCL 은 스트림에 넣기만 하므로 완료까지 기다려 우리와 같은 기준으로 만든다
        torch.cuda.current_stream().synchronize()
        d = time.perf_counter() - t
        _nz["t"] += d; _nz["n"] += 1
        b = nb.bit_length() - 1
        h = _nz["hist"].setdefault(b, [0, 0.0]); h[0] += 1; h[1] += d
        return r

    ps.GroupCoordinator.all_reduce = _timed_all_reduce
    if rank == 0:
        print("# NCCL 계측 모드(스트림 동기 포함 — 자체 구현과 같은 기준)", flush=True)

if a.tbvcoll:
    import torch
    import tbvcoll
    from vllm.distributed import parallel_state as ps
    _orig_ar = ps.GroupCoordinator.all_reduce
    _state = {"init": False, "tp": None}

    # 핫패스에 쓰는 것들을 미리 묶는다(전역/속성 조회를 매 호출 피한다)
    _fast = {}

    def _patched_all_reduce(self, input_):
        if self.world_size == 1:
            return input_
        if _state["tp"] is None:
            try:
                _state["tp"] = ps.get_tp_group().unique_name
            except Exception:
                _state["tp"] = False
        if _state["tp"] and self.unique_name == _state["tp"]:
            if not _state["init"]:
                # 지연 초기화: SPMD 라 두 랭크가 같은 지점에서 들어온다.
                tbvcoll.init(rank, a.tbv_devs, a.tbv_peer if rank == 1 else "",
                             port=int(os.environ.get("TBV_PORT", "18996")),
                             cap_bytes=a.tbv_cap)
                _state["init"] = True
                _fast["fn"] = tbvcoll.raw_fn()
                _fast["cap"] = tbvcoll.cap()
                _fast["dt"] = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
                _fast["empty"] = torch.empty_like
                _fast["cs"] = torch.cuda.current_stream
            _t = time.perf_counter()
            dt = _fast["dt"].get(input_.dtype)
            nb = input_.numel() * input_.element_size()
            if dt is not None and nb <= _fast["cap"] and input_.is_contiguous():
                out = _fast["empty"](input_)
                # 매 호출 현재 스트림을 읽는다(1.96us). 캐시하면 vLLM 의 스트림 전환에 깨진다.
                rc = _fast["fn"](input_.data_ptr(), out.data_ptr(), nb, dt,
                                 _fast["cs"]().cuda_stream)
                _tbv_stats["py"] += time.perf_counter() - _t
                if rc == 0:
                    _tbv_stats["n"] += 1
                    return out
                if rc != -2:
                    raise RuntimeError(f"tbv_ar_allreduce rc={rc}")
            else:
                _tbv_stats["py"] += time.perf_counter() - _t
            _tbv_stats["fallback"] += 1
        return _orig_ar(self, input_)

    ps.GroupCoordinator.all_reduce = _patched_all_reduce
    if rank == 0:
        print(f"# tbvcoll 패치 적용: devs={a.tbv_devs} cap={a.tbv_cap>>10}KiB eager={a.tbv_eager}", flush=True)
    if a.tbv_eager:
        # LLM() 전에 초기화. 두 랭크가 같은 지점이므로 핸드셰이크가 맞는다.
        torch.cuda.set_device(0)
        tbvcoll.init(rank, a.tbv_devs, a.tbv_peer if rank == 1 else "",
                     port=int(os.environ.get("TBV_PORT", "18996")), cap_bytes=a.tbv_cap)
        _state["init"] = True
        _fast["fn"] = tbvcoll.raw_fn()
        _fast["cap"] = tbvcoll.cap()
        _fast["dt"] = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
        _fast["empty"] = torch.empty_like
        _fast["cs"] = torch.cuda.current_stream
# ──────────────────────────────────────────────────────────────────────────────

t0 = time.perf_counter()
_extra = {}
if a.dbo:
    _extra["enable_dbo"] = True
    if a.dbo_thresh:
        _extra["dbo_decode_token_threshold"] = a.dbo_thresh
llm = LLM(model=a.model, tensor_parallel_size=a.tp,
          distributed_executor_backend="external_launcher",
          max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_mem_util,
          seed=1, enforce_eager=not a.compile, **_extra)
load_s = time.perf_counter() - t0
if rank == 0:
    print(f"# 모델 로드 {load_s:.1f}s  model={a.model} tp={a.tp} batch={a.batch} out={a.out_tokens} eager={not a.compile} dbo={a.dbo}", flush=True)

# 길이를 고정해 출력 토큰 수를 결정적으로 만든다(ignore_eos)
sp = SamplingParams(temperature=0.0, max_tokens=a.out_tokens, ignore_eos=True)
prompt = ("Explain the architecture of a modern GPU memory hierarchy, "
          "including cache levels, bandwidth considerations, and how it affects "
          "tensor parallel inference. Be detailed and technical. ")
prompts = [prompt + f"(variant {i})" for i in range(a.batch)]

# 워밍업(컴파일/그래프 캡처 제외)
llm.generate(prompts[:2], SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True))
dist.barrier()

results = []
for r in range(a.rounds_only or a.rounds):
    dist.barrier()
    t = time.perf_counter()
    outs = llm.generate(prompts, sp)
    dist.barrier()
    dt = time.perf_counter() - t
    gen = sum(len(o.outputs[0].token_ids) for o in outs)
    if rank == 0:
        # 라운드마다 해시를 낸다. 1라운드만 보면 간헐적 오염(2026-10-24 ASYNC 1회 관측)을 놓친다.
        import hashlib
        ids = b"".join(bytes(str(list(o.outputs[0].token_ids)), "utf8") for o in outs)
        h = hashlib.sha256(ids).hexdigest()[:16]
        if r == 0:
            print(f"# 출력해시 {h}  샘플: {outs[0].outputs[0].text[:90]!r}", flush=True)
        else:
            print(f"# 출력해시 r{r+1} {h}", flush=True)
    pin = sum(len(o.prompt_token_ids) for o in outs)
    results.append((dt, gen, pin))
    if rank == 0:
        print(f"  round{r+1}: {dt:6.2f}s  생성 {gen:5d} tok  → **{gen/dt:7.1f} tok/s**  "
              f"(프롬프트 {pin} tok, 요청 {a.batch}개)", flush=True)

if rank == 0:
    tps = [g/d for d, g, _ in results]
    if a.time_nccl and _nz["n"]:
        print(f"# NCCL: 호출 {_nz['n']}회  평균 {_nz['t']/_nz['n']*1e6:.1f}us", flush=True)
        for b in sorted(_nz["hist"]):
            c, tt = _nz["hist"][b]
            print(f"#   {1<<b:>9d} B~ : {c:>7d} 회  평균 {tt/c*1e6:>7.1f} us", flush=True)
    if a.tbvcoll:
        import tbvcoll as _tc
        pre, xf, post, enq, drain, n = _tc.stats5()
        _tc.hist(); _tc.rails()
        print(f"# tbvcoll: 호출 {_tbv_stats['n']}회  폴백 {_tbv_stats['fallback']}회  "
              f"모드 {'ASYNC' if _tc.is_async() else 'SYNC'}  "
              f"전송 {xf:.1f}us  호출비용 {enq:.1f}us  "
              f"드레인 {drain:.1f}us  "
              f"파이썬래퍼 {(_tbv_stats['py']/max(1,_tbv_stats['n'])*1e6 - enq):.1f}us  "
              f"(pre {pre:.1f} / post {post:.1f})", flush=True)
    print(f"# 결과: 중위 {statistics.median(tps):.1f} tok/s  최고 {max(tps):.1f}  "
          f"(배치 {a.batch}, 출력 {a.out_tokens} tok/요청)")
    print(f"# 요청당 평균: {statistics.median([d for d,_,_ in results])/a.batch:.2f}s")
