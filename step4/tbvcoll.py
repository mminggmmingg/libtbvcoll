"""tbvcoll — libtbvcoll.so 의 ctypes 바인딩. vLLM TP all-reduce 대체용.

쓰는 법:
    import tbvcoll
    tbvcoll.init(rank, "usb4_rdma0,usb4_rdma5", peer_tb_ip, port=18995, cap_bytes=8<<20)
    tbvcoll.all_reduce(tensor)        # in-place SUM. cap 초과 시 False → 호출자가 NCCL 폴백
"""
import ctypes, os, torch

_DT = {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}
_lib = None
_cap = 0

def init(rank, devs, peer_ip, port=18995, cap_bytes=8 << 20,
         so=os.path.join(os.path.dirname(os.path.abspath(__file__)), "libtbvcoll.so")):
    global _lib, _cap
    if _lib is not None:
        return
    l = ctypes.CDLL(so)
    l.tbv_ar_init.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_size_t]
    l.tbv_ar_init.restype = ctypes.c_int
    l.tbv_ar_allreduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
    l.tbv_ar_allreduce.restype = ctypes.c_int
    l.tbv_ar_stats.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_ulonglong)]
    l.tbv_ar_stats5.argtypes = [ctypes.POINTER(ctypes.c_double)]*5 + [ctypes.POINTER(ctypes.c_ulonglong)]
    l.tbv_ar_stats4.argtypes = [ctypes.POINTER(ctypes.c_double)]*4 + [ctypes.POINTER(ctypes.c_ulonglong)]
    l.tbv_ar_async.restype = ctypes.c_int
    l.tbv_ar_err.restype = ctypes.c_int
    l.tbv_ar_cap.restype = ctypes.c_int
    rc = l.tbv_ar_init(rank, devs.encode(), (peer_ip or "").encode(), port, cap_bytes)
    if rc != 0:
        raise RuntimeError(f"tbv_ar_init 실패 rc={rc}")
    _lib, _cap = l, l.tbv_ar_cap()

def all_reduce(t: torch.Tensor, out: torch.Tensor = None) -> bool:
    """SUM. out 이 None 이면 in-place. 지원 못 하면 False(호출자가 NCCL 로 폴백)."""
    if _lib is None or not t.is_cuda or not t.is_contiguous():
        return False
    if out is not None and (not out.is_contiguous() or out.dtype is not t.dtype
                            or out.numel() != t.numel()):
        return False
    dt = _DT.get(t.dtype)
    if dt is None:
        return False
    nb = t.numel() * t.element_size()
    if nb == 0:
        return True
    if nb > _cap:
        return False
    st = torch.cuda.current_stream().cuda_stream
    rc = _lib.tbv_ar_allreduce(ctypes.c_void_p(t.data_ptr()),
                               ctypes.c_void_p(out.data_ptr()) if out is not None else None,
                               nb, dt, ctypes.c_void_p(st))
    if rc == -2:
        return False
    if rc != 0:
        raise RuntimeError(f"tbv_ar_allreduce 실패 rc={rc}")
    return True

def is_async():
    return bool(_lib.tbv_ar_async())

def err():
    return bool(_lib.tbv_ar_err())

def rails():
    """레일별 post 완료 / 피어 도착 평균을 stderr 로 덤프(리셋됨)."""
    _lib.tbv_ar_rails()

def hist():
    """크기별 전송 히스토그램을 stderr 로 덤프(리셋됨)."""
    _lib.tbv_ar_hist()

def raw_fn():
    """핫패스용 사전바인딩된 ctypes 함수. (in_ptr, out_ptr, nbytes, dtype, stream) → rc.
    래퍼 프레임을 한 개 줄이려고 노출한다(유휴 측정 기준 Python 측 작업은 전체 4.1us).
    호출자가 dtype/크기/연속성을 검사할 책임을 진다."""
    return _lib.tbv_ar_allreduce

def cap():
    return _cap

def stats5():
    """(pre, xfer, post, enq, drain, count) us.
    drain = 플래그 해제 후 내 송신 완료를 전부 수거하는 데 걸린 시간.
      ~0   → 내 송신이 피어 도착보다 먼저 끝났다 = 피어 대기(스큐)가 xfer 를 지배
      큼   → 내 전송 자체가 느리다"""
    a,b,c,d,e = (ctypes.c_double() for _ in range(5))
    n = ctypes.c_ulonglong()
    _lib.tbv_ar_stats5(*[ctypes.byref(x) for x in (a,b,c,d,e)], ctypes.byref(n))
    return a.value, b.value, c.value, d.value, e.value, n.value

def stats4():
    """(pre_us, xfer_us, post_us, enq_us, count).
    pre=SYNC 의 스트림 대기, xfer=전송(ASYNC 에서는 진행 스레드 기준),
    post=가산커널 런치, enq=호출자가 이 함수 안에서 쓴 총 시간."""
    a, b, c, d = (ctypes.c_double() for _ in range(4))
    n = ctypes.c_ulonglong()
    _lib.tbv_ar_stats4(ctypes.byref(a), ctypes.byref(b), ctypes.byref(c),
                       ctypes.byref(d), ctypes.byref(n))
    return a.value, b.value, c.value, d.value, n.value

def stats3():
    pre, xf, post, enq, n = stats4()
    return pre, xf, post, n

def stats():
    """(전송구간 평균 us, 호출수). 읽으면 리셋된다. D2D/스트림동기/가산커널은 제외."""
    a, c = ctypes.c_double(), ctypes.c_ulonglong()
    _lib.tbv_ar_stats(ctypes.byref(a), ctypes.byref(c))
    return a.value, c.value
