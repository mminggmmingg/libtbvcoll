#!/usr/bin/env python3
"""dbo_patch — vLLM 의 마이크로배치(DBO)를 dense 모델 + ROCm 에서 쓰기 위한 우회 2건.
컨테이너 안에서만 수정하며 --rm 이라 일회성이다. 결과 정합성은 출력해시로 검증한다.

우회 1: config 어서션 (vllm/config/vllm.py)
  "Microbatching currently only supports the deepep_* all2all backend"
  → ubatch 머신러리(UBatchWrapper/maybe_create_ubatch_slices) 자체는 all2all 전용이 아니다.

우회 2: SM control 플랫폼 어서션 (vllm/v1/worker/gpu_ubatch_wrapper.py)
  "SM control is currently only supported on CUDA"
  → 이 어서션은 **잘못 놓여 있다**. _create_sm_control_context 를 읽으면
    set_comm_sms / set_compute_sms 는 expert-parallel + DeepEP + DeepGEMM 이 아닐 때
    모두 `lambda sms: None`(무연산)이다. dense 모델 + EP 없음 + DeepGEMM 없음인 우리 설정에서는
    SM control 이 아무 일도 하지 않으므로 플랫폼 어서션만 막고 있는 셈이다.
    VLLM_DBO_COMM_SMS=0 과 함께 쓰면 의미상 안전하다.
"""
import sys

def patch(mod_path_expr, needle, repl, label):
    import importlib
    m = importlib.import_module(mod_path_expr)
    p = m.__file__
    s = open(p).read()
    if repl in s:
        print(f"[dbo] 이미 우회됨({label}): {p}"); return True
    if needle not in s:
        print(f"[dbo] 실패({label}) — 패턴 없음: {p}"); return False
    open(p, "w").write(s.replace(needle, repl, 1))
    print(f"[dbo] 우회 적용({label}): {p}")
    return True

# 우회 3: should_ubatch 가 DP 전용으로 배선돼 있다 (vllm/v1/worker/gpu_model_runner.py)
#   should_ubatch 는 False 로 초기화되고 `if data_parallel_size > 1` 안에서만 설정된다.
#   TP=2 / DP=1 이면 영원히 False → maybe_create_ubatch_slices 가 (None,None) → 겹침 도달 불가.
#   **이건 잘못 놓인 어서션 제거가 아니라 실제 제어 흐름 추가다.**
#   결과(2026-10-21 실측): 이 우회만으로는 안 된다. 다음 벽이 연달아 나온다 —
#     gpu_ubatch_wrapper.py:452  assert dp_metadata is not None       (이것도 잘못 놓임: 값은 안 읽는다)
#     forward_context.py:105     assert parallel_config.data_parallel_size > 1
#     forward_context.py:106     assert parallel_config.is_moe_model is not False   ← dense 거부
#   즉 마이크로배치는 DP + MoE 기구 **위에 구현**돼 있고 게이팅만의 문제가 아니다.
#   여기서부터는 dense TP 모델에 DP+MoE 상태를 위조하는 일이라 조용히 틀릴 위험을 한정할 수 없다.
#   → DBO_TP 는 기본 0. 겹침은 자체 엔진(step5)에서 구현하는 것이 맞다.
UB_NEEDLE = """        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:"""
UB_REPL = """        should_ubatch, num_tokens_across_dp = False, None
        if (self.vllm_config.parallel_config.data_parallel_size == 1
                and allow_microbatching):
            # [zbrdma] TP 전용 경로에서도 마이크로배치를 켠다. 상류는 DP 협조 경로에만
            # should_ubatch 를 세팅하므로 TP=2/DP=1 에서는 겹침이 도달 불가였다.
            from vllm.v1.worker.ubatch_utils import check_ubatch_thresholds as _cut
            should_ubatch = _cut(self.parallel_config, num_tokens, uniform_decode)
        if self.vllm_config.parallel_config.data_parallel_size > 1:"""

ok = patch("vllm.config.vllm",
           "assert a2a_backend in [",
           "assert True or a2a_backend in [",
           "config a2a")
ok &= patch("vllm.v1.worker.gpu_ubatch_wrapper",
            "assert current_platform.is_cuda(), (",
            "assert current_platform.is_cuda() or True, (",
            "SM control")
import os as _os
if _os.environ.get("DBO_TP", "0") == "1":   # 기본 끔: DPMetadata.make 가 dp_size>1 + is_moe_model 을 단언한다
    ok &= patch("vllm.v1.worker.gpu_model_runner", UB_NEEDLE, UB_REPL, "should_ubatch TP")
sys.exit(0 if ok else 1)
