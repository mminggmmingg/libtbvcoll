# 자체 추론 엔진 초석 (zbengine) — dense 모델

측정일 2026-10-21 · A/B = HP ZBook Ultra G1a (gfx1151, 128GB UMA) · 모델 Qwen2.5-7B-Instruct (bf16)
코드: `experiments/tbv-adopt-20261020/step5/`

사용자 결정: **dense 모델로 초석 먼저.** `Qwen3.8-Flash-Next`(qwen4exp)는 별도 작업으로 분리(§7).

## 1. 무엇을 만들었나

| 파일 | 역할 |
|---|---|
| `engine.py` | TP=1/TP=2 디코드 엔진 182줄. 가중치 TP 샤딩 적재, RoPE, GQA, KV 캐시, greedy |
| `run_engine.py` | 실행기. TP=2 는 libtbvcoll 로 all-reduce |
| `validate.py` | HF transformers 와 생성 토큰 비교 |
| `cmp_logits.py` | **logits 직접 비교** (토큰 불일치가 버그인지 반올림인지 가린다) |
| `cmp_decode.py` | 궤적을 고정해 decode 경로(KV 캐시) 스텝별 비교 |
| `bound.py` / `bound2.py` | CPU/GPU 바운드 판정 |
| `prof.py` | 커널 프로파일 |
| `graph_probe.py` | HIP 그래프 캡처 타당성 확인 |
| `run_val.sh` / `run_tp2.sh` | 단일노드 / 2노드 실행 |

설계 원칙: **수학은 PyTorch 연산을 쓴다.** TP=1 이 이미 메모리 루프라인 근처라 자체 커널로 얻을 게
없다(reports/09 §1 #1). 우리가 직접 만드는 것은 **실행 루프와 통신 일정** — 측정으로 확인된 유일한 여지다.

## 2. 정합성 — HF 와 비트 동일

| 검증 | 결과 |
|---|---|
| prefill logits (배치 1, 길이 16) | **최대절대차 0.0000**, 상대오차 0.00e+00, top5 동일 |
| decode logits (배치 2, 궤적 고정, 24스텝) | 23스텝 **0.0000**, step0 만 0.0312(bf16 양자 0.125 의 1/4) |
| decode argmax | **24/24 일치** |
| 자유 생성 24토큰 | [1] 24/24 일치, [0] 21개 일치 후 갈라짐 |

자유 생성의 갈라짐은 **버그가 아니다**: 궤적을 고정하면 argmax 가 전부 일치하고, HF 의 1위-2위
logit 격차가 0.125~0.25(bf16 양자 1~2개)인 지점이 많아 서브양자 오차가 경계를 넘으면 뒤집힌다.
vLLM vs NCCL 에서도 같은 현상을 봤다(reports/10 §2).

**GEMM 융합(q/k/v→qkv, gate/up→gate_up) 후에도 logits 는 0.0000 을 유지했다** → 융합이 정확하다.

## 3. 성능

| 구성 | 배치 | 스텝 | tok/s |
|---|---:|---:|---:|
| 자체 TP=1 | 8 | 84.8 ms | 94.4 |
| **자체 TP=2 + libtbvcoll** | 8 | 61.4 ms | **130.3** (+41%) |
| vLLM TP=2 + NCCL | 8 | — | 139.7 |
| vLLM TP=2 + libtbvcoll | 8 | — | **170.9** |

TP=2 의 전송은 스텝당 56회 × 152~180 µs. 메모리 루프라인 하한은 15.2 GiB / 237 GB/s = **64.1 ms**.

**자체 엔진은 vLLM 보다 31% 느리다.** 원인은 통신이 아니라 **GPU 커널 효율**이다(§4).

## 4. 두 가지 정정 — 측정이 제 가설을 뒤집었다

### 4.1 "CPU 가 벽이다" → **틀렸다**
`bound.py` 에서 30스텝 큐잉이 스텝당 60.2 ms 로 나와 "CPU 디스패치가 스텝의 75%" 라고 판단했다.
`bound2.py` 로 **동기화 직후 소수 스텝만** 재니:

| N 스텝 | 스텝당 큐잉 |
|---:|---:|
| 1 | 7.80 ms |
| 2 | 7.48 ms |
| 5 | 6.30 ms |
| 10 | 17.82 ms |
| 30 | 60.26 ms |

**순수 CPU 디스패치는 6.3~7.8 ms(스텝의 8%)** 이고, 60.2 ms 는 **명령 큐 역압**(CPU 가 GPU 를
기다린 것)이었다. 엔진은 **GPU 바운드**이며 스텝 80 ms = 루프라인 64.1 ms 의 **80%** 다.

### 4.2 "HIP 그래프 캡처가 핵심" → **우선순위 대폭 하향**
`graph_probe.py` 로 타당성은 확인했다:
- 세그먼트(MLP부) CPU 큐잉 **28.5 → 7.1 µs (4×)**, 리플레이 결과 **최대절대차 0.00000**
- KV 캐시의 변하는 쓰기 위치도 **디바이스 텐서 `index_copy_` 로 캡처 가능**(위치 3/7/11 기록 확인)
- 집합통신은 CPU 가 post/폴링해야 하므로 그래프에 못 넣는다 → **경계마다 그래프를 쪼개는** 설계가 필요

그러나 §4.1 로 CPU 가 8% 뿐임이 드러났으므로 **회수 상한이 6 ms(7%)** 다. 먼저 할 일이 아니다.

### 4.3 융합 `rms_norm` 은 HF 와 비트 동일하지 않다
CPU op 7→1, 스텝 −2.2 ms 이지만 **최대절대차 0.2188(상대 1.08e-2)** 로 bf16 양자 0.125 보다 크다.
검증 능력을 유지하려고 `ZB_EXACT_RMS=1` 로 정확 경로를 쓸 수 있게 분리했다.
cmp_logits/cmp_decode 는 이 환경변수와 함께 돌려야 비트 비교가 성립한다.

## 5. 자체 엔진의 가치 명제 재평가

| 항목 | 값 | 근거 |
|---|---:|---|
| vLLM 대비 커널 효율 격차 | **−31%** | §3 (130.3 vs 170.9) |
| 마이크로배치 겹침으로 숨길 통신 | **+14%** | 8.5 ms / 61.4 ms |
| 그래프 캡처로 회수할 CPU | +7% | §4.2 |

**겹침 이득(14%)이 커널 격차(31%)보다 작다.** 따라서 "자체 엔진으로 vLLM 을 이긴다" 는
커널까지 다 만들어야 성립하고, 루프라인상 커널의 여지는 20% 뿐이다(80% → 100%).

→ **더 나은 경로: 겹침을 vLLM 안에 넣는 것.** vLLM 의 DBO 차단은 `vllm/config/vllm.py` 의
평범한 `assert` 이며 ubatch 머신러리(`UBatchWrapper`, `maybe_create_ubatch_slices`)는
all2all 전용으로 보이지 않는다. 우회 스크립트 `step3b/dbo_patch.py` 로 config 검증을 통과시켰고
(주의: import 경로는 `/app/vllm` 이 아니라 site-packages 다) 실행 결과는 §6.

## 6. 재현

```bash
cd experiments/tbv-adopt-20261020/step5
SCRIPT=cmp_logits.py ./run_val.sh                      # 정합성(ZB_EXACT_RMS=1 권장)
SCRIPT=cmp_decode.py ./run_val.sh --steps 24
SCRIPT=bound2.py     ./run_val.sh --batch 8            # CPU/GPU 바운드 판정
SCRIPT=graph_probe.py ./run_val.sh --batch 8
SCRIPT=run_engine.py ./run_val.sh --tp 1 --batch 8 --new 64 --rounds 2
./run_tp2.sh --batch 8 --new 64 --rounds 2             # 2노드 TP=2
```

## 7. 남은 일
1. vLLM DBO 우회 실행 결과 확정 — 되면 겹침 14% 를 vLLM 에서 바로 얻는다(§5).
2. 자체 엔진의 GPU 효율 80% → 원인 규명(skinny GEMM 의심, 미측정).
3. 마이크로배치 겹침을 자체 엔진에 구현 — DBO 우회가 실패할 때의 경로.
4. `qwen4exp`(Qwen3.8-Flash-Next) 는 별도. 하이브리드 SSM + 512전문가 MoE + hyper-connection
   + 디스크 PLE 구조라 재구현이 아니라 **llama.cpp 포크 기반 + 우리 전송** 경로가 맞다.
   이 모델에서는 MoE 때문에 TP 보다 PP 가 유리하다(활성 4.17 GiB, 스텝당 96회 vs 1회).

---

## 8. vLLM DBO 우회 시도 결과 — 차단 확정 (§5 의 "더 나은 경로" 철회)

§5 에서 "겹침을 vLLM 안에 넣는 것이 더 낫다" 고 했으나 **불가능하다.** 차단 사슬 6단계를
하나씩 통과시켜 가며 확인했다(전체: `local_analysis/VLLM_DBO_REQUIRES_DEEPEP_20261021.md`).

| # | 위치 | 성격 |
|---|---|---|
| 1 | `config/vllm.py:1169` DeepEP 백엔드 강제 | 잘못 놓인 어서션 → 우회 |
| 2 | `gpu_ubatch_wrapper.py:73` "SM control only on CUDA" | 잘못 놓인 어서션(setter 가 무연산) → 우회 |
| 3 | `gpu_model_runner.py:3615` `should_ubatch` 가 DP 안에서만 설정 | 제어 흐름 DP 전용 → 분기 추가 |
| 4 | `gpu_ubatch_wrapper.py:452` `assert dp_metadata is not None` | 잘못 놓인 어서션(값 미사용) |
| 5 | `forward_context.py:105` `assert data_parallel_size > 1` | **구현 의존** |
| 6 | `forward_context.py:106` `assert is_moe_model is not False` | **구현 의존 — dense 거부** |

5·6 에서 중단했다. dense TP 모델에 DP+MoE 상태를 위조하는 일이 되고 조용히 틀릴 위험을
출력해시만으로 한정할 수 없다.

### 1·2 만 우회한 상태의 통제 비교 (배치 64, 같은 빌드)
| | tok/s | 집합통신 | 전송 | 해시 |
|---|---:|---:|---:|---|
| dbo=0 | 814.6 | 14934회 | 370.0 µs | `464f721db71ab595` |
| dbo=1 | 826.2 | **14934회** | 355.6 µs | **동일** |

해시까지 동일 → **DBO 플래그는 아무 일도 하지 않는다.**
(앞서 "DBO 로 819.2 tok/s" 라고 보고한 것은 잘못된 귀속이었다. 그 값은 비동기 + 레일별
진행스레드 + cap 적정화로 오른 현재 빌드의 배치 64 성능이다.)

### 결론: 겹침은 자체 엔진에서 구현한다
실행 루프를 우리가 쥐고 있어 DP/MoE 기구에 의존하지 않는다.
측정된 상금: 배치 8 에서 14%(8.5/61.4 ms), **배치 64 에서 28%(21.5/77.5 ms)**.
자체 엔진은 vLLM 대비 커널이 31% 느리지만(§3), 겹침은 vLLM 이 **구조적으로 못 하는** 것이다.
