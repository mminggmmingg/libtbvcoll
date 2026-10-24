# 선행 연구 조사 — 자체 엔진/통신 최적화 방향 재설정

조사일 2026-10-21. 접근 URL 과 날짜는 각 항목에 기록.
목적: `reports/11` §8 의 "겹침을 자체 엔진에 구현" 계획을 **짓기 전에** 선행 연구로 검증한다.

---

## 1. Apple MLX distributed inference — **우리와 동일 토폴로지의 실측**
https://machinelearning.apple.com/research/mlx-distributed-inference (접근 2026-10-21)

Mac 2대 + Thunderbolt 케이블 2개 + RDMA, MLX 의 `jaccl` 집합통신. **우리 하드웨어 구성과 같다.**

| 모델 | 방식 | decode |
|---|---|---:|
| Muse Spark Open (816B, 활성 42B, NVFP4) | Pipeline | 15.74 tok/s |
| 같음 | **Tensor** | **24.99 tok/s** |

- **TP 가 PP 보다 decode 에서 1.59× 빠르다** — MoE 모델에서도.
- "Pipeline 은 decode 속도를 사실상 그대로 둔다"(대신 prefill 을 최대 1.8× 가속).
- 2케이블 메시는 prefill 에서도 이득: 65,536 토큰 TTFT 50.5s(1케이블) → 40.6s(2케이블).
- 단 **아주 긴 컨텍스트(65K)에서는 2머신 TP 가 PP 에 밀린다.** 머신을 늘리면 TP 가 유리해진다.

### → 자기 정정
`reports/11` §7 과 `local_analysis/PP-P2P-001` 에서 "qwen4exp 는 MoE 라 TP 보다 PP 가 유리"
(활성 4.17 GiB, 스텝당 96회 vs 1회 → TP 49 vs PP 52.9 tok/s)라고 **[추정]** 했다.
Apple 이 비슷한 규모의 MoE 를 같은 토폴로지에서 실측해 **반대 결과**를 냈다.
내 추정은 고정비만 세고 PP 의 직렬 지연(노드당 연산이 순차로 더해짐)을 과소평가했다.
**사용자의 "무조건 TP" 결정이 실측으로 지지된다.**

---

## 2. TokenWeave — **내 겹침 계획의 결함을 알려준다**
https://arxiv.org/abs/2505.11329 / https://arxiv.org/html/2505.11329v5 (접근 2026-10-21)
코드: https://github.com/microsoft/tokenweave (artifact-evaluation 브랜치, vLLM 0.8.5 V1 기반)

TP 통신 오버헤드가 **NVLink 에서도 20%**. 배치의 토큰을 두 부분으로 나눠 한쪽 연산과
다른쪽 통신을 겹친다. 최대 1.29× 지연 개선, 1.26× 처리량.

전이 가능한 설계:
1. **wave-aware 분할**: 분할 후 총 GPU 웨이브 수가 분할 전보다 늘지 않게 쪼갠다.
   H100(132 SM) 예: 300 CTA = 3웨이브(2 full + 1 partial) → 132 / 168 로 쪼개 3웨이브 유지.
   오프라인 프로파일로 `split_offset` 을 시퀀스 길이별로 결정.
2. **레이어당 일정**: 스트림 2개. split1 의 AllReduce+RMSNorm ∥ split2 의 attention →
   split1 의 FFN ∥ split2 의 AllReduce+RMSNorm. `stream_wait` 로 의존성 처리.
3. **RMSNorm 이 4~9% 를 먹는다.** ReduceScatter 후 각 랭크는 1/N 만 들고 있으므로
   자기 몫만 정규화하면 중복 연산이 없어진다. 융합 커널로 **1.34~1.39×**.
4. 통신이 **2~8 SM** 만 쓰게 만든 것이 핵심. 기존 겹침 기법은 16~20+ SM 을 먹어 연산을 굶겼다.

### 왜 우리 워크로드에는 안 맞는가
- **그들 스스로 dense 모델에서 1K 토큰 미만이면 분할을 끈다**(MoE 는 4K). 웨이브 양자화
  손실이 겹침 이득을 넘기 때문이다. **우리 decode 배치는 8~128 토큰** — 한참 아래다.
- 더 근본적으로: 메모리 지배 decode 에서 배치를 쪼개면 **레이어 가중치를 두 번 읽는다.**
  우리 실측으로 계산하면(배치 128, 연산 76 ms = 가중치 32 ms + 토큰비례 44 ms)
  2×64 분할은 각 절반이 32 + 22 = 54 ms, 합 **108 ms**(+32 ms). 숨길 통신이 41 ms 이므로
  완벽히 겹쳐도 순이득 ~9 ms = **7.5%**, 웨이브 손실까지 보면 0 에 가깝다.
- 작은 배치에서는 분할이 **음(-)** 이다.
- 부수 확인: **CUDA Graphs 가 스트림 기반 겹침과 비호환**이라고 명시 — `reports/11` §4.2 에서
  내가 본 "그래프 캡처 vs CPU 주도 집합통신" 긴장과 같은 결론이다.

### 우리에게 유리한 비대칭 하나
TokenWeave 가 2~8 SM 까지 줄이려 애쓴 통신을, **우리는 GPU 연산을 0 쓴다**(CPU 주도 RDMA +
DMA 엔진). 그래서 SM 경합 문제가 없다 — vLLM DBO 의 SM control 기구가 우리에게 불필요했던
이유이기도 하다(`local_analysis/DBO-001` 2단계). 다만 위의 가중치 재독 문제는 그대로다.

### 결론: **겹침은 짓지 않는다.** 이 조사로 헛일을 면했다.

---

## 3. Flash Communication — **대역폭 지배 구간의 올바른 기법**
https://arxiv.org/html/2412.04964v1 (접근 2026-10-21)

활성값(가중치 아님)을 저비트로 양자화해 와이어 바이트를 줄인다.
- **output projection 은 INT4, down projection 은 All-Gather 전 INT8 / 리덕션 후 INT4**
  (down 쪽이 누적오차 때문에 양자화가 더 어렵다)
- **비대칭 양자화, 그룹 크기 128** (그룹별 scale + zero-point)
- 정확도: LLaMA-3-70B perplexity 6.74 → 7.04 (+0.30), LLaMA-2-70B 5.52 → 5.60 (+0.08)
- **L40(PCIe 64 GB/s, TP=4)에서 TTFT 2×**, A100(NVLink 600 GB/s, TP=8)에서 1.19×
  → **느린 링크일수록 이득이 크다.** 우리 링크는 2.63 GB/s 로 L40 PCIe 보다 24× 느리다.
- 공개 코드 없음.

---

## 4. mccl-mlx — 압축의 **적용 판단 규칙**
https://github.com/american-code/mccl-mlx (접근 2026-10-21)

Apple Silicon 클러스터용 NCCL 대응 라이브러리. 토폴로지 플래너 + 와이어 압축.

**크로스오버 규칙**: *"코덱은 패브릭의 비압축 all-reduce 속도가 그 코덱 자신의
encode/decode 상한보다 낮을 때만 이득이다."*

우리 수치로 적용:
- 패브릭 all-reduce 속도(896 KiB, 2레일) = 917504 B / 605 µs = **1.52 GB/s**
- 코덱 상한: 양자화/역양자화는 메모리 지배. 896 KiB bf16 읽기 + 448 KiB int8 쓰기 ≈ 1.3 MB
  @ 211 GB/s(실측 D2D) ≈ **6 µs** → 약 150 GB/s 급
- → 패브릭이 코덱 상한보다 **약 100× 느리다. 규칙상 압축은 확실히 이득이다.** [도출]

압축 방식: downcast fp32→fp16(2.0×), **int8 blockwise(블록별 absmax, 3.94×)**,
Top-K + error feedback.

측정치(M1 Max 2대, TB5 직결): 대역폭 1.06 GB/s(probe) / 1.46 GB/s(쌍 최적화),
**지연 167.8 µs**, 2노드 all-reduce **16 MiB 에서 0.746 GB/s = 버스 이용률 70.4%**.

### 우리 전송과의 비교 (관례 차이가 있어 참고용)
| | mccl/JACCL (TB5, M1 Max) | 우리 (USB4 2레일, gfx1151) |
|---|---|---|
| 소형 지연 | 167.8 µs | **115 µs** (4 KiB 하한, reports/09) |
| all-reduce 이용률 | 70.4% @16 MiB | **90% @2 MiB**, 58% @896 KiB |
→ 우리 전송은 동급 이상이다. 더 짤 것은 전송이 아니라 **바이트 수**다.

---

## 5. JACCL (MLX) — 레퍼런스 구현, MIT
https://github.com/ml-explore/mlx/pull/2808 · 소스 `mlx/distributed/jaccl/lib` (접근 2026-10-21)
- macOS 26.2 의 Thunderbolt RDMA 와 공동 개발. TCP 대비 한 자릿수 빠름.
- **"현재는 지연 최적화이며 full mesh 만 지원. ring 과 대역폭 최적화는 나중."**
  → 우리와 같은 지연 우선 설계. 설계 비교 대상으로 읽을 가치가 있다.
- 알려진 문제: 집합통신 중 케이블 분리 시 `tbt_post_recv` segfault
  (ml-explore/mlx#4192) — 우리 워치독 설계와 대조할 사례.

---

## 6. SiFAR — **적용 불가**
https://arxiv.org/pdf/2607.08973 (접근 2026-10-21)
동기화 없는 all-reduce. 다만 **프로그래머블 스위치의 in-switch 리덕션**이 전제다.
우리는 직결 케이블이라 스위치가 없다. 진단은 유효: "저지연 추론에서는 배치를 작게 유지하므로
겹칠 연산이 부족해 마이크로배치 이득이 작다" — §2 의 결론과 일치.

---

## 7. 재설정된 다음 작업

| 순위 | 작업 | 근거 | 예상 |
|---|---|---|---|
| 1 | **all-reduce 페이로드 양자화** (bf16→int8 blockwise, 그룹 128, 비대칭) | §3 §4. 크로스오버 규칙상 100× 여유 | 배치 128 에서 와이어 349→175 µs, 전송 605→~420 µs, 스텝 120→~109 ms = **+10%** [추정] |
| 2 | int4 로 확장 (down proj 은 int8 유지) | §3 의 레이어별 비트 배분 | 스텝 ~104 ms = **+15%** [추정] |
| 3 | **집합통신 tail(`k_acc`) + 다음 RMSNorm 융합** | §2 의 3번. 커널 런치 + hidden state 1회 read/write 절약, 스텝당 56회 | 1.34~1.39× (해당 연산 기준) |
| 4 | JACCL 소스 대조 | §5 | 설계 점검 |
| ~~5~~ | ~~마이크로배치 겹침~~ | §2 로 **철회** | — |

주의: 1·2 는 **정확도를 바꾼다**(§3: perplexity +0.08~0.30). 출력해시 비교로는 판정할 수 없으므로
perplexity 또는 과제 정확도 측정을 도입해야 한다. 작은 배치(8~32)에서는 와이어가 전송의
13% 뿐이라 이득이 거의 없다 — **크기 임계값을 두고 켜야 한다**(§4 의 규칙과 같은 취지).
