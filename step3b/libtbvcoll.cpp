// libtbvcoll — 2랭크 TP all-reduce 라이브러리. vLLM/torch 에서 호출한다.
//
// 모드 2개:
//   SYNC  — 호출 스레드가 스트림 완료를 기다리고 RDMA/CQ 를 직접 돌린다. (TBV_ASYNC=0)
//   ASYNC — 호출 스레드는 큐잉만 하고 즉시 반환. 진행 스레드가 RDMA/CQ 를 돌린 뒤
//           hipStreamWaitValue64 플래그를 올려 GPU 를 해제한다. (TBV_ASYNC=1, 기본)
//
// 왜 ASYNC 인가(실측):
//   SYNC 는 스트림이 빌 때까지 바쁜-대기하므로 **스텝마다 58번 CPU 선행 큐잉을 파괴**한다.
//   배치 128 에서 스텝 131.8 ms 중 pre 76.1(GPU 연산) + 전송 40.0 + 미설명 15.5 ms 였고,
//   그 15.5 ms 가 GPU 를 굶기는 CPU 직렬화다. prefill 은 아니다(출력 128/256 2점 측정으로 분리).
//
// 실측으로 확정한 설계 결정:
//   1. dmabuf MR 은 **HSA export offset** 으로 등록. 버퍼마다 hipMemGetHandleForAddressRange 로
//      export 하면 같은 슬랩 fd 가 나와 offset 0 두 MR 이 동일 메모리를 가리킨다(정합성 깨짐).
//   2. 호스트 flag 별도 WR 금지 — dmabuf 는 RAW_STREAM, 호스트는 일반 경로라 같은 QP·RC 라도
//      순서 보장이 없다 → WRITE_WITH_IMM 의 recv 완료를 완료신호로 쓴다.
//   3. **GPU 커널이 호스트 flag 를 올리는 완료통지는 쓸 수 없다.** 앞선 복사의 L2 더티라인이
//      플러시되지 않아 NHI DMA 엔진이 오래된 데이터를 읽는다(verify=BAD). ASYNC 에서는
//      hipEventCreateWithFlags(hipEventReleaseToSystem) 로 시스템범위 release 를 강제한다.
//   4. 레일 교차점 96 KiB — 56KiB 는 1레일이 2레일보다 빠르다(133 vs 169 us).
//   5. 호출자 포인터가 매번 바뀌므로 사전등록 VRAM 바운스로 복사(D2D 211 GB/s, 56KiB 0.27 us).
//   6. out-of-place — vLLM 의 all-reduce 계약.
//   7. **대기 중인 hipStreamWaitValue64 는 GPU 사용률 100% 로 보인다**(스핀 폴링). 진행 스레드가
//      멈추면 GPU 가 영구 스핀하므로 **워치독 필수**. 2026-10-21 사고에서 확인.
//
// build: hipcc -O3 -fPIC -shared -o libtbvcoll.so libtbvcoll.cpp -libverbs
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <infiniband/verbs.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>
#include <dlfcn.h>
#include <pthread.h>
#include <sched.h>
#include <vector>
#include <string>
#include <algorithm>

#define GID_IDX 1
#define MAXRAIL 4
#define MAXSLOT 32                // 할당 상한
// 슬롯 링 깊이. 크면 메인 스레드가 더 앞서가지만 버퍼가 캐시에서 식는다
// (32슬롯 × 896KiB = 28.7MB 순환). TBV_NSLOT 로 조정한다.
#define NSLOT (G.nslot)
#define SPIN_US 30                // 조건변수 전 짧은 스핀. 이 안에 끝나면 깨우기 비용 0.
#define RAIL1_MAX (G.rail1max)    // 이하는 1레일이 빠르다(실측 교차점). TBV_RAIL1MAX 로 조정
// 워치독: 이 시간 넘으면 오류로 간주하고 GPU 를 풀어준다. TBV_WDOG(초)로 바꾼다(기본 10).
// 큰 모델의 첫 요청은 JIT 컴파일로 오래 걸린다(GLM-5.3 TCP 첫 요청 89.6s) — 그때 오탐하면 늘린다.
static double WDOG_SEC = 10.0;

static double nowt(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
static ssize_t rdall(int fd,void*p,size_t n){char*c=(char*)p;size_t g=0;while(g<n){ssize_t r=read(fd,c+g,n-g);if(r<=0)return r;g+=r;}return g;}
#define FAIL(m) do{ fprintf(stderr,"[tbvcoll] %s errno=%d(%s)\n",m,errno,strerror(errno)); return -1; }while(0)

struct exch { uint32_t qpn, rkey; uint64_t addr; uint8_t gid[16]; } __attribute__((packed));

struct Item { uint64_t seq; int slot; size_t nbytes; int R; size_t xbytes; };  // xbytes = 전송 바이트(태그 포함)

// 레일마다 진행 스레드를 둔다. post 는 드라이버가 4048B 프레임으로 분할하는 CPU 작업이라
// 한 스레드에서 순차 post 하면 두 번째 레일이 그만큼 늦게 시작한다
// (896KiB 2레일에서 rail0 post 37.4us / rail1 69.1us → rail1 도착이 31.7us 밀림, 실측).
struct Rail {
  ibv_context* ctx; ibv_pd* pd; ibv_cq* cq; ibv_qp* qp;
  ibv_mr *mr_send, *mr_recv;
  exch me, pe;
  int recv_left, pend_recv, outstanding;
  int idx;
  Item q[MAXSLOT];
  volatile uint64_t q_head, q_tail;
  pthread_t th;
};

static struct {
  int inited=0, rank=0, R=0, sfd=-1, async=1, nslot=8;
  Rail rr[MAXRAIL];
  char *sendbase=nullptr, *recvbase=nullptr;    // 각각 NSLOT*cap
  size_t cap=0;
  size_t stride=0;                              // 슬롯 간격. TBV_TAG=1 이면 cap+4096(태그 자리)
  // TBV_TAG=1 진단: 송신측 복사커널이 페이로드 뒤에 seq 태그를 써서 같은 WR 로 보내고,
  // 수신측은 플래그 해제 직후 태그를 일반 로드/시스템범위 로드로 읽어 기대 seq 와 비교한다.
  int uncached=0;
  int tag=0; unsigned long long* dbg=nullptr;   // dbg: [검사, 일반불일치, 시스템불일치, 첫seq, 첫일반값, 첫시스템값]
  void* last_stream=nullptr; uint64_t n_stream_sw=0;
  volatile unsigned long long* gflag=nullptr;   // 슬롯별, 호스트 coherent (GPU 가 대기)
  volatile unsigned long long* cflag=nullptr;   // 슬롯별, 복사완료 통지(GPU → 진행 스레드)
  unsigned int* arrive=nullptr;                 // 슬롯별 블록 도착 카운터(디바이스)
  int gpusig=1;
  int kwait=0;                                  // 1: GPU 대기를 CP 대기패킷 대신 스핀 커널로(진단용)
  unsigned long long* bc=nullptr;               // GPU 발자국(호스트 가시): [대기진입, 대기통과, 복사시작, 가산완료] seq
  size_t rail1max=96u*1024u;
  int cpu=-1;
  int wr_hwm=8;
  size_t chunkb=0;                              // WR 당 바이트. 0=레일 조각 전체를 WR 1개로                                 // 1 = 커널 플래그(진행 스레드에 HIP 호출 0)
  hipEvent_t ev[MAXSLOT];
  uint64_t seq=0;
  volatile unsigned rails_done[MAXSLOT]={0};     // 슬롯별 도착 완료 레일 수
  volatile int run=0;
  // 스핀만으로 기다리면 코어를 태워 진행 스레드와 경합한다(배치 128 에서 전송 673→756us 실측).
  // 짧게 스핀한 뒤 조건변수로 잠든다.
  pthread_mutex_t mx=PTHREAD_MUTEX_INITIALIZER;
  pthread_cond_t  cv_item=PTHREAD_COND_INITIALIZER;   // 메인 → 진행: 새 항목
  pthread_cond_t  cv_room=PTHREAD_COND_INITIALIZER;   // 진행 → 메인: 자리 생김
  volatile int err=0;                            // 워치독/전송 오류
  // 통계
  double t_acc=0, t_pre=0, t_post=0, t_enq=0, t_drain=0; uint64_t n_acc=0;
  double hb_t[32]={0}; uint64_t hb_n[32]={0}, hb_r[32]={0};
  double rl_post[MAXRAIL]={0}, rl_arr[MAXRAIL]={0}; uint64_t rl_n[MAXRAIL]={0};
} G;

// 복사 + 완료통지를 **한 커널**에서 한다. 블록마다 __syncthreads() 로 자기 쓰기를 캐시까지 내리고
// thread0 이 __threadfence_system() 으로 시스템범위 writeback 을 한 뒤 도착 카운터를 올린다.
// 마지막 블록이 플래그를 쓴다 → 플래그가 보이면 모든 복사가 메모리까지 내려가 있다(NHI DMA 가 읽음).
// 별도 k_sig 커널로 하면 앞선 복사의 L2 더티라인이 플러시되지 않아 DMA 가 오래된 데이터를 읽는다
// (2026-10-21 실측 verify=BAD). 반드시 같은 커널이어야 한다.
__device__ inline void bc_put(unsigned long long* bc, int i, unsigned long long v){
  if(bc) __hip_atomic_store(bc+i, v, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM); }
__global__ void k_copy_sig(char* __restrict__ dst, const char* __restrict__ src, size_t n,
                           volatile unsigned long long* flag, unsigned long long seq,
                           unsigned int* arrive, unsigned int nblk, unsigned long long* tagp,
                           unsigned long long* bc){
  if(blockIdx.x==0 && threadIdx.x==0) bc_put(bc,2,seq);
  size_t i=(blockIdx.x*(size_t)blockDim.x+threadIdx.x)*16, st=(size_t)gridDim.x*blockDim.x*16;
  for(;i+16<=n;i+=st) *(float4*)(dst+i)=*(const float4*)(src+i);
  if(blockIdx.x==0 && threadIdx.x==0){ size_t t=n&~(size_t)15; for(size_t j=t;j<n;j++) dst[j]=src[j];
    if(tagp) *tagp=seq; }
  __syncthreads();                 // 블록의 쓰기가 캐시까지 도달
  if(threadIdx.x==0){
    __threadfence_system();        // L2 → 시스템 writeback
    if(atomicAdd(arrive,1u)==nblk-1){ *arrive=0; __threadfence_system(); *flag=seq; }
  }
}

// vLLM 의 all-reduce 계약은 out-of-place 이므로 out 을 분리한다(out==in 이면 in-place).
__global__ void k_tagchk(const unsigned long long* tag, unsigned long long seq, unsigned long long* dbg){
  unsigned long long v1=*tag;
  unsigned long long v2=__hip_atomic_load(tag,__ATOMIC_RELAXED,__HIP_MEMORY_SCOPE_SYSTEM);
  atomicAdd_system(&dbg[0],1ull);
  if(v1!=seq) atomicAdd_system(&dbg[1],1ull);
  if(v2!=seq) atomicAdd_system(&dbg[2],1ull);
  if((v1!=seq||v2!=seq) && dbg[3]==0){ dbg[3]=seq; dbg[4]=v1; dbg[5]=v2; }
}

// TBV_KWAIT=1(진단용): hipStreamWaitValue64(CP 가 호스트 메모리를 폴링) 대신 웨이브 1개가 플래그를 스핀하고
// GPU 발자국(bc)을 남긴다. 처음엔 GFXOFF 로 대기 패킷이 안 깨어난다는 가설로 넣었으나 기각됐고(KWAIT 에서도 정지),
// 이 발자국으로 진짜 원인(rails_done 경합, 아래 tbv_ar_allreduce 주석)을 찾았다. >= 비교.
__global__ void k_waitflag(const unsigned long long* flag, unsigned long long seq, unsigned long long* bc){
  bc_put(bc,0,seq);
  while(__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq)
    __builtin_amdgcn_s_sleep(2);
  bc_put(bc,1,seq);
}
__global__ void k_mark(unsigned long long* bc, int i, unsigned long long seq){ bc_put(bc,i,seq); }
template<typename T> __global__ void k_acc(T* __restrict__ out, const T* __restrict__ a,
                                           const T* __restrict__ b, size_t n){
  size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x, s=(size_t)gridDim.x*blockDim.x;
  for(;i<n;i+=s) out[i]=a[i]+b[i];
}

static inline char* slot_send(int s){ return G.sendbase + (size_t)s*G.stride; }
static inline char* slot_recv(int s){ return G.recvbase + (size_t)s*G.stride; }

// 한 레일의 전송을 수행한다: post → 피어 조각 도착 대기. 송신 수거는 호출자가 뒤에서 한다.
static int do_xfer_rail(Rail& x, const Item& it){
  int R=it.R; size_t chunk=((it.xbytes/R)+4095)&~4095ull;
  size_t soff=(size_t)it.slot*G.stride;
  size_t roff=(size_t)x.idx*chunk;
  if(roff>=it.xbytes) return 0;                       // 이 레일은 담당 조각이 없다
  size_t rlen=std::min(chunk,it.xbytes-roff);
  double t0=nowt();
  if(x.recv_left<32){ for(int i=0;i<360;i++){ ibv_recv_wr rw{},*bad=0; rw.wr_id=9; rw.num_sge=0;
      if(ibv_post_recv(x.qp,&rw,&bad)) return -1; } x.recv_left+=360; }
  { size_t ck = G.chunkb? G.chunkb : rlen;
    for(size_t o=0;o<rlen;o+=ck){
      size_t len=std::min(ck,rlen-o); bool last=(o+len>=rlen);
      ibv_sge sge{}; sge.addr=(uint64_t)(uintptr_t)slot_send(it.slot)+roff+o;
      sge.length=len; sge.lkey=x.mr_send->lkey;
      ibv_send_wr w{},*bad=0; w.wr_id=1; w.sg_list=&sge; w.num_sge=1;
      w.opcode = last? IBV_WR_RDMA_WRITE_WITH_IMM : IBV_WR_RDMA_WRITE;
      if(last) w.imm_data=(uint32_t)it.seq;
      w.send_flags=IBV_SEND_SIGNALED;
      w.wr.rdma.remote_addr=x.pe.addr+soff+roff+o; w.wr.rdma.rkey=x.pe.rkey;
      if(ibv_post_send(x.qp,&w,&bad)) return -1;
      x.outstanding++;
      if(last) G.rl_post[x.idx] += nowt()-t0;
    } }
  // 피어 조각 도착만 기다린다(내 송신 완료는 플래그 해제 뒤로 미룬다 —
  // 임계경로에서 기다리면 56KiB 가 122→222us 로 악화된다, 실측).
  while(x.pend_recv==0){ ibv_wc wc; int c=ibv_poll_cq(x.cq,1,&wc);
    if(c>0){ if(wc.status!=IBV_WC_SUCCESS){
               fprintf(stderr,"[tbvcoll] rail%d wc=%d(%s)\n",x.idx,wc.status,ibv_wc_status_str(wc.status)); return -1; }
             if(wc.opcode==IBV_WC_RDMA_WRITE) x.outstanding--;
             else { x.pend_recv++; x.recv_left--; } }
    if(nowt()-t0>WDOG_SEC){ fprintf(stderr,"[tbvcoll] 워치독: rail%d 전송 %.1fs 초과 seq=%llu\n",
                                    x.idx,nowt()-t0,(unsigned long long)it.seq); return -1; } }
  x.pend_recv--;
  { double d=nowt()-t0;
    G.rl_arr[x.idx] += d; G.rl_n[x.idx]++;
    if(x.idx==0){ G.t_acc+=d; G.n_acc++;
      int b=0; { size_t v=it.nbytes; while(v>>=1) b++; } if(b>31) b=31;
      G.hb_t[b]+=d; G.hb_n[b]++; G.hb_r[b]+=R; } }
  return 0;
}

// 플래그 해제 뒤 내 송신 완료를 수거한다. outstanding <= wr_hwm 로 유지하면
// RC FIFO 완료 순서상 항목 i-NSLOT 의 송신이 끝났음이 보장되어 슬롯 재사용이 안전하다.
static void reap_sends(Rail& x){
  ibv_wc wcs[32]; int c=ibv_poll_cq(x.cq,32,wcs);
  for(int i=0;i<c;i++){ if(wcs[i].status!=IBV_WC_SUCCESS){ G.err=1; return; }
    if(wcs[i].opcode==IBV_WC_RDMA_WRITE) x.outstanding--; else { x.pend_recv++; x.recv_left--; } }
  double td=nowt();
  while(x.outstanding > G.wr_hwm){ ibv_wc wc; int c2=ibv_poll_cq(x.cq,1,&wc);
    if(c2>0){ if(wc.status!=IBV_WC_SUCCESS){ G.err=1; return; }
              if(wc.opcode==IBV_WC_RDMA_WRITE) x.outstanding--; else { x.pend_recv++; x.recv_left--; } }
    if(nowt()-td>WDOG_SEC){ fprintf(stderr,"[tbvcoll] 워치독: rail%d 송신수거 초과\n",x.idx); G.err=1; return; } }
}

static void* progress_rail(void* arg){
  Rail& x = *(Rail*)arg;
  if(G.cpu>=0){
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET((G.cpu + x.idx) % CPU_SETSIZE,&cs);
    if(pthread_setaffinity_np(pthread_self(),sizeof cs,&cs))
      fprintf(stderr,"[tbvcoll] rail%d CPU 고정 실패\n",x.idx);
  }
  while(G.run){
    if(__atomic_load_n(&x.q_head,__ATOMIC_ACQUIRE)==x.q_tail){
      double ts=nowt();
      while(nowt()-ts < SPIN_US*1e-6 && __atomic_load_n(&x.q_head,__ATOMIC_ACQUIRE)==x.q_tail && G.run) ;
      if(__atomic_load_n(&x.q_head,__ATOMIC_ACQUIRE)==x.q_tail){
        pthread_mutex_lock(&G.mx);
        while(G.run && __atomic_load_n(&x.q_head,__ATOMIC_ACQUIRE)==x.q_tail){
          struct timespec t2; clock_gettime(CLOCK_REALTIME,&t2);
          t2.tv_nsec += 2000000; if(t2.tv_nsec>=1000000000){ t2.tv_sec++; t2.tv_nsec-=1000000000; }
          pthread_cond_timedwait(&G.cv_item,&G.mx,&t2);
        }
        pthread_mutex_unlock(&G.mx);
      }
      if(!G.run) break;
      if(__atomic_load_n(&x.q_head,__ATOMIC_ACQUIRE)==x.q_tail) continue;
    }
    Item it=x.q[x.q_tail % NSLOT];
    // 복사 완료(시스템범위 release) 대기. 레일마다 같은 플래그를 본다.
    if(G.gpusig){
      double tw=nowt();
      while(__atomic_load_n((unsigned long long*)((char*)G.cflag + 64*(size_t)it.slot),
                            __ATOMIC_ACQUIRE) < it.seq){
        if(nowt()-tw>WDOG_SEC){ fprintf(stderr,"[tbvcoll] 워치독: rail%d 복사플래그 %.0fs 초과 seq=%llu\n",x.idx,WDOG_SEC,
                                        (unsigned long long)it.seq);
                                if(G.bc) fprintf(stderr,"[tbvcoll]   GPU발자국 대기진입=%llu 대기통과=%llu 복사시작=%llu 가산완료=%llu 호스트seq=%llu cflag=%llu gflag[직전]=%llu\n",
                                  G.bc[0],G.bc[1],G.bc[2],G.bc[3],(unsigned long long)G.seq,
                                  *(volatile unsigned long long*)((char*)G.cflag+64*(size_t)it.slot),
                                  *(volatile unsigned long long*)((char*)G.gflag+64*(size_t)((it.seq-1)%NSLOT)));
                                G.err=1; break; } }
      { double w=nowt()-tw;   // 1초 넘는 대기는 남긴다(JIT 지연인지 진짜 멈춤인지 구분용)
        if(w>1.0 && x.idx==0) fprintf(stderr,"[tbvcoll] 복사플래그 대기 %.1fs seq=%llu\n",w,(unsigned long long)it.seq); }
    } else if(x.idx==0 && hipEventSynchronize(G.ev[it.slot])!=hipSuccess){ G.err=1; }
    if(!G.err && do_xfer_rail(x,it)!=0) G.err=1;
    // 마지막으로 끝난 레일이 GPU 를 해제한다. 오류여도 반드시 올린다(영구 스핀 방지).
    // 다음 항목(seq+NSLOT)의 레일들은 GPU 가 이 gflag 를 통과해야 복사 플래그를 받으므로, 0 으로 되돌리는 것이 그보다 먼저다.
    if(__atomic_add_fetch(&G.rails_done[it.slot],1u,__ATOMIC_ACQ_REL) >= (unsigned)it.R){
      __atomic_store_n(&G.rails_done[it.slot],0u,__ATOMIC_RELAXED);
      __atomic_store_n((unsigned long long*)((char*)G.gflag + 64*(size_t)it.slot),
                       it.seq, __ATOMIC_RELEASE); }
    if(!G.err) reap_sends(x);
    __atomic_store_n(&x.q_tail, x.q_tail+1, __ATOMIC_RELEASE);
    pthread_mutex_lock(&G.mx); pthread_cond_broadcast(&G.cv_room); pthread_mutex_unlock(&G.mx);
  }
  return nullptr;
}

extern "C" int tbv_ar_init(int rank, const char* devs, const char* peer_ip, int port, size_t cap){
  if(G.inited) return 0;
  G.rank=rank; G.cap=(cap+4095)&~4095ull;
  { const char* e=getenv("TBV_ASYNC"); G.async = e? atoi(e):1; }
  // TBV_GPUSIG=1(기본): 복사 커널이 직접 완료 플래그를 올린다. 진행 스레드에 HIP 호출이 없어
  // vLLM 메인 스레드의 런치와 HIP 런타임 락에서 경합하지 않는다(배치 128 전송 673→756us 의 원인).
  { const char* e=getenv("TBV_GPUSIG"); G.gpusig = e? atoi(e):1; }
  { const char* e=getenv("TBV_KWAIT"); G.kwait = e? atoi(e):0; }
  { const char* e=getenv("TBV_NSLOT"); int v = e? atoi(e):8;
    if(v<2) v=2; if(v>MAXSLOT) v=MAXSLOT; G.nslot=v; }
  // TBV_CHUNK: 레일 조각을 이 크기의 WR 여러 개로 쪼갠다. 드라이버가 post 시점에 4048B 프레임으로
  // 분할하므로(2MiB 에서 post 90us 실측), WR 1개면 분할이 끝나야 송신이 시작된다.
  // 쪼개면 첫 WR 의 송신과 나머지 WR 의 분할이 겹친다. 0=쪼개지 않음.
  { const char* e=getenv("TBV_CHUNK"); G.chunkb = e? (size_t)strtoull(e,0,10):0; }
  { const char* e=getenv("TBV_RAIL1MAX"); if(e) G.rail1max=(size_t)strtoull(e,0,10); }
  { const char* e=getenv("TBV_CPU"); G.cpu = e? atoi(e):-1; }
  { const char* e=getenv("TBV_WDOG"); if(e && atof(e)>0) WDOG_SEC=atof(e); }
  { const char* e=getenv("TBV_TAG"); G.tag = (e && atoi(e) && G.async && G.gpusig)? 1:0; }
  G.stride = G.cap + (G.tag? 4096:0);
  // 슬롯 재사용 안전: 미수거 송신을 NSLOT-1 "항목" 분량 이하로 유지해야 한다.
  // 청크를 쓰면 항목당 WR 수가 늘어나므로 상한도 비례해 늘린다(항목당 최대 WR 수 = cap/chunk).
  { size_t per_item = G.chunkb? ((G.cap + G.chunkb - 1)/G.chunkb + 1) : 1;
    G.wr_hwm = (int)std::min<size_t>(96, (size_t)(G.nslot-1)*per_item); if(G.wr_hwm<1) G.wr_hwm=1; }

  typedef int (*pfn)(const void*,size_t,int*,unsigned long*);
  pfn hsa=(pfn)dlsym(RTLD_DEFAULT,"hsa_amd_portable_export_dmabuf");
  if(!hsa){ void*h=dlopen("libhsa-runtime64.so.1",RTLD_LAZY|RTLD_GLOBAL); if(h) hsa=(pfn)dlsym(h,"hsa_amd_portable_export_dmabuf"); }
  if(!hsa) FAIL("hsa_amd_portable_export_dmabuf 없음");

  size_t span=G.stride*NSLOT;
  // TBV_UNCACHED: 비트0=송신 바운스, 비트1=수신 바운스를 GPU L2 비캐시(hipDeviceMallocUncached)로.
  // **기본 1(송신 비캐시).** 7.0.0-1015-oem 에서 일반 hipMalloc 송신 버퍼는 k_copy_sig 의
  // __threadfence_system 뒤에도 NHI DMA 가 가끔 같은 슬롯의 이전 내용(seq-NSLOT)을 읽었다
  // (2026-10-24 vLLM TP 실측: 일반 16회 중 14회 오염, 송신 비캐시 14회·42라운드 0회, 수신만 비캐시는 6회 중 3회 오염).
  // 0 이면 예전처럼 한 번에 일반 hipMalloc.
  { const char* e=getenv("TBV_UNCACHED"); G.uncached = e? atoi(e):1; }
  if(!G.uncached){
    char* base;
    if(hipMalloc((void**)&base, span*2)!=hipSuccess) FAIL("hipMalloc bounce");
    G.sendbase=base; G.recvbase=base+span;
  } else {
    auto alloc=[&](char** p, bool unc)->bool{
      return unc ? hipExtMallocWithFlags((void**)p, span, hipDeviceMallocUncached)==hipSuccess
                 : hipMalloc((void**)p, span)==hipSuccess; };
    if(!alloc(&G.sendbase, G.uncached&1)) FAIL("alloc send bounce");
    if(!alloc(&G.recvbase, G.uncached&2)) FAIL("alloc recv bounce");
  }
  if(hipHostMalloc((void**)&G.gflag, 64*NSLOT, hipHostMallocCoherent)!=hipSuccess) FAIL("hipHostMalloc flag");
  for(int i=0;i<NSLOT;i++) *(unsigned long long*)((char*)G.gflag+64*(size_t)i)=0;
  if(hipHostMalloc((void**)&G.cflag, 64*NSLOT, hipHostMallocCoherent)!=hipSuccess) FAIL("hipHostMalloc cflag");
  if(hipHostMalloc((void**)&G.bc, 64, hipHostMallocCoherent)!=hipSuccess) FAIL("hipHostMalloc bc");
  memset(G.bc,0,64);
  if(G.tag){ if(hipHostMalloc((void**)&G.dbg, 64, hipHostMallocCoherent)!=hipSuccess) FAIL("hipHostMalloc dbg");
             memset(G.dbg,0,64); }
  for(int i=0;i<NSLOT;i++) *(unsigned long long*)((char*)G.cflag+64*(size_t)i)=0;
  if(hipMalloc((void**)&G.arrive, sizeof(unsigned int)*NSLOT)!=hipSuccess) FAIL("hipMalloc arrive");
  if(hipMemset(G.arrive,0,sizeof(unsigned int)*NSLOT)!=hipSuccess) FAIL("memset arrive");
  for(int i=0;i<NSLOT;i++){
    // hipEventReleaseToSystem: 이벤트 완료 시 시스템범위 release(캐시 쓰기되돌림)를 강제한다.
    if(hipEventCreateWithFlags(&G.ev[i], hipEventReleaseToSystem)!=hipSuccess){
      fprintf(stderr,"[tbvcoll] hipEventReleaseToSystem 불가 → 기본 이벤트로 대체\n");
      if(hipEventCreate(&G.ev[i])!=hipSuccess) FAIL("hipEventCreate");
    }
  }
  int fds=-1,fdr=-1; unsigned long offs=0,offr=0;
  if(hsa(G.sendbase,span,&fds,&offs)!=0) FAIL("export send");
  if(hsa(G.recvbase,span,&fdr,&offr)!=0) FAIL("export recv");

  std::vector<std::string> dn;
  { std::string d=devs; size_t p=0; while(p<=d.size()){ size_t q=d.find(',',p);
      if(q==std::string::npos){ dn.push_back(d.substr(p)); break; } dn.push_back(d.substr(p,q-p)); p=q+1; } }
  G.R=(int)dn.size(); if(G.R<1||G.R>MAXRAIL) FAIL("rail 수");

  int acc=IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ;
  int n=0; ibv_device** L=ibv_get_device_list(&n);
  for(int r=0;r<G.R;r++){ Rail& x=G.rr[r];
    ibv_device* D=nullptr; for(int i=0;i<n;i++) if(dn[r]==ibv_get_device_name(L[i])) D=L[i];
    if(!D) FAIL("device 없음");
    x.ctx=ibv_open_device(D); if(!x.ctx) FAIL("open");
    x.pd=ibv_alloc_pd(x.ctx);  if(!x.pd) FAIL("pd");
    x.cq=ibv_create_cq(x.ctx,1024,0,0,0); if(!x.cq) FAIL("cq");
    ibv_qp_init_attr q{}; q.send_cq=x.cq; q.recv_cq=x.cq; q.qp_type=IBV_QPT_RC;
    q.cap.max_send_wr=256; q.cap.max_recv_wr=512; q.cap.max_send_sge=1; q.cap.max_recv_sge=1;
    x.qp=ibv_create_qp(x.pd,&q); if(!x.qp) FAIL("create_qp");
    x.mr_send=ibv_reg_dmabuf_mr(x.pd,offs,span,(uint64_t)(uintptr_t)G.sendbase,fds,acc); if(!x.mr_send) FAIL("reg send");
    x.mr_recv=ibv_reg_dmabuf_mr(x.pd,offr,span,(uint64_t)(uintptr_t)G.recvbase,fdr,acc); if(!x.mr_recv) FAIL("reg recv");
    ibv_gid g; if(ibv_query_gid(x.ctx,1,GID_IDX,&g)) FAIL("query_gid");
    x.me.qpn=x.qp->qp_num; x.me.rkey=x.mr_recv->rkey;
    x.me.addr=(uint64_t)(uintptr_t)G.recvbase; memcpy(x.me.gid,&g,16);
    x.recv_left=0; x.pend_recv=0; x.outstanding=0; x.idx=r; x.q_head=0; x.q_tail=0;
  }

  if(rank==0){ int ls=socket(AF_INET,SOCK_STREAM,0); int on=1;
    setsockopt(ls,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on);
    sockaddr_in sa{}; sa.sin_family=AF_INET; sa.sin_addr.s_addr=INADDR_ANY; sa.sin_port=htons(port);
    if(bind(ls,(sockaddr*)&sa,sizeof sa)) FAIL("bind"); if(listen(ls,1)) FAIL("listen");
    G.sfd=accept(ls,0,0); close(ls); if(G.sfd<0) FAIL("accept");
  } else { G.sfd=socket(AF_INET,SOCK_STREAM,0);
    sockaddr_in sa{}; sa.sin_family=AF_INET; sa.sin_port=htons(port); inet_pton(AF_INET,peer_ip,&sa.sin_addr);
    int okc=-1; for(int i=0;i<120;i++){ if((okc=connect(G.sfd,(sockaddr*)&sa,sizeof sa))==0) break; usleep(250000); }
    if(okc) FAIL("connect"); }
  { std::vector<exch> tx(G.R),rx(G.R);
    for(int r=0;r<G.R;r++) tx[r]=G.rr[r].me;
    size_t sz=G.R*sizeof(exch);
    if(write(G.sfd,tx.data(),sz)!=(ssize_t)sz) FAIL("tx exch");
    if(rdall(G.sfd,rx.data(),sz)!=(ssize_t)sz) FAIL("rx exch");
    for(int r=0;r<G.R;r++) G.rr[r].pe=rx[r]; }

  for(int r=0;r<G.R;r++){ Rail& x=G.rr[r];
    { ibv_qp_attr a{}; a.qp_state=IBV_QPS_INIT; a.pkey_index=0; a.port_num=1; a.qp_access_flags=acc;
      if(ibv_modify_qp(x.qp,&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS)) FAIL("INIT"); }
    { ibv_qp_attr a{}; a.qp_state=IBV_QPS_RTR; a.path_mtu=IBV_MTU_4096; a.dest_qp_num=x.pe.qpn;
      a.rq_psn=0; a.max_dest_rd_atomic=1; a.min_rnr_timer=12;
      a.ah_attr.is_global=1; a.ah_attr.port_num=1; a.ah_attr.grh.sgid_index=GID_IDX;
      a.ah_attr.grh.hop_limit=1; memcpy(&a.ah_attr.grh.dgid,x.pe.gid,16);
      if(ibv_modify_qp(x.qp,&a,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN|
                               IBV_QP_MAX_DEST_RD_ATOMIC|IBV_QP_MIN_RNR_TIMER)) FAIL("RTR"); }
    { ibv_qp_attr a{}; a.qp_state=IBV_QPS_RTS; a.sq_psn=0; a.timeout=14; a.retry_cnt=7; a.rnr_retry=7; a.max_rd_atomic=1;
      if(ibv_modify_qp(x.qp,&a,IBV_QP_STATE|IBV_QP_SQ_PSN|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|
                               IBV_QP_RNR_RETRY|IBV_QP_MAX_QP_RD_ATOMIC)) FAIL("RTS"); }
    for(int i=0;i<400;i++){ ibv_recv_wr rw{},*bad=0; rw.wr_id=9; rw.num_sge=0;
      if(ibv_post_recv(x.qp,&rw,&bad)) FAIL("post_recv"); }
    x.recv_left=400;
  }
  { char s='S'; if(write(G.sfd,&s,1)!=1) FAIL("barrier tx"); if(rdall(G.sfd,&s,1)!=1) FAIL("barrier rx"); }

  if(G.async){ G.run=1;
    for(int r=0;r<G.R;r++){ G.rr[r].q_head=0; G.rr[r].q_tail=0;
      if(pthread_create(&G.rr[r].th,nullptr,progress_rail,&G.rr[r])) FAIL("pthread_create"); } }
  G.inited=1;
  fprintf(stderr,"[tbvcoll] rank%d ready: %d rail, cap=%zu KiB, 슬롯 %d, 모드 %s\n",
          rank,G.R,G.cap/1024,NSLOT,G.async?(G.gpusig?(G.kwait?"ASYNC+GPUSIG+KWAIT":"ASYNC+GPUSIG"):"ASYNC+EVENT"):"SYNC");
  fprintf(stderr,"[tbvcoll]   청크 %zu KiB, 송신WR 상한 %d, 레일별 진행스레드 %d개\n",G.chunkb/1024,G.wr_hwm,G.async?G.R:0);
  if(G.uncached||G.tag) fprintf(stderr,"[tbvcoll]   진단: TBV_UNCACHED=%d TBV_TAG=%d\n",G.uncached,G.tag);
  return 0;
}

// dtype: 0=float32 1=float16 2=bfloat16. out==nullptr 이면 in-place.
extern "C" int tbv_ar_allreduce(void* ptr, void* outp, size_t nbytes, int dtype, void* stream){
  if(!G.inited) FAIL("미초기화");
  if(G.err) FAIL("이전 전송 오류 — 복구 불가");
  if(nbytes>G.cap) return -2;                       // 호출자가 NCCL 로 폴백
  if(!outp) outp=ptr;
  hipStream_t st=(hipStream_t)stream;
  if(stream!=G.last_stream){ if(G.last_stream) G.n_stream_sw++; G.last_stream=stream;
    // 처음 보는 스트림이면 남긴다(요청 경계 정지가 다른 스트림 호출과 관련 있는지 확인용)
    static void* seen[16]; static int nseen=0; bool f=false;
    for(int i=0;i<nseen;i++) if(seen[i]==stream){ f=true; break; }
    if(!f && nseen<16){ seen[nseen++]=stream;
      fprintf(stderr,"[tbvcoll] 새 스트림 #%d %p seq=%llu nbytes=%zu\n",nseen,stream,(unsigned long long)(G.seq+1),nbytes); }
    else if(G.n_stream_sw<=20) fprintf(stderr,"[tbvcoll] 스트림 전환 %p seq=%llu\n",stream,(unsigned long long)(G.seq+1)); }
  size_t tagoff=(nbytes+7)&~(size_t)7, xbytes=G.tag? tagoff+8 : nbytes;
  int R = (nbytes<=RAIL1_MAX)?1:G.R;
  int th=256;

  double te=nowt();
  uint64_t myseq = ++G.seq;
  int slot = (int)(myseq % NSLOT);

  if(G.async){
    // 큐가 꽉 차면(진행 스레드가 NSLOT 만큼 밀렸으면) 비운다. 이벤트/슬롯 재사용 안전장치.
    // 모든 레일 큐에 자리가 생길 때까지 기다린다
    auto full=[&]()->bool{ for(int r=0;r<R;r++)
        if(G.rr[r].q_head - __atomic_load_n(&G.rr[r].q_tail,__ATOMIC_ACQUIRE) >= (uint64_t)NSLOT) return true;
      return false; };
    if(full()){
      double ts=nowt();
      while(nowt()-ts < SPIN_US*1e-6 && full() && !G.err) ;
      pthread_mutex_lock(&G.mx);
      while(!G.err && full()){
        struct timespec ts2; clock_gettime(CLOCK_REALTIME,&ts2);
        ts2.tv_nsec += 2000000; if(ts2.tv_nsec>=1000000000){ ts2.tv_sec++; ts2.tv_nsec-=1000000000; }
        pthread_cond_timedwait(&G.cv_room,&G.mx,&ts2);
      }
      pthread_mutex_unlock(&G.mx);
      if(G.err) FAIL("전송 오류");
    }
    if(G.gpusig){
      size_t v=(nbytes+15)/16; int bl=(int)std::min<size_t>(320,(v+th-1)/th); if(bl<1) bl=1;
      k_copy_sig<<<bl,th,0,st>>>(slot_send(slot),(const char*)ptr,nbytes,
                                 (volatile unsigned long long*)((char*)G.cflag + 64*(size_t)slot),myseq,&G.arrive[slot],(unsigned)bl,
                                 G.tag? (unsigned long long*)(slot_send(slot)+tagoff) : nullptr, G.bc);
      if(hipGetLastError()!=hipSuccess) FAIL("k_copy_sig 런치");
    } else {
      if(hipMemcpyAsync(slot_send(slot),ptr,nbytes,hipMemcpyDeviceToDevice,st)!=hipSuccess) FAIL("D2D in");
      if(hipEventRecord(G.ev[slot],st)!=hipSuccess) FAIL("eventRecord");
    }
    // rails_done[slot] 은 호스트가 지우지 않는다. 지우면 경합: 디코드(1레일) 뒤 prefill(2레일)처럼 레일별 큐 깊이가
    // 다를 때, 호스트가 같은 슬롯의 다음 항목(seq+NSLOT)을 넣으며 0 으로 만들 때 rail1 이 아직 seq 를 끝내지 않았으면
    // 카운트가 R 에 못 미쳐 gflag 가 영영 안 올라간다(2026-10-25 GLM-5.3 요청 경계 정지, GPU 발자국으로 확인:
    // 두 레일 모두 seq 를 끝냈는데 gflag[슬롯]=seq-NSLOT). 마지막 레일이 스스로 0 으로 되돌린다.
    for(int r=0;r<R;r++){ Rail& x=G.rr[r];
      uint64_t h=__atomic_load_n(&x.q_head,__ATOMIC_RELAXED);
      x.q[h % NSLOT] = Item{myseq,slot,nbytes,R,xbytes};
      __atomic_store_n(&x.q_head,h+1,__ATOMIC_RELEASE); }
    pthread_mutex_lock(&G.mx); pthread_cond_broadcast(&G.cv_item); pthread_mutex_unlock(&G.mx);
    // GPU 가 진행 스레드의 플래그를 기다린다. 이 대기 중 GPU 사용률은 100% 로 보인다(스핀).
    if(G.kwait){
      k_waitflag<<<1,1,0,st>>>((const unsigned long long*)((char*)G.gflag + 64*(size_t)slot),myseq,G.bc);
      if(hipGetLastError()!=hipSuccess) FAIL("k_waitflag 런치");
    } else if(hipStreamWaitValue64(st,(void*)((char*)G.gflag + 64*(size_t)slot),(uint64_t)myseq,
                            hipStreamWaitValueEq,0xffffffffffffffffull)!=hipSuccess) FAIL("streamWaitValue");
    if(G.tag) k_tagchk<<<1,1,0,st>>>((const unsigned long long*)(slot_recv(slot)+tagoff),myseq,G.dbg);
  } else {
    if(hipMemcpyAsync(slot_send(slot),ptr,nbytes,hipMemcpyDeviceToDevice,st)!=hipSuccess) FAIL("D2D in");
    double tp=nowt();
    for(;;){ hipError_t e=hipStreamQuery(st);
      if(e==hipSuccess) break;
      if(e!=hipErrorNotReady) FAIL("stream query");
      if(nowt()-tp>WDOG_SEC) FAIL("stream 워치독"); }
    G.t_pre+=nowt()-tp;
    Item it{myseq,slot,nbytes,R,xbytes};
    for(int r=0;r<R;r++) if(do_xfer_rail(G.rr[r],it)!=0) FAIL("전송");
    double tdr=nowt(); for(int r=0;r<R;r++) reap_sends(G.rr[r]); G.t_drain+=nowt()-tdr;
  }

  double tq=nowt();
  const char* rb = slot_recv(slot);
  if(dtype==0){ size_t m=nbytes/4; int bl=(int)std::min<size_t>(160,(m+th-1)/th);
    k_acc<float><<<bl,th,0,st>>>((float*)outp,(const float*)ptr,(const float*)rb,m); }
  else if(dtype==1){ size_t m=nbytes/2; int bl=(int)std::min<size_t>(160,(m+th-1)/th);
    k_acc<__half><<<bl,th,0,st>>>((__half*)outp,(const __half*)ptr,(const __half*)rb,m); }
  else { size_t m=nbytes/2; int bl=(int)std::min<size_t>(160,(m+th-1)/th);
    k_acc<__hip_bfloat16><<<bl,th,0,st>>>((__hip_bfloat16*)outp,(const __hip_bfloat16*)ptr,(const __hip_bfloat16*)rb,m); }
  if(hipGetLastError()!=hipSuccess) FAIL("k_acc");
  if(G.kwait && G.bc) k_mark<<<1,1,0,st>>>(G.bc,3,myseq);
  G.t_post+=nowt()-tq;
  G.t_enq+=nowt()-te;
  return 0;
}

extern "C" void tbv_ar_stats(double* avg_us, unsigned long long* count){
  if(avg_us) *avg_us = G.n_acc? G.t_acc/G.n_acc*1e6 : 0.0;
  if(count)  *count  = G.n_acc;
  G.t_acc=0; G.n_acc=0;
}
// 구간별: pre(SYNC 만) / xfer(전송, ASYNC 에서는 진행 스레드 기준) / post(가산커널 런치) / enq(호출자가 쓴 총 시간)
extern "C" void tbv_ar_stats5(double* pre_us, double* xfer_us, double* post_us, double* enq_us,
                              double* drain_us, unsigned long long* count){
  double n = G.n_acc? (double)G.n_acc : 1.0;
  if(pre_us)   *pre_us   = G.t_pre  /n*1e6;
  if(xfer_us)  *xfer_us  = G.t_acc  /n*1e6;
  if(post_us)  *post_us  = G.t_post /n*1e6;
  if(enq_us)   *enq_us   = G.t_enq  /n*1e6;
  if(drain_us) *drain_us = G.t_drain/n*1e6;
  if(count)    *count    = G.n_acc;
  G.t_pre=G.t_acc=G.t_post=G.t_enq=G.t_drain=0; G.n_acc=0;
}
extern "C" void tbv_ar_stats4(double* pre_us, double* xfer_us, double* post_us, double* enq_us,
                              unsigned long long* count){
  double n = G.n_acc? (double)G.n_acc : 1.0;
  if(pre_us)  *pre_us  = G.t_pre /n*1e6;
  if(xfer_us) *xfer_us = G.t_acc /n*1e6;
  if(post_us) *post_us = G.t_post/n*1e6;
  if(enq_us)  *enq_us  = G.t_enq /n*1e6;
  if(count)   *count   = G.n_acc;
  G.t_pre=G.t_acc=G.t_post=G.t_enq=0; G.n_acc=0;
}
extern "C" int tbv_ar_cap(void){ return (int)G.cap; }
extern "C" int tbv_ar_async(void){ return G.async; }
extern "C" int tbv_ar_err(void){ return G.err; }
// 레일별 post 완료 / 피어 도착 평균을 덤프한다. 두 레일의 도착 시각 차이가 크면 불균형,
// post 시각 차이가 크면 직렬 post 가 두 번째 레일의 시작을 늦춘 것이다.
extern "C" void tbv_ar_rails(void){
  fprintf(stderr,"[tbvcoll] 레일별 (post완료 / 피어도착, us):\n");
  for(int r=0;r<G.R;r++){ if(!G.rl_n[r]) continue;
    fprintf(stderr,"[tbvcoll]   rail%d: post %7.1f   도착 %7.1f   (%llu회)\n",
            r, G.rl_post[r]/G.rl_n[r]*1e6, G.rl_arr[r]/G.rl_n[r]*1e6,
            (unsigned long long)G.rl_n[r]);
    G.rl_post[r]=0; G.rl_arr[r]=0; G.rl_n[r]=0; }
}
extern "C" void tbv_ar_hist(void){
  fprintf(stderr,"[tbvcoll] 스트림 전환 %llu회\n",(unsigned long long)G.n_stream_sw);
  if(G.tag){ hipDeviceSynchronize();
    fprintf(stderr,"[tbvcoll] 태그검사 %llu회  일반로드 불일치 %llu  시스템로드 불일치 %llu  첫불일치 seq=%llu 일반=%llu 시스템=%llu\n",
            G.dbg[0],G.dbg[1],G.dbg[2],G.dbg[3],G.dbg[4],G.dbg[5]); }
  fprintf(stderr,"[tbvcoll] 크기별 전송 (버킷 = 1<<b 이상):\n");
  for(int b=0;b<32;b++){ if(!G.hb_n[b]) continue;
    fprintf(stderr,"[tbvcoll]   %8zu B~ : %7llu 회  평균 %7.1f us  평균레일 %.2f\n",
            (size_t)1<<b,(unsigned long long)G.hb_n[b],G.hb_t[b]/G.hb_n[b]*1e6,
            (double)G.hb_r[b]/G.hb_n[b]);
    G.hb_t[b]=0; G.hb_n[b]=0; G.hb_r[b]=0; }
}
extern "C" void tbv_ar_fini(void){
  if(G.async && G.run){ G.run=0;
    pthread_mutex_lock(&G.mx); pthread_cond_broadcast(&G.cv_item); pthread_mutex_unlock(&G.mx);
    for(int r=0;r<G.R;r++) pthread_join(G.rr[r].th,nullptr); }
}
