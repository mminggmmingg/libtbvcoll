#include <vector>
#include <dlfcn.h>
// allreduce2 — 2랭크 전용 최소 all-reduce. 커스텀 LLM 엔진 통신계층의 초석.
//
// 왜: TP=2 decode 의 스텝시간 58ms 중 **17.1ms(29%)가 NCCL all-reduce 56회**(56KiB 당 306us).
//     루프라인상 연산/메모리는 32ms 이므로 통신만 줄이면 138→187 tok/s(+36%) 가 열린다.
//     2랭크 점대점 링크에서 56KiB 는 와이어로 21us 면 되는데 NCCL 은 306us 를 쓴다 → 자체 구현으로 검증.
//
// 설계(2랭크라 링/트리 불필요 — 단순 교환 후 로컬 가산):
//   1. 내 버퍼를 피어의 staging(GPU) 으로 RDMA_WRITE
//   2. 같은 QP 로 flag(호스트 pinned) 에 RDMA_WRITE  ← RC 는 같은 QP 의 순서를 보장하므로
//      flag 가 보이면 데이터도 도착했다(별도 fence 불필요)
//   3. CPU 가 flag 를 폴링(호스트 메모리라 캐시에서 읽힘, syscall 0)
//   4. GPU 커널: out = mine + staging   (gfx1151 메모리대역 238GB/s 라 56KiB 는 ~1us)
// 트래픽은 NCCL 2랭크 ring 과 동일(각 방향 size). 차이는 **고정비**뿐이다.
//
// build(컨테이너): hipcc -O3 -o ar2 allreduce2.cpp -libverbs
// run server(rank0): ./ar2 0 <dev> <bytes> <iters>
//     client(rank1): ./ar2 1 <dev> <bytes> <iters> <peer_ip>
#include <hip/hip_runtime.h>
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
#include <algorithm>
#define GID_IDX 1
#define PORT 18990
#define CK(c,m) do{ if(!(c)){ fprintf(stderr,"[fail] %s errno=%d(%s)\n",m,errno,strerror(errno)); return 1; } }while(0)
#define HCK(c,m) do{ hipError_t e=(c); if(e!=hipSuccess){ fprintf(stderr,"[fail] %s: %s\n",m,hipGetErrorString(e)); return 1; } }while(0)
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
static ssize_t rdall(int fd,void*p,size_t n){char*c=(char*)p;size_t g=0;while(g<n){ssize_t r=read(fd,c+g,n-g);if(r<=0)return r;g+=r;}return g;}

struct exch { uint32_t qpn, psn, rkey_data, rkey_flag; uint64_t addr_data, addr_flag; uint8_t gid[16]; } __attribute__((packed));

__global__ void k_add(float* __restrict__ out, const float* __restrict__ a,
                      const float* __restrict__ b, size_t n){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x*blockDim.x;
  for (; i<n; i+=s) out[i] = a[i] + b[i];
}

int main(int argc, char** argv){
  if (argc < 5){ fprintf(stderr,"usage: %s <rank 0|1> <dev> <bytes> <iters> [peer_ip]\n",argv[0]); return 1; }
  int rank = atoi(argv[1]); const char* dev = argv[2];
  size_t len = strtoul(argv[3],0,0); int iters = atoi(argv[4]);
  const char* pip = argc>5 ? argv[5] : nullptr;
  if (rank==1 && !pip){ fprintf(stderr,"rank1 needs peer_ip\n"); return 1; }
  size_t nfloat = len/sizeof(float);

  // --- GPU 버퍼: mine(내 입력), stage(피어 데이터 수신), out(결과) ---
  // 중요: mine/stage 를 각각 hipMemGetHandleForAddressRange 로 export 하면 **같은 슬랩의 fd** 가 나와
  // offset 0 으로 등록한 두 MR 이 동일 메모리를 가리킨다(실측: 피어가 stage 에 쓴 값이 mine 을 덮었다).
  // → 한 번에 크게 할당하고 **HSA export 가 돌려주는 offset** 으로 각 영역을 등록한다.
  typedef int (*pfn_export)(const void*, size_t, int*, unsigned long*);
  pfn_export hsa_export = (pfn_export)dlsym(RTLD_DEFAULT,"hsa_amd_portable_export_dmabuf");
  if(!hsa_export){ void* h=dlopen("libhsa-runtime64.so.1",RTLD_LAZY|RTLD_GLOBAL);
                   if(h) hsa_export=(pfn_export)dlsym(h,"hsa_amd_portable_export_dmabuf"); }
  CK(hsa_export,"hsa_amd_portable_export_dmabuf 심볼 없음");
  float* base; size_t region = (len + 4095) & ~4095ull;
  HCK(hipMalloc(&base, region*3),"hipMalloc base");
  float *mine=base, *stage=(float*)((char*)base+region), *out=(float*)((char*)base+2*region);
  HCK(hipMemset(stage,0,len),"memset stage");
  int fd_mine=-1, fd_stage=-1; unsigned long off_mine=0, off_stage=0;
  CK(hsa_export(mine ,region,&fd_mine ,&off_mine )==0,"export mine");
  CK(hsa_export(stage,region,&fd_stage,&off_stage)==0,"export stage");
  { // 결정적 입력: rank0=1.0, rank1=2.0 → 합은 3.0 이어야 한다
    std::vector<float> h(nfloat, rank==0?1.0f:2.0f);
    HCK(hipMemcpy(mine,h.data(),len,hipMemcpyHostToDevice),"H2D mine");
  }


  // --- flag: 호스트 pinned (CPU 가 캐시에서 폴링, 피어가 RDMA_WRITE) ---
  volatile uint64_t* flag=nullptr;
  HCK(hipHostMalloc((void**)&flag, 64, hipHostMallocDefault),"hostMalloc flag");
  *flag = 0;

  int n=0; ibv_device** L=ibv_get_device_list(&n), *D=nullptr;
  for(int i=0;i<n;i++) if(!strcmp(ibv_get_device_name(L[i]),dev)) D=L[i];
  CK(D,"device not found");
  ibv_context* ctx=ibv_open_device(D); CK(ctx,"open");
  ibv_pd* pd=ibv_alloc_pd(ctx); CK(pd,"pd");
  ibv_cq* cq=ibv_create_cq(ctx,256,nullptr,nullptr,0); CK(cq,"cq");
  ibv_qp_init_attr qia{}; qia.send_cq=cq; qia.recv_cq=cq; qia.qp_type=IBV_QPT_RC;
  qia.cap.max_send_wr=64; qia.cap.max_recv_wr=512; qia.cap.max_send_sge=1; qia.cap.max_recv_sge=1;
  ibv_qp* qp=ibv_create_qp(pd,&qia); CK(qp,"create_qp");

  int acc=IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ;
  ibv_mr* mr_mine  = ibv_reg_dmabuf_mr(pd,off_mine ,len,(uint64_t)(uintptr_t)mine ,fd_mine ,acc); CK(mr_mine ,"reg mine");
  ibv_mr* mr_stage = ibv_reg_dmabuf_mr(pd,off_stage,len,(uint64_t)(uintptr_t)stage,fd_stage,acc); CK(mr_stage,"reg stage");
  if(rank==0) printf("  [MR] mine off=%lu lkey=0x%x | stage off=%lu rkey=0x%x\n",
                     off_mine,mr_mine->lkey,off_stage,mr_stage->rkey);
  ibv_mr* mr_flag  = ibv_reg_mr(pd,(void*)flag,64,acc); CK(mr_flag,"reg flag");

  ibv_gid mygid; CK(ibv_query_gid(ctx,1,GID_IDX,&mygid)==0,"query_gid");
  exch me{},pe{};
  me.qpn=qp->qp_num; me.psn=0;
  me.rkey_data=mr_stage->rkey; me.addr_data=(uint64_t)(uintptr_t)stage;   // 피어가 여기에 쓴다
  me.rkey_flag=mr_flag->rkey;  me.addr_flag=(uint64_t)(uintptr_t)flag;
  memcpy(me.gid,&mygid,16);

  int sfd=-1;
  if(rank==0){ int ls=socket(AF_INET,SOCK_STREAM,0); int on=1; setsockopt(ls,SOL_SOCKET,SO_REUSEADDR,&on,4);
    sockaddr_in sa{}; sa.sin_family=AF_INET; sa.sin_addr.s_addr=INADDR_ANY; sa.sin_port=htons(PORT);
    CK(bind(ls,(sockaddr*)&sa,sizeof sa)==0,"bind"); CK(listen(ls,1)==0,"listen");
    printf("[rank0] waiting :%d\n",PORT); fflush(stdout);
    sfd=accept(ls,0,0); CK(sfd>=0,"accept"); close(ls);
  } else { sfd=socket(AF_INET,SOCK_STREAM,0); sockaddr_in sa{}; sa.sin_family=AF_INET; sa.sin_port=htons(PORT);
    inet_pton(AF_INET,pip,&sa.sin_addr);
    for(int i=0;i<60;i++){ if(connect(sfd,(sockaddr*)&sa,sizeof sa)==0) break; usleep(500000); } }
  CK(write(sfd,&me,sizeof me)==(ssize_t)sizeof me,"tx exch");
  CK(rdall(sfd,&pe,sizeof pe)==(ssize_t)sizeof pe,"rx exch");

  { ibv_qp_attr a{}; a.qp_state=IBV_QPS_INIT; a.pkey_index=0; a.port_num=1; a.qp_access_flags=acc;
    CK(ibv_modify_qp(qp,&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS)==0,"INIT"); }
  { ibv_qp_attr a{}; a.qp_state=IBV_QPS_RTR; a.path_mtu=IBV_MTU_4096; a.dest_qp_num=pe.qpn;
    a.rq_psn=pe.psn; a.max_dest_rd_atomic=1; a.min_rnr_timer=12;
    a.ah_attr.is_global=1; a.ah_attr.port_num=1; a.ah_attr.grh.hop_limit=1;
    a.ah_attr.grh.sgid_index=GID_IDX; memcpy(&a.ah_attr.grh.dgid,pe.gid,16);
    CK(ibv_modify_qp(qp,&a,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN|
                     IBV_QP_MAX_DEST_RD_ATOMIC|IBV_QP_MIN_RNR_TIMER)==0,"RTR"); }
  { ibv_qp_attr a{}; a.qp_state=IBV_QPS_RTS; a.sq_psn=0; a.timeout=14; a.retry_cnt=7; a.rnr_retry=7; a.max_rd_atomic=1;
    CK(ibv_modify_qp(qp,&a,IBV_QP_STATE|IBV_QP_SQ_PSN|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|
                     IBV_QP_RNR_RETRY|IBV_QP_MAX_QP_RD_ATOMIC)==0,"RTS"); }
  { char s='S'; CK(write(sfd,&s,1)==1,"barrier tx"); CK(rdall(sfd,&s,1)==1,"barrier rx"); }  // RTS 배리어 필수

  int blocks=std::min<size_t>(160,(nfloat+255)/256), threads=256;
  if(blocks<1) blocks=1;
  uint64_t seq=0;
  double acc_post=0, acc_wait=0, acc_kern=0, acc_cq=0; int acc_n=0;
  // recv WR 을 넉넉히 걸어둔다(WRITE_WITH_IMM 이 소모)
  auto post_recvs=[&](int k){ for(int i=0;i<k;i++){ ibv_recv_wr rw{},*bad=0; rw.wr_id=9; rw.num_sge=0;
      if(ibv_post_recv(qp,&rw,&bad)) return 1; } return 0; };
  if(post_recvs(400)) { printf("post_recv fail\n"); return 1; }
  int recv_left = 400; int pend_recv=0, pend_send=0, outstanding=0;

  auto one_allreduce=[&](bool timed,bool sync_each)->int{
    seq++;
    double t0=now();
    if (recv_left < 16){ if(post_recvs(240)) return 1; recv_left += 240; }
    // 데이터 → 피어 staging, WRITE_WITH_IMM 으로 수신측에 완료를 알린다
    // (flag+별도 WR 방식은 dmabuf=RAW_STREAM 경로와 일반 경로의 순서가 보장되지 않아 정합성이 깨졌다 — 실측)
    ibv_sge sge{}; sge.addr=(uint64_t)(uintptr_t)mine; sge.length=len; sge.lkey=mr_mine->lkey;
    ibv_send_wr w{},*bad=0; w.wr_id=1; w.sg_list=&sge; w.num_sge=1;
    w.opcode=IBV_WR_RDMA_WRITE_WITH_IMM; w.imm_data=(uint32_t)seq;
    // 선택적 시그널링: 32 WR 마다 1개만 완료를 올린다(매번 SIGNALED 는 임계경로에 171us 를 더했다 — 실측)
    bool sig = (seq % 32 == 0);
    w.send_flags = sig ? IBV_SEND_SIGNALED : 0;
    w.wr.rdma.remote_addr=pe.addr_data; w.wr.rdma.rkey=pe.rkey_data;
    if(ibv_post_send(qp,&w,&bad)) { printf("post_send fail errno=%d\n",errno); return 1; }
    double t1=now();
    // 수신 완료(= 피어 데이터가 내 staging 에 도착) 대기.
    // 이미 관측해 둔 완료가 있으면 그걸 먼저 소비한다(예전 버전은 송신대기 루프가 다음 iteration 의
    // 수신완료를 먹어치워 타임아웃이 났다 — 실측 후 수정).
    while(pend_recv==0){ ibv_wc wc; int c=ibv_poll_cq(cq,1,&wc);
      if(c>0){ if(wc.status!=IBV_WC_SUCCESS){ printf("wc status=%d(%s) op=%d\n",wc.status,ibv_wc_status_str(wc.status),wc.opcode); return 1; }
               if(wc.opcode==IBV_WC_RDMA_WRITE) pend_send++; else { pend_recv++; recv_left--; } }
      if(now()-t1>5){ printf("recv timeout seq=%lu\n",seq); return 1; } }
    pend_recv--;
    double t2=now();
    k_add<<<blocks,threads>>>(out,mine,stage,nfloat);
    if(sync_each){ if(hipDeviceSynchronize()!=hipSuccess){ printf("kernel fail\n"); return 1; } }
    double t3=now();
    // 내 송신 완료는 **임계경로에서 기다리지 않는다**(실측 84us 낭비였다).
    // WR 슬롯이 마를 때만 드레인한다. 데이터 도착 보장은 피어의 recv 완료가 해준다.
    if(sig) outstanding++;
    { ibv_wc wcs[16]; int c=ibv_poll_cq(cq,16,wcs);   // 논블로킹 수거(유저공간 CQ 라 1.6ns)
      for(int i=0;i<c;i++){ if(wcs[i].status!=IBV_WC_SUCCESS){ printf("wc status=%d\n",wcs[i].status); return 1; }
        if(wcs[i].opcode==IBV_WC_RDMA_WRITE){ pend_send++; outstanding--; } else { pend_recv++; recv_left--; } } }
    while(outstanding >= 2){   // 시그널 WR 2개 이상 미결 시에만 블록(= 64 WR 분량)
      ibv_wc wc; int c=ibv_poll_cq(cq,1,&wc);
      if(c>0){ if(wc.status!=IBV_WC_SUCCESS){ printf("drain wc status=%d\n",wc.status); return 1; }
               if(wc.opcode==IBV_WC_RDMA_WRITE){ pend_send++; outstanding--; } else { pend_recv++; recv_left--; } }
      if(now()-t3>5){ printf("drain timeout\n"); return 1; } }
    double t4=now();
    if(timed){ acc_post+=t1-t0; acc_wait+=t2-t1; acc_kern+=t3-t2; acc_cq+=t4-t3; acc_n++; }
    return 0;
  };

  for(int i=0;i<20;i++) if(one_allreduce(false,true)) return 1;      // 워밍업
  { char s='W'; write(sfd,&s,1); rdall(sfd,&s,1); }
  double dt_sync=0, dt_stream=0;
  for(int mode=0;mode<2;mode++){
    bool se = (mode==0);
    acc_post=acc_wait=acc_kern=acc_cq=0; acc_n=0;
    { char c='M'; write(sfd,&c,1); rdall(sfd,&c,1); }
    double t=now();
    for(int i=0;i<iters;i++) if(one_allreduce(true,se)) return 1;
    if(!se) HCK(hipDeviceSynchronize(),"final sync");
    double d=(now()-t)/iters;
    if(se) dt_sync=d; else dt_stream=d;
    if(rank==0) printf("  %8zu B  %-14s lat **%7.1f us**  algbw %5.2f Gb/s   [post %5.1f | 수신대기 %6.1f | 커널 %5.1f | 송신CQ %5.1f]\n",
      len, se?"호스트동기":"스트림순서", d*1e6, len*8/d/1e9,
      acc_post/acc_n*1e6, acc_wait/acc_n*1e6, acc_kern/acc_n*1e6, acc_cq/acc_n*1e6);
  }
  double dt=dt_stream;

  // 정합성: 합이 3.0 이어야 한다
  std::vector<float> h(std::min<size_t>(nfloat,1024));
  HCK(hipMemcpy(h.data(),out,h.size()*sizeof(float),hipMemcpyDeviceToHost),"D2H out");
  bool ok=true; for(float v:h) if (v<2.99f||v>3.01f) { ok=false; break; }
  if(!ok){ float m0,s0,o0;
    hipMemcpy(&m0,mine ,4,hipMemcpyDeviceToHost);
    hipMemcpy(&s0,stage,4,hipMemcpyDeviceToHost);
    hipMemcpy(&o0,out  ,4,hipMemcpyDeviceToHost);
    printf("            [디버그] mine[0]=%.3f stage[0]=%.3f out[0]=%.3f (기대 %.1f + %.1f = 3.0)\n",
           m0,s0,o0, rank==0?1.0:2.0, rank==0?2.0:1.0); }
  if(rank==0){
    printf("            → %s\n", ok?"verify=OK":"verify=BAD");
  }
  { char s='D'; write(sfd,&s,1); rdall(sfd,&s,1); }
  close(sfd); return ok?0:1;
}
