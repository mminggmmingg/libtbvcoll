// allreduce_mr — 2랭크 N레일 all-reduce. 커스텀 LLM 엔진 통신계층의 초석.
//
// 왜: TP=2 decode 의 스텝 58ms 중 17.1ms(29%)가 NCCL all-reduce 56회(56KiB 당 306us).
//     루프라인상 연산/메모리는 32ms 이므로 통신만 줄이면 138→187 tok/s 가 열린다.
//
// 설계(2랭크라 ring/tree 불필요 — 전량 교환 후 로컬 가산. 바이트수는 ring 과 동일하고 홉은 1회로 최소):
//   1. 버퍼를 레일 수만큼 등분, 레일 r 은 자기 조각만 피어의 staging(GPU) 으로 RDMA_WRITE_WITH_IMM
//   2. 각 레일의 recv 완료 = 그 조각이 내 staging 에 도착했다는 신호 (별도 flag/fence 불필요)
//   3. 전 레일 도착 후 GPU 커널: out = mine + staging
//
// 1레일판(allreduce2.cpp) 에서 실측으로 확정한 설계 결정 3개를 그대로 승계한다:
//   a. dmabuf MR 은 **HSA export 가 주는 offset** 으로 등록해야 한다. hipMemGetHandleForAddressRange 로
//      버퍼마다 export 하면 같은 슬랩의 fd 가 나와 offset 0 두 MR 이 같은 메모리를 가리킨다(정합성 깨짐).
//   b. flag 별도 WR 방식은 못 쓴다. dmabuf 는 RAW_STREAM 경로, 호스트 flag 는 일반 경로라
//      같은 QP 라도 순서가 보장되지 않는다 → WRITE_WITH_IMM 의 recv 완료를 쓴다.
//   c. 매 WR SIGNALED 는 임계경로에 171us 를 더한다 → 32개마다 1개만 시그널.
//
// build(컨테이너): hipcc -O3 -o armr allreduce_mr.cpp -libverbs
// run: rank0: ./armr 0 usb4_rdma0,usb4_rdma5 <bytes> <iters>
//      rank1: ./armr 1 usb4_rdma0,usb4_rdma5 <bytes> <iters> <peer_ip>
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
#include <dlfcn.h>
#include <vector>
#include <string>
#include <algorithm>
#define GID_IDX 1
#define PORT 18992
#define SIGEVERY 32
#define CK(c,m) do{ if(!(c)){ fprintf(stderr,"[fail] %s errno=%d(%s)\n",m,errno,strerror(errno)); return 1; } }while(0)
#define HCK(c,m) do{ hipError_t e=(c); if(e!=hipSuccess){ fprintf(stderr,"[fail] %s: %s\n",m,hipGetErrorString(e)); return 1; } }while(0)
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+t.tv_nsec/1e9; }
static ssize_t rdall(int fd,void*p,size_t n){char*c=(char*)p;size_t g=0;while(g<n){ssize_t r=read(fd,c+g,n-g);if(r<=0)return r;g+=r;}return g;}

struct exch { uint32_t qpn, rkey_stage; uint64_t addr_stage; uint8_t gid[16]; } __attribute__((packed));

struct Rail {
  const char* name;
  ibv_context* ctx; ibv_pd* pd; ibv_cq* cq; ibv_qp* qp;
  ibv_mr *mr_mine, *mr_stage;
  size_t off, bytes;                 // 이 레일이 담당하는 조각
  exch me, pe;
  int recv_left, pend_recv, pend_send, outstanding;
};

__global__ void k_add(float* __restrict__ out, const float* __restrict__ a,
                      const float* __restrict__ b, size_t n){
  size_t i = blockIdx.x*(size_t)blockDim.x + threadIdx.x;
  size_t s = (size_t)gridDim.x*blockDim.x;
  for (; i<n; i+=s) out[i] = a[i] + b[i];
}

int main(int argc, char** argv){
  if(argc<5){ fprintf(stderr,"usage: %s <rank> <dev[,dev...]> <bytes> <iters> [peer_ip]\n",argv[0]); return 2; }
  int rank=atoi(argv[1]); std::string devs=argv[2];
  size_t len=strtoull(argv[3],0,10); int iters=atoi(argv[4]);
  const char* peer = argc>5?argv[5]:nullptr;
  size_t nfloat=len/4; CK(nfloat>0,"bytes>=4");

  std::vector<std::string> dnames;
  { size_t p=0; while(p<=devs.size()){ size_t q=devs.find(',',p);
      if(q==std::string::npos){ dnames.push_back(devs.substr(p)); break; }
      dnames.push_back(devs.substr(p,q-p)); p=q+1; } }
  int R=dnames.size(); CK(R>=1,"rail>=1");

  // ---- GPU 버퍼: 한 번에 할당하고 HSA export offset 으로 등록 (설계결정 a)
  typedef int (*pfn_export)(const void*, size_t, int*, unsigned long*);
  pfn_export hsa_export=(pfn_export)dlsym(RTLD_DEFAULT,"hsa_amd_portable_export_dmabuf");
  if(!hsa_export){ void* h=dlopen("libhsa-runtime64.so.1",RTLD_LAZY|RTLD_GLOBAL);
                   if(h) hsa_export=(pfn_export)dlsym(h,"hsa_amd_portable_export_dmabuf"); }
  CK(hsa_export,"hsa_amd_portable_export_dmabuf 없음");
  bool hostbuf = getenv("HOSTBUF") && atoi(getenv("HOSTBUF"));
  size_t region=(len+4095)&~4095ull;
  float* base;
  if(hostbuf) HCK(hipHostMalloc((void**)&base,region*3,0),"hipHostMalloc");
  else        HCK(hipMalloc(&base,region*3),"hipMalloc");
  float *mine=base, *stage=(float*)((char*)base+region), *out=(float*)((char*)base+2*region);
  if(hostbuf){ for(size_t i=0;i<nfloat;i++) mine[i]=rank==0?1.0f:2.0f; memset(stage,0,len); }
  else { std::vector<float> h(nfloat, rank==0?1.0f:2.0f);
         HCK(hipMemcpy(mine,h.data(),len,hipMemcpyHostToDevice),"H2D");
         HCK(hipMemset(stage,0,len),"memset stage"); }
  int fd_m=-1,fd_s=-1; unsigned long off_m=0,off_s=0;
  if(!hostbuf){ CK(hsa_export(mine ,region,&fd_m,&off_m)==0,"export mine");
                CK(hsa_export(stage,region,&fd_s,&off_s)==0,"export stage"); }
  int acc=IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ;

  // ---- 레일별 자원
  int n=0; ibv_device** L=ibv_get_device_list(&n);
  std::vector<Rail> rr(R);
  size_t chunk=((len/R)+4095)&~4095ull;        // 4K 정렬 등분
  for(int r=0;r<R;r++){
    Rail& x=rr[r]; x.name=dnames[r].c_str();
    x.off = r*chunk;
    x.bytes = (x.off>=len)?0:std::min(chunk, len-x.off);
    ibv_device* D=nullptr;
    for(int i=0;i<n;i++) if(dnames[r]==ibv_get_device_name(L[i])) D=L[i];
    CK(D,"device not found");
    x.ctx=ibv_open_device(D); CK(x.ctx,"open");
    x.pd=ibv_alloc_pd(x.ctx); CK(x.pd,"pd");
    x.cq=ibv_create_cq(x.ctx,512,nullptr,nullptr,0); CK(x.cq,"cq");
    ibv_qp_init_attr qia{}; qia.send_cq=x.cq; qia.recv_cq=x.cq; qia.qp_type=IBV_QPT_RC;
    qia.cap.max_send_wr=64; qia.cap.max_recv_wr=512; qia.cap.max_send_sge=1; qia.cap.max_recv_sge=1;
    x.qp=ibv_create_qp(x.pd,&qia); CK(x.qp,"create_qp");
    if(hostbuf){ x.mr_mine =ibv_reg_mr(x.pd,mine ,len,acc); CK(x.mr_mine ,"reg mine host");
                 x.mr_stage=ibv_reg_mr(x.pd,stage,len,acc); CK(x.mr_stage,"reg stage host"); }
    else       { x.mr_mine =ibv_reg_dmabuf_mr(x.pd,off_m,len,(uint64_t)(uintptr_t)mine ,fd_m,acc); CK(x.mr_mine ,"reg mine");
                 x.mr_stage=ibv_reg_dmabuf_mr(x.pd,off_s,len,(uint64_t)(uintptr_t)stage,fd_s,acc); CK(x.mr_stage,"reg stage"); }
    ibv_gid g; CK(ibv_query_gid(x.ctx,1,GID_IDX,&g)==0,"query_gid");
    x.me.qpn=x.qp->qp_num; x.me.rkey_stage=x.mr_stage->rkey;
    x.me.addr_stage=(uint64_t)(uintptr_t)stage + x.off;     // 피어가 쓸 내 staging 조각
    memcpy(x.me.gid,&g,16);
    x.recv_left=0; x.pend_recv=x.pend_send=x.outstanding=0;
  }

  // ---- 소켓으로 레일 정보 일괄 교환
  int sfd=-1;
  if(rank==0){ int ls=socket(AF_INET,SOCK_STREAM,0); int on=1;
    setsockopt(ls,SOL_SOCKET,SO_REUSEADDR,&on,sizeof on);
    sockaddr_in sa{}; sa.sin_family=AF_INET; sa.sin_addr.s_addr=INADDR_ANY; sa.sin_port=htons(PORT);
    CK(bind(ls,(sockaddr*)&sa,sizeof sa)==0,"bind"); CK(listen(ls,1)==0,"listen");
    printf("[rank0] waiting :%d (%d rail)\n",PORT,R); fflush(stdout);
    sfd=accept(ls,0,0); CK(sfd>=0,"accept"); close(ls);
  } else { CK(peer,"peer_ip 필요"); sfd=socket(AF_INET,SOCK_STREAM,0);
    sockaddr_in sa{}; sa.sin_family=AF_INET; sa.sin_port=htons(PORT); inet_pton(AF_INET,peer,&sa.sin_addr);
    for(int i=0;i<60;i++){ if(connect(sfd,(sockaddr*)&sa,sizeof sa)==0) break; usleep(500000); } }
  { std::vector<exch> tx(R),rx(R);
    for(int r=0;r<R;r++) tx[r]=rr[r].me;
    CK(write(sfd,tx.data(),R*sizeof(exch))==(ssize_t)(R*sizeof(exch)),"tx exch");
    CK(rdall(sfd,rx.data(),R*sizeof(exch))==(ssize_t)(R*sizeof(exch)),"rx exch");
    for(int r=0;r<R;r++) rr[r].pe=rx[r]; }

  for(int r=0;r<R;r++){ Rail& x=rr[r];
    { ibv_qp_attr a{}; a.qp_state=IBV_QPS_INIT; a.pkey_index=0; a.port_num=1; a.qp_access_flags=acc;
      CK(ibv_modify_qp(x.qp,&a,IBV_QP_STATE|IBV_QP_PKEY_INDEX|IBV_QP_PORT|IBV_QP_ACCESS_FLAGS)==0,"INIT"); }
    { ibv_qp_attr a{}; a.qp_state=IBV_QPS_RTR; a.path_mtu=IBV_MTU_4096; a.dest_qp_num=x.pe.qpn;
      a.rq_psn=0; a.max_dest_rd_atomic=1; a.min_rnr_timer=12;
      a.ah_attr.is_global=1; a.ah_attr.port_num=1; a.ah_attr.grh.sgid_index=GID_IDX;
      memcpy(&a.ah_attr.grh.dgid,x.pe.gid,16); a.ah_attr.grh.hop_limit=1;
      CK(ibv_modify_qp(x.qp,&a,IBV_QP_STATE|IBV_QP_AV|IBV_QP_PATH_MTU|IBV_QP_DEST_QPN|IBV_QP_RQ_PSN|
                               IBV_QP_MAX_DEST_RD_ATOMIC|IBV_QP_MIN_RNR_TIMER)==0,"RTR"); }
    { ibv_qp_attr a{}; a.qp_state=IBV_QPS_RTS; a.sq_psn=0; a.timeout=14; a.retry_cnt=7; a.rnr_retry=7; a.max_rd_atomic=1;
      CK(ibv_modify_qp(x.qp,&a,IBV_QP_STATE|IBV_QP_SQ_PSN|IBV_QP_TIMEOUT|IBV_QP_RETRY_CNT|
                               IBV_QP_RNR_RETRY|IBV_QP_MAX_QP_RD_ATOMIC)==0,"RTS"); }
  }
  { char s='S'; CK(write(sfd,&s,1)==1,"barrier tx"); CK(rdall(sfd,&s,1)==1,"barrier rx"); }  // RTS 배리어 필수

  int threads=256, blocks=(int)std::min<size_t>(160,(nfloat+threads-1)/threads);
  uint64_t seq=0;
  double acc_post=0,acc_wait=0,acc_kern=0,acc_cq=0; int acc_n=0;

  auto post_recvs=[&](Rail& x,int k)->int{
    for(int i=0;i<k;i++){ ibv_recv_wr rw{},*bad=0; rw.wr_id=9; rw.num_sge=0;
      if(ibv_post_recv(x.qp,&rw,&bad)) return 1; }
    x.recv_left+=k; return 0; };
  for(int r=0;r<R;r++) if(rr[r].bytes) CK(post_recvs(rr[r],400)==0,"post_recv");

  auto one=[&](bool timed,bool sync_each)->int{
    seq++; bool sig=(seq%SIGEVERY==0);
    double t0=now();
    for(int r=0;r<R;r++){ Rail& x=rr[r]; if(!x.bytes) continue;
      if(x.recv_left<32) { if(post_recvs(x,360)) return 1; }
      ibv_sge sge{}; sge.addr=(uint64_t)(uintptr_t)mine+x.off; sge.length=x.bytes; sge.lkey=x.mr_mine->lkey;
      ibv_send_wr w{},*bad=0; w.wr_id=1; w.sg_list=&sge; w.num_sge=1;
      w.opcode=IBV_WR_RDMA_WRITE_WITH_IMM; w.imm_data=(uint32_t)seq;
      w.send_flags = sig?IBV_SEND_SIGNALED:0;
      w.wr.rdma.remote_addr=x.pe.addr_stage; w.wr.rdma.rkey=x.pe.rkey_stage;
      if(ibv_post_send(x.qp,&w,&bad)){ printf("post_send r%d errno=%d\n",r,errno); return 1; }
      if(sig) x.outstanding++;
    }
    double t1=now();
    // 전 레일의 조각 도착 대기
    for(int r=0;r<R;r++){ Rail& x=rr[r]; if(!x.bytes) continue;
      while(x.pend_recv==0){ ibv_wc wc; int c=ibv_poll_cq(x.cq,1,&wc);
        if(c>0){ if(wc.status!=IBV_WC_SUCCESS){ printf("r%d wc=%d(%s) op=%d\n",r,wc.status,ibv_wc_status_str(wc.status),wc.opcode); return 1; }
                 if(wc.opcode==IBV_WC_RDMA_WRITE){ x.pend_send++; x.outstanding--; } else { x.pend_recv++; x.recv_left--; } }
        if(now()-t1>5){ printf("recv timeout r%d seq=%lu\n",r,seq); return 1; } }
      x.pend_recv--; }
    double t2=now();
    k_add<<<blocks,threads>>>(out,mine,stage,nfloat);
    if(sync_each) HCK(hipDeviceSynchronize(),"kernel");
    double t3=now();
    for(int r=0;r<R;r++){ Rail& x=rr[r]; if(!x.bytes) continue;
      ibv_wc wcs[16]; int c=ibv_poll_cq(x.cq,16,wcs);
      for(int i=0;i<c;i++){ if(wcs[i].status!=IBV_WC_SUCCESS){ printf("drain r%d wc=%d\n",r,wcs[i].status); return 1; }
        if(wcs[i].opcode==IBV_WC_RDMA_WRITE){ x.pend_send++; x.outstanding--; } else { x.pend_recv++; x.recv_left--; } }
      while(x.outstanding>=2){ ibv_wc wc; int c2=ibv_poll_cq(x.cq,1,&wc);
        if(c2>0){ if(wc.status!=IBV_WC_SUCCESS){ printf("drain2 r%d wc=%d\n",r,wc.status); return 1; }
                  if(wc.opcode==IBV_WC_RDMA_WRITE){ x.pend_send++; x.outstanding--; } else { x.pend_recv++; x.recv_left--; } }
        if(now()-t3>5){ printf("drain timeout r%d\n",r); return 1; } } }
    double t4=now();
    if(timed){ acc_post+=t1-t0; acc_wait+=t2-t1; acc_kern+=t3-t2; acc_cq+=t4-t3; acc_n++; }
    return 0; };

  for(int i=0;i<20;i++) if(one(false,true)) return 1;
  { char s='W'; write(sfd,&s,1); rdall(sfd,&s,1); }

  double dt_stream=0;
  for(int mode=0;mode<2;mode++){
    bool se=(mode==0);
    acc_post=acc_wait=acc_kern=acc_cq=0; acc_n=0;
    { char c='M'; write(sfd,&c,1); rdall(sfd,&c,1); }
    double t=now();
    for(int i=0;i<iters;i++) if(one(true,se)) return 1;
    if(!se) HCK(hipDeviceSynchronize(),"final");
    double d=(now()-t)/iters;
    if(!se) dt_stream=d;
    if(rank==0) printf("  %8zu B %drail%s %-10s lat **%7.1f us**  algbw %5.2f Gb/s  [post %5.1f | 수신대기 %6.1f | 커널 %5.1f | 송신CQ %5.1f]\n",
      len,R,hostbuf?"/host":"/gpu ",se?"호스트동기":"스트림순서",d*1e6,len*8/d/1e9,
      acc_post/acc_n*1e6,acc_wait/acc_n*1e6,acc_kern/acc_n*1e6,acc_cq/acc_n*1e6);
  }

  HCK(hipDeviceSynchronize(),"sync");
  std::vector<float> h(std::min<size_t>(nfloat,1024));
  HCK(hipMemcpy(h.data(),out,h.size()*4,hipMemcpyDefault),"D2H");
  bool ok=true; for(float v:h) if(v<2.99f||v>3.01f){ ok=false; break; }
  if(rank==0) printf("            → %s\n", ok?"verify=OK":"verify=BAD");
  { char s='D'; write(sfd,&s,1); rdall(sfd,&s,1); }
  close(sfd); return ok?0:1;
}
