// VLEN>128 hazard demo: with the OLD fixed step of 4 lanes on VLEN=256,
// a k that is a multiple of 8 but relied on k%4==0 alone would double-count.
// Here we test k=16 (2 iterations at VLMAX=8) and confirm the RVV path matches.
#include "llamafile/sgemm.h"
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml.h"
#include "tinyblas_cpu.h"
#include <cmath>
#include <cstdio>
#include <vector>
#include <random>

template <typename F>
static void naive(int m,int n,int k,const float* A,const float* B,float* C,F fa,F fb){
  for(int i=0;i<m;i++) for(int j=0;j<n;j++){
    double s=0; for(int t=0;t<k;t++) s+=(double)fa(A[i*k+t])*(double)fb(B[j*k+t]); C[i*n+j]=(float)s; }
}
int g_m,g_n;  // never used by main() below directly; kept for relerr prototype parity
static double relerr2(const std::vector<float>& c,const std::vector<float>& r,int m,int n){
  double mx=1e-9,e=0;
  for(int i=0;i<m;i++) for(int j=0;j<n;j++){ double v=r[i*n+j]; mx=std::max(mx,std::fabs(v)); e=std::max(e,std::fabs(c[j*m+i]-v)); }
  return e/mx;
}
int main(){
  size_t vlmax=__riscv_vsetvlmax_e32m1();
  printf("VLEN=%zu vlmax=%zu\n", __riscv_vlenb()*8, vlmax);
  int fails=0;
  for (int k : {8, 16, 24, 40, 48}) {
    if (k % (int)vlmax) { printf("k=%d skipped (not multiple of vlmax)\n", k); continue; }
    int m=2,n=3;
    std::mt19937 rng(7); std::normal_distribution<float> d(0,0.1f);
    std::vector<float> A(m*k),B(n*k),C(m*n),R(m*n);
    for(auto&x:A)x=d(rng); for(auto&x:B)x=d(rng);
    auto I=[](float v){return v;};
    naive(m,n,k,A.data(),B.data(),R.data(),I,I);
    bool ok=llamafile_sgemm(m,n,k,A.data(),k,B.data(),k,C.data(),m,0,1,GGML_TASK_TYPE_COMPUTE,
                            GGML_TYPE_F32,GGML_TYPE_F32,GGML_TYPE_F32,GGML_PREC_DEFAULT);
    double e=ok?relerr2(C,R,m,n):1e9;
    printf("k=%3d  supported=%s  max_rel=%.3e  %s\n", k, ok?"yes":"NO", e, e<1e-5?"PASS":"FAIL");
    if(!ok||e>=1e-5) fails++;
  }
  printf("RESULT: %s\n", fails?"FAIL":"ALL PASS");
  return fails?1:0;
}
