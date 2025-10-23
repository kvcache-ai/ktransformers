#include "../la/amx.hpp"

#include <omp.h>

#include "mat-test.hpp"
#define FMT_HEADER_ONLY
#include <fmt/core.h>

const int test_iter = 100;
const bool mt = true;
const bool cache_hit = false;

void q_latency_test_bf16(int m, int n, int k, ggml_bf16_t* qa, ggml_bf16_t* qb) {
  int nth = amx::GemmKernel224BF::recommended_nth(n);
  int m_ = (m + 31) / 32 * 32;
  Mat<float> d(m_, n, Layout::RowMajor);
  {
    int repeat = 100;
    std::vector<ggml_bf16_t*> vec_a;
    std::vector<ggml_bf16_t*> vec_b;
    std::vector<float*> vec_c;
    std::vector<std::shared_ptr<amx::GemmKernel224BF::BufferA>> vec_ba;
    std::vector<std::shared_ptr<amx::GemmKernel224BF::BufferB>> vec_bb;
    std::vector<std::shared_ptr<amx::GemmKernel224BF::BufferC>> vec_bc;
    for (int i = 0; i < repeat * 2; i++) {
      ggml_bf16_t* a = (ggml_bf16_t*)std::aligned_alloc(64, amx::GemmKernel224BF::BufferA::required_size(m_, k));
      std::shared_ptr<amx::GemmKernel224BF::BufferA> ba = std::make_shared<amx::GemmKernel224BF::BufferA>(m_, k, a);
      ggml_bf16_t* b = (ggml_bf16_t*)std::aligned_alloc(64, amx::GemmKernel224BF::BufferB::required_size(n, k));
      std::shared_ptr<amx::GemmKernel224BF::BufferB> bb = std::make_shared<amx::GemmKernel224BF::BufferB>(n, k, b);
      float* c = (float*)std::aligned_alloc(64, amx::GemmKernel224BF::BufferC::required_size(m_, n));
      std::shared_ptr<amx::GemmKernel224BF::BufferC> bc = std::make_shared<amx::GemmKernel224BF::BufferC>(m_, n, c);
      ba->from_mat(m, qa, 0, 1);
      int nth = amx::GemmKernel224BF::recommended_nth(n);
      for (int i = 0; i < nth; i++) {
        bb->from_mat(qb, i, nth);
      }
      vec_a.push_back(a);
      vec_b.push_back(b);
      vec_c.push_back(c);
      vec_ba.push_back(ba);
      vec_bb.push_back(bb);
      vec_bc.push_back(bc);
    }
    Timer t(fmt::format("m:{} n:{} k:{} t:{} repeat:{}, latency", m, n, k, test_iter, repeat));
    for (int t = 0; t < test_iter; t++) {
#pragma omp parallel for schedule(dynamic, 1)
      for (int ti = 0; ti < nth * repeat; ti++) {
        int mat_id = ti / nth + repeat * (t % 2);
        int ith = ti % nth;
        if (cache_hit) {
          mat_id = 0;
        }
        amx::mat_mul(m, n, k, vec_ba[mat_id], vec_bb[mat_id], vec_bc[mat_id], ith, nth);
      }
    }
    for (int i = 0; i < repeat * 2; i++) {
      free(vec_a[i]);
      free(vec_b[i]);
      free(vec_c[i]);
    }
  }
  d.dealloc();
}

void group_q_latency_test_bf16(int n_max, int k_max) {
  amx::GemmKernel224BF::config();

  int m_max = 1024;
  int m_start = 32;
  int m_step = 32;

  Mat<float> a(m_max, k_max, Layout::RowMajor), b(k_max, n_max, Layout::ColumnMajor);
  std::mt19937 gen(123);
  a.random(gen);
  b.random(gen);
  a.quant(GGML_TYPE_BF16);
  b.quant(GGML_TYPE_BF16);

  std::string method_name = "BF16";
  if (mt) {
    method_name += fmt::format("_mt{}", omp_get_max_threads());
  }
  if (cache_hit) {
    method_name += "-cache-hit";
  }

  auto output = fmt::format("{}-m:{}:{}:{}-n:{}-k:{}-x{}x{}.txt", method_name, m_start, m_max, m_step, n_max, k_max,
                            amx::GemmKernel224BF::N_BLOCK, amx::GemmKernel224BF::K_BLOCK);
  // std::cout << "Output to: " << output << std::endl;
  auto x = freopen(output.c_str(), "w", stdout);
  assert(x);

  for (int m = m_start; m <= m_max; m *= 2) {
    q_latency_test_bf16(m, n_max, k_max, a.quant_data<ggml_bf16_t>(), b.quant_data<ggml_bf16_t>());
  }
}

void q_latency_test_int8(int m, int n, int k, ggml_bf16_t* qa, ggml_bf16_t* qb) {
  int nth = amx::GemmKernel224Int8::recommended_nth(n);
  int m_ = (m + 31) / 32 * 32;
  Mat<float> d(m_, n, Layout::RowMajor);
  {
    int repeat = 100;
    std::vector<int8_t*> vec_a;
    std::vector<int8_t*> vec_b;
    std::vector<float*> vec_c;
    std::vector<std::shared_ptr<amx::GemmKernel224Int8::BufferA>> vec_ba;
    std::vector<std::shared_ptr<amx::GemmKernel224Int8::BufferB>> vec_bb;
    std::vector<std::shared_ptr<amx::GemmKernel224Int8::BufferC>> vec_bc;
    for (int i = 0; i < repeat * 2; i++) {
      int8_t* a = (int8_t*)std::aligned_alloc(64, amx::GemmKernel224Int8::BufferA::required_size(m_, k));
      std::shared_ptr<amx::GemmKernel224Int8::BufferA> ba = std::make_shared<amx::GemmKernel224Int8::BufferA>(m_, k, a);
      int8_t* b = (int8_t*)std::aligned_alloc(64, amx::GemmKernel224Int8::BufferB::required_size(n, k));
      std::shared_ptr<amx::GemmKernel224Int8::BufferB> bb = std::make_shared<amx::GemmKernel224Int8::BufferB>(n, k, b);
      float* c = (float*)std::aligned_alloc(64, amx::GemmKernel224Int8::BufferC::required_size(m_, n));
      std::shared_ptr<amx::GemmKernel224Int8::BufferC> bc = std::make_shared<amx::GemmKernel224Int8::BufferC>(m_, n, c);
      ba->from_mat(m, qa, 0, 1);
      int nth = amx::GemmKernel224Int8::recommended_nth(n);
      for (int i = 0; i < nth; i++) {
        bb->from_mat(qb, i, nth);
      }
      vec_a.push_back(a);
      vec_b.push_back(b);
      vec_c.push_back(c);
      vec_ba.push_back(ba);
      vec_bb.push_back(bb);
      vec_bc.push_back(bc);
    }
    Timer t(fmt::format("m:{} n:{} k:{} t:{} repeat:{}, latency", m, n, k, test_iter, repeat));
    for (int t = 0; t < test_iter; t++) {
#pragma omp parallel for schedule(dynamic, 1)
      for (int ti = 0; ti < nth * repeat; ti++) {
        int mat_id = ti / nth + repeat * (t % 2);
        int ith = ti % nth;
        if (cache_hit) {
          mat_id = 0;
        }
        amx::mat_mul(m, n, k, vec_ba[mat_id], vec_bb[mat_id], vec_bc[mat_id], ith, nth);
      }
    }
    for (int i = 0; i < repeat * 2; i++) {
      free(vec_a[i]);
      free(vec_b[i]);
      free(vec_c[i]);
    }
  }
  d.dealloc();
}

void group_q_latency_test_int8(int n_max, int k_max) {
  amx::GemmKernel224Int8::config();

  int m_max = 1024;
  int m_start = 32;
  int m_step = 32;

  Mat<float> a(m_max, k_max, Layout::RowMajor), b(k_max, n_max, Layout::ColumnMajor);
  std::mt19937 gen(123);
  a.random(gen);
  b.random(gen);
  a.quant(GGML_TYPE_BF16);
  b.quant(GGML_TYPE_BF16);

  std::string method_name = "INT8";
  if (mt) {
    method_name += fmt::format("_mt{}", omp_get_max_threads());
  }
  if (cache_hit) {
    method_name += "-cache-hit";
  }

  auto output = fmt::format("{}-m:{}:{}:{}-n:{}-k:{}-x{}x{}.txt", method_name, m_start, m_max, m_step, n_max, k_max,
                            amx::GemmKernel224Int8::N_BLOCK, amx::GemmKernel224Int8::K_BLOCK);
  // std::cout << "Output to: " << output << std::endl;
  auto x = freopen(output.c_str(), "w", stdout);
  assert(x);
  for (int m = m_start; m <= m_max; m *= 2) {
    q_latency_test_int8(m, n_max, k_max, a.quant_data<ggml_bf16_t>(), b.quant_data<ggml_bf16_t>());
  }
}

void correction_test_int4(int m, int n, int k) {
  amx::GemmKernel224Int4::config();

  int m_max = 1024;
  int m_start = 32;
  int m_step = 32;

  Mat<float> ma(m, k, Layout::RowMajor), mb(k, n, Layout::ColumnMajor);
  // std::mt19937 gen(123);

  // for(size_t i=0;i<m;i++){
  //   for(size_t j=0;j<k;j++){
  //     // ma.at(i,j) = std::max(int(-i+j),0);
  //     ma.at(i,j) = (i+j)%25/25.0;
  //   }
  // }
  // for (size_t i = 0; i < k; i++) {
  //   for (size_t j = 0; j < n; j++) {
  //     // mb.at(i,j) = std::max(int(-i+j),0);
  //     mb.at(i,j) = (i+j)%25/25.0;
  //   }
  // }
  std::mt19937 gena(123);
  std::mt19937 genb(312);
  ma.random(gena);
  mb.random(genb);
  // ma.random(gen);
  // mb.random(gen);

  auto mc = ma.mul_check(mb);
  // ma.print();
  // mb.print();

  ma.quant(GGML_TYPE_BF16);
  mb.quant(GGML_TYPE_BF16);

  using K = amx::GemmKernel224Int4;
  int8_t* a = (int8_t*)std::aligned_alloc(64, K::BufferA::required_size(m, k));
  std::shared_ptr<K::BufferA> ba = std::make_shared<K::BufferA>(m, k, a);
  int8_t* b = (int8_t*)std::aligned_alloc(64, K::BufferB::required_size(n, k));
  std::shared_ptr<K::BufferB> bb = std::make_shared<K::BufferB>(n, k, b);
  float* c = (float*)std::aligned_alloc(64, K::BufferC::required_size(m, n));
  std::shared_ptr<K::BufferC> bc = std::make_shared<K::BufferC>(m, n, c);

  ba->from_mat(m, ma.quant_data<ggml_bf16_t>(), 0, 1);
  // printf("%d\n",amx::GemmKernel224Int4::BufferA::required_size(m, k));
  // for(size_t i=0;i<amx::GemmKernel224Int4::BufferA::required_size(m, k);i++){
  //   if((i*2)%k==0)
  //     printf("\n");

  //   printf("%02x ", (unsigned char)(a[i]));
  // }
  // printf("\n");

  // int nth = amx::GemmKernel224Int4::recommended_nth(n);
  bb->from_mat(mb.quant_data<ggml_bf16_t>(), 0, 1);

  // for(size_t i=0;i<amx::GemmKernel224Int4::BufferB::required_size(n, k);i++){
  //  if((i*2)%k==0)
  //     printf("\n");

  //  printf("%02x ", (unsigned char)(b[i]));
  // }
  // printf("\n");

  amx::mat_mul(m, n, k, ba, bb, bc, 0, 1);

  // for(size_t i=0;i<m;i++){
  //   for(size_t j=0;j<n;j++){
  //     printf("%.2f ",c[i*n+j]);
  //   }
  //   printf("\n");
  // }

  // printf("\n");
  Mat<float> tc(m, n, Layout::RowMajor);
  tc.data = c;
  // std::cout<<"AMX OUTPUT:"<<std::endl;
  // tc.print_all();
  // std::cout<<"STD OUTPUT:"<<std::endl;
  // mc.print_all();

  mc.cmp(tc);

  // for(size_t i=0;i<m/32;i++){
  //   for(size_t j=0;j<n/32;j++){
  //     Mat<float> stdre(32,32,Layout::RowMajor);
  //     Mat<float> amxre(32,32,Layout::RowMajor);
  //     for(size_t ii=i*32;ii<i*32+32;ii++){
  //       for(size_t jj=j*32;jj<j*32+32;jj++){
  //         stdre.at(ii-i*32,jj-j*32) = mc.at(ii,jj);
  //         amxre.at(ii-i*32,jj-j*32) = tc.at(ii,jj);
  //       }
  //     }
  //     printf("%d %d ",i,j);
  //     stdre.cmp(amxre);
  //     // if(i==0&&j==0){
  //       std::cout<<"STD"<<std::endl;
  //       stdre.print_all();
  //       std::cout<<"AMX"<<std::endl;
  //       amxre.print_all();
  //     // }
  //   }
  // }
}

void correction_test_int4_1(int m, int n, int k) {
  using K = amx::GemmKernel224Int4_1;
  K::config();

  int m_max = 1024;
  int m_start = 32;
  int m_step = 32;

  Mat<float> ma(m, k, Layout::RowMajor), mb(k, n, Layout::ColumnMajor);
  // std::mt19937 gen(123);

  // for(size_t i=0;i<m;i++){
  //   for(size_t j=0;j<k;j++){
  //     // ma.at(i,j) = std::max(int(-i+j),0);
  //     ma.at(i,j) = (i+j)%25/25.0;
  //   }
  // }
  // for (size_t i = 0; i < k; i++) {
  //   for (size_t j = 0; j < n; j++) {
  //     // mb.at(i,j) = std::max(int(-i+j),0);
  //     mb.at(i,j) = (i+j)%25/25.0;
  //   }
  // }
  std::mt19937 gena(123);
  std::mt19937 genb(312);
  ma.random(gena);
  mb.random(genb);
  // ma.random(gen);
  // mb.random(gen);

  auto mc = ma.mul_check(mb);
  // ma.print();
  // mb.print();

  ma.quant(GGML_TYPE_BF16);
  mb.quant(GGML_TYPE_BF16);

  int8_t* a = (int8_t*)std::aligned_alloc(64, K::BufferA::required_size(m, k));
  std::shared_ptr<K::BufferA> ba = std::make_shared<K::BufferA>(m, k, a);
  int8_t* b = (int8_t*)std::aligned_alloc(64, K::BufferB::required_size(n, k));
  std::shared_ptr<K::BufferB> bb = std::make_shared<K::BufferB>(n, k, b);
  float* c = (float*)std::aligned_alloc(64, K::BufferC::required_size(m, n));
  std::shared_ptr<K::BufferC> bc = std::make_shared<K::BufferC>(m, n, c);

  ba->from_mat(m, ma.quant_data<ggml_bf16_t>(), 0, 1);
  // printf("%d\n",amx::GemmKernel224Int4::BufferA::required_size(m, k));
  // for(size_t i=0;i<amx::GemmKernel224Int4::BufferA::required_size(m, k);i++){
  //   if((i*2)%k==0)
  //     printf("\n");

  //   printf("%02x ", (unsigned char)(a[i]));
  // }
  // printf("\n");

  // int nth = amx::GemmKernel224Int4::recommended_nth(n);
  bb->from_mat(mb.quant_data<ggml_bf16_t>(), 0, 1);

  // for(size_t i=0;i<amx::GemmKernel224Int4::BufferB::required_size(n, k);i++){
  //  if((i*2)%k==0)
  //     printf("\n");

  //  printf("%02x ", (unsigned char)(b[i]));
  // }
  // printf("\n");

  amx::mat_mul(m, n, k, ba, bb, bc, 0, 1);

  // for(size_t i=0;i<m;i++){
  //   for(size_t j=0;j<n;j++){
  //     printf("%.2f ",c[i*n+j]);
  //   }
  //   printf("\n");
  // }

  // printf("\n");
  Mat<float> tc(m, n, Layout::RowMajor);
  tc.data = c;
  std::cout << "AMX OUTPUT:" << std::endl;
  tc.print_all();
  std::cout << "STD OUTPUT:" << std::endl;
  mc.print_all();

  mc.cmp(tc);

  // for(size_t i=0;i<m/32;i++){
  //   for(size_t j=0;j<n/32;j++){
  //     Mat<float> stdre(32,32,Layout::RowMajor);
  //     Mat<float> amxre(32,32,Layout::RowMajor);
  //     for(size_t ii=i*32;ii<i*32+32;ii++){
  //       for(size_t jj=j*32;jj<j*32+32;jj++){
  //         stdre.at(ii-i*32,jj-j*32) = mc.at(ii,jj);
  //         amxre.at(ii-i*32,jj-j*32) = tc.at(ii,jj);
  //       }
  //     }
  //     printf("%d %d ",i,j);
  //     stdre.cmp(amxre);
  //     // if(i==0&&j==0){
  //       std::cout<<"STD"<<std::endl;
  //       stdre.print_all();
  //       std::cout<<"AMX"<<std::endl;
  //       amxre.print_all();
  //     // }
  //   }
  // }
}

void q_latency_test_int4(int m, int n, int k, ggml_bf16_t* qa, ggml_bf16_t* qb) {
  int nth = amx::GemmKernel224Int4::recommended_nth(n);
  int m_ = (m + 31) / 32 * 32;
  Mat<float> d(m_, n, Layout::RowMajor);
  {
    int repeat = 100;
    std::vector<int8_t*> vec_a;
    std::vector<int8_t*> vec_b;
    std::vector<float*> vec_c;
    std::vector<std::shared_ptr<amx::GemmKernel224Int4::BufferA>> vec_ba;
    std::vector<std::shared_ptr<amx::GemmKernel224Int4::BufferB>> vec_bb;
    std::vector<std::shared_ptr<amx::GemmKernel224Int4::BufferC>> vec_bc;
    for (int i = 0; i < repeat * 2; i++) {
      int8_t* a = (int8_t*)std::aligned_alloc(64, amx::GemmKernel224Int4::BufferA::required_size(m_, k));
      std::shared_ptr<amx::GemmKernel224Int4::BufferA> ba = std::make_shared<amx::GemmKernel224Int4::BufferA>(m_, k, a);
      int8_t* b = (int8_t*)std::aligned_alloc(64, amx::GemmKernel224Int4::BufferB::required_size(n, k));
      std::shared_ptr<amx::GemmKernel224Int4::BufferB> bb = std::make_shared<amx::GemmKernel224Int4::BufferB>(n, k, b);
      float* c = (float*)std::aligned_alloc(64, amx::GemmKernel224Int4::BufferC::required_size(m_, n));
      std::shared_ptr<amx::GemmKernel224Int4::BufferC> bc = std::make_shared<amx::GemmKernel224Int4::BufferC>(m_, n, c);
      ba->from_mat(m, qa, 0, 1);
      int nth = amx::GemmKernel224Int4::recommended_nth(n);
      for (int i = 0; i < nth; i++) {
        bb->from_mat(qb, i, nth);
      }
      vec_a.push_back(a);
      vec_b.push_back(b);
      vec_c.push_back(c);
      vec_ba.push_back(ba);
      vec_bb.push_back(bb);
      vec_bc.push_back(bc);
    }
    Timer t(fmt::format("m:{} n:{} k:{} t:{} repeat:{}, latency", m, n, k, test_iter, repeat));
    for (int t = 0; t < test_iter; t++) {
#pragma omp parallel for schedule(dynamic, 1)
      for (int ti = 0; ti < nth * repeat; ti++) {
        int mat_id = ti / nth + repeat * (t % 2);
        int ith = ti % nth;
        if (cache_hit) {
          mat_id = 0;
        }
        amx::mat_mul(m, n, k, vec_ba[mat_id], vec_bb[mat_id], vec_bc[mat_id], ith, nth);
      }
    }
    for (int i = 0; i < repeat * 2; i++) {
      free(vec_a[i]);
      free(vec_b[i]);
      free(vec_c[i]);
    }
  }
  d.dealloc();
}

void group_q_latency_test_int4(int n_max, int k_max) {
  amx::GemmKernel224Int4::config();

  int m_max = 1024;
  int m_start = 32;
  int m_step = 32;

  Mat<float> a(m_max, k_max, Layout::RowMajor), b(k_max, n_max, Layout::ColumnMajor);
  std::mt19937 gen(123);
  a.random(gen);
  b.random(gen);
  a.quant(GGML_TYPE_BF16);
  b.quant(GGML_TYPE_BF16);

  std::string method_name = "INT4";
  if (mt) {
    method_name += fmt::format("_mt{}", omp_get_max_threads());
  }
  if (cache_hit) {
    method_name += "-cache-hit";
  }

  auto output = fmt::format("{}-m:{}:{}:{}-n:{}-k:{}-x{}x{}.txt", method_name, m_start, m_max, m_step, n_max, k_max,
                            amx::GemmKernel224Int4::N_BLOCK, amx::GemmKernel224Int4::K_BLOCK);
  // std::cout << "Output to: " << output << std::endl;
  auto x = freopen(output.c_str(), "w", stdout);
  assert(x);

  for (int m = m_start; m <= m_max; m *= 2) {
    q_latency_test_int4(m, n_max, k_max, a.quant_data<ggml_bf16_t>(), b.quant_data<ggml_bf16_t>());
  }
}

void q_latency_test_int4_1(int m, int n, int k, ggml_bf16_t* qa, ggml_bf16_t* qb) {
  int nth = amx::GemmKernel224Int4_1::recommended_nth(n);
  int m_ = (m + 31) / 32 * 32;
  Mat<float> d(m_, n, Layout::RowMajor);
  {
    int repeat = 100;
    std::vector<int8_t*> vec_a;
    std::vector<int8_t*> vec_b;
    std::vector<float*> vec_c;
    std::vector<std::shared_ptr<amx::GemmKernel224Int4_1::BufferA>> vec_ba;
    std::vector<std::shared_ptr<amx::GemmKernel224Int4_1::BufferB>> vec_bb;
    std::vector<std::shared_ptr<amx::GemmKernel224Int4_1::BufferC>> vec_bc;
    for (int i = 0; i < repeat * 2; i++) {
      int8_t* a = (int8_t*)std::aligned_alloc(64, amx::GemmKernel224Int4_1::BufferA::required_size(m_, k));
      std::shared_ptr<amx::GemmKernel224Int4_1::BufferA> ba =
          std::make_shared<amx::GemmKernel224Int4_1::BufferA>(m_, k, a);
      int8_t* b = (int8_t*)std::aligned_alloc(64, amx::GemmKernel224Int4_1::BufferB::required_size(n, k));
      std::shared_ptr<amx::GemmKernel224Int4_1::BufferB> bb =
          std::make_shared<amx::GemmKernel224Int4_1::BufferB>(n, k, b);
      float* c = (float*)std::aligned_alloc(64, amx::GemmKernel224Int4_1::BufferC::required_size(m_, n));
      std::shared_ptr<amx::GemmKernel224Int4_1::BufferC> bc =
          std::make_shared<amx::GemmKernel224Int4_1::BufferC>(m_, n, c);
      ba->from_mat(m, qa, 0, 1);
      int nth = amx::GemmKernel224Int4_1::recommended_nth(n);
      for (int i = 0; i < nth; i++) {
        bb->from_mat(qb, i, nth);
      }
      vec_a.push_back(a);
      vec_b.push_back(b);
      vec_c.push_back(c);
      vec_ba.push_back(ba);
      vec_bb.push_back(bb);
      vec_bc.push_back(bc);
    }
    Timer t(fmt::format("m:{} n:{} k:{} t:{} repeat:{}, latency", m, n, k, test_iter, repeat));
    for (int t = 0; t < test_iter; t++) {
#pragma omp parallel for schedule(dynamic, 1)
      for (int ti = 0; ti < nth * repeat; ti++) {
        int mat_id = ti / nth + repeat * (t % 2);
        int ith = ti % nth;
        if (cache_hit) {
          mat_id = 0;
        }
        amx::mat_mul(m, n, k, vec_ba[mat_id], vec_bb[mat_id], vec_bc[mat_id], ith, nth);
      }
    }
    for (int i = 0; i < repeat * 2; i++) {
      free(vec_a[i]);
      free(vec_b[i]);
      free(vec_c[i]);
    }
  }
  d.dealloc();
}

void group_q_latency_test_int4_1(int n_max, int k_max) {
  amx::GemmKernel224Int4_1::config();

  int m_max = 1024;
  int m_start = 32;
  int m_step = 32;

  Mat<float> a(m_max, k_max, Layout::RowMajor), b(k_max, n_max, Layout::ColumnMajor);
  std::mt19937 gen(123);
  a.random(gen);
  b.random(gen);
  a.quant(GGML_TYPE_BF16);
  b.quant(GGML_TYPE_BF16);

  std::string method_name = "INT4_1";
  if (mt) {
    method_name += fmt::format("_mt{}", omp_get_max_threads());
  }
  if (cache_hit) {
    method_name += "-cache-hit";
  }

  auto output = fmt::format("{}-m:{}:{}:{}-n:{}-k:{}-x{}x{}.txt", method_name, m_start, m_max, m_step, n_max, k_max,
                            amx::GemmKernel224Int4_1::N_BLOCK, amx::GemmKernel224Int4_1::K_BLOCK);
  // std::cout << "Output to: " << output << std::endl;
  auto x = freopen(output.c_str(), "w", stdout);
  assert(x);

  for (int m = m_start; m <= m_max; m *= 2) {
    q_latency_test_int4_1(m, n_max, k_max, a.quant_data<ggml_bf16_t>(), b.quant_data<ggml_bf16_t>());
  }
}

int main() {
  amx::enable_amx();
  init();

  // group_q_latency_test_bf16(5120, 1536);
  // group_q_latency_test_bf16(3584, 2560);
  // group_q_latency_test_bf16(2560, 3584);
  // group_q_latency_test_bf16(1536, 5120);
  // group_q_latency_test_bf16(7168, 2048);
  // group_q_latency_test_bf16(2048, 7168);

  // group_q_latency_test_int8(5120, 1536);
  // group_q_latency_test_int8(3584, 2560);
  // group_q_latency_test_int8(2560, 3584);
  // group_q_latency_test_int8(1536, 5120);
  // group_q_latency_test_int8(7168, 2048);
  // group_q_latency_test_int8(2048, 7168);

  group_q_latency_test_int4(5120, 1536);
  group_q_latency_test_int4(3584, 2560);
  group_q_latency_test_int4(2560, 3584);
  group_q_latency_test_int4(1536, 5120);
  group_q_latency_test_int4(7168, 2048);
  group_q_latency_test_int4(2048, 7168);

  // group_q_latency_test_int4_1(5120, 1536);
  // group_q_latency_test_int4_1(3584, 2560);
  // group_q_latency_test_int4_1(2560, 3584);
  // group_q_latency_test_int4_1(1536, 5120);
  // group_q_latency_test_int4_1(7168, 2048);
  // group_q_latency_test_int4_1(2048, 7168);

  // int k = 2048;
  // correction_test_int4_1(32, 32, k);
  // correction_test_int4(256, 256, 2048);
  // correction_test_int4(32, 32, 4096);
  // correction_test_int4(256, 256, 4096);
  // correction_test_int4(32, 32, k);
  // correction_test_int4(256, 32, 128);
  // correction_test_int4(32, 64, 128);
  // correction_test_int4(64, 32, 128);
  // correction_test_int4(256, 256, 128);
}
