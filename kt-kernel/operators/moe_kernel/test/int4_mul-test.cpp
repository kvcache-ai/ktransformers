#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "../la/arm_kml.hpp"
#include "debug.hpp"
#include "kblas.h"
const int M = 1, K = 7168, N = 8;

int main() {
  // 随机生成a, b, c矩阵
  arm_kml::GemmKernelInt4::BufferA buffer_a(M, K);
  arm_kml::GemmKernelInt4::BufferB buffer_b(N, K, true);
  arm_kml::GemmKernelInt4::BufferC buffer_c(M, N);

  arm_kml::GemmKernelInt8::BufferA buffer_a_check(M, K);
  arm_kml::GemmKernelInt8::BufferB buffer_b_check(N, K, true);
  arm_kml::GemmKernelInt8::BufferC buffer_c_check(M, N);

  float* a = (float*)aligned_alloc(64, sizeof(float) * M * K);
  float* b = (float*)aligned_alloc(64, sizeof(float) * K * N);
  float* c = (float*)aligned_alloc(64, sizeof(float) * M * N);
  float* c_check = (float*)aligned_alloc(64, sizeof(float) * M * N);
  int8_t* buffer_a_data = (int8_t*)aligned_alloc(64, buffer_a.required_size());
  int4_2_t* buffer_b_data = (int4_2_t*)aligned_alloc(64, buffer_b.required_size());
  int32_t* c_data = (int32_t*)aligned_alloc(64, buffer_c.required_size());
  int8_t* buffer_a_data_check = (int8_t*)aligned_alloc(64, buffer_a_check.required_size());
  int8_t* buffer_b_data_check = (int8_t*)aligned_alloc(64, buffer_b_check.required_size());
  int32_t* c_data_check = (int32_t*)aligned_alloc(64, buffer_c_check.required_size());
  // 初始化元素内容
  load_bin("input.bin", a, M * K);
  load_bin("local_q_a_proj_quant.bin", b, N * K);

  // for (int i = 0; i < M * K; i++) {
  //   // 随机浮点数
  //   // a[i] = (static_cast<float>(rand()) / (float)RAND_MAX) / 25 - 0.02;
  //   a[i] = -(static_cast<float>(rand()) / (float)RAND_MAX) / 25;
  //   // a[i] = i % 10;
  //   // a[i] = 1;
  // }
  // for (int i = 0; i < K * N; i++) {
  //   // 随机浮点数
  //   // b[i] = (static_cast<float>(rand()) / (float)RAND_MAX) / 25 - 0.02;
  //   b[i] = -(static_cast<float>(rand()) / (float)RAND_MAX) / 25;
  //   // b[i] = i % 10;
  //   // b[i] = 1;
  // }
  // // // // 设置离群值
  // for (int i = 0; i < N; i++) {
  //   b[i * K] = 0.06f; // 设置第一列为离群值
  // }
  // // 打印一下输入矩阵和权重矩阵
  // printf("Input matrix a:\n");
  // for (int i = 0; i < M; i++) {
  //   for (int j = 0; j < K; j++) {
  //     printf("%f ", a[i * K + j]);
  //   }
  //   printf("\n");
  // }
  // printf("Weight matrix b:\n");
  // for (int i = 0; i < N; i++) {
  //   for (int j = 0; j < K; j++) {
  //     printf("%f ", b[i * K + j]);
  //   }
  //   printf("\n");
  // }
  buffer_a.set_data(buffer_a_data);
  buffer_b.set_data(buffer_b_data);
  buffer_c.set_data(c_data);
  buffer_a_check.set_data(buffer_a_data_check);
  buffer_b_check.set_data(buffer_b_data_check);
  buffer_c_check.set_data(c_data_check);
  //   调用 from mat 进行量化
  buffer_a.from_mat(M, a, 0, M);
  for (int i = 0; i <= arm_kml::GemmKernelInt4::recommended_nth(N); i++) {
    buffer_b.from_mat(b, i, arm_kml::GemmKernelInt4::recommended_nth(N));
  }
  buffer_a_check.from_mat(M, a, 0, M);
  for (int i = 0; i <= arm_kml::GemmKernelInt8::recommended_nth(N); i++) {
    buffer_b_check.from_mat(b, i, arm_kml::GemmKernelInt8::recommended_nth(N));
  }
  // 进行乘法
  arm_kml::MatRef<int8_t> a_ref(buffer_a.a, M, K, K, CblasRowMajor);
  arm_kml::MatRef<int4_2_t> b_ref(buffer_b.b, K, N, K, CblasColMajor, CblasNoTrans, buffer_b.if_pack);
  arm_kml::MatRef<int32_t> c_ref(buffer_c.c, M, N, N, CblasRowMajor);
  b_ref = b_ref.offset_col(0, N);

  arm_kml::MatRef<int8_t> a_ref_check(buffer_a_check.a, M, K, K, CblasRowMajor);
  arm_kml::MatRef<int8_t> b_ref_check(buffer_b_check.b, K, N, K, CblasColMajor, CblasNoTrans, buffer_b_check.if_pack);
  arm_kml::MatRef<int32_t> c_ref_check(buffer_c_check.c, M, N, N, CblasRowMajor);

  arm_kml::decode_mul_mat_clearc(a_ref, b_ref, c_ref);
  arm_kml::decode_mul_mat_clearc(a_ref_check, b_ref_check, c_ref_check);
  //   反量化，apply scale
  arm_kml::GemmKernelInt4::apply_scale(c, N, &buffer_a, &buffer_b, &buffer_c, 0, M, 0, N, true);
  arm_kml::GemmKernelInt8::apply_scale(c_check, N, &buffer_a_check, &buffer_b_check, &buffer_c_check, 0, M, 0, N, true);
  // 打印结果,比较 c 和 c_check
  const float threashold = 0.05;
  for (int i = 0; i < M * N; i++) {
    float diff_relative = (c[i] - c_check[i]) / (c_check[i] + 1e-6);

    if (diff_relative > threashold || diff_relative < -threashold) {
      printf("diff_relative: %f\n", diff_relative);
      printf("Mismatch at index %d: c = %f, c_check = %f\n", i, c[i], c_check[i]);
    } else {
      printf("Match at index %d: c = %f, c_check = %f\n", i, c[i], c_check[i]);
    }
  }
  return 0;
}