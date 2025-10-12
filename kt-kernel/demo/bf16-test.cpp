#define BGEMM

#include <arm_sve.h>
#include <dlfcn.h>
#include <kblas.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

int main() {
  // 矩阵维度 M 是 1024，K 是 1024，N 是 1024（行主序）
  int M = 512;         // 行主序时，A 的行长度为 K
  const int K = 7168;  // B 的行长度为 N
  const int N = 512;   // C 的行长度为 N
  const int iter = 1;  // 迭代次数
  // int M = 10;        // 行主序时，A 的行长度为 K
  // const int K = 10; // B 的行长度为 N
  // const int N = 10;  // C 的行长度为 N

  // 分配矩阵内存
  bfloat16_t* A = new bfloat16_t[M * K];
  bfloat16_t* B = new bfloat16_t[K * N];
  bfloat16_t* C = new bfloat16_t[M * N];
  srand(123);

  // 初始化随机种子
  // std::mt19937 rng(124);
  // std::uniform_real_distribution <float> dist(0.0, 1.0);

  for (int j = 0; j < M * K; j++) {
    A[j] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    // A[j] = dist(rng);
    // A[j] = j;
  }
  for (int j = 0; j < K * N; j++) {
    B[j] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    // B[j] = dist(rng);
    // B[j] = j;
  }
  for (int j = 0; j < M * N; j++) {
    C[j] = 0.0;
  }

  // 设置 cblas_gemm_s8u8s32 的参数
  float alpha = 1.0f;
  float beta = 0.0f;

  // 打印矩阵 A、B
  // printf("A=\n");
  // for (int i = 0; i < M; i++) {
  //   for (int j = 0; j < K; j++) {
  //     printf("%f ", A[i * K + j]);
  //   }
  //   printf("\n");
  // }
  // printf("B=\n");
  // for (int i = 0; i < N; i++) {
  //   for (int j = 0; j < K; j++) {
  //     printf("%f ", B[i * K + j]);
  //   }
  //   printf("\n");
  // }
  // cblas_shgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, C, N);
  // // 打印结果
  // printf("C=\n");
  // for (int i = 0; i < M; i++) {
  //   for (int j = 0; j < N; j++) {
  //     printf("%f ", C[i * N + j]);
  //   }
  //   printf("\n");
  // }
  // return 0;

  auto fout = fopen("test.out", "w");
  int stride = 16;
  for (int n = stride; n <= N; n += stride)
    for (int m = stride; m <= M; m += stride) {
      // 记录开始时间
      auto start = std::chrono::high_resolution_clock::now();
      // #pragma GCC unroll 8
      for (int i = 0; i < iter; i++) {
        cblas_bgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, K, alpha, A, K, B, N, beta, C, N);
        // cblas_gemm_s8s8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset, m, N, K, alpha, A, K, oa, B, K,
        // ob,
        //  beta, C, N, &oc);
      }

      // 打印结果
      // printf("result:\n");
      // for (int i = 0; i < M; i++) {
      //   for (int j = 0; j < N; j++) {
      //     printf("%f ", C[i * N + j]);
      //   }
      //   printf("\n");
      // }
      // return 0;

      // 记录结束时间
      auto end = std::chrono::high_resolution_clock::now();

      // 计算总时长（秒）
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
      double time_sec = duration.count() / 1e6;  // 转换为秒

      // 计算理论浮点运算次数并转换为 TFLOPS
      double ops = iter * 2.0 * m * n * K;
      double tflops = ops / (duration.count() * 1e6);  // 转换为 TFLOPS

      // 输出结果
      printf("execute end time %f us, m n:%d %d\n", time_sec * 1e6, m, n);
      // printf("执行时间: %.4f 秒\n", time_sec);
      printf("计算性能: %.4f TFLOPS\n", tflops);
      printf("\n");

      fprintf(fout, "%d %d %f\n", m, n, tflops);
    }

  // 释放资源
  free(A);
  free(B);
  free(C);
  return 0;
}