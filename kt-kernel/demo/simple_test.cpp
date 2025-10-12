#include <dlfcn.h>
#include <kblas.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main() {
  // 矩阵维度 M 是 1024，K 是 1024，N 是 1024（行主序）
  int M = 1024;        // 行主序时，A 的行长度为 K
  const int K = 1024;  // B 的行长度为 N
  const int N = 1024;  // C 的行长度为 N
  const int iter = 1;  // 迭代次数

  // 分配矩阵内存
  int8_t* A = (int8_t*)malloc(M * K * sizeof(int8_t));
  int8_t* B = (int8_t*)malloc(K * N * sizeof(int8_t));
  int32_t* C = (int32_t*)malloc(M * N * sizeof(int32_t));

  // 初始化随机种子
  srand((unsigned)time(NULL));

  // 随机初始化 A（范围 0 到 255）和 B（范围 -128 到 127）
  // 初始化矩阵 A 和 B
  for (int j = 0; j < M * K; j++) {
    // A[j] = rand() % 256;
    A[j] = j;
  }
  for (int j = 0; j < K * N; j++) {
    // B[j] = rand() % 256;
    B[j] = j;
  }
  // 初始化矩阵 C
  for (int j = 0; j < M * N; j++) {
    C[j] = 0;
  }

  // 设置 cblas_gemm_s8u8s32 的参数
  float alpha = 1.0f;
  float beta = 0.0f;
  int8_t oa = 0, ob = 0;
  int32_t oc = 0;

  // 打印矩阵 A、B
  // printf("A=\n");
  // for (int i = 0; i < M; i++) {
  //   for (int j = 0; j < K; j++) {
  //     printf("%d ", A[i * K + j]);
  //   }
  //   printf("\n");
  // }
  // printf("B=\n");
  // for (int i = 0; i < N; i++) {
  //   for (int j = 0; j < K; j++) {
  //     printf("%d ", B[i * K + j]);
  //   }
  //   printf("\n");
  // }

  // printf("format: 'generate end'\n");
  // 调用 cblas_gemm_s8u8s32 执行矩阵乘法：C = i1(A+ao)(B+bo) + 0*C + oc
  // 从m=10～256 都测一遍速度，步长是 stride
  int stride = 2;
  int start_m = M;
  for (int m = start_m; m <= M; m += stride) {
    // 记录开始时间
    auto start = std::chrono::high_resolution_clock::now();
#pragma GCC unroll 8
    for (int i = 0; i < iter; i++) {
      cblas_gemm_s8s8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset, m, N / 2, K, alpha, A, K, oa, B, K,
                         ob, beta, C, N, &oc);
      int8_t* B_high = B + K * N / 2;
      int32_t* C_high = C + N / 2;
      cblas_gemm_s8s8s32(CblasRowMajor, CblasNoTrans, CblasTrans, CblasFixOffset, m, N / 2, K, alpha, A, K, oa, B_high,
                         K, ob, beta, C_high, N, &oc);
    }

    // 打印结果
    // printf("result:\n");
    // for (int i = 0; i < M; i++) {
    //   for (int j = 0; j < N; j++) {
    //     printf("%d ", C[i * N + j]);
    //   }
    //   printf("\n");
    // }

    // 记录结束时间
    auto end = std::chrono::high_resolution_clock::now();

    // 计算总时长（秒）
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double time_sec = duration.count() / 1e6;  // 转换为秒

    // 计算理论浮点运算次数并转换为 TFLOPS
    double ops = iter * 2.0 * m * N * K;
    double tflops = ops / (duration.count() * 1e6);  // 转换为 TFLOPS

    // 输出结果
    printf("execute end,m is:%d\n", m);
    // printf("执行时间: %.4f 秒\n", time_sec);
    printf("计算性能: %.4f TFLOPS\n", tflops);
    printf("\n");
  }

  // 释放资源
  free(A);
  free(B);
  free(C);
  return 0;
}