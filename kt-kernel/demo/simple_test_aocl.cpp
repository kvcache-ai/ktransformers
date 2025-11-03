#include <blis.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
// #define CHECK
namespace {
// B matrix is in col-major order
constexpr int kM = 3;
constexpr int kK = 7168;
constexpr int kN = 2048;
void fill_inputs(int8_t* a, int8_t* b) {
  srand(static_cast<unsigned>(time(nullptr)));
  for (int i = 0; i < kM * kK; ++i) {
    a[i] = static_cast<int8_t>(rand() % 127);
  }
  for (int i = 0; i < kK * kN; ++i) {
    b[i] = static_cast<int8_t>(rand() % 127);
  }
}

void compute_reference(const int8_t* a, const int8_t* b, int32_t* ref) {
  for (int m = 0; m < kM; ++m) {
    for (int n = 0; n < kN; ++n) {
      int32_t acc = 0;
      for (int k = 0; k < kK; ++k) {
        acc += static_cast<int32_t>(a[m * kK + k]) * static_cast<int32_t>(b[k * kN + n]);
      }
      ref[m * kN + n] = acc;
    }
  }
}

bool check_result(const int32_t* got, const int32_t* ref) {
  for (int idx = 0; idx < kM * kN; ++idx) {
    if (got[idx] != ref[idx]) {
      std::printf("Mismatch at %d: got %d, expected %d\n", idx, got[idx], ref[idx]);
      return false;
    }
  }
  return true;
}
}  // namespace

int main() {
  err_t err = BLIS_SUCCESS;
  int8_t* a = static_cast<int8_t*>(bli_malloc_user(kM * kK, &err));
  int8_t* b = static_cast<int8_t*>(bli_malloc_user(kK * kN, &err));
  int8_t* b_rowmajor = static_cast<int8_t*>(bli_malloc_user(kK * kN, &err));
  int8_t* b_reordered = nullptr;
  int32_t* c = static_cast<int32_t*>(bli_malloc_user(kM * kN * sizeof(int32_t), &err));
  int32_t* c_unp = static_cast<int32_t*>(bli_malloc_user(kM * kN * sizeof(int32_t), &err));
  int32_t* ref = static_cast<int32_t*>(bli_malloc_user(kM * kN * sizeof(int32_t), &err));

  if (!a || !b || !c || !ref || !c_unp) {
    std::fprintf(stderr, "Allocation failed\n");
    bli_free_user(a);
    bli_free_user(b);
    bli_free_user(c);
    bli_free_user(ref);
    bli_free_user(c_unp);
    return EXIT_FAILURE;
  }

  fill_inputs(a, b);
  // transform B from col-major to row-major
  for (int k = 0; k < kK; ++k) {
    for (int n = 0; n < kN; ++n) {
      // original B is in col-major: b[n * ld + k], here ld = kK
      int8_t val = b[n * kK + k];
      // target row-major: row index = k, col index = n
      b_rowmajor[k * kN + n] = val;
    }
  }
#ifdef CHECK
  // CHECK: printf inputs
  std::puts("\nMatrix A:\n");
  for (int m = 0; m < kM; ++m) {
    for (int k = 0; k < kK; ++k) {
      std::printf("%4d ", a[m * kK + k]);
    }
    std::puts("");
  }
  std::puts("\nMatrix B:\n");
  for (int k = 0; k < kK; ++k) {
    for (int n = 0; n < kN; ++n) {
      std::printf("%4d ", b[n * kK + k]);
    }
    std::puts("");
  }
#endif
  std::memset(c, 0, kM * kN * sizeof(int32_t));
  std::memset(c_unp, 0, kM * kN * sizeof(int32_t));
  std::memset(ref, 0, kM * kN * sizeof(int32_t));
  compute_reference(a, b_rowmajor, ref);
#ifdef CHECK
  // CHECK: printf reference
  std::puts("\nReference result:\n");
  for (int m = 0; m < kM; ++m) {
    for (int n = 0; n < kN; ++n) {
      std::printf("%6d ", ref[m * kN + n]);
    }
    std::puts("");
  }
#endif
  const dim_t reorder_size = aocl_get_reorder_buf_size_s8s8s32os32('c', 'n', 'B', kK, kN);
  b_reordered = static_cast<int8_t*>(bli_malloc_user(reorder_size, &err));
  if (!b_reordered) {
    std::fprintf(stderr, "Reorder buffer allocation failed\n");
    bli_free_user(a);
    bli_free_user(b);
    bli_free_user(c);
    bli_free_user(ref);
    return EXIT_FAILURE;
  }
  aocl_reorder_s8s8s32os32('c', 'n', 'B', b, b_reordered, kK, kN, kK);
#ifdef CHECK
  // CHECK: printf reordered B
  std::puts("\nReordered Matrix B:\n");
  for (int k = 0; k < kK; ++k) {
    for (int n = 0; n < kN; ++n) {
      std::printf("%4d ", b_reordered[k * kN + n]);
    }
    std::puts("");
  }
  std::printf("\nReorder buffer size: %zu bytes\n", reorder_size);
#endif

  const int32_t alpha = 1;
  const int32_t beta = 0;
  aocl_gemm_s8s8s32os32('r', 'n', 't', kM, kN, kK, alpha, a, kK, 'n', b_reordered, kK, 'r', beta, c, kN, nullptr);
  aocl_gemm_s8s8s32os32('r', 'n', 't', kM, kN, kK, alpha, a, kK, 'n', b, kK, 'n', beta, c_unp, kN, nullptr);
#ifdef CHECK
  // CHECK: printf AOCL result
  std::puts("\nAOCL GEMM result (with reordered B):\n");
  for (int m = 0; m < kM; ++m) {
    for (int n = 0; n < kN; ++n) {
      std::printf("%6d ", c[m * kN + n]);
    }
    std::puts("");
  }
  std::puts("\nAOCL GEMM result (without reordered B):\n");
  for (int m = 0; m < kM; ++m) {
    for (int n = 0; n < kN; ++n) {
      std::printf("%6d ", c_unp[m * kN + n]);
    }
    std::puts("");
  }
#endif

  if (check_result(c, ref)) {
    std::puts("AOCL GEMM output matches reference.");
  } else {
    std::puts("AOCL GEMM output mismatch detected.");
  }

  if (check_result(c_unp, ref)) {
    std::puts("unpack AOCL GEMM output matches reference.");
  } else {
    std::puts("unpack AOCL GEMM output mismatch detected.");
  }

  bli_free_user(a);
  bli_free_user(b);
  bli_free_user(b_rowmajor);
  bli_free_user(b_reordered);
  bli_free_user(c);
  bli_free_user(c_unp);
  bli_free_user(ref);
  return 0;
}