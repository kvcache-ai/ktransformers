#include <malloc.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

/* matrix saved in rows or cols */
typedef enum CBLAS_ORDER { CblasRowMajor = 101, CblasColMajor = 102 } CBLAS_ORDER;

/* matrix transpose or conjugate transpose */
typedef enum CBLAS_TRANSPOSE {
  CblasNoTrans = 111,
  CblasTrans = 112,
  CblasConjTrans = 113,  // conjugate transpose
  CblasConjNoTrans = 114
} CBLAS_TRANSPOSE;

typedef CBLAS_ORDER CBLAS_LAYOUT;

typedef enum CBLAS_OFFSET { CblasRowOffset = 171, CblasColOffset = 172, CblasFixOffset = 173 } CBLAS_OFFSET;

typedef int8_t BLASINT8;
typedef uint8_t BLASUINT8;

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

void cblas_gemm_s8s8s32(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                        const CBLAS_OFFSET offsetc, const size_t m, const size_t n, const size_t k, const float alpha,
                        const void* a, const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb,
                        const BLASINT8 ob, const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void cblas_gemm_u8u8s32(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                        const CBLAS_OFFSET offsetc, const size_t m, const size_t n, const size_t k, const float alpha,
                        const void* a, const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb,
                        const BLASINT8 ob, const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void cblas_gemm_s8u8s32(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                        const CBLAS_OFFSET offsetc, const size_t m, const size_t n, const size_t k, const float alpha,
                        const void* a, const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb,
                        const BLASINT8 ob, const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

void cblas_gemm_u8s8s32(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                        const CBLAS_OFFSET offsetc, const size_t m, const size_t n, const size_t k, const float alpha,
                        const void* a, const size_t lda, const BLASINT8 oa, const void* b, const size_t ldb,
                        const BLASINT8 ob, const float beta, int32_t* c, const size_t ldc, const int32_t* oc);

#ifdef __cplusplus
}
#endif /* __cplusplus */

namespace test {

namespace tools {

template <typename T, std::size_t alignment = 128>
struct aligned_allocator {
  using value_type = T;
  using pointer = T*;
  using const_pointer = const T*;
  using reference = T&;
  using const_reference = const T&;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;

  template <typename U>
  struct rebind {
    typedef aligned_allocator<U, alignment> other;
  };

  [[nodiscard]] T* allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::bad_array_new_length();
    if (auto p = static_cast<T*>(memalign(alignment, n * sizeof(T)))) {
      return p;
    }

    throw std::bad_alloc();
  }

  void deallocate(T* p, std::size_t n) noexcept {
    (void)(n);
    free(p);
  }

  ~aligned_allocator() {}
};
template <typename T, std::size_t alignment_1, typename U, std::size_t alignment_2>
bool operator==(const aligned_allocator<T, alignment_1>&, const aligned_allocator<U, alignment_2>&) {
  return (alignment_1 == alignment_2) && std::is_same_v<T, U>;
}

template <typename T, std::size_t alignment_1, typename U, std::size_t alignment_2>
bool operator!=(const aligned_allocator<T, alignment_1>& lhs, const aligned_allocator<U, alignment_2>& rhs) {
  return !(lhs == rhs);
}
template <typename Func, typename... Args>
double timing(Func&& func, Args&&... args) {
  double time = 0.0;
  double time_begin = 0.0;
  std::size_t n_run = 0;

  auto start_begin = std::chrono::steady_clock::now();
  std::forward<Func>(func)(std::forward<Args>(args)...);
  auto end_begin = std::chrono::steady_clock::now();

  time_begin = std::chrono::duration_cast<std::chrono::nanoseconds>(end_begin - start_begin).count() / 1e9;
  n_run = std::max<std::size_t>(std::size_t(1.0 / time_begin), 3);

  auto start = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < n_run; ++i) {
    std::forward<Func>(func)(std::forward<Args>(args)...);
  }
  auto end = std::chrono::steady_clock::now();

  time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e9;
  return time / n_run;
}
}  // namespace tools
namespace helpers {
std::size_t get_oc_size(CBLAS_OFFSET offset, std::size_t m, std::size_t n) {
  std::size_t ret_val = 0;
  switch (offset) {
    case CblasFixOffset:
      ret_val = 1;
      break;
    case CblasColOffset:
      ret_val = m;
      break;
    case CblasRowOffset:
      ret_val = n;
      break;
    default:
      std::cout << "Incorrect value of offset to the function " << __PRETTY_FUNCTION__ << std::endl;
  }
  return ret_val;
}
template <typename T>
auto get_ab_matrix(CBLAS_LAYOUT lt, CBLAS_TRANSPOSE trans_, T&& non_trans_mtx, T&& trans_mtx) {
  if (lt == CblasColMajor) {
    if (trans_ == CblasNoTrans) {
      return non_trans_mtx.data();
    } else {
      return trans_mtx.data();
    }
  } else {
    if (trans_ == CblasNoTrans) {
      return trans_mtx.data();
    } else {
      return non_trans_mtx.data();
    }
  }
}
auto get_ldab(CBLAS_LAYOUT lt, CBLAS_TRANSPOSE trans_mtx, std::size_t ld_n, std::size_t ld_t) {
  if (lt == CblasColMajor) {
    if (trans_mtx == CblasNoTrans) {
      return ld_n;
    } else {
      return ld_t;
    }
  } else {
    if (trans_mtx == CblasNoTrans) {
      return ld_t;
    } else {
      return ld_n;
    }
  }
}

// returns copy of the matrix
template <typename T>
auto get_c_matrix(CBLAS_LAYOUT lt, T&& non_trans_mtx, T&& trans_mtx) {
  if (lt == CblasColMajor) {
    return non_trans_mtx;
  } else {
    return trans_mtx;
  }
}

auto get_ldc(CBLAS_LAYOUT lt, std::size_t ldc_n, std::size_t ldc_t) {
  if (lt == CblasColMajor) {
    return ldc_n;
  } else {
    return ldc_t;
  }
}
template <typename A_Type, typename B_Type>
void cblas_gemm_wrapper(const CBLAS_LAYOUT Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb,
                        const CBLAS_OFFSET offsetc, const size_t m, const size_t n, const size_t k, const float alpha,
                        const A_Type* a, const size_t lda, const int8_t oa, const B_Type* b, const size_t ldb,
                        const int8_t ob, const float beta, int32_t* c, const size_t ldc, const int32_t* oc) {
  if constexpr (std::is_same_v<A_Type, std::int8_t>) {
    if constexpr (std::is_same_v<B_Type, std::int8_t>) {
      cblas_gemm_s8s8s32(Layout, transa, transb, offsetc, m, n, k, alpha, a, lda, oa, b, ldb, ob, beta, c, ldc, oc);
    } else {
      cblas_gemm_s8u8s32(Layout, transa, transb, offsetc, m, n, k, alpha, a, lda, oa, b, ldb, ob, beta, c, ldc, oc);
    }
  } else {
    if constexpr (std::is_same_v<B_Type, std::int8_t>) {
      cblas_gemm_u8s8s32(Layout, transa, transb, offsetc, m, n, k, alpha, a, lda, oa, b, ldb, ob, beta, c, ldc, oc);
    } else {
      cblas_gemm_u8u8s32(Layout, transa, transb, offsetc, m, n, k, alpha, a, lda, oa, b, ldb, ob, beta, c, ldc, oc);
    }
  }
}

std::size_t return_oc_idx(const CBLAS_OFFSET offsetc, std::size_t mi, std::size_t ni) {
  return (offsetc == CblasFixOffset) ? 0 : ((offsetc == CblasColOffset) ? mi : ni);
}
}  // namespace helpers
enum class status_t { passed, failed };

std::ostream& operator<<(std::ostream& os, const status_t& st) {
  if (status_t::passed == st) {
    os << "PASSED";
  } else if (status_t::failed == st) {
    os << "FAILED";
  }
  return os;
}
// column major
template <typename A_Type, typename B_Type>
void ref_gemm(const CBLAS_OFFSET offsetc, const std::size_t m, const std::size_t n, const std::size_t k,
              const float alpha, const A_Type* a, const std::size_t lda, const std::int8_t oa, const B_Type* b,
              const std::size_t ldb, const std::int8_t ob, const float beta, std::int32_t* c, const std::size_t ldc,
              const std::int32_t* oc) {
  for (std::size_t mi = 0; mi < m; ++mi) {
    for (std::size_t ni = 0; ni < n; ++ni) {
      std::int32_t tmp = 0;
      for (std::size_t ki = 0; ki < k; ++ki) {
        tmp += (a[mi + ki * lda] + oa) * (b[ki + ni * ldb] + ob);
      }
      c[mi + ni * ldc] = std::round(alpha * static_cast<double>(tmp) +
                                    static_cast<double>(beta * static_cast<float>(c[mi + ni * ldc])) +
                                    static_cast<float>(oc[helpers::return_oc_idx(offsetc, mi, ni)]));
    }
  }
}
template <typename DataType>
void fill_random(DataType* buffer, std::size_t len) {
  static std::mt19937 generator(0);
  std::uniform_int_distribution<DataType> dist(0, 64);
  for (std::size_t i = 0; i < len; i++) {
    buffer[i] = static_cast<DataType>(dist(generator));
  }
}

template <typename DataType>
void fill_const(DataType* buffer, std::size_t len) {
  for (std::size_t i = 0; i < len; i++) {
    buffer[i] = DataType{-8};
  }
}
// performs transposition (n0 * n1) -> (n1 * n0), assuming col major
template <typename T>
void simplest_transpose(T* in, T* out, std::size_t n0, std::size_t n1, std::size_t ld0, std::size_t ld1) {
  for (std::size_t i = 0; i < n0; ++i) {
    for (std::size_t j = 0; j < n1; ++j) {
      out[i + j * ld1] = in[j + i * ld0];
    }
  }
}

template <typename DataType>
status_t compare(DataType* ref, DataType* test, std::size_t m, std::size_t n, std::size_t ld) {
  for (std::size_t mi = 0; mi < m; ++mi) {
    for (std::size_t ni = 0; ni < n; ++ni) {
      if (ref[mi + ni * ld] != test[mi + ni * ld]) {
        return status_t::failed;
      }
    }
  }
  return status_t::passed;
}
template <typename DataType>
void print_matrix(DataType* buffer, std::size_t m, std::size_t n) {
  for (std::size_t mi = 0; mi < m; ++mi) {
    for (std::size_t ni = 0; ni < n; ++ni) {
      std::cout << static_cast<std::int32_t>(buffer[mi + ni * m]) << ", ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}
template <typename A_Type, typename B_Type>
status_t gemm(std::size_t m, std::size_t n, std::size_t k, float alpha, float beta) {
  std::int8_t oa = 4;
  std::int8_t ob = 9;

  std::size_t lda_n = m;
  std::size_t ldb_n = k;
  std::size_t ldc_n = m;

  std::size_t lda_t = k;
  std::size_t ldb_t = n;
  std::size_t ldc_t = n;

  if (std::getenv("LD_STRIDE")) {
    lda_n += 2;
    ldb_n += 7;
    ldc_n += 3;

    lda_t += 8;
    ldb_t += 3;
    ldc_t += 23;
  }

  bool only_performance = false;
  if (std::getenv("ONLY_PERF")) {
    only_performance = true;
  }
  std::vector<A_Type, tools::aligned_allocator<A_Type, 128>> a_n(lda_n * k);
  std::vector<A_Type, tools::aligned_allocator<A_Type, 128>> a_t(m * lda_t);
  std::vector<B_Type, tools::aligned_allocator<B_Type, 128>> b_n(ldb_n * n);
  std::vector<B_Type, tools::aligned_allocator<B_Type, 128>> b_t(k * ldb_t);
  std::vector<std::int32_t, tools::aligned_allocator<std::int32_t, 128>> c_ref(ldc_n * n);
  std::vector<std::int32_t, tools::aligned_allocator<std::int32_t, 128>> c_n(ldc_n * n);
  std::vector<std::int32_t, tools::aligned_allocator<std::int32_t, 128>> c_t(m * ldc_t);

  // fill the whole array even if ld* > corresponding dim
  fill_random(a_n.data(), a_n.size());
  fill_random(b_n.data(), b_n.size());

  simplest_transpose(a_n.data(), a_t.data(), m, k, lda_n, lda_t);
  simplest_transpose(b_n.data(), b_t.data(), k, n, ldb_n, ldb_t);

  fill_const(c_ref.data(), ldc_n * n);
  c_n = c_ref;

  simplest_transpose(c_n.data(), c_t.data(), m, ldc_n, n, ldc_t);

  auto return_st = status_t::passed;
  double total = 0;
  size_t cnt = 0;
  for (auto c_offset : {CblasFixOffset, CblasColOffset, CblasRowOffset}) {
    std::vector<std::int32_t, tools::aligned_allocator<std::int32_t, 128>> oc(helpers::get_oc_size(c_offset, m, n));
    for (std::size_t i = 0; i < oc.size(); ++i) {
      oc[i] = i + i;
    }

    std::vector<std::int32_t, tools::aligned_allocator<std::int32_t, 128>> c_ref_copy = c_ref;
    if (!only_performance) {
      ref_gemm(c_offset, m, n, k, alpha, a_n.data(), lda_n, oa, b_n.data(), ldb_n, ob, beta, c_ref_copy.data(), ldc_n,
               oc.data());
    }
    for (auto layout : {CblasColMajor, CblasRowMajor}) {
      for (auto transa : {CblasNoTrans, CblasTrans}) {
        for (auto transb : {CblasNoTrans, CblasTrans}) {
          auto&& c_tested = helpers::get_c_matrix(layout, c_n, c_t);
          if (!only_performance) {
            helpers::cblas_gemm_wrapper(
                layout, transa, transb, c_offset, m, n, k, alpha, helpers::get_ab_matrix(layout, transa, a_n, a_t),
                helpers::get_ldab(layout, transa, lda_n, lda_t), oa, helpers::get_ab_matrix(layout, transb, b_n, b_t),
                helpers::get_ldab(layout, transb, ldb_n, ldb_t), ob, beta, c_tested.data(),
                helpers::get_ldc(layout, ldc_n, ldc_t), oc.data());

            // transpose c_tested to col-major if required
            auto loc_st = status_t::passed;
            if (layout == CblasRowMajor) {
              std::vector<std::int32_t, tools::aligned_allocator<std::int32_t, 128>> c_tested_n(ldc_n * n);
              simplest_transpose(c_tested.data(), c_tested_n.data(), n, ldc_t, m, ldc_n);
              loc_st = compare(c_ref_copy.data(), c_tested_n.data(), m, n, ldc_n);
            } else {
              loc_st = compare(c_ref_copy.data(), c_tested.data(), m, n, ldc_n);
            }
            if (loc_st != status_t::passed) {
              std::cout << "-";
              return_st = status_t::failed;
            } else {
              std::cout << "+";
            }
          } else {
            double cur = (2.0 * m * n * k) /
                         tools::timing(helpers::cblas_gemm_wrapper<A_Type, B_Type>, layout, transa, transb, c_offset, m,
                                       n, k, alpha, helpers::get_ab_matrix(layout, transa, a_n, a_t),
                                       helpers::get_ldab(layout, transa, lda_n, lda_t), oa,
                                       helpers::get_ab_matrix(layout, transb, b_n, b_t),
                                       helpers::get_ldab(layout, transb, ldb_n, ldb_t), ob, beta, c_tested.data(),
                                       helpers::get_ldc(layout, ldc_n, ldc_t), oc.data()) /
                         1e12;
            total += cur;
            ++cnt;

            std::cout << cur << ", ";
          }
        }
      }
    }
  }
  if (only_performance) {
    std::cout << "Average " << total / cnt << " TFlops";
  }
  std::cout << "    ";
  return return_st;
}
}  // namespace test

int main(int argc, char** argv) {
  std::size_t m = 128;
  std::size_t n = 128;
  std::size_t k = 128;
  float alpha = 1.0f;
  float beta = 1.0f;

  if (argc > 1) {
    m = std::stoi(argv[1]);
    if (argc > 2) {
      n = std::stoi(argv[2]);
      if (argc > 3) {
        k = std::stoi(argv[3]);
        if (argc > 4) {
          alpha = std::stof(argv[4]);
          if (argc > 5) {
            beta = std::stof(argv[5]);
          }
        }
      }
    }
  }
  std::cout << "Testing matrix m = " << m << ", n = " << n << ", k = " << k << ", alpha = " << alpha
            << ", beta = " << beta << std::endl;

  std::cout << "\tTesting i8i8i32: " << test::gemm<std::int8_t, std::int8_t>(m, n, k, alpha, beta) << std::endl;
  std::cout << "\tTesting i8u8i32: " << test::gemm<std::int8_t, std::uint8_t>(m, n, k, alpha, beta) << std::endl;
  std::cout << "\tTesting u8i8i32: " << test::gemm<std::uint8_t, std::int8_t>(m, n, k, alpha, beta) << std::endl;
  std::cout << "\tTesting u8u8i32: " << test::gemm<std::uint8_t, std::uint8_t>(m, n, k, alpha, beta) << std::endl;

  return 0;
}