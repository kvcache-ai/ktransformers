// SPDX-License-Identifier: Apache-2.0
#ifndef KT_SYCL_GPTQ_INT4_XMX_KERNELS_HPP
#define KT_SYCL_GPTQ_INT4_XMX_KERNELS_HPP

#include <cstring>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>

#include "gptq_int4_sycl-runtime.hpp"

namespace sycl_xmx {
namespace es = sycl::ext::intel::esimd;
namespace xmx = es::xmx;

// One tile belongs to one expert. Rows are indices in the routed workspace.
struct Tile {
  int expert;
  int begin;
  int rows;
};

template <typename T>
class USM {
 public:
  T* data = nullptr;
  size_t capacity = 0;
  USM() = default;
  USM(const USM&) = delete;
  USM& operator=(const USM&) = delete;
  ~USM() { sycl_int4::usm_free(data); }
  void reserve(size_t n) {
    if (n <= capacity) return;
    T* replacement = sycl_int4::usm_alloc<T>(n, "XMX workspace");
    sycl_int4::usm_free(data);
    data = replacement;
    capacity = n;
  }
};

struct Weight {
  USM<uint32_t> packed;  // [expert, N/16, K/8, 16], original GPTQ nibbles
  USM<float> scales;     // [expert, N/16, K/group_size, 16]
  int k = 0, n = 0, gs = 0;
  void init(int experts, int n_, int k_, int gs_) {
    n = n_;
    k = k_;
    gs = gs_;
    packed.reserve(static_cast<size_t>(experts) * n * k / 8);
    scales.reserve(static_cast<size_t>(experts) * n * (k / gs));
  }
  void from_mat(int expert, const uint32_t* qw, const float* sc) {
    if (!qw || !sc) throw std::runtime_error("XMX weight/scale pointer is null");
    uint32_t* dst = packed.data + static_cast<size_t>(expert) * n * k / 8;
    float* sd = scales.data + static_cast<size_t>(expert) * n * (k / gs);
    for (int nb = 0; nb < n / 16; ++nb) {
      for (int p = 0; p < k / 8; ++p)
        std::memcpy(dst + (static_cast<size_t>(nb) * (k / 8) + p) * 16, qw + p * n + nb * 16, 64);
      for (int g = 0; g < k / gs; ++g)
        std::memcpy(sd + (static_cast<size_t>(nb) * (k / gs) + g) * 16, sc + g * n + nb * 16, 64);
    }
  }
};

ESIMD_INLINE es::simd<float, 16> bf16_load(const uint16_t* p) {
  es::simd<uint16_t, 16> b;
  b.copy_from(p);
  es::simd<uint32_t, 16> u = es::convert<uint32_t>(b) << 16;
  return u.template bit_cast_view<float>();
}

// DPAS B uses two FP16 K values per dword, then consecutive output columns.
ESIMD_INLINE es::simd<sycl::half, 256> dequant_f16(const uint32_t* p, es::simd<float, 16> scale) {
  es::simd<uint32_t, 32> w;
  w.copy_from(p);
  es::simd<sycl::half, 256> b;
#pragma unroll
  for (int pair = 0; pair < 8; ++pair) {
    es::simd<uint32_t, 16> q = w.template select<16, 1>((pair / 4) * 16);
    const int shift = (pair % 4) * 8;
    es::simd<float, 16> lo = (es::convert<float>((q >> shift) & 15) - 8.0f) * scale;
    es::simd<float, 16> hi = (es::convert<float>((q >> (shift + 4)) & 15) - 8.0f) * scale;
    es::simd<sycl::half, 16> lh = es::convert<sycl::half>(lo), hh = es::convert<sycl::half>(hi);
    es::simd<uint32_t, 16> word = es::convert<uint32_t>(lh.template bit_cast_view<uint16_t>().read()) |
                                  (es::convert<uint32_t>(hh.template bit_cast_view<uint16_t>().read()) << 16);
    b.template bit_cast_view<uint32_t>().template select<16, 1>(pair * 16) = word;
  }
  return b;
}

inline sycl::event quantize(const uint16_t* src, int8_t* dst, float* scales, int rows, int k, int gs,
                            const sycl::event* dep = nullptr) {
  const int groups = k / gs;
  return sycl_int4::queue().submit([&](sycl::handler& h) {
    if (dep) h.depends_on(*dep);
    h.parallel_for(sycl::nd_range<1>(static_cast<size_t>(rows) * groups * 128, 128), [=](sycl::nd_item<1> it) {
      int rg = it.get_group(0), lane = it.get_local_id(0);
      int row = rg / groups, g = rg % groups;
      float maximum = 0.0f;
      for (int j = lane; j < gs; j += 128)
        maximum = sycl::fmax(maximum, sycl::fabs(sycl_int4::bf16_to_fp32(src[row * k + g * gs + j])));
      maximum = sycl::reduce_over_group(it.get_group(), maximum, sycl::maximum<float>());
      float scale = maximum / 127.0f;
      if (lane == 0) scales[rg] = scale;
      for (int j = lane; j < gs; j += 128) {
        float v = sycl_int4::bf16_to_fp32(src[row * k + g * gs + j]);
        dst[row * k + g * gs + j] =
            static_cast<int8_t>(sycl::clamp(sycl::rint(scale > 0 ? v / scale : 0), -127.0f, 127.0f));
      }
    });
  });
}

template <int Rows, bool Q8, bool GateUp, bool FusedQuant = false, int KSplits = 1, bool FusedMerge = false>
inline sycl::event projection(const Tile* tiles, int tile_count, const int* row_tokens, const uint16_t* input,
                              const int8_t* qinput, const float* ascales, const Weight& first, const Weight* second,
                              uint16_t* activated, float* down, float limit, float alpha,
                              const sycl::event* dep = nullptr, int8_t* qoutput = nullptr,
                              float* output_scales = nullptr, const int* merge_slots = nullptr,
                              const float* routing = nullptr, float* merged = nullptr, int topk = 0) {
  static_assert(!FusedQuant || (Q8 && GateUp));
  static_assert(KSplits == 1 || KSplits == 2 || KSplits == 4 || KSplits == 8);
  static_assert(!FusedQuant || KSplits == 1);
  static_assert(!FusedMerge || (Rows == 1 && !GateUp && !FusedQuant && KSplits == 1));
  const int n = first.n, k = first.k, gs = first.gs, groups = k / gs;
  const uint32_t* w0 = first.packed.data;
  const float* s0 = first.scales.data;
  const uint32_t* w1 = second ? second->packed.data : nullptr;
  const float* s1 = second ? second->scales.data : nullptr;
  auto compute = [=](size_t index, int lane) SYCL_ESIMD_FUNCTION {
    if constexpr (FusedQuant) es::slm_init<Rows * 8 * sizeof(float)>();
    if constexpr (FusedMerge) es::slm_init<8 * 16 * sizeof(float)>();
    if constexpr (KSplits > 1) es::slm_init<Rows * 16 * KSplits * sizeof(float) * (GateUp ? 2 : 1)>();
    const int nb = index % (n / 16);
    const bool valid = !FusedMerge || index / (n / 16) < static_cast<size_t>(tile_count);
    const Tile tile = valid ? tiles[index / (n / 16)] : Tile{0, 0, 0};
    const size_t wi = (static_cast<size_t>(tile.expert) * (n / 16) + nb) * (k / 8) * 16;
    const size_t si = (static_cast<size_t>(tile.expert) * (n / 16) + nb) * groups * 16;
    es::simd<float, Rows * 16> acc0 = 0.0f, acc1 = 0.0f;
    for (int g = KSplits > 1 ? lane : 0; g < (valid ? groups : 0); g += KSplits) {
      es::simd<float, 16> sc0;
      sc0.copy_from(s0 + si + g * 16);
      es::simd<float, 16> sc1;
      if constexpr (GateUp) sc1.copy_from(s1 + si + g * 16);
      if constexpr (Q8) {
        es::simd<int, Rows * 16> c0 = 0, c1 = 0;
        for (int j = 0; j < gs; j += 32) {
          es::simd<int8_t, Rows * 32> a = 0;
#pragma unroll
          for (int r = 0; r < Rows; ++r) {
            if (r < tile.rows) {
              int row = GateUp ? row_tokens[tile.begin + r] : tile.begin + r;
              es::simd<int8_t, 32> v;
              v.copy_from(qinput + static_cast<size_t>(row) * k + g * gs + j);
              a.template select<32, 1>(r * 32) = v;
            }
          }
          es::simd<uint32_t, 64> b;
          b.copy_from(w0 + wi + (g * gs + j) / 8 * 16);
          b ^= 0x88888888u;
          constexpr int D = Rows > 8 ? 8 : Rows;
#pragma unroll
          for (int row_base = 0; row_base < Rows; row_base += D) {
            es::simd<int, D * 16> partial = c0.template select<D * 16, 1>(row_base * 16);
            es::simd<int8_t, D * 32> src = a.template select<D * 32, 1>(row_base * 32);
            c0.template select<D * 16, 1>(row_base * 16) =
                xmx::dpas<8, D, int, int, uint32_t, int8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(
                    partial, b, src);
          }
          if constexpr (GateUp) {
            b.copy_from(w1 + wi + (g * gs + j) / 8 * 16);
            b ^= 0x88888888u;
#pragma unroll
            for (int row_base = 0; row_base < Rows; row_base += D) {
              es::simd<int, D * 16> partial = c1.template select<D * 16, 1>(row_base * 16);
              es::simd<int8_t, D * 32> src = a.template select<D * 32, 1>(row_base * 32);
              c1.template select<D * 16, 1>(row_base * 16) =
                  xmx::dpas<8, D, int, int, uint32_t, int8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(
                      partial, b, src);
            }
          }
        }
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
          if (r < tile.rows) {
            int row = GateUp ? row_tokens[tile.begin + r] : tile.begin + r;
            float sa = ascales[row * groups + g];
            acc0.template select<16, 1>(r * 16) +=
                es::convert<float>(c0.template select<16, 1>(r * 16).read()) * (sc0 * sa);
            if constexpr (GateUp)
              acc1.template select<16, 1>(r * 16) +=
                  es::convert<float>(c1.template select<16, 1>(r * 16).read()) * (sc1 * sa);
          }
        }
      } else {
        for (int j = 0; j < gs; j += 16) {
          es::simd<sycl::half, Rows * 16> a = 0;
#pragma unroll
          for (int r = 0; r < Rows; ++r) {
            if (r < tile.rows) {
              int row = GateUp ? row_tokens[tile.begin + r] : tile.begin + r;
              a.template select<16, 1>(r * 16) =
                  es::convert<sycl::half>(bf16_load(input + static_cast<size_t>(row) * k + g * gs + j));
            }
          }
          auto b = dequant_f16(w0 + wi + (g * gs + j) / 8 * 16, sc0);
          constexpr int D = Rows > 8 ? 8 : Rows;
#pragma unroll
          for (int row_base = 0; row_base < Rows; row_base += D) {
            es::simd<float, D * 16> partial = acc0.template select<D * 16, 1>(row_base * 16);
            es::simd<sycl::half, D * 16> src = a.template select<D * 16, 1>(row_base * 16);
            acc0.template select<D * 16, 1>(row_base * 16) = xmx::dpas<8, D, float>(partial, b, src);
          }
          if constexpr (GateUp) {
            b = dequant_f16(w1 + wi + (g * gs + j) / 8 * 16, sc1);
#pragma unroll
            for (int row_base = 0; row_base < Rows; row_base += D) {
              es::simd<float, D * 16> partial = acc1.template select<D * 16, 1>(row_base * 16);
              es::simd<sycl::half, D * 16> src = a.template select<D * 16, 1>(row_base * 16);
              acc1.template select<D * 16, 1>(row_base * 16) = xmx::dpas<8, D, float>(partial, b, src);
            }
          }
        }
      }
      // Groups carry distinct scales; integer accumulation never crosses them.
    }
    if constexpr (KSplits > 1) {
      constexpr int bytes = Rows * 16 * sizeof(float);
      es::slm_block_store<float, Rows * 16>(lane * bytes, acc0);
      if constexpr (GateUp) es::slm_block_store<float, Rows * 16>((KSplits + lane) * bytes, acc1);
      es::barrier();
      if (lane != 0) return;
      acc0 = 0.0f;
      acc1 = 0.0f;
#pragma unroll
      for (int split = 0; split < KSplits; ++split) {
        acc0 += es::slm_block_load<float, Rows * 16>(split * bytes);
        if constexpr (GateUp) acc1 += es::slm_block_load<float, Rows * 16>((KSplits + split) * bytes);
      }
    }
#pragma unroll
    for (int r = 0; r < Rows; ++r) {
      if (r >= tile.rows) continue;
      es::simd<float, 16> value = acc0.template select<16, 1>(r * 16);
      if constexpr (GateUp) {
        es::simd<float, 16> up = acc1.template select<16, 1>(r * 16);
        if (limit > 0) {
          value = es::min(value, limit);
          if (alpha > 0) value = es::max(value, -limit);
          up = es::min(es::max(up, -limit), limit);
        }
        value = value / (1.0f + es::exp(-value * (alpha > 0 ? alpha : 1.0f))) * (alpha > 0 ? up + 1.0f : up);
      }
      es::simd<uint32_t, 16> bits = value.template bit_cast_view<uint32_t>();
      bits += 0x7fffu + ((bits >> 16) & 1u);
      es::simd<uint16_t, 16> bf = es::convert<uint16_t>(bits >> 16);
      if constexpr (FusedQuant) {
        es::simd<uint32_t, 16> u = es::convert<uint32_t>(bf) << 16;
        es::simd<float, 16> rounded = u.template bit_cast_view<float>();
        acc0.template select<16, 1>(r * 16) = rounded;
        float maximum = es::hmax<float>(es::abs(rounded));
        es::slm_scatter<float, 1>(es::simd<uint32_t, 1>((r * 8 + lane) * sizeof(float)), es::simd<float, 1>(maximum));
      } else if constexpr (GateUp) {
        bf.copy_to(activated + static_cast<size_t>(tile.begin + r) * n + nb * 16);
      } else {
        es::simd<uint32_t, 16> u = es::convert<uint32_t>(bf) << 16;
        es::simd<float, 16> rounded = u.template bit_cast_view<float>();
        if constexpr (FusedMerge)
          acc0 = rounded;
        else
          rounded.copy_to(down + static_cast<size_t>(tile.begin + r) * n + nb * 16);
      }
    }
    if constexpr (FusedMerge) {
      // Decode tiles have exactly one row. Eight work-items own a column
      // block across routed rows, keeping BF16 rounding before slot weights.
      // Inactive padded lanes still reach the work-group barrier.
      es::slm_block_store<float, 16>(lane * 16 * sizeof(float), acc0);
      es::barrier();
      if (lane == 0) {
        es::simd<float, 16> sum = 0.0f;
        for (int slot = 0; slot < topk; ++slot) {
          const int row = merge_slots[slot];
          if (row >= 0) {
            const auto values = es::slm_block_load<float, 16>(row * 16 * sizeof(float));
            sum += values * routing[slot];
          }
        }
        sum.copy_to(merged + nb * 16);
      }
    }
    if constexpr (FusedQuant) {
      // Eight adjacent N=16 work-items jointly own one group of 128 values.
      es::barrier();
#pragma unroll
      for (int r = 0; r < Rows; ++r) {
        if (r >= tile.rows) continue;
        auto maxima = es::slm_block_load<float, 8>(r * 8 * sizeof(float));
        float scale = es::hmax<float>(maxima) / 127.0f;
        if (lane == 0) output_scales[(tile.begin + r) * (n / 128) + nb / 8] = scale;
        es::simd<float, 16> values = acc0.template select<16, 1>(r * 16);
        values = es::rnde<float>(values / (scale > 0 ? scale : 1.0f));
        es::simd<int8_t, 16> quantized = es::convert<int8_t>(es::min(es::max(values, -127.0f), 127.0f));
        quantized.copy_to(qoutput + static_cast<size_t>(tile.begin + r) * n + nb * 16);
      }
    }
  };
  return sycl_int4::queue().submit([&](sycl::handler& h) {
    if (dep) h.depends_on(*dep);
    const size_t work = static_cast<size_t>(tile_count) * (n / 16);
    if constexpr (FusedMerge) {
      h.parallel_for(sycl::nd_range<1>(static_cast<size_t>(n / 16) * 8, 8), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        const int lane = it.get_local_linear_id();
        compute(static_cast<size_t>(lane) * (n / 16) + it.get_group_linear_id(), lane);
      });
    } else if constexpr (FusedQuant) {
      h.parallel_for(sycl::nd_range<1>(work, 8), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        compute(it.get_global_linear_id(), it.get_local_linear_id());
      });
    } else if constexpr (KSplits > 1) {
      h.parallel_for(sycl::nd_range<1>(work * KSplits, KSplits), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        compute(it.get_group_linear_id(), it.get_local_linear_id());
      });
    } else {
      h.parallel_for(sycl::range<1>(work), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL { compute(id[0], 0); });
    }
  });
}

// Preserve the original slot summation order and FP32 precision. A work-item
// handles a contiguous output vector, amortizing route lookup and loads.
template <int Width>
inline sycl::event merge_vector(const int* slot_rows, const float* weights, const float* down, float* output, int rows,
                                int topk, int hidden, const sycl::event& dep) {
  return sycl_int4::queue().submit([&](sycl::handler& h) {
    h.depends_on(dep);
    h.parallel_for(sycl::range<1>(static_cast<size_t>(rows) * (hidden / Width)), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
      const int token = id[0] / (hidden / Width);
      const int col = (id[0] % (hidden / Width)) * Width;
      es::simd<float, Width> sum = 0;
      for (int slot = 0; slot < topk; ++slot) {
        const int row = slot_rows[token * topk + slot];
        if (row >= 0) {
          es::simd<float, Width> values;
          values.copy_from(down + static_cast<size_t>(row) * hidden + col);
          sum += values * weights[token * topk + slot];
        }
      }
      sum.copy_to(output + static_cast<size_t>(token) * hidden + col);
    });
  });
}

inline sycl::event merge(const int* slot_rows, const float* weights, const float* down, float* output, int rows,
                         int topk, int hidden, const sycl::event& dep) {
  return sycl_int4::queue().submit([&](sycl::handler& h) {
    h.depends_on(dep);
    h.parallel_for(sycl::range<1>(static_cast<size_t>(rows) * hidden), [=](sycl::id<1> id) {
      int token = id[0] / hidden, col = id[0] % hidden;
      float sum = 0;
      for (int slot = 0; slot < topk; ++slot) {
        int row = slot_rows[token * topk + slot];
        if (row >= 0) sum += down[static_cast<size_t>(row) * hidden + col] * weights[token * topk + slot];
      }
      output[id] = sum;
    });
  });
}
}  // namespace sycl_xmx
#endif
