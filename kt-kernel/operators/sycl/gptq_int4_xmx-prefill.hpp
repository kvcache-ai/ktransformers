// SPDX-License-Identifier: Apache-2.0
#ifndef KT_SYCL_GPTQ_INT4_XMX_PREFILL_HPP
#define KT_SYCL_GPTQ_INT4_XMX_PREFILL_HPP
#include "gptq_int4_xmx-kernels.hpp"

namespace sycl_xmx {
enum class PrefillLoad { Block, Gather };

// Routed rows are not contiguous. Gather dwords across the rows in one vector
// instead of issuing a separate 32-byte block load for every DPAS input row.
template <int Rows>
ESIMD_INLINE es::simd<int8_t, Rows * 32> prefill_gather(const int8_t* input, es::simd<uint32_t, Rows * 8> offsets,
                                                        es::simd_mask<Rows * 8> mask, uint32_t column) {
  es::simd<uint32_t, Rows * 8> words = es::gather<uint32_t, Rows * 8>(reinterpret_cast<const uint32_t*>(input),
                                                                      es::simd<uint32_t, Rows * 8>(offsets + column),
                                                                      mask, es::simd<uint32_t, Rows * 8>(0));
  return words.template bit_cast_view<int8_t>();
}

// Prefill-only N32 tile: two N16 DPAS blocks share activation loads. No
// resident weight conversion; precision/group boundaries match the N16 path.
template <int Rows, bool GateUp, bool FusedQuant = false, PrefillLoad Load = PrefillLoad::Block>
inline sycl::event prefill_n32(const Tile* tiles, int tile_count, const int* row_tokens, const int8_t* input,
                               const float* input_scales, const Weight& first, const Weight* second,
                               uint16_t* activated, float* down, float limit, float alpha,
                               const sycl::event* dep = nullptr, int8_t* qoutput = nullptr,
                               float* output_scales = nullptr) {
  static_assert(!FusedQuant || GateUp);
  static_assert(Rows == 1 || Rows == 2 || Rows == 4 || Rows == 8 || Rows == 16);
  constexpr int D = Rows > 8 ? 8 : Rows;
  const int n = first.n, k = first.k, gs = first.gs, groups = k / gs;
  const auto* w0 = first.packed.data;
  const auto* s0 = first.scales.data;
  const auto* w1 = second ? second->packed.data : nullptr;
  const auto* s1 = second ? second->scales.data : nullptr;
  return sycl_int4::queue().submit([&](sycl::handler& h) {
    if (dep) h.depends_on(*dep);
    const size_t work = static_cast<size_t>(tile_count) * (n / 32);
    auto compute = [=](size_t index, int lane) SYCL_ESIMD_FUNCTION {
      if constexpr (FusedQuant) es::slm_init<Rows * 4 * sizeof(float)>();
      const int nb = (index % (n / 32)) * 2;
      const Tile tile = tiles[index / (n / 32)];
      es::simd<float, Rows * 16> acc[2], up[2];
#pragma unroll
      for (int block = 0; block < 2; ++block) {
        acc[block] = 0;
        up[block] = 0;
      }
      es::simd<uint32_t, Rows * 8> input_offsets = 0;
      es::simd_mask<Rows * 8> input_mask = 0;
      if constexpr (Load == PrefillLoad::Gather) {
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
          if (r < tile.rows) {
            const int row = GateUp ? row_tokens[tile.begin + r] : tile.begin + r;
            input_offsets.template select<8, 1>(r * 8) = es::simd<uint32_t, 8>(row * k, 4);
            input_mask.template select<8, 1>(r * 8) = 1;
          }
        }
      }
      for (int g = 0; g < groups; ++g) {
        es::simd<int, Rows * 16> c[2], u[2];
#pragma unroll
        for (int block = 0; block < 2; ++block) {
          c[block] = 0;
          u[block] = 0;
        }
        for (int j = 0; j < gs; j += 32) {
          es::simd<int8_t, Rows * 32> a = 0;
          if constexpr (Load == PrefillLoad::Gather) {
            a = prefill_gather<Rows>(input, input_offsets, input_mask, g * gs + j);
          } else {
#pragma unroll
            for (int r = 0; r < Rows; ++r) {
              if (r < tile.rows) {
                const int row = GateUp ? row_tokens[tile.begin + r] : tile.begin + r;
                es::simd<int8_t, 32> value;
                value.copy_from(input + static_cast<size_t>(row) * k + g * gs + j);
                a.template select<32, 1>(r * 32) = value;
              }
            }
          }
#pragma unroll
          for (int block = 0; block < 2; ++block) {
            const size_t wi = (static_cast<size_t>(tile.expert) * (n / 16) + nb + block) * (k / 8) * 16;
            es::simd<uint32_t, 64> b;
            b.copy_from(w0 + wi + (g * gs + j) / 8 * 16);
            b ^= 0x88888888u;
#pragma unroll
            for (int r = 0; r < Rows; r += D) {
              es::simd<int, D * 16> partial = c[block].template select<D * 16, 1>(r * 16);
              es::simd<int8_t, D * 32> av = a.template select<D * 32, 1>(r * 32);
              c[block].template select<D * 16, 1>(r * 16) =
                  xmx::dpas<8, D, int, int, uint32_t, int8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(
                      partial, b, av);
            }
            if constexpr (GateUp) {
              b.copy_from(w1 + wi + (g * gs + j) / 8 * 16);
              b ^= 0x88888888u;
#pragma unroll
              for (int r = 0; r < Rows; r += D) {
                es::simd<int, D * 16> partial = u[block].template select<D * 16, 1>(r * 16);
                es::simd<int8_t, D * 32> av = a.template select<D * 32, 1>(r * 32);
                u[block].template select<D * 16, 1>(r * 16) =
                    xmx::dpas<8, D, int, int, uint32_t, int8_t, xmx::dpas_argument_type::s4,
                              xmx::dpas_argument_type::s8>(partial, b, av);
              }
            }
          }
        }
#pragma unroll
        for (int block = 0; block < 2; ++block) {
          const size_t si = (static_cast<size_t>(tile.expert) * (n / 16) + nb + block) * groups * 16;
          es::simd<float, 16> sc, su;
          sc.copy_from(s0 + si + g * 16);
          if constexpr (GateUp) su.copy_from(s1 + si + g * 16);
#pragma unroll
          for (int r = 0; r < Rows; ++r) {
            if (r < tile.rows) {
              const int row = GateUp ? row_tokens[tile.begin + r] : tile.begin + r;
              const float sa = input_scales[row * groups + g];
              acc[block].template select<16, 1>(r * 16) +=
                  es::convert<float>(c[block].template select<16, 1>(r * 16).read()) * (sc * sa);
              if constexpr (GateUp)
                up[block].template select<16, 1>(r * 16) +=
                    es::convert<float>(u[block].template select<16, 1>(r * 16).read()) * (su * sa);
            }
          }
        }
      }
#pragma unroll
      for (int r = 0; r < Rows; ++r) {
        if (r >= tile.rows) continue;
        float maximum = 0;
#pragma unroll
        for (int block = 0; block < 2; ++block) {
          es::simd<float, 16> value = acc[block].template select<16, 1>(r * 16);
          if constexpr (GateUp) {
            es::simd<float, 16> uv = up[block].template select<16, 1>(r * 16);
            if (limit > 0) {
              value = es::min(value, limit);
              if (alpha > 0) value = es::max(value, -limit);
              uv = es::min(es::max(uv, -limit), limit);
            }
            value = value / (1.0f + es::exp(-value * (alpha > 0 ? alpha : 1.0f))) * (alpha > 0 ? uv + 1 : uv);
          }
          es::simd<uint32_t, 16> bits = value.template bit_cast_view<uint32_t>();
          bits += 0x7fffu + ((bits >> 16) & 1u);
          bits &= 0xffff0000u;
          es::simd<float, 16> rounded = bits.template bit_cast_view<float>();
          if constexpr (FusedQuant) {
            acc[block].template select<16, 1>(r * 16) = rounded;
            maximum = es::max(maximum, es::hmax<float>(es::abs(rounded)));
          } else if constexpr (GateUp) {
            es::convert<uint16_t>(bits >> 16)
                .copy_to(activated + static_cast<size_t>(tile.begin + r) * n + (nb + block) * 16);
          } else {
            rounded.copy_to(down + static_cast<size_t>(tile.begin + r) * n + (nb + block) * 16);
          }
        }
        if constexpr (FusedQuant)
          es::slm_scatter<float, 1>(es::simd<uint32_t, 1>((r * 4 + lane) * sizeof(float)), es::simd<float, 1>(maximum));
      }
      if constexpr (FusedQuant) {
        es::barrier();
#pragma unroll
        for (int r = 0; r < Rows; ++r) {
          if (r >= tile.rows) continue;
          auto maxima = es::slm_block_load<float, 4>(r * 4 * sizeof(float));
          const float scale = es::hmax<float>(maxima) / 127.0f;
          if (lane == 0) output_scales[(tile.begin + r) * (n / 128) + nb / 8] = scale;
#pragma unroll
          for (int block = 0; block < 2; ++block) {
            es::simd<float, 16> value = acc[block].template select<16, 1>(r * 16);
            value = es::rnde<float>(value / (scale > 0 ? scale : 1.0f));
            es::convert<int8_t>(es::min(es::max(value, -127.0f), 127.0f))
                .copy_to(qoutput + static_cast<size_t>(tile.begin + r) * n + (nb + block) * 16);
          }
        }
      }
    };
    if constexpr (FusedQuant) {
      h.parallel_for(sycl::nd_range<1>(work, 4), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
        compute(it.get_global_linear_id(), it.get_local_linear_id());
      });
    } else {
      h.parallel_for(sycl::range<1>(work), [=](sycl::id<1> id) SYCL_ESIMD_KERNEL { compute(id[0], 0); });
    }
  });
}
}  // namespace sycl_xmx
#endif
