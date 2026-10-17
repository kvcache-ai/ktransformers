// SPDX-License-Identifier: Apache-2.0
#ifndef KT_SYCL_GPTQ_INT4_VECTOR_KERNELS_HPP
#define KT_SYCL_GPTQ_INT4_VECTOR_KERNELS_HPP

#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/experimental/esimd/math.hpp>

#include "gptq_int4_xmx-kernels.hpp"

namespace sycl_xmx {

// GPTQ nibbles use offset binary, not two's-complement S4. Expand four
// nibbles to signed bytes without carries between the packed byte lanes.
ESIMD_INLINE es::simd<int32_t, 16> unpack_s4_bytes(es::simd<uint32_t, 16> packed) {
  auto v = packed & 0xffffu;
  v = (v | (v << 8)) & 0x00ff00ffu;
  v = (v | (v << 4)) & 0x0f0f0f0fu;
  auto negative = (~v) & 0x08080808u;
  v = (v & 0x07070707u) | negative | (negative << 1) | (negative << 2) | (negative << 3) | (negative << 4);
  return v.template bit_cast_view<int32_t>();
}

ESIMD_INLINE es::simd<float, 16> round_bf16(es::simd<float, 16> value) {
  es::simd<uint32_t, 16> bits = value.template bit_cast_view<uint32_t>();
  bits += 0x7fffu + ((bits >> 16) & 1u);
  bits &= 0xffff0000u;
  return bits.template bit_cast_view<float>();
}

// All candidates read the same N16/K8 layout. Lanes own adjacent output
// columns; work-items split K groups, then sum their FP32 partials in SLM.
// This vector path does not issue DPAS or pad an activation into a matrix.
template <bool Q8, bool GateUp, int Splits>
inline sycl::event vector_projection(const Tile* tiles, int tile_count, int tile_rows, const int* row_tokens,
                                     const uint16_t* input, const int8_t* qinput, const float* ascales,
                                     const Weight& first, const Weight* second, uint16_t* activated, float* down,
                                     float limit, float alpha, const sycl::event* dep = nullptr) {
  static_assert(Splits == 1 || Splits == 2 || Splits == 4 || Splits == 8);
  const int n = first.n, k = first.k, gs = first.gs, groups = k / gs;
  const uint32_t* w0 = first.packed.data;
  const uint32_t* w1 = second ? second->packed.data : nullptr;
  const float* s0 = first.scales.data;
  const float* s1 = second ? second->scales.data : nullptr;
  const size_t workgroups = static_cast<size_t>(tile_count) * tile_rows * (n / 16);
  return sycl_int4::queue().submit([&](sycl::handler& h) {
    if (dep) h.depends_on(*dep);
    h.parallel_for(sycl::nd_range<1>(workgroups * Splits, Splits), [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
      if constexpr (Splits > 1) es::slm_init<Splits * 16 * sizeof(float) * (GateUp ? 2 : 1)>();
      const size_t task = it.get_group_linear_id();
      const int nb = task % (n / 16), split = it.get_local_linear_id();
      const int r = (task / (n / 16)) % tile_rows;
      const Tile tile = tiles[task / (static_cast<size_t>(n / 16) * tile_rows)];
      // Every work-item in this work-group takes the same branch.
      if (r >= tile.rows) return;
      const int row = GateUp ? row_tokens[tile.begin + r] : tile.begin + r;
      const size_t wi = (static_cast<size_t>(tile.expert) * (n / 16) + nb) * (k / 8) * 16;
      const size_t si = (static_cast<size_t>(tile.expert) * (n / 16) + nb) * groups * 16;
      es::simd<float, 16> acc0 = 0.0f, acc1 = 0.0f;
      for (int g = split; g < groups; g += Splits) {
        es::simd<int32_t, 16> c0 = 0, c1 = 0;
        es::simd<float, 16> f0 = 0.0f, f1 = 0.0f;
        for (int j = 0; j < gs; j += 8) {
          es::simd<uint32_t, 16> b0;
          b0.copy_from(w0 + wi + (g * gs + j) / 8 * 16);
          es::simd<uint32_t, 16> b1;
          if constexpr (GateUp) b1.copy_from(w1 + wi + (g * gs + j) / 8 * 16);
          if constexpr (Q8) {
            es::simd<int8_t, 8> a;
            a.copy_from(qinput + static_cast<size_t>(row) * k + g * gs + j);
            es::simd<int32_t, 2> av = a.template bit_cast_view<int32_t>();
            es::simd<int32_t, 16> al = static_cast<int32_t>(av[0]), ah = static_cast<int32_t>(av[1]);
            c0 = es::dp4a<int32_t>(c0, unpack_s4_bytes(b0), al);
            c0 = es::dp4a<int32_t>(c0, unpack_s4_bytes(b0 >> 16), ah);
            if constexpr (GateUp) {
              c1 = es::dp4a<int32_t>(c1, unpack_s4_bytes(b1), al);
              c1 = es::dp4a<int32_t>(c1, unpack_s4_bytes(b1 >> 16), ah);
            }
          } else {
            es::simd<uint16_t, 8> bits;
            bits.copy_from(input + static_cast<size_t>(row) * k + g * gs + j);
            es::simd<uint32_t, 8> u = es::convert<uint32_t>(bits) << 16;
            es::simd<float, 8> a = u.template bit_cast_view<float>();
#pragma unroll
            for (int x = 0; x < 8; ++x) {
              es::simd<float, 16> av = static_cast<float>(a[x]);
              auto v0 = es::convert<float>((b0 >> (x * 4)) & 15) - 8.0f;
              f0 = sycl::ext::intel::experimental::esimd::fma(av, v0, f0);
              if constexpr (GateUp) {
                auto v1 = es::convert<float>((b1 >> (x * 4)) & 15) - 8.0f;
                f1 = sycl::ext::intel::experimental::esimd::fma(av, v1, f1);
              }
            }
          }
        }
        es::simd<float, 16> scale;
        scale.copy_from(s0 + si + g * 16);
        if constexpr (Q8) {
          float sa = ascales[row * groups + g];
          acc0 += es::convert<float>(c0) * (scale * sa);
          if constexpr (GateUp) {
            scale.copy_from(s1 + si + g * 16);
            acc1 += es::convert<float>(c1) * (scale * sa);
          }
        } else {
          acc0 += f0 * scale;
          if constexpr (GateUp) {
            scale.copy_from(s1 + si + g * 16);
            acc1 += f1 * scale;
          }
        }
      }
      if constexpr (Splits > 1) {
        es::slm_block_store<float, 16>(split * 16 * sizeof(float), acc0);
        if constexpr (GateUp) es::slm_block_store<float, 16>((Splits + split) * 16 * sizeof(float), acc1);
        es::barrier();
        if (split != 0) return;
        acc0 = 0.0f;
        acc1 = 0.0f;
#pragma unroll
        for (int s = 0; s < Splits; ++s) {
          acc0 += es::slm_block_load<float, 16>(s * 16 * sizeof(float));
          if constexpr (GateUp) acc1 += es::slm_block_load<float, 16>((Splits + s) * 16 * sizeof(float));
        }
      }
      if constexpr (GateUp) {
        if (limit > 0) {
          acc0 = es::min(acc0, limit);
          if (alpha > 0) acc0 = es::max(acc0, -limit);
          acc1 = es::min(es::max(acc1, -limit), limit);
        }
        acc0 = acc0 / (1.0f + es::exp(-acc0 * (alpha > 0 ? alpha : 1.0f))) * (alpha > 0 ? acc1 + 1.0f : acc1);
        auto rounded = round_bf16(acc0);
        es::simd<uint32_t, 16> bits = rounded.template bit_cast_view<uint32_t>();
        es::simd<uint16_t, 16> bf = es::convert<uint16_t>(bits >> 16);
        bf.copy_to(activated + static_cast<size_t>(tile.begin + r) * n + nb * 16);
      } else {
        round_bf16(acc0).copy_to(down + static_cast<size_t>(tile.begin + r) * n + nb * 16);
      }
    });
  });
}

template <bool Q8, bool GateUp>
inline sycl::event vector_dispatch(int splits, const Tile* tiles, int tile_count, int tile_rows, const int* row_tokens,
                                   const uint16_t* input, const int8_t* qinput, const float* ascales,
                                   const Weight& first, const Weight* second, uint16_t* activated, float* down,
                                   float limit, float alpha, const sycl::event* dep = nullptr) {
#define KT_VECTOR_DISPATCH(S)                                                                                       \
  vector_projection<Q8, GateUp, S>(tiles, tile_count, tile_rows, row_tokens, input, qinput, ascales, first, second, \
                                   activated, down, limit, alpha, dep)
  switch (splits) {
    case 1:
      return KT_VECTOR_DISPATCH(1);
    case 2:
      return KT_VECTOR_DISPATCH(2);
    case 4:
      return KT_VECTOR_DISPATCH(4);
    case 8:
      return KT_VECTOR_DISPATCH(8);
    default:
      throw std::runtime_error("Vector K splits must be 1, 2, 4 or 8");
  }
#undef KT_VECTOR_DISPATCH
}
}  // namespace sycl_xmx
#endif
