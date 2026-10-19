// SPDX-License-Identifier: Apache-2.0
#ifndef KT_SYCL_GPTQ_INT4_XMX_FUSED_EXPERT_HPP
#define KT_SYCL_GPTQ_INT4_XMX_FUSED_EXPERT_HPP

#include "gptq_int4_xmx-kernels.hpp"

namespace sycl_xmx {
// Hardware-specific decode candidate: one work-group owns a complete expert.
// Gate/up -> BF16 SwiGLU -> groupwise Q8 -> down stays inside SLM. Global
// expert merge remains separate; there is no cross-work-group synchronization.
// Only H=2048, I=512 and group=128 are implemented for the measured model.
template <int KSplits>
inline sycl::event fused_expert_decode(const Tile* tiles, int tile_count, const int8_t* input,
                                       const float* input_scales, const Weight& gate, const Weight& up,
                                       const Weight& down, float* output, float limit, float alpha,
                                       const sycl::event* dep = nullptr) {
  static_assert(KSplits == 1 || KSplits == 2);
  constexpr int H = 2048, I = 512, GS = 128, NB = I / 16, Threads = NB * KSplits;
  constexpr int PartialBytes = KSplits > 1 ? KSplits * I * 2 * sizeof(float) : 0;
  constexpr int Middle = PartialBytes;
  constexpr int Quantized = Middle + I * sizeof(uint16_t);
  constexpr int Scales = Quantized + I;
  constexpr int SLM = Scales + I / GS * sizeof(float);
  const auto* gw = gate.packed.data;
  const auto* gs = gate.scales.data;
  const auto* uw = up.packed.data;
  const auto* us = up.scales.data;
  const auto* dw = down.packed.data;
  const auto* ds = down.scales.data;
  return sycl_int4::queue().submit([&](sycl::handler& handler) {
    if (dep) handler.depends_on(*dep);
    handler.parallel_for(
        sycl::nd_range<1>(static_cast<size_t>(tile_count) * Threads, Threads),
        [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
          es::slm_init<SLM>();
          const Tile tile = tiles[item.get_group_linear_id()];
          const int lane = item.get_local_linear_id(), nb = lane % NB, split = lane / NB;
          const size_t wi = (static_cast<size_t>(tile.expert) * NB + nb) * (H / 8) * 16;
          const size_t si = (static_cast<size_t>(tile.expert) * NB + nb) * (H / GS) * 16;
          es::simd<float, 16> gacc = 0, uacc = 0;
          for (int group = split; group < H / GS; group += KSplits) {
            es::simd<int32_t, 16> gi = 0, ui = 0;
            for (int j = 0; j < GS; j += 32) {
              es::simd<int8_t, 32> a;
              a.copy_from(input + group * GS + j);  // single input token for all experts
              es::simd<uint32_t, 64> b;
              b.copy_from(gw + wi + (group * GS + j) / 8 * 16);
              b ^= 0x88888888u;
              gi =
                  xmx::dpas<8, 1, int, int, uint32_t, int8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(
                      gi, b, a);
              b.copy_from(uw + wi + (group * GS + j) / 8 * 16);
              b ^= 0x88888888u;
              ui =
                  xmx::dpas<8, 1, int, int, uint32_t, int8_t, xmx::dpas_argument_type::s4, xmx::dpas_argument_type::s8>(
                      ui, b, a);
            }
            es::simd<float, 16> scale;
            scale.copy_from(gs + si + group * 16);
            gacc += es::convert<float>(gi) * (scale * input_scales[group]);
            scale.copy_from(us + si + group * 16);
            uacc += es::convert<float>(ui) * (scale * input_scales[group]);
          }
          if constexpr (KSplits > 1) {
            es::slm_block_store<float, 16>(lane * 16 * sizeof(float), gacc);
            es::slm_block_store<float, 16>((Threads + lane) * 16 * sizeof(float), uacc);
            es::barrier();
            if (split == 0) {
              gacc = 0;
              uacc = 0;
#pragma unroll
              for (int part = 0; part < KSplits; ++part) {
                gacc += es::slm_block_load<float, 16>((part * NB + nb) * 16 * sizeof(float));
                uacc += es::slm_block_load<float, 16>((Threads + part * NB + nb) * 16 * sizeof(float));
              }
            }
          }
          if (split == 0) {
            if (limit > 0) {
              gacc = es::min(gacc, limit);
              if (alpha > 0) gacc = es::max(gacc, -limit);
              uacc = es::min(es::max(uacc, -limit), limit);
            }
            auto value = gacc / (1.0f + es::exp(-gacc * (alpha > 0 ? alpha : 1.0f))) * (alpha > 0 ? uacc + 1.0f : uacc);
            es::simd<uint32_t, 16> bits = value.template bit_cast_view<uint32_t>();
            bits += 0x7fffu + ((bits >> 16) & 1u);
            es::slm_block_store<uint16_t, 16>(Middle + nb * 16 * sizeof(uint16_t), es::convert<uint16_t>(bits >> 16));
          }
          es::barrier();
          if (lane < I / GS) {
            const auto bf = es::slm_block_load<uint16_t, GS>(Middle + lane * GS * sizeof(uint16_t));
            es::simd<uint32_t, GS> bits = es::convert<uint32_t>(bf) << 16;
            const es::simd<float, GS> values = bits.template bit_cast_view<float>();
            const float scale = es::hmax<float>(es::abs(values)) / 127.0f;
            es::slm_scatter<float, 1>(es::simd<uint32_t, 1>(Scales + lane * sizeof(float)), es::simd<float, 1>(scale));
            auto quantized = es::rnde<float>(values / (scale > 0 ? scale : 1.0f));
            es::slm_block_store<int8_t, GS>(Quantized + lane * GS,
                                            es::convert<int8_t>(es::min(es::max(quantized, -127.0f), 127.0f)));
          }
          es::barrier();
          const auto activation_scales = es::slm_block_load<float, I / GS>(Scales);
          for (int out_nb = lane; out_nb < H / 16; out_nb += Threads) {
            const size_t wi = (static_cast<size_t>(tile.expert) * (H / 16) + out_nb) * (I / 8) * 16;
            const size_t si = (static_cast<size_t>(tile.expert) * (H / 16) + out_nb) * (I / GS) * 16;
            es::simd<float, 16> accum = 0;
            for (int group = 0; group < I / GS; ++group) {
              es::simd<int32_t, 16> partial = 0;
              for (int j = 0; j < GS; j += 32) {
                const auto a = es::slm_block_load<int8_t, 32>(Quantized + group * GS + j);
                es::simd<uint32_t, 64> b;
                b.copy_from(dw + wi + (group * GS + j) / 8 * 16);
                b ^= 0x88888888u;
                partial = xmx::dpas<8, 1, int, int, uint32_t, int8_t, xmx::dpas_argument_type::s4,
                                    xmx::dpas_argument_type::s8>(partial, b, a);
              }
              es::simd<float, 16> scale;
              scale.copy_from(ds + si + group * 16);
              accum += es::convert<float>(partial) * (scale * static_cast<float>(activation_scales[group]));
            }
            es::simd<uint32_t, 16> bits = accum.template bit_cast_view<uint32_t>();
            bits += 0x7fffu + ((bits >> 16) & 1u);
            bits &= 0xffff0000u;
            es::simd<float, 16> rounded = bits.template bit_cast_view<float>();
            rounded.copy_to(output + static_cast<size_t>(tile.begin) * H + out_nb * 16);
          }
        });
  });
}
}  // namespace sycl_xmx
#endif
