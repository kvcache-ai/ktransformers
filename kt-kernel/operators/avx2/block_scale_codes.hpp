/**
 * One-byte block scale codes for the AVX2 FP4 kernel (MXFP4, NVFP4).
 *
 * Every block scale of both formats is one byte in the checkpoint (UE8M0, or E4M3 times a per-tensor global), so a
 * matrix holds at most 256 distinct values. With KT_SCALE_CODES=1 a matrix keeps one byte a group indexing a
 * 256-entry FP32 table of those values, beside the FP32 scales it always had: the decode reads a quarter of the scale
 * bytes and multiplies the same FP32 bit patterns. The AMX/AVX-512 MXFP4 backend keeps UE8M0 as one exponent byte
 * already; this brings the AVX2 kernel the same.
 */
#pragma once

#include <cstdint>
#include <cstdlib>

namespace avx2 {

// read at every load, so one process can load both ways
inline bool scale_codes_enabled() {
  const char* e = std::getenv("KT_SCALE_CODES");
  return e != nullptr && e[0] == '1';
}

// A row's group scales: FP32 values, or one-byte codes into the matrix's table. The kernels are generic over the
// source, so the choice is made once a row and the inner loops carry no branch; the FP32 value multiplied is the same
// bit pattern either way.
struct BlockScaleF32 {
  const float* s;
  const float* operator()(int g) const { return s + g; }
};

struct BlockScaleCode {
  const uint8_t* c;
  const float* lut;
  const float* operator()(int g) const { return lut + c[g]; }
};

}  // namespace avx2
