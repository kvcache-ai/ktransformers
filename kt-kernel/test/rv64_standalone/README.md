# Standalone RVV backend checks (no torch required)

Two C++ programs that verify the RVV tinyBLAS backend directly, so they can run
on boards without a Python/torch stack (e.g. SpacemiT K1, VLEN=256).

| file | purpose |
|---|---|
| `rv_tinyblas_check.cpp` | F32 / BF16 / F16 / Q8_0 / Q4_0 sgemm vs scalar reference; fp16→fp32 decode (inf / NaN / -0) |
| `k_align_check.cpp` | k = VLMAX multiples (8 at VLEN=256); catches a fixed-4-lane step regression |

## Build & run

Assumes a `third_party/llama.cpp` layout matching the kt-kernel build
(include roots: `third_party`, `third_party/llamafile`, `third_party/llama.cpp`).

```bash
cd <repo>
gcc -O2 -march=rv64gcv -D_GNU_SOURCE -DGGML_USE_CPU=1 \
    -c third_party/llama.cpp/ggml-quants.c -o gq.o
gcc -O2 -march=rv64gcv -D_GNU_SOURCE -DGGML_USE_CPU=1 \
    -c third_party/llama.cpp/ggml.c -o gg.o

g++ -O2 -std=c++17 -DNDEBUG -march=rv64gcv -DKT_RVV=1 \
    -I third_party -I third_party/llamafile -I third_party/llama.cpp \
    kt-kernel/test/rv64_standalone/rv_tinyblas_check.cpp \
    third_party/llamafile/tinyblas_cpu_sgemm_amd_rv64.cpp \
    third_party/llamafile/tinyblas_cpu_mixmul_amd_rv64.cpp \
    third_party/llamafile/flags.cpp gq.o gg.o \
    -o rvcheck -lm -lpthread
./rvcheck
```

## Notes / gotchas

- `-DNDEBUG` is required. The sgemm wrapper asserts `ldb >= k`, which the
  quantized block types deliberately violate (addresses are block-strided).
- `ggml.c` must be built **as C** (`gcc`, not `g++`) and with `-D_GNU_SOURCE`
  (it uses `CPU_ZERO`/`CPU_SET_S`).
- `GGML_FP16_TO_FP32` relies on a lookup table populated by `ggml_init()`, so the
  test uses its own IEEE half decoder for the reference.
- sgemm writes `C` column-major with `ldc = m`: element (i,j) is `C[j*m+i]`.
- Q4_0 is supported only as the **A (weight)** side: call with
  `Atype=Q4_0, Btype=Q8_0`.

## Expected output (VLEN=256, K1)

```
VLEN(bits)=256  vlmax_e32m1=8  k=128
F32  supported=yes
F32 x F32                max_rel=1.181e-07  PASS
BF16 supported=yes
BF16 x BF16              max_rel=5.901e-08  PASS
F16  supported=yes
F16 x F16                max_rel=8.856e-08  PASS
F16 decode: ... PASS
Q8_0 supported=yes
Q8_0 x Q8_0              max_rel=0.000e+00  PASS
Q4_0 supported=yes
Q8_0 x Q4_0              max_rel=0.000e+00  PASS

RESULT: ALL PASS
```
