// Standalone RVV tinyBLAS verification for VLEN != 128 (e.g. SpacemiT K1, 256-bit).
// Compares llamafile_sgemm_rv64 (our RVV backend) against ggml dequant + naive dot.
#include "llamafile/sgemm.h"
#include "llama.cpp/ggml-impl.h"
#include "llama.cpp/ggml-quants.h"
#include "tinyblas_cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>

// Self-contained IEEE-754 half decode (ggml's GGML_FP16_TO_FP32 needs a table
// populated by ggml_init, which this standalone test does not call).
static float fp16_to_f32_ref(uint16_t h) {
    uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1fu, m = h & 0x3ffu, b;
    if (e == 0) {                      // zero / subnormal (flush to zero here)
        b = (s << 31);
    } else if (e == 0x1f) {            // inf / nan
        b = (s << 31) | 0x7f800000u | (m << 13);
    } else {
        b = (s << 31) | ((e + 112u) << 23) | (m << 13);
    }
    float f; memcpy(&f, &b, 4); return f;
}

static float bf16_to_f32(uint16_t v) {
    uint32_t u = ((uint32_t)v) << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

template <typename F>
static void naive(int m, int n, int k, const float* A, const float* B, float* C, F fa, F fb) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            double s = 0;
            for (int t = 0; t < k; t++) s += (double)fa(A[i * k + t]) * (double)fb(B[j * k + t]);
            C[i * n + j] = (float)s;  // reference is row-major, compared below after transposing
        }
}

static int g_fail = 0;
static void report(const char* name, double maxrel, double tol) {
    printf("%-24s max_rel=%.3e  %s\n", name, maxrel, maxrel <= tol ? "PASS" : "FAIL");
    if (maxrel > tol) g_fail++;
}
// sgemm writes C in column-major order with ldc=m: element (i,j) is C[j*m+i].
// `ref` holds the reference row-major (ref[i*n+j]).
int g_m, g_n;
static double relerr(const std::vector<float>& c, const std::vector<float>& ref) {
    double mx = 1e-9, e = 0;
    for (int i = 0; i < g_m; i++)
        for (int j = 0; j < g_n; j++) {
            double r = ref[i * g_n + j];
            mx = std::max(mx, std::fabs(r));
            e = std::max(e, std::fabs(c[j * g_m + i] - r));
        }
    return e / mx;
}

int main() {
    const int m = 3, n = 5, k = 128;  // k multiple of VLMAX (8 at VLEN=256, 4 at 128)
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    printf("VLEN(bits)=%zu  vlmax_e32m1=%zu  k=%d\n", __riscv_vlenb() * 8, vlmax, k);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0, 0.1f);
    auto I = [](float v) { return v; };
    g_m = m; g_n = n;

    std::vector<float> A(m * k), B(n * k);
    for (auto& x : A) x = dist(rng);
    for (auto& x : B) x = dist(rng);
    std::vector<float> C(m * n), R(m * n);  // C is column-major: ldc=m
    int ok;

    // ---- F32 x F32
    naive(m, n, k, A.data(), B.data(), R.data(), I, I);
    ok = llamafile_sgemm(m, n, k, A.data(), k, B.data(), k, C.data(), m, 0, 1, GGML_TASK_TYPE_COMPUTE,
                         GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32, GGML_PREC_DEFAULT);
    printf("F32  supported=%s\n", ok ? "yes" : "NO");
    report("F32 x F32", ok ? relerr(C, R) : 1e9, 1e-5);

    // ---- BF16 x BF16
    std::vector<float> Ad(m * k), Bd(n * k);
    std::vector<uint16_t> Ab(m * k), Bb(n * k);
    std::vector<float> Af(m * k), Bf(n * k);
    for (int i = 0; i < m * k; i++) {
        float t = A[i];
        uint32_t u; memcpy(&u, &t, 4);
        u = (u + 0x8000u) >> 16;  // round-to-nearest-even bf16
        uint16_t h = (uint16_t)u;
        Ab[i] = h;
        Ad[i] = bf16_to_f32(h);
    }
    for (int i = 0; i < n * k; i++) {
        float t = B[i];
        uint32_t u; memcpy(&u, &t, 4);
        u = (u + 0x8000u) >> 16;
        uint16_t h = (uint16_t)u;
        Bb[i] = h;
        Bd[i] = bf16_to_f32(h);
    }
    naive(m, n, k, Ad.data(), Bd.data(), R.data(), I, I);
    ok = llamafile_sgemm(m, n, k, Ab.data(), k, Bb.data(), k, C.data(), m, 0, 1, GGML_TASK_TYPE_COMPUTE,
                         GGML_TYPE_BF16, GGML_TYPE_BF16, GGML_TYPE_F32, GGML_PREC_DEFAULT);
    printf("BF16 supported=%s\n", ok ? "yes" : "NO");
    report("BF16 x BF16", ok ? relerr(C, R) : 1e9, 2e-3);

    // ---- F16 x F16
    std::vector<uint16_t> Ah(m * k), Bh(n * k);
    for (int i = 0; i < m * k; i++) Ah[i] = ggml_fp32_to_fp16(A[i]);
    for (int i = 0; i < n * k; i++) Bh[i] = ggml_fp32_to_fp16(B[i]);
    for (int i = 0; i < m * k; i++) Ad[i] = fp16_to_f32_ref(Ah[i]);
    for (int i = 0; i < n * k; i++) Bd[i] = fp16_to_f32_ref(Bh[i]);
    naive(m, n, k, Ad.data(), Bd.data(), R.data(), I, I);
    ok = llamafile_sgemm(m, n, k, Ah.data(), k, Bh.data(), k, C.data(), m, 0, 1, GGML_TASK_TYPE_COMPUTE,
                         GGML_TYPE_F16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_PREC_DEFAULT);
    printf("F16  supported=%s\n", ok ? "yes" : "NO");
    report("F16 x F16", ok ? relerr(C, R) : 1e9, 2e-3);

    // ---- fp16 -> fp32 decode path (inf / -inf / NaN / -0 / 1.0 / 65504 / -1.0 / 0)
    {
        const uint16_t in[8] = {0x7C00, 0xFC00, 0x7E00, 0x8000, 0x3C00, 0x7BFF, 0xBC00, 0x0000};
        size_t n2 = std::min((size_t)8, vlmax);
        vfloat32m1_t v = load<vfloat32m1_t>((const ggml_fp16_t*)in);
        float out[8] = {0};
        __riscv_vse32_v_f32m1(out, v, n2);
        bool bad = false;
        printf("F16 decode:");
        for (size_t i = 0; i < n2; i++) {
            uint32_t got, want;
            memcpy(&got, &out[i], 4);
            float wf = fp16_to_f32_ref(in[i]);
            memcpy(&want, &wf, 4);
            printf(" %04x->%08x", in[i], got);
            bool same = (std::isnan(out[i]) && std::isnan(wf)) || got == want;
            if (!same) bad = true;
        }
        printf("  %s\n", bad ? "FAIL" : "PASS");
        if (bad) g_fail++;
    }

    // ---- Q8_0 x Q8_0 and Q8_0 x Q4_0 (A = quantized activation, B = weight)
    {
        std::vector<float> X(n * k);
        for (auto& x : X) x = dist(rng);
        std::vector<block_q8_0> Q8((size_t)n * (k / QK8_0));
        std::vector<block_q4_0> Q4((size_t)n * (k / QK4_0));
        std::vector<block_q8_0> A8((size_t)m * (k / QK8_0));
        for (int r = 0; r < n; r++) quantize_row_q8_0(X.data() + r * k, Q8.data() + (size_t)r * (k / QK8_0), k);
        for (int r = 0; r < n; r++) quantize_row_q4_0(X.data() + r * k, Q4.data() + (size_t)r * (k / QK4_0), k);
        for (int r = 0; r < m; r++) quantize_row_q8_0(A.data() + r * k, A8.data() + (size_t)r * (k / QK8_0), k);

        std::vector<float> A8d(m * k), Q8d(n * k), Q4d(n * k);
        ggml_internal_get_type_traits(GGML_TYPE_Q8_0).to_float(A8.data(), A8d.data(), m * k);
        ggml_internal_get_type_traits(GGML_TYPE_Q8_0).to_float(Q8.data(), Q8d.data(), n * k);
        ggml_internal_get_type_traits(GGML_TYPE_Q4_0).to_float(Q4.data(), Q4d.data(), n * k);

        naive(m, n, k, A8d.data(), Q8d.data(), R.data(), I, I);
        ok = llamafile_sgemm(m, n, k, A8.data(), k / QK8_0, Q8.data(), k / QK8_0, C.data(), m, 0, 1,
                             GGML_TASK_TYPE_COMPUTE, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, GGML_TYPE_F32,
                             GGML_PREC_DEFAULT);
        printf("Q8_0 supported=%s\n", ok ? "yes" : "NO");
        report("Q8_0 x Q8_0", ok ? relerr(C, R) : 1e9, 1e-4);

        // Q4_0 is supported as the *weight* (A) side: A=Q4_0, B=Q8_0.
        naive(m, n, k, Q8d.data(), Q4d.data(), R.data(), I, I);
        ok = llamafile_sgemm(m, n, k, Q8.data(), k / QK8_0, Q4.data(), k / QK4_0, C.data(), m, 0, 1,
                             GGML_TASK_TYPE_COMPUTE, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_F32,
                             GGML_PREC_DEFAULT);
        printf("Q4_0 supported=%s\n", ok ? "yes" : "NO");
        report("Q8_0 x Q4_0", ok ? relerr(C, R) : 1e9, 1e-4);
    }

    printf("\nRESULT: %s\n", g_fail ? "FAIL" : "ALL PASS");
    return g_fail ? 1 : 0;
}
