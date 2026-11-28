// AscendC FUSED MXFP4 -> W8A8: int8 weight + per-output-channel oscale in ONE pass (reads MXFP4
// once). Block-partitioned: each core owns a contiguous, ACC-aligned row range; per row it does
// decode/scale/reduce once, emits int8 (two contiguous planes [lo|hi]) AND accumulates oscale into
// a UB block; each full block is flushed as ONE large contiguous DataCopy (the idiom that makes
// the small per-channel scale store survive alongside loads + int8 stores).
#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t HALF_MAX = 2048;
constexpr int32_t IN_MAX = 4096;
constexpr int32_t NB_MAX = 128;
constexpr int32_t ACC = 512;   // oscale flush block (floats), 8-aligned

extern "C" __global__ __aicore__ void mxfp4_fused(
    GM_ADDR codes, GM_ADDR scaleg, GM_ADDR outg, GM_ADDR oscaleg,
    GM_ADDR lutLoG, GM_ADDR lutHiG, GM_ADDR lutE8G, GM_ADDR scOffG,
    uint32_t R, uint32_t HALF, uint32_t NB, uint32_t IN)
{
    const int32_t blkid = GetBlockIdx();
    const int32_t nblk = GetBlockNum();

    GlobalTensor<uint8_t> gCodes, gScale, gOut;
    GlobalTensor<float> gOscale, gLutLo, gLutHi, gLutE8;
    GlobalTensor<uint32_t> gScOff;
    gCodes.SetGlobalBuffer((__gm__ uint8_t *)codes);
    gScale.SetGlobalBuffer((__gm__ uint8_t *)scaleg);
    gOut.SetGlobalBuffer((__gm__ uint8_t *)outg);
    gOscale.SetGlobalBuffer((__gm__ float *)oscaleg);
    gLutLo.SetGlobalBuffer((__gm__ float *)lutLoG);
    gLutHi.SetGlobalBuffer((__gm__ float *)lutHiG);
    gLutE8.SetGlobalBuffer((__gm__ float *)lutE8G);
    gScOff.SetGlobalBuffer((__gm__ uint32_t *)scOffG);

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> qCodes, qScale;
    TQue<QuePosition::VECOUT, 1> qOut;
    pipe.InitBuffer(qCodes, 1, HALF_MAX * sizeof(uint8_t));
    pipe.InitBuffer(qScale, 1, (NB_MAX + 32) * sizeof(uint8_t));
    pipe.InitBuffer(qOut, 1, IN_MAX * sizeof(uint8_t));
    TBuf<TPosition::VECCALC> tLutLo, tLutHi, tLutE8, tScOff;
    TBuf<TPosition::VECCALC> tComb, tOff, tOffH, tScI, tScF, tScHalf, tAbs, tWork, tAcc;
    pipe.InitBuffer(tLutLo, 256 * sizeof(float));
    pipe.InitBuffer(tLutHi, 256 * sizeof(float));
    pipe.InitBuffer(tLutE8, 256 * sizeof(float));
    pipe.InitBuffer(tScOff, HALF_MAX * sizeof(uint32_t));
    pipe.InitBuffer(tComb, 2 * HALF_MAX * sizeof(float));
    pipe.InitBuffer(tOff, HALF_MAX * sizeof(int32_t));
    pipe.InitBuffer(tOffH, HALF_MAX * sizeof(half));
    pipe.InitBuffer(tScI, NB_MAX * sizeof(int32_t));
    pipe.InitBuffer(tScF, NB_MAX * sizeof(float));
    pipe.InitBuffer(tScHalf, HALF_MAX * sizeof(float));
    pipe.InitBuffer(tAbs, HALF_MAX * sizeof(float));
    pipe.InitBuffer(tWork, HALF_MAX * sizeof(float));
    pipe.InitBuffer(tAcc, ACC * sizeof(float));

    LocalTensor<float> lutLo = tLutLo.Get<float>();
    LocalTensor<float> lutHi = tLutHi.Get<float>();
    LocalTensor<float> lutE8 = tLutE8.Get<float>();
    LocalTensor<uint32_t> scOff = tScOff.Get<uint32_t>();
    LocalTensor<float> comb = tComb.Get<float>();
    LocalTensor<int32_t> off = tOff.Get<int32_t>();
    LocalTensor<half> offH = tOffH.Get<half>();
    LocalTensor<int32_t> scI = tScI.Get<int32_t>();
    LocalTensor<float> scF = tScF.Get<float>();
    LocalTensor<float> scHalf = tScHalf.Get<float>();
    LocalTensor<float> absb = tAbs.Get<float>();
    LocalTensor<float> work = tWork.Get<float>();
    LocalTensor<float> acc = tAcc.Get<float>();

    DataCopy(lutLo, gLutLo, 256);
    DataCopy(lutHi, gLutHi, 256);
    DataCopy(lutE8, gLutE8, 256);
    DataCopy(scOff, gScOff, HALF);
    PipeBarrier<PIPE_ALL>();

    const uint32_t chunk = ((R + nblk - 1) / nblk + (ACC - 1)) / ACC * ACC;
    const uint32_t rStart = (uint32_t)blkid * chunk;
    uint32_t rEnd = rStart + chunk;
    if (rEnd > R) rEnd = R;
    const uint32_t scLoad = (NB + 31) / 32 * 32;

    for (uint32_t base = rStart; base < rEnd; base += ACC) {
        uint32_t bend = base + ACC;
        if (bend > rEnd) bend = rEnd;
        for (uint32_t r = base; r < bend; r++) {
            LocalTensor<float> vlo = comb;
            LocalTensor<float> vhi = comb[HALF];

            LocalTensor<uint8_t> cu = qCodes.AllocTensor<uint8_t>();
            DataCopy(cu, gCodes[(uint64_t)r * HALF], HALF);
            qCodes.EnQue(cu);
            LocalTensor<uint8_t> cuU = qCodes.DeQue<uint8_t>();
            LocalTensor<uint8_t> su = qScale.AllocTensor<uint8_t>();
            DataCopy(su, gScale[(uint64_t)r * NB], scLoad);
            qScale.EnQue(su);
            LocalTensor<uint8_t> suU = qScale.DeQue<uint8_t>();

            Cast(offH, cuU, RoundMode::CAST_NONE, HALF);
            Muls(offH, offH, (half)4.0, HALF);
            Cast(off, offH, RoundMode::CAST_RINT, HALF);
            LocalTensor<uint32_t> offU = off.ReinterpretCast<uint32_t>();
            Gather(vlo, lutLo, offU, (uint32_t)0, HALF);
            Gather(vhi, lutHi, offU, (uint32_t)0, HALF);
            qCodes.FreeTensor(cuU);

            Cast(offH, suU, RoundMode::CAST_NONE, NB);
            Muls(offH, offH, (half)4.0, NB);
            Cast(scI, offH, RoundMode::CAST_RINT, NB);
            Gather(scF, lutE8, scI.ReinterpretCast<uint32_t>(), (uint32_t)0, NB);
            qScale.FreeTensor(suU);
            PipeBarrier<PIPE_V>();
            Gather(scHalf, scF, scOff, (uint32_t)0, HALF);
            Mul(vlo, vlo, scHalf, HALF);
            Mul(vhi, vhi, scHalf, HALF);
            PipeBarrier<PIPE_V>();

            Abs(absb, vlo, HALF);
            Abs(work, vhi, HALF);
            Max(scHalf, absb, work, HALF);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> fa = scHalf, fb = absb;
            for (uint32_t h = HALF >> 1; h >= 8; h >>= 1) {
                Max(fb, fa, fa[h], h);
                PipeBarrier<PIPE_V>();
                LocalTensor<float> tmp = fa; fa = fb; fb = tmp;
            }
            PipeBarrier<PIPE_ALL>();
            float amax = fa.GetValue(0);
            for (int i = 1; i < 8; i++) { float v = fa.GetValue(i); if (v > amax) amax = v; }
            if (amax < 1e-8f) amax = 1e-8f;
            acc.SetValue(r - base, amax / 127.0f);     // accumulate oscale (flushed per block)
            float inv = 127.0f / amax;

            PipeBarrier<PIPE_ALL>();                    // scalar inv -> vector
            Muls(vlo, vlo, inv, HALF);
            Muls(vhi, vhi, inv, HALF);
            PipeBarrier<PIPE_V>();
            Mins(vlo, vlo, 127.0f, HALF); Maxs(vlo, vlo, -127.0f, HALF);
            Mins(vhi, vhi, 127.0f, HALF); Maxs(vhi, vhi, -127.0f, HALF);
            PipeBarrier<PIPE_V>();

            LocalTensor<uint8_t> outrow = qOut.AllocTensor<uint8_t>();
            LocalTensor<int8_t> outI = outrow.ReinterpretCast<int8_t>();
            Cast(offH, vlo, RoundMode::CAST_NONE, HALF);
            PipeBarrier<PIPE_V>();
            Cast(outI, offH, RoundMode::CAST_RINT, HALF);
            PipeBarrier<PIPE_V>();
            Cast(offH, vhi, RoundMode::CAST_NONE, HALF);
            PipeBarrier<PIPE_V>();
            Cast(outI[HALF], offH, RoundMode::CAST_RINT, HALF);
            PipeBarrier<PIPE_V>();
            qOut.EnQue(outrow);
            LocalTensor<uint8_t> outU = qOut.DeQue<uint8_t>();
            DataCopy(gOut[(uint64_t)r * IN], outU, IN);
            qOut.FreeTensor(outU);
        }
        // flush the oscale block as one large contiguous DataCopy (8-aligned base)
        PipeBarrier<PIPE_ALL>();
        DataCopy(gOscale[base], acc, ACC);
        PipeBarrier<PIPE_ALL>();
    }
}

extern "C" void launch_mxfp4_fused(void *stream, uint32_t blockdim,
    uint8_t *codes, uint8_t *scale, uint8_t *out, uint8_t *oscale,
    uint8_t *lutLo, uint8_t *lutHi, uint8_t *lutE8, uint8_t *scOff,
    uint32_t R, uint32_t HALF, uint32_t NB, uint32_t IN)
{
    mxfp4_fused<<<blockdim, nullptr, stream>>>(
        (GM_ADDR)codes, (GM_ADDR)scale, (GM_ADDR)out, (GM_ADDR)oscale,
        (GM_ADDR)lutLo, (GM_ADDR)lutHi, (GM_ADDR)lutE8, (GM_ADDR)scOff,
        R, HALF, NB, IN);
}

// ---- block-input variant: reads raw GGUF block_mxfp4 ([nb*17] per row = nb x (1 e8m0 scale + 16
// codes)) and de-interleaves IN UB via Gather (same gather-from-UB-by-offset op the base kernel
// already uses for scHalf), so the host does NO de-interleave (the slow 16-of-17 strided int8 copy).
// codeOff[j] = byte offset of code j in the half-cast block buffer = ((j/16)*17 + 1 + j%16)*2;
// scaleOff[b] = (b*17)*2. Everything after the input load is byte-identical to mxfp4_fused.
constexpr int32_t BLK_MAX = (HALF_MAX / 16) * 17;   // 2176

extern "C" __global__ __aicore__ void mxfp4_fused_blk(
    GM_ADDR blocks, GM_ADDR outg, GM_ADDR oscaleg,
    GM_ADDR lutLoG, GM_ADDR lutHiG, GM_ADDR lutE8G, GM_ADDR scOffG,
    GM_ADDR codeOffG, GM_ADDR scaleOffG,
    uint32_t R, uint32_t HALF, uint32_t NB, uint32_t IN)
{
    const int32_t blkid = GetBlockIdx();
    const int32_t nblk = GetBlockNum();

    GlobalTensor<uint8_t> gBlocks, gOut;
    GlobalTensor<float> gOscale, gLutLo, gLutHi, gLutE8;
    GlobalTensor<uint32_t> gScOff, gCodeOff, gScaleOff;
    gBlocks.SetGlobalBuffer((__gm__ uint8_t *)blocks);
    gOut.SetGlobalBuffer((__gm__ uint8_t *)outg);
    gOscale.SetGlobalBuffer((__gm__ float *)oscaleg);
    gLutLo.SetGlobalBuffer((__gm__ float *)lutLoG);
    gLutHi.SetGlobalBuffer((__gm__ float *)lutHiG);
    gLutE8.SetGlobalBuffer((__gm__ float *)lutE8G);
    gScOff.SetGlobalBuffer((__gm__ uint32_t *)scOffG);
    gCodeOff.SetGlobalBuffer((__gm__ uint32_t *)codeOffG);
    gScaleOff.SetGlobalBuffer((__gm__ uint32_t *)scaleOffG);

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> qBlk;
    TQue<QuePosition::VECOUT, 1> qOut;
    pipe.InitBuffer(qBlk, 1, BLK_MAX * sizeof(uint8_t));
    pipe.InitBuffer(qOut, 1, IN_MAX * sizeof(uint8_t));
    TBuf<TPosition::VECCALC> tLutLo, tLutHi, tLutE8, tScOff, tCodeOff, tScaleOff, tBlkH;
    TBuf<TPosition::VECCALC> tComb, tOff, tOffH, tScI, tScF, tScHalf, tAbs, tWork, tAcc;
    pipe.InitBuffer(tLutLo, 256 * sizeof(float));
    pipe.InitBuffer(tLutHi, 256 * sizeof(float));
    pipe.InitBuffer(tLutE8, 256 * sizeof(float));
    pipe.InitBuffer(tScOff, HALF_MAX * sizeof(uint32_t));
    pipe.InitBuffer(tCodeOff, HALF_MAX * sizeof(uint32_t));
    pipe.InitBuffer(tScaleOff, NB_MAX * sizeof(uint32_t));
    pipe.InitBuffer(tBlkH, BLK_MAX * sizeof(half));
    pipe.InitBuffer(tComb, 2 * HALF_MAX * sizeof(float));
    pipe.InitBuffer(tOff, HALF_MAX * sizeof(int32_t));
    pipe.InitBuffer(tOffH, HALF_MAX * sizeof(half));
    pipe.InitBuffer(tScI, NB_MAX * sizeof(int32_t));
    pipe.InitBuffer(tScF, NB_MAX * sizeof(float));
    pipe.InitBuffer(tScHalf, HALF_MAX * sizeof(float));
    pipe.InitBuffer(tAbs, HALF_MAX * sizeof(float));
    pipe.InitBuffer(tWork, HALF_MAX * sizeof(float));
    pipe.InitBuffer(tAcc, ACC * sizeof(float));

    LocalTensor<float> lutLo = tLutLo.Get<float>();
    LocalTensor<float> lutHi = tLutHi.Get<float>();
    LocalTensor<float> lutE8 = tLutE8.Get<float>();
    LocalTensor<uint32_t> scOff = tScOff.Get<uint32_t>();
    LocalTensor<uint32_t> codeOff = tCodeOff.Get<uint32_t>();
    LocalTensor<uint32_t> scaleOff = tScaleOff.Get<uint32_t>();
    LocalTensor<half> blkH = tBlkH.Get<half>();
    LocalTensor<float> comb = tComb.Get<float>();
    LocalTensor<int32_t> off = tOff.Get<int32_t>();
    LocalTensor<half> offH = tOffH.Get<half>();
    LocalTensor<int32_t> scI = tScI.Get<int32_t>();
    LocalTensor<float> scF = tScF.Get<float>();
    LocalTensor<float> scHalf = tScHalf.Get<float>();
    LocalTensor<float> absb = tAbs.Get<float>();
    LocalTensor<float> work = tWork.Get<float>();
    LocalTensor<float> acc = tAcc.Get<float>();

    DataCopy(lutLo, gLutLo, 256);
    DataCopy(lutHi, gLutHi, 256);
    DataCopy(lutE8, gLutE8, 256);
    DataCopy(scOff, gScOff, HALF);
    DataCopy(codeOff, gCodeOff, HALF);
    DataCopy(scaleOff, gScaleOff, NB);
    PipeBarrier<PIPE_ALL>();

    const uint32_t nb17 = (HALF / 16) * 17;
    const uint32_t chunk = ((R + nblk - 1) / nblk + (ACC - 1)) / ACC * ACC;
    const uint32_t rStart = (uint32_t)blkid * chunk;
    uint32_t rEnd = rStart + chunk;
    if (rEnd > R) rEnd = R;

    for (uint32_t base = rStart; base < rEnd; base += ACC) {
        uint32_t bend = base + ACC;
        if (bend > rEnd) bend = rEnd;
        for (uint32_t r = base; r < bend; r++) {
            LocalTensor<float> vlo = comb;
            LocalTensor<float> vhi = comb[HALF];

            LocalTensor<uint8_t> bu = qBlk.AllocTensor<uint8_t>();
            DataCopy(bu, gBlocks[(uint64_t)r * nb17], (nb17 + 31) / 32 * 32);
            qBlk.EnQue(bu);
            LocalTensor<uint8_t> buU = qBlk.DeQue<uint8_t>();
            Cast(blkH, buU, RoundMode::CAST_NONE, nb17);   // blocks -> half
            qBlk.FreeTensor(buU);
            PipeBarrier<PIPE_V>();

            Gather(offH, blkH, codeOff, (uint32_t)0, HALF);   // de-interleave codes -> half
            Muls(offH, offH, (half)4.0, HALF);
            Cast(off, offH, RoundMode::CAST_RINT, HALF);
            LocalTensor<uint32_t> offU = off.ReinterpretCast<uint32_t>();
            Gather(vlo, lutLo, offU, (uint32_t)0, HALF);
            Gather(vhi, lutHi, offU, (uint32_t)0, HALF);

            Gather(offH, blkH, scaleOff, (uint32_t)0, NB);    // de-interleave scale -> half
            Muls(offH, offH, (half)4.0, NB);
            Cast(scI, offH, RoundMode::CAST_RINT, NB);
            Gather(scF, lutE8, scI.ReinterpretCast<uint32_t>(), (uint32_t)0, NB);
            PipeBarrier<PIPE_V>();
            Gather(scHalf, scF, scOff, (uint32_t)0, HALF);
            Mul(vlo, vlo, scHalf, HALF);
            Mul(vhi, vhi, scHalf, HALF);
            PipeBarrier<PIPE_V>();

            Abs(absb, vlo, HALF);
            Abs(work, vhi, HALF);
            Max(scHalf, absb, work, HALF);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> fa = scHalf, fb = absb;
            for (uint32_t h = HALF >> 1; h >= 8; h >>= 1) {
                Max(fb, fa, fa[h], h);
                PipeBarrier<PIPE_V>();
                LocalTensor<float> tmp = fa; fa = fb; fb = tmp;
            }
            PipeBarrier<PIPE_ALL>();
            float amax = fa.GetValue(0);
            for (int i = 1; i < 8; i++) { float v = fa.GetValue(i); if (v > amax) amax = v; }
            if (amax < 1e-8f) amax = 1e-8f;
            acc.SetValue(r - base, amax / 127.0f);
            float inv = 127.0f / amax;

            PipeBarrier<PIPE_ALL>();
            Muls(vlo, vlo, inv, HALF);
            Muls(vhi, vhi, inv, HALF);
            PipeBarrier<PIPE_V>();
            Mins(vlo, vlo, 127.0f, HALF); Maxs(vlo, vlo, -127.0f, HALF);
            Mins(vhi, vhi, 127.0f, HALF); Maxs(vhi, vhi, -127.0f, HALF);
            PipeBarrier<PIPE_V>();

            LocalTensor<uint8_t> outrow = qOut.AllocTensor<uint8_t>();
            LocalTensor<int8_t> outI = outrow.ReinterpretCast<int8_t>();
            Cast(offH, vlo, RoundMode::CAST_NONE, HALF);
            PipeBarrier<PIPE_V>();
            Cast(outI, offH, RoundMode::CAST_RINT, HALF);
            PipeBarrier<PIPE_V>();
            Cast(offH, vhi, RoundMode::CAST_NONE, HALF);
            PipeBarrier<PIPE_V>();
            Cast(outI[HALF], offH, RoundMode::CAST_RINT, HALF);
            PipeBarrier<PIPE_V>();
            qOut.EnQue(outrow);
            LocalTensor<uint8_t> outU = qOut.DeQue<uint8_t>();
            DataCopy(gOut[(uint64_t)r * IN], outU, IN);
            qOut.FreeTensor(outU);
        }
        PipeBarrier<PIPE_ALL>();
        DataCopy(gOscale[base], acc, ACC);
        PipeBarrier<PIPE_ALL>();
    }
}

extern "C" void launch_mxfp4_fused_blk(void *stream, uint32_t blockdim,
    uint8_t *blocks, uint8_t *out, uint8_t *oscale,
    uint8_t *lutLo, uint8_t *lutHi, uint8_t *lutE8, uint8_t *scOff,
    uint8_t *codeOff, uint8_t *scaleOff,
    uint32_t R, uint32_t HALF, uint32_t NB, uint32_t IN)
{
    mxfp4_fused_blk<<<blockdim, nullptr, stream>>>(
        (GM_ADDR)blocks, (GM_ADDR)out, (GM_ADDR)oscale,
        (GM_ADDR)lutLo, (GM_ADDR)lutHi, (GM_ADDR)lutE8, (GM_ADDR)scOff,
        (GM_ADDR)codeOff, (GM_ADDR)scaleOff,
        R, HALF, NB, IN);
}

// ---- direct FRACTAL_NZ block-input variant (v3):
// Reads raw GGUF block_mxfp4 [Ec, OUT, (IN/32)*17], decodes/scales/quantizes 32 output rows in UB
// with S_HALF=1024 (2048 weights = 64 MXFP4 blocks per strip), performs hardware batched 16x16
// transpose via TransDataTo5HD<half>(repeatTimes=8) over 128 columns at a time (zero cpSplit16),
// and writes 4,096 B contiguous bursts (dstStride=0) directly into FRACTAL_NZ layout
// [Ec, OUT/32, IN/16, 16, 32] (int8) + per-output-channel float32 oscale [Ec, OUT].
//
// Eliminates PyTorch cat/fp16/transpose/contiguous/int8/npu_format_cast(29)/copy_ completely.
// Supported shapes: IN in {2048, 4096}, OUT % 32 == 0.
// UB footprint: 181.1 KB < 192 KB on dav-c220 via non-overlapping Phase 1 / Phase 2 reuse.
constexpr int32_t NZ_S_HALF = 1024;         // codes per strip (2048 weights = 64 MXFP4 blocks)
constexpr int32_t NZ_S_NB = 64;             // blocks per strip
constexpr int32_t NZ_S_IN = 2048;           // weights per strip
constexpr int32_t NZ_S_BLK_BYTES = 1088;    // 64 * 17 bytes (exact 32B multiple: 34 * 32)
constexpr int32_t NZ_STRIP_ROWS = 32;
constexpr int32_t NZ_OUT_BATCH_PAIRS = 4;   // 4 pairs = 128 cols = 8 fractal tiles = 4,096 B burst

extern "C" __global__ __aicore__ void mxfp4_fused_blk_nz(
    GM_ADDR blocks, GM_ADDR outg, GM_ADDR oscaleg,
    GM_ADDR lutLoG, GM_ADDR lutHiG, GM_ADDR lutE8G,
    GM_ADDR scOffG, GM_ADDR codeOffG, GM_ADDR scaleOffG,
    uint32_t Ec, uint32_t OUT, uint32_t IN)
{
    const uint32_t blkid = (uint32_t)GetBlockIdx();
    const uint32_t nblk = (uint32_t)GetBlockNum();
    const uint32_t n1Count = OUT / NZ_STRIP_ROWS;
    const uint32_t m1Count = IN / 16;
    const uint32_t totalStrips = Ec * n1Count;
    const uint32_t numStripsIn = IN / NZ_S_IN;
    const uint32_t nb17 = (IN / 32) * 17;

    GlobalTensor<uint8_t> gBlocks;
    GlobalTensor<int8_t> gOut;
    GlobalTensor<float> gOscale, gLutLo, gLutHi, gLutE8;
    GlobalTensor<uint32_t> gScOff, gCodeOff, gScaleOff;
    gBlocks.SetGlobalBuffer((__gm__ uint8_t *)blocks);
    gOut.SetGlobalBuffer((__gm__ int8_t *)outg);
    gOscale.SetGlobalBuffer((__gm__ float *)oscaleg);
    gLutLo.SetGlobalBuffer((__gm__ float *)lutLoG);
    gLutHi.SetGlobalBuffer((__gm__ float *)lutHiG);
    gLutE8.SetGlobalBuffer((__gm__ float *)lutE8G);
    gScOff.SetGlobalBuffer((__gm__ uint32_t *)scOffG);
    gCodeOff.SetGlobalBuffer((__gm__ uint32_t *)codeOffG);
    gScaleOff.SetGlobalBuffer((__gm__ uint32_t *)scaleOffG);

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> qBlk;
    TQue<QuePosition::VECOUT, 2> qOut;
    pipe.InitBuffer(qBlk, 1, NZ_S_BLK_BYTES * sizeof(uint8_t));               // 1,088 B
    pipe.InitBuffer(qOut, 2, NZ_OUT_BATCH_PAIRS * 1024 * sizeof(int8_t));     // 8,192 B (2 x 4 KB)

    TBuf<TPosition::VECCALC> tQRows, tRowF32, tScratch12K;
    TBuf<TPosition::VECCALC> tLutLo, tLutHi, tLutE8, tScOff, tCodeOff, tScaleOff;
    TBuf<TPosition::VECCALC> tBlkH, tOffH, tScI, tScF, tAcc;
    pipe.InitBuffer(tQRows, NZ_STRIP_ROWS * IN_MAX * sizeof(int8_t));         // 131,072 B (128 KB)
    pipe.InitBuffer(tRowF32, IN_MAX * sizeof(float));                         // 16,384 B (16 KB)
    pipe.InitBuffer(tScratch12K, 3 * NZ_S_HALF * sizeof(float));              // 12,288 B (12 KB)
    pipe.InitBuffer(tLutLo, 256 * sizeof(float));                             // 1,024 B
    pipe.InitBuffer(tLutHi, 256 * sizeof(float));                             // 1,024 B
    pipe.InitBuffer(tLutE8, 256 * sizeof(float));                             // 1,024 B
    pipe.InitBuffer(tScOff, NZ_S_HALF * sizeof(uint32_t));                    // 4,096 B
    pipe.InitBuffer(tCodeOff, NZ_S_HALF * sizeof(uint32_t));                  // 4,096 B
    pipe.InitBuffer(tScaleOff, NZ_S_NB * sizeof(uint32_t));                   // 256 B
    pipe.InitBuffer(tBlkH, ((NZ_S_BLK_BYTES + 15) / 16 * 16) * sizeof(half)); // 2,176 B
    pipe.InitBuffer(tOffH, NZ_S_HALF * sizeof(half));                         // 2,048 B
    pipe.InitBuffer(tScI, NZ_S_NB * sizeof(int32_t));                         // 256 B
    pipe.InitBuffer(tScF, NZ_S_NB * sizeof(float));                           // 256 B
    pipe.InitBuffer(tAcc, NZ_STRIP_ROWS * sizeof(float));                     // 128 B

    LocalTensor<int8_t> qRows = tQRows.Get<int8_t>();
    LocalTensor<float> rowF32 = tRowF32.Get<float>();
    LocalTensor<float> lutLo = tLutLo.Get<float>();
    LocalTensor<float> lutHi = tLutHi.Get<float>();
    LocalTensor<float> lutE8 = tLutE8.Get<float>();
    LocalTensor<uint32_t> scOff = tScOff.Get<uint32_t>();
    LocalTensor<uint32_t> codeOff = tCodeOff.Get<uint32_t>();
    LocalTensor<uint32_t> scaleOff = tScaleOff.Get<uint32_t>();
    LocalTensor<half> blkH = tBlkH.Get<half>();
    LocalTensor<half> offH = tOffH.Get<half>();
    LocalTensor<int32_t> scI = tScI.Get<int32_t>();
    LocalTensor<float> scF = tScF.Get<float>();
    LocalTensor<float> acc = tAcc.Get<float>();

    // Phase 1 partition of tScratch12K (12 KB = 3 x 4 KB):
    //   slot 0 [0..4095]:     rowMax (float[1024])
    //   slot 1 [4096..8191]:  absb   (float[1024])
    //   slot 2 [8192..12287]: off (int32[1024]) -> scHalf (float[1024]) -> work (float[1024]) -> outH (half[2048])
    LocalTensor<float> s12F = tScratch12K.Get<float>();
    LocalTensor<int32_t> s12I = tScratch12K.Get<int32_t>();
    LocalTensor<half> s12H = tScratch12K.Get<half>();
    LocalTensor<float> rowMax = s12F[0];
    LocalTensor<float> absb = s12F[NZ_S_HALF];
    LocalTensor<int32_t> off = s12I[2 * NZ_S_HALF];
    LocalTensor<float> scHalf = s12F[2 * NZ_S_HALF];
    LocalTensor<float> work = s12F[2 * NZ_S_HALF];
    LocalTensor<half> outH = s12H[4 * NZ_S_HALF];

    // Phase 2 partition (reuses dead tRowF32 [16 KB] + tScratch12K [12 KB]):
    //   In tRowF32 (16 KB):
    //     b128A: 2 KB [0..2047],     b128B: 2 KB [2048..4095]
    //     h128A: 4 KB [4096..8191],  h128B: 4 KB [8192..12287]  (16 rows x 128 cols)
    //     trA:   4 KB [12288..16383] (8 tiles of 16x16)
    //   In tScratch12K (12 KB):
    //     trB:   4 KB [0..4095]      (8 tiles of 16x16)
    //     combH: 8 KB [4096..12287]  (8 tiles of 16x32)
    LocalTensor<int8_t> rBytes = tRowF32.Get<int8_t>();
    LocalTensor<half> rHalves = tRowF32.Get<half>();
    LocalTensor<int8_t> b128A = rBytes[0];
    LocalTensor<int8_t> b128B = rBytes[2048];
    LocalTensor<half> h128A = rHalves[2048];
    LocalTensor<half> h128B = rHalves[4096];
    LocalTensor<half> trA = rHalves[6144];
    LocalTensor<half> trB = s12H[0];
    LocalTensor<half> combH = s12H[2048];

    DataCopy(lutLo, gLutLo, 256);
    DataCopy(lutHi, gLutHi, 256);
    DataCopy(lutE8, gLutE8, 256);
    DataCopy(scOff, gScOff, NZ_S_HALF);
    DataCopy(codeOff, gCodeOff, NZ_S_HALF);
    DataCopy(scaleOff, gScaleOff, NZ_S_NB);
    PipeBarrier<PIPE_ALL>();

    DataCopyParams cpHalfInterleave;
    cpHalfInterleave.blockCount = NZ_S_NB;
    cpHalfInterleave.blockLen = 1;
    cpHalfInterleave.srcStride = 0;
    cpHalfInterleave.dstStride = 1;

    DataCopyParams cpGather128;
    cpGather128.blockCount = 16;
    cpGather128.blockLen = 4;             // 128 int8 = 4 x 32B
    cpGather128.srcStride = (uint16_t)((IN - 128) / 32);
    cpGather128.dstStride = 0;

    DataCopyParams cpMerge128;
    cpMerge128.blockCount = 128;          // 8 tiles x 16 rows of 16 halves (32B)
    cpMerge128.blockLen = 1;
    cpMerge128.srcStride = 0;
    cpMerge128.dstStride = 1;

    TransDataTo5HDParams transParams8;
    transParams8.dstHighHalf = false;
    transParams8.srcHighHalf = false;
    transParams8.repeatTimes = 8;         // 8 tiles of 16x16 in one hardware instruction
    transParams8.srcRepStride = 1;        // advance 16 halves (1 block) along 128-col row
    transParams8.dstRepStride = 16;       // advance 16x16 halves (16 blocks) per output tile

    LocalTensor<half> srcListA[16], dstListA[16];
    LocalTensor<half> srcListB[16], dstListB[16];
    for (int32_t i = 0; i < 16; i++) {
        srcListA[i] = h128A[i * 128];
        dstListA[i] = trA[i * 16];
        srcListB[i] = h128B[i * 128];
        dstListB[i] = trB[i * 16];
    }

    for (uint32_t strip = blkid; strip < totalStrips; strip += nblk) {
        const uint32_t e = strip / n1Count;
        const uint32_t n1 = strip % n1Count;
        const uint32_t r0 = e * OUT + n1 * NZ_STRIP_ROWS;

        // Phase 1: Decode & quantize all 32 rows into qRows[32, IN] (final halfblock K-order).
        for (uint32_t k = 0; k < NZ_STRIP_ROWS; k++) {
            const uint32_t r = r0 + k;

            for (uint32_t si = 0; si < numStripsIn; si++) {
                LocalTensor<float> vlo = rowF32[si * NZ_S_IN];
                LocalTensor<float> vhi = rowF32[si * NZ_S_IN + NZ_S_HALF];

                LocalTensor<uint8_t> bu = qBlk.AllocTensor<uint8_t>();
                DataCopy(bu, gBlocks[(uint64_t)r * nb17 + (uint64_t)si * NZ_S_BLK_BYTES], NZ_S_BLK_BYTES);
                qBlk.EnQue(bu);
                LocalTensor<uint8_t> buU = qBlk.DeQue<uint8_t>();
                Cast(blkH, buU, RoundMode::CAST_NONE, NZ_S_BLK_BYTES);
                qBlk.FreeTensor(buU);
                PipeBarrier<PIPE_V>();

                Gather(offH, blkH, codeOff, (uint32_t)0, NZ_S_HALF);
                Muls(offH, offH, (half)4.0, NZ_S_HALF);
                Cast(off, offH, RoundMode::CAST_RINT, NZ_S_HALF);
                LocalTensor<uint32_t> offU = off.ReinterpretCast<uint32_t>();
                Gather(vlo, lutLo, offU, (uint32_t)0, NZ_S_HALF);
                Gather(vhi, lutHi, offU, (uint32_t)0, NZ_S_HALF);

                Gather(offH, blkH, scaleOff, (uint32_t)0, NZ_S_NB);
                Muls(offH, offH, (half)4.0, NZ_S_NB);
                Cast(scI, offH, RoundMode::CAST_RINT, NZ_S_NB);
                Gather(scF, lutE8, scI.ReinterpretCast<uint32_t>(), (uint32_t)0, NZ_S_NB);
                PipeBarrier<PIPE_V>();
                Gather(scHalf, scF, scOff, (uint32_t)0, NZ_S_HALF);
                Mul(vlo, vlo, scHalf, NZ_S_HALF);
                Mul(vhi, vhi, scHalf, NZ_S_HALF);
                PipeBarrier<PIPE_V>();

                Abs(absb, vlo, NZ_S_HALF);
                Abs(work, vhi, NZ_S_HALF);
                PipeBarrier<PIPE_V>();
                if (si == 0) {
                    Max(rowMax, absb, work, NZ_S_HALF);
                } else {
                    Max(absb, absb, work, NZ_S_HALF);
                    PipeBarrier<PIPE_V>();
                    Max(rowMax, rowMax, absb, NZ_S_HALF);
                }
                PipeBarrier<PIPE_V>();
            }

            // Single reduction & single PIPE_ALL per row.
            LocalTensor<float> fa = rowMax, fb = absb;
            for (uint32_t h = NZ_S_HALF >> 1; h >= 8; h >>= 1) {
                Max(fb, fa, fa[h], h);
                PipeBarrier<PIPE_V>();
                LocalTensor<float> tmp = fa; fa = fb; fb = tmp;
            }
            PipeBarrier<PIPE_ALL>();
            float rowAmax = fa.GetValue(0);
            for (int32_t i = 1; i < 8; i++) {
                float v = fa.GetValue(i);
                if (v > rowAmax) rowAmax = v;
            }
            if (rowAmax < 1e-8f) rowAmax = 1e-8f;
            acc.SetValue(k, rowAmax / 127.0f);
            const float inv = 127.0f / rowAmax;
            PipeBarrier<PIPE_ALL>();

            for (uint32_t si = 0; si < numStripsIn; si++) {
                LocalTensor<float> vlo = rowF32[si * NZ_S_IN];
                LocalTensor<float> vhi = rowF32[si * NZ_S_IN + NZ_S_HALF];

                Muls(vlo, vlo, inv, NZ_S_HALF);
                Muls(vhi, vhi, inv, NZ_S_HALF);
                PipeBarrier<PIPE_V>();
                Mins(vlo, vlo, 127.0f, NZ_S_HALF); Maxs(vlo, vlo, -127.0f, NZ_S_HALF);
                Mins(vhi, vhi, 127.0f, NZ_S_HALF); Maxs(vhi, vhi, -127.0f, NZ_S_HALF);
                PipeBarrier<PIPE_V>();

                Cast(offH, vlo, RoundMode::CAST_NONE, NZ_S_HALF);
                PipeBarrier<PIPE_V>();
                DataCopy(outH[0], offH, cpHalfInterleave);
                PipeBarrier<PIPE_V>();
                Cast(offH, vhi, RoundMode::CAST_NONE, NZ_S_HALF);
                PipeBarrier<PIPE_V>();
                DataCopy(outH[16], offH, cpHalfInterleave);
                PipeBarrier<PIPE_V>();

                Cast(qRows[k * IN + si * NZ_S_IN], outH, RoundMode::CAST_RINT, NZ_S_IN);
                PipeBarrier<PIPE_V>();
            }
        }

        // Flush 32 oscales (128B) to GM.
        PipeBarrier<PIPE_ALL>();
        DataCopy(gOscale[r0], acc, NZ_STRIP_ROWS);
        PipeBarrier<PIPE_ALL>();

        // Phase 2: Batched 128-column transpose -> 4,096 B contiguous FRACTAL_NZ bursts.
        const uint64_t nzStripBase = (uint64_t)(e * n1Count + n1) * m1Count * 512;
        const uint32_t nPairs = IN / 32;
        for (uint32_t pBase = 0; pBase < nPairs; pBase += NZ_OUT_BATCH_PAIRS) {
            DataCopy(b128A, qRows[pBase * 32], cpGather128);
            DataCopy(b128B, qRows[16 * IN + pBase * 32], cpGather128);
            PipeBarrier<PIPE_V>();

            Cast(h128A, b128A, RoundMode::CAST_NONE, 2048);
            Cast(h128B, b128B, RoundMode::CAST_NONE, 2048);
            PipeBarrier<PIPE_V>();

            TransDataTo5HD<half>(dstListA, srcListA, transParams8);
            TransDataTo5HD<half>(dstListB, srcListB, transParams8);
            PipeBarrier<PIPE_V>();

            DataCopy(combH[0], trA, cpMerge128);
            DataCopy(combH[16], trB, cpMerge128);
            PipeBarrier<PIPE_V>();

            LocalTensor<int8_t> outBatch = qOut.AllocTensor<int8_t>();
            Cast(outBatch, combH, RoundMode::CAST_NONE, 4096);
            PipeBarrier<PIPE_V>();
            qOut.EnQue(outBatch);
            LocalTensor<int8_t> outReady = qOut.DeQue<int8_t>();
            DataCopy(gOut[nzStripBase + (uint64_t)pBase * 1024], outReady, 4096);
            qOut.FreeTensor(outReady);
        }
        PipeBarrier<PIPE_ALL>();
    }
}

extern "C" void launch_mxfp4_fused_blk_nz(void *stream, uint32_t blockdim,
    uint8_t *blocks, int8_t *out_nz, float *oscale,
    float *lutLo, float *lutHi, float *lutE8,
    uint32_t *scOff, uint32_t *codeOff, uint32_t *scaleOff,
    uint32_t Ec, uint32_t OUT, uint32_t IN)
{
    mxfp4_fused_blk_nz<<<blockdim, nullptr, stream>>>(
        (GM_ADDR)blocks, (GM_ADDR)out_nz, (GM_ADDR)oscale,
        (GM_ADDR)lutLo, (GM_ADDR)lutHi, (GM_ADDR)lutE8,
        (GM_ADDR)scOff, (GM_ADDR)codeOff, (GM_ADDR)scaleOff,
        Ec, OUT, IN);
}
