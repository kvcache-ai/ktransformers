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
