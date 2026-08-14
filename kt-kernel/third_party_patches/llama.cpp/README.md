# llama.cpp patches

`third_party/llama.cpp` is pinned to the upstream commit
`a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f`, which is `refs/tags/b3173` in
<https://github.com/ggerganov/llama.cpp>. The pin deliberately points at a
commit that exists upstream, so `git clone --recursive` of this repository
works for everyone. Local-only fork commits must **not** be pinned here.

b3173 predates MXFP4, so the deltas kt-kernel needs on top of it live in this
directory as a patch series and are applied at configure time by
`kt-kernel/CMakeLists.txt` (right before `add_subdirectory(... llama.cpp)`).

## Patches

### `0001-ggml-mxfp4-type.patch`

Adds the OCP microscaling FP4 type (`GGML_TYPE_MXFP4`, id 39 — the same id
upstream llama.cpp later assigned) to ggml:

* `ggml-common.h` — `block_mxfp4` (one `ue8m0` byte + 16 nibble-packed `E2M1`
  codes per 32 weights), matching the upstream half-block interleave.
* `ggml-quants.{c,h}` — `dequantize_row_mxfp4` and `ggml_vec_dot_mxfp4_q8_0`
  (NEON dot-product path that avoids SVE/i8mm so it runs on Kunpeng 920, plus a
  scalar fallback for every other architecture).
* `ggml.c` / `ggml.h` — the enum value and the `type_traits` entry wiring the
  dequantizer and the `vec_dot` (`vec_dot_type = GGML_TYPE_Q8_0`).
* `gguf-py/gguf/constants.py` — `GGMLQuantizationType.MXFP4` and its
  `GGML_QUANT_SIZES` entry.

This is required because the DeepSeek-V4 GGUF stores its CPU-offloaded expert
tensors as MXFP4; without it ggml rejects the tensor type and kt-kernel cannot
load the experts.

## Applying by hand

The build applies these automatically. To do it manually (e.g. when debugging):

```bash
cd third_party/llama.cpp
for p in ../../kt-kernel/third_party_patches/llama.cpp/*.patch; do
    git apply "$p"
done
```

To revert:

```bash
cd third_party/llama.cpp
for p in $(ls -r ../../kt-kernel/third_party_patches/llama.cpp/*.patch); do
    git apply --reverse "$p"
done
```

The CMake step is idempotent and needs no marker file: for each patch it first
tries `git apply --check`; if that fails it tries `git apply --reverse --check`,
and a success there means the patch is already applied, so it is skipped. Only
when both checks fail does the configure step abort.
