# llama.cpp (b3173) patches required by kt-kernel

kt-kernel pins `third_party/llama.cpp` at tag `b3173`, which predates two things the
CPU-MoE path needs:

- `0001-gguf-py-numpy2-reader.patch` — makes the in-tree `gguf-py` reader work under
  NumPy 2.x (used by the GGUF conversion/verification tooling that runs against the
  in-tree `gguf-py`).
- `0002-ggml-mxfp4-type.patch` — adds the `GGML_TYPE_MXFP4` block type (16-value table,
  E8M0 block scale, 17-byte blocks) plus NEON/scalar `vec_dot` paths. Byte-compatible
  with the MXFP4 format modern llama.cpp ships; backported here because the pinned
  tree cannot be bumped (kt-kernel couples to the b3173 source layout).

Apply once after checking out submodules:

    cd third_party/llama.cpp
    git apply ../../kt-kernel/third_party_patches/llama.cpp/*.patch

The kt-kernel CMake build applies these automatically at configure time when the
tree is unpatched (see `kt-kernel/CMakeLists.txt`).
