# Experimental RAWINT4/KGroup SFT

## Important: compile this branch yourself

This directory documents the experimental Kimi-K2 RAWINT4 training path that
was exercised on qj5090 and sap4. The delivery point is:

- repository: `yyj6666667/sft-profiling-and-perf` (private)
- branch: `share/rawint4-kgroup-sft-qj5090`
- base commit: `880a27b4588d705b49be1601805c0ab48ae2b495`
- package version in this branch: `0.6.1.post1`

**Do not replace this checkout with `pip install ktransformers[sft]`.** The
current public KT 0.7.x build does not expose the RAWINT4/KGroup SFT native
binding used here. A working installation must compile `kt-kernel` from this
branch with AMX enabled. Installing a public wheel after the local build can
silently replace the experimental extension.

This is an experimental research branch, not a supported release. Build on the
target machine: a `-march=native` binary is not portable across CPU models.

## What is implemented

The YAML backend name `AMXINT4_KGroup` resolves through all three required
layers:

1. `kt-kernel/python/sft/wrapper.py` maps it to
   `AMXINT4_KGroup_SFT`.
2. `kt-kernel/python/sft/amx.py` imports and selects
   `AMXInt4_KGroup_SFT_MOE`.
3. `kt-kernel/ext_bindings.cpp` exports the native
   `AMXInt4_KGroup_SFT_MOE` binding.

It consumes native compressed-tensors, pack-quantized routed-expert weights:

- `weight_packed`: INT32 packed RAWINT4
- `weight_scale`: BF16
- group size: 32 for the validated Kimi checkpoints

Attention, embeddings, `lm_head`, shared experts, activations, and LoRA compute
remain BF16. Therefore `bf16: true` in the training YAML does **not** mean that
routed-expert base weights are expanded to BF16.

## Evidence

### qj5090: Kimi-K2.6

The four-rank FSDP2 run used the two K2.6 YAML files in this directory. It
wrapped 60 MoE layers and executed complete forward, loss, backward, and
optimizer steps. Recorded samples include finite losses `0.5126`, `0.5128`,
and `0.4266`, with nonzero gradient norms `0.4367`, `0.4494`, and `0.3482`.

Original run records on qj5090:

```text
/mnt/sft_yyj_yyj/kimi-k2-sft-smoke/configs/train/kimi_k26_sft_tp2_fused_20260710T092135Z_kimi_k26_r8a32_10step_synth128_cutoff512_cuda4567.yaml
/mnt/sft_yyj_yyj/kimi-k2-sft-smoke/configs/accelerate/fsdp2_kimi_k26_4gpu_tp2_kgroup_fused_20260710T092135Z_kimi_k26_r8a32_10step_synth128_cutoff512_cuda4567.yaml
/mnt/sft_yyj_yyj/kimi-k2-sft-smoke/logs/train/kimi_k26_tp2_fused_20260710T092135Z_kimi_k26_r8a32_10step_synth128_cutoff512_cuda4567.log
```

### sap4: Kimi-K2.5

The four-rank FSDP2 K2.5 configuration completed one optimizer step with loss
`2.159`; the recorded training runtime was `888.6 s`. The two K2.5 YAML files
in this directory retain that run's training semantics while replacing private
output paths with placeholders.

## Hardware and software boundary

- Linux x86_64 with Intel AMX (`amx_tile`, `amx_int8`, and preferably
  `amx_bf16` in `/proc/cpuinfo`).
- Sufficient host RAM for the checkpoint, FSDP state, activations, and native
  workspaces.
- The validated dependency family was Python 3.11, `ktransformers==0.6.1.post1`,
  `kt-kernel==0.6.1.post1`, `transformers-kt==5.6.0.post1`, and
  `accelerate-kt==1.14.0.post1`.
- Use a LLaMA-Factory checkout that supports `use_kt`, Accelerate `kt_config`,
  and compressed-tensors checkpoints under KT KGroup + FSDP2. A stock checkout
  that rejects compressed-tensors PTQ under FSDP is not compatible with this
  path.

## Build and verify

Choose the PyTorch/CUDA build required by the target GPUs before compiling.
The following commands deliberately install the native kernel first and the
lightweight root package with `--no-deps`, so pip cannot substitute the public
`kt-kernel` wheel.

```bash
git clone --branch share/rawint4-kgroup-sft-qj5090 \
  https://github.com/yyj6666667/sft-profiling-and-perf.git
cd sft-profiling-and-perf

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel cmake ninja pybind11 packaging

# Install the PyTorch build required by the target CUDA stack first.
# Example only; choose the correct index/version for the machine:
# python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130

python -m pip install \
  transformers-kt==5.6.0.post1 \
  accelerate-kt==1.14.0.post1

cd kt-kernel
python -m pip install -r requirements.txt
export CPUINFER_CPU_INSTRUCT=NATIVE
export CPUINFER_ENABLE_AMX=ON
export CPUINFER_BUILD_TYPE=Release
export CPUINFER_PARALLEL=16
export KT_KERNEL_PIP_NO_BUILD_ISOLATION=1
./install.sh build --manual
cd ..

python -m pip install --no-deps -e .
```

The build is accepted only if the native class imports successfully:

```bash
python -c "from kt_kernel_ext.moe import AMXInt4_KGroup_SFT_MOE; print('RAWINT4_KGROUP_SFT_OK')"
python -c "from kt_kernel.sft.amx import _HAS_AMX_SFT_SUPPORT; assert _HAS_AMX_SFT_SUPPORT"
```

Also verify that Python is loading this checkout rather than another wheel:

```bash
python -c "import kt_kernel, kt_kernel.sft.wrapper as w; print(kt_kernel.__file__); print(w.__file__)"
```

## Configure and launch

The training YAML owns the LLaMA-Factory settings and top-level `use_kt`
switches. In this 0.6.1 integration, the Accelerate YAML owns `kt_config`.
Do not move `kt_config` to the training YAML without updating the integration.

Before launching, replace these values:

- `/path/to/Kimi-K2.5` or `/path/to/Kimi-K2.6` with the native RAWINT4
  checkpoint directory;
- `/path/to/output/...` with a writable output directory;
- `synthetic_over4192_512` with a dataset registered in LLaMA-Factory's
  `data/dataset_info.json`, if that synthetic dataset is unavailable;
- `num_processes`, LoRA rank/alpha, CPU thread count, and model length only when
  the corresponding values are changed in both YAML files.

Launch from a compatible LLaMA-Factory checkout:

```bash
export ACCELERATE_USE_KT=true
export ACCELERATE_KT_BACKEND=AMXINT4_KGroup

accelerate launch \
  --config_file /path/to/fsdp2_kimi_k26_rawint4_4gpu.yaml \
  /path/to/LLaMA-Factory/src/train.py \
  /path/to/kimi_k26_rawint4_sft.yaml
```

For K2.5, substitute the two `k25` YAML files.

## Failure checks

- `cannot import name AMXInt4_KGroup_SFT_MOE`: the public/prebuilt kernel was
  loaded, AMX was disabled at compile time, or the wrong Python environment is
  active. Rebuild and inspect `kt_kernel.__file__`.
- Fallback to `AMXBF16_SFT`: the backend string was not read. Confirm
  `ACCELERATE_USE_KT=true` and that `kt_config` remains in the Accelerate YAML.
- compressed-tensors rejected under FSDP: use the KT-compatible LLaMA-Factory
  integration; this is not fixed by changing the model precision in YAML.
- `is_torch_fx_available` import failure in Kimi remote model code: align the
  checkpoint's remote Python code with the pinned `transformers-kt` version.
  This is a model-code/Transformers compatibility issue, not a RAWINT4 kernel
  failure.
- Do not replace `AMXINT4_KGroup` with `AMXINT4`: they are different packing
  layouts and training backends.
