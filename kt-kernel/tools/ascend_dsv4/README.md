# DeepSeek-V4-Flash single-NPU deployment scripts

Companion scripts for [`doc/en/DeepSeek-V4-Flash_tutorial_for_Ascend_NPU.md`](../../../doc/en/DeepSeek-V4-Flash_tutorial_for_Ascend_NPU.md).
Read the tutorial first — it explains what each step is for and what to check
afterwards. This file is the reference for the scripts themselves.

Everything is driven by [`dsv4_env.sh`](./dsv4_env.sh). It auto-detects the CANN
root and version, the NUMA node count, the SoC and the Python interpreter, and
every value it derives can be overridden from the environment. Nothing is
hardcoded to a particular machine.

## Order

```bash
export DSV4_MODEL_ROOT=/path/to/models
export DSV4_NPU_DEVICE_ID=0

bash dsv4_env.sh --show          # 1. check what got detected
bash build_cann_ops.sh all       # 2. only if the CANN vendors are missing
bash build_kt_kernel.sh          # 3. build kt-kernel, produce a wheel
bash build_sgl_kernel_npu.sh     # 4. only on a native install (images ship it)
bash install_python_deps.sh      # 5. SGLang's NPU deps, without touching torch
bash convert_mxfp4_gguf.sh       # 6. build + verify the per-layer GGUF set
bash preflight.sh                # 7. must print PREFLIGHT OK
bash serve.sh                    # 8. launch
bash verify.sh                   # 9. four acceptance gates
```

Steps 2, 4 and 6 are the slow ones (40–90 min, 20–40 min, and a few hours) and
all three are skippable when the artifacts already exist.

## Scripts

| Script | What it does |
|--------|--------------|
| `dsv4_env.sh` | Single source of truth. Sourced by every other script. `--show` prints the resolved configuration without changing your shell. |
| `build_cann_ops.sh` | Builds and installs the `customize` and `custom_transformer` CANN vendor packages plus the `custom_ops` torch bindings, from two pinned Gitcode commits. Takes `all` (default), `customize`, `custom_ops` or `transformer`. |
| `build_kt_kernel.sh` | Builds kt-kernel with `CPUINFER_USE_ASCEND_NPU=1` and produces a wheel under `${DSV4_ARTIFACT_DIR}/wheels`. `inplace` builds the extension in tree instead. |
| `build_sgl_kernel_npu.sh` | Builds and installs `sgl_kernel_npu`, `deep_ep`, `attentions` and `torch_memory_saver` from a pinned tag. Container images ship these; a native install must build them, and the import is not optional. |
| `install_python_deps.sh` | Installs SGLang's NPU dependency list from `pyproject_npu.toml`, constrained to the torch family already present. `--dry-run` shows the list. |
| `convert_mxfp4_gguf.sh` | Converts the official checkpoint into the per-layer MXFP4 GGUF set and verifies it. `verify` re-runs verification only. |
| `preflight.sh` | Checks everything the server needs. Exit 0 means safe to launch. |
| `serve.sh` | Launches the server. `--foreground` stays attached. |
| `verify.sh` | Four acceptance gates against a running server. |

The GGUF conversion and verification tools themselves live in
[`../mxfp4_gguf/`](../mxfp4_gguf) and can be run directly.

## Things these scripts exist to prevent

Each of these is a failure that is silent, or reported somewhere far from its
cause. They are the reason this is a set of scripts rather than a list of
commands in a document.

- **`PYTHONPATH` ordering.** The CANN `set_env.sh` scripts prepend their own
  entries, so a `PYTHONPATH` exported before them loses. The server then starts,
  answers normally, and runs a different SGLang than your clone.
  `dsv4_env.sh` exports it last; `preflight.sh` and `serve.sh` both assert on
  `sglang.__file__`.
- **`PYTHONNOUSERSITE` on a non-root host.** Disabling user-site keeps a stray
  `~/.local` from shadowing an image's packages, but where you are not root pip
  installs *into* `~/.local` — so `dsv4_env.sh` only sets it when the
  interpreter's own `site-packages` is writable.
- **The `{layer_idx}` placeholder.** Writing the template inside a
  `${VAR:-default}` expansion silently truncates it to
  `dsv4_layer{layer_idx_mxfp4.gguf}`, because bash reads the first `}` as the end
  of the expansion. `dsv4_env.sh` uses a plain assignment; `preflight.sh` checks
  the placeholder survived.
- **The compiler CMake actually uses.** `kt-kernel/CMakeLists.txt` may
  force-select `/usr/bin/gcc`, in which case exporting `CC` does nothing.
  `build_kt_kernel.sh` detects which behaviour the checkout has and fails early,
  with the remedy, if the effective GCC is older than 11 (kt-kernel is C++20 and
  needs `<barrier>`).
- **ARM `-march` extensions.** `setup.py` enables `+bf16`/`+i8mm` from
  `/proc/cpuinfo`; GCC < 10 cannot encode them, and the SVE path of the llamafile
  sgemm has no MXFP4 tile so decode dies with `llamafile not supported`. All
  three default to OFF, which is the validated NEON configuration.
- **Truncated GGUF files.** An interrupted conversion leaves a short file that
  loads and serves and produces wrong tokens. `verify_mxfp4_gguf_set.py` checks
  exact sizes, optionally sha256, and optionally bit-exactness against the
  source checkpoint.
- **aarch64 static TLS.** `libgomp` needs static TLS, and by the time SGLang's
  import chain dlopen()s it the loader's surplus is gone —
  `cannot allocate memory in static TLS block`. `dsv4_env.sh` preloads libgomp so
  it lands in the initial link set.
- **A `torchaudio` built against another torch.** `transformers` imports it
  unconditionally through `loss_rnnt`, so an ABI mismatch takes down the whole
  stack with `undefined symbol: torch_library_impl`. `install_python_deps.sh`
  pins only `torch` and `torch_npu` — never a mismatched companion — and
  `preflight.sh` imports the companions to catch it early.
- **HTTP 200 as an acceptance signal.** With a broken CPU expert path the server
  still binds its port and answers health probes. `verify.sh` requires a
  non-empty greedy completion.
- **Silent streaming-prefill fallback.** When streaming is enabled but a layer
  OOMs, it falls back to hybrid per layer and answers requests normally.
  `verify.sh` checks `inline resident > 0` and `hybrid fallback == 0`.
