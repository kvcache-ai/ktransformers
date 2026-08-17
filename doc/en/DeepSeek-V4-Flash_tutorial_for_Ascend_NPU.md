# Running DeepSeek-V4-Flash on a Single Ascend NPU

This tutorial demonstrates how to run **DeepSeek-V4-Flash** inference on a **single Ascend NPU die** using SGLang with KT-Kernel CPU expert offload. Attention, the dense layers and a small resident subset of experts run on the NPU in W8A8; the remaining routed experts are served from the host CPU out of per-layer MXFP4 GGUF files through the KT-Kernel `LLAMAFILE` MoE kernel.

Every step below is driven by a script in [`kt-kernel/tools/ascend_dsv4/`](../../kt-kernel/tools/ascend_dsv4), and every path is a variable — nothing is hardcoded to the machine this was validated on.

> **Important:** This deployment needs a KT-Kernel built with the **Ascend NPU backend** (`CPUINFER_USE_ASCEND_NPU=1`) and the MXFP4 llama.cpp patch series it carries. If `kt-kernel/cpu_backend/vendors/ascend_npu.h` is not in your checkout, that support is not present yet and [Step 4](#step-4-build-kt-kernel) will not produce a usable wheel.

## Table of Contents

- [Running DeepSeek-V4-Flash on a Single Ascend NPU](#running-deepseek-v4-flash-on-a-single-ascend-npu)
  - [Table of Contents](#table-of-contents)
  - [Hardware Requirements](#hardware-requirements)
  - [Software Stack](#software-stack)
  - [What You Need Before You Start](#what-you-need-before-you-start)
  - [Step 1: Get the Code](#step-1-get-the-code)
  - [Step 2: Configure Your Environment](#step-2-configure-your-environment)
  - [Step 3: Build the CANN Custom Operator Packages](#step-3-build-the-cann-custom-operator-packages)
  - [Step 4: Build KT-Kernel](#step-4-build-kt-kernel)
  - [Step 5: Build sgl-kernel-npu](#step-5-build-sgl-kernel-npu)
  - [Step 6: Install the Python Dependencies](#step-6-install-the-python-dependencies)
  - [Step 7: Prepare the Weights](#step-7-prepare-the-weights)
  - [Step 8: Preflight](#step-8-preflight)
  - [Step 9: Launch the Server](#step-9-launch-the-server)
  - [Step 10: Acceptance Checks](#step-10-acceptance-checks)
  - [Step 11: Talk to the Model](#step-11-talk-to-the-model)
  - [Step 12: Accuracy Validation (GPQA-Diamond)](#step-12-accuracy-validation-gpqa-diamond)
  - [Step 13: Throughput Validation](#step-13-throughput-validation)
  - [Tuning](#tuning)
    - [Resident Experts vs. the KV Pool](#resident-experts-vs-the-kv-pool)
    - [Optional: Streaming Prefill](#optional-streaming-prefill)
  - [Measured Results](#measured-results)
  - [Troubleshooting](#troubleshooting)
  - [Known Limitations](#known-limitations)
  - [Additional Resources](#additional-resources)

## Hardware Requirements

**Validated Configuration (this tutorial):**
- **NPU**: 1× Atlas 800I A2 (910B3) die, 64 GB HBM
- **CPU**: Kunpeng 920, 192 cores, 8 NUMA nodes (aarch64)
- **RAM**: ≥200 GB — the offloaded experts alone hold about 140 GB resident
- **Storage**: ~840 GB for all three weight artifacts (see [Step 7](#step-7-prepare-the-weights))

**Supported NPU platforms:**

| Platform | SoC name (`DSV4_SOC`) | HBM per die | Host | End-to-end validated |
|----------|----------------------|-------------|------|----------------------|
| Atlas 800I A2 (910B3) | `ascend910b` | 64 GB | Kunpeng 920, 192 cores, 8 NUMA, 1.5 TB RAM | ✓ container image |
| Atlas A3 (910_93 series) | `ascend910_93` | 61 GB | 40 cores, **1 NUMA**, 229 GB RAM | ✓ native install |

Both were walked end to end with the scripts in this tutorial. The throughput and
accuracy tables below are from the A2 host; the A3 host has one NUMA node and a
fifth of the cores, so its numbers are not comparable and are not published here.

> **Note:** The A2 platform above is a container (Ubuntu 22.04, GCC 11) and the A3
> one is a native install (Ubuntu 20.04). That difference is about the host image,
> not the NPU: the GCC requirement in [Software Stack](#software-stack) is what
> decides whether a given machine can build KT-Kernel, and Ubuntu 20.04's stock
> GCC 9.4 does not meet it.

> **Note:** Only **one** die is used. Tensor parallelism, expert parallelism and A2A dispatch are rejected at startup when KT CPU offload is active — see [Known Limitations](#known-limitations).

The single most important host property is memory **bandwidth**, not core count. Decode is dominated by streaming expert weights out of DRAM every token, so a busy neighbour on a shared host changes measured throughput by 5–6×.

## Software Stack

| Component | Version | Notes |
|-----------|---------|-------|
| CANN toolkit | 9.0.0 | Both validated stacks report `innerversion=V100R001C10SPC001B250` |
| Driver | 25.5.1 | |
| torch / torch_npu | 2.10.0 / 2.10.0 | A2 container image |
| torch / torch_npu | 2.8.0 / 2.8.0.post4 | A3 native install |
| Python | 3.11 | The operator wheels are `cp311`-tagged |
| transformers | 5.12.1 | Pinned by SGLang's `pyproject_npu.toml`; installed in [Step 6](#step-6-install-the-python-dependencies) |
| GCC at `/usr/bin/gcc` | **≥ 11** | Hard requirement — see the note below |
| hwloc, libnuma | any | `libhwloc-dev`, `libnuma-dev`, `pkg-config` — KT-Kernel marks hwloc `REQUIRED` |

> **Important:** The GCC requirement applies to **`/usr/bin/gcc` specifically**, not to whatever `CC` points at. `kt-kernel/CMakeLists.txt` force-selects `/usr/bin/gcc` whenever that path exists — deliberately, so a conda toolchain on `PATH` is never picked up by accident — which means exporting `CC`/`CXX` does not change the compiler, and installing a newer GCC alongside an old one does not help.
>
> Two distinct failures follow from an old `/usr/bin/gcc`:
>
> - **GCC < 11**: `cpu_backend/worker_pool.h` includes the C++20 `<barrier>`, which libstdc++ only ships from GCC 11. The build stops with `fatal error: barrier: No such file or directory`, and there is no workaround short of making `/usr/bin/gcc` a newer compiler (`update-alternatives`, a newer base image, or a container).
> - **GCC < 10**: it also cannot encode the `+bf16` / `+i8mm` `-march` modifiers that `setup.py` enables from `/proc/cpuinfo`, giving `invalid feature modifier 'bf16'`. `build_kt_kernel.sh` disables those extensions, which is the configuration the CPU MoE path was validated on anyway.
>
> `build_kt_kernel.sh` checks the effective compiler up front and stops with this explanation rather than letting the build fail deep in the compile.

The quickest starting point on an A2 host is the Ascend SGLang container image, which already ships CANN 9.0.0, Python 3.11 and the torch stack:

```bash
docker run -d --name dsv4 \
  --ipc=host --privileged --security-opt seccomp=unconfined \
  --ulimit memlock=-1 --ulimit stack=67108864 --shm-size 16g \
  $(for d in /dev/davinci[0-9]*; do printf -- '--device %s ' "$d"; done) \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/sbin:/usr/local/sbin:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /var/queue_schedule:/var/queue_schedule:ro \
  -v /path/to/models:/workspace/models \
  -v /path/to/workspace:/workspace/dsv4 \
  -p 18080:18080 \
  quay.io/ascend/sglang:main-cann9.0.0-910b sleep infinity

docker exec -it dsv4 bash
```

Replace `/path/to/models` and `/path/to/workspace` with your own directories. Everything after this point runs identically inside the container or on a bare-metal host that already has CANN 9.0.0.

> **Note:** Use `docker exec -it dsv4 bash`, not `bash -lc`. A login shell re-runs the image's profile scripts, which prepend their own entries to `PYTHONPATH` and shadow your clone.

## What You Need Before You Start

Four things, and the deployment fails at a different place for each:

1. **CANN 9.0.0**, with `<root>/ascend-toolkit/set_env.sh` present.
2. **Two CANN custom operator vendor packages** — `customize` and `custom_transformer` — plus the `custom_ops` torch bindings. Without them the model fails to load with missing-operator errors. Many vendor images already ship them; [Step 3](#step-3-build-the-cann-custom-operator-packages) tells you how to check and how to build them if not.
3. **KT-Kernel built with the Ascend backend** ([Step 4](#step-4-build-kt-kernel)), and **sgl-kernel-npu** ([Step 5](#step-5-build-sgl-kernel-npu)). Container images ship the latter; native installs must build it.
4. **Three weight artifacts** ([Step 7](#step-7-prepare-the-weights)). The MXFP4 GGUF set is the long pole: about 138 GiB, derived from the official checkpoint.

## Step 1: Get the Code

Two repositories. SGLang is cloned next to KTransformers by default; a checkout under `third_party/sglang` is detected too.

```bash
export DSV4_WORKSPACE=$HOME/dsv4-workspace     # same value you will put in ~/dsv4.env
mkdir -p "${DSV4_WORKSPACE}" && cd "${DSV4_WORKSPACE}"

git clone https://github.com/kvcache-ai/ktransformers.git
git clone -b dsv4-cann9-no-patch https://github.com/Pan-Boyi/sglang.git

cd ktransformers
git submodule update --init --progress third_party/llama.cpp third_party/pybind11
```

> **Note:** If you clone KTransformers under a different directory name, use that name
> for `KTRANSFORMERS_REPO` in [Step 2](#step-2-configure-your-environment) — the tutorial
> assumes `${DSV4_WORKSPACE}/ktransformers`.

> **Note:** Pass `--progress`. Git 2.25 (the Ubuntu 20.04 default) prints one `Cloning into ...` line and then stays silent for the whole fetch, which looks like a hang. `third_party/llama.cpp` is about 419 MB of history; `--depth 1` cuts that to roughly 30 MB if you do not need `git describe` to work inside the submodule.

> **Important:** Both submodules are required. `kt-kernel/CMakeLists.txt` calls `add_subdirectory()` on **`third_party/pybind11`** as well as on `third_party/llama.cpp`, so initializing only `llama.cpp` fails at configure time. The other submodules in `.gitmodules` are not part of this build and can stay uninitialized.

`third_party/llama.cpp` is pinned to upstream tag `b3173` (`a94e6ff8774b7c9f950d9545baf0ce35e8d1ed2f`), a commit that exists in the public llama.cpp repository, so a plain recursive clone works for everyone. That tag predates MXFP4, so the delta ships with the Ascend backend as a patch series under `kt-kernel/third_party_patches/llama.cpp/` and is applied automatically at CMake configure time. **Never run `git apply` by hand** — the configure step detects an already-patched tree and skips it, and a manual apply makes that detection report "already applied" when you expected "applied".

## Step 2: Configure Your Environment

Everything downstream reads [`dsv4_env.sh`](../../kt-kernel/tools/ascend_dsv4/dsv4_env.sh). It auto-detects the CANN root, the CANN version, the NUMA node count, the SoC and the Python interpreter; you only export what it cannot know.

Put your settings in one file and source it in every shell you use from here on.
**This is the only place you edit paths** — everything else in this tutorial is
copy-paste as written.

```bash
cat > ~/dsv4.env <<'EOF'
# ---- edit these ----
export DSV4_WORKSPACE=$HOME/dsv4-workspace   # clones and build artifacts
export DSV4_MODEL_ROOT=/path/to/models       # parent of the three weight directories
export DSV4_NPU_DEVICE_ID=0                  # an idle die
export DSV4_PORT=18080

# ---- the directory name you cloned ktransformers into ----
export KTRANSFORMERS_REPO=$DSV4_WORKSPACE/ktransformers
export DSV4_TOOLS=$KTRANSFORMERS_REPO/kt-kernel/tools/ascend_dsv4

# ---- leave alone ----
export DSV4_ARTIFACT_DIR=$DSV4_WORKSPACE/dsv4-artifacts
export DSV4_LOG_DIR=$DSV4_WORKSPACE/dsv4-logs
EOF

source ~/dsv4.env
mkdir -p "$DSV4_WORKSPACE" "$DSV4_LOG_DIR" "$DSV4_ARTIFACT_DIR"
bash "$DSV4_TOOLS/dsv4_env.sh" --show
```

> **Important:** Do **not** export `SGLANG_REPO` in that file. `dsv4_env.sh` finds it
> itself — `${KTRANSFORMERS_REPO}/third_party/sglang` if that exists, otherwise
> `${DSV4_WORKSPACE}/sglang`. Setting it by hand overrides the detection, and if your
> clone is in the other location every later step fails with
> `not found: .../python/pyproject_npu.toml`. Only set it if your SGLang is in a third
> place entirely.

Verify the two repository paths resolved to directories that actually exist before
going on:

```bash
source ~/dsv4.env
bash "$DSV4_TOOLS/dsv4_env.sh" --show | grep -E 'KTRANSFORMERS_REPO|SGLANG_REPO'
ls "$(bash "$DSV4_TOOLS/dsv4_env.sh" --show 2>/dev/null | awk '/SGLANG_REPO/{print $2}')/python/pyproject_npu.toml"
```

### Platform profiles

`dsv4_env.sh` derives the hardware-dependent values, but it is worth knowing what it
should come up with on each platform, because a wrong NUMA count or SoC is not obvious
later:

| | **A2** (Atlas 800I A2 / 910B3) | **A3** (910_93 series) |
|---|---|---|
| Typical shape | container from the Ascend SGLang image | native install, often non-root |
| `ASCEND_INSTALL_ROOT` | `/usr/local/Ascend` | often `$HOME/Ascend` |
| `DSV4_SOC` | `ascend910b` | `ascend910_93` |
| torch / torch_npu | 2.10 / 2.10 | 2.8.0 / 2.8.0.post4 |
| `/usr/bin/gcc` | 11.4 (Ubuntu 22.04) | often 9.4 (Ubuntu 20.04) — see [Step 4](#step-4-build-kt-kernel) |
| NUMA nodes → `DSV4_THREADPOOL_COUNT` | 8 | 1 |
| → `DSV4_CPUINFER` | 128 | 16 |
| HBM per die | 64 GB | 61 GB |
| `sgl_kernel_npu` | shipped by the image | must be built ([Step 5](#step-5-build-sgl-kernel-npu)) |
| `npu-smi` | works | may fail with `libc_sec.so`; use the torch probe below |

If `npu-smi info` does not run, pick the idle die through the runtime instead:

```bash
source ~/dsv4.env
source "$ASCEND_INSTALL_ROOT/ascend-toolkit/set_env.sh"
python3 -c "
import torch, torch_npu
for i in range(torch.npu.device_count()):
    free, total = torch.npu.mem_get_info(i)
    print(f'[{i}] {torch.npu.get_device_name(i)}  free={free/2**30:.1f}G / total={total/2**30:.1f}G')"
```

Only a die whose `free` is close to `total` is idle.

`--show` prints everything it resolved. Check it before going further:

```
CANN
  ASCEND_INSTALL_ROOT     /usr/local/Ascend
  CANN_ROOT               /usr/local/Ascend/cann-9.0.0
  CANN_VERSION            9.0.0
  vendors present         customize custom_transformer
...
Hardware / serving
  DSV4_SOC                ascend910b
  DSV4_THREADPOOL_COUNT   8   (NUMA nodes)
  DSV4_CPUINFER           128
```

The full variable list, with what each one defaults to:

| Variable | Default | Description |
|----------|---------|-------------|
| `ASCEND_INSTALL_ROOT` | first of `$HOME/Ascend`, `/usr/local/Ascend`, `/opt/Ascend` that has `ascend-toolkit/set_env.sh` | CANN install prefix. A user-scoped install under `$HOME` is common on shared hosts. |
| `CANN_ROOT` | resolved from `ascend-toolkit/latest` | The versioned directory, e.g. `.../cann-9.0.0`. |
| `CANN_VENDORS_DIR` | `${CANN_ROOT}/opp/vendors` | Where the two operator vendor packages install. |
| `KTRANSFORMERS_REPO` | the repository this script lives in | |
| `SGLANG_REPO` | `${KTRANSFORMERS_REPO}/third_party/sglang` if present, else `${DSV4_WORKSPACE}/sglang` | |
| `DSV4_MODEL_ROOT` | `${DSV4_WORKSPACE}/models` | Parent of the three weight artifacts. |
| `DSV4_MODEL_PATH` | `${DSV4_MODEL_ROOT}/DeepSeek-V4-Flash-W8A8` | W8A8 checkpoint, served on the NPU. |
| `DSV4_NATIVE_CKPT` | `${DSV4_MODEL_ROOT}/DeepSeek-V4-Flash` | Official checkpoint. Only read when converting the GGUF set. |
| `DSV4_GGUF_DIR` | `${DSV4_MODEL_ROOT}/cache` | The 43 per-layer MXFP4 GGUF files. |
| `DSV4_GGUF_TEMPLATE` | `${DSV4_GGUF_DIR}/dsv4_layer{layer_idx}_mxfp4.gguf` | Passed to `--kt-weight-path`. |
| `DSV4_SOC` | detected from `npu-smi`, else from `torch.npu.get_device_name` | `ascend910b` or `ascend910_93`. |
| `DSV4_NPU_DEVICE_ID` | `0` | Becomes `ASCEND_RT_VISIBLE_DEVICES`. |
| `DSV4_THREADPOOL_COUNT` | NUMA node count | One CPU-MoE sub-pool per NUMA node. |
| `DSV4_CPUINFER` | `16 ×` NUMA nodes, capped at 3/4 of `nproc` | Total CPU MoE threads. |
| `DSV4_NUM_GPU_EXPERTS` | `32` | Experts kept resident on the NPU per layer, ~1.0 GiB HBM each. |
| `DSV4_MEM_FRACTION` | `0.81` | Static HBM budget. Sized for 64 GB; re-derive on other HBM sizes. |
| `DSV4_CONTEXT_LENGTH` | `65536` | |
| `DSV4_CHUNKED_PREFILL_SIZE` | `32768` | Must be a positive multiple of `--page-size 128`. |
| `DSV4_ARTIFACT_DIR` | `${DSV4_WORKSPACE}/dsv4-artifacts` | Built wheels and `.run` packages land here. |
| `DSV4_LOG_DIR` | `${DSV4_WORKSPACE}/dsv4-logs` | |
| `DSV4_PYTHON` | `python3.11`, else `python3` | |

> **Important:** `dsv4_env.sh` exports `PYTHONPATH` **after** sourcing every CANN environment script, and that ordering is not cosmetic. Those scripts *prepend* their own entries, so a `PYTHONPATH` exported before them ends up behind theirs. The symptom is the worst kind: the server starts, answers requests normally, and runs a completely different SGLang than your clone. `preflight.sh` and `serve.sh` both assert on `sglang.__file__` for exactly this reason.

> **Note:** `dsv4_env.sh` sets `PYTHONNOUSERSITE=1` only when the interpreter's own `site-packages` is writable. If you are not root, pip installs into `~/.local` by design, and disabling user-site there would hide every dependency you are about to install.

## Step 3: Build the CANN Custom Operator Packages

First check whether you need this at all — vendor images frequently ship them:

```bash
source kt-kernel/tools/ascend_dsv4/dsv4_env.sh
ls "${CANN_VENDORS_DIR}"                        # want: customize  custom_transformer
python3 -c "import torch, torch_npu, custom_ops; print('ok')"
```

If both vendors are present and `custom_ops` imports, skip to [Step 4](#step-4-build-kt-kernel).

> **Note:** `import torch` has to come first. `custom_ops_lib*.so` lists `libc10.so`,
> `libtorch_cpu.so` and `libtorch_npu.so` as `NEEDED` but carries no `RPATH` to torch's
> `lib/` directory, which is not on the loader's search path. Importing torch first pulls
> those libraries into the process so the later `dlopen` resolves against them; importing
> `custom_ops` on its own fails with `ImportError: libc10.so: cannot open shared object
> file`. Adding torch's `lib/` to `LD_LIBRARY_PATH` is not a fix — it resolves `libc10`
> and then fails on the next dependency, and once all of them resolve at `dlopen` time the
> bundled libgomp hits the aarch64 static-TLS limit instead.

Otherwise:

```bash
bash kt-kernel/tools/ascend_dsv4/build_cann_ops.sh all
```

This clones two Gitcode repositories, pins them to the commits this tutorial was validated against, builds three packages and installs them into `${CANN_ROOT}/opp`. Budget 40–90 minutes.

| Package | Source | Pinned commit | Provides |
|---------|--------|---------------|----------|
| `customize` vendor | `gitcode.com/cann/cann-recipes-infer` | `1c8e6bcc2333d95b3db47d873210f921113d6d11` | Fused ops: `RmsNormDynamicQuant`, `SwigluClipQuant`, `MoeGatingTopKHash`, … |
| `custom_ops` wheel | same repository | same | The `torch.ops.custom.*` python bindings |
| `custom_transformer` vendor | `gitcode.com/cann/ops-transformer` | `8edcd591e83e536e9ee98a9ce0de3af02ea4f3ea` | NSA/DSA attention: `compressor`, `sparse_attn_sharedkv`, `quant_lightning_indexer` |

> **Important:** Both repositories track a moving `master` with no tags or releases, so the commits above are pinned in `dsv4_env.sh` as `CANN_RECIPES_COMMIT` and `OPS_TRANSFORMER_COMMIT`. Pulling `master` a week later gets you different operators. The NSA operators in particular exist **only** on `master` — the 9.0.0 release branch dropped them.

Two quirks worth knowing, both already handled by the script:

- Every build step runs under `umask 0022` and `chmod -R go-w .`. CANN's `msopgen` refuses to process group- or world-writable intermediate files and aborts the entire build with a security message.
- `ops-transformer` appends `_transformer` to the vendor name itself, so the script passes `--vendor_name=custom` and the result is the vendor `custom_transformer`.

Re-source the environment afterwards, so `ASCEND_CUSTOM_OPP_PATH` picks up the newly installed vendors:

```bash
source kt-kernel/tools/ascend_dsv4/dsv4_env.sh
```

## Step 4: Build KT-Kernel

```bash
# Debian/Ubuntu; on openEuler/CentOS use hwloc-devel numactl-devel pkgconfig
sudo apt-get install -y build-essential cmake git libhwloc-dev libhwloc15 libnuma-dev patchelf pkg-config

bash kt-kernel/tools/ascend_dsv4/build_kt_kernel.sh
```

10–30 minutes on a first build. The configure log must contain all four of these lines:

```
-- ARM target: -march=armv8.2-a+fp16+dotprod
-- llama.cpp: applied patch .../0001-ggml-mxfp4-type.patch
-- Ascend NPU (CANN) backend selected
-- NUMA library found: ... - enabling NUMA support
```

The second line is the one to read carefully. `applied patch` means the MXFP4 delta went in on this configure; `patch already applied, skipping` is also fine on a rebuild. Anything else means `third_party/llama.cpp` is not at the pinned commit.

Install the resulting wheel:

```bash
python3 -m pip install --no-deps "${DSV4_ARTIFACT_DIR}"/wheels/kt_kernel-*.whl
```

> **Important:** `--no-deps` is not optional. `kt-kernel`'s dependency metadata includes `torch`, and letting pip resolve it on a CANN image replaces the NPU torch build with a generic one. `torch_npu` is then bound to a torch that no longer exists and the whole NPU stack has to be rebuilt.

## Step 5: Build sgl-kernel-npu

Check first — the Ascend SGLang container images already ship it:

```bash
python3 -c "import sgl_kernel_npu; print('ok')"
```

If that works, skip to [Step 6](#step-6-install-the-python-dependencies). On a native CANN install it almost certainly does not, and it is **not** an optional dependency: `sglang/srt/mem_cache/pool_host/mha.py` imports it unconditionally, so the server dies at startup with `ModuleNotFoundError: No module named 'sgl_kernel_npu'`.

```bash
bash kt-kernel/tools/ascend_dsv4/build_sgl_kernel_npu.sh
```

This clones `sgl-project/sgl-kernel-npu` at tag `2026.6.2`, builds it and installs the four wheels it produces (`sgl_kernel_npu`, `deep_ep`, `attentions`, `torch_memory_saver`). Budget 20–40 minutes.

The script works around three issues in that tag, all of which otherwise present as a build that "succeeded" with an empty `output/`:

- `csrc/attentions/csrc/CMakeLists.txt` does not link `libdl`, so `PTAExtensionOPS` fails with `undefined reference to dlopen`. `build.sh` runs under `set -e`, so the only visible symptom is a missing wheel.
- The `deep_ep` vendor tree is installed read-only, so a second build stops at `rm uninstall.sh: Permission denied`.
- `csrc/attentions/build/` is a **tracked source directory**, not build output. Deleting it to "clean" the tree breaks the build; restore it with `git checkout -- csrc/attentions/build/`. Only `csrc/build_out/` and `output/` are generated.

## Step 6: Install the Python Dependencies

```bash
bash kt-kernel/tools/ascend_dsv4/install_python_deps.sh
```

This reads the dependency list out of SGLang's `python/pyproject_npu.toml`, writes a pip constraints file pinning the torch family that is already installed, and installs the list against those constraints. Anything that wants a different torch fails here, loudly, instead of silently upgrading it later. Pass `--dry-run` to see the list first.

> **Important:** Do **not** run `pip install -e python/` from the SGLang clone. The default `python/pyproject.toml` is the **CUDA** variant: it pulls `torch`, `flashinfer` and `cuda-python`, replaces the image's torch with a CUDA build, and leaves `torch_npu` bound to a torch that is gone. On a container the only recovery is to recreate it. Running from a clone is supported through `PYTHONPATH` only, which is what `dsv4_env.sh` sets up.

## Step 7: Prepare the Weights

Three artifacts. The first two are downloads; the third is produced from the second.

| Artifact | Variable | Size | Used for |
|----------|----------|------|----------|
| W8A8 `compressed-tensors` checkpoint | `DSV4_MODEL_PATH` | ~275 GB | `--model-path`: attention, dense layers, resident experts |
| Official DeepSeek-V4-Flash checkpoint | `DSV4_NATIVE_CKPT` | ~150 GB | Source for the GGUF conversion; **not** read at serving time |
| 43 per-layer MXFP4 GGUF files | `DSV4_GGUF_DIR` | ~138 GiB | `--kt-weight-path`: the CPU-offloaded experts |

Download the official checkpoint:

```bash
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir "${DSV4_MODEL_ROOT}/DeepSeek-V4-Flash"
```

For the W8A8 side, use a **published** W8A8 quantization of the same model rather than quantizing it yourself. `preflight.sh` checks that its `config.json` reports `quant_method: compressed-tensors` with `format: int-quantized`, int8 channel-symmetric weights and int8 token-dynamic activations — that is what `--quantization compressed-tensors` expects.

> **Important:** The CPU-side GGUF and the NPU-side W8A8 must come from the **same quantization basis**. Using a published W8A8 alongside the published MXFP4 experts keeps them consistent. If you re-quantize W8A8 yourself with a tool that applies its own rotation, the two halves disagree, and the failure mode is garbled output with no error anywhere.

Then build the GGUF set:

```bash
bash kt-kernel/tools/ascend_dsv4/convert_mxfp4_gguf.sh
```

The conversion is a **lossless bit repack**, not a re-quantization: the checkpoint already stores E2M1 codes with a ue8m0 per-32 scale, and only the nibble ordering within each 32-element block differs from GGUF's half-block interleave. It is also byte-deterministic, so the same checkpoint always yields the same files. Budget several hours and ~138 GiB of free space.

The script finishes by running a three-level check ([`verify_mxfp4_gguf_set.py`](../../kt-kernel/tools/mxfp4_gguf/verify_mxfp4_gguf_set.py)):

- **L1** every layer present, every file the exact expected byte count.
- **L2** sha256 against a manifest, if you pass `--sha256-manifest`.
- **L3** dequantize a sample of layers from both the GGUF and the checkpoint and compare element-wise — bit-exact equality is required, since the repack is lossless.

> **Important:** L1 is not optional busywork. An interrupted or doubly-scheduled conversion leaves a **truncated** file behind, and nothing downstream notices: the server loads it, serves, and produces wrong tokens.

Generate your own manifest to check the set on a second machine:

```bash
cd "${DSV4_GGUF_DIR}" && sha256sum dsv4_layer*_mxfp4.gguf | sort -V -k2 > manifest.txt
```

## Step 8: Preflight

```bash
bash kt-kernel/tools/ascend_dsv4/preflight.sh
```

Every check corresponds to a failure that is silent or badly reported at serve time: the CANN vendors and `ASCEND_CUSTOM_OPP_PATH`, the torch/torch_npu/transformers versions, `torch.ops.custom.*` resolution, the Ascend callback-worker binding in `kt_kernel_ext`, `GGMLQuantizationType.MXFP4 == 39`, `sglang.__file__` pointing at your clone, the GGUF file count, the `{layer_idx}` placeholder, and the `--chunked-prefill-size` divisibility rule.

`PREFLIGHT OK` means it is safe to launch.

> **Note:** `MXFP4 == 39` is the check that proves the llama.cpp patch series actually took effect. Without it the loader raises on the first expert tensor.

## Step 9: Launch the Server

```bash
source ~/dsv4.env
bash "$DSV4_TOOLS/serve.sh"
tail -f "${DSV4_LOG_DIR}/serve.log"
```

Weight loading takes 8–10 minutes, and for that whole window **the process exists but the
port is not listening yet** — that is normal, not a hang. Wait for readiness before
sending anything:

```bash
source ~/dsv4.env
until curl -sf -m5 --noproxy '*' "http://127.0.0.1:${DSV4_PORT}/health" >/dev/null; do
  echo "$(date +%T) still loading..."; sleep 30
done
echo "ready"
```

> **Note:** Run the server under `tmux`/`screen`, or make sure it is the backgrounded
> `serve.sh` form. Accuracy and throughput runs take hours; if the server dies partway
> the client reports a connection error and the whole run is wasted.

The script prints the resolved `sglang.__file__` before it launches anything and refuses to start if it points outside your clone. It then sets the runtime environment and starts the server:

```bash
export ASCEND_RT_VISIBLE_DEVICES=${DSV4_NPU_DEVICE_ID}
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export SGLANG_SET_CPU_AFFINITY=1
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
ulimit -n 65536

python3 -m sglang.launch_server \
    --model-path "${DSV4_MODEL_PATH}" \
    --device npu \
    --attention-backend ascend \
    --tensor-parallel-size 1 \
    --expert-parallel-size 1 \
    --moe-a2a-backend none \
    --page-size 128 \
    --quantization compressed-tensors \
    --disable-shared-experts-fusion \
    --dtype bfloat16 \
    --trust-remote-code \
    --disable-radix-cache \
    --mem-fraction-static "${DSV4_MEM_FRACTION}" \
    --context-length "${DSV4_CONTEXT_LENGTH}" \
    --chunked-prefill-size "${DSV4_CHUNKED_PREFILL_SIZE}" \
    --watchdog-timeout 18000 \
    --kt-method LLAMAFILE \
    --kt-num-gpu-experts "${DSV4_NUM_GPU_EXPERTS}" \
    --kt-weight-path "${DSV4_GGUF_TEMPLATE}" \
    --kt-threadpool-count "${DSV4_THREADPOOL_COUNT}" \
    --kt-cpuinfer "${DSV4_CPUINFER}" \
    --max-running-requests 1 \
    --host "${DSV4_HOST}" --port "${DSV4_PORT}"
```

> **Important:** `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` is load-bearing. Without it HBM fragments across prefill chunks, the reported weight footprint grows by about 1.8 GB, the KV pool is sized down to match, and long prompts fail with what looks like a genuine OOM. Verify it landed by reading `/proc/<pid>/environ`, not by trusting what the script echoed.

Why each parameter is what it is:

| Parameter | Why |
|-----------|-----|
| `--tensor-parallel-size 1`, `--expert-parallel-size 1` | Only single-die KT offload is supported; anything else is rejected at startup. |
| `--moe-a2a-backend none` | The Ascend dispatcher hook KT attaches to does not support A2A dispatchers. |
| `--quantization compressed-tensors` | Matches the on-disk W8A8 (`int-quantized`) NPU weights. |
| `--disable-shared-experts-fusion` | Shared-expert fusion is incompatible with a split GPU/CPU expert set. |
| `--page-size 128` | `--chunked-prefill-size` must be a positive multiple of it. |
| `--chunked-prefill-size 32768` | Must be at least as large as the longest prompt you will serve. Do **not** pass `-1`: the `LLAMAFILE` CPU MoE sizes its per-NUMA fp32 output buffer from the maximum chunk length, `-1` collapses to 1, and the first prefill longer than one token writes past the allocation and aborts inside glibc. |
| `--kt-num-gpu-experts 32` | Experts resident on the NPU per layer, about 1.0 GiB HBM each, taken from the same static budget as the KV pool. |
| `--kt-threadpool-count <numa>` | One sub-pool per NUMA node. Exceeding the host's node count fails at startup with `NUMA node N not found`. |
| `--kt-cpuinfer <numa×16>` | Total CPU MoE threads. Do not allocate every core — the spin threads, the ACL host callback and the OS need headroom. |
| `--max-running-requests 1` | This configuration is validated for single-request serving only. |
| `--disable-radix-cache` | Prefix caching is not validated with a split expert set. |
| *(no `--cuda-graph-backend-*` flag)* | NPU graph capture is on by default and is the recommended setting. Disabling it costs roughly 5× decode throughput. |

Stop the server with `SIGINT` first, then `SIGTERM`:

```bash
kill -INT  "$(cat "${DSV4_LOG_DIR}/serve.log.pid")"
sleep 5
kill -TERM "$(cat "${DSV4_LOG_DIR}/serve.log.pid")"
```

> **Note:** Do not `pkill -f sglang.launch_server`. If the container's PID 1 is `sleep infinity` it does not reap children, so `kill -0` succeeds on a zombie and the process looks alive. Check for `Z <defunct>` in `ps` before concluding anything.

## Step 10: Acceptance Checks

```bash
bash kt-kernel/tools/ascend_dsv4/verify.sh
```

Four gates, all of which must pass:

1. **HBM accounting.** On the validated A2 configuration: 60.49 GB free before load, 18.11 GB after, so 42.37 GB of weights, and KV pools `full=915584` / `swa=91520`. A weight figure noticeably above that means HBM fragmented — check the four environment variables in Step 8.
2. **NPU graph captured.** The log must contain `Capture target decode NPU graph end`. Capture costs about 6 s of startup and 0.26 GB of HBM, against roughly 5× decode throughput.
3. **Health.** `/health` and `/health_generate` both return 200.
4. **Numerical probe.** A greedy `/generate` must return a non-empty completion, and the same completion every time.

> **Important:** HTTP 200 is not an acceptance signal. With a broken CPU expert path the server still binds its port and answers health probes while generating nothing — `"text": ""` with status 200 is a failure. Greedy decoding here is deterministic, so re-run the probe and compare byte for byte, and compare against a second machine before trusting any measurement.

## Step 11: Talk to the Model

`verify.sh` proves the plumbing works. This step is the part you do by hand, to see the
model actually behaving like DeepSeek-V4 rather than emitting plausible noise.

**A real question, through the OpenAI-compatible endpoint:**

```bash
source ~/dsv4.env
curl -s --noproxy '*' -X POST "http://127.0.0.1:${DSV4_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4","messages":[{"role":"user","content":"Explain in three sentences what a Mixture-of-Experts model is and why it saves memory."}],"temperature":0.6,"max_tokens":300}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content']); print('---', d['usage'])"
```

**Reasoning spot-checks.** `temperature=0` makes each answer deterministic and each of
these has exactly one right answer, so a broken numerical path shows up immediately:

```bash
ask() {
  curl -s --noproxy '*' -X POST "http://127.0.0.1:${DSV4_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"dsv4\",\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"temperature\":0,\"max_tokens\":64}" \
    | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'].strip())"
}

ask "What is 27 * 43? Answer with just the number."                             # 1161
ask "How many letter r are in the word strawberry? Answer with just the number." # 3
ask "Alice is taller than Bob. Bob is taller than Carol. Who is shortest? Name only."  # Carol
```

**Streaming**, so you can watch tokens arrive rather than waiting for one blob:

```bash
curl -sN --noproxy '*' -X POST "http://127.0.0.1:${DSV4_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4","stream":true,"messages":[{"role":"user","content":"Write one sentence about Ascend NPUs."}],"temperature":0.6,"max_tokens":60}' \
  | sed -u 's/^data: //' | grep -v '^\[DONE\]' \
  | python3 -c "
import sys, json
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    try: d = json.loads(line)
    except Exception: continue
    c = d['choices'][0].get('delta', {}).get('content')
    if c: print(c, end='', flush=True)
print()"
```

**An interactive session** with multi-turn context:

```bash
python3 "${DSV4_TOOLS}/dsv4_chat.py" "${DSV4_PORT}"
```

`/reset` clears the context, `/quit` exits. To confirm the context is really being kept,
tell it your name and then ask for it in the next turn.

> **Note:** The server is launched with `--max-running-requests 1`, so requests are served
> one at a time. A second client does not fail, it queues.

> **Note:** Warm up before measuring anything. The dynamic hot-expert residency needs a few
> requests to converge, so the first few are noticeably slower than steady state.

## Step 12: Accuracy Validation (GPQA-Diamond)

```bash
source ~/dsv4.env
python3 -m pip install evalscope

cd "$DSV4_WORKSPACE"
[ -d cann-recipes-infer ] || git clone --depth 1 https://gitcode.com/cann/cann-recipes-infer.git
export DSV4_RECIPES="$DSV4_WORKSPACE/cann-recipes-infer/integration/sglang/dsv4-flash-single-npu-moe-offload"
echo "export DSV4_RECIPES=$DSV4_RECIPES" >> ~/dsv4.env
```

**Wait for the server before starting.** A round takes about 1 h 50 min and evalscope
retries five times and then gives up with `Connection error`, which looks like an
evalscope problem but almost always means the server is not listening yet — weight
loading alone is 8–10 minutes:

```bash
source ~/dsv4.env
until curl -sf -m5 --noproxy '*' "http://127.0.0.1:${DSV4_PORT}/health" >/dev/null; do
  echo "$(date +%T) still loading..."; sleep 30
done
echo "ready"
```

Then run it — in `tmux` or `screen`, because a dropped terminal loses the whole round:

```bash
source ~/dsv4.env
REPEATS=3 \
PORT="$DSV4_PORT" \
MODEL_PATH="$DSV4_MODEL_ROOT/DeepSeek-V4-Flash-W8A8" \
OUT_DIR="$DSV4_LOG_DIR/gpqa" \
bash "$DSV4_RECIPES/scripts/tools/gpqa_accuracy_repeat.sh" 2>&1 | tee "$DSV4_LOG_DIR/gpqa.log"
```

A single round is also available directly:

```bash
source ~/dsv4.env
evalscope eval \
  --model "$DSV4_MODEL_ROOT/DeepSeek-V4-Flash-W8A8" \
  --api-url "http://127.0.0.1:${DSV4_PORT}/v1/chat/completions" \
  --api-key EMPTY --eval-type openai_api \
  --datasets gpqa_diamond \
  --generation-config '{"temperature":1,"top_p":1,"max_tokens":32768,"extra_body":{"chat_template_kwargs":{"thinking":false,"high_effort":false}}}' \
  --eval-batch-size 1 --repeats 1 \
  --work-dir "$DSV4_LOG_DIR/gpqa/R1"
```

> **Important:** Report a multi-round mean, never a single round. GPQA-Diamond is 198
> questions and at `temperature=1` the binomial standard error of one round is about
> ±3.2 pp — wider than the spread between rounds actually measured on A2
> (72.22 / 71.72 / 75.76, mean **73.23%**, sample SD 2.20 pp). Those three are
> statistically indistinguishable from each other. Chasing a "regression" smaller than
> roughly 5 pp means averaging about ten rounds first.

> **Note:** evalscope uses the OpenAI Python client, which honours `http_proxy` /
> `all_proxy`. If you set a proxy to download the dataset, keep localhost out of it:
> `export no_proxy="127.0.0.1,localhost,$no_proxy"`.

## Step 13: Throughput Validation

```bash
source ~/dsv4.env
uptime            # decode is host-memory-bandwidth bound; measure on an idle machine
```

**Baseline — feature switches off:**

```bash
source ~/dsv4.env
TARGET_TOKENS_LIST="130 1000 4000 8000" \
MAX_NEW=1000 REPEAT=3 WARMUP=1 \
PORT="$DSV4_PORT" PY=python3 \
bash "$DSV4_RECIPES/scripts/tools/decode_throughput_test.sh" \
  2>&1 | tee "$DSV4_LOG_DIR/perf_baseline.log"
```

Read the `warm-pf(s)` and `dec-med` columns. `WARMUP=1` is not optional — the dynamic
hot-expert residency needs a few requests to converge.

**All feature switches on.** Restart the server first; streaming needs its own HBM slot,
so `mem-fraction` has to go up with it:

```bash
source ~/dsv4.env
P=$(cat "$DSV4_LOG_DIR/serve.log.pid"); kill -INT $P; sleep 8; kill -TERM $P; sleep 20

export DSV4_PREFILL_STREAM=1      # streaming prefill + depool + dynamic resident + side stream
export DSV4_MEM_FRACTION=0.86     # 0.81 will not start with streaming enabled
bash "$DSV4_TOOLS/serve.sh"

until curl -sf -m5 --noproxy '*' "http://127.0.0.1:${DSV4_PORT}/health" >/dev/null; do sleep 30; done

TARGET_TOKENS_LIST="130 1000 4000 8000" \
MAX_NEW=1000 REPEAT=3 WARMUP=1 \
PORT="$DSV4_PORT" PY=python3 \
bash "$DSV4_RECIPES/scripts/tools/decode_throughput_test.sh" \
  2>&1 | tee "$DSV4_LOG_DIR/perf_allon.log"
```

**Confirm streaming actually engaged before believing the numbers:**

```bash
source ~/dsv4.env
echo "inline resident : $(grep -c 'inline resident' "$DSV4_LOG_DIR/serve.log")"
echo "hybrid fallback : $(grep -cE 'streaming failed|hybrid fallback' "$DSV4_LOG_DIR/serve.log")"
```

> **Important:** Only `inline resident > 0` is positive proof. `maybe_streaming_forward`
> has several early returns that log nothing, so a run that never streamed at all also
> reports zero fallbacks. Two silent non-streaming modes both answer requests normally:
> a prompt shorter than `KT_PREFILL_STREAM_THRESHOLD` (512) never triggers it — so the
> 130-token step cannot be used to evaluate streaming at all — and a layer that OOMs falls
> back to hybrid per layer. One A2 run with 27 experts at `mem-fraction 0.86` logged
> `inline resident=0, hybrid fallback=240`: 43 layers each attempting, OOMing by 1.00 GiB
> and falling back, burning ~211 s of the 213.2 s it reported for an 8000-token prefill,
> with no error anywhere.

Both runs must use the **same** `--mem-fraction-static` if you want the comparison to be
about the switches; on A2, raising it from 0.81 to 0.86 on its own is worth 0.18–0.71%.

## Tuning

### Resident Experts vs. the KV Pool

`--kt-num-gpu-experts` and `--mem-fraction-static` share one HBM budget with the KV cache, and they interact in a way that defeats the obvious adjustment.

> **Important:** Lowering `--kt-num-gpu-experts` on its own does **not** free runtime headroom. SGLang's KV pool sizer immediately claims whatever the experts gave up, and you end up with *less* margin than before. Measured, with streaming prefill enabled:
>
> | Configuration | Weights | KV `full` | Free after load | Streaming |
> |---------------|---------|-----------|-----------------|-----------|
> | 32 experts / `mem-fraction 0.86` | 51.50 GB | 66,688 | 8.00 GB | engaged |
> | 27 experts / `mem-fraction 0.86` | 46.47 GB | 766,976 | **5.63 GB** | OOM per layer, silently fell back to hybrid |
>
> Lower `--mem-fraction-static` together with the expert count, or the reduction buys nothing.

The SWA sub-pool is a fixed **10%** of the full pool, and it is what runs out first:

```bash
grep -ohE 'swa=[0-9]+' "${DSV4_LOG_DIR}/serve.log" | sort -u
```

> **Important:** A prompt longer than the reported `swa` value cannot be scheduled, and the scheduler **spins instead of reporting the condition**: CPU pegged, `SIGINT` and `SIGTERM` ignored, only `kill -9` works. Check `swa` against your longest prompt right after startup.

### Optional: Streaming Prefill

Streaming prefill stages a whole layer's expert set from DDR into HBM and runs the MoE on the NPU. It turns prefill time into a roughly constant cost instead of one that grows with prompt length. It is **off by default**:

```bash
export DSV4_PREFILL_STREAM=1
bash kt-kernel/tools/ascend_dsv4/serve.sh
```

Measured on the validated A2 configuration:

| Prompt tokens | Streaming off | Streaming on | Speed-up |
|--------------:|--------------:|-------------:|---------:|
| 801 | 10.9 s | 18.5 s | 0.6× (slower) |
| 3,944 | 43.1 s | 19.0 s | 2.3× |
| 7,823 | 85.1 s | 19.2 s | 4.4× |
| 15,568 | 169.3 s | 19.9 s | 8.5× |
| 31,540 | 343.9 s | 22.1 s | 15.6× |

The crossover is around **1,500 tokens** — below that the fixed cost never pays back, which is why `KT_PREFILL_STREAM_THRESHOLD` defaults to 512 and shorter prompts stay on the hybrid path.

Pick the configuration from the longest prompt you intend to serve:

| Longest prompt | `--kt-num-gpu-experts` | `--mem-fraction-static` |
|----------------|-----------------------:|------------------------:|
| ≤ 16k | 32 | 0.83 |
| 32k | 27 | 0.785 |

No single configuration covers 1k through 32k. The boundaries are narrow and each fails differently: below about **0.7985** the server does not start at all (the streaming slots count against the static budget); at **0.81** the SWA pool is only 9,088 and the 16k case deadlocks the scheduler; at **0.86** the remaining headroom is 6.42 GB and a 32k prompt OOMs by 1.93 GiB inside the attention `einsum`.

> **Important:** Confirm streaming actually engaged before you believe any streaming measurement:
>
> ```bash
> grep -c 'inline resident' "${DSV4_LOG_DIR}/serve.log"                       # must be > 0
> grep -cE 'streaming failed|hybrid fallback' "${DSV4_LOG_DIR}/serve.log"     # must be 0
> ```
>
> Both ways of not-streaming are silent and the server answers normally: the prompt was shorter than the threshold, or every layer OOMed and fell back to hybrid. `verify.sh` checks both when `DSV4_PREFILL_STREAM=1`.

> **Note:** Streaming and non-streaming are **not** numerically equivalent. With the same prompt and the same greedy parameters the two paths diverge at the first generated token, because streaming puts all 256 experts on the NPU while hybrid splits them 32/224 across NPU and CPU. Different arithmetic, different rounding, and near-ties flip. Validate accuracy separately on the streaming path.

## Measured Results

All figures from a single 910B3 die, NPU graph on, one request at a time, on an otherwise idle host.

**Throughput.** Decode is the median steady-state inter-token rate after a warmup run; prefill is the mean warmed prefill time.

| Prompt tokens | Decode | Prefill |
|--------------:|-------:|--------:|
| 118 | 16.84 tok/s | 2.0 s |
| 801 | 16.86 tok/s | 10.9 s |
| 3,944 | 16.93 tok/s | 43.1 s |
| 7,823 | 16.76 tok/s | 85.1 s |
| 15,568 | 16.72 tok/s | 169.3 s |
| 31,540 | 16.57 tok/s | 343.9 s |

Decode is flat across context length: it is dominated by the per-token host-side expert reads, not by KV growth. Prefill grows linearly, roughly `2.3 s + 0.0107 s × prompt`, because every chunk pays the full CPU MoE cost.

**Accuracy.** GPQA-Diamond (198 questions) via evalscope, thinking disabled, `temperature=1`, `top_p=1`, `max_tokens=32768`, one question at a time:

```bash
evalscope eval \
  --model "${DSV4_MODEL_PATH}" \
  --api-url "http://127.0.0.1:${DSV4_PORT}/v1/chat/completions" \
  --api-key EMPTY --eval-type openai_api \
  --datasets gpqa_diamond \
  --generation-config '{"temperature":1,"top_p":1,"max_tokens":32768,"extra_body":{"chat_template_kwargs":{"thinking":false,"high_effort":false}}}' \
  --eval-batch-size 1 --repeats 1 \
  --work-dir "${DSV4_LOG_DIR}/gpqa/R1"
```

| Round | Score |
|-------|-------|
| 1 | 72.22% |
| 2 | 71.72% |
| 3 | 75.76% |
| **Mean** | **73.23%** (sample SD 2.20 pp) |

About 1 h 50 min per round.

> **Important:** Report the multi-round mean, never a single round. At 198 questions and `temperature=1` the binomial standard error of one round is roughly ±3.2 pp — wider than the spread between these three, which are therefore statistically indistinguishable.

> **Important:** Decode here is **memory-bandwidth bound on the host**. On a shared machine, contention from other tenants has been observed to drop decode from 16.8 to 1.5 tok/s. A single-threaded `memcpy` benchmark does not detect it — it reported 4.4 GB/s while real throughput was 1.5 tok/s. Measure by sending a 200-token request, benchmark on an idle host, and state the host load alongside any number you publish.

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| Server runs, but your code changes have no effect | `PYTHONPATH` was shadowed by a CANN `set_env.sh`. Export it after every environment script and check `python -c "import sglang; print(sglang.__file__)"`. |
| `ModuleNotFoundError: No module named 'sgl_kernel_npu'` | Not an optional dependency. Build it — see [Step 5](#step-5-build-sgl-kernel-npu). |
| `import custom_ops` fails with `libc10.so: cannot open shared object file` | Import torch first: `python3 -c "import torch, torch_npu, custom_ops"`. `custom_ops_lib.so` needs torch's libraries but has no `RPATH` to them. Do not reach for `LD_LIBRARY_PATH` — adding torch's `lib/` just moves the error to `libtorch_npu.so`, and adding both then hits the aarch64 static-TLS limit on the bundled libgomp. |
| `OSError: libgomp.so.1: cannot allocate memory in static TLS block` | aarch64 static-TLS exhaustion when libgomp is dlopened late. `dsv4_env.sh` preloads it; if you set up the environment by hand, `export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1`. |
| `OSError: … _torchaudio.abi3.so: undefined symbol: torch_library_impl` | `torchaudio` (or `torchvision`) is built against a different torch. `transformers` imports `torchaudio` unconditionally via `loss_rnnt`, so this breaks everything. Install the matching release: `pip install --no-deps torchaudio==$(python -c 'import torch;print(torch.__version__.split("+")[0])')`. |
| CMake configure fails | `third_party/pybind11` not initialized, or `libhwloc-dev` missing. |
| `invalid feature modifier 'bf16' in '-march=…'` | `/usr/bin/gcc` is older than 10. `build_kt_kernel.sh` disables `+bf16`/`+i8mm`/`+sve`; note that exporting `CC` does not help because CMakeLists force-selects `/usr/bin/gcc`. |
| `llamafile not supported` at the first decode step | Built with `CPUINFER_ARM_SVE=ON`. The SVE branch of the sgemm has no MXFP4 tile. Rebuild with SVE off. |
| Startup: `ValueError: … Raise --mem-fraction-static above 0.853` | Streaming prefill is on but `--mem-fraction-static` is still 0.81. |
| Long prompt hangs, Ctrl-C does nothing | The SWA pool is smaller than the prompt. `kill -9`, then lower `--kt-num-gpu-experts` and re-check `swa=`. |
| Prefill aborts inside glibc | `--chunked-prefill-size` is `-1` or not a multiple of `--page-size`. |
| Weights use ~1.8 GB more HBM than expected | `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` is missing; HBM fragmented. |
| ACL error `107011` at startup | Two ACL report subscribers on one stream. torch_npu 2.10 pre-subscribes the capture stream; use the companion kt-kernel build, which honours `KT_EXTERNAL_NPU_REPORT_SUBSCRIBER=1`. |
| `NUMA node N not found` | `--kt-threadpool-count` exceeds the host's NUMA node count. |
| Decode around 3 tok/s | NPU graph mode is disabled. |
| Decode around 1.5 tok/s | DRAM bandwidth contention from another tenant — not a core-count problem. |
| Server exits during warmup with `res=<Response [502]>` | An HTTP proxy is set and intercepts the warmup request to the server's own port. |
| `ModuleNotFoundError` for an SGLang dependency, as a non-root user | `PYTHONNOUSERSITE` is hiding `~/.local`. Current `dsv4_env.sh` only sets it when system `site-packages` is writable. |
| `git submodule update` appears to hang | Git 2.25 does not forward sub-clone progress. Add `--progress`, and consider `--depth 1`. |
| `not found: .../sglang/python/pyproject_npu.toml` | `SGLANG_REPO` was exported by hand and overrode the auto-detection. Unset it, or point it at the clone that exists — see [Step 2](#step-2-configure-your-environment). |
| evalscope or any client reports `Connection error` and retries 5× | The server is not listening — either still loading weights (8–10 min) or no longer running. `curl /health` first; see the readiness loop in [Step 9](#step-9-launch-the-server). |
| The process is alive but the port is closed | Still loading. `grep 'Load weight end' ` the log; `Uvicorn running` is the line that means the port is open. |

## Known Limitations

Stated as measured, not as expected behaviour:

- **Single die, single request.** Tensor parallelism, expert parallelism, A2A dispatch, PD disaggregation and speculative decoding are rejected at startup with KT offload active. Validated at `--max-running-requests 1` only.
- **Measured up to a 31,540-token prompt**, not to the full 65,536-token context. The server comes up and serves at `--context-length 65536`; prompts between 31,540 and 65,536 tokens are untested.
- **No single `--mem-fraction-static` covers 1k–32k** with streaming prefill enabled. See [Tuning](#tuning).
- **Decode throughput is not reproducible across hosts** — it tracks available DRAM bandwidth.
- **The published throughput and accuracy figures are A2 (910B3) only.** The A3 walkthrough reached the same weight footprint (42.38 GB against the A2's 42.37 GB) and passed all four acceptance gates, but its host has one NUMA node and 40 cores against the A2 host's eight nodes and 192, so no performance number transfers between them.
- **The SGLang side is not upstreamed yet.** It is served from the `dsv4-cann9-no-patch` branch of the fork referenced in [Step 1](#step-1-get-the-code).

## Additional Resources

- [DeepSeek-V4-Flash on CUDA](./DeepSeek-V4-Flash.md) — the x86 + NVIDIA variant of this deployment
- [KT-Kernel Parameters](https://github.com/kvcache-ai/ktransformers/tree/main/kt-kernel#kt-kernel-parameters)
- `kt-kernel/third_party_patches/llama.cpp/README.md` — what is patched into `b3173` and why
- [Deployment scripts](../../kt-kernel/tools/ascend_dsv4) and the [MXFP4 GGUF tools](../../kt-kernel/tools/mxfp4_gguf)
- `cann-recipes-infer` — the accuracy and throughput harnesses used in Steps 12 and 13
- [DeepSeek-V4-Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
