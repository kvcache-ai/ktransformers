# Running DeepSeek-V4-Flash on a Single Ascend NPU

DeepSeek-V4-Flash served from **one** Ascend die. Attention, the dense layers and 32 resident
experts per layer run on the NPU in W8A8; the remaining 224 routed experts stay on the host as
MXFP4 and are computed by KT-Kernel's `LLAMAFILE` MoE. Follow the steps in order.

## Requirements

| Platform | `DSV4_SOC` | HBM per die | Host |
|---|---|---|---|
| Atlas 800I A2 (910B3) | `ascend910b` | 64 GB | Kunpeng 920, 192 cores, 8 NUMA, 1.5 TB RAM |
| Atlas A3 (910_93) | `ascend910_93` | 61 GB | 40 cores, 1 NUMA, 229 GB RAM |

| Component | Version |
|---|---|
| CANN toolkit | 9.0.0 |
| Driver | 25.5.1 |
| Python | 3.11 |
| torch / torch_npu | 2.9.1 / 2.9.1 |
| GCC at `/usr/bin/gcc` | ≥ 11 |
| System packages | `libhwloc-dev`, `libnuma-dev`, `pkg-config`, `cmake`, `patchelf` |

Disk: about 570 GB for the three weight artifacts.

## Step 1: Start the Container

Skip this on a native install.

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

## Step 2: Get the Code

```bash
export DSV4_WORKSPACE=$HOME/dsv4-workspace
mkdir -p "${DSV4_WORKSPACE}" && cd "${DSV4_WORKSPACE}"

git clone https://github.com/kvcache-ai/ktransformers.git
git clone -b dsv4-cann9-no-patch https://github.com/Pan-Boyi/sglang.git

cd ktransformers
git submodule update --init --progress third_party/llama.cpp third_party/pybind11
```

Install the system packages if they are missing:

```bash
sudo apt-get install -y build-essential cmake git libhwloc-dev libhwloc15 \
                        libnuma-dev patchelf pkg-config
# openEuler / CentOS: dnf install -y hwloc-devel numactl-devel pkgconfig
```

## Step 3: Configure

Everything is driven by one file. Edit the four values at the top; the rest is derived.

```bash
cat > ~/dsv4.env <<'EOF'
# ---- edit these ----
export DSV4_WORKSPACE=$HOME/dsv4-workspace   # clones and build artifacts
export DSV4_MODEL_ROOT=/path/to/models       # parent of the three weight directories
export DSV4_NPU_DEVICE_ID=0                  # an idle die
export DSV4_PORT=18080

# ---- leave alone ----
export KTRANSFORMERS_REPO=$DSV4_WORKSPACE/ktransformers
export DSV4_TOOLS=$KTRANSFORMERS_REPO/kt-kernel/tools/ascend_dsv4
export DSV4_ARTIFACT_DIR=$DSV4_WORKSPACE/dsv4-artifacts
export DSV4_LOG_DIR=$DSV4_WORKSPACE/dsv4-logs
EOF

source ~/dsv4.env
mkdir -p "$DSV4_WORKSPACE" "$DSV4_LOG_DIR" "$DSV4_ARTIFACT_DIR"
bash "$DSV4_TOOLS/dsv4_env.sh" --show
```

`--show` prints everything that was auto-detected — CANN root, SoC, NUMA node count, thread
counts, weight paths. Check that they look right before continuing. Any of them can be
overridden by exporting it in `~/dsv4.env`; see [Configuration Reference](#configuration-reference).

## Step 4: Download the Weights

```bash
source ~/dsv4.env

huggingface-cli download deepseek-ai/DeepSeek-V4-Flash \
  --local-dir "${DSV4_MODEL_ROOT}/DeepSeek-V4-Flash"

# and a published W8A8 compressed-tensors quantization of the same model into
#   ${DSV4_MODEL_ROOT}/DeepSeek-V4-Flash-W8A8
```

| Artifact | Variable | Size | Used for |
|---|---|---|---|
| W8A8 checkpoint | `DSV4_MODEL_PATH` | ~275 GB | `--model-path`: attention, dense layers, resident experts |
| Official checkpoint | `DSV4_NATIVE_CKPT` | ~150 GB | source for the GGUF conversion only |
| 43 per-layer MXFP4 GGUF | `DSV4_GGUF_DIR` | ~138 GiB | `--kt-weight-path`: the CPU experts, produced in Step 5 |

## Step 5: Build and Convert

```bash
source ~/dsv4.env
bash "$DSV4_TOOLS/setup.sh" all
```

That runs six steps in order. Each is separately invocable as `setup.sh <step>`, and a vendor
image already satisfies some of them:

| Step | What it does | Time |
|---|---|---|
| `deps` | SGLang's NPU runtime dependencies | minutes |
| `kt-kernel` | build KT-Kernel, produce and install a wheel | 10–30 min |
| `sgl-kernel` | build `sgl_kernel_npu`, `deep_ep`, `attentions`, `torch_memory_saver` | 20–40 min |
| `cann-ops` | build the `customize`, `custom_ops` and `custom_transformer` packages | 40–90 min |
| `gguf` | convert the checkpoint to the per-layer MXFP4 GGUF set | hours |
| `check` | verify the environment; exit 0 means safe to launch | seconds |

Check what is already present before spending the time:

```bash
source ~/dsv4.env
ls "${CANN_VENDORS_DIR}"                                  # want: customize  custom_transformer
python3 -c "import torch, torch_npu, custom_ops; print('custom_ops ok')"
python3 -c "import sgl_kernel_npu; print('sgl_kernel_npu ok')"
```

Install the KT-Kernel wheel if you ran `kt-kernel` on its own:

```bash
python3 -m pip install --no-deps "${DSV4_ARTIFACT_DIR}"/wheels/kt_kernel-*.whl
```

The `cann-ops` step builds from two pinned upstream commits:

| Package | Source | Commit |
|---|---|---|
| `customize`, `custom_ops` | `gitcode.com/cann/cann-recipes-infer` | `1c8e6bcc2333d95b3db47d873210f921113d6d11` |
| `custom_transformer` | `gitcode.com/cann/ops-transformer` | `8edcd591e83e536e9ee98a9ce0de3af02ea4f3ea` |

When Step 5 finishes, `setup.sh check` must print `PREFLIGHT OK`.

## Step 6: Launch

```bash
source ~/dsv4.env
bash "$DSV4_TOOLS/serve.sh"
tail -f "${DSV4_LOG_DIR}/serve.log"
```

Wait until it answers:

```bash
until curl -sf -m5 --noproxy '*' "http://127.0.0.1:${DSV4_PORT}/health" >/dev/null; do
  echo "$(date +%T) still loading..."; sleep 30
done
echo "ready"
```

Stop it with:

```bash
kill -INT  "$(cat "${DSV4_LOG_DIR}/serve.log.pid")"; sleep 5
kill -TERM "$(cat "${DSV4_LOG_DIR}/serve.log.pid")"
```

## Step 7: Verify

```bash
source ~/dsv4.env
bash "$DSV4_TOOLS/verify.sh"
```

All gates must pass. The last one sends a greedy prompt and requires a non-empty completion.

## Step 8: Use It

```bash
source ~/dsv4.env
bash "${DSV4_TOOLS}/verify.sh" chat "${DSV4_PORT}"     # interactive
```

Or over the OpenAI-compatible API:

```bash
source ~/dsv4.env
curl -s --noproxy '*' -X POST "http://127.0.0.1:${DSV4_PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"dsv4","messages":[{"role":"user","content":"Explain in three sentences what a Mixture-of-Experts model is."}],"temperature":0.6,"max_tokens":300}' \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content']); print('---', d['usage'])"
```

## Configuration Reference

Export any of these in `~/dsv4.env` before sourcing. `dsv4_env.sh --show` prints the resolved
values.

| Variable | Default | Description |
|---|---|---|
| `ASCEND_INSTALL_ROOT` | first of `$HOME/Ascend`, `/usr/local/Ascend`, `/opt/Ascend` | CANN install prefix |
| `DSV4_MODEL_ROOT` | `${DSV4_WORKSPACE}/models` | parent of the three weight artifacts |
| `DSV4_MODEL_PATH` | `${DSV4_MODEL_ROOT}/DeepSeek-V4-Flash-W8A8` | W8A8 checkpoint |
| `DSV4_NATIVE_CKPT` | `${DSV4_MODEL_ROOT}/DeepSeek-V4-Flash` | official checkpoint |
| `DSV4_GGUF_DIR` | `${DSV4_MODEL_ROOT}/cache` | the 43 per-layer GGUF files |
| `DSV4_SOC` | detected | `ascend910b` or `ascend910_93` |
| `DSV4_NPU_DEVICE_ID` | `0` | becomes `ASCEND_RT_VISIBLE_DEVICES` |
| `DSV4_THREADPOOL_COUNT` | NUMA node count | `--kt-threadpool-count` |
| `DSV4_CPUINFER` | NUMA × 16 | `--kt-cpuinfer`, total CPU MoE threads |
| `DSV4_NUM_GPU_EXPERTS` | `32` | resident experts per layer, ~1.0 GiB HBM each |
| `DSV4_MEM_FRACTION` | `0.81` | `--mem-fraction-static` |
| `DSV4_CONTEXT_LENGTH` | `65536` | `--context-length` |
| `DSV4_CHUNKED_PREFILL_SIZE` | `32768` | must be a positive multiple of 128 and ≥ your longest prompt |
| `DSV4_PREFILL_STREAM` | unset | `1` enables streaming prefill, see below |
| `DSV4_EXTRA_FLAGS` | empty | appended to the launch command |

### Streaming prefill

Off by default. It makes long prompts much faster and short ones slower, so enable it only if
you serve long contexts:

```bash
source ~/dsv4.env
export DSV4_PREFILL_STREAM=1
bash "$DSV4_TOOLS/serve.sh"
```

Long prompts need matching expert and memory settings:

| Longest prompt | `DSV4_NUM_GPU_EXPERTS` | `DSV4_MEM_FRACTION` | `DSV4_EXTRA_FLAGS` |
|---:|---:|---:|:---|
| up to ~12k | 32 | 0.81 | — |
| up to ~19k | 32 | 0.81 | `--swa-full-tokens-ratio 0.35` |
| up to ~39k | 28 | 0.775 | — |

Confirm streaming actually engaged:

```bash
echo "inline resident : $(grep -c 'inline resident' "$DSV4_LOG_DIR/serve.log")"   # want > 0
echo "hybrid fallback : $(grep -cE 'streaming failed|hybrid fallback' "$DSV4_LOG_DIR/serve.log")"   # want 0
```

## Optional: Accuracy Validation

```bash
source ~/dsv4.env
python3 -m pip install evalscope

cd "$DSV4_WORKSPACE"
[ -d cann-recipes-infer ] || git clone --depth 1 https://gitcode.com/cann/cann-recipes-infer.git
export DSV4_RECIPES="$DSV4_WORKSPACE/cann-recipes-infer/integration/sglang/dsv4-flash-single-npu-moe-offload"
echo "export DSV4_RECIPES=$DSV4_RECIPES" >> ~/dsv4.env

REPEATS=3 PORT="$DSV4_PORT" \
MODEL_PATH="$DSV4_MODEL_ROOT/DeepSeek-V4-Flash-W8A8" \
OUT_DIR="$DSV4_LOG_DIR/gpqa" \
bash "$DSV4_RECIPES/scripts/tools/gpqa_accuracy_repeat.sh" 2>&1 | tee "$DSV4_LOG_DIR/gpqa.log"
```

GPQA-Diamond, three rounds: 72.22% / 71.72% / 75.76%, mean **73.23%** (sample SD 2.20 pp).

## Optional: Throughput Validation

```bash
source ~/dsv4.env
TARGET_TOKENS_LIST="130 1000 4000 8000" \
MAX_NEW=1000 REPEAT=3 WARMUP=1 \
PORT="$DSV4_PORT" PY=python3 \
bash "$DSV4_RECIPES/scripts/tools/decode_throughput_test.sh" | tee "$DSV4_LOG_DIR/perf.log"
```

Decode is host-memory-bandwidth bound, so measure on an idle machine.

## Measured Results

Single Atlas A3 die (`Ascend910_9362`, 61.3 GB HBM), 1 NUMA node, 40 cores,
`--kt-cpuinfer 32 --kt-threadpool-count 1`, NPU graph on, one request at a time,
`max_new_tokens=1000`, one warmup and three measured iterations per bucket. All five feature
switches enabled:

```
KT_PREFILL_STREAM=1  KT_PREFILL_STREAM_THRESHOLD=512  KT_MXFP4_DEPOOL=1
KT_DYNAMIC_RESIDENT=1  KT_SIDE_STREAM=1  KT_MXFP4_GGUF_DEDUP=1
```

| Prompt tokens | Prefill | Decode | Settings |
|--------------:|--------:|-------:|:---------|
| 118 | 1.5 s <sup>1</sup> | 18.32 tok/s | 32 experts, `mf` 0.81 |
| 801 | 15.6 s | 20.12 tok/s | 32 experts, `mf` 0.81 |
| 3,944 | 16.0 s | 19.58 tok/s | 32 experts, `mf` 0.81 |
| 7,823 | 16.5 s | 19.86 tok/s | 32 experts, `mf` 0.81 |
| 15,568 | 17.5 s | 20.48 tok/s | + `--swa-full-tokens-ratio 0.35` |
| 31,540 | 20.4 s | 19.84 tok/s | 28 experts, `mf` 0.775 |

<sup>1</sup> Below `KT_PREFILL_STREAM_THRESHOLD`, so this row uses the hybrid path.

Spread across the three measured iterations is under 0.3% in every bucket. Prefill is
effectively a fixed cost: across a 39× range of prompt length it moves only from 15.6 s to
20.4 s. Decode is flat across context length because it is dominated by per-token host-side
expert reads, not by KV growth.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'sgl_kernel_npu'` | `bash "$DSV4_TOOLS/setup.sh" sgl-kernel` |
| `import custom_ops` fails with `libc10.so: cannot open shared object file` | import torch first: `python3 -c "import torch, torch_npu, custom_ops"` |
| `invalid feature modifier 'bf16'` during the build | `/usr/bin/gcc` is older than 10; install GCC ≥ 11 |
| Startup: `Raise --mem-fraction-static above …` | streaming prefill needs a higher `DSV4_MEM_FRACTION`, see the table above |
| Long prompt hangs | the prompt exceeds the SWA pool; use the long-prompt settings above |

## Known Limitations

- Single die only: `--tensor-parallel-size 1 --expert-parallel-size 1`.
- `--max-running-requests 1`; concurrent serving is not validated.
- Prefix caching (`--disable-radix-cache`) and shared-expert fusion are not supported with a
  split expert set.
- `--chunked-prefill-size` must be at least your longest prompt; chunked prefill across the
  NSA compressor is not supported.

## Additional Resources

- [`kt-kernel/tools/ascend_dsv4/README.md`](../../kt-kernel/tools/ascend_dsv4/README.md) — script reference
- [CANN recipes](https://gitcode.com/cann/cann-recipes-infer) — the upstream Ascend recipe this deployment follows
