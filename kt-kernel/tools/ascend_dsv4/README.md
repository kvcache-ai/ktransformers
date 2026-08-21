# DeepSeek-V4-Flash single-NPU deployment scripts

Companion scripts for
[`doc/en/DeepSeek-V4-Flash_tutorial_for_Ascend_NPU.md`](../../../doc/en/DeepSeek-V4-Flash_tutorial_for_Ascend_NPU.md).
Read the tutorial for the walkthrough; this file is the script reference.

All configuration lives in [`dsv4_env.sh`](./dsv4_env.sh), which every other script sources.
Override any value by exporting it first; `--show` prints what got resolved.

## Order

```bash
export DSV4_MODEL_ROOT=/path/to/models
export DSV4_NPU_DEVICE_ID=0

bash dsv4_env.sh --show      # 1. check the detected configuration
bash setup.sh all            # 2. build and convert everything
bash serve.sh                # 3. launch
bash verify.sh               # 4. acceptance checks
bash verify.sh chat          # 5. talk to it
```

## Files

| File | What it does |
|---|---|
| `dsv4_env.sh` | Configuration and environment. Sourced by everything else. `--show` prints the resolved values without changing your shell. |
| `setup.sh` | `deps`, `kt-kernel`, `sgl-kernel`, `cann-ops`, `gguf`, `check`, or `all`. Run with no arguments for the list. |
| `serve.sh` | Launches the server. `--foreground` stays attached. |
| `verify.sh` | Acceptance checks against a running server. `chat [port]` opens an interactive client. |

Weight conversion tools are in [`../mxfp4_gguf/`](../mxfp4_gguf):

| File | What it does |
|---|---|
| `convert_mxfp4_gguf.py` | `batch` converts every layer, `layer` converts one. |
| `verify_mxfp4_gguf.py` | `set` checks a converted directory, `layer` checks one file. |

`setup.sh gguf` drives both.
