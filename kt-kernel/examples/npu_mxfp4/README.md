# Two-device MXFP4 LoRA on Ascend

This example trains the DeepSeek V4-Flash or V4.1-Flash-1017 text backbone with two expert-parallel NPU ranks. Each rank owns half of the frozen routed experts and their LoRA weights. Attention runs on the NPU and the shared expert runs on the CPU. The example uses a bounded forward operand cache and transfers transposed backward weights in groups of up to 16 expert IDs. The host transpose uses SSE2 and OpenMP on Linux x86-64.

Install KT-Kernel and the [KT Transformers V4 release source](https://github.com/kvcache-ai/transformers/tree/46da1fcfcf4c5d4ffbfd5c9a43d1d393a7ef513b) in the NPU environment. V4.1 uses the text model included in KT-Kernel. Its Apache-2.0 configuration and modeling files are adapted from [Michel Belleau's model contribution](https://github.com/huggingface/transformers/pull/48721) and the [KT text training backport](https://github.com/kvcache-ai/transformers/tree/85a62231652d4c40b7ab32ca6e0e8c5d8b1156e9). This entry point loads them explicitly. The standard KT and LLaMA-Factory entry points continue to select their usual expert backend.

Supply the native checkpoint directory with its configuration, tokenizer, safetensors index and weights. For a repackaged V4 checkpoint, use `--config` to specify the matching canonical model configuration directory. Configure CANN, HCCL and the two allocated devices before launching. The measured runtime uses torch 2.10.0+cpu, torch-npu 2.10.0.post4, CANN 9.2.0-beta.2, accelerate 1.10.1, safetensors 0.8.0 and NumPy 1.26.4.

From this directory, build the host transpose and run the example:

```sh
g++ -std=c++17 -O3 -fopenmp -fPIC -shared transpose.cpp -o transpose.so
python prepare_batch.py "$PACKED_ALPACA_DIR"
export ASCEND_RT_VISIBLE_DEVICES="$ALLOCATED_DEVICE_PAIR"
export RANK_TABLE_FILE="$TWO_RANK_TABLE"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
torchrun --standalone --nproc-per-node=2 train.py \
  --model "$MODEL_DIR" --batch batch.json --transpose-library ./transpose.so \
  --steps 2 --output "$RESULT_DIR"
```

The JSON batch contains one record per rank, each with integer arrays for `input_ids`, `attention_mask` and `labels`. The example uses equal sequence lengths and supervised-token counts on both ranks. The supplied extraction script reads packed Alpaca rows 677 and 1340, yielding two 1,024-token windows with 768 shifted supervised labels each. The measured batch SHA-256 is `a3c731d6490852df6aa1f1e85582d1ae3468fa801bbc631efbc185889b644c57`.

Each update uses the same batch, LoRA rank 8/alpha 16, FP32-master AdamW at 1e-5, gradient accumulation 1 and non-reentrant gradient checkpointing. Each rank writes one JSON record per update. Use step 2 for comparison. Input throughput divides 2,048 by the slower rank's elapsed seconds from forward start through gradient synchronization, optimizer update and parameter copy. The same interval accounts for 1,536 supervised tokens. The record also includes backward time, top-6 assignment counts, finite gradient counts and sampled adapter changes.
