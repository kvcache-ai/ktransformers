## How to use ktransformers long context framework

Currently, long context is only supported by our **local_chat.py** interface, and the integration with the server interface is under development.

To facilitate user management, we have uploaded the model config, gguf, and tokenizer to a repo. URL: https://huggingface.co/nilv234/internlm2_5_to_llama_1m/tree/main

By setting the model_path and gguf_path in the local_chat function to **/path/to/repo** and setting the mode to **"long_context"**, you can use the InternLM2.5-7B-Chat-1M model with 1m functionality on a 24G VRAM.

After running local_chat.py for the first time, a config.yaml file will be automatically created under ** ~/.ktransformers**. The relevant configurations for long context are as follows:

```python
chunk_size: 4096 # prefill chunk size
max_seq_len: 100000 # KVCache length
block_size: 128 # KVCache block size
local_windows_len: 4096 # The KVCache of length local_windows_len is stored on the GPU.
second_select_num: 96 # After preselection, each time select the number of KVCache blocks. If >= preselect_block_count, use the preselected blocks.
threads_num: 64 # CPU thread num
anchor_type: DYNAMIC # KVCache block representative token selection method.
kv_type: FP16
dense_layer_num: 0 # The first few layers do not need to fill or select KVCache
anchor_num: 1 # The number of representative tokens within a KVCache block.
preselect_block: False # Whether to preselect.
head_select_mode: SHARED # All kv_heads jointly select.
preselect_block_count: 96 # Number of preselected blocks.
layer_step: 1 # Select every few layers.
token_step: 1 # Select every few tokens.
```

The memory required for different context lengths is shown in the table below:

|                | 4K  | 32K  | 64K  | 128K | 512K | 1M     |
| -------------- | --- | ---- | ---- | ---- | ---- | ------ |
| DRAM Size (GB) | 0.5 | 4.29 | 8.58 | 17.1 | 68.7 | 145.49 |

Please choose an appropriate max_seq_len based on your DRAM size.
For example:
```python
python local_chat.py --model_path="/data/model/internlm2_5_to_llama_1m"  --gguf_path="/data/model/internlm2_5_to_llama_1m" --max_new_tokens=500 --cpu_infer=10  --use_cuda_graph=True  --mode="long_context" --prompt_file="/path/to/file"
```

If you've already specified the input text via the prompt_file, just press Enter when the terminal displays chat: to begin.