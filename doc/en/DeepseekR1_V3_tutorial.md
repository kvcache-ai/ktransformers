## prerequisites
We run our best performance tests on <br>
cpu: Intel(R) Xeon(R) Gold 6454S 1T DRAM(2 NUMA nodes)<br>
gpu: 4090D 24G VRAM <br>
## bench result
### V0.2
#### settings
- model: DeepseekV3-q4km（int4）<br>
- CPU: cpu_model_name：Intel(R) Xeon(R) Gold 6454S, 32 cores per socket, 2 socket, 2numa nodes
- GPU: 4090D 24GVRAM
- we test after enough warm up!
#### memory consumption:
  - single socket: 382G DRAM, 12G VRAM
  - dual socket: 1T DRAM, 12G VRAM

#### Benchmark Results

"6 experts" case is part of v0.3's preview

| Prompt<br>(500 tokens) | Dual socket Ktrans (6 experts) | Dual socket Ktrans (8 experts) | Single socket Ktrans (6 experts) | Single socket Ktrans (8 experts)| Llama (8 experts) | 
| --- | --- | --- | --- | --- | --- | 
| Prefill token/s | 97.32 | 82.94 | 65.14 | 54.21 | 10.31 |
| Decode token/s | 13.69 | 12.208 | 10.303 | 8.73 |4.51 |

**The highest speedup reaches up to <u>x3.03</u> in decoding and <u>x9.44</u> in prefill.**

### V0.3-Preview
#### settings
- model: DeepseekV3-BF16 (online quant into int8 for CPU and int4 for GPU)
- CPU: cpu_model_name：Intel(R) Xeon(R) Gold 6454S, 32 cores per socket, 2 socket, 2 numa nodes
- GPU: (1~4)x 4090D 24GVRAM (requires more VRAM for longer prompt)

#### memory consumptions:
- 644GB DRAM, at least 12GB VRAM

#### Benchmark Results
| Prompt length  | 1K  | 2K  | 4K  | 8K |
|---------------|-----|-----|-----|-----|
| KTrans (8 experts) Prefill token/s |   185.96  |  255.26   |  252.58   |  195.62   |
| KTrans (6 experts) Prefill token/s |   203.70  |  286.55   |  271.08   |  207.20   |

**The prefill of KTrans V0.3 is up to <u>x3.45</u> times faster than KTrans V0.2, and is up to <u>x63.53</u> times faster than Llama.**
**The decoding speed is the same as KTrans V0.2 (6 experts version) so it is omitted.**

The main acceleration comes from 
- Intel AMX instruction set and our specially designed cache friendly memory layout
- Expert selection strategy that selects fewer experts based on offline profile results of out of domain data

## how to run
### v0.2 showcase
#### single socket version(32 cores)
our local_chat test command is:
``` shell
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
numactl -N 1 -m 1 python ./ktransformers/local_chat.py --model_path <your model path> --gguf_path <your gguf path>  --prompt_file <your promt txt file>  --cpu_infer 33  --cache_lens 1536 
<when you see chat, then press enter to load the text prompt_file>
```
\<your model path\> can be local or set from onlie hugging face like deepseek-ai/DeepSeek-V3. If onlie encounters connection problem, try use mirror(hf-mirror.com) <br>
\<your gguf path\> can also be onlie, but as its large we recommend you download it and quantize the model to what you want.<br>
the command numactl -N 1 -m 1 aims to adoid data transfer between numa nodes.
### dual socket version(64 cores)
make suer before you install(use install.sh or `make dev_install`), setting the env var `USE_NUMA=1` by `export USE_NUMA=1`(if already installed, reinstall it with this env var set) <br>
our local_chat test command is:
``` shell
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
export USE_NUMA=1
make dev_install # or sh ./install.sh
python ./ktransformers/local_chat.py --model_path <your model path> --gguf_path <your gguf path>  --prompt_file <your promt txt file>  --cpu_infer 65  --cache_lens 1536 
<when you see chat, then press enter to load the text prompt_file>
```
The parameters meaning is the same. But As we  use dual socket, so we set cpu_infer to 65.
## some explanations
1. From our perspective on DeepSeekV2, DeepSeekV3 and DeepSeekR1, 
when we slightly decrease the activation experts num in inference, 
the output quality doesn't change(within 1% accuracy drop),But the speed of decoding and prefill 
is speed up about 30% which is inspiring. So our showcase makes use of this finding, 
changing the activation experts of DeepSeekV3/R1 from 8 to 6. <br>
2. Also we want to make further use of our two NUMA nodes on Xeon Gold cpu. 
To avoid the cost of data transfer between nodes, we "copy" the critical matrix on 
both nodes which takes more memory consumption but accelerates the prefill and decoding process.
But this method takes huge memory and slow when loading weights, So be patient when loading
and monitor the memory usage.(we are considering to make this method as an option)<br>
3. the command args `--cpu_infer 65` specifies how many cores to use(it's ok that it exceeds the physical number, 
but it's not the more the better. Adjust it slight lower to your actual number of cores)<br>
