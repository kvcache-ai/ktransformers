<!-- omit in toc -->
# How to Run DeepSeek-R1
- [Preparation](#preparation)
- [Installation](#installation)
  - [Attention](#attention)
  - [Supported models include:](#supported-models-include)
  - [Support quantize format:](#support-quantize-format)

In this document, we will show you how to install and run KTransformers on your local machine. There are two versions: 
* V0.2 is the current main branch.
* V0.3 is a preview version only provides binary distribution for now.
* To reproduce our DeepSeek-R1/V3 results, please refer to [Deepseek-R1/V3 Tutorial](./DeepseekR1_V3_tutorial.md) for more detail settings after installation.
## Preparation
Some preparation:

- CUDA 12.1 and above, if you didn't have it yet, you may install from [here](https://developer.nvidia.com/cuda-downloads).
  
  ```sh
  # Adding CUDA to PATH
  if [ -d "/usr/local/cuda/bin" ]; then
      export PATH=$PATH:/usr/local/cuda/bin
  fi

  if [ -d "/usr/local/cuda/lib64" ]; then
      export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
      # Or you can add it to /etc/ld.so.conf and run ldconfig as root:
      # echo "/usr/local/cuda-12.x/lib64" | sudo tee -a /etc/ld.so.conf
      # sudo ldconfig
  fi

  if [ -d "/usr/local/cuda" ]; then
      export CUDA_PATH=$CUDA_PATH:/usr/local/cuda
  fi
  ```

- Linux-x86_64 with gcc, g++ and cmake (using Ubuntu as an example)
  
  ```sh
  sudo apt-get update
  sudo apt-get install build-essential cmake ninja-build
  ```

- We recommend using [Miniconda3](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) or [Anaconda3](https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh) to create a virtual environment with Python=3.11 to run our program. Assuming your Anaconda installation directory is `~/anaconda3`, you should ensure that the version identifier of the GNU C++standard library used by Anaconda includes `GLIBCXX-3.4.32`

  
  ```sh
  conda create --name ktransformers python=3.11
  conda activate ktransformers # you may need to run ‘conda init’ and reopen shell first
  
  conda install -c conda-forge libstdcxx-ng # Anaconda provides a package called `libstdcxx-ng` that includes a newer version of `libstdc++`, which can be installed via `conda-forge`.

  strings ~/anaconda3/envs/ktransformers-0.3/lib/libstdc++.so.6 | grep GLIBCXX
  ```

- Make sure that PyTorch, packaging, ninja is installed You can also [install previous versions of PyTorch](https://pytorch.org/get-started/previous-versions/)
  
  ```
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
  pip3 install packaging ninja cpufeature numpy
  ```

 - At the same time, you should download and install the corresponding version of flash-attention from https://github.com/Dao-AILab/flash-attention/releases.

## Installation
### Attention
If you want to use numa support, not only do you need to set USE_NUMA=1, but you also need to make sure you have installed the libnuma-dev (`sudo apt-get install libnuma-dev` may help you).

<!-- 1. ~~Use a Docker image, see [documentation for Docker](./doc/en/Docker.md)~~
   
   >We are working on the latest docker image, please wait for a while.

2. ~~You can install using Pypi (for linux):~~
    > We are working on the latest pypi package, please wait for a while.
   
   ```
   pip install ktransformers --no-build-isolation
   ```
   
   for windows we prepare a pre compiled whl package on [ktransformers-0.2.0+cu125torch24avx2-cp312-cp312-win_amd64.whl](https://github.com/kvcache-ai/ktransformers/releases/download/v0.2.0/ktransformers-0.2.0+cu125torch24avx2-cp312-cp312-win_amd64.whl), which require cuda-12.5, torch-2.4, python-3.11, more pre compiled package are being produced.  -->

* Download source code and compile:
   
   - init source code 
     
     ```sh
     git clone https://github.com/kvcache-ai/ktransformers.git
     cd ktransformers
     git submodule init
     git submodule update
     ```

   - [Optional] If you want to run with website, please [compile the website](./api/server/website.md) before execute ```bash install.sh```

   - For Linux
     - For simple install:
     
        ```shell
        bash install.sh
        ```
     - For those who have two cpu and 1T RAM:

       ```shell
        # Make sure your system has dual sockets and double size RAM than the model's size (e.g. 1T RAM for 512G model)
        apt install libnuma-dev
        export USE_NUMA=1
        bash install.sh # or #make dev_install
        ```

   - For Windows
     
     ```shell
     install.bat
     ```

* If you are developer, you can make use of the makefile to compile and format the code. <br> the detailed usage of makefile is [here](./makefile_usage.md) 

<h3>Local Chat</h3>
We provide a simple command-line local chat Python script that you can run for testing.

> Note: this is a very simple test tool only support one round chat without any memory about last input, if you want to try full ability of the model, you may go to [RESTful API and Web UI](#id_666). 

<h4>Run Example</h4>

```shell
# Begin from root of your cloned repo!
# Begin from root of your cloned repo!!
# Begin from root of your cloned repo!!! 

# Download mzwing/DeepSeek-V2-Lite-Chat-GGUF from huggingface
mkdir DeepSeek-V2-Lite-Chat-GGUF
cd DeepSeek-V2-Lite-Chat-GGUF

wget https://huggingface.co/mradermacher/DeepSeek-V2-Lite-GGUF/resolve/main/DeepSeek-V2-Lite.Q4_K_M.gguf -O DeepSeek-V2-Lite-Chat.Q4_K_M.gguf

cd .. # Move to repo's root dir

# Start local chat
python -m ktransformers.local_chat --model_path deepseek-ai/DeepSeek-V2-Lite-Chat --gguf_path ./DeepSeek-V2-Lite-Chat-GGUF

# If you see “OSError: We couldn't connect to 'https://huggingface.co' to load this file”, try：
# GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite
# python  ktransformers.local_chat --model_path ./DeepSeek-V2-Lite --gguf_path ./DeepSeek-V2-Lite-Chat-GGUF
```

It features the following arguments:

- `--model_path` (required): Name of the model (such as "deepseek-ai/DeepSeek-V2-Lite-Chat" which will automatically download configs from [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)). Or if you already got local files  you may directly use that path to initialize the model.  
  
  > Note: <strong>.safetensors</strong> files are not required in the directory. We only need config files to build model and tokenizer.

- `--gguf_path` (required): Path of a directory containing GGUF files which could that can be downloaded from [Hugging Face](https://huggingface.co/mzwing/DeepSeek-V2-Lite-Chat-GGUF/tree/main). Note that the directory should only contains GGUF of current model, which means you need one separate directory for each model.

- `--optimize_config_path` (required except for Qwen2Moe and DeepSeek-V2): Path of YAML file containing optimize rules. There are two rule files pre-written in the [ktransformers/optimize/optimize_rules](ktransformers/optimize/optimize_rules) directory for optimizing DeepSeek-V2 and Qwen2-57B-A14, two SOTA MoE models.

- `--max_new_tokens`: Int (default=1000). Maximum number of new tokens to generate.

- `--cpu_infer`: Int (default=10). The number of CPUs used for inference. Should ideally be set to the (total number of cores - 2).

<details>
<summary>Supported Models/quantization</summary>

### Supported models include:

| ✅ **Supported Models** | ❌ **Deprecated Models** |
|------------------------|------------------------|
| DeepSeek-R1 | ~~InternLM2.5-7B-Chat-1M~~ |
| DeepSeek-V3 |  |
| DeepSeek-V2 |  |
| DeepSeek-V2.5 |  |
| Qwen2-57B |  |
| DeepSeek-V2-Lite |  |
| Mixtral-8x7B |  |
| Mixtral-8x22B |  |

### Support quantize format:

| ✅ **Supported Formats** | ❌ **Deprecated Formats** |
|--------------------------|--------------------------|
| Q2_K_L | ~~IQ2_XXS~~ |
| Q2_K_XS |  |
| Q3_K_M |  |
| Q4_K_M |  |
| Q5_K_M |  |
| Q6_K |  |
| Q8_0 |  |
</details>

<details>
<summary>Suggested Model</summary>

| Model Name                     | Model Size | VRAM  | Minimum DRAM    | Recommended DRAM  |
| ------------------------------ | ---------- | ----- | --------------- | ----------------- |
| DeepSeek-R1-q4_k_m		 | 377G       | 14G   | 382G            | 512G		    |
| DeepSeek-V3-q4_k_m		 | 377G       | 14G   | 382G            | 512G		    |
| DeepSeek-V2-q4_k_m             | 133G       | 11G   | 136G            | 192G              |
| DeepSeek-V2.5-q4_k_m           | 133G       | 11G   | 136G            | 192G              |
| DeepSeek-V2.5-IQ4_XS           | 117G       | 10G   | 107G            | 128G              |
| Qwen2-57B-A14B-Instruct-q4_k_m | 33G        | 8G    | 34G             | 64G               |
| DeepSeek-V2-Lite-q4_k_m        | 9.7G       | 3G    | 13G             | 16G               |
| Mixtral-8x7B-q4_k_m            | 25G        | 1.6G  | 51G             | 64G               |
| Mixtral-8x22B-q4_k_m           | 80G        | 4G    | 86.1G           | 96G               |
| InternLM2.5-7B-Chat-1M         | 15.5G      | 15.5G | 8G(32K context) | 150G (1M context) |


More will come soon. Please let us know which models you are most interested in. 

Be aware that you need to be subject to their corresponding model licenses when using [DeepSeek](https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/LICENSE) and [QWen](https://huggingface.co/Qwen/Qwen2-72B-Instruct/blob/main/LICENSE).
</details>


<details>
  <summary>Click To Show how to run other examples</summary>

* Qwen2-57B

  ```sh
  pip install flash_attn # For Qwen2

  mkdir Qwen2-57B-GGUF && cd Qwen2-57B-GGUF

  wget https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct-GGUF/resolve/main/qwen2-57b-a14b-instruct-q4_k_m.gguf?download=true -O qwen2-57b-a14b-instruct-q4_k_m.gguf

  cd ..

  python -m ktransformers.local_chat --model_name Qwen/Qwen2-57B-A14B-Instruct --gguf_path ./Qwen2-57B-GGUF

  # If you see “OSError: We couldn't connect to 'https://huggingface.co' to load this file”, try：
  # GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct
  # python  ktransformers/local_chat.py --model_path ./Qwen2-57B-A14B-Instruct --gguf_path ./DeepSeek-V2-Lite-Chat-GGUF
  ```

* Deepseek-V2
  
  ```sh
  mkdir DeepSeek-V2-Chat-0628-GGUF && cd DeepSeek-V2-Chat-0628-GGUF
  # Download weights
  wget https://huggingface.co/bartowski/DeepSeek-V2-Chat-0628-GGUF/resolve/main/DeepSeek-V2-Chat-0628-Q4_K_M/DeepSeek-V2-Chat-0628-Q4_K_M-00001-of-00004.gguf -o DeepSeek-V2-Chat-0628-Q4_K_M-00001-of-00004.gguf
  wget https://huggingface.co/bartowski/DeepSeek-V2-Chat-0628-GGUF/resolve/main/DeepSeek-V2-Chat-0628-Q4_K_M/DeepSeek-V2-Chat-0628-Q4_K_M-00002-of-00004.gguf -o DeepSeek-V2-Chat-0628-Q4_K_M-00002-of-00004.gguf
  wget https://huggingface.co/bartowski/DeepSeek-V2-Chat-0628-GGUF/resolve/main/DeepSeek-V2-Chat-0628-Q4_K_M/DeepSeek-V2-Chat-0628-Q4_K_M-00003-of-00004.gguf -o DeepSeek-V2-Chat-0628-Q4_K_M-00003-of-00004.gguf
  wget https://huggingface.co/bartowski/DeepSeek-V2-Chat-0628-GGUF/resolve/main/DeepSeek-V2-Chat-0628-Q4_K_M/DeepSeek-V2-Chat-0628-Q4_K_M-00004-of-00004.gguf -o DeepSeek-V2-Chat-0628-Q4_K_M-00004-of-00004.gguf

  cd ..

  python -m ktransformers.local_chat --model_name deepseek-ai/DeepSeek-V2-Chat-0628 --gguf_path ./DeepSeek-V2-Chat-0628-GGUF

  # If you see “OSError: We couldn't connect to 'https://huggingface.co' to load this file”, try：

  # GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat-0628

  # python -m ktransformers.local_chat --model_path ./DeepSeek-V2-Chat-0628 --gguf_path ./DeepSeek-V2-Chat-0628-GGUF
  ```

| model name | weights download link |
|----------|----------|
| Qwen2-57B | [Qwen2-57B-A14B-gguf-Q4K-M](https://huggingface.co/Qwen/Qwen2-57B-A14B-Instruct-GGUF/tree/main) |
| DeepseekV2-coder |[DeepSeek-Coder-V2-Instruct-gguf-Q4K-M](https://huggingface.co/LoneStriker/DeepSeek-Coder-V2-Instruct-GGUF/tree/main) |
| DeepseekV2-chat |[DeepSeek-V2-Chat-gguf-Q4K-M](https://huggingface.co/bullerwins/DeepSeek-V2-Chat-0628-GGUF/tree/main) |
| DeepseekV2-lite | [DeepSeek-V2-Lite-Chat-GGUF-Q4K-M](https://huggingface.co/mzwing/DeepSeek-V2-Lite-Chat-GGUF/tree/main) |
| DeepSeek-R1 | [DeepSeek-R1-gguf-Q4K-M](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-Q4_K_M) |

</details>

<!-- pin block for jump -->
<span id='id_666'> 

<h3>RESTful API and Web UI  </h3>


Start without website:

```sh
ktransformers --model_path deepseek-ai/DeepSeek-V2-Lite-Chat --gguf_path /path/to/DeepSeek-V2-Lite-Chat-GGUF --port 10002
```

Start with website:

```sh
ktransformers --model_path deepseek-ai/DeepSeek-V2-Lite-Chat --gguf_path /path/to/DeepSeek-V2-Lite-Chat-GGUF  --port 10002 --web True
```

Or you want to start server with transformers, the model_path should include safetensors

```bash
ktransformers --type transformers --model_path /mnt/data/model/Qwen2-0.5B-Instruct --port 10002 --web True
```

Access website with url [http://localhost:10002/web/index.html#/chat](http://localhost:10002/web/index.html#/chat) :

<p align="center">
  <picture>
    <img alt="Web UI" src="https://github.com/user-attachments/assets/615dca9b-a08c-4183-bbd3-ad1362680faf" width=90%>
  </picture>
</p>

More information about the RESTful API server can be found [here](doc/en/api/server/server.md). You can also find an example of integrating with Tabby [here](doc/en/api/server/tabby.md).
