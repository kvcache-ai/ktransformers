# Intel GPU Docker Guide (Beta)

## Prerequisites

* Docker must be installed and running on your system.
* Create a folder to store big models & intermediate files (e.g., /mnt/models)
* **Before proceeding, ensure the Intel GPU driver is installed correctly on your host:** [Installation Guide](./xpu.md#1-install-intel-gpu-driver)

---

## Building the Docker Image Locally

1. Clone the repository and navigate to the project directory:

   ```bash
   git clone https://github.com/kvcache-ai/ktransformers.git
   cd ktransformers
   ```

2. Build the Docker image using the XPU-specific [Dockerfile.xpu](../../Dockerfile.xpu):

   ```bash
   sudo http_proxy=$HTTP_PROXY \
        https_proxy=$HTTPS_PROXY \
        docker build \
          --build-arg http_proxy=$HTTP_PROXY \
          --build-arg https_proxy=$HTTPS_PROXY \
          -t kt_xpu:0.3.1 \
          -f Dockerfile.xpu \
          .
   ```

---

## Running the Container

### 1. Start the container

```bash
sudo docker run -td --privileged \
    --net=host \
    --device=/dev/dri \
    --shm-size="16g" \
    -v /path/to/models:/models \
    -e http_proxy=$HTTP_PROXY \
    -e https_proxy=$HTTPS_PROXY \
    --name ktransformers_xpu \
    kt_xpu:0.3.1
```

**Note**: Replace `/path/to/models` with your actual model directory path (e.g., `/mnt/models`).

---

### 2. Access the container

```bash
sudo docker exec -it ktransformers_xpu /bin/bash
```

---

### 3. Set required XPU environment variables (inside the container)

```bash
export SYCL_CACHE_PERSISTENT=1
export ONEAPI_DEVICE_SELECTOR=level_zero:0
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
```

---

### 4. Run the sample script

```bash
python ktransformers/local_chat.py \
  --model_path deepseek-ai/DeepSeek-R1 \
  --gguf_path <path_to_gguf_files> \
  --optimize_config_path ktransformers/optimize/optimize_rules/xpu/DeepSeek-V3-Chat.yaml \
  --cpu_infer <cpu_cores + 1> \
  --device xpu \
  --max_new_tokens 200
```

**Note**:

* Replace `<path_to_gguf_files>` with the path to your GGUF model files.
* Replace `<cpu_cores + 1>` with the number of CPU cores you want to use plus one.

---

## Additional Information

For more configuration options and usage details, refer to the [project README](../../README.md). To run KTransformers natively on XPU (outside of Docker), please refer to [xpu.md](./xpu.md).
