# KTransformers Docker Packaging Guide

This directory contains scripts for building and distributing KTransformers Docker images with standardized naming conventions.

## Overview

The packaging system provides:

- **Automated version detection** from sglang, ktransformers, and LLaMA-Factory
- **Multi-CPU variant support** (AMX, AVX512, AVX2) with runtime auto-detection
- **Standardized naming convention** for easy identification and management
- **Two distribution methods**:
  - Local tar file export for offline distribution
  - DockerHub publishing for online distribution

## Naming Convention

Docker images follow this naming pattern:

```
sglang-v{sglang版本}_ktransformers-v{ktransformers版本}_{cpu信息}_{gpu信息}_{功能模式}_{时间戳}
```

### Example Names

**Tar file:**
```
sglang-v0.5.6_ktransformers-v0.4.3_x86-intel-multi_cu128_sft_llamafactory-v0.9.3_20241212143022.tar
```

**DockerHub tags:**
```
Full tag:
kvcache/ktransformers:sglang-v0.5.6_ktransformers-v0.4.3_x86-intel-multi_cu128_sft_llamafactory-v0.9.3_20241212143022

Simplified tag:
kvcache/ktransformers:v0.4.3-cu128
```

### Name Components

| Component | Description | Example |
|-----------|-------------|---------|
| sglang version | SGLang package version | `v0.5.6` |
| ktransformers version | KTransformers version | `v0.4.3` |
| cpu info | CPU instruction set support | `x86-intel-multi` (includes AMX/AVX512/AVX2) |
| gpu info | CUDA version | `cu128` (CUDA 12.8) |
| functionality | Feature mode | `sft_llamafactory-v0.9.3` or `infer` |
| timestamp | Build time (Beijing/UTC+8) | `20241212143022` |

## Files

| File | Purpose |
|------|---------|
| `Dockerfile` | Main Dockerfile with multi-CPU build and version extraction |
| `docker-utils.sh` | Shared utility functions for both scripts |
| `build-docker-tar.sh` | Build and export Docker image to tar file |
| `push-to-dockerhub.sh` | Build and push Docker image to DockerHub |

## Prerequisites

- Docker installed and running
- For DockerHub push: Docker Hub account and login (`docker login`)
- Sufficient disk space (at least 20GB recommended)
- Internet access (or local mirrors configured)

## Quick Start

### Build Local Tar File

```bash
cd docker

# Basic build
./build-docker-tar.sh

# With specific CUDA version and mirror
./build-docker-tar.sh \
  --cuda-version 12.8.1 \
  --ubuntu-mirror 1

# With proxy (for China mainland)
./build-docker-tar.sh \
  --cuda-version 12.8.1 \
  --ubuntu-mirror 1 \
  --http-proxy "http://127.0.0.1:16981" \
  --https-proxy "http://127.0.0.1:16981" \
  --output-dir /path/to/output
```

### Push to DockerHub

```bash
cd docker

# Basic push (requires --repository)
./push-to-dockerhub.sh \
  --repository kvcache/ktransformers

# With simplified tag
./push-to-dockerhub.sh \
  --cuda-version 12.8.1 \
  --repository kvcache/ktransformers \
  --also-push-simplified

# Skip build if image exists
./push-to-dockerhub.sh \
  --repository kvcache/ktransformers \
  --skip-build
```

## Script Options

### build-docker-tar.sh

```
Build Configuration:
  --cuda-version VERSION       CUDA version (default: 12.8.1)
  --ubuntu-mirror 0|1         Use Tsinghua mirror (default: 0)
  --http-proxy URL            HTTP proxy URL
  --https-proxy URL           HTTPS proxy URL
  --cpu-variant VARIANT       CPU variant (default: x86-intel-multi)
  --functionality TYPE        Mode: sft or infer (default: sft)

Paths:
  --dockerfile PATH           Path to Dockerfile (default: ./Dockerfile)
  --context-dir PATH          Build context directory (default: .)
  --output-dir PATH           Output directory for tar (default: .)

Options:
  --dry-run                   Preview without building
  --keep-image                Keep Docker image after export
  --build-arg KEY=VALUE       Additional build arguments
  -h, --help                  Show help message
```

### push-to-dockerhub.sh

```
All options from build-docker-tar.sh, plus:

Registry Settings:
  --registry REGISTRY         Docker registry (default: docker.io)
  --repository REPO           Repository name (REQUIRED)

Options:
  --skip-build                Skip build if image exists
  --also-push-simplified      Also push simplified tag
  --max-retries N             Max push retries (default: 3)
  --retry-delay SECONDS       Delay between retries (default: 5)
```

## Usage Examples

### Example 1: Local Development Build

For testing on your local machine:

```bash
./build-docker-tar.sh \
  --cuda-version 12.8.1 \
  --output-dir ./builds \
  --keep-image
```

This will:
1. Build the Docker image
2. Export to tar in `./builds/` directory
3. Keep the Docker image for local testing

### Example 2: Production Build for Distribution

For creating a production build with mirrors and proxy:

```bash
./build-docker-tar.sh \
  --cuda-version 12.8.1 \
  --ubuntu-mirror 1 \
  --http-proxy "http://127.0.0.1:16981" \
  --https-proxy "http://127.0.0.1:16981" \
  --output-dir /mnt/data/releases
```

### Example 3: Publish to DockerHub

For publishing to DockerHub:

```bash
# First, login to Docker Hub
docker login

# Then push
./push-to-dockerhub.sh \
  --cuda-version 12.8.1 \
  --repository kvcache/ktransformers \
  --also-push-simplified
```

This creates two tags:
- Full: `kvcache/ktransformers:sglang-v0.5.6_ktransformers-v0.4.3_x86-intel-multi_cu128_sft_llamafactory-v0.9.3_20241212143022`
- Simplified: `kvcache/ktransformers:v0.4.3-cu128`

### Example 4: Dry Run

Preview the build without actually building:

```bash
./build-docker-tar.sh --cuda-version 12.8.1 --dry-run
```

### Example 5: Custom Build Arguments

Pass additional Docker build arguments:

```bash
./build-docker-tar.sh \
  --cuda-version 12.8.1 \
  --build-arg SGL_VERSION=0.5.7 \
  --build-arg FLASHINFER_VERSION=0.5.4
```

## Using the Built Images

### Load from Tar File

```bash
# Load the image
docker load -i sglang-v0.5.6_ktransformers-v0.4.3_x86-intel-multi_cu128_sft_llamafactory-v0.9.3_20241212143022.tar

# Run the container
docker run -it --rm \
  --gpus all \
  sglang-v0.5.6_ktransformers-v0.4.3_x86-intel-multi_cu128_sft_llamafactory-v0.9.3_20241212143022 \
  /bin/bash
```

### Pull from DockerHub

```bash
# Pull with full tag
docker pull kvcache/ktransformers:sglang-v0.5.6_ktransformers-v0.4.3_x86-intel-multi_cu128_sft_llamafactory-v0.9.3_20241212143022

# Or pull with simplified tag
docker pull kvcache/ktransformers:v0.4.3-cu128

# Run the container
docker run -it --rm \
  --gpus all \
  kvcache/ktransformers:v0.4.3-cu128 \
  /bin/bash
```

### Inside the Container

The image contains two conda environments:

```bash
# Activate serve environment (for inference with sglang)
conda activate serve
# or use the alias:
serve

# Activate fine-tune environment (for training with LLaMA-Factory)
conda activate fine-tune
# or use the alias:
finetune
```

## Multi-CPU Variant Support

The Docker image includes all three CPU variants:
- **AMX** - For Intel Sapphire Rapids and newer (4th Gen Xeon+)
- **AVX512** - For Intel Skylake-X, Ice Lake, Cascade Lake
- **AVX2** - Maximum compatibility for older CPUs

The runtime automatically detects your CPU and loads the appropriate variant. To override:

```bash
# Force use of AVX2 variant
export KT_KERNEL_CPU_VARIANT=avx2
python your_script.py

# Enable debug output to see which variant is loaded
export KT_KERNEL_DEBUG=1
python your_script.py
```

## Version Extraction

Versions are automatically extracted during Docker build from:

- **SGLang**: From `sglang.__version__` in serve environment
- **KTransformers**: From `version.py` in ktransformers repository
- **LLaMA-Factory**: From `llamafactory.__version__` in fine-tune environment

The versions are saved to `/workspace/versions.env` in the image:

```bash
# View versions in running container
cat /workspace/versions.env

# Output:
SGLANG_VERSION=0.5.6
KTRANSFORMERS_VERSION=0.4.3
LLAMAFACTORY_VERSION=0.9.3
```

## Troubleshooting

### Build Fails with Out of Disk Space

Check available disk space:
```bash
df -h
```

The build requires approximately 15-20GB of disk space. Clean up Docker:
```bash
docker system prune -a
```

### Version Extraction Fails

If version extraction fails (shows "unknown"), check:

1. The cloned repositories have the correct branches
2. Python packages are properly installed in conda environments
3. Version files exist in expected locations

You can manually verify by running:
```bash
docker run --rm <image> /bin/bash -c "
  source /opt/miniconda3/etc/profile.d/conda.sh &&
  conda activate serve &&
  python -c 'import sglang; print(sglang.__version__)'
"
```

### Push to DockerHub Fails

1. **Check login**: `docker login`
2. **Check repository name**: Must include namespace (e.g., `kvcache/ktransformers`, not just `ktransformers`)
3. **Network issues**: Use `--max-retries` and `--retry-delay` options
4. **Rate limiting**: DockerHub has pull/push rate limits for free accounts

### Proxy Configuration Issues

For builds in China mainland, use Tsinghua mirrors:

```bash
./build-docker-tar.sh \
  --ubuntu-mirror 1 \
  --http-proxy "http://127.0.0.1:16981" \
  --https-proxy "http://127.0.0.1:16981"
```

Verify proxy is working:
```bash
curl -x http://127.0.0.1:16981 https://www.google.com
```

## Advanced Topics

### Custom Dockerfile Location

```bash
./build-docker-tar.sh \
  --dockerfile /path/to/custom/Dockerfile \
  --context-dir /path/to/build/context
```

### Building Only Inference Image (Future)

Currently, the image always includes both serve and fine-tune environments. To create an inference-only image, modify the Dockerfile to skip the fine-tune environment section.

### Customizing CPU Variants

To build only specific CPU variants, modify `kt-kernel/install.sh` or set environment variables in the Dockerfile.

### CI/CD Integration

The scripts are designed for manual execution but can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Build and push Docker image
  run: |
    cd docker
    ./push-to-dockerhub.sh \
      --cuda-version ${{ matrix.cuda_version }} \
      --repository ${{ secrets.DOCKER_REPOSITORY }} \
      --also-push-simplified
```

## Support

For issues and questions:
- File an issue at: https://github.com/kvcache-ai/ktransformers/issues
- Check documentation: https://github.com/kvcache-ai/ktransformers

## License

This packaging system is part of KTransformers and follows the same license.
