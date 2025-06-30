## Enabling Prefix Cache Mode in KTransformers

To enable **Prefix Cache Mode** in KTransformers, you need to modify the configuration file and recompile the project.

### Step 1: Modify the Configuration File

Edit the `./ktransformers/configs/config.yaml` file with the following content (you can adjust the values according to your needs):

```yaml
attn:
  page_size: 16 # Size of a page in KV Cache.
  chunk_size: 256
kvc2:
  gpu_only: false # Set to false to enable prefix cache mode (Disk + CPU + GPU KV storage)
  utilization_percentage: 1.0
  cpu_memory_size_GB: 500 # Amount of CPU memory allocated for KV Cache
```

### Step 2: Update Submodules and Recompile

If this is your first time using prefix cache mode, please update the submodules first:

```bash
git submodule update --init --recursive # Update PhotonLibOS submodule
```

Then recompile the project:

```bash
# Install single NUMA dependencies
USE_BALANCE_SERVE=1  bash ./install.sh
# For those who have two cpu and 1T RAM（Dual NUMA）:
USE_BALANCE_SERVE=1 USE_NUMA=1 bash ./install.sh
```