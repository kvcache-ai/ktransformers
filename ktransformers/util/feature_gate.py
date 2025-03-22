import os
from ktransformers.operators.flashinfer_wrapper import flashinfer_enabled
from ktransformers.util.utils import get_compute_capability
from ktransformers.util.vendors import device_manager, GPUVendor

# Feature gate default values
KTRANSFORMERS_USE_TORCH_NATIVE = False
KTRANSFORMERS_USE_FLASHINFER = False

if os.name == "nt" or get_compute_capability() < 8 or device_manager.gpu_vendor != GPUVendor.NVIDIA:
    print("Using torch native for Windows or Nvidia GPUs before Ampere.")
    KTRANSFORMERS_USE_TORCH_NATIVE = True

if not KTRANSFORMERS_USE_TORCH_NATIVE and flashinfer_enabled:
    print("Using FlashInfer for Nvidia GPUs after Ampere.")
    KTRANSFORMERS_USE_FLASHINFER = True

print(
    f"Feature gate initialized: KTRANSFORMERS_USE_TORCH_NATIVE={KTRANSFORMERS_USE_TORCH_NATIVE},"
    f" KTRANSFORMERS_USE_FLASHINFER={KTRANSFORMERS_USE_FLASHINFER}"
)
