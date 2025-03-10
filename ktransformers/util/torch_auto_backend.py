import sys
import torch
from torch.utils.cpp_extension import CUDA_HOME
try:
    from torch_musa.utils.musa_extension import MUSA_HOME
except ImportError:
    MUSA_HOME=None

if CUDA_HOME is not None:
    CUDA = "cuda"
elif MUSA_HOME is not None:
    CUDA = "musa"

    torch.cuda = torch.musa
    torch.cuda.CUDAGraph = torch.musa.MUSAGraph

    # **Monkey Patch `torch.Tensor.cuda()`**
    def tensor_cuda(self, device=None, non_blocking=False, memory_format=None):
        if device is None:
            device = CUDA
        elif isinstance(device, int):
            device = f"{CUDA}:{device}"
        return self.to(device, non_blocking=non_blocking, memory_format=memory_format)

    torch.Tensor.cuda = tensor_cuda

    # **Monkey Patch `torch.cuda.current_stream`**
    original_musa_current_stream = torch.musa.current_stream

    def patch_stream_object(stream):
        if not hasattr(stream, "cuda_stream"):
            stream.cuda_stream = stream.musa_stream
        return stream

    def patched_current_stream(device=None):
        return patch_stream_object(original_musa_current_stream(device))

    torch.cuda.current_stream = patched_current_stream

else:
    raise ValueError("Unsupported platform: {}".format(sys.platform))

CUDA0 = f"{CUDA}:0"
CUDA1 = f"{CUDA}:1"
CUDA2 = f"{CUDA}:2"

print(f"Torch backend loaded: CUDA={CUDA}, CUDA0={CUDA0}")
