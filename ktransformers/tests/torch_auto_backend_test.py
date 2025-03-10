import torch
from ktransformers.util.torch_auto_backend import CUDA, CUDA0

if __name__ == "__main__":
    print(CUDA, CUDA0)
    a = torch.tensor([1.2, 2.3], dtype=torch.float32, device=CUDA)
    print(a)
    b = torch.tensor([1.2, 2.3], dtype=torch.float32, device=CUDA0)
    print(b)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.device(0))
    print(b.cuda())
    print(torch.cuda.current_stream(CUDA).cuda_stream)
