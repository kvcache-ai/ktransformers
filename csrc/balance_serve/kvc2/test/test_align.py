import ctypes
import torch

def aligned_tensor(size, alignment=4096):
    num_bytes = size 
    mem = ctypes.c_void_p()
    error_code = ctypes.CDLL(None).posix_memalign(
        ctypes.byref(mem), ctypes.c_size_t(alignment), ctypes.c_size_t(num_bytes)
    )

    if error_code != 0:
        raise MemoryError(f"posix_memalign failed with error code {error_code}")

    array_type = (ctypes.c_int8 * size) 
    raw_array = array_type.from_address(mem.value)

    tensor = torch.frombuffer(raw_array, dtype=torch.int8)

    if tensor.data_ptr() % alignment != 0:
        raise ValueError(f"Tensor data_ptr {tensor.data_ptr()} is not aligned to {alignment} bytes")

    return tensor, mem


size = 5124380
tensor, mem_ptr = aligned_tensor(size, alignment=4096)

print(f"Tensor: {tensor}, size: {tensor.size()}, dataptr: {tensor.data_ptr()}")
print(f"Tensor memory alignment: {tensor.data_ptr() % 4096 == 0}")
print(f"Allocated memory address: {mem_ptr.value}")

ctypes.CDLL(None).free(mem_ptr)
