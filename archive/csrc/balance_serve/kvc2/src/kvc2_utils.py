import torch
import ctypes

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

def alloc_aligned_cache(layer_count,block_count,element_size):
    cache = []
    cache_mem = []
    for i in range(layer_count):
        layer_data = []
        layer_mem = []
        for j in range(block_count):
            tensor, mem_ptr = aligned_tensor(element_size, alignment=4096)
            layer_data.append(tensor)
            layer_mem.append(mem_ptr)
        cache.append(layer_data)
        cache_mem.append(layer_mem)
    return cache,cache_mem

def dealloc_aligned_cache(cache_mem):
    for layer_mem in cache_mem:
        for mem_ptr in layer_mem:
            ctypes.CDLL(None).free(mem_ptr)

def get_tensor_ptr(tensors):
    tensor_ptr = []
    for layer in tensors:
        layer_ptr = []
        for data in layer:
            layer_ptr.append(data.data_ptr())
        tensor_ptr.append(layer_ptr)
    return tensor_ptr

def get_tensor_from_data_ptr(matched_data,element_size):
    re = []
    for layer in matched_data:
        re_layer = []
        for data_ptr in layer:
            array_type = (ctypes.c_int8 * element_size) 
            raw_array = array_type.from_address(data_ptr)
            tensor = torch.frombuffer(raw_array, dtype=torch.int8)
            re_layer.append(tensor)
        re.append(re_layer)
    return re
if __name__ == "__main__":
    pass