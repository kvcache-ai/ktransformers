import sys
sys.path.append('./build')
sys.path.append('./src')
import torch
import kvc2_ext
from kvc2_utils import alloc_aligned_cache,dealloc_aligned_cache,get_tensor_ptr,get_tensor_from_data_ptr

# Create a kvc2 instance
path = "/mnt/data/kvc2"
kvc2_instance = kvc2_ext.create_kvc2(path,int(10e9)) # 10 G memory pool

# Start IO thread
print("Start IO thread")
kvc2_ext.start_io_thread(kvc2_instance)
print("IO thread started")

# Create CacheInfoInput
test_info = kvc2_ext.CacheInfoInput()
test_info.model_type = kvc2_ext.ModelType.MT_DeepseekV2
test_info.cache_type = kvc2_ext.CacheType.CT_KeyCache
test_info.quant_type = kvc2_ext.QuantType.QT_F32

print("Element size: ", test_info.element_size())

# Generate random test IDs (length = 2560)
torch.manual_seed(123)
length = 2560
test_id = torch.randint(0, 65536, (length,), dtype=torch.uint16).contiguous()
block_count = (length+255) // 256
# print("Test ID: ", test_id)

# Generate test data based on element size and hidden layer count
element_size = test_info.element_size()
hidden_layer_count = test_info.hidden_layer_count()

write_data,write_data_mem = alloc_aligned_cache(hidden_layer_count,block_count,element_size)
# print(test_data,test_data_mem)
print('Generate Insert Data')
for layer in write_data:
    for data in layer:
        random_values = torch.randint(0, 256, (element_size,), dtype=torch.uint8)
        data.copy_(random_values)

print('Insert New data')
# Insert raw data
kvc2_ext.raw_insert(kvc2_instance, test_info, test_id.data_ptr(), length, get_tensor_ptr(write_data))


def read_cmp_and_release(kvc2_instance,cache_info,ids,length):
    handle = kvc2_ext.lookup(kvc2_instance, cache_info, ids, length)
    if kvc2_ext.is_nullptr(handle):
        print("Handle is nullptr.")
        exit()
    matched_length = kvc2_ext.matched_length(handle)
    matched_data = kvc2_ext.handle_data(handle)
    print('Matched length: ', matched_length)
    if matched_length >0:
        print(f'First layer address {[hex(x) for x in matched_data[0]]}')
    read_data = get_tensor_from_data_ptr(matched_data,element_size)
    
    for layer_w,layer_r in zip(write_data,read_data):
        for data_w,data_r in zip(layer_w,layer_r):
            # print(data_w,data_r)
            assert torch.equal(data_w,data_r)
    print("Lookup read check ok.")
    kvc2_ext.release(handle)


l = 128
while l<=length:
    read_cmp_and_release(kvc2_instance,test_info,test_id.data_ptr(),l)
    l+=128



dealloc_aligned_cache(write_data_mem)


kvc2_ext.save(kvc2_instance)
kvc2_ext.destroy_kvc2(kvc2_instance)


print("Test completed successfully.")
