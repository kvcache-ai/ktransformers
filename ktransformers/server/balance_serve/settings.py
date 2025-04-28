'''
Date: 2024-11-13 09:43:39
LastEditors: djw
LastEditTime: 2024-11-18 16:41:03
'''
import sys, os
import yaml, json
from time import sleep


import sched_ext
from transformers import AutoConfig

from ktransformers.models.configuration_qwen3_moe import Qwen3MoeConfig

def create_sched_settings(args):
    default_sample_options = sched_ext.SampleOptions()
    model_name = os.path.basename(os.path.normpath(args.model_dir))
    input_model_settings = sched_ext.ModelSettings()
    input_model_settings.model_path = args.model_dir
    input_model_settings.params_count = int(0)
    model_config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    input_model_settings.layer_count = model_config.num_hidden_layers
    input_model_settings.num_k_heads = 1 # model_config["num_key_value_heads"]
    input_model_settings.k_head_dim = 576
    input_model_settings.bytes_per_params = 2
    input_model_settings.bytes_per_kv_cache_element = 2
    settings = sched_ext.Settings()
    settings.model_name = model_name
    settings.quant_type = "BF16"
    settings.model_settings = input_model_settings
    settings.page_size = args.page_size
    settings.gpu_device_count = 1 # tp
    settings.gpu_device_id = [i for i in range(settings.gpu_device_count)]
    # settings.gpu_memory_size = args.cache_lens*576*2
    settings.gpu_memory_size = args.gpu_memory_size
    settings.memory_utilization_percentage = args.utilization_percentage
    max_batch_size = args.max_batch_size
    chunk_size = args.chunk_size

    max_decode_batch_size = max_batch_size - 2

    settings.max_batch_size = max_batch_size
    settings.recommended_chunk_prefill_token_count = (chunk_size - max_decode_batch_size) // 2
    settings.sample_options = default_sample_options
    settings.sched_metrics_port = args.sched_metrics_port
    settings.gpu_only = args.memory_gpu_only
    settings.use_self_defined_head_dim = True
    settings.self_defined_head_dim = 576
    settings.full_kv_cache_on_each_gpu = True
    settings.k_cache_on = True
    settings.v_cache_on = False

    settings.kvc2_root_path = '/mnt/data/persist-kvc'
    settings.kvc2_config_path = args.kvc2_config_dir
    settings.memory_pool_size_GB = args.cpu_memory_size_GB
    settings.evict_count = 40
    settings.kvc2_metrics_port = args.kvc2_metrics_port
    settings.load_from_disk = False
    settings.save_to_disk = True


    settings.strategy_name = args.sched_strategy

    settings.auto_derive()
    return settings


def create_sched_settings_qwen2moe(args):
    default_sample_options = sched_ext.SampleOptions()
    model_name = os.path.basename(os.path.normpath(args.model_dir))
    input_model_settings = sched_ext.ModelSettings()
    input_model_settings.model_path = args.model_dir
    input_model_settings.params_count = int(0)
    model_config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    input_model_settings.layer_count = model_config.num_hidden_layers
    input_model_settings.num_k_heads = model_config.num_key_value_heads # model_config["num_key_value_heads"]
    input_model_settings.k_head_dim = 128
    input_model_settings.bytes_per_params = 2
    input_model_settings.bytes_per_kv_cache_element = 2
    settings = sched_ext.Settings()
    settings.model_name = model_name
    settings.quant_type = "BF16"
    settings.model_settings = input_model_settings
    settings.page_size = args.page_size
    settings.gpu_device_count = 1 # tp
    settings.gpu_device_id = [i for i in range(settings.gpu_device_count)]
    # settings.gpu_memory_size = args.cache_lens*576*2
    settings.gpu_memory_size = args.gpu_memory_size
    settings.memory_utilization_percentage = args.utilization_percentage
    max_batch_size = args.max_batch_size
    chunk_size = args.chunk_size

    max_decode_batch_size = max_batch_size - 2

    settings.max_batch_size = max_batch_size
    settings.recommended_chunk_prefill_token_count = (chunk_size - max_decode_batch_size) // 2
    settings.sample_options = default_sample_options
    settings.sched_metrics_port = args.sched_metrics_port
    settings.gpu_only = args.memory_gpu_only
    settings.use_self_defined_head_dim = False
    settings.self_defined_head_dim = 576
    settings.full_kv_cache_on_each_gpu = True
    settings.k_cache_on = True
    settings.v_cache_on = True

    settings.kvc2_root_path = '/mnt/data/persist-kvc'
    settings.kvc2_config_path = args.kvc2_config_dir
    settings.memory_pool_size_GB = args.cpu_memory_size_GB
    settings.evict_count = 40
    settings.kvc2_metrics_port = args.kvc2_metrics_port
    settings.load_from_disk = False
    settings.save_to_disk = True


    settings.strategy_name = args.sched_strategy

    settings.auto_derive()
    return settings



def create_sched_settings_qwen3moe(args):
    default_sample_options = sched_ext.SampleOptions()
    model_name = os.path.basename(os.path.normpath(args.model_dir))
    input_model_settings = sched_ext.ModelSettings()
    input_model_settings.model_path = args.model_dir
    input_model_settings.params_count = int(0)
    model_config = Qwen3MoeConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    input_model_settings.layer_count = model_config.num_hidden_layers
    input_model_settings.num_k_heads = model_config.num_key_value_heads # model_config["num_key_value_heads"]
    input_model_settings.k_head_dim = 128
    input_model_settings.bytes_per_params = 2
    input_model_settings.bytes_per_kv_cache_element = 2
    settings = sched_ext.Settings()
    settings.model_name = model_name
    settings.quant_type = "BF16"
    settings.model_settings = input_model_settings
    settings.page_size = args.page_size
    settings.gpu_device_count = 1 # tp
    settings.gpu_device_id = [i for i in range(settings.gpu_device_count)]
    # settings.gpu_memory_size = args.cache_lens*576*2
    settings.gpu_memory_size = args.gpu_memory_size
    settings.memory_utilization_percentage = args.utilization_percentage
    max_batch_size = args.max_batch_size
    chunk_size = args.chunk_size

    max_decode_batch_size = max_batch_size - 2

    settings.max_batch_size = max_batch_size
    settings.recommended_chunk_prefill_token_count = (chunk_size - max_decode_batch_size) // 2
    settings.sample_options = default_sample_options
    settings.sched_metrics_port = args.sched_metrics_port
    settings.gpu_only = args.memory_gpu_only
    settings.use_self_defined_head_dim = False
    settings.self_defined_head_dim = 576
    settings.full_kv_cache_on_each_gpu = True
    settings.k_cache_on = True
    settings.v_cache_on = True

    settings.kvc2_root_path = '/mnt/data/persist-kvc'
    settings.kvc2_config_path = args.kvc2_config_dir
    settings.memory_pool_size_GB = args.cpu_memory_size_GB
    settings.evict_count = 40
    settings.kvc2_metrics_port = args.kvc2_metrics_port
    settings.load_from_disk = False
    settings.save_to_disk = True


    settings.strategy_name = args.sched_strategy

    settings.auto_derive()
    return settings






