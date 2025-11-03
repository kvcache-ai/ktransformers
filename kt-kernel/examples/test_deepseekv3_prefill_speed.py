import os, sys
import time
os.environ["BLAS_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(__file__) + "/../build")
import kt_kernel_ext
from kt_kernel_ext.kvcache import ggml_type
import torch
import logging
import sys
import json
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    GenerationConfig,
    TextStreamer,
)

logger = logging.getLogger("reader")

from gguf.gguf_reader import GGUFReader
# load_layers = 3
load_layers = None
worker_config = kt_kernel_ext.WorkerPoolConfig()
worker_config.subpool_count = 2
worker_config.subpool_numa_map= [0,1]
worker_config.subpool_thread_count = [72,72]
CPUInfer = kt_kernel_ext.CPUInfer(worker_config)

max_qlen = 4096
max_kvlen = 4096
page_size = 256
pages_count = 200


def read_gguf_file(gguf_file_path):
    """
    Reads and prints key-value pairs and tensor information from a GGUF file in an improved format.

    Parameters:
    - gguf_file_path: Path to the GGUF file.
    """

    reader = GGUFReader(gguf_file_path)

    # List all key-value pairs in a columnized format
    # print("Key-Value Pairs:") # noqa: NP100
    # max_key_length = max(len(key) for key in reader.fields.keys())
    for key, field in reader.fields.items():
        value = field.parts[field.data[0]]
        # print(f"{key:{max_key_length}} : {value}") # noqa: NP100
    # print("----") # noqa: NP100

    # List all tensors
    # print("Tensors:") # noqa: NP100
    # tensor_info_format = "{:<30} | Shape: {:<15} | Size: {:<12} | Quantization: {}"
    # print(tensor_info_format.format("Tensor Name", "Shape", "Size", "Quantization")) # noqa: NP100
    # print("-" * 80) # noqa: NP100
    re = []
    for tensor in reader.tensors:
        shape_str = "x".join(map(str, tensor.shape))
        size_str = str(tensor.n_elements)
        quantization_str = tensor.tensor_type.name
        # print(tensor_info_format.format(tensor.name, shape_str, size_str, quantization_str)) # noqa: NP100
        re.append(tensor)
    return re


def read_gguf_directory(directory):
    """
    Reads all GGUF files in a directory and prints their contents.

    Parameters:
    - directory: Path to the directory containing GGUF files.
    """
    if not os.path.isdir(directory):
        logger.error(f"Directory {directory} does not exist.")
        return

    # List all GGUF files in the directory
    files = [f for f in os.listdir(directory) if f.endswith(".gguf")]
    if not files:
        logger.info(f"No GGUF files found in {directory}.")
        return

    re = []
    for file in files:
        file_path = os.path.join(directory, file)
        # print(f"Reading {file_path}:") # noqa: NP100
        # print("\n") # noqa: NP100
        re.extend(read_gguf_file(file_path))
    re = {r.name: r for r in re}
    return re


def find_weights(name, weights):
    """
    Finds and returns the weights for a given name from the list of weights.

    Parameters:
    - name: The name of the weights to find.
    - weights: List of weight tensors.

    Returns:
    - The weight tensor if found, otherwise None.
    """
    for weight in weights:
        if weight.name == name:
            return weight
    raise ValueError(f"Weight with name {name} not found in the provided weights list.")


def get_torch_tensor_from_gguf(gguf_weights, name):
    return torch.from_numpy(gguf_weights[name].data).contiguous()


def get_torch_tensor_and_type_from_gguf(gguf_weights, name):
    return torch.from_numpy(gguf_weights[name].data).contiguous(), gguf_weights[name].tensor_type.name


def type_to_ggml_type(type):
    if type == "F32":
        return ggml_type.FP32
    elif type == "F16":
        return ggml_type.FP16
    elif type == "BF16":
        return ggml_type.BF16
    else:
        raise ValueError(f"Unsupported data type: {type}")


def build_mla(layer_idx, json_config, gguf_weights):
    hidden_size = json_config["hidden_size"]
    num_heads = json_config["num_attention_heads"]
    q_lora_rank = json_config["q_lora_rank"]
    kv_lora_rank = json_config["kv_lora_rank"]
    nope_size = json_config["qk_nope_head_dim"]
    rope_size = json_config["qk_rope_head_dim"]
    max_position_embeddings = json_config["max_position_embeddings"]
    rope_theta = json_config["rope_theta"]
    rope_scaling = json_config["rope_scaling"]

    config = kt_kernel_ext.mla.MLAConfig(
        hidden_size,
        q_lora_rank,
        kv_lora_rank,
        num_heads,
        nope_size,
        rope_size,
    )
    config.max_qlen = max_qlen
    config.max_kvlen = max_kvlen
    config.max_position_embeddings = max_position_embeddings
    config.rope_scaling_factor = rope_scaling["factor"]
    config.rope_theta = rope_theta
    config.rope_scaling_beta_fast = rope_scaling["beta_fast"]
    config.rope_scaling_beta_slow = rope_scaling["beta_slow"]
    config.rope_scaling_mscale = rope_scaling["mscale"]
    config.rope_scaling_mscale_all_dim = rope_scaling["mscale_all_dim"]
    config.rope_scaling_original_max_position_embeddings = rope_scaling["original_max_position_embeddings"]

    q_a_proj_weight, type = get_torch_tensor_and_type_from_gguf(gguf_weights, f"blk.{layer_idx}.attn_q_a.weight")
    config.q_a_proj = q_a_proj_weight.data_ptr()
    config.q_a_proj_type = type_to_ggml_type(type)
    q_a_type = type

    q_a_norm_weight, type = get_torch_tensor_and_type_from_gguf(gguf_weights, f"blk.{layer_idx}.attn_q_a_norm.weight")
    config.q_a_norm = q_a_norm_weight.data_ptr()
    config.q_a_norm_type = type_to_ggml_type(type)

    q_b_proj_weight, type = get_torch_tensor_and_type_from_gguf(gguf_weights, f"blk.{layer_idx}.attn_q_b.weight")
    config.q_b_proj = q_b_proj_weight.data_ptr()
    config.q_b_proj_type = type_to_ggml_type(type)

    kv_a_proj_with_mqa_weight, type = get_torch_tensor_and_type_from_gguf(
        gguf_weights, f"blk.{layer_idx}.attn_kv_a_mqa.weight"
    )
    config.kv_a_proj_with_mqa = kv_a_proj_with_mqa_weight.data_ptr()
    config.kv_a_proj_with_mqa_type = type_to_ggml_type(type)

    kv_a_norm_weight, type = get_torch_tensor_and_type_from_gguf(gguf_weights, f"blk.{layer_idx}.attn_kv_a_norm.weight")
    config.kv_a_norm = kv_a_norm_weight.data_ptr()
    config.kv_a_norm_type = type_to_ggml_type(type)

    kv_b_proj_weight, type = get_torch_tensor_and_type_from_gguf(gguf_weights, f"blk.{layer_idx}.attn_kv_b.weight")
    config.kv_b_proj = kv_b_proj_weight.data_ptr()
    config.kv_b_proj_type = type_to_ggml_type(type)

    o_proj_weight, type = get_torch_tensor_and_type_from_gguf(gguf_weights, f"blk.{layer_idx}.attn_output.weight")
    config.o_proj = o_proj_weight.data_ptr()
    config.w_o_type = type_to_ggml_type(type)

    config.layer_idx = layer_idx
    config.pool = CPUInfer.backend_
    config.page_count = pages_count

    if q_a_type == "F32":
        mla = kt_kernel_ext.mla.MLA_F32(config)
    elif q_a_type == "F16":
        mla = kt_kernel_ext.mla.MLA_F16(config)
    elif q_a_type == "BF16":
        # mla = kt_kernel_ext.mla.MLA_F32(config)
        mla = kt_kernel_ext.mla.MLA_QUAN_F32(config)
    else:
        raise ValueError(f"Unsupported data type: {q_a_type}")

    mla.load_weights()
    mla.set_local_pages(pages_count)
    return mla


def build_ffn(layer_idx, json_config, gguf_weights):
    if f"blk.{layer_idx}.ffn_gate.weight" in gguf_weights:  # dense
        config = kt_kernel_ext.moe.MOEConfig(
            json_config["num_experts_per_tok"] + json_config["n_shared_experts"],
            json_config["num_experts_per_tok"] + json_config["n_shared_experts"],
            json_config["hidden_size"],
            json_config["moe_intermediate_size"],
        )
        config.layer_idx = layer_idx
        config.max_len = max_qlen
        config.pool = CPUInfer.backend_
        gate, gate_type = get_torch_tensor_and_type_from_gguf(gguf_weights, f"blk.{layer_idx}.ffn_gate.weight")
        up, up_type = get_torch_tensor_and_type_from_gguf(gguf_weights, f"blk.{layer_idx}.ffn_up.weight")
        down, down_type = get_torch_tensor_and_type_from_gguf(gguf_weights, f"blk.{layer_idx}.ffn_down.weight")

        config.gate_proj = gate.data_ptr()
        config.gate_type = type_to_ggml_type(gate_type)
        config.up_proj = up.data_ptr()
        config.up_type = type_to_ggml_type(up_type)
        config.down_proj = down.data_ptr()
        config.down_type = type_to_ggml_type(down_type)

        moe = kt_kernel_ext.moe.KMLInt8_MOE(config)
        moe.load_weights()
        return moe

    elif f"blk.{layer_idx}.ffn_gate_exps.weight" in gguf_weights:
        config = kt_kernel_ext.moe.MOEConfig(
            json_config["n_routed_experts"] + json_config["n_shared_experts"],
            json_config["num_experts_per_tok"] + json_config["n_shared_experts"],
            json_config["hidden_size"],
            json_config["moe_intermediate_size"],
        )
        config.layer_idx = layer_idx
        config.max_len = max_qlen
        config.pool = CPUInfer.backend_
        gate, gate_type = get_torch_tensor_and_type_from_gguf(gguf_weights, f"blk.{layer_idx}.ffn_gate_exps.weight")
        up, up_type = get_torch_tensor_and_type_from_gguf(gguf_weights, f"blk.{layer_idx}.ffn_up_exps.weight")
        down, down_type = get_torch_tensor_and_type_from_gguf(gguf_weights, f"blk.{layer_idx}.ffn_down_exps.weight")

        gate_sh, gate_sh_type = get_torch_tensor_and_type_from_gguf(
            gguf_weights, f"blk.{layer_idx}.ffn_gate_shexp.weight"
        )
        up_sh, up_sh_type = get_torch_tensor_and_type_from_gguf(gguf_weights, f"blk.{layer_idx}.ffn_up_shexp.weight")
        down_sh, down_sh_type = get_torch_tensor_and_type_from_gguf(
            gguf_weights, f"blk.{layer_idx}.ffn_down_shexp.weight"
        )

        gate_sh_expanded = gate_sh.unsqueeze(0)
        gate = torch.cat([gate, gate_sh_expanded], dim=0).contiguous()
        up_sh_expanded = up_sh.unsqueeze(0)
        up = torch.cat([up, up_sh_expanded], dim=0).contiguous()
        down_sh_expanded = down_sh.unsqueeze(0)
        down = torch.cat([down, down_sh_expanded], dim=0).contiguous()

        config.gate_proj = gate.data_ptr()
        config.gate_type = type_to_ggml_type(gate_type)
        config.up_proj = up.data_ptr()
        config.up_type = type_to_ggml_type(up_type)
        config.down_proj = down.data_ptr()
        config.down_type = type_to_ggml_type(down_type)

        moe = kt_kernel_ext.moe.KMLInt8_MOE(config)
        moe.load_weights()
        return moe

    else:
        raise ValueError(f"Unsupported FFN type for layer {layer_idx}")


def build_moegate(layer_idx, json_config, gguf_weights):
    config = kt_kernel_ext.gate.GateConfig(
        json_config["hidden_size"],
        json_config["num_experts_per_tok"],
        json_config["n_routed_experts"],
        json_config["n_group"],
        json_config["topk_group"],
    )

    config.routed_scaling_factor = json_config['routed_scaling_factor']

    config.pool = CPUInfer.backend_

    weight,weight_type = get_torch_tensor_and_type_from_gguf(gguf_weights, f"blk.{layer_idx}.ffn_gate_inp.weight")
    config.weight = weight.data_ptr()
    config.weight_type = type_to_ggml_type(weight_type)

    bias,bias_type = get_torch_tensor_and_type_from_gguf(gguf_weights, f"blk.{layer_idx}.exp_probs_b.bias")
    config.e_score_correction_bias = bias.data_ptr()
    config.e_score_correction_bias_type = type_to_ggml_type(bias_type)

    gate = kt_kernel_ext.gate.MoEGate(config)
    
    return gate
   


def build_llm(json_config, gguf_weights):

    general_config = kt_kernel_ext.GeneralConfig()
    general_config.vocab_size = json_config["vocab_size"]
    general_config.hidden_size = json_config["hidden_size"]
    general_config.num_experts_per_tok = json_config["num_experts_per_tok"]
    general_config.n_routed_experts = json_config["n_routed_experts"]
    general_config.n_shared_experts = json_config["n_shared_experts"]
    general_config.max_qlen = max_qlen

    lm_heads,lm_heads_type = get_torch_tensor_and_type_from_gguf(gguf_weights, "output.weight")
    general_config.lm_heads_ptr = lm_heads.data_ptr()
    general_config.lm_heads_type = type_to_ggml_type(lm_heads_type)

    output_norm, output_norm_type = get_torch_tensor_and_type_from_gguf(gguf_weights, "output_norm.weight")
    general_config.norm_weights_ptr = output_norm.data_ptr()
    general_config.norm_weights_type = type_to_ggml_type(output_norm_type)    

    token_embd,token_embd_type = get_torch_tensor_and_type_from_gguf(weights, "token_embd.weight")
    general_config.token_embd_ptr = token_embd.data_ptr()
    general_config.token_embd_type = type_to_ggml_type(token_embd_type)

    general_config.pool = CPUInfer.backend_

    llm = kt_kernel_ext.DeepseekV3ForCausalLM(general_config)
    model = kt_kernel_ext.DeepseekV3Model(general_config)
    llm.model = model


    decoder_layers = []
    real_load_layers = json_config["num_hidden_layers"] if load_layers is None else load_layers

    for i in range(real_load_layers):
    # for i in [2,3]:
        layer = kt_kernel_ext.DeepseekV3DecoderLayer(general_config,i)
        attn_norm, attn_norm_type = get_torch_tensor_and_type_from_gguf(gguf_weights, f"blk.{i}.attn_norm.weight")
        ffn_norm, ffn_norm_type = get_torch_tensor_and_type_from_gguf(gguf_weights, f"blk.{i}.ffn_norm.weight")

        layer.load_norm(
            attn_norm.data_ptr(),
            type_to_ggml_type(attn_norm_type),
            ffn_norm.data_ptr(),
            type_to_ggml_type(ffn_norm_type),
        )
        layer.self_attn = build_mla(i, json_config, gguf_weights)
        if f"blk.{i}.ffn_gate_inp.weight" in gguf_weights:
            layer.gate = build_moegate(i, json_config, gguf_weights)
        layer.ffn = build_ffn(i, json_config, gguf_weights)
        decoder_layers.append(layer)

    model.layers = decoder_layers 
    return llm


safetensor_path = '/home/bd/models/DeepSeek-R1'
json_path = os.path.join(safetensor_path, "config.json")
json_config = json.load(open(json_path, "r"))
print(json_config)

gguf_path = "/home/bd/models/DeepSeek-R1-BF16"
weights = read_gguf_directory(gguf_path)
weights = dict(sorted(weights.items()))


# for name, t in weights.items():
    # if not name.startswith("blk"):
    # if name.startswith("blk.10."):
        # if "ffn_gate." in name:
        # print(f"Found weight: {t.name}, Shape: {t.shape}, Type: {t.tensor_type.name}, Size: {t.n_elements}")
    # print(f"Found weight: {t.name}, Shape: {t.shape}, Type: {t.tensor_type.name}, Size: {t.n_elements}")
    
print("Building LLM ...") 
load_start_time = time.perf_counter()
llm = build_llm(json_config, weights)
load_end_time = time.perf_counter()
print(f"Load time: {load_end_time - load_start_time:.4f} seconds")

print("Release Weight Tensors ...")
weights = None
print("Loading Configs ...")


tokenizer = AutoTokenizer.from_pretrained(safetensor_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(safetensor_path, trust_remote_code=True)

force_think = False


output_logits = torch.zeros((max_qlen, json_config['vocab_size']), dtype=torch.float32)


def start_chat(content=None):
    if content is None:
        content = input("Chat: ")
    
    messages = [{"role": "user", "content": content}]
    input_tensor = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    if force_think:
        token_thinks = torch.tensor([tokenizer.encode("<think>\\n",add_special_tokens=False)],device=input_tensor.device)
        input_tensor = torch.cat(
            [input_tensor, token_thinks], dim=1
        )
    input_tensor = input_tensor.squeeze(0)  # Add batch dimension

    print(f"Input tensor: {input_tensor}, type {input_tensor.dtype}, shape {input_tensor.shape}")
    kvlen = 0
    step = 2
    while True or step > 0:
        step -= 1
        stream = TextStreamer(tokenizer)

        qlen = input_tensor.shape[0]
        qlens = [qlen]
        kvlens = [0]
        page_tables = [list(range(pages_count))]
        start_time = time.perf_counter()
        llm.forward(qlens,page_tables, kvlens, input_tensor.data_ptr(), output_logits.data_ptr())
        end_time = time.perf_counter()
        print(f"Forward time: {end_time - start_time:.4f} seconds, tps: {qlens[0] / (end_time - start_time)} tokens/sec")
        
        logits = output_logits[0]
        # print(logits)
        # sample 
        next_token = torch.argmax(logits).item()
        # print(f"Next token: {next_token}, {tokenizer.decode(next_token)}")
        # kvlen = input_tensor.shape[0]
        input_tensor = torch.cat((input_tensor, torch.tensor([next_token])), dim=-1)
                
        if next_token == tokenizer.eos_token_id or tokenizer.decode(next_token) == '<|im_end|>':
            stream.end()
            break
        else:
            stream.put(torch.tensor([next_token]))
job_id = 0
while True:
    try:
        # ---------- 让用户决定是否继续 ----------
        choice = input(
            "\n【回车】开始对话 | 输入 1 读取文件 | 输入 q/quit/exit 退出程序： "
        ).strip().lower()
        if choice in {"q", "quit", "exit"}:
            print("收到退出指令，程序结束。")
            break
        elif choice == "1":
            file_path = input("请输入要读取的文件路径：").strip()
            if not Path(file_path).is_file():
                print(f"文件 {file_path} 不存在，请检查路径。")
                continue
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            print(f"读取到内容：\n{content}\n")
            start_chat(content)
        else:
            start_chat()

    except KeyboardInterrupt:
        # 随时 Ctrl-C：放弃当前任务并重启
        print(f"\n检测到 Ctrl-C，已终止对话 #{job_id}，马上重启…")
    except Exception as e:
        # 其他异常：打印错误信息并重启
        print(f"\n发生错误：{e}\n已终止对话 #{job_id}，马上重启…")
        logger.error(f"Error in job {job_id}: {e}", exc_info=True)
    finally:
        job_id += 1                # 不管中断与否，都给下一任务换编号











