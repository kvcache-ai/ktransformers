import os
from datetime import timedelta

import torch
import torch_npu
import torch.distributed as dist

_DATA_PARALLEL_SIZE = 0
_TENSOR_PARALLEL_SIZE = 0
_DATA_PARALLEL_GROUP = None
_TENSOR_PARALLEL_RANKS = None
_TENSOR_PARALLEL_GROUP = None
_DATA_PARALLEL_GROUP_GLOO = None
_DATA_PARALLEL_RANKS = None


def setup_model_parallel(distributed_timeout_minutes: int = 30, tp: int = 1):
    global _DATA_PARALLEL_SIZE
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_RANKS
    global _TENSOR_PARALLEL_SIZE
    global _TENSOR_PARALLEL_RANKS
    global _TENSOR_PARALLEL_GROUP

    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12345"
    local_rank = int(os.getenv("LOCAL_RANK", '0'))
    world_size = int(os.getenv("WORLD_SIZE", '1'))
    torch_npu.npu.set_device(local_rank)
    tp_size = tp
    dp_size = world_size // tp_size
    _DATA_PARALLEL_SIZE = dp_size
    _TENSOR_PARALLEL_SIZE = tp_size

    torch.set_num_threads(8)
    timeout = timedelta(minutes=distributed_timeout_minutes)
    print(f"start to init process group ------rank is {local_rank}, world_size is {world_size}")
    torch.distributed.init_process_group(
        backend='hccl',
        world_size=world_size, rank=local_rank
    )
    print(f"init process group success ------rank is {local_rank}, world_size is {world_size}")

    rank = torch.distributed.get_rank()
    nccl_comm_cfgs = {}
    # DP 组由每隔 tp_size 的进程组成
    for dp_group_id in range(tp_size):
        ranks = list(range(dp_group_id, world_size, tp_size))
        dp_group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('dp', nccl_comm_cfgs)
        )
        if rank in ranks:
            global _DATA_PARALLEL_GROUP
            _DATA_PARALLEL_GROUP = dp_group
            _DATA_PARALLEL_RANKS = ranks

    # TP 组由连续的 dp_size 个进程组成
    for tp_group_id in range(dp_size):
        start_rank = tp_group_id * tp_size
        end_rank = (tp_group_id + 1) * tp_size
        ranks = list(range(start_rank, end_rank))
        tp_group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=get_nccl_options('tp', nccl_comm_cfgs)
        )
        if rank in ranks:
            global _TENSOR_PARALLEL_GROUP
            _TENSOR_PARALLEL_GROUP = tp_group
            _TENSOR_PARALLEL_RANKS = ranks
    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def get_tensor_parallel_size():
    assert _TENSOR_PARALLEL_SIZE is not None, "tensor parallel size is not set"
    return _TENSOR_PARALLEL_SIZE


def get_tensor_parallel_group():
    assert _TENSOR_PARALLEL_GROUP is not None, "tensor parallel group is not initialized"
    return _TENSOR_PARALLEL_GROUP


def get_tensor_parallel_rank():
    assert _TENSOR_PARALLEL_RANKS is not None, "tensor parallel rank is not initialized"
    return _TENSOR_PARALLEL_RANKS


def get_data_parallel_size():
    assert _DATA_PARALLEL_SIZE is not None, "data parallel size is not initialized"
    return _DATA_PARALLEL_SIZE


def get_data_parallel_gloo():
    assert _DATA_PARALLEL_GROUP_GLOO is not None, "data parallel gloo group is not initialized"
    return _DATA_PARALLEL_GROUP_GLOO


def get_data_parallel_group():
    assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP


def get_data_parallel_rank():
    assert _DATA_PARALLEL_RANKS is not None, "data parallel rank is not initialized"
    return _DATA_PARALLEL_RANKS



def get_nccl_options(pg_name, nccl_comm_cfgs):
    if pg_name in nccl_comm_cfgs:
        nccl_options = torch.distributed.ProcessGroupNCCL.Options()
        nccl_options.config.cga_cluster_size = nccl_comm_cfgs[pg_name].get('cga_cluster_size', 4)
        nccl_options.config.max_ctas = nccl_comm_cfgs[pg_name].get('max_ctas', 32)
        nccl_options.config.min_ctas = nccl_comm_cfgs[pg_name].get('min_ctas', 1)
        return nccl_options
    else:
        return None


def get_safetensors_cut_weight(name: str, weights: torch.Tensor):
    translate_col_cut_tensors = ["ffn_down", "attn_output"]  # "kv_b_proj"
    translate_row_cut_tensors = ["ffn_gate", "ffn_up", "attn_q_b"]
    tp = get_tensor_parallel_size()
    if tp == 1 or weights.shape == torch.Size([1]):
        return weights
    rank = torch.distributed.get_rank()
    rank %= tp
    assert 0 <= rank < tp and tp > 0, f"unexpected {rank=}, {tp=}"
    if any(t in name for t in translate_col_cut_tensors):
        if weights.dim() == 1:
            return weights
        dim = weights.shape[-1]
        assert dim % tp == 0, f"unexpected division {dim=}, {tp=}"
        chunk_size = dim // tp
        output_weights = weights[:, rank * chunk_size:(rank + 1) * chunk_size]
        # print(f"col cut weights {name=} from {weights.shape=} to {output_weights.shape=}")
        return output_weights
    elif any(t in name for t in translate_row_cut_tensors):
        dim = weights.shape[0]
        assert dim % tp == 0, f"unexpected division {dim=}, {tp=}"
        chunk_size = dim // tp
        output_weights = weights[rank * chunk_size: (rank + 1) * chunk_size:]
        # print(f"row cut weights {name=} from {weights.shape=} to {output_weights.shape=}")
        return output_weights
    else:
        return weights


def get_absort_weight(model, config):
    # 新增q_absorb， out_absorb属性
    local_rank = torch.distributed.get_rank()
    tp = get_tensor_parallel_size()
    local_rank %= tp
    tp_heads = config.num_attention_heads // tp
    for i in range(config.num_hidden_layers):
        self = model.model.layers[i].self_attn
        if not (hasattr(self, 'q_absorb') and hasattr(self, 'out_absorb')):
            kv_b_proj = self.kv_b_proj.weight.view(config.num_attention_heads, -1, self.kv_lora_rank)
            q_absorb = kv_b_proj[:, :self.qk_nope_head_dim, :].clone()
            out_absorb = kv_b_proj[:, self.qk_nope_head_dim:, :].clone()
            q_absorb = q_absorb[local_rank * tp_heads: (local_rank + 1) * tp_heads, :, :].contiguous()
            out_absorb = out_absorb[local_rank * tp_heads: (local_rank + 1) * tp_heads, :, :].contiguous()
            out_absorb = out_absorb.transpose(1, 2).contiguous()
            setattr(self, "q_absorb", q_absorb)
            setattr(self, "out_absorb", out_absorb)
            del self.orig_module.kv_b_proj
    torch.distributed.barrier(get_tensor_parallel_group())


def allredeuce_warpper(func):
    def wrapper(*args, **kwargs):
        orig_output = func(*args, **kwargs)
        if isinstance(orig_output, tuple):
            if get_tensor_parallel_size() > 1:
                org_dtype = orig_output[0].dtype
                if org_dtype == torch.bfloat16:
                    dist.all_reduce(orig_output[0].to(dtype=torch.float16), op=dist.ReduceOp.SUM,
                                    group=get_tensor_parallel_group())
                else:
                    dist.all_reduce(orig_output[0], op=dist.ReduceOp.SUM, group=get_tensor_parallel_group())
                if org_dtype == torch.bfloat16:
                    bf_orig_output = orig_output[0].to(dtype=org_dtype)
                else:
                    bf_orig_output = orig_output[0]
            else:
                bf_orig_output = orig_output[0]
            return (bf_orig_output,) + orig_output[1:]
        else:
            if get_tensor_parallel_size() > 1:
                org_dtype = orig_output.dtype
                if org_dtype == torch.bfloat16:
                    orig_output = orig_output.to(dtype=torch.float16)
                dist.all_reduce(orig_output, op=dist.ReduceOp.SUM, group=get_tensor_parallel_group())
                if org_dtype == torch.bfloat16:
                    orig_output = orig_output.to(dtype=org_dtype)
            return orig_output

    return wrapper