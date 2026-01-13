#!/usr/bin/env python3
"""
分析MoE模型的专家分布和大小
"""
import sys
import json
import hashlib
from pathlib import Path
from collections import defaultdict

try:
    from safetensors import safe_open
except ImportError:
    print("请先安装safetensors: pip install safetensors")
    sys.exit(1)


def _get_cache_file():
    """获取集中式缓存文件路径

    所有模型的缓存信息存储在一个文件中
    缓存位置：~/.ktransformers/cache/moe_analysis.json
    """
    cache_dir = Path.home() / ".ktransformers" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "moe_analysis.json"


def _load_all_cache():
    """加载所有缓存数据

    返回:
        dict: {model_path: cache_data}
    """
    cache_file = _get_cache_file()
    if not cache_file.exists():
        return {}

    try:
        with open(cache_file, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_all_cache(cache_data):
    """保存所有缓存数据

    参数:
        cache_data: dict, {model_path: cache_data}
    """
    cache_file = _get_cache_file()
    try:
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        import warnings

        warnings.warn(f"Failed to save MoE cache: {e}")


def _compute_model_fingerprint(model_path):
    """计算模型指纹（用于检测模型是否变化）"""
    model_path = Path(model_path)
    safetensors_files = sorted(model_path.glob("*.safetensors"))

    if not safetensors_files:
        return None

    # 使用文件列表、文件大小和修改时间作为指纹
    fingerprint_data = []
    for f in safetensors_files:
        stat = f.stat()
        fingerprint_data.append({"name": f.name, "size": stat.st_size, "mtime": int(stat.st_mtime)})

    # 生成哈希
    fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
    return hashlib.md5(fingerprint_str.encode()).hexdigest()


def _load_cache(model_path):
    """加载指定模型的缓存

    参数:
        model_path: 模型路径

    返回:
        dict 或 None: 缓存的分析结果
    """
    model_path = Path(model_path).resolve()
    model_path_str = str(model_path)

    all_cache = _load_all_cache()

    if model_path_str not in all_cache:
        return None

    try:
        cache_entry = all_cache[model_path_str]

        # 验证缓存版本（兼容旧版本）
        cache_version = cache_entry.get("cache_version", 0)
        if cache_version > 1:
            # 版本太新，无法读取
            return None

        # 验证指纹
        current_fingerprint = _compute_model_fingerprint(model_path)
        if cache_entry.get("fingerprint") != current_fingerprint:
            return None  # 模型已变化，缓存失效

        return cache_entry.get("result")
    except Exception:
        return None


def _save_cache(model_path, result):
    """保存指定模型的缓存

    参数:
        model_path: 模型路径
        result: 分析结果
    """
    model_path = Path(model_path).resolve()
    model_path_str = str(model_path)

    try:
        fingerprint = _compute_model_fingerprint(model_path)

        # 加载所有缓存
        all_cache = _load_all_cache()

        # 更新此模型的缓存
        all_cache[model_path_str] = {
            "fingerprint": fingerprint,
            "result": result,
            "cache_version": 1,
            "last_updated": __import__("datetime").datetime.now().isoformat(),
        }

        # 保存所有缓存
        _save_all_cache(all_cache)
    except Exception as e:
        # 缓存保存失败不影响主流程，但可以记录警告
        import warnings

        warnings.warn(f"Failed to save MoE cache for {model_path}: {e}")


def _load_model_config(model_path):
    """从 config.json 加载模型配置（参考 convert_cpu_weights.py）

    参数:
        model_path: 模型路径

    返回:
        dict 或 None: 包含 num_experts 和 num_layers 的字典，失败返回 None
    """
    config_path = model_path / "config.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        # 支持嵌套的 text_config（参考 convert_cpu_weights.py）
        if "text_config" in config:
            text_cfg = config["text_config"]
        else:
            text_cfg = config

        # 提取专家数量和层数
        num_experts = text_cfg.get("n_routed_experts") or text_cfg.get("num_experts")
        num_layers = text_cfg.get("num_hidden_layers")

        if num_experts and num_layers:
            return {"num_experts": num_experts, "num_layers": num_layers}
    except Exception:
        pass

    return None


def analyze_moe_model(model_path, use_cache=True):
    """
    分析MoE模型的专家分布和大小

    参数:
        model_path: 模型路径（字符串或Path对象）
        use_cache: 是否使用缓存（默认True）

    返回:
        dict: {
            'num_experts': 专家数量（按编号计，与层数无关）,
            'num_layers': 层数,
            'single_expert_size_gb': 单个专家在所有层的大小(GB),
            'rest_size_gb': 其余部分大小，即非expert部分(GB),
            'total_size_gb': 总大小(GB),
            'all_experts_size_gb': 所有专家大小(GB),
            'cached': 是否从缓存读取
        }
        如果失败返回 None
    """
    model_path = Path(model_path)

    if not model_path.exists():
        print(f"错误: 路径不存在 {model_path}")
        return None

    # 尝试加载缓存
    if use_cache:
        cached_result = _load_cache(model_path)
        if cached_result:
            cached_result["cached"] = True
            return cached_result

    # 尝试从 config.json 读取配置（参考 convert_cpu_weights.py）
    config_info = _load_model_config(model_path)

    # 查找所有safetensors文件
    safetensors_files = sorted(model_path.glob("*.safetensors"))

    if not safetensors_files:
        print(f"错误: 在 {model_path} 中没有找到safetensors文件")
        return None

    # 统计变量
    total_size_bytes = 0
    expert_0_weights = defaultdict(lambda: defaultdict(lambda: {"size": 0}))  # {layer_idx: {weight_name: size}}
    num_layers = 0
    num_experts_per_layer = 0
    layout = "base"  # "base" or "fused"

    # 确定主前缀（通常是 'model'）
    # 有些模型可能有多个前缀（如 model. 和 mtp.），我们只统计主前缀的
    main_prefix = None

    # 遍历所有文件
    for file_path in safetensors_files:
        try:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                # 第一次遍历：确定主前缀和layout（参考 convert_cpu_weights.py）
                if main_prefix is None:
                    prefix_counts = defaultdict(int)
                    for key in f.keys():
                        # 检测 fused layout（参考 convert_cpu_weights.py）
                        if ".mlp.experts." in key and "gate_up" in key:
                            layout = "fused"

                        if ".mlp.experts." in key or ".block_sparse_moe.experts." in key:
                            prefix = key.split(".")[0]
                            prefix_counts[prefix] += 1
                    if prefix_counts:
                        # 选择出现次数最多的前缀作为主前缀
                        main_prefix = max(prefix_counts.items(), key=lambda x: x[1])[0]

                for key in f.keys():
                    tensor = f.get_tensor(key)

                    # 计算大小
                    num_params = 1
                    for dim in tensor.shape:
                        num_params *= dim
                    size_bytes = num_params * tensor.element_size()
                    total_size_bytes += size_bytes

                    # 判断是否是expert权重
                    # 支持多种命名格式：.mlp.experts. 或 .block_sparse_moe.experts.
                    is_expert_weight = ".mlp.experts." in key or ".block_sparse_moe.experts." in key

                    if is_expert_weight:
                        parts = key.split(".")

                        # 只统计主前缀的专家
                        if main_prefix and not key.startswith(f"{main_prefix}."):
                            continue

                        if "layers" in parts:
                            layer_idx = int(parts[parts.index("layers") + 1])
                            num_layers = max(num_layers, layer_idx + 1)
                        else:
                            layer_idx = None

                        if layout == "fused":
                            # Fused layout: model.layers.{layer}.mlp.experts.{proj}
                            # 所有专家合并在一个张量里，从 config.json 获取专家数
                            if config_info and layer_idx is not None:
                                # 使用 config.json 的信息
                                num_experts_per_layer = config_info["num_experts"]
                                # Fused 格式下，张量是 [E, ...] 形状，E 是专家数
                                # 单个专家大小 = tensor_size / num_experts
                                single_expert_in_tensor = size_bytes / num_experts_per_layer
                                weight_name = parts[-1]  # gate_up 或 down 等
                                expert_0_weights[layer_idx][weight_name]["size"] += single_expert_in_tensor
                        elif "experts" in parts:
                            expert_idx = int(parts[parts.index("experts") + 1])
                            num_experts_per_layer = max(num_experts_per_layer, expert_idx + 1)

                            # 收集所有层的 expert 0 权重大小
                            if expert_idx == 0 and layer_idx is not None:
                                weight_name = parts[-2]
                                expert_0_weights[layer_idx][weight_name]["size"] += size_bytes

        except Exception as e:
            print(f"读取文件 {file_path.name} 时出错: {e}")
            return None

    # 优先使用 config.json 的信息（参考 convert_cpu_weights.py）
    if config_info:
        if num_experts_per_layer == 0:
            num_experts_per_layer = config_info["num_experts"]
        if num_layers == 0:
            num_layers = config_info["num_layers"]

    # 计算
    # 选择第一个包含专家的层（层号最小）
    if not expert_0_weights:
        # 没有找到任何专家权重
        single_expert_size_bytes = 0
        first_expert_layer = None
        num_expert_layers = 0
    else:
        first_expert_layer = min(expert_0_weights.keys())
        expert_weights = expert_0_weights[first_expert_layer]

        # 1. 单个expert在单层的大小
        single_expert_size_bytes = sum(w["size"] for w in expert_weights.values())

        # 计算有多少层包含专家
        num_expert_layers = len(expert_0_weights)

    # 2. 单个expert在所有专家层的大小
    single_expert_across_layers_size = single_expert_size_bytes * num_expert_layers

    # 3. 所有experts的大小 = 单个expert × 专家层数 × 专家数
    all_experts_size = single_expert_size_bytes * num_expert_layers * num_experts_per_layer

    # 4. 其余部分 = 总大小 - 所有experts
    rest_size = total_size_bytes - all_experts_size

    # 转换为GB
    result = {
        "num_experts": num_experts_per_layer,
        "num_layers": num_layers,
        "single_expert_size_gb": single_expert_across_layers_size / (1024**3),
        "rest_size_gb": rest_size / (1024**3),
        "total_size_gb": total_size_bytes / (1024**3),
        "all_experts_size_gb": all_experts_size / (1024**3),
        "cached": False,
    }

    # 保存缓存
    if use_cache:
        _save_cache(model_path, result)

    return result


def print_analysis(model_path):
    """打印模型分析结果"""
    print(f"分析模型: {model_path}\n")

    result = analyze_moe_model(model_path)

    if result is None:
        print("分析失败")
        return

    print("=" * 70)
    print("MoE模型分析结果")
    if result.get("cached"):
        print("[使用缓存]")
    print("=" * 70)
    print(f"模型结构:")
    print(f"  - 专家数量: {result['num_experts']} (按编号计，每层都有这些experts)")
    print(f"  - 层数: {result['num_layers']}")
    print()
    print(f"大小统计:")
    print(
        f"  - 单个专家大小: {result['single_expert_size_gb']:.4f} GB "
        f"({result['single_expert_size_gb']/result['total_size_gb']*100:.2f}%)"
    )
    print(
        f"  - 其余部分大小 (非expert): {result['rest_size_gb']:.4f} GB "
        f"({result['rest_size_gb']/result['total_size_gb']*100:.2f}%)"
    )
    print(f"  - 总大小: {result['total_size_gb']:.4f} GB")
    print()
    print(f"详细分解:")
    print(f"  - 非Expert部分: {result['rest_size_gb']:.4f} GB")
    print(f"  - 所有Experts ({result['num_experts']}个): {result['all_experts_size_gb']:.4f} GB")
    print(f"    其中单个expert: {result['single_expert_size_gb']:.4f} GB")
    print(
        f"    其他{result['num_experts']-1}个experts: {result['all_experts_size_gb'] - result['single_expert_size_gb']:.4f} GB"
    )
    print("=" * 70)
    print()


def main():
    # 测试两个模型
    models = ["/mnt/data2/models/Qwen3-30B-A3B", "/mnt/data2/models/Qwen3-235B-A22B-Instruct-2507"]

    if len(sys.argv) > 1:
        # 如果提供了命令行参数，使用它
        models = [sys.argv[1]]

    for model_path in models:
        print_analysis(model_path)


if __name__ == "__main__":
    main()
