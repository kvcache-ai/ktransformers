#!/usr/bin/env python3
"""
快速分析 MoE 模型 - 基于 config.json
(复用 sglang 的模型注册表和判断逻辑)
"""
import json
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any


def _get_sglang_moe_architectures():
    """
    从 sglang 的模型注册表获取所有 MoE 架构

    复用 sglang 的代码，这样 sglang 更新后自动支持新模型
    """
    try:
        import sys

        # 添加 sglang 路径到 sys.path
        sglang_path = Path("/mnt/data2/ljq/sglang/python")
        if sglang_path.exists() and str(sglang_path) not in sys.path:
            sys.path.insert(0, str(sglang_path))

        # 直接导入 sglang 的 ModelRegistry
        # 注意：这需要 sglang 及其依赖正确安装
        from sglang.srt.models.registry import ModelRegistry

        # 获取所有支持的架构
        supported_archs = ModelRegistry.get_supported_archs()

        # 过滤出 MoE 模型（名称包含 Moe）
        moe_archs = {arch for arch in supported_archs if "Moe" in arch or "moe" in arch.lower()}

        # 手动添加一些不带 "Moe" 字样但是 MoE 模型的架构
        # DeepSeek V2/V3 系列
        deepseek_moe = {arch for arch in supported_archs if arch.startswith("Deepseek") or arch.startswith("deepseek")}
        moe_archs.update(deepseek_moe)

        # DBRX 也是 MoE 模型
        dbrx_moe = {arch for arch in supported_archs if "DBRX" in arch or "dbrx" in arch.lower()}
        moe_archs.update(dbrx_moe)

        # Grok 也是 MoE 模型
        grok_moe = {arch for arch in supported_archs if "Grok" in arch or "grok" in arch.lower()}
        moe_archs.update(grok_moe)

        return moe_archs
    except Exception as e:
        # 如果 sglang 不可用，返回空集合
        # 这种情况下，后续会使用配置文件中的其他判断方法
        import warnings

        warnings.warn(f"Failed to load MoE architectures from sglang: {e}. Using fallback detection methods.")
        return set()


# 获取 MoE 架构列表（优先从 sglang 获取）
MOE_ARCHITECTURES = _get_sglang_moe_architectures()


def _get_cache_file():
    """获取集中式缓存文件路径"""
    cache_dir = Path.home() / ".ktransformers" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "moe_analysis_v2.json"


def _load_all_cache():
    """加载所有缓存数据"""
    cache_file = _get_cache_file()
    if not cache_file.exists():
        return {}

    try:
        with open(cache_file, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_all_cache(cache_data):
    """保存所有缓存数据"""
    cache_file = _get_cache_file()
    try:
        with open(cache_file, "w") as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        import warnings

        warnings.warn(f"Failed to save MoE cache: {e}")


def _compute_config_fingerprint(config_path: Path) -> Optional[str]:
    """计算 config.json 指纹"""
    if not config_path.exists():
        return None

    try:
        stat = config_path.stat()
        # 使用文件大小和修改时间作为指纹
        fingerprint_str = f"{config_path.name}:{stat.st_size}:{int(stat.st_mtime)}"
        return hashlib.md5(fingerprint_str.encode()).hexdigest()
    except Exception:
        return None


def _load_cache(model_path: Path) -> Optional[Dict[str, Any]]:
    """加载指定模型的缓存"""
    model_path_str = str(model_path.resolve())
    all_cache = _load_all_cache()

    if model_path_str not in all_cache:
        return None

    try:
        cache_entry = all_cache[model_path_str]

        # 验证缓存版本
        cache_version = cache_entry.get("cache_version", 0)
        if cache_version != 2:
            return None

        # 验证 config.json 指纹
        config_path = model_path / "config.json"
        current_fingerprint = _compute_config_fingerprint(config_path)
        if cache_entry.get("fingerprint") != current_fingerprint:
            return None

        return cache_entry.get("result")
    except Exception:
        return None


def _save_cache(model_path: Path, result: Dict[str, Any]):
    """保存指定模型的缓存"""
    model_path_str = str(model_path.resolve())

    try:
        config_path = model_path / "config.json"
        fingerprint = _compute_config_fingerprint(config_path)

        all_cache = _load_all_cache()

        all_cache[model_path_str] = {
            "fingerprint": fingerprint,
            "result": result,
            "cache_version": 2,
            "last_updated": __import__("datetime").datetime.now().isoformat(),
        }

        _save_all_cache(all_cache)
    except Exception as e:
        import warnings

        warnings.warn(f"Failed to save MoE cache for {model_path}: {e}")


def _load_config_json(model_path: Path) -> Optional[Dict[str, Any]]:
    """读取 config.json 文件

    参考 sglang 的 get_config() 实现
    """
    config_path = model_path / "config.json"

    if not config_path.exists():
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except Exception:
        return None


def _is_moe_model(config: Dict[str, Any]) -> bool:
    """判断是否是 MoE 模型

    参考 sglang 的模型注册表和架构识别方式
    """
    # 方法1: 检查架构名称
    architectures = config.get("architectures", [])
    if any(arch in MOE_ARCHITECTURES for arch in architectures):
        return True

    # 方法2: 检查是否有 MoE 相关字段（Mistral 格式）
    if config.get("moe"):
        return True

    # 方法3: 检查是否有 num_experts 或其变体字段
    # 需要检查 text_config（对于某些多模态模型）
    text_config = config.get("text_config", config)

    # 检查各种专家数量字段
    if (
        text_config.get("num_experts") or text_config.get("num_local_experts") or text_config.get("n_routed_experts")
    ):  # Kimi-K2 使用这个字段
        return True

    return False


def _extract_moe_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """从 config 中提取 MoE 参数

    参考 sglang 的各种 MoE 模型实现
    """
    # 处理嵌套的 text_config
    text_config = config.get("text_config", config)

    # 提取基本参数
    result = {
        "architectures": config.get("architectures", []),
        "model_type": config.get("model_type", "unknown"),
    }

    # 专家数量（不同模型字段名不同）
    num_experts = (
        text_config.get("num_experts")  # Qwen2/3 MoE, DeepSeek V2
        or text_config.get("num_local_experts")  # Mixtral
        or text_config.get("n_routed_experts")  # Kimi-K2, DeepSeek V3
        or config.get("moe", {}).get("num_experts")  # Mistral 格式
    )

    # 每个 token 激活的专家数
    num_experts_per_tok = (
        text_config.get("num_experts_per_tok")
        or text_config.get("num_experts_per_token")
        or config.get("moe", {}).get("num_experts_per_tok")
        or 2  # 默认值
    )

    # 层数
    num_hidden_layers = text_config.get("num_hidden_layers") or text_config.get("n_layer") or 0

    # 隐藏层维度
    hidden_size = text_config.get("hidden_size") or text_config.get("d_model") or 0

    # MoE 专家中间层大小
    moe_intermediate_size = (
        text_config.get("moe_intermediate_size")
        or text_config.get("intermediate_size")  # 如果没有特殊的 moe_intermediate_size
        or 0
    )

    # 共享专家中间层大小（Qwen2/3 MoE）
    shared_expert_intermediate_size = text_config.get("shared_expert_intermediate_size", 0)

    result.update(
        {
            "num_experts": num_experts or 0,
            "num_experts_per_tok": num_experts_per_tok,
            "num_hidden_layers": num_hidden_layers,
            "hidden_size": hidden_size,
            "moe_intermediate_size": moe_intermediate_size,
            "shared_expert_intermediate_size": shared_expert_intermediate_size,
        }
    )

    # 提取其他有用的参数
    result["num_attention_heads"] = text_config.get("num_attention_heads", 0)
    result["num_key_value_heads"] = text_config.get("num_key_value_heads", 0)
    result["vocab_size"] = text_config.get("vocab_size", 0)
    result["max_position_embeddings"] = text_config.get("max_position_embeddings", 0)

    return result


def _estimate_model_size(model_path: Path) -> float:
    """估算模型总大小（GB）

    快速统计 safetensors 文件总大小
    """
    try:
        total_size = 0
        for file_path in model_path.glob("*.safetensors"):
            total_size += file_path.stat().st_size
        return total_size / (1024**3)
    except Exception:
        return 0.0


def analyze_moe_model(model_path, use_cache=True):
    """
    快速分析 MoE 模型 - 只读取 config.json

    参数:
        model_path: 模型路径（字符串或Path对象）
        use_cache: 是否使用缓存（默认True）

    返回:
        dict: {
            'is_moe': 是否是 MoE 模型,
            'num_experts': 专家总数,
            'num_experts_per_tok': 每个 token 激活的专家数,
            'num_hidden_layers': 层数,
            'hidden_size': 隐藏层维度,
            'moe_intermediate_size': MoE 专家中间层大小,
            'shared_expert_intermediate_size': 共享专家中间层大小,
            'architectures': 模型架构列表,
            'model_type': 模型类型,
            'total_size_gb': 模型总大小（估算，GB）,
            'cached': 是否从缓存读取
        }
        如果不是 MoE 模型或失败，返回 None
    """
    model_path = Path(model_path)

    if not model_path.exists():
        return None

    # 尝试加载缓存
    if use_cache:
        cached_result = _load_cache(model_path)
        if cached_result:
            cached_result["cached"] = True
            return cached_result

    # 读取 config.json
    config = _load_config_json(model_path)
    if not config:
        return None

    # 判断是否是 MoE 模型
    if not _is_moe_model(config):
        return None

    # 提取 MoE 参数
    params = _extract_moe_params(config)

    # 验证必要参数
    if params["num_experts"] == 0:
        return None

    # 估算模型大小
    total_size_gb = _estimate_model_size(model_path)

    # 组装结果
    result = {
        "is_moe": True,
        "num_experts": params["num_experts"],
        "num_experts_per_tok": params["num_experts_per_tok"],
        "num_hidden_layers": params["num_hidden_layers"],
        "hidden_size": params["hidden_size"],
        "moe_intermediate_size": params["moe_intermediate_size"],
        "shared_expert_intermediate_size": params["shared_expert_intermediate_size"],
        "architectures": params["architectures"],
        "model_type": params["model_type"],
        "total_size_gb": total_size_gb,
        "cached": False,
        # 额外参数
        "num_attention_heads": params.get("num_attention_heads", 0),
        "num_key_value_heads": params.get("num_key_value_heads", 0),
        "vocab_size": params.get("vocab_size", 0),
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
        print("不是 MoE 模型或分析失败")
        return

    print("=" * 70)
    print("MoE 模型分析结果")
    if result.get("cached"):
        print("[使用缓存]")
    print("=" * 70)
    print(f"模型架构:")
    print(f"  - 架构: {', '.join(result['architectures'])}")
    print(f"  - 类型: {result['model_type']}")
    print()
    print(f"MoE 结构:")
    print(f"  - 专家总数: {result['num_experts']}")
    print(f"  - 激活专家数: {result['num_experts_per_tok']} experts/token")
    print(f"  - 层数: {result['num_hidden_layers']}")
    print(f"  - 隐藏维度: {result['hidden_size']}")
    print(f"  - MoE 中间层: {result['moe_intermediate_size']}")
    if result["shared_expert_intermediate_size"] > 0:
        print(f"  - 共享专家中间层: {result['shared_expert_intermediate_size']}")
    print()
    print(f"大小统计:")
    print(f"  - 模型总大小: {result['total_size_gb']:.2f} GB")
    print("=" * 70)
    print()


def main():
    import sys

    models = ["/mnt/data2/models/Qwen3-30B-A3B", "/mnt/data2/models/Qwen3-235B-A22B-Instruct-2507"]

    if len(sys.argv) > 1:
        models = [sys.argv[1]]

    for model_path in models:
        print_analysis(model_path)


if __name__ == "__main__":
    main()
