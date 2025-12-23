"""
Model registry for kt-cli.

Provides a registry of supported models with fuzzy matching capabilities.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from kt_kernel.cli.config.settings import get_settings


@dataclass
class ModelInfo:
    """Information about a supported model."""

    name: str
    hf_repo: str
    aliases: list[str] = field(default_factory=list)
    type: str = "moe"  # moe, dense
    gpu_vram_gb: float = 0
    cpu_ram_gb: float = 0
    default_params: dict = field(default_factory=dict)
    description: str = ""
    description_zh: str = ""


# Built-in model registry
BUILTIN_MODELS: list[ModelInfo] = [
    ModelInfo(
        name="DeepSeek-V3",
        hf_repo="deepseek-ai/DeepSeek-V3",
        aliases=["deepseek-v3", "dsv3", "deepseek3"],
        type="moe",
        gpu_vram_gb=27,
        cpu_ram_gb=350,
        default_params={
            "kt-num-gpu-experts": 1,
            "attention-backend": "triton",
            "disable-shared-experts-fusion": True,
            "kt-method": "AMXINT4",
        },
        description="DeepSeek V3 671B MoE model",
        description_zh="DeepSeek V3 671B MoE 模型",
    ),
    ModelInfo(
        name="DeepSeek-V3.2",
        hf_repo="deepseek-ai/DeepSeek-V3.2",
        aliases=["deepseek-v3.2", "dsv3.2", "deepseek3.2", "v3.2"],
        type="moe",
        gpu_vram_gb=27,
        cpu_ram_gb=350,
        default_params={
            "kt-num-gpu-experts": 1,
            "attention-backend": "triton",
            "disable-shared-experts-fusion": True,
            "kt-method": "AMXINT4",
        },
        description="DeepSeek V3.2 671B MoE model (latest)",
        description_zh="DeepSeek V3.2 671B MoE 模型（最新）",
    ),
    ModelInfo(
        name="DeepSeek-V2",
        hf_repo="deepseek-ai/DeepSeek-V2",
        aliases=["deepseek-v2", "dsv2", "deepseek2"],
        type="moe",
        gpu_vram_gb=16,
        cpu_ram_gb=128,
        default_params={
            "kt-num-gpu-experts": 2,
            "attention-backend": "triton",
        },
        description="DeepSeek V2 236B MoE model",
        description_zh="DeepSeek V2 236B MoE 模型",
    ),
    ModelInfo(
        name="DeepSeek-V2.5",
        hf_repo="deepseek-ai/DeepSeek-V2.5",
        aliases=["deepseek-v2.5", "dsv2.5", "deepseek2.5"],
        type="moe",
        gpu_vram_gb=16,
        cpu_ram_gb=128,
        default_params={
            "kt-num-gpu-experts": 2,
            "attention-backend": "triton",
        },
        description="DeepSeek V2.5 236B MoE model",
        description_zh="DeepSeek V2.5 236B MoE 模型",
    ),
    ModelInfo(
        name="Qwen3-30B-A3B",
        hf_repo="Qwen/Qwen3-30B-A3B",
        aliases=["qwen3-30b-a3b", "qwen3-30b", "qwen3-moe", "qwen3"],
        type="moe",
        gpu_vram_gb=12,
        cpu_ram_gb=64,
        default_params={
            "kt-num-gpu-experts": 2,
            "attention-backend": "triton",
        },
        description="Qwen3 30B MoE model with 3B active parameters",
        description_zh="Qwen3 30B MoE 模型，3B 活跃参数",
    ),
    ModelInfo(
        name="Qwen2.5-57B-A14B",
        hf_repo="Qwen/Qwen2.5-57B-A14B",
        aliases=["qwen2.5-57b-a14b", "qwen2.5-57b", "qwen2.5-moe"],
        type="moe",
        gpu_vram_gb=16,
        cpu_ram_gb=128,
        default_params={
            "kt-num-gpu-experts": 2,
            "attention-backend": "triton",
        },
        description="Qwen2.5 57B MoE model with 14B active parameters",
        description_zh="Qwen2.5 57B MoE 模型，14B 活跃参数",
    ),
    ModelInfo(
        name="Kimi-K2",
        hf_repo="moonshotai/Kimi-K2",
        aliases=["kimi-k2", "kimi", "k2"],
        type="moe",
        gpu_vram_gb=24,
        cpu_ram_gb=256,
        default_params={
            "kt-num-gpu-experts": 1,
            "attention-backend": "triton",
        },
        description="Moonshot Kimi K2 MoE model",
        description_zh="月之暗面 Kimi K2 MoE 模型",
    ),
    ModelInfo(
        name="Mixtral-8x7B",
        hf_repo="mistralai/Mixtral-8x7B-v0.1",
        aliases=["mixtral-8x7b", "mixtral", "mixtral-moe"],
        type="moe",
        gpu_vram_gb=12,
        cpu_ram_gb=48,
        default_params={
            "kt-num-gpu-experts": 2,
            "attention-backend": "triton",
        },
        description="Mistral Mixtral 8x7B MoE model",
        description_zh="Mistral Mixtral 8x7B MoE 模型",
    ),
    ModelInfo(
        name="Mixtral-8x22B",
        hf_repo="mistralai/Mixtral-8x22B-v0.1",
        aliases=["mixtral-8x22b", "mixtral-22b"],
        type="moe",
        gpu_vram_gb=24,
        cpu_ram_gb=176,
        default_params={
            "kt-num-gpu-experts": 1,
            "attention-backend": "triton",
        },
        description="Mistral Mixtral 8x22B MoE model",
        description_zh="Mistral Mixtral 8x22B MoE 模型",
    ),
    ModelInfo(
        name="MiniMax-M2",
        hf_repo="MiniMaxAI/MiniMax-M2",
        aliases=["minimax-m2", "minimax", "m2"],
        type="moe",
        gpu_vram_gb=80,  # Example: 4 GPUs with 20GB each
        cpu_ram_gb=400,
        default_params={
            "kt-num-gpu-experts": 60,
            "kt-method": "RAWFP8",
            "kt-gpu-prefill-token-threshold": 4096,
            "attention-backend": "flashinfer",
            "tensor-parallel-size": 4,
            "max-total-tokens": 100000,
            "max-running-requests": 16,
            "chunked-prefill-size": 32768,
            "mem-fraction-static": 0.80,
            "watchdog-timeout": 3000,
            "served-model-name": "DeepSeek",
            "disable-shared-experts-fusion": True,
        },
        description="MiniMax M2 MoE model (example configuration)",
        description_zh="MiniMax M2 MoE 模型（示例配置）",
    ),
]


class ModelRegistry:
    """Registry of supported models with fuzzy matching."""

    def __init__(self):
        """Initialize the model registry."""
        self._models: dict[str, ModelInfo] = {}
        self._aliases: dict[str, str] = {}
        self._load_builtin_models()
        self._load_user_models()

    def _load_builtin_models(self) -> None:
        """Load built-in models."""
        for model in BUILTIN_MODELS:
            self._register(model)

    def _load_user_models(self) -> None:
        """Load user-defined models from config."""
        settings = get_settings()
        registry_file = settings.config_dir / "registry.yaml"

        if registry_file.exists():
            try:
                with open(registry_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}

                for name, info in data.get("models", {}).items():
                    model = ModelInfo(
                        name=name,
                        hf_repo=info.get("hf_repo", ""),
                        aliases=info.get("aliases", []),
                        type=info.get("type", "moe"),
                        gpu_vram_gb=info.get("gpu_vram_gb", 0),
                        cpu_ram_gb=info.get("cpu_ram_gb", 0),
                        default_params=info.get("default_params", {}),
                        description=info.get("description", ""),
                        description_zh=info.get("description_zh", ""),
                    )
                    self._register(model)
            except (yaml.YAMLError, OSError):
                pass

    def _register(self, model: ModelInfo) -> None:
        """Register a model."""
        self._models[model.name.lower()] = model

        # Register aliases
        for alias in model.aliases:
            self._aliases[alias.lower()] = model.name.lower()

    def get(self, name: str) -> Optional[ModelInfo]:
        """Get a model by exact name or alias."""
        name_lower = name.lower()

        # Check direct match
        if name_lower in self._models:
            return self._models[name_lower]

        # Check aliases
        if name_lower in self._aliases:
            return self._models[self._aliases[name_lower]]

        return None

    def search(self, query: str, limit: int = 10) -> list[ModelInfo]:
        """Search for models using fuzzy matching.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching models, sorted by relevance
        """
        query_lower = query.lower()
        results: list[tuple[float, ModelInfo]] = []

        for model in self._models.values():
            score = self._match_score(query_lower, model)
            if score > 0:
                results.append((score, model))

        # Sort by score descending
        results.sort(key=lambda x: x[0], reverse=True)

        return [model for _, model in results[:limit]]

    def _match_score(self, query: str, model: ModelInfo) -> float:
        """Calculate match score for a model.

        Returns a score between 0 and 1, where 1 is an exact match.
        """
        # Check exact match
        if query == model.name.lower():
            return 1.0

        # Check alias exact match
        for alias in model.aliases:
            if query == alias.lower():
                return 0.95

        # Check if query is contained in name
        if query in model.name.lower():
            return 0.8

        # Check if query is contained in aliases
        for alias in model.aliases:
            if query in alias.lower():
                return 0.7

        # Check if query is contained in hf_repo
        if query in model.hf_repo.lower():
            return 0.6

        # Fuzzy matching - check if all query parts are present
        query_parts = re.split(r"[-_.\s]", query)
        name_lower = model.name.lower()

        matches = sum(1 for part in query_parts if part and part in name_lower)
        if matches > 0:
            return 0.5 * (matches / len(query_parts))

        return 0.0

    def list_all(self) -> list[ModelInfo]:
        """List all registered models."""
        return list(self._models.values())

    def find_local_models(self) -> list[tuple[ModelInfo, Path]]:
        """Find models that are downloaded locally in any configured model path.

        Returns:
            List of (ModelInfo, path) tuples for local models
        """
        settings = get_settings()
        model_paths = settings.get_model_paths()
        results = []

        for model in self._models.values():
            found = False
            # Search in all configured model directories
            for models_dir in model_paths:
                if not models_dir.exists():
                    continue

                # Check common path patterns
                possible_paths = [
                    models_dir / model.name,
                    models_dir / model.name.lower(),
                    models_dir / model.hf_repo.split("/")[-1],
                    models_dir / model.hf_repo.replace("/", "--"),
                ]

                for path in possible_paths:
                    if path.exists() and (path / "config.json").exists():
                        results.append((model, path))
                        found = True
                        break

                if found:
                    break

        return results


# Global registry instance
_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry
