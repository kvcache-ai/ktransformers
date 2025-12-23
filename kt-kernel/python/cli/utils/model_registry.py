"""
Model registry for kt-cli.

Provides a registry of supported models with fuzzy matching capabilities.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

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
    max_tensor_parallel_size: Optional[int] = None  # Maximum tensor parallel size for this model


# Built-in model registry
BUILTIN_MODELS: list[ModelInfo] = [
    ModelInfo(
        name="DeepSeek-V3-0324",
        hf_repo="deepseek-ai/DeepSeek-V3-0324",
        aliases=["deepseek-v3-0324", "deepseek-v3", "dsv3", "deepseek3", "v3-0324"],
        type="moe",
        default_params={
            "kt-num-gpu-experts": 1,
            "attention-backend": "triton",
            "disable-shared-experts-fusion": True,
            "kt-method": "AMXINT4",
        },
        description="DeepSeek V3-0324 685B MoE model (March 2025, improved benchmarks)",
        description_zh="DeepSeek V3-0324 685B MoE 模型（2025年3月，改进的基准测试）",
    ),
    ModelInfo(
        name="DeepSeek-V3.2",
        hf_repo="deepseek-ai/DeepSeek-V3.2",
        aliases=["deepseek-v3.2", "dsv3.2", "deepseek3.2", "v3.2"],
        type="moe",
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
        name="DeepSeek-R1-0528",
        hf_repo="deepseek-ai/DeepSeek-R1-0528",
        aliases=["deepseek-r1-0528", "deepseek-r1", "dsr1", "r1", "r1-0528"],
        type="moe",
        default_params={
            "kt-num-gpu-experts": 1,
            "attention-backend": "triton",
            "disable-shared-experts-fusion": True,
            "kt-method": "AMXINT4",
        },
        description="DeepSeek R1-0528 reasoning model (May 2025, improved reasoning depth)",
        description_zh="DeepSeek R1-0528 推理模型（2025年5月，改进的推理深度）",
    ),
    ModelInfo(
        name="Kimi-K2-Thinking",
        hf_repo="moonshotai/Kimi-K2-Thinking",
        aliases=["kimi-k2-thinking", "kimi-thinking", "k2-thinking", "kimi", "k2"],
        type="moe",
        default_params={
            "kt-num-gpu-experts": 1,
            "attention-backend": "triton",
        },
        description="Moonshot Kimi K2 Thinking MoE model",
        description_zh="月之暗面 Kimi K2 Thinking MoE 模型",
    ),
    ModelInfo(
        name="MiniMax-M2",
        hf_repo="MiniMaxAI/MiniMax-M2",
        aliases=["minimax-m2", "m2"],
        type="moe",
        default_params={
            "kt-method": "FP8",
            "kt-gpu-prefill-token-threshold": 4096,
            "attention-backend": "flashinfer",
            "fp8-gemm-backend": "triton",
            "max-total-tokens": 100000,
            "max-running-requests": 16,
            "chunked-prefill-size": 32768,
            "mem-fraction-static": 0.80,
            "watchdog-timeout": 3000,
            "served-model-name": "MiniMax-M2",
            "disable-shared-experts-fusion": True,
            "tool-call-parser": "minimax-m2",
            "reasoning-parser": "minimax-append-think"
        },
        description="MiniMax M2 MoE model",
        description_zh="MiniMax M2 MoE 模型",
        max_tensor_parallel_size=4,  # M2 only supports up to 4-way tensor parallelism
    ),
    ModelInfo(
        name="MiniMax-M2.1",
        hf_repo="MiniMaxAI/MiniMax-M2.1",
        aliases=["minimax-m2.1", "m2.1"],
        type="moe",
        default_params={
            "kt-method": "FP8",
            "kt-gpu-prefill-token-threshold": 4096,
            "attention-backend": "flashinfer",
            "fp8-gemm-backend": "triton",
            "max-total-tokens": 100000,
            "max-running-requests": 16,
            "chunked-prefill-size": 32768,
            "mem-fraction-static": 0.80,
            "watchdog-timeout": 3000,
            "served-model-name": "MiniMax-M2.1",
            "disable-shared-experts-fusion": True,
            "tool-call-parser": "minimax-m2",
            "reasoning-parser": "minimax-append-think"
        },
        description="MiniMax M2.1 MoE model (enhanced multi-language programming)",
        description_zh="MiniMax M2.1 MoE 模型（增强多语言编程能力）",
        max_tensor_parallel_size=4,  # M2.1 only supports up to 4-way tensor parallelism
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
                        max_tensor_parallel_size=info.get("max_tensor_parallel_size"),
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


# ============================================================================
# Model-specific parameter computation functions
# ============================================================================


def compute_deepseek_v3_gpu_experts(tensor_parallel_size: int, vram_per_gpu_gb: float) -> int:
    per_gpu_gb = 16
    if vram_per_gpu_gb < per_gpu_gb:
        return int(0)
    total_vram =  int(tensor_parallel_size * (vram_per_gpu_gb - per_gpu_gb))

    return total_vram // 1


def compute_kimi_k2_thinking_gpu_experts(tensor_parallel_size: int, vram_per_gpu_gb: float) -> int:
    """Compute kt-num-gpu-experts for Kimi K2 Thinking."""
    per_gpu_gb = 16
    if vram_per_gpu_gb < per_gpu_gb:
        return int(0)
    total_vram =  int(tensor_parallel_size * (vram_per_gpu_gb - per_gpu_gb))

    return total_vram * 2


def compute_minimax_m2_gpu_experts(tensor_parallel_size: int, vram_per_gpu_gb: float) -> int:
    """Compute kt-num-gpu-experts for MiniMax M2/M2.1."""
    per_gpu_gb = 16
    if vram_per_gpu_gb < per_gpu_gb:
        return int(0)
    total_vram =  int(tensor_parallel_size * (vram_per_gpu_gb - per_gpu_gb))

    return total_vram // 1


# Model name to computation function mapping
MODEL_COMPUTE_FUNCTIONS: dict[str, Callable[[int, float], int]] = {
    "DeepSeek-V3-0324": compute_deepseek_v3_gpu_experts,
    "DeepSeek-V3.2": compute_deepseek_v3_gpu_experts,  # Same as V3-0324
    "DeepSeek-R1-0528": compute_deepseek_v3_gpu_experts,  # Same as V3-0324
    "Kimi-K2-Thinking": compute_kimi_k2_thinking_gpu_experts,
    "MiniMax-M2": compute_minimax_m2_gpu_experts,
    "MiniMax-M2.1": compute_minimax_m2_gpu_experts,  # Same as M2
}
