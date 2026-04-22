"""
KTransformers — CPU-GPU heterogeneous fine-tuning for MoE models.

``pip install ktransformers`` gives you everything:
- ``kt_kernel`` — C++ AMX kernel engine
- ``ktransformers.integrations`` — auto-patching for transformers & accelerate

When ``ACCELERATE_USE_KT=true``, importing this package automatically
patches transformers and accelerate so custom forks are NOT needed.
"""

import os as _os

__all__: list[str] = []


def _ensure_patches() -> None:
    """Apply KT patches to transformers/accelerate if KT is enabled."""
    if _os.environ.get("ACCELERATE_USE_KT", "").lower() not in ("1", "true", "yes"):
        return
    try:
        from .integrations._patch import apply_all as _apply_all
        _apply_all()
    except Exception:
        pass


_ensure_patches()
