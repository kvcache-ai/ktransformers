"""
Configuration management for kt-cli.

Handles reading and writing configuration from ~/.ktransformers/config.yaml
"""

import os
from pathlib import Path
from typing import Any, Optional

import yaml

# Default configuration directory
DEFAULT_CONFIG_DIR = Path.home() / ".ktransformers"
DEFAULT_CONFIG_FILE = DEFAULT_CONFIG_DIR / "config.yaml"
DEFAULT_MODELS_DIR = DEFAULT_CONFIG_DIR / "models"
DEFAULT_CACHE_DIR = DEFAULT_CONFIG_DIR / "cache"

# Default configuration values
DEFAULT_CONFIG = {
    "general": {
        "language": "auto",  # auto, en, zh
        "color": True,
        "verbose": False,
    },
    "paths": {
        "models": str(DEFAULT_MODELS_DIR),
        "cache": str(DEFAULT_CACHE_DIR),
        "weights": "",  # Custom quantized weights path
    },
    "server": {
        "host": "0.0.0.0",
        "port": 30000,
    },
    "inference": {
        "cpu_threads": 0,  # 0 = auto-detect
        "numa_nodes": 0,  # 0 = auto-detect
        "gpu_experts": 1,
        "attention_backend": "triton",
        "max_total_tokens": 40000,
        "max_running_requests": 32,
        "chunked_prefill_size": 4096,
        "mem_fraction_static": 0.98,
    },
    "download": {
        "mirror": "",  # HuggingFace mirror URL
        "resume": True,
        "verify": True,
    },
    "advanced": {
        # Environment variables to set when running
        "env": {},
        # Extra arguments to pass to sglang
        "sglang_args": [],
        # Extra arguments to pass to llamafactory
        "llamafactory_args": [],
    },
}


class Settings:
    """Configuration manager for kt-cli."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize settings manager.

        Args:
            config_path: Path to config file. Defaults to ~/.ktransformers/config.yaml
        """
        self.config_path = config_path or DEFAULT_CONFIG_FILE
        self.config_dir = self.config_path.parent
        self._config: dict[str, Any] = {}
        self._load()

    def _ensure_dirs(self) -> None:
        """Ensure configuration directories exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        Path(self.get("paths.models", DEFAULT_MODELS_DIR)).mkdir(parents=True, exist_ok=True)
        Path(self.get("paths.cache", DEFAULT_CACHE_DIR)).mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        """Load configuration from file."""
        self._config = self._deep_copy(DEFAULT_CONFIG)

        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    user_config = yaml.safe_load(f) or {}
                self._deep_merge(self._config, user_config)
            except (yaml.YAMLError, OSError) as e:
                # Log warning but continue with defaults
                print(f"Warning: Failed to load config: {e}")

        self._ensure_dirs()

    def _save(self) -> None:
        """Save configuration to file."""
        self._ensure_dirs()
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)
        except OSError as e:
            raise RuntimeError(f"Failed to save config: {e}")

    def _deep_copy(self, obj: Any) -> Any:
        """Create a deep copy of a nested dict."""
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        return obj

    def _deep_merge(self, base: dict, override: dict) -> None:
        """Deep merge override into base."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot-separated key.

        Args:
            key: Dot-separated key path (e.g., "server.port")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        parts = key.split(".")
        value = self._config

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """Set a configuration value by dot-separated key.

        Args:
            key: Dot-separated key path (e.g., "server.port")
            value: Value to set
        """
        parts = key.split(".")
        config = self._config

        # Navigate to parent
        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]

        # Set value
        config[parts[-1]] = value
        self._save()

    def delete(self, key: str) -> bool:
        """Delete a configuration value.

        Args:
            key: Dot-separated key path

        Returns:
            True if key was deleted, False if not found
        """
        parts = key.split(".")
        config = self._config

        # Navigate to parent
        for part in parts[:-1]:
            if part not in config:
                return False
            config = config[part]

        # Delete key
        if parts[-1] in config:
            del config[parts[-1]]
            self._save()
            return True
        return False

    def reset(self) -> None:
        """Reset configuration to defaults."""
        self._config = self._deep_copy(DEFAULT_CONFIG)
        self._save()

    def get_all(self) -> dict[str, Any]:
        """Get all configuration values."""
        return self._deep_copy(self._config)

    def get_env_vars(self) -> dict[str, str]:
        """Get environment variables to set."""
        env_vars = {}

        # Get from advanced.env
        advanced_env = self.get("advanced.env", {})
        if isinstance(advanced_env, dict):
            env_vars.update({k: str(v) for k, v in advanced_env.items()})

        return env_vars

    @property
    def models_dir(self) -> Path:
        """Get the models directory path."""
        return Path(self.get("paths.models", DEFAULT_MODELS_DIR))

    @property
    def cache_dir(self) -> Path:
        """Get the cache directory path."""
        return Path(self.get("paths.cache", DEFAULT_CACHE_DIR))

    @property
    def weights_dir(self) -> Optional[Path]:
        """Get the custom weights directory path."""
        weights = self.get("paths.weights", "")
        return Path(weights) if weights else None


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset the global settings instance."""
    global _settings
    _settings = None
