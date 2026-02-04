"""
Configuration save/load for kt run command.

Manages saved run configurations bound to specific models.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import yaml


CONFIG_FILE = Path.home() / ".ktransformers" / "run_configs.yaml"


class RunConfigManager:
    """Manager for saved run configurations."""

    def __init__(self):
        self.config_file = CONFIG_FILE
        self._ensure_config_file()

    def _ensure_config_file(self):
        """Ensure config file exists."""
        if not self.config_file.exists():
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            self._save_data({"version": "1.0", "configs": {}})

    def _load_data(self) -> Dict:
        """Load raw config data."""
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {"version": "1.0", "configs": {}}
        except Exception:
            return {"version": "1.0", "configs": {}}

    def _save_data(self, data: Dict):
        """Save raw config data."""
        with open(self.config_file, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    def list_configs(self, model_id: str) -> List[Dict[str, Any]]:
        """List all saved configs for a model.

        Returns:
            List of config dicts with 'config_name' and other fields.
        """
        data = self._load_data()
        configs = data.get("configs", {}).get(model_id, [])
        return configs if isinstance(configs, list) else []

    def save_config(self, model_id: str, config: Dict[str, Any]):
        """Save a configuration for a model.

        Args:
            model_id: Model ID to bind config to
            config: Configuration dict with all run parameters
        """
        data = self._load_data()

        if "configs" not in data:
            data["configs"] = {}

        if model_id not in data["configs"]:
            data["configs"][model_id] = []

        # Add timestamp
        config["created_at"] = datetime.now().isoformat()

        # Append config
        data["configs"][model_id].append(config)

        self._save_data(data)

    def delete_config(self, model_id: str, config_index: int) -> bool:
        """Delete a saved configuration.

        Args:
            model_id: Model ID
            config_index: Index of config to delete (0-based)

        Returns:
            True if deleted, False if not found
        """
        data = self._load_data()

        if model_id not in data.get("configs", {}):
            return False

        configs = data["configs"][model_id]
        if config_index < 0 or config_index >= len(configs):
            return False

        configs.pop(config_index)
        self._save_data(data)
        return True

    def get_config(self, model_id: str, config_index: int) -> Optional[Dict[str, Any]]:
        """Get a specific saved configuration.

        Args:
            model_id: Model ID
            config_index: Index of config to get (0-based)

        Returns:
            Config dict or None if not found
        """
        configs = self.list_configs(model_id)
        if config_index < 0 or config_index >= len(configs):
            return None
        return configs[config_index]
