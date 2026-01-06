"""
User Model Registry

Manages user-registered models in ~/.ktransformers/user_models.yaml
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml


# Constants
USER_MODELS_FILE = Path.home() / ".ktransformers" / "user_models.yaml"
REGISTRY_VERSION = "1.0"


@dataclass
class UserModel:
    """Represents a user-registered model"""

    name: str  # User-editable name (default: folder name)
    path: str  # Absolute path to model directory
    format: str  # "safetensors" | "gguf"
    id: Optional[str] = None  # Unique UUID for this model (auto-generated if None)
    repo_type: Optional[str] = None  # "huggingface" | "modelscope" | None
    repo_id: Optional[str] = None  # e.g., "deepseek-ai/DeepSeek-V3"
    sha256_status: str = "not_checked"  # "not_checked" | "checking" | "passed" | "failed" | "no_repo"
    gpu_model_ids: Optional[List[str]] = None  # For llamafile/AMX: list of GPU model UUIDs to run with
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_verified: Optional[str] = None  # ISO format datetime

    def __post_init__(self):
        """Ensure ID is set after initialization"""
        if self.id is None:
            import uuid

            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserModel":
        """Create from dictionary loaded from YAML"""
        return cls(**data)

    def path_exists(self) -> bool:
        """Check if model path still exists"""
        return Path(self.path).exists()


class UserModelRegistry:
    """Manages the user model registry"""

    def __init__(self, registry_file: Optional[Path] = None):
        """
        Initialize the registry

        Args:
            registry_file: Path to the registry YAML file (default: USER_MODELS_FILE)
        """
        self.registry_file = registry_file or USER_MODELS_FILE
        self.models: List[UserModel] = []
        self.version = REGISTRY_VERSION

        # Ensure directory exists
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing registry
        self.load()

    def load(self) -> None:
        """Load models from YAML file"""
        if not self.registry_file.exists():
            # Initialize empty registry
            self.models = []
            self.save()  # Create the file
            return

        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not data:
                self.models = []
                return

            # Load version
            self.version = data.get("version", REGISTRY_VERSION)

            # Load models
            models_data = data.get("models", [])
            self.models = [UserModel.from_dict(m) for m in models_data]

            # Migrate: ensure all models have UUIDs (for backward compatibility)
            needs_save = False
            for model in self.models:
                if model.id is None:
                    import uuid

                    model.id = str(uuid.uuid4())
                    needs_save = True

            if needs_save:
                self.save()

        except Exception as e:
            raise RuntimeError(f"Failed to load user model registry: {e}")

    def save(self) -> None:
        """Save models to YAML file"""
        data = {"version": self.version, "models": [m.to_dict() for m in self.models]}

        try:
            with open(self.registry_file, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        except Exception as e:
            raise RuntimeError(f"Failed to save user model registry: {e}")

    def add_model(self, model: UserModel) -> None:
        """
        Add a model to the registry

        Args:
            model: UserModel instance to add

        Raises:
            ValueError: If a model with the same name already exists
        """
        if self.check_name_conflict(model.name):
            raise ValueError(f"Model with name '{model.name}' already exists")

        self.models.append(model)
        self.save()

    def remove_model(self, name: str) -> bool:
        """
        Remove a model from the registry

        Args:
            name: Name of the model to remove

        Returns:
            True if model was removed, False if not found
        """
        original_count = len(self.models)
        self.models = [m for m in self.models if m.name != name]

        if len(self.models) < original_count:
            self.save()
            return True
        return False

    def update_model(self, name: str, updates: Dict[str, Any]) -> bool:
        """
        Update a model's attributes

        Args:
            name: Name of the model to update
            updates: Dictionary of attributes to update

        Returns:
            True if model was updated, False if not found
        """
        model = self.get_model(name)
        if not model:
            return False

        # Update attributes
        for key, value in updates.items():
            if hasattr(model, key):
                setattr(model, key, value)

        self.save()
        return True

    def get_model(self, name: str) -> Optional[UserModel]:
        """
        Get a model by name

        Args:
            name: Name of the model

        Returns:
            UserModel instance or None if not found
        """
        for model in self.models:
            if model.name == name:
                return model
        return None

    def get_model_by_id(self, model_id: str) -> Optional[UserModel]:
        """
        Get a model by its unique ID

        Args:
            model_id: UUID of the model

        Returns:
            UserModel instance or None if not found
        """
        for model in self.models:
            if model.id == model_id:
                return model
        return None

    def list_models(self) -> List[UserModel]:
        """
        List all models

        Returns:
            List of all UserModel instances
        """
        return self.models.copy()

    def find_by_path(self, path: str) -> Optional[UserModel]:
        """
        Find a model by its path

        Args:
            path: Model directory path

        Returns:
            UserModel instance or None if not found
        """
        # Normalize paths for comparison
        search_path = str(Path(path).resolve())

        for model in self.models:
            model_path = str(Path(model.path).resolve())
            if model_path == search_path:
                return model
        return None

    def check_name_conflict(self, name: str, exclude_name: Optional[str] = None) -> bool:
        """
        Check if a name conflicts with existing models

        Args:
            name: Name to check
            exclude_name: Optional name to exclude from check (for rename operations)

        Returns:
            True if conflict exists, False otherwise
        """
        for model in self.models:
            if model.name == name and model.name != exclude_name:
                return True
        return False

    def refresh_status(self) -> Dict[str, List[str]]:
        """
        Check all models and identify missing ones

        Returns:
            Dictionary with 'valid' and 'missing' lists of model names
        """
        valid = []
        missing = []

        for model in self.models:
            if model.path_exists():
                valid.append(model.name)
            else:
                missing.append(model.name)

        return {"valid": valid, "missing": missing}

    def get_model_count(self) -> int:
        """Get total number of registered models"""
        return len(self.models)

    def suggest_name(self, base_name: str) -> str:
        """
        Suggest a unique name based on base_name

        Args:
            base_name: Base name to derive from

        Returns:
            A unique name (may have suffix like -2, -3 etc.)
        """
        if not self.check_name_conflict(base_name):
            return base_name

        counter = 2
        while True:
            candidate = f"{base_name}-{counter}"
            if not self.check_name_conflict(candidate):
                return candidate
            counter += 1
