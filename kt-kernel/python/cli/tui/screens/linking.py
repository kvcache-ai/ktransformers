"""
Linking screens.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer, Center
from textual.widgets import Button, Checkbox, DataTable, Input, Label, RichLog, Select, Static
from textual.screen import ModalScreen
from textual.worker import Worker, WorkerState


class LinkModelsScreen(ModalScreen):
    """Modal screen for linking GPU/CPU models"""

    CSS_PATH = str(Path(__file__).parent.parent / "styles" / "linking.tcss")

    def __init__(self, source_model, target_type: str):
        """
        Args:
            source_model: The model being configured
            target_type: "gpu" or "cpu" - what type of models to link to
        """
        super().__init__()
        self.source_model = source_model
        self.target_type = target_type
        self.checkboxes = {}  # model_id -> Checkbox widget

    def _detect_amx_quant_type(self, model_path: str) -> str:
        """
        Detect AMX quantization type from config.json metadata or model path

        Priority:
        1. Read from config.json amx_quantization.method (new format)
        2. Fallback to path-based detection (legacy)

        Returns:
            "amx-int4", "amx-int8", "amx-awq", "amx-moe_int4", "amx-moe_int8", or "amx"
        """
        from pathlib import Path
        import json

        path = Path(model_path)

        # Priority 1: Read from config.json
        config_path = path / "config.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)

                # Check for amx_quantization metadata
                amx_meta = config.get("amx_quantization")
                if amx_meta and amx_meta.get("converted"):
                    method = amx_meta.get("method", "").lower()
                    if method:
                        # Map method to display format
                        method_map = {
                            "int4": "amx-int4",
                            "int8": "amx-int8",
                            "awq": "amx-awq",
                            "moe_int4": "amx-moe-int4",
                            "moe_int8": "amx-moe-int8",
                        }
                        return method_map.get(method, f"amx-{method}")
            except Exception:
                pass  # Fallback to path detection

        # Priority 2: Fallback to path-based detection (legacy)
        path_str = str(path).upper()

        if "INT4" in path_str or "AMXINT4" in path_str:
            return "amx-int4"
        elif "INT8" in path_str or "AMXINT8" in path_str:
            return "amx-int8"
        elif "AWQ" in path_str:
            return "amx-awq"

        # Default to int4 as it's the most common
        return "amx-int4"

    def compose(self) -> ComposeResult:
        from kt_kernel.cli.utils.user_model_registry import UserModelRegistry
        from kt_kernel.cli.commands.model import is_amx_weights

        registry = UserModelRegistry()
        all_models = registry.list_models()

        # Filter to get target type models
        target_models = []
        for model in all_models:
            if model.id == self.source_model.id:
                continue  # Skip self

            if self.target_type == "gpu":
                # Looking for GPU models (safetensors non-AMX)
                if model.format == "safetensors":
                    is_amx, _ = is_amx_weights(model.path)
                    if not is_amx:
                        target_models.append(model)
            elif self.target_type == "cpu":
                # Looking for CPU models (AMX or GGUF)
                if model.format == "gguf":
                    target_models.append(model)
                elif model.format == "safetensors":
                    is_amx, _ = is_amx_weights(model.path)
                    if is_amx:
                        target_models.append(model)

        # Get currently linked model IDs
        current_links = self.source_model.gpu_model_ids or []

        with Container(id="link-dialog"):
            title = f"Link {self.target_type.upper()} Models"
            yield Label(f"[bold cyan]{title}[/bold cyan]")
            yield Label(f"Source: {self.source_model.name}")
            yield Label("")

            with ScrollableContainer(id="link-content"):
                if not target_models:
                    yield Static(f"[yellow]No {self.target_type.upper()} models found[/yellow]")
                else:
                    for model in target_models:
                        # Check if currently linked
                        is_linked = model.id in current_links

                        # Detect format display
                        if model.format == "safetensors":
                            # Check if it's AMX
                            is_amx, _ = is_amx_weights(model.path)
                            if is_amx:
                                format_display = self._detect_amx_quant_type(model.path)
                            else:
                                format_display = model.format
                        else:
                            format_display = model.format

                        checkbox = Checkbox(f"{model.name} ({format_display})", value=is_linked, id=f"check-{model.id}")
                        self.checkboxes[model.id] = checkbox
                        yield checkbox

            with Horizontal(id="link-buttons"):
                yield Button("Save", id="btn-save", variant="success")
                yield Button("Cancel", id="btn-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-save":
            # Collect selected model IDs
            selected_ids = []
            for model_id, checkbox in self.checkboxes.items():
                if checkbox.value:
                    selected_ids.append(model_id)

            self.dismiss(selected_ids)
        elif event.button.id == "btn-cancel":
            self.dismiss(None)
