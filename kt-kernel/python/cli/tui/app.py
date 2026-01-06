"""
KTransformers Interactive Model Manager

A Textual-based TUI for managing AI models.
"""

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer, Center
from textual.widgets import Header, Footer, DataTable, Static, Label, Button, Input, Select, Checkbox
from textual.screen import ModalScreen
from textual.worker import Worker, WorkerState
from textual.message import Message
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
import asyncio

# Local - Screen modules
from .screens.dialogs import ConfirmDialog, PathInputScreen, PathSelectScreen
from .screens.model_info import (
    InfoScreen,
    EditModelScreen,
    RenameInputScreen,
    RepoEditScreen,
    RepoInputScreen,
    AutoRepoSelectScreen,
)
from .screens.config import QuantConfigScreen, RunConfigScreen
from .screens.progress import QuantProgressScreen, DownloadProgressScreen
from .screens.download import DownloadScreen
from .screens.linking import LinkModelsScreen
from .screens.system import DoctorScreen, SettingsScreen


# InfoScreen, RepoEditScreen, RepoInputScreen, RenameInputScreen,
# AutoRepoSelectScreen, EditModelScreen moved to screens/model_info.py


# QuantConfigScreen moved to screens/config.py


# QuantProgressScreen moved to screens/

# DoctorScreen moved to screens/

# LinkModelsScreen moved to screens/

# DownloadScreen moved to screens/

# DownloadProgressScreen moved to screens/


class SettingsScreen(ModalScreen):
    """Modal screen for application settings"""

    CSS = """
    SettingsScreen {
        align: center middle;
    }

    #settings-dialog {
        width: 90;
        height: auto;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    #settings-content {
        width: 100%;
        height: auto;
        max-height: 30;
        overflow-y: auto;
        padding: 1 0;
    }

    #settings-actions {
        width: 100%;
        height: auto;
        padding: 1 0;
    }

    #settings-actions Button {
        margin: 0 1;
    }

    #settings-buttons {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1 0 0 0;
    }

    #settings-buttons Button {
        margin: 0 1;
    }
    """

    def __init__(self):
        super().__init__()

    def compose(self) -> ComposeResult:
        from kt_kernel.cli.config.settings import get_settings

        settings = get_settings()
        model_paths = settings.get_model_paths()

        with Container(id="settings-dialog"):
            yield Label("[bold cyan]⚙ Settings[/bold cyan]")
            yield Label("")

            with Container(id="settings-content"):
                yield Label("[bold]Model Paths:[/bold]")
                for i, path in enumerate(model_paths, 1):
                    exists = "✓" if path.exists() else "✗"
                    yield Label(f"  {i}. {exists} {path}")

                yield Label("")
                yield Label("[dim]Config: {0}[/dim]".format(settings.config_path))

            with Horizontal(id="settings-actions"):
                yield Button("➕ Add Path", id="btn-add-path", variant="success")
                yield Button("✏ Edit Path", id="btn-edit-path")
                yield Button("➖ Remove Path", id="btn-remove-path", variant="error")

            yield Label("")

            with Horizontal(id="settings-buttons"):
                yield Button("🔄 Force Refresh All", id="btn-force-refresh", variant="warning")
                yield Button("🔍 Auto-Repo", id="btn-auto-repo", variant="success")
                yield Button("✓ Verify-All", id="btn-verify-all", variant="primary")
                yield Button("Close", id="btn-close")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-close":
            self.dismiss(None)
        elif event.button.id == "btn-force-refresh":
            self.dismiss("force_refresh")
        elif event.button.id == "btn-add-path":
            self.dismiss("add_path")
        elif event.button.id == "btn-edit-path":
            self.dismiss("edit_path")
        elif event.button.id == "btn-remove-path":
            self.dismiss("remove_path")
        elif event.button.id == "btn-auto-repo":
            self.dismiss("auto_repo")
        elif event.button.id == "btn-verify-all":
            self.dismiss("verify_all")

    def on_key(self, event) -> None:
        """Handle key presses"""
        if event.key == "escape":
            self.dismiss(None)


# EditModelScreen moved to screens/model_info.py


class ModelManagerApp(App):
    """Interactive Model Manager TUI Application"""

    class VerifyProgress(Message):
        """Message sent during verification progress"""

        def __init__(
            self, model_id: str, progress_msg: str, total: int = None, current: int = None, is_milestone: bool = False
        ) -> None:
            self.model_id = model_id
            self.progress_msg = progress_msg
            self.total = total
            self.current = current
            self.is_milestone = is_milestone
            super().__init__()

    class VerifyComplete(Message):
        """Message sent when verification completes"""

        def __init__(self, model_id: str, result: Dict[str, Any]) -> None:
            self.model_id = model_id
            self.result = result
            super().__init__()

    CSS_PATH = "app.tcss"
    TITLE = "KTransformers Model Manager"

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("r", "refresh", "Refresh"),
        Binding("s", "scan", "Scan"),
        Binding("c", "settings", "Settings"),
        Binding("d", "download", "Download"),
        Binding("o", "doctor", "Doctor"),
        Binding("enter", "run", "Run Model"),
        Binding("1", "action_1", "", show=False),
        Binding("2", "action_2", "", show=False),
        Binding("3", "action_3", "", show=False),
        Binding("4", "action_4", "", show=False),
        Binding("5", "action_5", "", show=False),
        Binding("6", "action_6", "", show=False),
        Binding("7", "action_7", "", show=False),
        Binding("tab", "next_table", "Next Section", show=False),
        Binding("shift+tab", "prev_table", "Prev Section", show=False),
    ]

    def __init__(self):
        super().__init__()
        self.selected_model = None
        self.moe_results = {}
        self.active_table_id = "gpu-table"
        self.linked_cpu_rows = []  # Track which CPU rows are linked to selected GPU
        self.info_screen_open = False  # Track if info screen is currently displayed

    def compose(self) -> ComposeResult:
        """Create child widgets for the app"""
        yield Header(show_clock=True)

        with ScrollableContainer(id="main-container"):
            # MoE GPU Models Section
            with Vertical(id="moe-section", classes="model-section"):
                yield Label("MoE Models (GPU)", classes="section-header")
                yield DataTable(id="gpu-table", classes="model-table", cursor_type="row")

            # GGUF Models Section
            with Vertical(id="gguf-section", classes="model-section"):
                yield Label("GGUF Models (Llamafile)", classes="section-header")
                yield DataTable(id="gguf-table", classes="model-table", cursor_type="row")

            # AMX Models Section
            with Vertical(id="amx-section", classes="model-section"):
                yield Label("AMX Models (CPU)", classes="section-header")
                yield DataTable(id="amx-table", classes="model-table", cursor_type="row")

            # Non-MoE GPU Models Section (Last, informational only)
            with Vertical(id="non-moe-section", classes="model-section"):
                yield Label("Non-MoE GPU Models (Not Supported)", classes="section-header")
                yield DataTable(id="non-moe-table", classes="model-table", cursor_type="none")

        # Verification progress widget
        yield Static("", id="verify-progress", classes="hidden")

        # Action Panel
        with Container(id="action-panel"):
            yield Static("No model selected", id="selected-info")
            yield Static("")  # Empty line for spacing
            with Horizontal(id="action-buttons"):
                yield Static("[1] Run", classes="action-btn")
                yield Static("[2] Edit", classes="action-btn")
                yield Static("[3] Verify", classes="action-btn")
                yield Static("[4] Quant", classes="action-btn")
                yield Static("[5] Remove", classes="action-btn")
                yield Static("[6] Info", classes="action-btn")
                yield Static("[7] Links", classes="action-btn")

        yield Footer()

    def on_mount(self) -> None:
        """Initialize when app starts"""
        self.title = "KTransformers Model Manager"
        self.sub_title = "Loading models..."

        # Set up tables
        self.setup_tables()

        # Load models
        self.load_models()

    def setup_tables(self) -> None:
        """Configure DataTable columns"""
        # MoE GPU table
        gpu_table = self.query_one("#gpu-table", DataTable)
        # Exps = Experts×Layers (e.g., 256×64 = 256 experts per layer, 64 layers)
        gpu_table.add_columns("Name", "Total", "Exps×Layers", "1-Expert", "Skeleton", "Repo", "SHA256", "Path")
        gpu_table.cursor_type = "row"
        gpu_table.zebra_stripes = True
        gpu_table.focus()

        # GGUF table
        gguf_table = self.query_one("#gguf-table", DataTable)
        gguf_table.add_columns("Name", "Size", "GPU Links", "Repo", "SHA256", "Path")
        gguf_table.cursor_type = "row"
        gguf_table.zebra_stripes = True

        # AMX table
        amx_table = self.query_one("#amx-table", DataTable)
        amx_table.add_columns("Name", "NUMA", "Size", "GPU Links", "Path")
        amx_table.cursor_type = "row"
        amx_table.zebra_stripes = True

        # Non-MoE GPU table (last, not selectable)
        non_moe_table = self.query_one("#non-moe-table", DataTable)
        non_moe_table.add_columns("Name", "Size", "Repo", "SHA256", "Reason", "Path")
        non_moe_table.cursor_type = "none"  # Not selectable
        non_moe_table.zebra_stripes = True

    def load_models(self) -> None:
        """Load models from registry"""
        try:
            from kt_kernel.cli.utils.user_model_registry import UserModelRegistry
            from kt_kernel.cli.commands.model import is_amx_weights
            from kt_kernel.cli.utils.model_scanner import format_size
            from kt_kernel.cli.utils.analyze_moe_model import analyze_moe_model

            registry = UserModelRegistry()
            models = registry.list_models()

            # Categorize models
            gpu_models = []
            gguf_models = []
            amx_models = []

            for model in models:
                if not model.path_exists():
                    continue

                if model.format == "gguf":
                    gguf_models.append(model)
                elif model.format == "safetensors":
                    is_amx, numa_count = is_amx_weights(model.path)
                    if is_amx:
                        amx_models.append((model, numa_count))
                    else:
                        gpu_models.append(model)

            # Pre-analyze GPU models and categorize into MoE and Non-MoE
            moe_models = []
            non_moe_models = []

            if gpu_models:
                self.sub_title = f"Analyzing {len(gpu_models)} GPU models..."
                for i, model in enumerate(gpu_models):
                    # Update progress in subtitle
                    self.sub_title = f"Analyzing GPU models... ({i+1}/{len(gpu_models)})"
                    try:
                        result = analyze_moe_model(model.path, use_cache=True)
                        if result and result.get("num_experts", 0) > 0:
                            # Valid MoE model
                            self.moe_results[model.id] = result
                            moe_models.append(model)
                        else:
                            # Not a MoE model or has 0 experts
                            reason = "Not a MoE model"
                            non_moe_models.append((model, reason))
                    except Exception as e:
                        # Analysis failed
                        reason = f"Analysis failed"
                        non_moe_models.append((model, reason))

            # Populate tables
            self.populate_gpu_table(moe_models)
            self.populate_non_moe_table(non_moe_models)
            self.populate_gguf_table(gguf_models)
            self.populate_amx_table(amx_models)

            # Update subtitle
            total = len(moe_models) + len(non_moe_models) + len(gguf_models) + len(amx_models)
            self.sub_title = f"Total: {total} models | MoE: {len(moe_models)} | Non-MoE: {len(non_moe_models)} | GGUF: {len(gguf_models)} | AMX: {len(amx_models)}"

            # Auto-select first model in GPU table
            gpu_table = self.query_one("#gpu-table", DataTable)
            if gpu_table.row_count > 0:
                gpu_table.move_cursor(row=0)

            # Check if no models found - auto scan on first run
            if total == 0 and not hasattr(self, "_initial_scan_done"):
                self._initial_scan_done = True
                # Check if this is truly first run (no paths configured or only default empty path)
                from kt_kernel.cli.config.settings import get_settings, DEFAULT_MODELS_DIR

                settings = get_settings()
                model_paths = settings.get_model_paths()

                # Check if we only have the default path and it doesn't exist (first run)
                is_first_run = not model_paths or (
                    len(model_paths) == 1
                    and str(model_paths[0]) == str(DEFAULT_MODELS_DIR)
                    and not model_paths[0].exists()
                )

                if is_first_run:
                    # First run - automatically scan all disks in background
                    self.notify(
                        "🌍 First run detected - scanning all disks for models...", severity="information", timeout=5
                    )
                    self.run_worker(self._global_scan_async(), name="global-scan", group="scan", exclusive=True)
                else:
                    # Paths exist but no models - just notify
                    self.notify(
                        "No models found. Add models or scan paths in Settings (press 's' or '7')",
                        severity="warning",
                        timeout=5,
                    )

        except Exception as e:
            self.sub_title = f"Error loading models: {e}"

    def populate_gpu_table(self, models: List) -> None:
        """Populate GPU models table"""
        from kt_kernel.cli.utils.model_scanner import format_size

        table = self.query_one("#gpu-table", DataTable)
        table.clear()

        for model in models:
            # Calculate total size
            try:
                model_path = Path(model.path)
                files = list(model_path.glob("*.safetensors"))
                total_size = sum(f.stat().st_size for f in files if f.exists())
                size_str = format_size(total_size)
            except:
                size_str = "-"

            # Get repo info with prefix
            if model.repo_id:
                prefix = "hf:" if model.repo_type == "huggingface" else "ms:" if model.repo_type == "modelscope" else ""
                repo_str = f"{prefix}{model.repo_id}"
            else:
                repo_str = "-"

            # Get SHA256 status
            sha256_map = {"passed": "✓", "failed": "✗", "not_checked": "-", "checking": "...", "no_repo": "-"}
            sha256_str = sha256_map.get(model.sha256_status, "-")

            # MoE info from pre-analysis
            moe_result = self.moe_results.get(model.id)
            if moe_result and "error" not in moe_result:
                # Success - show MoE data
                num_experts = moe_result.get("num_experts", 0)
                num_layers = moe_result.get("num_layers", 0)
                # Format: 256×64 (experts per layer × number of layers)
                exps = f"{num_experts}×{num_layers}"

                expert_size_gb = moe_result.get("single_expert_size_gb", 0)
                expert_size = format_size(expert_size_gb * 1024**3)

                skeleton_gb = moe_result.get("rest_size_gb", 0)
                skeleton = format_size(skeleton_gb * 1024**3)
            elif moe_result and "error" in moe_result:
                # Failed analysis
                exps = "✗"
                expert_size = "Failed"
                skeleton = "-"
            else:
                # No analysis result
                exps = "-"
                expert_size = "-"
                skeleton = "-"

            table.add_row(
                model.name,
                size_str,  # Total
                exps,  # Exps×Layers
                expert_size,  # 1-Expert
                skeleton,  # Skeleton
                repo_str,  # Repo
                sha256_str,  # SHA256
                model.path,  # Path
                key=model.id,
            )

    def populate_non_moe_table(self, models: List[Tuple]) -> None:
        """Populate Non-MoE GPU models table"""
        from kt_kernel.cli.utils.model_scanner import format_size

        table = self.query_one("#non-moe-table", DataTable)
        table.clear()

        for model, reason in models:
            # Calculate total size
            try:
                model_path = Path(model.path)
                files = list(model_path.glob("*.safetensors"))
                total_size = sum(f.stat().st_size for f in files if f.exists())
                size_str = format_size(total_size)
            except:
                size_str = "-"

            # Get repo info with prefix
            if model.repo_id:
                prefix = "hf:" if model.repo_type == "huggingface" else "ms:" if model.repo_type == "modelscope" else ""
                repo_str = f"{prefix}{model.repo_id}"
            else:
                repo_str = "-"

            # Get SHA256 status
            sha256_map = {"passed": "✓", "failed": "✗", "not_checked": "-", "checking": "...", "no_repo": "-"}
            sha256_str = sha256_map.get(model.sha256_status, "-")

            table.add_row(model.name, size_str, repo_str, sha256_str, reason, model.path, key=model.id)  # Path

    def populate_gguf_table(self, models: List) -> None:
        """Populate GGUF models table"""
        from kt_kernel.cli.utils.model_scanner import format_size

        table = self.query_one("#gguf-table", DataTable)
        table.clear()

        from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

        registry = UserModelRegistry()

        for model in models:
            # Calculate size
            try:
                model_path = Path(model.path)
                files = list(model_path.glob("*.gguf"))
                total_size = sum(f.stat().st_size for f in files if f.exists())
                size_str = format_size(total_size)
            except:
                size_str = "-"

            # Get GPU links
            gpu_links = "-"
            if model.gpu_model_ids:
                gpu_names = []
                for gpu_id in model.gpu_model_ids:
                    gpu_obj = registry.get_model_by_id(gpu_id)
                    if gpu_obj:
                        gpu_names.append(gpu_obj.name)
                if gpu_names:
                    gpu_links = ", ".join(gpu_names[:2])
                    if len(gpu_names) > 2:
                        gpu_links += f" +{len(gpu_names) - 2}"

            # Get repo info with prefix
            if model.repo_id:
                prefix = "hf:" if model.repo_type == "huggingface" else "ms:" if model.repo_type == "modelscope" else ""
                repo_str = f"{prefix}{model.repo_id}"
            else:
                repo_str = "-"

            # Get SHA256 status
            sha256_map = {"passed": "✓", "failed": "✗", "not_checked": "-", "checking": "...", "no_repo": "-"}
            sha256_str = sha256_map.get(model.sha256_status, "-")

            table.add_row(
                model.name,
                size_str,
                gpu_links,
                repo_str,  # Repo
                sha256_str,  # SHA256
                model.path,  # Path
                key=model.id,
            )

    def populate_amx_table(self, models: List[Tuple]) -> None:
        """Populate AMX models table"""
        from kt_kernel.cli.utils.model_scanner import format_size

        table = self.query_one("#amx-table", DataTable)
        table.clear()

        from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

        registry = UserModelRegistry()

        for model, numa_count in models:
            # Calculate size
            try:
                model_path = Path(model.path)
                files = list(model_path.glob("*.safetensors"))
                total_size = sum(f.stat().st_size for f in files if f.exists())
                size_str = format_size(total_size)
            except:
                size_str = "-"

            # Get GPU links
            gpu_links = "-"
            if model.gpu_model_ids:
                gpu_names = []
                for gpu_id in model.gpu_model_ids:
                    gpu_obj = registry.get_model_by_id(gpu_id)
                    if gpu_obj:
                        gpu_names.append(gpu_obj.name)
                if gpu_names:
                    gpu_links = ", ".join(gpu_names[:2])
                    if len(gpu_names) > 2:
                        gpu_links += f" +{len(gpu_names) - 2}"

            table.add_row(model.name, f"{numa_count} NUMA", size_str, gpu_links, model.path, key=model.id)  # Path

    def on_model_manager_app_verify_progress(self, message: VerifyProgress) -> None:
        """Handle verification progress message"""
        try:
            # Show all progress messages in the progress widget
            progress_widget = self.query_one("#verify-progress", Static)

            # Remove "hidden" class to show the widget
            progress_widget.remove_class("hidden")

            # Build progress display
            lines = []

            # Line 1: Title
            lines.append("🔍 [bold cyan]Verifying Model[/bold cyan]")

            # Line 2: Progress message
            msg = message.progress_msg

            # If we have total and current, show progress bar
            if message.total and message.current is not None:
                percentage = int((message.current / message.total) * 100)
                bar_width = 30
                filled = int((message.current / message.total) * bar_width)
                bar = "█" * filled + "░" * (bar_width - filled)

                # Extract the main message (remove [X/Y] prefix if present)
                clean_msg = msg
                if msg.startswith("[") and "]" in msg:
                    # Keep the [X/Y] part
                    bracket_end = msg.index("]")
                    counter = msg[: bracket_end + 1]
                    rest = msg[bracket_end + 1 :].strip()

                    # Show full message without truncation
                    lines.append(f"{counter} {percentage}% {bar}")
                    if rest:
                        lines.append(f"  {rest}")
                else:
                    lines.append(f"{percentage}% {bar}")
                    # Show full message without truncation
                    lines.append(f"  {clean_msg}")
            else:
                # No progress info, just show full message
                lines.append(f"  {msg}")

            # Update widget
            progress_widget.update("\n".join(lines))

            # Log for debugging
            self.log.info(f"Verify progress: {message.progress_msg} ({message.current}/{message.total})")
        except Exception as e:
            self.log.error(f"Error updating progress: {e}")

    def on_model_manager_app_verify_complete(self, message: VerifyComplete) -> None:
        """Handle verification completion message"""
        from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

        try:
            result = message.result
            registry = UserModelRegistry()

            # Get model by ID first
            model = registry.get_model_by_id(message.model_id)
            if not model:
                self.notify("Model not found", severity="error")
                self._hide_verify_progress()
                return

            # Update progress widget with final status
            progress_widget = self.query_one("#verify-progress", Static)

            if result["status"] == "passed":
                final_msg = f"✓ Verification Passed: {result['files_passed']}/{result['files_checked']} files OK"
                progress_widget.update(final_msg)
                # Update model with correct parameter format: name and dictionary
                registry.update_model(model.name, {"sha256_status": "passed"})
            elif result["status"] == "failed":
                failed_count = len(result.get("files_failed", []))
                final_msg = f"✗ Verification Failed: {failed_count} file(s) failed"
                progress_widget.update(final_msg)
                registry.update_model(model.name, {"sha256_status": "failed"})
            else:
                # Error status
                error_msg = result.get("error_message", "Unknown error")
                # Show full error message without truncation
                final_msg = f"✗ Verification Error: {error_msg}"
                progress_widget.update(final_msg)

            # Log completion
            self.log.info(f"Verification complete: {final_msg}")

            # Refresh to show updated status
            self.load_models()

            # Check if we're in verify-all (batch) mode
            if hasattr(self, "_verify_models_queue") and self._verify_models_queue:
                # Batch mode: continue with next model
                self.log.info(f"Verify-all mode: {len(self._verify_models_queue)} models remaining")
                next_model = self._verify_models_queue.pop(0)
                self._start_verification(next_model, is_batch=True)
            elif hasattr(self, "_verify_models_queue"):
                # Batch mode complete (queue exists but empty)
                self.notify("✓ Verification complete for all models", severity="information", timeout=5)
                # Clean up batch state
                delattr(self, "_verify_models_queue")
                delattr(self, "_verify_total_count")
                # Auto-hide after showing completion
                self.set_timer(3.0, self._hide_verify_progress)
            else:
                # Single verification mode - auto-hide after 3 seconds
                self.set_timer(3.0, self._hide_verify_progress)

        except Exception as e:
            self.log.error(f"Error in verify complete handler: {e}")
            self._hide_verify_progress()

            # If in verify-all mode and error occurred, continue with next
            if hasattr(self, "_verify_models_queue") and self._verify_models_queue:
                next_model = self._verify_models_queue.pop(0)
                self._start_verification(next_model, is_batch=True)

    def _hide_verify_progress(self) -> None:
        """Hide the verification progress widget"""
        try:
            progress_widget = self.query_one("#verify-progress", Static)
            progress_widget.add_class("hidden")
            self.log.info("Progress widget hidden")
        except Exception as e:
            self.log.warning(f"Could not hide progress widget: {e}")

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Handle row highlight (cursor movement) in any table"""
        try:
            from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

            # Close info screen if open when switching models
            if self.info_screen_open:
                # Pop the info screen
                self.pop_screen()
                self.info_screen_open = False

            model_id = event.row_key.value
            registry = UserModelRegistry()
            self.selected_model = registry.get_model_by_id(model_id)

            # Update which table is active
            self.active_table_id = event.data_table.id

            if self.selected_model:
                self.update_action_panel()

                # If GPU model selected, show linked CPU models
                if event.data_table.id == "gpu-table":
                    self.show_linked_cpu_models(model_id)
                else:
                    # Clear highlights if selecting non-GPU model
                    self.clear_cpu_highlights()
            else:
                # Model not found in registry - log warning
                self.log.warning(f"Model {model_id} not found in registry")
        except Exception as e:
            # Catch any errors to prevent crashes
            self.log.error(f"Error in row highlight handler: {e}")

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection (Enter key) - same as highlight"""
        # Trigger run action when user presses Enter
        self.action_run()

    def show_linked_cpu_models(self, gpu_model_id: str) -> None:
        """Highlight CPU models linked to selected GPU model"""
        # Safety check: ensure selected_model is set
        if not self.selected_model:
            return

        from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

        # Clear previous highlights
        self.clear_cpu_highlights()

        try:
            # Find CPU models that link to this GPU model
            registry = UserModelRegistry()
            all_models = registry.list_models()

            amx_table = self.query_one("#amx-table", DataTable)

            for model in all_models:
                if model.gpu_model_ids and gpu_model_id in model.gpu_model_ids:
                    # This CPU model is linked to the selected GPU model
                    # Find its row in the AMX table and highlight it
                    for row_key in amx_table.rows:
                        if row_key.value == model.id:
                            self.linked_cpu_rows.append(row_key)
                            # Update row to show link indicator
                            current_values = list(amx_table.get_row(row_key))
                            if len(current_values) >= 4:
                                # Add indicator to GPU Links column
                                current_values[3] = f"→ {self.selected_model.name}"
                                amx_table.update_cell(row_key, "GPU Links", current_values[3])
                            break
        except Exception as e:
            # Silently ignore errors in highlighting to avoid crashes
            self.log.warning(f"Error highlighting CPU models: {e}")

    def clear_cpu_highlights(self) -> None:
        """Clear CPU model highlights"""
        if not self.linked_cpu_rows:
            return

        try:
            from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

            registry = UserModelRegistry()
            amx_table = self.query_one("#amx-table", DataTable)

            for row_key in self.linked_cpu_rows:
                try:
                    # Restore original GPU links text
                    model_id = row_key.value
                    model = registry.get_model_by_id(model_id)
                    if model and model.gpu_model_ids:
                        gpu_names = []
                        for gpu_id in model.gpu_model_ids:
                            gpu_obj = registry.get_model_by_id(gpu_id)
                            if gpu_obj:
                                gpu_names.append(gpu_obj.name)
                        if gpu_names:
                            gpu_links = ", ".join(gpu_names[:2])
                            if len(gpu_names) > 2:
                                gpu_links += f" (+{len(gpu_names)-2})"
                            amx_table.update_cell(row_key, "GPU Links", gpu_links)
                        else:
                            amx_table.update_cell(row_key, "GPU Links", "-")
                    else:
                        amx_table.update_cell(row_key, "GPU Links", "-")
                except Exception:
                    # Silently skip errors for individual rows
                    pass

            self.linked_cpu_rows.clear()
        except Exception as e:
            # Catch any outer errors
            self.log.warning(f"Error clearing CPU highlights: {e}")
            self.linked_cpu_rows.clear()

    def update_action_panel(self) -> None:
        """Update action buttons based on selected model"""
        if not self.selected_model:
            return

        # Update selected info with repo status
        info = self.query_one("#selected-info", Static)
        model = self.selected_model

        # Build info string with repo status
        info_parts = [f"Selected: [cyan]{model.name}[/cyan] ({model.format})"]

        # Show repo status for GPU and GGUF models (both can be verified)
        if model.format in ["safetensors", "gguf"]:
            if model.repo_id:
                prefix = "hf:" if model.repo_type == "huggingface" else "ms:" if model.repo_type == "modelscope" else ""
                repo_display = f"{prefix}{model.repo_id}"
                info_parts.append(f"[dim]| Repo: {repo_display}[/dim]")
            else:
                info_parts.append(f"[dim yellow]| No repo (verification unavailable)[/dim yellow]")

        info.update(" ".join(info_parts))

        # Update action buttons based on model type
        actions = self.get_actions_for_model(self.selected_model)
        action_btns = self.query("#action-buttons > .action-btn")

        for i, btn in enumerate(action_btns):
            if i < len(actions):
                btn.update(f"[{i+1}] {actions[i]}")
            else:
                btn.update("")

    def get_actions_for_model(self, model) -> List[str]:
        """Get available actions for model type"""
        from kt_kernel.cli.commands.model import is_amx_weights

        if model.format == "gguf":
            # GGUF models can also be verified if they have repo
            verify_action = "Verify" if model.repo_id else "No Repo"
            return ["Link GPU", "Edit", verify_action, "Unlink", "Remove", "Info"]
        elif model.format == "safetensors":
            is_amx, _ = is_amx_weights(model.path)
            if is_amx:
                # AMX models cannot be verified (no repo support)
                return ["Link GPU", "Edit", "Unlink", "Remove", "Info", "Test"]
            else:
                # For GPU models, show Verify only if repo is set
                verify_action = "Verify" if model.repo_id else "No Repo"
                return ["Run", "Edit", verify_action, "Quant", "Remove", "Info", "Link CPU"]
        return []

    # Action handlers
    def action_quit(self) -> None:
        """Quit the app"""
        self.exit()

    def action_refresh(self) -> None:
        """Refresh model list"""
        self.sub_title = "Refreshing..."
        self.load_models()

    def action_run(self) -> None:
        """Run selected model (only for GPU MoE models)"""
        if not self.selected_model:
            self.notify("No model selected", severity="warning")
            return

        # Check if it's a GPU MoE model
        if self.selected_model.format != "safetensors":
            self.notify("Only GPU models can be run", severity="warning")
            return

        from kt_kernel.cli.commands.model import is_amx_weights

        is_amx, _ = is_amx_weights(self.selected_model.path)
        if is_amx:
            self.notify("AMX models cannot be run directly", severity="warning")
            return

        # Get MoE result for this model
        moe_result = self.moe_results.get(self.selected_model.id)
        if not moe_result:
            self.notify("This is not a MoE model", severity="warning")
            return

        # Show run configuration screen
        self.push_screen(RunConfigScreen(self.selected_model, moe_result), callback=self._on_run_config)

    def _on_run_config(self, config) -> None:
        """Handle run configuration result"""
        if config is None:
            return

        # TODO: Implement actual run logic
        gpu_model = config["gpu_model"]
        cpu_model_id = config["cpu_model_id"]
        gpu_experts = config["gpu_experts"]
        cpu_threads = config["cpu_threads"]
        numa_nodes = config["numa_nodes"]
        total_tokens = config["total_tokens"]

        self.notify(
            f"Run config ready: GPU={gpu_model.name}, Experts={gpu_experts}, "
            f"Threads={cpu_threads}, NUMA={numa_nodes}, Tokens={total_tokens}",
            severity="information",
        )

    def action_edit(self) -> None:
        """Edit selected model"""
        if not self.selected_model:
            return

        moe_info = self.moe_results.get(self.selected_model.id, {})

        # Show edit dialog
        self.push_screen(EditModelScreen(self.selected_model, moe_info), callback=self._on_edit_choice)

    def _on_edit_choice(self, result) -> None:
        """Handle edit dialog result"""
        if result == "rename":
            self.edit_rename()
        elif result == "repo":
            self.edit_repo()
        elif result == "delete":
            self.edit_delete()

    def edit_rename(self) -> None:
        """Handle rename action"""
        # Show rename input dialog
        self.push_screen(RenameInputScreen(self.selected_model.name), callback=self._on_rename_done)

    def _on_rename_done(self, new_name) -> None:
        """Handle rename completion"""
        from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

        if not new_name:
            return

        registry = UserModelRegistry()
        old_name = self.selected_model.name

        # Check for name conflicts
        if registry.check_name_conflict(new_name, exclude_name=old_name):
            self.notify(f"Model name '{new_name}' already exists", severity="error")
            return

        # Update the model name
        success = registry.update_model(old_name, {"name": new_name})

        if success:
            self.notify(f"✓ Renamed '{old_name}' to '{new_name}'", severity="information")
            self.load_models()
        else:
            self.notify(f"Failed to rename model", severity="error")

    def edit_repo(self) -> None:
        """Handle repo edit action"""
        # Show repo edit dialog
        self.push_screen(RepoEditScreen(self.selected_model), callback=self._on_repo_edit_done)

    def _on_repo_edit_done(self, result) -> None:
        """Handle repo edit result"""
        from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

        if not result:
            return

        registry = UserModelRegistry()
        model_name = self.selected_model.name

        if result["action"] == "remove":
            # Remove repo info
            updates = {"repo_type": None, "repo_id": None, "sha256_status": "no_repo"}
            registry.update_model(model_name, updates)
            self.notify(f"✓ Removed repo info for {model_name}", severity="information")
            self.load_models()

        elif result["action"] == "set":
            # Set repo info
            updates = {"repo_type": result["repo_type"], "repo_id": result["repo_id"], "sha256_status": "not_checked"}
            registry.update_model(model_name, updates)
            prefix = (
                "hf:" if result["repo_type"] == "huggingface" else "ms:" if result["repo_type"] == "modelscope" else ""
            )
            self.notify(f"✓ Updated repo: {prefix}{result['repo_id']}", severity="information")
            self.load_models()

    def edit_delete(self) -> None:
        """Handle delete action"""
        from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

        registry = UserModelRegistry()

        # Show confirmation dialog
        self.push_screen(
            ConfirmDialog(
                f"Delete model '{self.selected_model.name}'?",
                "This will remove the model from the registry (files will NOT be deleted).",
            ),
            callback=self._on_delete_confirmed,
        )

    def _on_delete_confirmed(self, confirm) -> None:
        """Handle delete confirmation result"""
        from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

        if confirm:
            registry = UserModelRegistry()
            model_name = self.selected_model.name
            registry.remove_model(model_name)
            self.notify(f"✓ Deleted {model_name}", severity="information")
            self.load_models()

    def _start_verification(self, model, is_batch=False) -> None:
        """
        Start verification for a model (shared logic for single and batch verify)

        Args:
            model: Model to verify
            is_batch: Whether this is part of batch verification
        """
        try:
            # Show verification progress widget
            progress_widget = self.query_one("#verify-progress", Static)
            progress_widget.remove_class("hidden")

            # Update progress message based on mode
            if is_batch and hasattr(self, "_verify_models_queue"):
                # Batch mode: show progress in queue
                remaining = len(self._verify_models_queue)
                total = self._verify_total_count
                current = total - remaining
                progress_widget.update(
                    f"🔍 [bold cyan]Verify-All Progress[/bold cyan]\n"
                    f"  [{current}/{total}] Verifying: {model.name}..."
                )
            else:
                # Single model mode
                progress_widget.update(
                    "🔍 [bold cyan]Verifying Model[/bold cyan]\n" f"  Initializing verification for {model.name}..."
                )

            # Log
            self.log.info(f"Starting verification for {model.name}")

            # Run verification in background worker (same for both modes)
            self.run_worker(
                self.verify_model_async(model),
                name=f"verify-{model.id}",
                group="verification",
                description=f"Verifying {model.name}",
            )
        except Exception as e:
            self.notify(f"Error starting verification: {e}", severity="error")
            self.log.error(f"Verification error: {e}")

            # If in batch mode, continue with next
            if is_batch and hasattr(self, "_verify_models_queue") and self._verify_models_queue:
                model = self._verify_models_queue.pop(0)
                self._start_verification(model, is_batch=True)

    def action_verify(self) -> None:
        """Verify selected model"""
        if not self.selected_model:
            self.notify("No model selected", severity="warning")
            return

        # Verify requires repo_id
        if not self.selected_model.repo_id:
            self.notify(
                f"Cannot verify '{self.selected_model.name}': No repository configured.\n"
                f"Press 'e' (Edit) → 'Repo Info' to set HuggingFace or ModelScope repository.",
                severity="warning",
                timeout=8,
            )
            return

        # Use shared verification logic
        self._start_verification(self.selected_model, is_batch=False)

    async def verify_model_async(self, model):
        """Async worker to verify model with progress updates"""
        from kt_kernel.cli.utils.model_verifier import (
            verify_model_integrity_with_progress,
            check_huggingface_connectivity,
        )
        from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

        try:
            # Determine repo type
            repo_type = model.repo_type or "huggingface"

            # Check HuggingFace connectivity and decide whether to use mirror
            use_mirror = False
            if repo_type == "huggingface":
                # Check connectivity
                is_accessible, message = await asyncio.to_thread(check_huggingface_connectivity, timeout=5)

                if not is_accessible:
                    # HuggingFace is not accessible, use mirror
                    use_mirror = True
                    self.log.warning(f"HuggingFace not accessible: {message}")
                    self.log.info("Auto-switching to HuggingFace mirror: hf-mirror.com")

                    # Notify user
                    self.post_message(
                        self.VerifyProgress(
                            model.id, "⚠ HuggingFace not accessible, using mirror: hf-mirror.com", is_milestone=True
                        )
                    )

            # Progress callback to update UI
            def progress_callback(msg, total=None, current=None):
                # Post progress message to UI with total and current info
                # Determine if this is a milestone (phase change)
                is_milestone = any(keyword in msg for keyword in ["Fetching", "Calculating", "Comparing", "Using"])

                self.post_message(
                    self.VerifyProgress(model.id, msg, total=total, current=current, is_milestone=is_milestone)
                )

            # For GGUF models, only verify files that exist locally
            # (since GGUF repos often contain multiple quantizations)
            files_to_verify = None
            if model.format == "gguf":
                model_path = Path(model.path)
                local_gguf_files = [f.name for f in model_path.glob("*.gguf") if f.is_file()]
                if local_gguf_files:
                    files_to_verify = local_gguf_files

            # Run verification with progress (with mirror support)
            result = await asyncio.to_thread(
                verify_model_integrity_with_progress,
                repo_type=repo_type,
                repo_id=model.repo_id,
                local_dir=Path(model.path),
                progress_callback=progress_callback,
                files_to_verify=files_to_verify,
                use_mirror=use_mirror,  # Pass mirror flag
            )

            # Post completion message
            self.post_message(self.VerifyComplete(model.id, result))

        except Exception as e:
            self.post_message(self.VerifyComplete(model.id, {"status": "error", "error_message": str(e)}))

    def action_quant(self) -> None:
        """Quantize selected model"""
        if not self.selected_model:
            self.notify("No model selected", severity="warning")
            return

        # Only GPU models (safetensors, non-AMX) can be quantized
        if self.selected_model.format != "safetensors":
            self.notify("Only GPU models (safetensors) can be quantized", severity="warning")
            return

        from kt_kernel.cli.commands.model import is_amx_weights

        is_amx, _ = is_amx_weights(self.selected_model.path)
        if is_amx:
            self.notify("This model is already quantized (AMX format)", severity="warning")
            return

        # Show quantization config screen
        self.push_screen(QuantConfigScreen(self.selected_model), callback=self._on_quant_config_done)

    def _on_quant_config_done(self, config) -> None:
        """Handle quantization config completion"""
        if config is None:
            # User cancelled
            return

        # Store config for the worker
        self._quant_config = config
        self._quant_config["model"] = self.selected_model

        # Show progress screen and start quantization
        progress_screen = QuantProgressScreen(self.selected_model.name)
        self._quant_progress_screen = progress_screen

        # Push screen first, then start worker after a small delay to ensure screen is mounted
        self.push_screen(progress_screen, callback=self._on_quant_complete)

        # Start quantization worker after screen is mounted (use call_later)
        self.call_later(self._start_quant_worker)

    def _start_quant_worker(self) -> None:
        """Start the quantization worker after screen is mounted"""
        config = self._quant_config
        model = config["model"]
        progress_screen = self._quant_progress_screen

        # Start quantization worker
        self.run_worker(
            self.quantize_model_async(model, config, progress_screen),
            name=f"quant-{model.id}",
            group="quantization",
            exclusive=True,
        )

    async def quantize_model_async(self, model, config, progress_screen):
        """Async worker to quantize model with real-time output"""
        import sys
        import subprocess
        from pathlib import Path

        try:
            # Update initial status
            progress_screen.update_status("🔄 [bold yellow]Preparing quantization...[/bold yellow]")

            # Find conversion script
            try:
                import kt_kernel

                kt_kernel_path = Path(kt_kernel.__file__).parent.parent
            except ImportError:
                kt_kernel_path = None

            if kt_kernel_path is None or not kt_kernel_path.exists():
                raise Exception("kt-kernel not found. Install with: kt install inference")

            script_path = kt_kernel_path / "scripts" / "convert_cpu_weights.py"
            if not script_path.exists():
                raise Exception(f"Conversion script not found: {script_path}")

            # Build command
            cmd = [
                sys.executable,
                str(script_path),
                "--input-path",
                model.path,
                "--input-type",
                config["input_type"],
                "--output",
                config["output_path"],
                "--quant-method",
                config["method"],
                "--cpuinfer-threads",
                str(config["cpu_threads"]),
                "--threadpool-count",
                str(config["numa_nodes"]),
            ]

            # Add --gpu flag if enabled
            if config.get("use_gpu", False):
                cmd.append("--gpu")

            progress_screen.append_output(f"Command: {' '.join(cmd)}\n")
            progress_screen.update_status("🔄 [bold yellow]Quantizing model...[/bold yellow]")

            # Run subprocess with real-time output
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=str(kt_kernel_path)
            )

            # Read output line by line
            while True:
                line = await process.stdout.readline()
                if not line:
                    break

                line_str = line.decode("utf-8", errors="replace").rstrip()
                if line_str:
                    progress_screen.append_output(line_str)

            # Wait for completion
            await process.wait()

            if process.returncode == 0:
                progress_screen.update_status(
                    "[bold green]✓ Quantization completed successfully![/bold green]", running=False
                )
                progress_screen.set_success(True)
                return {
                    "status": "success",
                    "output_path": config["output_path"],
                    "method": config["method"],
                    "numa_nodes": config["numa_nodes"],
                }
            else:
                progress_screen.update_status(
                    f"[bold red]✗ Quantization failed (exit code {process.returncode})[/bold red]", running=False
                )
                progress_screen.set_success(False)
                return {"status": "error", "error": f"Exit code {process.returncode}"}

        except Exception as e:
            progress_screen.update_status(f"[bold red]✗ Error: {str(e)}[/bold red]", running=False)
            progress_screen.set_success(False)
            return {"status": "error", "error": str(e)}

    def _on_quant_complete(self, success) -> None:
        """Handle quantization completion"""
        if success:
            # Quantization succeeded - add to model registry and set GPU links
            from kt_kernel.cli.utils.user_model_registry import UserModelRegistry
            from pathlib import Path

            registry = UserModelRegistry()
            config = self._quant_config
            output_path = Path(config["output_path"])

            self.log.info(f"Checking output path: {output_path}")

            if not output_path.exists():
                self.log.error(f"Output path does not exist: {output_path}")
                self.notify(f"Quantization succeeded, but output path not found: {output_path}", severity="warning")
                return

            # Scan the quantized model directory
            from kt_kernel.cli.utils.model_scanner import scan_directory

            try:
                self.log.info(f"Scanning directory: {output_path}")
                scanned_models, warnings = scan_directory(
                    output_path, min_size_gb=0.1, exclude_paths=[]  # Lower threshold for AMX models
                )

                self.log.info(f"Found {len(scanned_models)} models")

                if not scanned_models:
                    self.log.error("No models found in scanned directory")
                    self.notify(
                        "Quantization succeeded, but no valid model files found in output directory", severity="warning"
                    )
                    return

                # Add the quantized model
                quantized_model = scanned_models[0]
                self.log.info(f"Adding model: {quantized_model.folder_name}, format: {quantized_model.format}")

                # Check if model already exists
                existing = registry.find_by_path(str(output_path))
                if existing:
                    self.log.info(f"Model already exists: {existing.name}")
                    quantized_name = existing.name
                else:
                    from kt_kernel.cli.utils.user_model_registry import UserModel

                    # Suggest unique name to avoid conflicts
                    unique_name = registry.suggest_name(quantized_model.folder_name)
                    self.log.info(f"Using name: {unique_name} for quantized model")

                    new_model = UserModel(name=unique_name, path=str(output_path), format=quantized_model.format)
                    registry.add_model(new_model)
                    quantized_name = unique_name
                    self.notify(f"✓ Quantized model added: {quantized_name}", severity="information")

                # Set GPU link: AMX model should link to original GPU model
                # gpu_model_ids is for AMX/CPU models to store which GPU models they can run with
                original_model_name = config["model"].name
                self.log.info(f"Setting GPU link on AMX model: {quantized_name} → {original_model_name}")

                # Get fresh copy of the original GPU model from registry
                original_model_obj = registry.get_model(original_model_name)
                if not original_model_obj or not original_model_obj.id:
                    self.log.error(f"Could not find original GPU model in registry: {original_model_name}")
                    self.notify(
                        f"✓ Quantized model added, but original GPU model not found for linking", severity="warning"
                    )
                    # Still reload models to show the new quantized model
                    self.load_models()
                    return

                self.log.info(f"Original GPU model ID: {original_model_obj.id}")

                # Get the quantized AMX model
                quant_model_obj = registry.get_model(quantized_name)
                if not quant_model_obj:
                    self.log.error(f"Could not find quantized model by name: {quantized_name}")
                    self.notify(f"✓ Quantized model added, but failed to retrieve it for GPU link", severity="warning")
                    # Still reload models to show the new quantized model
                    self.load_models()
                    return

                # Set gpu_model_ids on the AMX model to point to the original GPU model
                existing_links = quant_model_obj.gpu_model_ids or []
                self.log.info(f"Existing GPU links for AMX model {quantized_name}: {existing_links}")

                # Add original GPU model ID to AMX model's gpu_model_ids if not already there
                if original_model_obj.id not in existing_links:
                    existing_links.append(original_model_obj.id)

                    success = registry.update_model(quantized_name, {"gpu_model_ids": existing_links})

                    if success:
                        self.log.info(f"Successfully updated GPU links on AMX model: {existing_links}")
                        self.notify(f"✓ GPU link set: {quantized_name} → {original_model_name}", severity="information")
                    else:
                        self.log.error(f"Failed to update GPU links on AMX model")
                        self.notify(f"✓ Quantized model added, but failed to update GPU link", severity="warning")
                else:
                    self.log.info(f"GPU link already exists on AMX model")
                    self.notify(f"✓ Quantized model added (GPU link already exists)", severity="information")

                # Reload models to show changes
                self.load_models()

            except Exception as e:
                import traceback

                self.log.error(f"Failed to add quantized model: {e}")
                self.log.error(traceback.format_exc())
                self.notify(f"Quantization succeeded, but failed to add model: {e}", severity="warning")
                # Try to reload models anyway in case partial success
                try:
                    self.load_models()
                except:
                    pass

        # Clean up
        if hasattr(self, "_quant_config"):
            delattr(self, "_quant_config")
        if hasattr(self, "_quant_progress_screen"):
            delattr(self, "_quant_progress_screen")

    def action_link_gpu(self) -> None:
        """Link CPU model to GPU models"""
        if not self.selected_model:
            self.notify("No model selected", severity="warning")
            return

        # Show link screen
        self.push_screen(LinkModelsScreen(self.selected_model, "gpu"), callback=self._on_link_models)

    def action_link_cpu(self) -> None:
        """Link GPU model to CPU models"""
        if not self.selected_model:
            self.notify("No model selected", severity="warning")
            return

        # Show link screen
        self.push_screen(LinkModelsScreen(self.selected_model, "cpu"), callback=self._on_link_models)

    def action_unlink(self) -> None:
        """Clear all links for the selected model with bidirectional sync"""
        if not self.selected_model:
            self.notify("No model selected", severity="warning")
            return

        from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

        registry = UserModelRegistry()
        all_models = registry.list_models()

        # Get current links before clearing
        old_links = set(self.selected_model.gpu_model_ids or [])

        # Clear source model's links
        success = registry.update_model(self.selected_model.name, {"gpu_model_ids": []})

        if not success:
            self.notify(f"Failed to clear links", severity="error")
            return

        # Bidirectional sync: remove this model from all previously linked models
        for model_id in old_links:
            target_model = next((m for m in all_models if m.id == model_id), None)
            if target_model:
                target_links = set(target_model.gpu_model_ids or [])
                target_links.discard(self.selected_model.id)
                registry.update_model(target_model.name, {"gpu_model_ids": list(target_links)})

        self.notify(f"✓ Cleared all links for {self.selected_model.name} (bidirectional)", severity="information")
        self.load_models()

    def _on_link_models(self, selected_ids) -> None:
        """Handle link models screen result with bidirectional sync"""
        if selected_ids is None:
            return

        from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

        registry = UserModelRegistry()
        all_models = registry.list_models()

        # Get current links before update
        old_links = set(self.selected_model.gpu_model_ids or [])
        new_links = set(selected_ids)

        # Models that were added
        added_links = new_links - old_links
        # Models that were removed
        removed_links = old_links - new_links

        # Update source model
        success = registry.update_model(self.selected_model.name, {"gpu_model_ids": list(selected_ids)})

        if not success:
            self.notify(f"Failed to update links", severity="error")
            return

        # Bidirectional sync: update all linked models
        # For each newly added link, add source model to target's links
        for model_id in added_links:
            target_model = next((m for m in all_models if m.id == model_id), None)
            if target_model:
                target_links = set(target_model.gpu_model_ids or [])
                target_links.add(self.selected_model.id)
                registry.update_model(target_model.name, {"gpu_model_ids": list(target_links)})

        # For each removed link, remove source model from target's links
        for model_id in removed_links:
            target_model = next((m for m in all_models if m.id == model_id), None)
            if target_model:
                target_links = set(target_model.gpu_model_ids or [])
                target_links.discard(self.selected_model.id)
                registry.update_model(target_model.name, {"gpu_model_ids": list(target_links)})

        count = len(selected_ids)
        self.notify(f"✓ Updated links: {count} model(s) linked (bidirectional)", severity="information")
        self.load_models()

    def action_remove(self) -> None:
        """Remove selected model with confirmation dialog"""
        if not self.selected_model:
            self.notify("No model selected", severity="warning")
            return

        # Show confirmation dialog
        message = f"Are you sure you want to remove:\n\n"
        message += f"  [cyan]{self.selected_model.name}[/cyan]\n\n"
        message += f"[dim]Path: {self.selected_model.path}[/dim]\n\n"
        message += f"[yellow]Note: This only removes from registry, files will not be deleted.[/yellow]"

        self.push_screen(
            ConfirmDialog("Remove Model", message, confirm_variant="error"), callback=self._on_remove_confirm
        )

    def _on_remove_confirm(self, confirmed: bool) -> None:
        """Handle remove confirmation result"""
        if not confirmed or not self.selected_model:
            return

        from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

        registry = UserModelRegistry()
        all_models = registry.list_models()
        model_to_remove = self.selected_model
        model_uuid = model_to_remove.id

        # Check if this model has links (bidirectional cleanup needed)
        old_links = set(model_to_remove.gpu_model_ids or [])

        # Check if other models link to this one
        linked_by = []
        for m in all_models:
            if m.gpu_model_ids and model_uuid in m.gpu_model_ids:
                linked_by.append(m)

        # Remove the model
        success = registry.remove_model(model_to_remove.name)

        if success:
            # Bidirectional cleanup: remove this model from all linked models
            for model_id in old_links:
                target_model = next((m for m in all_models if m.id == model_id), None)
                if target_model:
                    target_links = set(target_model.gpu_model_ids or [])
                    target_links.discard(model_uuid)
                    registry.update_model(target_model.name, {"gpu_model_ids": list(target_links)})

            # Remove links from models that linked to this one
            for linked_model in linked_by:
                linked_links = set(linked_model.gpu_model_ids or [])
                linked_links.discard(model_uuid)
                registry.update_model(linked_model.name, {"gpu_model_ids": list(linked_links)})

            self.notify(f"✓ Removed: {model_to_remove.name}", severity="information")
            self.load_models()
        else:
            self.notify(f"Failed to remove model", severity="error")

    def action_info(self) -> None:
        """Toggle model info display"""
        if not self.selected_model:
            return

        # If info screen is already open, close it
        if self.info_screen_open:
            self.pop_screen()
            self.info_screen_open = False
            return

        model = self.selected_model
        moe_info = self.moe_results.get(model.id, {})

        # Show info screen
        self.info_screen_open = True
        self.push_screen(InfoScreen(model, moe_info), callback=self._on_info_dismissed)

    def _on_info_dismissed(self, result=None) -> None:
        """Called when info screen is dismissed"""
        self.info_screen_open = False

    def action_scan(self) -> None:
        """Scan for models"""
        from kt_kernel.cli.utils.model_scanner import scan_directory
        from kt_kernel.cli.utils.user_model_registry import UserModelRegistry
        from kt_kernel.cli.config.settings import get_settings
        from pathlib import Path

        self.notify("Scanning for models...")
        try:
            settings = get_settings()
            model_paths = settings.get_model_paths()

            if not model_paths:
                self.notify("No model path configured", severity="warning")
                return

            # Get existing models to exclude
            registry = UserModelRegistry()
            existing_paths = {m.path for m in registry.list_models()}

            # Scan all configured paths
            all_scanned = []
            all_warnings = []

            for model_path in model_paths:
                scanned_models, warnings = scan_directory(
                    Path(model_path), min_size_gb=1.0, exclude_paths=existing_paths
                )
                all_scanned.extend(scanned_models)
                all_warnings.extend(warnings)

            # Show warnings if any
            for warning in all_warnings:
                self.log.warning(warning)

            if not all_scanned:
                self.notify("No new models found", severity="information")
                return

            # Add to registry
            added_count = 0
            from kt_kernel.cli.utils.user_model_registry import UserModel

            for scanned in all_scanned:
                if not registry.find_by_path(scanned.path):
                    # Suggest unique name to avoid conflicts
                    unique_name = registry.suggest_name(scanned.folder_name)

                    new_model = UserModel(name=unique_name, path=scanned.path, format=scanned.format, repo_id=None)
                    registry.add_model(new_model)
                    added_count += 1

            if added_count > 0:
                self.notify(f"Added {added_count} new model(s)", severity="information")
                self.load_models()  # Refresh tables
            else:
                self.notify("All found models already registered", severity="information")

        except Exception as e:
            self.notify(f"Scan failed: {e}", severity="error")
            self.log.error(f"Scan error: {e}")

    def action_doctor(self) -> None:
        """Show system diagnostics"""
        self.push_screen(DoctorScreen())

    def action_settings(self) -> None:
        """Open settings dialog"""
        self.push_screen(SettingsScreen(), callback=self._on_settings_action)

    def _on_settings_action(self, result) -> None:
        """Handle settings dialog result"""
        if result == "force_refresh":
            self._force_refresh_all()
        elif result == "add_path":
            self._add_path()
        elif result == "edit_path":
            self._edit_path()
        elif result == "remove_path":
            self._remove_path()
        elif result == "auto_repo":
            self._auto_repo()
        elif result == "verify_all":
            self._verify_all()

    def _force_refresh_all(self) -> None:
        """Force refresh all models and recalculate SHA256"""
        from kt_kernel.cli.utils.model_scanner import scan_directory
        from kt_kernel.cli.utils.user_model_registry import UserModelRegistry, UserModel
        from kt_kernel.cli.config.settings import get_settings
        from pathlib import Path

        try:
            self.notify("🔄 Force refreshing all models...", severity="information", timeout=3)

            settings = get_settings()
            model_paths = settings.get_model_paths()

            # If no paths configured, use global scan
            if not model_paths or not any(p.exists() for p in model_paths):
                self._global_scan()
                return

            # Scan all configured paths
            self.sub_title = "Scanning directories..."

            all_scanned = []
            all_warnings = []

            for model_path in model_paths:
                scanned_models, warnings = scan_directory(
                    Path(model_path), min_size_gb=1.0, exclude_paths=None  # Don't exclude anything for force refresh
                )
                all_scanned.extend(scanned_models)
                all_warnings.extend(warnings)

            # Show warnings if any
            for warning in all_warnings:
                self.log.warning(warning)

            # Clear existing registry and rebuild
            registry = UserModelRegistry()

            # Get existing models to preserve custom settings
            existing_models = {m.path: m for m in registry.list_models()}

            # Clear all models
            for model in registry.list_models():
                registry.remove_model(model.name)

            # Re-add all scanned models
            added_count = 0
            models_without_repo = []  # Track models without repo_id for auto-detection
            self.sub_title = f"Adding models..."

            for scanned in all_scanned:
                path = scanned.path

                # Preserve repo info if it existed
                repo_id = None
                repo_type = None
                if path in existing_models:
                    old_model = existing_models[path]
                    repo_id = old_model.repo_id
                    repo_type = old_model.repo_type

                # Suggest unique name
                base_name = scanned.folder_name
                name = registry.suggest_name(base_name)

                new_model = UserModel(
                    name=name,
                    path=path,
                    format=scanned.format,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    sha256_status="not_checked",  # Reset SHA256 status
                )
                registry.add_model(new_model)
                added_count += 1

                # Track models without repo_id for auto-detection
                if not repo_id:
                    models_without_repo.append(new_model)

            # Auto-detect repo_id for models without it
            if models_without_repo:
                from kt_kernel.cli.utils.repo_detector import detect_repo_for_model

                self.sub_title = "Detecting repository information..."
                repo_detected_count = 0

                for i, model in enumerate(models_without_repo, 1):
                    self.sub_title = f"Analyzing model {i}/{len(models_without_repo)}: {model.name}"

                    try:
                        repo_info = detect_repo_for_model(model.path)
                        if repo_info:
                            repo_id, repo_type = repo_info
                            registry.update_model(model.name, {"repo_id": repo_id, "repo_type": repo_type})
                            repo_detected_count += 1
                    except Exception as e:
                        self.log.warning(f"Failed to detect repo for {model.name}: {e}")

                if repo_detected_count > 0:
                    self.notify(f"✓ Detected {repo_detected_count} repository IDs", severity="information", timeout=3)

            # Analyze MoE models (all models, using cache)
            if all_scanned:
                from kt_kernel.cli.utils.analyze_moe_model import analyze_moe_model

                self.sub_title = "Analyzing MoE models..."
                moe_analyzed_count = 0
                all_models_list = registry.list_models()

                for i, model in enumerate(all_models_list, 1):
                    if model.format != "safetensors":
                        continue

                    self.sub_title = f"Analyzing MoE {i}/{len(all_models_list)}: {model.name}"

                    try:
                        result = analyze_moe_model(model.path, use_cache=True)
                        if result and result.get("num_experts"):
                            moe_analyzed_count += 1
                    except Exception as e:
                        self.log.warning(f"Failed to analyze MoE for {model.name}: {e}")

                if moe_analyzed_count > 0:
                    self.notify(f"✓ Analyzed {moe_analyzed_count} MoE models", severity="information", timeout=3)

            self.notify(f"✓ Force refresh complete: {added_count} models", severity="information", timeout=5)
            self.sub_title = ""

            # Reload models display
            self.load_models()

        except Exception as e:
            self.notify(f"Force refresh failed: {e}", severity="error", timeout=10)
            self.log.error(f"Force refresh error: {e}")

    async def _global_scan_async(self) -> None:
        """
        Async version: Scan all disks and common locations for models.
        Used when no model paths are configured.
        Runs in background worker to avoid blocking UI.
        """
        from kt_kernel.cli.utils.environment import scan_storage_locations, scan_models_in_location
        from kt_kernel.cli.utils.user_model_registry import UserModelRegistry, UserModel
        from kt_kernel.cli.utils.model_scanner import scan_single_path
        from kt_kernel.cli.config.settings import get_settings
        from pathlib import Path

        try:
            self.call_from_thread(self.notify, "🌍 Scanning all disks for models...", "information", 3)
            self.call_from_thread(setattr, self, "sub_title", "Scanning storage locations...")

            # Scan all storage locations (disks + common paths)
            locations = scan_storage_locations(min_size_gb=10.0)

            if not locations:
                self.call_from_thread(self.notify, "No suitable storage locations found", "warning", 5)
                return

            # Find models in each location
            all_models = []

            for i, loc in enumerate(locations[:10], 1):  # Scan top 10 locations
                self.call_from_thread(
                    setattr, self, "sub_title", f"Scanning location {i}/{min(len(locations), 10)}: {loc.path}"
                )

                try:
                    models = scan_models_in_location(loc, max_depth=3)
                    if models:
                        all_models.extend(models)
                except Exception as e:
                    self.log.warning(f"Failed to scan {loc.path}: {e}")

            if not all_models:
                self.call_from_thread(self.notify, "No models found in any location", "warning", 5)
                self.call_from_thread(setattr, self, "sub_title", "")
                return

            # Add models to registry
            registry = UserModelRegistry()
            added_count = 0
            added_models = []  # Track newly added models for post-processing
            self.call_from_thread(setattr, self, "sub_title", "Adding models to registry...")

            for model in all_models:
                # Scan the specific path to get detailed info
                scanned = scan_single_path(Path(model.path))
                if not scanned:
                    continue

                # Suggest unique name
                name = registry.suggest_name(scanned.folder_name)

                new_model = UserModel(
                    name=name,
                    path=scanned.path,
                    format=scanned.format,
                    repo_id=None,
                    repo_type=None,
                    sha256_status="not_checked",
                )
                registry.add_model(new_model)
                added_models.append(new_model)
                added_count += 1

            # Auto-detect repo_id from README.md
            if added_models:
                from kt_kernel.cli.utils.repo_detector import detect_repo_for_model

                self.call_from_thread(setattr, self, "sub_title", "Detecting repository information...")
                repo_detected_count = 0

                for i, model in enumerate(added_models, 1):
                    self.call_from_thread(
                        setattr, self, "sub_title", f"📖 Analyzing model {i}/{len(added_models)}: {model.name}"
                    )

                    try:
                        # Detect repo from README
                        repo_info = detect_repo_for_model(model.path)
                        if repo_info:
                            repo_id, repo_type = repo_info
                            registry.update_model(model.name, {"repo_id": repo_id, "repo_type": repo_type})
                            repo_detected_count += 1
                            self.log.info(f"Detected repo for {model.name}: {repo_id}")
                    except Exception as e:
                        self.log.warning(f"Failed to detect repo for {model.name}: {e}")

                if repo_detected_count > 0:
                    self.call_from_thread(
                        self.notify,
                        f"✓ Detected {repo_detected_count}/{len(added_models)} repository IDs",
                        "information",
                        3,
                    )

            # Analyze MoE models
            if added_models:
                from kt_kernel.cli.utils.analyze_moe_model import analyze_moe_model

                self.call_from_thread(setattr, self, "sub_title", "Analyzing MoE models...")
                moe_analyzed_count = 0

                for i, model in enumerate(added_models, 1):
                    # Only analyze safetensors models
                    if model.format != "safetensors":
                        continue

                    self.call_from_thread(
                        setattr, self, "sub_title", f"🔬 Analyzing MoE {i}/{len(added_models)}: {model.name}"
                    )

                    try:
                        result = analyze_moe_model(model.path, use_cache=True)
                        if result and result.get("num_experts"):
                            moe_analyzed_count += 1
                            self.log.info(
                                f"MoE detected for {model.name}: "
                                f"{result['num_experts']} experts, "
                                f"{result.get('experts_per_tok', 'N/A')} per token"
                            )
                    except Exception as e:
                        self.log.warning(f"Failed to analyze MoE for {model.name}: {e}")

                if moe_analyzed_count > 0:
                    self.call_from_thread(self.notify, f"✓ Analyzed {moe_analyzed_count} MoE models", "information", 3)

            # Calculate common parent directories for all found models
            # This groups models by their parent directory
            model_parent_dirs = {}  # parent_dir -> [model_paths]
            for model in all_models:
                parent = str(Path(model.path).parent)
                if parent not in model_parent_dirs:
                    model_parent_dirs[parent] = []
                model_parent_dirs[parent].append(model.path)

            # Save discovered parent directories to settings
            settings = get_settings()
            current_paths = settings.get_model_paths()
            current_path_strs = {str(p) for p in current_paths}

            # Add new unique parent paths (directories that contain models)
            for parent_dir in model_parent_dirs.keys():
                if parent_dir not in current_path_strs:
                    settings.add_model_path(parent_dir)
                    self.log.info(f"Added path: {parent_dir} ({len(model_parent_dirs[parent_dir])} models)")

            self.call_from_thread(self.notify, f"✓ Global scan complete: {added_count} models found", "information", 5)
            self.call_from_thread(setattr, self, "sub_title", "")

            # Reload models display
            self.call_from_thread(self.load_models)

        except Exception as e:
            self.call_from_thread(self.notify, f"Global scan failed: {e}", "error", 10)
            self.log.error(f"Global scan error: {e}")
            self.call_from_thread(setattr, self, "sub_title", "")

    def _add_path(self) -> None:
        """Add a new model path"""
        self.push_screen(PathInputScreen("Add Model Path"), callback=self._on_path_added)

    def _on_path_added(self, new_path) -> None:
        """Handle new path input"""
        if not new_path:
            return

        from kt_kernel.cli.config.settings import get_settings
        from pathlib import Path

        try:
            settings = get_settings()
            model_paths = settings.get_model_paths()

            # Convert to Path object
            new_path_obj = Path(new_path).expanduser().resolve()

            # Check if path already exists in config
            if new_path_obj in model_paths:
                self.notify(f"Path already exists: {new_path_obj}", severity="warning")
                # Reopen settings
                self.push_screen(SettingsScreen(), callback=self._on_settings_action)
                return

            # Add to settings
            settings.add_model_path(str(new_path_obj))
            self.notify(f"✓ Added path: {new_path_obj}", severity="information")

            # Reopen settings to show updated paths
            self.push_screen(SettingsScreen(), callback=self._on_settings_action)

        except Exception as e:
            self.notify(f"Failed to add path: {e}", severity="error")
            self.log.error(f"Add path error: {e}")
            # Reopen settings even on error
            self.push_screen(SettingsScreen(), callback=self._on_settings_action)

    def _edit_path(self) -> None:
        """Edit an existing model path"""
        from kt_kernel.cli.config.settings import get_settings

        try:
            settings = get_settings()
            model_paths = settings.get_model_paths()

            if not model_paths:
                self.notify("No model paths to edit", severity="warning")
                self.push_screen(SettingsScreen(), callback=self._on_settings_action)
                return

            # Show path selection dialog
            self.push_screen(
                PathSelectScreen("Select Path to Edit", [str(p) for p in model_paths]),
                callback=self._on_path_selected_for_edit,
            )

        except Exception as e:
            self.notify(f"Failed to open path editor: {e}", severity="error")
            self.log.error(f"Edit path error: {e}")
            self.push_screen(SettingsScreen(), callback=self._on_settings_action)

    def _on_path_selected_for_edit(self, selected_index) -> None:
        """Handle path selection for editing"""
        if selected_index is None:
            # User cancelled, reopen settings
            self.push_screen(SettingsScreen(), callback=self._on_settings_action)
            return

        from kt_kernel.cli.config.settings import get_settings

        try:
            settings = get_settings()
            model_paths = settings.get_model_paths()
            old_path = str(model_paths[selected_index])

            # Store old path for later
            self.editing_path_index = selected_index
            self.editing_old_path = old_path

            # Show input dialog with current path
            self.push_screen(PathInputScreen("Edit Model Path", default_value=old_path), callback=self._on_path_edited)

        except Exception as e:
            self.notify(f"Failed to edit path: {e}", severity="error")
            self.log.error(f"Path selection error: {e}")
            self.push_screen(SettingsScreen(), callback=self._on_settings_action)

    def _on_path_edited(self, new_path) -> None:
        """Handle edited path"""
        if not new_path:
            # User cancelled, reopen settings
            self.push_screen(SettingsScreen(), callback=self._on_settings_action)
            return

        from kt_kernel.cli.config.settings import get_settings
        from pathlib import Path

        try:
            settings = get_settings()
            model_paths = settings.get_model_paths()

            # Convert to Path object
            new_path_obj = Path(new_path).expanduser().resolve()
            old_path = self.editing_old_path

            # Check if unchanged
            if str(new_path_obj) == old_path:
                self.notify("Path unchanged", severity="information")
                self.push_screen(SettingsScreen(), callback=self._on_settings_action)
                return

            # Check if new path already exists
            if new_path_obj in model_paths:
                self.notify(f"Path already exists: {new_path_obj}", severity="warning")
                self.push_screen(SettingsScreen(), callback=self._on_settings_action)
                return

            # Remove old path and add new one
            settings.remove_model_path(old_path)
            settings.add_model_path(str(new_path_obj))

            self.notify(f"✓ Updated path: {new_path_obj}", severity="information")

            # Reopen settings
            self.push_screen(SettingsScreen(), callback=self._on_settings_action)

        except Exception as e:
            self.notify(f"Failed to update path: {e}", severity="error")
            self.log.error(f"Path edit error: {e}")
            self.push_screen(SettingsScreen(), callback=self._on_settings_action)

    def _remove_path(self) -> None:
        """Remove a model path"""
        from kt_kernel.cli.config.settings import get_settings

        try:
            settings = get_settings()
            model_paths = settings.get_model_paths()

            if not model_paths:
                self.notify("No model paths to remove", severity="warning")
                self.push_screen(SettingsScreen(), callback=self._on_settings_action)
                return

            # Show path selection dialog
            self.push_screen(
                PathSelectScreen("Select Path to Remove", [str(p) for p in model_paths]),
                callback=self._on_path_selected_for_remove,
            )

        except Exception as e:
            self.notify(f"Failed to open path remover: {e}", severity="error")
            self.log.error(f"Remove path error: {e}")
            self.push_screen(SettingsScreen(), callback=self._on_settings_action)

    def _on_path_selected_for_remove(self, selected_index) -> None:
        """Handle path selection for removal"""
        if selected_index is None:
            # User cancelled, reopen settings
            self.push_screen(SettingsScreen(), callback=self._on_settings_action)
            return

        from kt_kernel.cli.config.settings import get_settings

        try:
            settings = get_settings()
            model_paths = settings.get_model_paths()
            path_to_remove = str(model_paths[selected_index])

            # Show confirmation dialog
            self.push_screen(
                ConfirmDialog(
                    f"Remove path '{path_to_remove}'?", "Models in this path will no longer be accessible in the TUI."
                ),
                callback=lambda confirm: self._on_path_remove_confirmed(confirm, path_to_remove),
            )

        except Exception as e:
            self.notify(f"Failed to remove path: {e}", severity="error")
            self.log.error(f"Path removal error: {e}")
            self.push_screen(SettingsScreen(), callback=self._on_settings_action)

    def _on_path_remove_confirmed(self, confirm, path_to_remove) -> None:
        """Handle path removal confirmation"""
        if not confirm:
            # User cancelled, reopen settings
            self.push_screen(SettingsScreen(), callback=self._on_settings_action)
            return

        from kt_kernel.cli.config.settings import get_settings

        try:
            settings = get_settings()
            settings.remove_model_path(path_to_remove)

            self.notify(f"✓ Removed path: {path_to_remove}", severity="information")

            # Reopen settings
            self.push_screen(SettingsScreen(), callback=self._on_settings_action)

        except Exception as e:
            self.notify(f"Failed to remove path: {e}", severity="error")
            self.log.error(f"Path removal error: {e}")
            self.push_screen(SettingsScreen(), callback=self._on_settings_action)

    def _auto_repo(self) -> None:
        """Auto-detect repository information for models"""
        from kt_kernel.cli.utils.user_model_registry import UserModelRegistry
        from kt_kernel.cli.utils.repo_detector import scan_models_for_repo

        try:
            self.notify("Scanning for repository information...", severity="information")

            registry = UserModelRegistry()
            models = registry.list_models()

            if not models:
                self.notify("No models found", severity="warning")
                self.push_screen(SettingsScreen(), callback=self._on_settings_action)
                return

            # Scan for repo information
            results = scan_models_for_repo(models)

            if not results["detected"]:
                self.notify("No repository information detected", severity="information")
                self.push_screen(SettingsScreen(), callback=self._on_settings_action)
                return

            # Show selection screen
            self.push_screen(AutoRepoSelectScreen(results["detected"]), callback=self._on_auto_repo_selected)

        except Exception as e:
            self.notify(f"Auto-repo detection failed: {e}", severity="error")
            self.log.error(f"Auto-repo error: {e}")
            self.push_screen(SettingsScreen(), callback=self._on_settings_action)

    def _on_auto_repo_selected(self, selected_models) -> None:
        """Handle auto-repo selection result"""
        if not selected_models:
            # User cancelled, reopen settings
            self.push_screen(SettingsScreen(), callback=self._on_settings_action)
            return

        from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

        try:
            registry = UserModelRegistry()
            updated_count = 0

            for model, repo_id, repo_type in selected_models:
                success = registry.update_model(model.name, {"repo_id": repo_id, "repo_type": repo_type})

                if success:
                    updated_count += 1

            if updated_count > 0:
                self.notify(f"✓ Updated {updated_count} model(s) with repository information", severity="information")
                self.load_models()  # Refresh tables
            else:
                self.notify("No models were updated", severity="warning")

            # Reopen settings
            self.push_screen(SettingsScreen(), callback=self._on_settings_action)

        except Exception as e:
            self.notify(f"Failed to apply repository information: {e}", severity="error")
            self.log.error(f"Auto-repo apply error: {e}")
            self.push_screen(SettingsScreen(), callback=self._on_settings_action)

    def _verify_all(self) -> None:
        """Verify all models with repo_id but unverified SHA256"""
        from kt_kernel.cli.utils.user_model_registry import UserModelRegistry

        try:
            registry = UserModelRegistry()
            models = registry.list_models()

            # Filter models that need verification
            models_to_verify = [m for m in models if m.repo_id and m.sha256_status != "passed"]

            if not models_to_verify:
                self.notify("All models with repo are already verified", severity="information")
                self.push_screen(SettingsScreen(), callback=self._on_settings_action)
                return

            self.notify(
                f"Starting verification for {len(models_to_verify)} model(s)...", severity="information", timeout=3
            )

            # Setup batch verification queue
            self._verify_models_queue = models_to_verify.copy()
            self._verify_total_count = len(models_to_verify)

            # Start with first model (use shared verification logic)
            first_model = self._verify_models_queue.pop(0)
            self._start_verification(first_model, is_batch=True)

        except Exception as e:
            self.notify(f"Verify-all failed: {e}", severity="error")
            self.log.error(f"Verify-all error: {e}")
            self.push_screen(SettingsScreen(), callback=self._on_settings_action)

    def action_download(self) -> None:
        """Download a model"""
        self.push_screen(DownloadScreen(), callback=self._on_download_config)

    def _on_download_config(self, config) -> None:
        """Handle download configuration and start download"""
        if config is None:
            return

        # Start download in background
        import subprocess
        import sys
        from pathlib import Path

        repo_id = config["repo_id"]
        repo_type = config["repo_type"]
        pattern = config["pattern"]
        model_name = config["model_name"]
        save_path = config["save_path"]

        # Get already-verified files from DownloadScreen
        files = config.get("files", [])
        total_size = config.get("total_size", 0)

        # Build command
        # Note: kt model download uses Rich console which may suppress output in non-TTY
        # We rely on subprocess to capture whatever output there is
        cmd = [
            sys.executable,
            "-u",  # Force unbuffered output
            "-m",
            "kt_kernel.cli.commands.model",
            "download",
            repo_id,
            "--repo-type",
            repo_type,
            "--pattern",
            pattern,
            "--local-dir",
            save_path,
            "--yes",
        ]

        self.log.info(f"Download command: {' '.join(cmd)}")
        self.log.info(f"Repository: {repo_id}")
        self.log.info(f"Pattern: {pattern}")
        self.log.info(f"Destination: {save_path}")
        self.log.info(f"Files count: {len(files)}")
        self.log.info(f"Total size: {total_size}")

        # Show progress screen and start download
        progress_screen = DownloadProgressScreen(cmd, model_name, repo_id, repo_type, save_path, files, total_size)
        self.push_screen(progress_screen, callback=self._on_download_complete)

    def _on_download_complete(self, success: bool) -> None:
        """Handle download completion"""
        if success:
            self.notify("Download completed successfully", severity="information")
            self.load_models()
        else:
            self.notify("Download failed or cancelled", severity="warning")

    def action_action_1(self) -> None:
        """Execute action 1"""
        if self.selected_model:
            actions = self.get_actions_for_model(self.selected_model)
            if actions:
                self.execute_action(actions[0])

    def action_action_2(self) -> None:
        """Execute action 2"""
        if self.selected_model:
            actions = self.get_actions_for_model(self.selected_model)
            if len(actions) > 1:
                self.execute_action(actions[1])

    def action_action_3(self) -> None:
        """Execute action 3"""
        if self.selected_model:
            actions = self.get_actions_for_model(self.selected_model)
            if len(actions) > 2:
                self.execute_action(actions[2])

    def action_action_4(self) -> None:
        """Execute action 4"""
        if self.selected_model:
            actions = self.get_actions_for_model(self.selected_model)
            if len(actions) > 3:
                self.execute_action(actions[3])

    def action_action_5(self) -> None:
        """Execute action 5"""
        if self.selected_model:
            actions = self.get_actions_for_model(self.selected_model)
            if len(actions) > 4:
                self.execute_action(actions[4])

    def action_action_6(self) -> None:
        """Execute action 6"""
        if self.selected_model:
            actions = self.get_actions_for_model(self.selected_model)
            if len(actions) > 5:
                self.execute_action(actions[5])

    def action_action_7(self) -> None:
        """Execute action 7"""
        if self.selected_model:
            actions = self.get_actions_for_model(self.selected_model)
            if len(actions) > 6:
                self.execute_action(actions[6])

    def action_next_table(self) -> None:
        """Move focus to next table"""
        tables = ["gpu-table", "gguf-table", "amx-table"]
        current_idx = tables.index(self.active_table_id)
        next_idx = (current_idx + 1) % len(tables)
        next_table = self.query_one(f"#{tables[next_idx]}", DataTable)
        next_table.focus()

    def action_prev_table(self) -> None:
        """Move focus to previous table"""
        tables = ["gpu-table", "gguf-table", "amx-table"]
        current_idx = tables.index(self.active_table_id)
        prev_idx = (current_idx - 1) % len(tables)
        prev_table = self.query_one(f"#{tables[prev_idx]}", DataTable)
        prev_table.focus()

    def execute_action(self, action_name: str) -> None:
        """Execute named action"""
        # Handle special case: No Repo
        if action_name == "No Repo":
            self.notify(
                f"Cannot verify '{self.selected_model.name}': No repository configured.\n"
                f"Press 'e' (Edit) → 'Repo Info' to set HuggingFace or ModelScope repository.",
                severity="warning",
                timeout=8,
            )
            return

        action_map = {
            "Run": self.action_run,
            "Edit": self.action_edit,
            "Verify": self.action_verify,
            "Quant": self.action_quant,
            "Remove": self.action_remove,
            "Info": self.action_info,
            "Link GPU": self.action_link_gpu,
            "Link CPU": self.action_link_cpu,
            "Unlink": self.action_unlink,
            "Test": self.action_run,
        }

        handler = action_map.get(action_name)
        if handler:
            handler()
        else:
            self.notify(f"Action '{action_name}' not implemented yet")


if __name__ == "__main__":
    app = ModelManagerApp()
    app.run()
