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


class InfoScreen(ModalScreen):
    """Modal screen for displaying model information"""

    CSS = """
    InfoScreen {
        align: center middle;
    }

    #info-dialog {
        width: 80;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    #info-content {
        width: 100%;
        height: auto;
        padding: 1 0;
    }

    #info-buttons {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1 0 0 0;
    }
    """

    def __init__(self, model, moe_info=None):
        super().__init__()
        self.model = model
        self.moe_info = moe_info or {}

    def compose(self) -> ComposeResult:
        status_map = {
            "passed": "✓ Passed",
            "failed": "✗ Failed",
            "not_checked": "Not Checked",
            "checking": "Checking...",
            "no_repo": "-",
        }
        sha256_display = status_map.get(self.model.sha256_status, self.model.sha256_status)

        # Build info content
        info_lines = [
            f"[bold]Name:[/bold] {self.model.name}",
            f"[bold]Format:[/bold] {self.model.format}",
            f"[bold]Path:[/bold] {self.model.path}",
        ]

        if self.model.repo_id:
            prefix = (
                "hf:"
                if self.model.repo_type == "huggingface"
                else "ms:" if self.model.repo_type == "modelscope" else ""
            )
            repo_display = f"{prefix}{self.model.repo_id}"
            info_lines.append(f"[bold]Repo:[/bold] {repo_display}")
        elif self.model.repo_type:
            info_lines.append(f"[bold]Repo Type:[/bold] {self.model.repo_type}")

        info_lines.append(f"[bold]SHA256:[/bold] {sha256_display}")

        # Add MoE info if available
        if self.moe_info and "error" not in self.moe_info:
            info_lines.append("")
            info_lines.append("[bold cyan]MoE Analysis:[/bold cyan]")
            num_experts = self.moe_info.get("num_experts", 0)
            num_layers = self.moe_info.get("num_layers", 0)
            info_lines.append(f"  Experts per layer: {num_experts}")
            info_lines.append(f"  Number of layers: {num_layers}")
            info_lines.append(f"  Total experts: {num_experts}×{num_layers} = {num_experts * num_layers}")
            info_lines.append(f"  Total size: {self.moe_info.get('total_size_gb', 0):.2f} GB")
            info_lines.append(f"  Single expert size: {self.moe_info.get('single_expert_size_gb', 0):.2f} GB")
            info_lines.append(f"  Skeleton size: {self.moe_info.get('rest_size_gb', 0):.2f} GB")

        with Container(id="info-dialog"):
            yield Label(f"[bold cyan]Model Information[/bold cyan]")
            yield Label("")
            with Container(id="info-content"):
                yield Static("\n".join(info_lines))
            yield Label("")
            yield Label("[dim]Press 'i' again or ESC to close[/dim]")
            with Horizontal(id="info-buttons"):
                yield Button("Close", id="btn-close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss()

    def on_key(self, event) -> None:
        """Handle key presses"""
        if event.key == "i" or event.key == "escape":
            self.dismiss()


class RepoEditScreen(ModalScreen):
    """Modal screen for editing repository information"""

    CSS = """
    RepoEditScreen {
        align: center middle;
    }

    #repo-dialog {
        width: 70;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    #repo-providers {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1 0;
    }

    #repo-providers Button {
        min-width: 20;
        margin: 0 1;
    }

    #repo-actions {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1 0;
    }

    #repo-actions Button {
        margin: 0 1;
    }

    #btn-huggingface {
        background: $success;
        color: $text;
    }

    #btn-modelscope {
        background: $primary;
        color: $text;
    }
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.selected_type = None

    def compose(self) -> ComposeResult:
        if self.model.repo_id:
            prefix = (
                "hf:"
                if self.model.repo_type == "huggingface"
                else "ms:" if self.model.repo_type == "modelscope" else ""
            )
            current_repo = f"{prefix}{self.model.repo_id}"
        else:
            current_repo = "None"

        with Container(id="repo-dialog"):
            yield Label(f"[bold cyan]Edit Repository Info: {self.model.name}[/bold cyan]")
            yield Label("")
            yield Label(f"[bold]Current:[/bold] {current_repo}")
            yield Label("")
            yield Label("[bold]Select Repository Type:[/bold]")

            # Repository providers in one row
            with Horizontal(id="repo-providers"):
                yield Button("🤗 HuggingFace", id="btn-huggingface", variant="success")
                yield Button("🔷 ModelScope", id="btn-modelscope", variant="primary")

            # Actions in another row
            with Horizontal(id="repo-actions"):
                yield Button("Remove Repo Info", id="btn-remove", variant="warning")
                yield Button("Cancel", id="btn-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id

        if button_id == "btn-cancel":
            self.dismiss(None)
        elif button_id == "btn-remove":
            self.dismiss({"action": "remove"})
        elif button_id == "btn-huggingface":
            self.selected_type = "huggingface"
            self._show_input_dialog()
        elif button_id == "btn-modelscope":
            self.selected_type = "modelscope"
            self._show_input_dialog()

    def _show_input_dialog(self) -> None:
        """Show input dialog for repo ID"""
        example = "deepseek-ai/DeepSeek-V3" if self.selected_type == "huggingface" else "deepseek/DeepSeek-V3"
        self.app.push_screen(
            RepoInputScreen(
                self.selected_type,
                example,
                model_name=self.model.name,
                model_path=self.model.path,
                default_value=self.model.repo_id or "",
            ),
            callback=self._on_input_done,
        )

    def _on_input_done(self, result) -> None:
        """Handle input dialog result"""
        if result:
            self.dismiss({"action": "set", "repo_type": self.selected_type, "repo_id": result})

    def on_key(self, event) -> None:
        """Handle key presses"""
        if event.key == "escape":
            self.dismiss(None)


class RepoInputScreen(ModalScreen):
    """Modal screen for inputting repository ID"""

    CSS = """
    RepoInputScreen {
        align: center middle;
    }

    #input-dialog {
        width: 70;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    Input {
        width: 100%;
        margin: 1 0;
    }

    #input-buttons {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1 0 0 0;
    }

    Button {
        margin: 0 1;
    }
    """

    def __init__(
        self, repo_type: str, example: str, model_name: str = "", model_path: str = "", default_value: str = ""
    ):
        super().__init__()
        self.repo_type = repo_type
        self.example = example
        self.model_name = model_name
        self.model_path = model_path
        self.default_value = default_value or ""

    def compose(self) -> ComposeResult:
        type_display = "HuggingFace" if self.repo_type == "huggingface" else "ModelScope"

        with Container(id="input-dialog"):
            yield Label(f"[bold cyan]Enter {type_display} Repository ID[/bold cyan]")
            yield Label("")
            yield Label(f"[dim]Example: {self.example}[/dim]")
            if self.model_name:
                yield Label(f"[dim]Model: {self.model_name}[/dim]")
            if self.model_path:
                yield Label(f"[dim]Path: {self.model_path}[/dim]")
            yield Input(placeholder=self.example, value=self.default_value, id="repo-input")
            with Horizontal(id="input-buttons"):
                yield Button("OK", id="btn-ok", variant="primary")
                yield Button("Cancel", id="btn-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-ok":
            self._submit()
        else:
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input"""
        self._submit()

    def _submit(self) -> None:
        """Submit the input value"""
        input_widget = self.query_one("#repo-input", Input)
        value = input_widget.value.strip()
        if value:
            self.dismiss(value)
        else:
            self.notify("Repository ID cannot be empty", severity="warning")

    def on_mount(self) -> None:
        """Focus input when screen mounts"""
        self.query_one("#repo-input", Input).focus()

    def on_key(self, event) -> None:
        """Handle key presses"""
        if event.key == "escape":
            self.dismiss(None)


class RenameInputScreen(ModalScreen):
    """Modal screen for renaming a model"""

    CSS = """
    RenameInputScreen {
        align: center middle;
    }

    #rename-dialog {
        width: 70;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    Input {
        width: 100%;
        margin: 1 0;
    }

    #rename-buttons {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1 0 0 0;
    }

    Button {
        margin: 0 1;
    }
    """

    def __init__(self, current_name: str):
        super().__init__()
        self.current_name = current_name

    def compose(self) -> ComposeResult:
        with Container(id="rename-dialog"):
            yield Label("[bold cyan]Rename Model[/bold cyan]")
            yield Label("")
            yield Label(f"[dim]Current name: {self.current_name}[/dim]")
            yield Input(placeholder="New model name", value=self.current_name, id="rename-input")
            with Horizontal(id="rename-buttons"):
                yield Button("Rename", id="btn-rename", variant="primary")
                yield Button("Cancel", id="btn-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-rename":
            self._submit()
        else:
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input"""
        self._submit()

    def _submit(self) -> None:
        """Submit the new name"""
        input_widget = self.query_one("#rename-input", Input)
        new_name = input_widget.value.strip()

        if not new_name:
            self.notify("Name cannot be empty", severity="warning")
            return

        if new_name == self.current_name:
            self.notify("Name unchanged", severity="warning")
            return

        self.dismiss(new_name)

    def on_mount(self) -> None:
        """Focus input when screen mounts and select all text"""
        input_widget = self.query_one("#rename-input", Input)
        input_widget.focus()
        # Select all text for easy replacement
        input_widget.action_end()
        input_widget.action_select_all()

    def on_key(self, event) -> None:
        """Handle key presses"""
        if event.key == "escape":
            self.dismiss(None)


class ConfirmDialog(ModalScreen):
    """Modal confirmation dialog"""

    CSS = """
    ConfirmDialog {
        align: center middle;
    }

    #confirm-dialog {
        width: 60;
        height: auto;
        border: thick $warning;
        background: $surface;
        padding: 1 2;
    }

    #confirm-buttons {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1 0 0 0;
    }

    Button {
        margin: 0 1;
    }
    """

    def __init__(self, title: str, message: str):
        super().__init__()
        self.title = title
        self.message = message

    def compose(self) -> ComposeResult:
        with Container(id="confirm-dialog"):
            yield Label(f"[bold yellow]{self.title}[/bold yellow]")
            yield Label("")
            yield Static(self.message)
            yield Label("")
            with Horizontal(id="confirm-buttons"):
                yield Button("Yes", id="btn-yes", variant="error")
                yield Button("No", id="btn-no", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-yes":
            self.dismiss(True)
        else:
            self.dismiss(False)

    def on_key(self, event) -> None:
        """Handle key presses"""
        if event.key == "escape":
            self.dismiss(False)


class PathInputScreen(ModalScreen):
    """Modal screen for inputting a path"""

    CSS = """
    PathInputScreen {
        align: center middle;
    }

    #path-input-dialog {
        width: 80;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    Input {
        width: 100%;
        margin: 1 0;
    }

    #path-input-buttons {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1 0 0 0;
    }

    Button {
        margin: 0 1;
    }
    """

    def __init__(self, title: str = "Enter Path", default_value: str = ""):
        super().__init__()
        self.title = title
        self.default_value = default_value

    def compose(self) -> ComposeResult:
        with Container(id="path-input-dialog"):
            yield Label(f"[bold cyan]{self.title}[/bold cyan]")
            yield Label("")
            yield Label("[dim]Enter absolute path to model directory[/dim]")
            yield Input(placeholder="/path/to/models", value=self.default_value, id="path-input")
            with Horizontal(id="path-input-buttons"):
                yield Button("OK", id="btn-ok", variant="primary")
                yield Button("Cancel", id="btn-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-ok":
            self._submit()
        else:
            self.dismiss(None)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input"""
        self._submit()

    def _submit(self) -> None:
        """Submit the path"""
        input_widget = self.query_one("#path-input", Input)
        path = input_widget.value.strip()

        if not path:
            self.notify("Path cannot be empty", severity="warning")
            return

        self.dismiss(path)

    def on_mount(self) -> None:
        """Focus input when screen mounts"""
        self.query_one("#path-input", Input).focus()

    def on_key(self, event) -> None:
        """Handle key presses"""
        if event.key == "escape":
            self.dismiss(None)


class PathSelectScreen(ModalScreen):
    """Modal screen for selecting a path from a list"""

    CSS = """
    PathSelectScreen {
        align: center middle;
    }

    #path-select-dialog {
        width: 80;
        height: auto;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    #path-select-buttons {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1 0 0 0;
    }

    #path-list {
        width: 100%;
        height: auto;
        max-height: 20;
        overflow-y: auto;
        padding: 1 0;
    }

    .path-item {
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
    }

    Button {
        margin: 0 1;
    }
    """

    def __init__(self, title: str, paths: list):
        super().__init__()
        self.title = title
        self.paths = paths

    def compose(self) -> ComposeResult:
        with Container(id="path-select-dialog"):
            yield Label(f"[bold cyan]{self.title}[/bold cyan]")
            yield Label("")

            with Container(id="path-list"):
                for i, path in enumerate(self.paths):
                    yield Button(f"{i+1}. {path}", id=f"path-{i}", classes="path-item")

            with Horizontal(id="path-select-buttons"):
                yield Button("Cancel", id="btn-cancel", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel":
            self.dismiss(None)
        elif event.button.id.startswith("path-"):
            index = int(event.button.id.split("-")[1])
            self.dismiss(index)

    def on_key(self, event) -> None:
        """Handle key presses"""
        if event.key == "escape":
            self.dismiss(None)


class AutoRepoSelectScreen(ModalScreen):
    """Modal screen for selecting models to apply auto-detected repo info"""

    CSS = """
    AutoRepoSelectScreen {
        align: center middle;
    }

    #auto-repo-dialog {
        width: 100;
        height: auto;
        max-height: 90%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    #auto-repo-table {
        width: 100%;
        height: auto;
        max-height: 30;
        margin: 1 0;
    }

    #auto-repo-buttons {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1 0 0 0;
    }

    #auto-repo-buttons Button {
        margin: 0 1;
    }

    #auto-repo-hint {
        width: 100%;
        padding: 1 0;
    }
    """

    def __init__(self, detected_models):
        """
        Args:
            detected_models: List of (model, repo_id, repo_type) tuples
        """
        super().__init__()
        self.detected_models = detected_models
        self.selected_indices = set(range(len(detected_models)))  # All selected by default

    def compose(self) -> ComposeResult:
        from textual.widgets import DataTable

        with Container(id="auto-repo-dialog"):
            yield Label("[bold cyan]🔍 Auto-Detect Repository[/bold cyan]")
            yield Label("")
            yield Label(f"[bold]Detected {len(self.detected_models)} model(s) with repository information[/bold]")

            with Container(id="auto-repo-hint"):
                yield Label("[dim]↑↓ Navigate  │  Space Toggle  │  Enter/OK Apply  │  ESC/Cancel Exit[/dim]")

            # Create table
            table = DataTable(id="auto-repo-table", cursor_type="row")
            table.add_columns("✓", "Model Name", "Repository", "Type")

            for i, (model, repo_id, repo_type) in enumerate(self.detected_models):
                checkmark = "[green]✓[/green]" if i in self.selected_indices else "[dim] [/dim]"
                # Add prefix to repo display
                prefix = "hf:" if repo_type == "huggingface" else "ms:" if repo_type == "modelscope" else ""
                repo_display = f"{prefix}{repo_id}"
                table.add_row(checkmark, model.name, repo_display, repo_type, key=str(i))

            yield table

            with Horizontal(id="auto-repo-buttons"):
                yield Button("OK", id="btn-ok", variant="success")
                yield Button("Select All", id="btn-select-all")
                yield Button("Deselect All", id="btn-deselect-all")
                yield Button("Cancel", id="btn-cancel")

    def on_mount(self) -> None:
        """Focus the table when mounted"""
        table = self.query_one("#auto-repo-table", DataTable)
        table.focus()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Handle row selection with space key"""
        pass

    def on_key(self, event) -> None:
        """Handle key presses"""
        if event.key == "space":
            table = self.query_one("#auto-repo-table", DataTable)
            if table.cursor_row is not None:
                # Get row index from cursor position
                row_index = table.cursor_row
                self._toggle_selection(row_index)
        elif event.key == "enter":
            self._apply_selection()
        elif event.key == "escape":
            self.dismiss(None)

    def _toggle_selection(self, row_index: int) -> None:
        """Toggle selection state for a row"""
        table = self.query_one("#auto-repo-table", DataTable)

        if row_index in self.selected_indices:
            self.selected_indices.remove(row_index)
            checkmark = "[dim] [/dim]"
        else:
            self.selected_indices.add(row_index)
            checkmark = "[green]✓[/green]"

        # Update the checkmark column
        table.update_cell_at((table.cursor_row, 0), checkmark)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-ok":
            self._apply_selection()
        elif event.button.id == "btn-cancel":
            self.dismiss(None)
        elif event.button.id == "btn-select-all":
            self._select_all(True)
        elif event.button.id == "btn-deselect-all":
            self._select_all(False)

    def _select_all(self, select: bool) -> None:
        """Select or deselect all items"""
        table = self.query_one("#auto-repo-table", DataTable)

        if select:
            self.selected_indices = set(range(len(self.detected_models)))
            checkmark = "[green]✓[/green]"
        else:
            self.selected_indices.clear()
            checkmark = "[dim] [/dim]"

        # Update all checkmarks
        for i in range(len(self.detected_models)):
            try:
                table.update_cell_at((i, 0), checkmark)
            except:
                pass

    def _apply_selection(self) -> None:
        """Apply selected items"""
        selected_models = [self.detected_models[i] for i in sorted(self.selected_indices)]
        self.dismiss(selected_models)


class QuantConfigScreen(ModalScreen):
    """Modal screen for configuring quantization parameters"""

    CSS = """
    QuantConfigScreen {
        align: center middle;
    }

    #quant-dialog {
        width: 80;
        height: auto;
        background: $surface;
        border: heavy $primary;
        padding: 1 2;
    }

    #quant-params {
        width: 100%;
        height: auto;
    }

    #quant-params > Label {
        height: 1;
    }

    #quant-params > Horizontal {
        height: 3;
        margin-bottom: 0;
    }

    #quant-params > Input {
        height: 3;
        margin-bottom: 0;
    }

    #quant-params > Checkbox {
        height: 3;
        margin: 1 0;
    }

    #quant-buttons {
        width: 100%;
        height: 3;
        align: center middle;
        layout: horizontal;
        margin-top: 1;
    }

    #quant-buttons > Button {
        min-width: 20;
        margin: 0 1;
    }
    """

    def __init__(self, model):
        """
        Args:
            model: UserModel to quantize
        """
        super().__init__()
        self.model = model
        self.method = "int4"  # Default
        self.input_type = "fp8"  # Default input type
        self.use_gpu = False  # Default: CPU only
        self.numa_nodes = None  # Will be set in on_mount
        self.cpu_threads = None  # Will be set in on_mount
        self.output_path = None  # Will be set in on_mount

    def compose(self) -> ComposeResult:
        from kt_kernel.cli.utils.environment import detect_cpu_info

        # Detect CPU info
        cpu_info = detect_cpu_info()
        self.numa_nodes = cpu_info.numa_nodes
        self.cpu_threads = cpu_info.cores

        # Generate default output path
        from pathlib import Path

        model_path = Path(self.model.path)
        default_output = model_path.parent / f"{model_path.name}-AMXINT4-NUMA{self.numa_nodes}"
        self.output_path = str(default_output)

        with Container(id="quant-dialog"):
            yield Label(f"[bold cyan]Quantize Model: {self.model.name}[/bold cyan]")
            yield Label("")

            # Check AMX support
            amx_available = any("amx" in s.lower() for s in cpu_info.instruction_sets)
            if amx_available:
                yield Label("[green]✓ AMX supported on this CPU[/green]")
            else:
                yield Label("[yellow]⚠ AMX not detected (will use fallback)[/yellow]")
            yield Label("")

            with Vertical(id="quant-params"):
                # Method selection
                yield Label("[bold]Quantization Method:[/bold]")
                with Horizontal():
                    yield Button("INT4", id="btn-int4", variant="primary")
                    yield Button("INT8", id="btn-int8")

                # Input type selection
                yield Label("[bold]Input Weight Type:[/bold]")
                with Horizontal():
                    yield Button("FP8", id="btn-fp8", variant="primary")
                    yield Button("FP16", id="btn-fp16")
                    yield Button("BF16", id="btn-bf16")

                # GPU option
                yield Label("[bold]GPU Acceleration:[/bold]")
                yield Checkbox("Use GPU (add --gpu flag)", id="check-gpu", value=False)

                # NUMA nodes
                yield Label(f"[bold]NUMA Nodes:[/bold] (Max: {self.numa_nodes})")
                yield Input(value=str(self.numa_nodes), placeholder=f"1-{self.numa_nodes}", id="input-numa")

                # CPU threads
                yield Label(f"[bold]CPU Threads:[/bold] (Max: {self.cpu_threads})")
                yield Input(value=str(self.cpu_threads), placeholder=f"1-{self.cpu_threads}", id="input-threads")

                # Output path
                yield Label("[bold]Output Path:[/bold]")
                yield Input(value=self.output_path, id="input-output", placeholder="Output directory")

            with Horizontal(id="quant-buttons"):
                yield Button("Start Quantization", id="btn-start", variant="success")
                yield Button("Cancel", id="btn-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-int4":
            self.method = "int4"
            event.button.variant = "primary"
            self.query_one("#btn-int8", Button).variant = "default"
            self._update_output_path()
        elif event.button.id == "btn-int8":
            self.method = "int8"
            event.button.variant = "primary"
            self.query_one("#btn-int4", Button).variant = "default"
            self._update_output_path()
        elif event.button.id == "btn-fp8":
            self.input_type = "fp8"
            event.button.variant = "primary"
            self.query_one("#btn-fp16", Button).variant = "default"
            self.query_one("#btn-bf16", Button).variant = "default"
        elif event.button.id == "btn-fp16":
            self.input_type = "fp16"
            event.button.variant = "primary"
            self.query_one("#btn-fp8", Button).variant = "default"
            self.query_one("#btn-bf16", Button).variant = "default"
        elif event.button.id == "btn-bf16":
            self.input_type = "bf16"
            event.button.variant = "primary"
            self.query_one("#btn-fp8", Button).variant = "default"
            self.query_one("#btn-fp16", Button).variant = "default"
        elif event.button.id == "btn-start":
            self._start_quantization()
        elif event.button.id == "btn-cancel":
            self.dismiss(None)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Update output path when NUMA nodes change"""
        if event.input.id == "input-numa":
            self._update_output_path()

    def _update_output_path(self) -> None:
        """Update output path based on current settings"""
        try:
            numa_input = self.query_one("#input-numa", Input)
            numa_value = int(numa_input.value) if numa_input.value else self.numa_nodes

            from pathlib import Path

            model_path = Path(self.model.path)
            method_str = self.method.upper()
            new_output = model_path.parent / f"{model_path.name}-AMX{method_str}-NUMA{numa_value}"

            output_input = self.query_one("#input-output", Input)
            output_input.value = str(new_output)
        except:
            pass

    def _start_quantization(self) -> None:
        """Validate inputs and start quantization"""
        from kt_kernel.cli.utils.environment import detect_cpu_info

        cpu_info = detect_cpu_info()

        # Get values
        try:
            numa_input = self.query_one("#input-numa", Input)
            numa_value = int(numa_input.value) if numa_input.value else self.numa_nodes

            threads_input = self.query_one("#input-threads", Input)
            threads_value = int(threads_input.value) if threads_input.value else self.cpu_threads

            output_input = self.query_one("#input-output", Input)
            output_value = output_input.value.strip()

            # Get GPU checkbox value
            gpu_checkbox = self.query_one("#check-gpu", Checkbox)
            use_gpu = gpu_checkbox.value
        except ValueError:
            self.app.notify("Invalid input values", severity="error")
            return

        # Validate NUMA
        if numa_value < 1 or numa_value > cpu_info.numa_nodes:
            self.app.notify(f"NUMA nodes must be between 1 and {cpu_info.numa_nodes}", severity="error")
            return

        # Validate threads
        if threads_value < 1 or threads_value > cpu_info.cores:
            self.app.notify(f"CPU threads must be between 1 and {cpu_info.cores}", severity="error")
            return

        # Validate output
        if not output_value:
            self.app.notify("Output path cannot be empty", severity="error")
            return

        # Build config and return
        config = {
            "method": self.method,
            "input_type": self.input_type,
            "use_gpu": use_gpu,
            "numa_nodes": numa_value,
            "cpu_threads": threads_value,
            "output_path": output_value,
        }

        self.dismiss(config)


class QuantProgressScreen(ModalScreen):
    """Modal screen for showing quantization progress"""

    CSS = """
    QuantProgressScreen {
        align: center middle;
    }

    #quant-progress-dialog {
        width: 100;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: heavy $primary;
        padding: 1 2;
    }

    #quant-output {
        width: 100%;
        height: 30;
        background: $surface-darken-1;
        color: $text;
        border: solid $primary;
    }

    #quant-status {
        width: 100%;
        height: auto;
        margin: 1 0;
        text-align: center;
    }

    #quant-close-container {
        width: 100%;
        height: auto;
        align: center middle;
    }

    Button#btn-close {
        min-width: 20;
        height: auto;
        min-height: 3;
        padding: 1 2;
    }
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name
        self._quant_running = True
        self.success = None

    def compose(self) -> ComposeResult:
        from textual.widgets import RichLog

        with Container(id="quant-progress-dialog"):
            yield Label(f"[bold cyan]Quantizing: {self.model_name}[/bold cyan]")
            yield Label("")
            yield Static("", id="quant-status")
            yield RichLog(id="quant-output", wrap=False, highlight=True, markup=True)
            yield Label("")
            with Center():
                yield Button("Close", id="btn-close", variant="primary", disabled=True)

    def append_output(self, line: str) -> None:
        """Append a line to the output display"""
        from textual.widgets import RichLog

        try:
            output_widget = self.query_one("#quant-output", RichLog)
            output_widget.write(line)
            # Auto-scroll to bottom
            self.call_after_refresh(output_widget.scroll_end, animate=False)
        except:
            pass  # Screen not mounted yet

    def update_status(self, status: str, running: bool = True) -> None:
        """Update the status message"""
        self._quant_running = running

        try:
            status_widget = self.query_one("#quant-status", Static)
            status_widget.update(status)

            # Enable close button when done
            if not running:
                close_btn = self.query_one("#btn-close", Button)
                close_btn.disabled = False
        except:
            pass  # Screen not mounted yet

    def set_success(self, success: bool) -> None:
        """Mark quantization as successful or failed"""
        self.success = success

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-close":
            self.dismiss(self.success)


class DoctorScreen(ModalScreen):
    """Modal screen for system diagnostics"""

    CSS = """
    DoctorScreen {
        align: center middle;
    }

    #doctor-dialog {
        width: 100;
        height: auto;
        max-height: 90%;
        background: $surface;
        border: heavy $primary;
        padding: 1 2;
    }

    #doctor-content {
        width: 100%;
        height: auto;
        max-height: 60;
        overflow-y: auto;
        padding: 1 0;
    }

    #doctor-buttons {
        width: 100%;
        height: auto;
        align: center middle;
        margin-top: 1;
    }

    #doctor-buttons > Button {
        min-width: 20;
    }

    .doctor-section {
        width: 100%;
        height: auto;
        margin: 1 0;
    }

    .doctor-item {
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
    }
    """

    def compose(self) -> ComposeResult:
        from kt_kernel.cli.utils.environment import (
            detect_cpu_info,
            detect_gpus,
            detect_cuda_version,
            detect_ram_gb,
        )
        import platform

        with Container(id="doctor-dialog"):
            yield Label("[bold cyan]System Diagnostics[/bold cyan]")
            yield Label("")

            with ScrollableContainer(id="doctor-content"):
                # Python info
                python_version = platform.python_version()
                yield Static(f"[bold]Python Version:[/bold] {python_version}", classes="doctor-item")

                # CPU info
                cpu_info = detect_cpu_info()
                yield Static(f"[bold]CPU:[/bold] {cpu_info.name}", classes="doctor-item")
                yield Static(
                    f"[bold]Cores/Threads:[/bold] {cpu_info.cores} cores / {cpu_info.threads} threads",
                    classes="doctor-item",
                )
                yield Static(f"[bold]NUMA Nodes:[/bold] {cpu_info.numa_nodes}", classes="doctor-item")

                # CPU Instruction Sets
                isa_list = cpu_info.instruction_sets
                has_amx = any("amx" in s.lower() for s in isa_list)
                has_avx512 = any("avx512" in s.lower() for s in isa_list)
                has_avx2 = "AVX2" in isa_list

                if has_amx:
                    isa_status = "[green]AMX available - best performance[/green]"
                elif has_avx512:
                    isa_status = "[yellow]AVX512 available - good performance[/yellow]"
                elif has_avx2:
                    isa_status = "[yellow]AVX2 only - basic support[/yellow]"
                else:
                    isa_status = "[red]No advanced instruction sets[/red]"

                # Display all instruction sets (no truncation)
                display_isa = ", ".join(isa_list)

                yield Static(f"[bold]Instruction Sets:[/bold]", classes="doctor-item")
                yield Static(f"  {display_isa}", classes="doctor-item")
                yield Static(f"  Status: {isa_status}", classes="doctor-item")

                # GPU info
                gpus = detect_gpus()
                if gpus:
                    gpu_names = ", ".join(g.name for g in gpus)
                    total_vram = sum(g.vram_gb for g in gpus)
                    yield Static(f"[bold]GPUs:[/bold] {len(gpus)} found", classes="doctor-item")
                    yield Static(f"  {gpu_names}", classes="doctor-item")
                    yield Static(f"  Total VRAM: {total_vram:.1f} GB", classes="doctor-item")
                else:
                    yield Static("[bold]GPUs:[/bold] [yellow]No GPU detected[/yellow]", classes="doctor-item")

                # CUDA info
                cuda_version = detect_cuda_version()
                if cuda_version:
                    yield Static(f"[bold]CUDA Version:[/bold] {cuda_version}", classes="doctor-item")
                else:
                    yield Static("[bold]CUDA Version:[/bold] [yellow]Not available[/yellow]", classes="doctor-item")

                # RAM info
                ram_gb = detect_ram_gb()
                yield Static(f"[bold]System RAM:[/bold] {ram_gb:.1f} GB", classes="doctor-item")

                # kt-kernel info
                try:
                    import kt_kernel

                    kt_version = getattr(kt_kernel, "__version__", "unknown")
                    kt_variant = getattr(kt_kernel, "__cpu_variant__", "unknown")
                    yield Static(f"[bold]kt-kernel:[/bold] v{kt_version} ({kt_variant})", classes="doctor-item")
                except ImportError:
                    yield Static("[bold]kt-kernel:[/bold] [red]Not installed[/red]", classes="doctor-item")

            yield Label("")
            with Horizontal(id="doctor-buttons"):
                yield Button("Close", id="btn-close", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-close":
            self.dismiss()


class LinkModelsScreen(ModalScreen):
    """Modal screen for linking GPU/CPU models"""

    CSS = """
    LinkModelsScreen {
        align: center middle;
    }

    #link-dialog {
        width: 80;
        height: auto;
        background: $surface;
        border: heavy $primary;
        padding: 1 2;
    }

    #link-content {
        width: 100%;
        height: auto;
        max-height: 30;
        overflow-y: auto;
        padding: 1 0;
        margin-bottom: 1;
    }

    #link-buttons {
        width: 100%;
        height: auto;
        align: center middle;
        layout: horizontal;
        margin-top: 1;
        padding: 1 0;
    }

    #link-buttons > Button {
        min-width: 20;
        margin: 0 1;
    }

    .link-item {
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
    }
    """

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


class ConfirmDialog(ModalScreen):
    """Modal dialog for confirmation"""

    CSS = """
    ConfirmDialog {
        align: center middle;
    }

    #confirm-dialog {
        width: 60;
        height: auto;
        background: $surface;
        border: heavy $primary;
        padding: 1 2;
    }

    #confirm-message {
        width: 100%;
        height: auto;
        padding: 1 0;
        margin-bottom: 1;
    }

    #confirm-buttons {
        width: 100%;
        height: auto;
        align: center middle;
        layout: horizontal;
        margin-top: 1;
    }

    #confirm-buttons > Button {
        min-width: 15;
        margin: 0 1;
    }
    """

    def __init__(self, title: str, message: str, confirm_variant: str = "error"):
        """
        Args:
            title: Dialog title
            message: Confirmation message
            confirm_variant: Variant for confirm button (default: "error" for destructive actions)
        """
        super().__init__()
        self.title = title
        self.message = message
        self.confirm_variant = confirm_variant

    def compose(self) -> ComposeResult:
        with Container(id="confirm-dialog"):
            yield Label(f"[bold]{self.title}[/bold]")
            yield Static(self.message, id="confirm-message")
            with Horizontal(id="confirm-buttons"):
                yield Button("Confirm", id="btn-confirm", variant=self.confirm_variant)
                yield Button("Cancel", id="btn-cancel", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-confirm":
            self.dismiss(True)
        elif event.button.id == "btn-cancel":
            self.dismiss(False)


class DownloadScreen(ModalScreen):
    """Modal screen for downloading models"""

    CSS = """
    DownloadScreen {
        align: center middle;
    }

    #download-dialog {
        width: 100;
        height: auto;
        max-height: 90%;
        background: $surface;
        border: heavy $primary;
        padding: 1 2;
    }

    #download-content {
        width: 100%;
        height: auto;
        max-height: 35;
        overflow-y: auto;
        padding: 1 0;
        margin-bottom: 1;
    }

    #download-params {
        width: 100%;
        height: auto;
    }

    #download-params > Label {
        height: 1;
        margin: 1 0 0 0;
    }

    #download-params > Horizontal {
        height: 3;
        margin-bottom: 0;
    }

    #download-params > Input {
        height: 3;
        margin-bottom: 1;
    }

    #download-buttons {
        width: 100%;
        height: auto;
        align: center middle;
        layout: horizontal;
        margin-top: 1;
        padding: 1 0;
    }

    #download-buttons > Button {
        min-width: 20;
        margin: 0 1;
    }

    #download-status {
        width: 100%;
        height: auto;
        margin: 1 0;
    }

    .file-item {
        width: 100%;
        height: auto;
    }
    """

    def __init__(self):
        super().__init__()
        self.repo_type = "huggingface"
        self.verified = False
        self.files = []
        self.total_size = 0

    def compose(self) -> ComposeResult:
        from kt_kernel.cli.config.settings import get_settings

        settings = get_settings()
        model_paths = settings.get_model_paths()
        default_dir = str(model_paths[0]) if model_paths else ""

        with Container(id="download-dialog"):
            yield Label("[bold cyan]Download Model[/bold cyan]")
            yield Label("")

            with ScrollableContainer(id="download-content"):
                yield Static("", id="download-status")

                with Vertical(id="download-params"):
                    # Repository type selection
                    yield Label("[bold]Repository Source:[/bold]")
                    with Horizontal():
                        yield Button("HuggingFace", id="btn-hf", variant="primary")
                        yield Button("ModelScope", id="btn-ms")

                    # Repository ID input
                    yield Label("[bold]Repository ID:[/bold]")
                    yield Input(placeholder="e.g., deepseek-ai/DeepSeek-V3", id="input-repo")

                    # File pattern input
                    yield Label("[bold]File Pattern:[/bold]")
                    yield Input(
                        value="*", placeholder="* for all files, or *.safetensors, *.gguf, etc.", id="input-pattern"
                    )

                    # Model name input (added)
                    yield Label("[bold]Model Name:[/bold]")
                    yield Input(placeholder="Auto-filled after verification", id="input-model-name")

                    # Save path input (added)
                    yield Label("[bold]Save Path:[/bold]")
                    yield Input(value=default_dir, placeholder="Path to save the model", id="input-save-path")

            yield Label("")
            with Horizontal(id="download-buttons"):
                yield Button("Verify Repository", id="btn-verify", variant="primary")
                yield Button("Download", id="btn-download", variant="success", disabled=True)
                yield Button("Cancel", id="btn-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-hf":
            self.repo_type = "huggingface"
            event.button.variant = "primary"
            self.query_one("#btn-ms", Button).variant = "default"
            self.verified = False
            self.query_one("#btn-download", Button).disabled = True
        elif event.button.id == "btn-ms":
            self.repo_type = "modelscope"
            event.button.variant = "primary"
            self.query_one("#btn-hf", Button).variant = "default"
            self.verified = False
            self.query_one("#btn-download", Button).disabled = True
        elif event.button.id == "btn-verify":
            self._verify_repository()
        elif event.button.id == "btn-download":
            self._start_download()
        elif event.button.id == "btn-cancel":
            self.dismiss(None)

    def _verify_repository(self) -> None:
        """Verify repository and list files"""
        repo_input = self.query_one("#input-repo", Input)
        pattern_input = self.query_one("#input-pattern", Input)
        status_widget = self.query_one("#download-status", Static)

        repo_id = repo_input.value.strip()
        pattern = pattern_input.value.strip() or "*"

        # Validate repo_id
        if not repo_id:
            status_widget.update("[red]Please enter a repository ID[/red]")
            return

        if "/" not in repo_id:
            status_widget.update("[red]Repository ID must be in format: namespace/model-name[/red]")
            return

        # Show verifying message
        status_widget.update(f"[yellow]Verifying repository: {repo_id}...[/yellow]")
        self.query_one("#btn-verify", Button).disabled = True

        # Run verification in worker
        from kt_kernel.cli.utils.download_helper import (
            verify_repo_exists,
            list_remote_files_hf,
            list_remote_files_ms,
            filter_files_by_pattern,
            calculate_total_size,
        )
        from kt_kernel.cli.utils.model_verifier import check_huggingface_connectivity

        try:
            # Check connectivity for HuggingFace
            use_mirror = False
            if self.repo_type == "huggingface":
                is_accessible, msg = check_huggingface_connectivity(timeout=5)
                if not is_accessible:
                    use_mirror = True
                    status_widget.update("[yellow]HuggingFace not accessible, using mirror...[/yellow]")

            # Verify repository exists
            exists, msg = verify_repo_exists(repo_id, self.repo_type, use_mirror)
            if not exists:
                status_widget.update(f"[red]Repository not found: {msg}[/red]")
                self.query_one("#btn-verify", Button).disabled = False
                return

            # List files
            if self.repo_type == "huggingface":
                all_files = list_remote_files_hf(repo_id, use_mirror)
            else:
                all_files = list_remote_files_ms(repo_id)

            # Filter by pattern
            filtered_files = filter_files_by_pattern(all_files, pattern)

            if not filtered_files:
                status_widget.update(f"[yellow]No files match pattern '{pattern}'[/yellow]")
                self.query_one("#btn-verify", Button).disabled = False
                return

            # Calculate total size
            self.total_size = calculate_total_size(filtered_files)
            self.files = filtered_files

            # Auto-fill model name (default: last part of repo_id)
            model_name = repo_id.split("/")[-1]
            model_name_input = self.query_one("#input-model-name", Input)
            if not model_name_input.value:
                model_name_input.value = model_name

            # Update save path with model name
            save_path_input = self.query_one("#input-save-path", Input)
            from pathlib import Path

            current_path = Path(save_path_input.value or "")

            # Determine base directory
            if current_path.is_dir():
                # It's an existing directory, use it as base
                base_dir = current_path
            elif current_path.parent.exists():
                # Parent exists, use parent as base (current path might be a full path with model name)
                base_dir = current_path.parent
            else:
                # Use the path as-is (might be a default value)
                base_dir = current_path if current_path.is_absolute() else Path(save_path_input.value or "").parent

            # Build full path with model name
            if base_dir:
                full_path = base_dir / model_name
                # Check if path already exists
                if full_path.exists():
                    # Find a unique name
                    counter = 1
                    while (base_dir / f"{model_name}-{counter}").exists():
                        counter += 1
                    full_path = base_dir / f"{model_name}-{counter}"

                save_path_input.value = str(full_path)

            # Format size
            size_gb = self.total_size / (1024**3)

            # Show results
            file_count = len(filtered_files)
            status_text = f"[green]✓ Repository found![/green]\n\n"
            status_text += f"Files to download: {file_count}\n"
            status_text += f"Total size: {size_gb:.2f} GB\n\n"

            # Show first few files
            status_text += "Files:\n"
            for i, f in enumerate(filtered_files[:10]):
                file_size_mb = f["size"] / (1024**2)
                status_text += f"  • {f['path']} ({file_size_mb:.1f} MB)\n"

            if len(filtered_files) > 10:
                status_text += f"  ... and {len(filtered_files) - 10} more files"

            status_widget.update(status_text)

            # Enable download button
            self.verified = True
            self.query_one("#btn-download", Button).disabled = False
            self.query_one("#btn-verify", Button).disabled = False

        except Exception as e:
            status_widget.update(f"[red]Verification failed: {e}[/red]")
            self.query_one("#btn-verify", Button).disabled = False

    def _start_download(self) -> None:
        """Start download process"""
        if not self.verified:
            return

        repo_input = self.query_one("#input-repo", Input)
        pattern_input = self.query_one("#input-pattern", Input)
        model_name_input = self.query_one("#input-model-name", Input)
        save_path_input = self.query_one("#input-save-path", Input)

        model_name = model_name_input.value.strip()
        save_path = save_path_input.value.strip()

        if not model_name:
            model_name = repo_input.value.strip().split("/")[-1]

        if not save_path:
            from kt_kernel.cli.config.settings import get_settings

            settings = get_settings()
            model_paths = settings.get_model_paths()
            from pathlib import Path

            save_path = str(Path(model_paths[0]) / model_name) if model_paths else model_name

        config = {
            "repo_type": self.repo_type,
            "repo_id": repo_input.value.strip(),
            "pattern": pattern_input.value.strip() or "*",
            "model_name": model_name,
            "save_path": save_path,
            "files": self.files,
            "total_size": self.total_size,
        }

        self.dismiss(config)


class DownloadProgressScreen(ModalScreen):
    """Modal screen for showing download progress"""

    CSS = """
    DownloadProgressScreen {
        align: center middle;
    }

    #download-progress-dialog {
        width: auto;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: heavy $primary;
        padding: 1 2;
    }

    #download-progress-output {
        width: 100%;
        height: 30;
        background: $surface-darken-1;
        color: $text;
        border: solid $primary;
    }

    #download-progress-status {
        width: 100%;
        height: auto;
        margin: 1 0;
        text-align: center;
    }

    #download-buttons {
        width: 100%;
        height: auto;
        align: center middle;
        layout: horizontal;
    }

    Button#btn-close-download, Button#btn-cancel-download {
        min-width: 15;
        margin: 0 1;
    }
    """

    def __init__(
        self,
        cmd: list,
        model_name: str,
        repo_id: str,
        repo_type: str,
        save_path: str,
        files: list = None,
        total_size: int = 0,
    ):
        super().__init__()
        self.cmd = cmd
        self.model_name = model_name
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.save_path = save_path
        self.files = files or []  # Already verified files from DownloadScreen
        self.total_size = total_size  # Already calculated total size
        self._download_running = True
        self.success = None
        self.process = None

    def compose(self) -> ComposeResult:
        from textual.widgets import RichLog
        from textual.containers import Horizontal

        with Container(id="download-progress-dialog"):
            yield Label(f"[bold cyan]Downloading: {self.repo_id}[/bold cyan]")
            yield Label("")
            yield Static("[yellow]Starting download...[/yellow]", id="download-progress-status")
            yield RichLog(id="download-progress-output", wrap=False, highlight=True, markup=True)
            yield Label("")
            with Center():
                with Horizontal(id="download-buttons"):
                    yield Button("Cancel", id="btn-cancel-download", variant="error")
                    yield Button("Close", id="btn-close-download", variant="primary", disabled=True)

    def on_mount(self) -> None:
        """Start download process when screen is mounted"""
        # Get parent app to use run_worker
        app = self.app
        if hasattr(app, "run_worker"):
            app.run_worker(self.download_async(), name=f"download-{self.repo_id}", group="download", exclusive=False)
        else:
            # Fallback to thread
            import threading

            threading.Thread(target=self._run_download_sync, daemon=True).start()

    def append_output(self, line: str) -> None:
        """Append a line to the output display"""
        from textual.widgets import RichLog

        try:
            output_widget = self.query_one("#download-progress-output", RichLog)
            output_widget.write(line)
            self.call_after_refresh(output_widget.scroll_end, animate=False)
        except:
            pass

    def update_status(self, status: str, running: bool = True) -> None:
        """Update the status message"""
        self._download_running = running

        try:
            status_widget = self.query_one("#download-progress-status", Static)
            status_widget.update(status)
            # Note: Button states are now managed explicitly in download_async and _cancel_download
        except:
            pass

    async def download_async(self):
        """Download model in subprocess and capture stdout"""
        import sys
        import subprocess
        from pathlib import Path

        try:
            # Update initial status
            self.update_status("[bold yellow]Preparing download...[/bold yellow]")

            self.append_output(f"[cyan]Repository: {self.repo_id}[/cyan]")
            self.append_output(f"[cyan]Type: {self.repo_type}[/cyan]")
            self.append_output(f"[cyan]Destination: {self.save_path}[/cyan]")
            self.append_output("")

            # Show verified file info
            if self.files:
                from kt_kernel.cli.utils.model_scanner import format_size

                self.append_output(
                    f"[green]Files to download: {len(self.files)} files ({format_size(self.total_size)})[/green]"
                )
                for i, file in enumerate(self.files[:5]):
                    file_name = Path(file["path"]).name
                    self.append_output(f"  • {file_name} ({format_size(file['size'])})")
                if len(self.files) > 5:
                    self.append_output(f"  ... and {len(self.files) - 5} more")
                self.append_output("")

            # Extract pattern
            pattern = "*"
            if "--pattern" in self.cmd:
                pattern_idx = self.cmd.index("--pattern") + 1
                if pattern_idx < len(self.cmd):
                    pattern = self.cmd[pattern_idx]

            # Create download directory
            download_path = Path(self.save_path)
            download_path.mkdir(parents=True, exist_ok=True)

            self.update_status("[bold yellow]Downloading files...[/bold yellow]")
            self.append_output("[yellow]Starting download...[/yellow]")
            self.append_output("")

            # Build Python code to execute in subprocess
            if self.repo_type == "huggingface":
                download_code = f"""
import sys
from huggingface_hub import snapshot_download

print("Downloading from HuggingFace...", file=sys.stderr)
sys.stderr.flush()

snapshot_download(
    repo_id="{self.repo_id}",
    local_dir="{download_path}",
    allow_patterns={repr(pattern if pattern != "*" else None)},
    local_dir_use_symlinks=False,
    resume_download=True
)

print("Download complete!", file=sys.stderr)
"""
            else:  # modelscope
                download_code = f"""
import sys
from modelscope.hub.snapshot_download import snapshot_download

print("Downloading from ModelScope...", file=sys.stderr)
sys.stderr.flush()

snapshot_download(
    model_id="{self.repo_id}",
    local_dir="{download_path}",
    allow_file_pattern={repr(pattern if pattern != "*" else None)}
)

print("Download complete!", file=sys.stderr)
"""

            # Execute in subprocess and capture output
            self.process = await asyncio.create_subprocess_exec(
                sys.executable, "-u", "-c", download_code, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )

            # Read output line by line
            while True:
                line = await self.process.stdout.readline()
                if not line:
                    break

                line_str = line.decode("utf-8", errors="replace").rstrip()
                if line_str:
                    self.append_output(line_str)

            # Wait for completion
            await self.process.wait()

            # Check result
            if self.process.returncode == 0:
                # Check downloaded files
                files_on_disk = list(download_path.glob("*"))
                self.append_output("")
                self.append_output(f"[green]✓ Download completed![/green]")
                self.append_output(f"[green]✓ {len(files_on_disk)} files in destination[/green]")

                self.update_status("[bold green]✓ Download completed successfully![/bold green]", running=False)
                self.set_success(True)
                # Add to registry
                self._add_to_registry()
                # Enable Close button, disable Cancel
                self.query_one("#btn-close-download", Button).disabled = False
                self.query_one("#btn-cancel-download", Button).disabled = True
            else:
                self.append_output("")
                self.append_output(f"[red]✗ Download failed with exit code {self.process.returncode}[/red]")
                self.update_status(f"[bold red]✗ Download failed[/bold red]", running=False)
                self.set_success(False)
                # Enable Close button, disable Cancel
                self.query_one("#btn-close-download", Button).disabled = False
                self.query_one("#btn-cancel-download", Button).disabled = True

        except Exception as e:
            self.append_output("")
            self.append_output(f"[red]✗ Error: {e}[/red]")
            import traceback

            tb = traceback.format_exc()
            for line in tb.split("\n")[:10]:
                if line.strip():
                    self.append_output(f"[dim]{line}[/dim]")

            self.update_status(f"[bold red]✗ Download failed[/bold red]", running=False)
            self.set_success(False)
            # Enable Close button, disable Cancel
            try:
                self.query_one("#btn-close-download", Button).disabled = False
                self.query_one("#btn-cancel-download", Button).disabled = True
            except:
                pass

    def _run_download_sync(self) -> None:
        """Fallback sync download (kept for compatibility)"""
        import subprocess
        from pathlib import Path

        try:
            self.call_from_thread(self.append_output, f"[cyan]Repository: {self.repo_id}[/cyan]")
            self.call_from_thread(self.append_output, f"[cyan]Type: {self.repo_type}[/cyan]")
            self.call_from_thread(self.append_output, f"[cyan]Destination: {self.save_path}[/cyan]")
            self.call_from_thread(self.append_output, "")
            self.call_from_thread(self.append_output, f"Command: {' '.join(self.cmd)}")
            self.call_from_thread(self.append_output, "")

            # Run subprocess
            process = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

            # Read output line by line
            for line in process.stdout:
                line = line.rstrip()
                if line:
                    self.call_from_thread(self.append_output, line)

            process.wait()

            if process.returncode == 0:
                self.call_from_thread(self.append_output, "")
                self.call_from_thread(self.append_output, "[green]✓ Download completed![/green]")
                self.call_from_thread(self.update_status, "[green]✓ Download completed successfully[/green]", False)
                self.call_from_thread(self.set_success, True)
                self.call_from_thread(self._add_to_registry)
            else:
                self.call_from_thread(
                    self.update_status, f"[red]✗ Download failed (exit code {process.returncode})[/red]", False
                )
                self.call_from_thread(self.set_success, False)

        except Exception as e:
            self.call_from_thread(self.append_output, f"[red]Error: {e}[/red]")
            self.call_from_thread(self.update_status, "[red]✗ Download failed[/red]", False)
            self.call_from_thread(self.set_success, False)

    def _add_to_registry(self) -> None:
        """Add downloaded model to registry"""
        from pathlib import Path
        from kt_kernel.cli.utils.user_model_registry import UserModelRegistry, UserModel
        from kt_kernel.cli.utils.model_scanner import scan_single_path

        try:
            output_path = Path(self.save_path)
            if output_path.exists():
                scanned = scan_single_path(output_path)
                if scanned:
                    registry = UserModelRegistry()
                    # Check if already exists
                    if not registry.find_by_path(str(output_path)):
                        new_model = UserModel(
                            name=self.model_name,
                            path=str(output_path),
                            format=scanned.format,
                            repo_id=self.repo_id,
                            repo_type=self.repo_type,
                        )
                        registry.add_model(new_model)
                        self.append_output(f"[green]✓ Model added to registry: {self.model_name}[/green]")
        except Exception as e:
            self.append_output(f"[yellow]Warning: Failed to add to registry: {e}[/yellow]")

    def set_success(self, success: bool) -> None:
        """Mark download as successful or failed"""
        self.success = success

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button clicks"""
        if event.button.id == "btn-close-download":
            # Close button - just dismiss
            self.dismiss(self.success)
        elif event.button.id == "btn-cancel-download":
            # Cancel button - kill process and cleanup
            self._cancel_download()

    def _cancel_download(self) -> None:
        """Cancel the download and cleanup"""
        import shutil
        from pathlib import Path

        self.append_output("")
        self.append_output("[yellow]Cancelling download...[/yellow]")

        # Kill the download process if it's running
        if self.process and self.process.returncode is None:
            try:
                self.process.kill()
                self.append_output("[yellow]Download process terminated[/yellow]")
            except Exception as e:
                self.append_output(f"[yellow]Warning: Failed to kill process: {e}[/yellow]")

        # Delete the download directory
        download_path = Path(self.save_path)
        if download_path.exists():
            try:
                shutil.rmtree(download_path)
                self.append_output(f"[yellow]Deleted directory: {self.save_path}[/yellow]")
            except Exception as e:
                self.append_output(f"[yellow]Warning: Failed to delete directory: {e}[/yellow]")

        self.append_output("")
        self.append_output("[red]✗ Download cancelled by user[/red]")
        self.update_status("[bold red]✗ Download cancelled[/bold red]", running=False)
        self.set_success(False)

        # Disable Cancel button, enable Close
        try:
            self.query_one("#btn-cancel-download", Button).disabled = True
            self.query_one("#btn-close-download", Button).disabled = False
        except:
            pass


class RunConfigScreen(ModalScreen):
    """Modal screen for configuring model run parameters"""

    CSS = """
    RunConfigScreen {
        align: center middle;
    }

    #run-dialog {
        width: 90;
        height: auto;
        max-height: 90%;
        background: $surface;
        border: heavy $primary;
        padding: 1 2;
    }

    #run-content {
        width: 100%;
        height: auto;
        max-height: 35;
        overflow-y: auto;
        padding: 1 0;
        margin-bottom: 1;
    }

    #run-params {
        width: 100%;
        height: auto;
    }

    #run-params > Label {
        height: 1;
        margin: 1 0 0 0;
    }

    #run-params > Input {
        height: 3;
        margin-bottom: 1;
    }

    #cpu-model-list {
        width: 100%;
        height: auto;
        max-height: 15;
        overflow-y: auto;
        border: solid $primary;
        padding: 1;
        margin-bottom: 1;
    }

    #vram-display {
        width: 100%;
        height: auto;
        padding: 1;
        background: $surface-lighten-1;
        border: solid $accent;
        margin: 1 0;
    }

    #run-buttons {
        width: 100%;
        height: auto;
        align: center middle;
        layout: horizontal;
        margin-top: 1;
        padding: 1 0;
    }

    #run-buttons > Button {
        min-width: 20;
        margin: 0 1;
    }

    .cpu-model-item {
        width: 100%;
        height: auto;
        margin: 0 0 1 0;
    }

    .cpu-model-linked {
        color: $accent;
        text-style: bold;
    }
    """

    def __init__(self, gpu_model, moe_result: dict):
        """
        Args:
            gpu_model: The GPU model to run
            moe_result: MoE analysis result containing expert info
        """
        super().__init__()
        self.gpu_model = gpu_model
        self.moe_result = moe_result
        self.selected_cpu_model_id = None
        self._is_mounted = False  # Track mount status to avoid duplicate updates
        self._updating_cpu_list = False  # Lock to prevent concurrent updates

        # Get system info
        from kt_kernel.cli.utils.environment import detect_cpu_info

        cpu_info = detect_cpu_info()
        self.max_cpu_cores = cpu_info.cores
        self.max_numa_nodes = cpu_info.numa_nodes

        # Get MoE info
        self.num_experts = moe_result.get("num_experts", 0)
        self.rest_size_gb = moe_result.get("rest_size_gb", 0)
        self.single_expert_size_gb = moe_result.get("single_expert_size_gb", 0)

    def compose(self) -> ComposeResult:
        with Container(id="run-dialog"):
            yield Label(f"[bold cyan]Run Configuration: {self.gpu_model.name}[/bold cyan]")
            yield Label("")

            with ScrollableContainer(id="run-content"):
                with Vertical(id="run-params"):
                    # GPU Experts
                    yield Label(f"[bold]GPU Experts[/bold] (0 to {self.num_experts}):")
                    yield Input(placeholder=f"0-{self.num_experts}", id="input-gpu-experts")

                    # CPU Threads
                    yield Label(f"[bold]CPU Threads[/bold] (1 to {self.max_cpu_cores}):")
                    yield Input(placeholder=f"1-{self.max_cpu_cores}", id="input-cpu-threads")

                    # NUMA Nodes
                    yield Label(f"[bold]NUMA Nodes[/bold] (1 to {self.max_numa_nodes}):")
                    yield Input(placeholder=f"1-{self.max_numa_nodes}", id="input-numa-nodes")

                    # Total Tokens
                    yield Label("[bold]Total Tokens[/bold] (1 to 10000):")
                    yield Input(placeholder="1-10000", id="input-total-tokens")

                    # CPU Model Selection
                    yield Label("[bold]CPU Model:[/bold]")
                    yield Container(id="cpu-model-list")

                # VRAM Display
                yield Container(id="vram-display")

            yield Label("")
            with Horizontal(id="run-buttons"):
                yield Button("Run", id="btn-run", variant="success")
                yield Button("Cancel", id="btn-cancel")

    async def on_mount(self) -> None:
        """Initialize CPU model list and VRAM display"""
        # First set flag to prevent event handling
        self._is_mounted = False

        # Use prevent context manager to block Input.Changed events
        numa_input = self.query_one("#input-numa-nodes", Input)
        gpu_input = self.query_one("#input-gpu-experts", Input)
        cpu_input = self.query_one("#input-cpu-threads", Input)
        tokens_input = self.query_one("#input-total-tokens", Input)

        with (
            numa_input.prevent(Input.Changed),
            gpu_input.prevent(Input.Changed),
            cpu_input.prevent(Input.Changed),
            tokens_input.prevent(Input.Changed),
        ):
            gpu_input.value = "1"
            cpu_input.value = str(int(self.max_cpu_cores * 0.8))
            numa_input.value = str(self.max_numa_nodes)
            tokens_input.value = "4096"

        # Populate initial data
        await self._update_cpu_model_list()
        self._update_vram_display()

        # Now allow input changes to trigger updates
        self._is_mounted = True

    async def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes"""
        # Only respond to changes after mount to avoid duplicate initialization
        if not self._is_mounted:
            return

        if event.input.id == "input-numa-nodes":
            # NUMA changed, refresh CPU model list
            await self._update_cpu_model_list()
        elif event.input.id == "input-gpu-experts":
            # GPU experts changed, update VRAM
            self._update_vram_display()

    async def _update_cpu_model_list(self) -> None:
        """Update CPU model list based on NUMA selection"""
        # Prevent concurrent updates
        if self._updating_cpu_list:
            return

        self._updating_cpu_list = True

        try:
            from kt_kernel.cli.utils.user_model_registry import UserModelRegistry
            from kt_kernel.cli.commands.model import is_amx_weights

            registry = UserModelRegistry()
            all_models = registry.list_models()

            # Get selected NUMA value
            try:
                numa_input = self.query_one("#input-numa-nodes", Input)
                selected_numa = int(numa_input.value) if numa_input.value else self.max_numa_nodes
            except:
                selected_numa = self.max_numa_nodes

            # Get linked CPU models
            linked_ids = set(self.gpu_model.gpu_model_ids or [])

            # Filter CPU models
            cpu_models = []
            for model in all_models:
                if model.format == "gguf":
                    # GGUF models: always show
                    is_linked = model.id in linked_ids
                    cpu_models.append((model, is_linked, True))
                elif model.format == "safetensors":
                    is_amx, numa_count = is_amx_weights(model.path)
                    if is_amx:
                        # AMX models: only show if NUMA matches
                        if numa_count == selected_numa:
                            is_linked = model.id in linked_ids
                            cpu_models.append((model, is_linked, True))

            # Sort: linked first, then by name
            cpu_models.sort(key=lambda x: (not x[1], x[0].name))

            # Update display - safely remove all children first
            cpu_list_container = self.query_one("#cpu-model-list", Container)

            # Use remove_children() to safely remove all widgets (async)
            await cpu_list_container.remove_children()

            # Now add new widgets
            if not cpu_models:
                await cpu_list_container.mount(
                    Static("[yellow]No compatible CPU models found[/yellow]", classes="cpu-model-item")
                )
            else:
                widgets_to_mount = []
                for model, is_linked, _ in cpu_models:
                    if is_linked:
                        label = f"✓ [bold]{model.name}[/bold] ({model.format}) [dim]- linked[/dim]"
                        style_class = "cpu-model-item cpu-model-linked"
                    else:
                        label = f"  {model.name} ({model.format})"
                        style_class = "cpu-model-item"

                    # Create clickable button for each CPU model
                    btn = Button(label, id=f"cpu-{model.id}", classes=style_class)
                    widgets_to_mount.append(btn)

                # Mount all widgets at once
                if widgets_to_mount:
                    await cpu_list_container.mount(*widgets_to_mount)

        finally:
            # Always release the lock
            self._updating_cpu_list = False

    def _update_vram_display(self) -> None:
        """Update GPU VRAM requirement display"""
        try:
            gpu_experts_input = self.query_one("#input-gpu-experts", Input)
            gpu_experts = int(gpu_experts_input.value) if gpu_experts_input.value else 0
        except:
            gpu_experts = 0

        # Calculate VRAM: backbone + (gpu_experts × expert_size)
        total_vram = self.rest_size_gb + (gpu_experts * self.single_expert_size_gb)

        vram_display = self.query_one("#vram-display", Container)

        # Safely remove old content
        try:
            for child in list(vram_display.children):
                child.remove()
        except Exception:
            pass

        vram_text = f"[bold]GPU VRAM Requirement:[/bold]\n"
        vram_text += f"  Backbone: {self.rest_size_gb:.2f} GB\n"
        vram_text += f"  Experts ({gpu_experts}): {gpu_experts * self.single_expert_size_gb:.2f} GB\n"
        vram_text += f"  [bold cyan]Total: {total_vram:.2f} GB[/bold cyan]"

        vram_display.mount(Static(vram_text))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-run":
            self._validate_and_run()
        elif event.button.id == "btn-cancel":
            self.dismiss(None)
        elif event.button.id and event.button.id.startswith("cpu-"):
            # CPU model selected
            model_id = event.button.id[4:]  # Remove "cpu-" prefix
            self.selected_cpu_model_id = model_id
            # Highlight selected
            for btn in self.query("Button"):
                if btn.id and btn.id.startswith("cpu-"):
                    if btn.id == event.button.id:
                        btn.variant = "primary"
                    else:
                        btn.variant = "default"

    def _validate_and_run(self) -> None:
        """Validate inputs and prepare to run"""
        try:
            # Get all inputs
            gpu_experts_input = self.query_one("#input-gpu-experts", Input)
            cpu_threads_input = self.query_one("#input-cpu-threads", Input)
            numa_nodes_input = self.query_one("#input-numa-nodes", Input)
            total_tokens_input = self.query_one("#input-total-tokens", Input)

            gpu_experts = int(gpu_experts_input.value) if gpu_experts_input.value else 0
            cpu_threads = int(cpu_threads_input.value) if cpu_threads_input.value else 1
            numa_nodes = int(numa_nodes_input.value) if numa_nodes_input.value else self.max_numa_nodes
            total_tokens = int(total_tokens_input.value) if total_tokens_input.value else 4096

            # Validate ranges
            if gpu_experts < 0 or gpu_experts > self.num_experts:
                self.app.notify(f"GPU experts must be between 0 and {self.num_experts}", severity="error")
                return

            if cpu_threads < 1 or cpu_threads > self.max_cpu_cores:
                self.app.notify(f"CPU threads must be between 1 and {self.max_cpu_cores}", severity="error")
                return

            if numa_nodes < 1 or numa_nodes > self.max_numa_nodes:
                self.app.notify(f"NUMA nodes must be between 1 and {self.max_numa_nodes}", severity="error")
                return

            if total_tokens < 1 or total_tokens > 10000:
                self.app.notify("Total tokens must be between 1 and 10000", severity="error")
                return

            # Check if CPU model selected
            if not self.selected_cpu_model_id:
                self.app.notify("Please select a CPU model", severity="warning")
                return

            # Prepare config
            config = {
                "gpu_model": self.gpu_model,
                "cpu_model_id": self.selected_cpu_model_id,
                "gpu_experts": gpu_experts,
                "cpu_threads": cpu_threads,
                "numa_nodes": numa_nodes,
                "total_tokens": total_tokens,
            }

            self.dismiss(config)

        except ValueError:
            self.app.notify("Please enter valid numbers", severity="error")


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


class EditModelScreen(ModalScreen):
    """Modal screen for editing model information"""

    CSS = """
    EditModelScreen {
        align: center middle;
    }

    #edit-dialog {
        width: 80;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    #edit-buttons {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1 0 0 0;
    }

    Button {
        margin: 0 1;
    }
    """

    def __init__(self, model, moe_info=None):
        super().__init__()
        self.model = model
        self.moe_info = moe_info or {}

    def compose(self) -> ComposeResult:
        status_map = {
            "passed": "✓ Passed",
            "failed": "✗ Failed",
            "not_checked": "Not Checked",
            "checking": "Checking...",
            "no_repo": "-",
        }
        sha256_display = status_map.get(self.model.sha256_status, self.model.sha256_status)

        # Build info content
        info_parts = [
            f"[bold]Name:[/bold] {self.model.name}",
            f"[bold]Format:[/bold] {self.model.format}",
            f"[bold]Path:[/bold] {self.model.path}",
        ]

        if self.model.repo_id:
            prefix = (
                "hf:"
                if self.model.repo_type == "huggingface"
                else "ms:" if self.model.repo_type == "modelscope" else ""
            )
            repo_display = f"{prefix}{self.model.repo_id}"
            info_parts.append(f"[bold]Repo:[/bold] {repo_display}")
        elif self.model.repo_type:
            info_parts.append(f"[bold]Repo Type:[/bold] {self.model.repo_type}")

        info_parts.append(f"[bold]SHA256:[/bold] {sha256_display}")

        if self.moe_info and "error" not in self.moe_info:
            info_parts.append("")
            info_parts.append(
                f"[bold]MoE:[/bold] {self.moe_info.get('num_experts', 0)}×{self.moe_info.get('num_layers', 0)}, {self.moe_info.get('total_size_gb', 0):.1f}GB"
            )

        with Container(id="edit-dialog"):
            yield Label(f"[bold cyan]Edit Model: {self.model.name}[/bold cyan]")
            yield Label("")
            yield Static("\n".join(info_parts))
            yield Label("")
            yield Label("[bold]What would you like to edit?[/bold]")
            yield Label("")
            with Horizontal(id="edit-buttons"):
                yield Button("Rename", id="btn-rename", variant="primary")
                yield Button("Repo Info", id="btn-repo")
                yield Button("Delete", id="btn-delete", variant="error")
                yield Button("Cancel", id="btn-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id

        if button_id == "btn-cancel":
            self.dismiss(None)
        elif button_id == "btn-rename":
            self.dismiss("rename")
        elif button_id == "btn-repo":
            self.dismiss("repo")
        elif button_id == "btn-delete":
            self.dismiss("delete")


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

            if not model_paths:
                self.notify("No model paths configured", severity="warning")
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

            self.notify(f"✓ Force refresh complete: {added_count} models", severity="information", timeout=5)

            # Reload models display
            self.load_models()

        except Exception as e:
            self.notify(f"Force refresh failed: {e}", severity="error", timeout=10)
            self.log.error(f"Force refresh error: {e}")

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
