"""
Model information and editing screens.

Screens for viewing model details, editing metadata, and managing repositories.
"""

from pathlib import Path
from typing import Optional, List, Tuple

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, DataTable, Input, Label, Static
from textual.screen import ModalScreen


class InfoScreen(ModalScreen):
    """Modal screen for displaying model information"""

    CSS_PATH = str(Path(__file__).parent.parent / "styles" / "model_info.tcss")

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

    CSS_PATH = str(Path(__file__).parent.parent / "styles" / "model_info.tcss")

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

    CSS_PATH = str(Path(__file__).parent.parent / "styles" / "model_info.tcss")

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

    CSS_PATH = str(Path(__file__).parent.parent / "styles" / "model_info.tcss")

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


class AutoRepoSelectScreen(ModalScreen):
    """Modal screen for selecting models to apply auto-detected repo info"""

    CSS_PATH = str(Path(__file__).parent.parent / "styles" / "model_info.tcss")

    def __init__(self, detected_models):
        """
        Args:
            detected_models: List of (model, repo_id, repo_type) tuples
        """
        super().__init__()
        self.detected_models = detected_models
        self.selected_indices = set(range(len(detected_models)))  # All selected by default

    def compose(self) -> ComposeResult:
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


class EditModelScreen(ModalScreen):
    """Modal screen for editing model information"""

    CSS_PATH = str(Path(__file__).parent.parent / "styles" / "model_info.tcss")

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
