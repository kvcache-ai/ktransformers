"""
Common dialog screens.

Reusable dialog components for user interactions: confirmation, path input,
and path selection.
"""

from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Button, Input, Label, Static
from textual.screen import ModalScreen


class ConfirmDialog(ModalScreen):
    """Modal dialog for confirmation"""

    CSS_PATH = str(Path(__file__).parent.parent / "styles" / "dialogs.tcss")

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


class PathInputScreen(ModalScreen):
    """Modal screen for inputting a path"""

    CSS_PATH = str(Path(__file__).parent.parent / "styles" / "dialogs.tcss")

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

    CSS_PATH = str(Path(__file__).parent.parent / "styles" / "dialogs.tcss")

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
