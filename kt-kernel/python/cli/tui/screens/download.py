"""
Download screens.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer, Center
from textual.widgets import Button, Checkbox, DataTable, Input, Label, RichLog, Select, Static
from textual.screen import ModalScreen
from textual.worker import Worker, WorkerState


class DownloadScreen(ModalScreen):
    """Modal screen for downloading models"""

    CSS_PATH = str(Path(__file__).parent.parent / "styles" / "download.tcss")

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
