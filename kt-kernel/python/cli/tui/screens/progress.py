"""
Progress screens.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer, Center
from textual.widgets import Button, Checkbox, DataTable, Input, Label, RichLog, Select, Static
from textual.screen import ModalScreen
from textual.worker import Worker, WorkerState


class QuantProgressScreen(ModalScreen):
    """Modal screen for showing quantization progress"""

    CSS_PATH = str(Path(__file__).parent.parent / "styles" / "progress.tcss")

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


class DownloadProgressScreen(ModalScreen):
    """Modal screen for showing download progress"""

    CSS_PATH = str(Path(__file__).parent.parent / "styles" / "progress.tcss")

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


# RunConfigScreen moved to screens/config.py
