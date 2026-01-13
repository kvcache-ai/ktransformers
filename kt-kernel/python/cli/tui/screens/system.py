"""
System screens.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer, Center
from textual.widgets import Button, Checkbox, DataTable, Input, Label, RichLog, Select, Static
from textual.screen import ModalScreen
from textual.worker import Worker, WorkerState


class DoctorScreen(ModalScreen):
    """Modal screen for system diagnostics"""

    CSS_PATH = str(Path(__file__).parent.parent / "styles" / "system.tcss")

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


class SettingsScreen(ModalScreen):
    """Modal screen for application settings"""

    CSS_PATH = str(Path(__file__).parent.parent / "styles" / "system.tcss")

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
