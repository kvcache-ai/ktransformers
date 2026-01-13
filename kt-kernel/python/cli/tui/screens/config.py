"""
Configuration screens.

Screens for configuring operations before execution (quantization, model run).
"""

from pathlib import Path
from typing import Optional, Dict, Any

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Button, Checkbox, Input, Label, Select, Static
from textual.screen import ModalScreen


class QuantConfigScreen(ModalScreen):
    """Modal screen for configuring quantization parameters"""

    CSS_PATH = str(Path(__file__).parent.parent / "styles" / "config.tcss")

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


class RunConfigScreen(ModalScreen):
    """Modal screen for configuring model run parameters"""

    CSS_PATH = str(Path(__file__).parent.parent / "styles" / "config.tcss")

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
        self.selected_gpu_ids = set()  # Set of selected GPU IDs
        self.gpu_buttons = {}  # gpu_id -> Button

        # Get system info
        from kt_kernel.cli.utils.environment import detect_cpu_info

        cpu_info = detect_cpu_info()
        self.max_cpu_cores = cpu_info.cores
        self.max_numa_nodes = cpu_info.numa_nodes

        # Get MoE info
        self.num_experts = moe_result.get("num_experts", 0)
        self.rest_size_gb = moe_result.get("rest_size_gb", 0)
        self.single_expert_size_gb = moe_result.get("single_expert_size_gb", 0)

    def _get_gpu_info(self):
        """Dynamically get current GPU information with real-time VRAM"""
        from kt_kernel.cli.utils.environment import detect_gpus
        import torch

        gpus = detect_gpus()
        if not gpus:
            return []

        # Enrich with real-time available VRAM
        gpu_info_list = []
        for i, gpu in enumerate(gpus):
            gpu_data = {
                "id": i,
                "name": gpu.name,
                "total_vram_gb": gpu.vram_gb,
                "free_vram_gb": gpu.vram_gb,  # Default fallback
            }

            # Try to get real-time free VRAM
            if torch.cuda.is_available() and i < torch.cuda.device_count():
                try:
                    free_vram_bytes, total_vram_bytes = torch.cuda.mem_get_info(i)
                    gpu_data["free_vram_gb"] = free_vram_bytes / (1024**3)
                    gpu_data["total_vram_gb"] = total_vram_bytes / (1024**3)
                except Exception as e:
                    pass  # Use fallback values

            gpu_info_list.append(gpu_data)

        return gpu_info_list

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

                    # GPU Selection (dynamically populated)
                    yield Label("[bold]GPU Selection[/bold] (Click to select, TP must be power of 2):")
                    with ScrollableContainer(id="gpu-selection"):
                        pass  # Will be populated dynamically

                    # Total Tokens (KV Cache)
                    yield Label("[bold]Total Tokens[/bold] (1 to 10000):")
                    yield Input(placeholder="1-10000", id="input-total-tokens")

                    # CPU Model Selection
                    yield Label("[bold]CPU Model:[/bold]")
                    with ScrollableContainer(id="cpu-model-list"):
                        pass  # Will be populated dynamically

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

        # Populate GPU selection dynamically
        self._populate_gpu_selection()

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

    def _populate_gpu_selection(self) -> None:
        """Populate GPU selection buttons with real-time VRAM info"""
        gpu_container = self.query_one("#gpu-selection", ScrollableContainer)

        # Clear existing content
        try:
            for child in list(gpu_container.children):
                child.remove()
        except Exception:
            pass

        # Get current GPU info
        gpu_info_list = self._get_gpu_info()

        if not gpu_info_list:
            gpu_container.mount(Label("[yellow]No GPUs detected[/yellow]"))
            return

        # Initially select all GPUs
        self.selected_gpu_ids = set(gpu["id"] for gpu in gpu_info_list)

        # Create buttons with real-time VRAM info
        self.gpu_buttons = {}
        for gpu in gpu_info_list:
            gpu_id = gpu["id"]
            free_vram = gpu["free_vram_gb"]
            total_vram = gpu["total_vram_gb"]

            # Detailed GPU label format
            label_text = f"[GPU {gpu_id}] {gpu['name']:<28} │ Free: {free_vram:>5.1f}GB / Total: {total_vram:>5.1f}GB"

            # Create button with appropriate style class
            btn = Button(label_text, id=f"gpu-{gpu_id}", classes="gpu-item gpu-item-selected")  # Initially all selected
            self.gpu_buttons[gpu_id] = btn
            gpu_container.mount(btn)

    async def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes"""
        # Only respond to changes after mount to avoid duplicate initialization
        if not self._is_mounted:
            return

        if event.input.id == "input-numa-nodes":
            # NUMA changed, refresh CPU model list
            await self._update_cpu_model_list()
        elif event.input.id in ["input-gpu-experts", "input-total-tokens"]:
            # GPU experts or tokens changed, update VRAM
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
            cpu_list_container = self.query_one("#cpu-model-list", ScrollableContainer)

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
        """Update GPU VRAM requirement display with KV cache"""
        try:
            gpu_experts_input = self.query_one("#input-gpu-experts", Input)
            gpu_experts = int(gpu_experts_input.value) if gpu_experts_input.value else 0
        except:
            gpu_experts = 0

        try:
            tokens_input = self.query_one("#input-total-tokens", Input)
            total_tokens = int(tokens_input.value) if tokens_input.value else 4096
        except:
            total_tokens = 4096

        # Get selected GPU count (TP size)
        tp = max(1, len(self.selected_gpu_ids))

        # Calculate KV cache size
        kv_cache_gb = 0
        try:
            from kt_kernel.cli.utils.kv_cache_calculator import get_kv_size_gb

            kv_result = get_kv_size_gb(
                model_path=self.gpu_model.path, max_total_tokens=total_tokens, tp=tp, dtype="auto", verbose=False
            )
            kv_cache_gb = kv_result["total_size_gb"]
        except Exception as e:
            kv_cache_gb = 0  # Fallback if calculation fails

        # Calculate VRAM per GPU
        skeleton_per_gpu = self.rest_size_gb / tp  # Skeleton sharded across GPUs
        moe_per_gpu = gpu_experts * self.single_expert_size_gb / tp  # MoE experts also sharded
        # KV cache: calculator already returns per-GPU size (MHA: sharded, MLA: full)
        kv_per_gpu = kv_cache_gb
        total_per_gpu = skeleton_per_gpu + moe_per_gpu + kv_per_gpu

        vram_display = self.query_one("#vram-display", Container)

        # Safely remove old content
        try:
            for child in list(vram_display.children):
                child.remove()
        except Exception:
            pass

        # Build detailed VRAM display
        selected_count = len(self.selected_gpu_ids)
        vram_text = f"[bold cyan]━━━ VRAM Requirements (TP={tp}, {selected_count} GPU{'s' if selected_count != 1 else ''} selected) ━━━[/bold cyan]\n\n"

        vram_text += f"[bold]Per-GPU Breakdown:[/bold]\n"
        vram_text += f"  • Skeleton (sharded):     {skeleton_per_gpu:>7.2f} GB\n"
        vram_text += f"  • MoE ({gpu_experts:>2} experts, sharded): {moe_per_gpu:>7.2f} GB\n"
        vram_text += f"  • KV Cache ({total_tokens} tokens):   {kv_per_gpu:>7.2f} GB\n"
        vram_text += f"  {'-' * 35}\n"
        vram_text += f"  [bold yellow]Total per GPU:           {total_per_gpu:>7.2f} GB[/bold yellow]\n\n"

        vram_text += f"[bold]Total Cluster:[/bold]\n"
        # Cluster total = skeleton (full) + MoE (full) + KV per GPU * GPU count
        total_cluster = self.rest_size_gb + (gpu_experts * self.single_expert_size_gb) + (kv_per_gpu * selected_count)
        vram_text += f"  Across {selected_count} GPU{'s' if selected_count != 1 else ''}:  [bold cyan]{total_cluster:.2f} GB[/bold cyan]"

        vram_display.mount(Static(vram_text))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-run":
            self._validate_and_run()
        elif event.button.id == "btn-cancel":
            self.dismiss(None)
        elif event.button.id and event.button.id.startswith("gpu-"):
            # GPU selection toggle
            if not self._is_mounted:
                return

            gpu_id = int(event.button.id[4:])  # Remove "gpu-" prefix

            # Toggle selection
            if gpu_id in self.selected_gpu_ids:
                # Deselect
                self.selected_gpu_ids.remove(gpu_id)
                event.button.remove_class("gpu-item-selected")
                event.button.add_class("gpu-item-unselected")
            else:
                # Select
                self.selected_gpu_ids.add(gpu_id)
                event.button.remove_class("gpu-item-unselected")
                event.button.add_class("gpu-item-selected")

            # Debug: Show selection count
            count = len(self.selected_gpu_ids)
            self.app.notify(
                f"Selected {count} GPU{'s' if count != 1 else ''}: {sorted(self.selected_gpu_ids)}", timeout=2
            )

            # Update VRAM display
            self._update_vram_display()

        elif event.button.id and event.button.id.startswith("cpu-"):
            # CPU model selected
            if not self._is_mounted:
                return

            model_id = event.button.id[4:]  # Remove "cpu-" prefix
            self.selected_cpu_model_id = model_id

            # Highlight selected using class
            for btn in self.query("Button"):
                if btn.id and btn.id.startswith("cpu-"):
                    if btn.id == event.button.id:
                        btn.add_class("cpu-model-selected")
                    else:
                        btn.remove_class("cpu-model-selected")

            # Notify selection
            self.app.notify(f"Selected CPU model: {model_id}", timeout=2)

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

            # Get selected GPUs
            selected_gpus = sorted(list(self.selected_gpu_ids))

            if not selected_gpus:
                self.app.notify("Please select at least one GPU", severity="error")
                return

            tensor_parallel = len(selected_gpus)

            # Validate TP is power of 2
            if tensor_parallel & (tensor_parallel - 1) != 0:
                self.app.notify(
                    f"Tensor Parallel (selected GPUs count) must be a power of 2 (1, 2, 4, 8, ...). "
                    f"Currently selected: {tensor_parallel} GPUs",
                    severity="error",
                )
                return

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

            # Calculate VRAM requirements
            try:
                vram_check = self._check_vram_requirements(gpu_experts, total_tokens, tensor_parallel, selected_gpus)
                if not vram_check["success"]:
                    self.app.notify(vram_check["error"], severity="error", timeout=10)
                    return
            except Exception as e:
                self.app.notify(f"Failed to check VRAM: {e}", severity="error")
                return

            # Prepare config
            config = {
                "gpu_model": self.gpu_model,
                "cpu_model_id": self.selected_cpu_model_id,
                "gpu_experts": gpu_experts,
                "cpu_threads": cpu_threads,
                "numa_nodes": numa_nodes,
                "total_tokens": total_tokens,
                "tensor_parallel": tensor_parallel,
                "selected_gpus": selected_gpus,
            }

            self.dismiss(config)

        except ValueError:
            self.app.notify("Please enter valid numbers", severity="error")

    def _check_vram_requirements(self, gpu_experts: int, total_tokens: int, tp: int, selected_gpus: list) -> dict:
        """
        Check if selected GPUs have enough VRAM (using real-time GPU info)

        Returns:
            dict with 'success' (bool) and optional 'error' (str)
        """
        try:
            from kt_kernel.cli.utils.kv_cache_calculator import get_kv_size_gb

            # Get real-time GPU information
            gpu_info_list = self._get_gpu_info()
            if not gpu_info_list:
                return {"success": False, "error": "No GPUs detected"}

            # Create a mapping for quick lookup
            gpu_info_map = {gpu["id"]: gpu for gpu in gpu_info_list}

            # Calculate KV cache size
            kv_result = get_kv_size_gb(
                model_path=self.gpu_model.path, max_total_tokens=total_tokens, tp=tp, dtype="auto", verbose=False
            )
            kv_cache_gb = kv_result["total_size_gb"]

            # Calculate total VRAM per GPU
            # Skeleton (rest) + MoE experts + KV Cache
            skeleton_per_gpu = self.rest_size_gb / tp  # Skeleton is sharded across GPUs
            moe_per_gpu = gpu_experts * self.single_expert_size_gb / tp  # MoE experts also sharded
            # KV cache: calculator already returns per-GPU size
            # - MHA models: KV heads sharded (calculator divides by tp)
            # - MLA models: full KV cache on each GPU (calculator doesn't divide)
            kv_per_gpu = kv_cache_gb

            total_vram_per_gpu = skeleton_per_gpu + moe_per_gpu + kv_per_gpu

            # Check each selected GPU
            for gpu_id in selected_gpus:
                if gpu_id not in gpu_info_map:
                    return {"success": False, "error": f"Invalid GPU ID: {gpu_id}"}

                gpu = gpu_info_map[gpu_id]
                free_vram_gb = gpu["free_vram_gb"]

                # Apply 0.95 safety margin
                available_vram = free_vram_gb * 0.95

                if available_vram < total_vram_per_gpu:
                    error_msg = (
                        f"GPU {gpu_id} ({gpu['name']}) insufficient VRAM:\n"
                        f"  Required: {total_vram_per_gpu:.2f} GB\n"
                        f"  Available: {available_vram:.2f} GB (95% of {free_vram_gb:.2f} GB free)\n"
                        f"  Breakdown:\n"
                        f"    Skeleton: {skeleton_per_gpu:.2f} GB\n"
                        f"    MoE ({gpu_experts} experts): {moe_per_gpu:.2f} GB\n"
                        f"    KV Cache ({total_tokens} tokens): {kv_per_gpu:.2f} GB"
                    )
                    return {"success": False, "error": error_msg}

            return {"success": True}

        except ImportError as e:
            return {"success": False, "error": f"Failed to import KV cache calculator: {e}"}
        except Exception as e:
            return {"success": False, "error": f"VRAM check failed: {e}"}
