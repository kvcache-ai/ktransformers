"""
Tuna engine for auto-tuning GPU experts configuration.

Automatically finds the maximum viable num-gpu-experts through binary search
by testing actual server launches with different configurations.
"""

import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from kt_kernel.cli.utils.console import console, print_error, print_info, print_warning


def get_num_experts(model_path: Path) -> int:
    """
    Get the number of experts per layer from model config.

    Args:
        model_path: Path to the model directory

    Returns:
        Number of experts per layer

    Raises:
        ValueError: If config.json not found or num_experts field missing
    """
    config_file = model_path / "config.json"

    if not config_file.exists():
        raise ValueError(f"config.json not found in {model_path}")

    try:
        config = json.loads(config_file.read_text())
    except Exception as e:
        raise ValueError(f"Failed to parse config.json: {e}")

    # Different models may use different field names
    possible_keys = [
        "num_experts_per_tok",  # DeepSeek
        "num_local_experts",  # Mixtral
        "n_routed_experts",  # Qwen
        "num_experts",  # Generic
    ]

    for key in possible_keys:
        if key in config:
            return config[key]

    raise ValueError(f"Cannot find num_experts field in {config_file}. " f"Tried: {', '.join(possible_keys)}")


def detect_oom(log_line: Optional[str]) -> bool:
    """
    Detect OOM (Out Of Memory) errors from log output.

    Args:
        log_line: A line from server output

    Returns:
        True if OOM detected, False otherwise
    """
    if log_line is None:
        return False

    log_lower = log_line.lower()

    oom_patterns = [
        "cuda out of memory",
        "out of memory",
        "outofmemoryerror",
        "oom",
        "failed to allocate",
        "cumemalloc failed",
        "cumemallocasync failed",
        "allocation failed",
    ]

    return any(pattern in log_lower for pattern in oom_patterns)


def test_config(
    num_gpu_experts: int,
    model_path: Path,
    config: dict,
    verbose: bool = False,
) -> tuple[bool, float]:
    """
    Test if a configuration with given num_gpu_experts works.

    Args:
        num_gpu_experts: Number of GPU experts to test
        model_path: Path to the model
        config: Configuration dict with all parameters
        verbose: Whether to show detailed logs

    Returns:
        (success: bool, elapsed_time: float)
        - success: True if server starts and inference works
        - elapsed_time: Time taken for the test
    """
    start_time = time.time()

    # Use random port to avoid conflicts
    test_port = random.randint(30000, 40000)

    # Build command
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model",
        str(model_path),
        "--port",
        str(test_port),
        "--host",
        "127.0.0.1",
        "--tensor-parallel-size",
        str(config["tensor_parallel_size"]),
        "--kt-num-gpu-experts",
        str(num_gpu_experts),
        "--max-total-tokens",
        str(config["max_total_tokens"]),
    ]

    # Add kt-kernel options
    if config.get("weights_path"):
        cmd.extend(["--kt-weight-path", str(config["weights_path"])])
    else:
        cmd.extend(["--kt-weight-path", str(model_path)])

    cmd.extend(
        [
            "--kt-cpuinfer",
            str(config.get("cpu_threads", 64)),
            "--kt-threadpool-count",
            str(config.get("numa_nodes", 2)),
            "--kt-method",
            config.get("kt_method", "AMXINT4"),
            "--kt-gpu-prefill-token-threshold",
            str(config.get("kt_gpu_prefill_threshold", 4096)),
        ]
    )

    # Add other SGLang options
    if config.get("attention_backend"):
        cmd.extend(["--attention-backend", config["attention_backend"]])

    cmd.extend(
        [
            "--trust-remote-code",
            "--mem-fraction-static",
            str(config.get("mem_fraction_static", 0.98)),
            "--chunked-prefill-size",
            str(config.get("chunked_prefill_size", 4096)),
            "--max-running-requests",
            str(config.get("max_running_requests", 1)),  # Use 1 for faster testing
            "--watchdog-timeout",
            str(config.get("watchdog_timeout", 3000)),
            "--enable-mixed-chunk",
            "--enable-p2p-check",
        ]
    )

    # Add disable-shared-experts-fusion if specified
    if config.get("disable_shared_experts_fusion"):
        cmd.append("--disable-shared-experts-fusion")

    # Add extra args
    if config.get("extra_args"):
        cmd.extend(config["extra_args"])

    if verbose:
        console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")

    # Start process
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=config.get("env"),
        )
    except Exception as e:
        if verbose:
            print_error(f"Failed to start process: {e}")
        return False, time.time() - start_time

    # Monitor process output
    timeout = 60  # Maximum 60 seconds to wait
    server_ready = False

    try:
        while time.time() - start_time < timeout:
            # Check if process has output
            if process.poll() is not None:
                # Process exited
                if verbose:
                    print_warning("Process exited early")
                return False, time.time() - start_time

            # Read output line (non-blocking)
            try:
                line = process.stdout.readline()
                if not line:
                    time.sleep(0.1)
                    continue

                if verbose:
                    console.print(f"[dim]{line.rstrip()}[/dim]")

                # Fast OOM detection
                if detect_oom(line):
                    if verbose:
                        print_warning(f"OOM detected: {line.rstrip()}")
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    return False, time.time() - start_time

                # Check for startup success
                if "Uvicorn running" in line or "Application startup complete" in line:
                    server_ready = True
                    break

            except Exception as e:
                if verbose:
                    print_warning(f"Error reading output: {e}")
                break

        if not server_ready:
            # Timeout or failed to start
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
            return False, time.time() - start_time

        # Server is ready, test inference
        success = test_inference(test_port, verbose=verbose)

        # Cleanup
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)

        return success, time.time() - start_time

    except KeyboardInterrupt:
        # User cancelled
        process.terminate()
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    except Exception as e:
        if verbose:
            print_error(f"Test failed with exception: {e}")
        try:
            process.terminate()
            process.wait(timeout=2)
        except:
            try:
                process.kill()
            except:
                pass
        return False, time.time() - start_time


def test_inference(port: int, verbose: bool = False) -> bool:
    """
    Test if the server can handle a simple inference request.

    Args:
        port: Server port
        verbose: Whether to show detailed logs

    Returns:
        True if inference succeeds, False otherwise
    """
    try:
        # Wait a bit for server to be fully ready
        time.sleep(2)

        # Try to import OpenAI client
        try:
            from openai import OpenAI
        except ImportError:
            if verbose:
                print_warning("OpenAI package not available, skipping inference test")
            return True  # Assume success if we can't test

        client = OpenAI(
            base_url=f"http://127.0.0.1:{port}/v1",
            api_key="test",
        )

        # Send a simple test request
        response = client.chat.completions.create(
            model="test",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1,
            temperature=0,
            timeout=10,
        )

        # Check if we got a valid response
        success = response.choices and len(response.choices) > 0 and response.choices[0].message.content is not None

        if verbose:
            if success:
                print_info(f"Inference test passed: {response.choices[0].message.content}")
            else:
                print_warning("Inference test failed: no valid response")

        return success

    except Exception as e:
        if verbose:
            print_warning(f"Inference test failed: {e}")
        return False


def find_max_gpu_experts(
    model_path: Path,
    config: dict,
    verbose: bool = False,
) -> int:
    """
    Binary search to find the maximum viable num_gpu_experts.

    Args:
        model_path: Path to the model
        config: Configuration dict
        verbose: Whether to show detailed logs

    Returns:
        Maximum number of GPU experts that works
    """
    # Get number of experts from model config
    try:
        num_experts = get_num_experts(model_path)
    except ValueError as e:
        print_error(str(e))
        raise

    console.print()
    console.print(f"Binary search range: [0, {num_experts}]")
    console.print()

    left, right = 0, num_experts
    result = 0
    iteration = 0
    total_iterations = math.ceil(math.log2(num_experts + 1))

    while left <= right:
        iteration += 1
        mid = (left + right) // 2

        console.print(f"[{iteration}/{total_iterations}] Testing gpu-experts={mid}... ", end="")

        success, elapsed = test_config(mid, model_path, config, verbose=verbose)

        if success:
            console.print(f"[green]✓ OK[/green] ({elapsed:.1f}s)")
            result = mid
            left = mid + 1
        else:
            console.print(f"[red]✗ FAILED[/red] ({elapsed:.1f}s)")
            right = mid - 1

    return result


def run_tuna(
    model_path: Path,
    tensor_parallel_size: int,
    max_total_tokens: int,
    kt_method: str,
    verbose: bool = False,
    **kwargs,
) -> int:
    """
    Run tuna auto-tuning to find optimal num_gpu_experts.

    Args:
        model_path: Path to the model
        tensor_parallel_size: Tensor parallel size
        max_total_tokens: Maximum total tokens
        kt_method: KT quantization method
        verbose: Whether to show detailed logs
        **kwargs: Additional configuration parameters

    Returns:
        Optimal num_gpu_experts value

    Raises:
        ValueError: If tuning fails completely
    """
    # Prepare configuration
    config = {
        "tensor_parallel_size": tensor_parallel_size,
        "max_total_tokens": max_total_tokens,
        "kt_method": kt_method,
        **kwargs,
    }

    # Run binary search
    try:
        result = find_max_gpu_experts(model_path, config, verbose=verbose)
    except KeyboardInterrupt:
        console.print()
        print_warning("Tuning cancelled by user")
        raise

    console.print()

    # Check if even 0 doesn't work
    if result == 0:
        console.print("[yellow]Testing if gpu-experts=0 is viable...[/yellow]")
        success, _ = test_config(0, model_path, config, verbose=verbose)

        if not success:
            # Even 0 doesn't work
            console.print()
            print_error("Failed to start server even with all experts on CPU (gpu-experts=0)")
            console.print()
            console.print("[bold]Possible reasons:[/bold]")
            console.print("  • Insufficient GPU memory for base model layers")
            console.print("  • max-total-tokens is too large for available VRAM")
            console.print("  • Tensor parallel configuration issue")
            console.print()
            console.print("[bold]Suggestions:[/bold]")
            console.print(f"  • Reduce --max-total-tokens (current: {max_total_tokens})")
            console.print(f"  • Reduce --tensor-parallel-size (current: {tensor_parallel_size})")
            console.print("  • Use more GPUs or GPUs with more VRAM")
            console.print("  • Try a smaller model")
            console.print()
            raise ValueError("Minimum GPU memory requirements not met")
        else:
            # 0 works but nothing more
            console.print()
            print_warning("All experts will run on CPU (gpu-experts=0). " "Performance will be limited by CPU speed.")

    return result
