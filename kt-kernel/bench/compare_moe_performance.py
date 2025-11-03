#!/usr/bin/env python
# coding=utf-8
"""
MoE Performance Comparison Script
Compares performance between KTransformers AMX MoE and SGL CPU MoE implementations
"""
import os
import sys
import time
import json
import platform
import subprocess
import argparse
import logging
import signal
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment configuration
@dataclass
class EnvironmentConfig:
    malloc_conf: str = "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1"
    jemalloc_path: str = "/home/xwy/Projects/jemalloc/lib/libjemalloc.so"
    
    def apply(self):
        os.environ['MALLOC_CONF'] = self.malloc_conf
        if os.path.exists(self.jemalloc_path):
            os.environ['LD_PRELOAD'] = self.jemalloc_path
        else:
            logger.warning(f"jemalloc not found at {self.jemalloc_path}")

# Apply environment configuration
env_config = EnvironmentConfig()
env_config.apply()

# Add paths for both implementations
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
sys.path.insert(0, '/home/xwy/Projects/sgl-cpu-tests')

import torch

# Try importing both implementations
try:
    import kt_kernel_ext
    KTRANSFORMERS_AVAILABLE = True
    logger.info("KTransformers kt_kernel_ext loaded successfully")
except ImportError as e:
    KTRANSFORMERS_AVAILABLE = False
    logger.warning(f"KTransformers kt_kernel_ext not available: {e}")

try:
    from sgl_kernel.common_ops import fused_experts_cpu
    from sgl_kernel.common_ops import convert_weight_packed
    SGL_AVAILABLE = True
    logger.info("SGL kernel loaded successfully")
except ImportError as e:
    SGL_AVAILABLE = False
    logger.warning(f"SGL kernel not available: {e}")

# Try importing int4 support
try:
    # For SGL INT4, we'll check if the sglang-jianan directory exists
    import os
    sglang_path = "/home/xwy/Projects/sglang-jianan"
    if os.path.exists(sglang_path) and os.path.exists(os.path.join(sglang_path, "benchmark/kernels/int4_moe/benchmark_int4_moe.py")):
        SGL_INT4_AVAILABLE = True
        logger.info("SGL INT4 support available (via sglang-jianan)")
    else:
        SGL_INT4_AVAILABLE = False
        logger.warning("SGL INT4 support not available: sglang-jianan directory not found")
except Exception as e:
    SGL_INT4_AVAILABLE = False
    logger.warning(f"SGL INT4 support not available: {e}")

def get_cpu_count() -> int:
    """Get logical CPU core count (including hyperthreading)"""
    cpu_count = None
    
    # Method 1: os.cpu_count()
    try:
        cpu_count = os.cpu_count()
        if cpu_count and cpu_count > 0:
            logger.info(f"Detected {cpu_count} logical CPU cores via os.cpu_count()")
            return cpu_count
    except Exception as e:
        logger.debug(f"os.cpu_count() failed: {e}")
    
    # Method 2: Check /proc/cpuinfo
    try:
        with open('/proc/cpuinfo', 'r') as f:
            cpu_count = sum(1 for line in f if line.strip().startswith('processor'))
        if cpu_count > 0:
            logger.info(f"Detected {cpu_count} logical CPU cores via /proc/cpuinfo")
            return cpu_count
    except Exception as e:
        logger.debug(f"Failed to read /proc/cpuinfo: {e}")
    
    # Default fallback
    logger.warning("Could not detect CPU count, defaulting to 32")
    return 32

def get_physical_cpu_count() -> int:
    """Get physical CPU core count (excluding hyperthreading)"""
    
    # Method 1: Try lscpu command
    try:
        result = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            cores_per_socket = None
            sockets = None
            for line in result.stdout.split('\n'):
                if 'Core(s) per socket:' in line:
                    cores_per_socket = int(line.split(':')[1].strip())
                elif 'Socket(s):' in line:
                    sockets = int(line.split(':')[1].strip())
            
            if cores_per_socket and sockets:
                physical_cores = cores_per_socket * sockets
                logger.info(f"Detected {physical_cores} physical CPU cores via lscpu")
                return physical_cores
    except Exception as e:
        logger.debug(f"lscpu failed: {e}")
    
    # Method 2: Check /sys/devices/system/cpu/
    try:
        cpu_path = '/sys/devices/system/cpu/'
        if os.path.exists(cpu_path):
            # Count unique physical core IDs
            physical_cores = set()
            for cpu_dir in os.listdir(cpu_path):
                if cpu_dir.startswith('cpu') and cpu_dir[3:].isdigit():
                    core_id_path = os.path.join(cpu_path, cpu_dir, 'topology/core_id')
                    if os.path.exists(core_id_path):
                        with open(core_id_path, 'r') as f:
                            core_id = f.read().strip()
                            physical_cores.add(core_id)
            
            if physical_cores:
                count = len(physical_cores)
                logger.info(f"Detected {count} physical CPU cores via sysfs")
                return count
    except Exception as e:
        logger.debug(f"Failed to check sysfs: {e}")
    
    # Method 3: Parse /proc/cpuinfo for unique core ids
    try:
        with open('/proc/cpuinfo', 'r') as f:
            content = f.read()
            cores = set()
            current_physical_id = None
            
            for line in content.split('\n'):
                if line.startswith('physical id'):
                    current_physical_id = line.split(':')[1].strip()
                elif line.startswith('core id') and current_physical_id is not None:
                    core_id = line.split(':')[1].strip()
                    cores.add(f"{current_physical_id}:{core_id}")
            
            if cores:
                count = len(cores)
                logger.info(f"Detected {count} physical CPU cores via /proc/cpuinfo")
                return count
    except Exception as e:
        logger.debug(f"Failed to parse /proc/cpuinfo: {e}")
    
    # Fallback: assume hyperthreading is enabled and divide logical cores by 2
    try:
        logical_count = get_cpu_count()
        if logical_count > 0:
            # Assume hyperthreading, so physical cores = logical cores / 2
            physical_count = logical_count // 2
            logger.warning(f"Could not detect physical cores directly. Assuming hyperthreading enabled: {logical_count} logical cores -> {physical_count} physical cores")
            return physical_count
    except:
        pass
    
    # Default fallback
    logger.warning("Could not detect physical CPU count, defaulting to 32")
    return 32

# Test configuration dataclass
@dataclass
class TestConfig:
    expert_num: int = 256
    hidden_size: int = 7168
    intermediate_size: int = 2048
    max_len: int = 25600
    num_experts_per_tok: int = 8
    layer_num: int = 5
    warm_up_iter: int = 100
    test_iter: int = 10000
    qlen_values: List[int] = None
    thread_count_values: List[int] = None
    
    def __post_init__(self):
        if self.qlen_values is None:
            self.qlen_values = [1, 4, 16, 64, 256, 1024, 2048]
        if self.thread_count_values is None:
            # Default to physical CPU core count
            physical_cores = get_physical_cpu_count()
            self.thread_count_values = [physical_cores]
    
    @property
    def total_configurations(self) -> int:
        return len(self.qlen_values) * len(self.thread_count_values)

def get_numa_count() -> int:
    """Get NUMA node count from system with multiple fallback methods"""
    # Method 1: Try numactl
    try:
        result = subprocess.run(['numactl', '--hardware'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'available:' in line and 'nodes' in line:
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].isdigit():
                        numa_count = int(parts[1])
                        logger.info(f"Detected {numa_count} NUMA nodes via numactl")
                        return numa_count
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.debug(f"numactl not available: {e}")
    
    # Method 2: Check /sys/devices/system/node/
    try:
        node_path = '/sys/devices/system/node/'
        if os.path.exists(node_path):
            numa_dirs = [d for d in os.listdir(node_path) if d.startswith('node')]
            if numa_dirs:
                numa_count = len(numa_dirs)
                logger.info(f"Detected {numa_count} NUMA nodes via sysfs")
                return numa_count
    except Exception as e:
        logger.debug(f"Failed to check sysfs: {e}")
    
    # Default fallback
    logger.warning("Could not detect NUMA configuration, defaulting to 2 nodes")
    return 2

# System configuration
@dataclass
class SystemConfig:
    numa_count: int = 0
    cpu_cores: int = 0
    
    def __post_init__(self):
        if self.numa_count == 0:
            self.numa_count = get_numa_count()
        if self.cpu_cores == 0:
            self.cpu_cores = get_cpu_count()

sys_config = SystemConfig()

@dataclass
class ThreadConfig:
    thread_count: int
    threads_per_numa: int
    sgl_thread_count: int
    numa_prefix: str
    
    @classmethod
    def from_thread_count(cls, thread_count: int, numa_count: int, cpu_cores: int) -> 'ThreadConfig':
        """Create thread configuration for a specific thread count"""
        # Validate thread count
        if thread_count > cpu_cores:
            logger.warning(f"thread_count ({thread_count}) > cpu_cores ({cpu_cores}), using all cores")
            thread_count = cpu_cores
        
        threads_per_numa = thread_count // numa_count
        sgl_thread_count = threads_per_numa
        last_core = sgl_thread_count - 1
        numa_prefix = f"numactl --physcpubind=0-{last_core} --membind=0"
        
        return cls(
            thread_count=thread_count,
            threads_per_numa=threads_per_numa,
            sgl_thread_count=sgl_thread_count,
            numa_prefix=numa_prefix
        )

def get_system_info() -> Dict[str, any]:
    """Get comprehensive system information"""
    info = {}
    
    # Basic system info
    uname = platform.uname()
    info["system_name"] = uname.system
    info["node_name"] = uname.node
    info["release"] = uname.release
    info["machine"] = uname.machine
    info["cpu_count"] = sys_config.cpu_cores
    info["numa_nodes"] = sys_config.numa_count
    
    # CPU model information
    if os.path.exists('/proc/cpuinfo'):
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpu_info = f.read()
                for line in cpu_info.split('\n'):
                    if "model name" in line:
                        info["cpu_model"] = line.split(":", 1)[1].strip()
                        break
                # Check for CPU features
                if "flags" in cpu_info:
                    flags_line = next(line for line in cpu_info.split('\n') if "flags" in line)
                    flags = flags_line.split(":", 1)[1].strip().split()
                    info["cpu_features"] = {
                        "avx2": "avx2" in flags,
                        "avx512": any(f.startswith("avx512") for f in flags),
                        "amx": any("amx" in f for f in flags)
                    }
        except Exception as e:
            logger.debug(f"Failed to read CPU info: {e}")
    
    # Memory information
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["total_memory_gb"] = round(mem.total / (1024**3), 2)
        info["available_memory_gb"] = round(mem.available / (1024**3), 2)
    except ImportError:
        pass
    
    # Python and PyTorch versions
    info["python_version"] = sys.version.split()[0]
    info["torch_version"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
    
    return info

@dataclass
class BenchmarkResult:
    implementation: str
    quant_mode: str
    qlen: int
    thread_count: int
    total_time: float
    time_per_iter_us: float
    bandwidth_gbs: float
    tflops: float
    iterations: int
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class CheckpointState:
    """State information for checkpoint/resume functionality"""
    test_config: TestConfig
    completed_configs: List[Tuple[int, int, str, str]]  # (thread_count, qlen, implementation, quant_mode)
    results: List[BenchmarkResult]
    start_time: str
    last_update: str
    
    def to_dict(self) -> Dict:
        return {
            'test_config': asdict(self.test_config),
            'completed_configs': self.completed_configs,
            'results': [r.to_dict() for r in self.results],
            'start_time': self.start_time,
            'last_update': self.last_update
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CheckpointState':
        test_config = TestConfig(**data['test_config'])
        results = [BenchmarkResult(**r) for r in data['results']]
        return cls(
            test_config=test_config,
            completed_configs=data['completed_configs'],
            results=results,
            start_time=data['start_time'],
            last_update=data['last_update']
        )

class CheckpointManager:
    """Manages checkpoint saving and loading"""
    def __init__(self, checkpoint_dir: str = None):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path.cwd() / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "moe_benchmark_checkpoint.json"
        self.interrupted = False
        
        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        logger.warning(f"Received signal {signum}, will save checkpoint after current test...")
        self.interrupted = True
    
    def save_checkpoint(self, state: CheckpointState):
        """Save checkpoint to file"""
        state.last_update = datetime.now().isoformat()
        
        # Save to temporary file first for atomicity
        temp_file = self.checkpoint_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w') as f:
                json.dump(state.to_dict(), f, indent=2)
            
            # Atomically rename
            temp_file.replace(self.checkpoint_file)
            logger.info(f"Checkpoint saved: {len(state.results)} results, {len(state.completed_configs)} configs completed")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_file.exists():
                temp_file.unlink()
    
    def load_checkpoint(self) -> Optional[CheckpointState]:
        """Load checkpoint from file if exists"""
        if not self.checkpoint_file.exists():
            return None
        
        try:
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
            state = CheckpointState.from_dict(data)
            logger.info(f"Loaded checkpoint: {len(state.results)} results, {len(state.completed_configs)} configs completed")
            logger.info(f"Checkpoint started at {state.start_time}, last updated {state.last_update}")
            return state
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def clear_checkpoint(self):
        """Remove checkpoint file"""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            logger.info("Checkpoint cleared")

def bench_ktransformers_moe(test_config: TestConfig, quant_mode: str, qlen: int, 
                           thread_config: ThreadConfig) -> Optional[BenchmarkResult]:
    """Benchmark KTransformers AMX MoE implementation"""
    if not KTRANSFORMERS_AVAILABLE:
        logger.error("KTransformers not available, skipping benchmark")
        return None
    
    # Adjust iterations based on qlen to maintain reasonable runtime
    adjusted_iterations = test_config.test_iter
    adjusted_warmup = test_config.warm_up_iter
    if qlen >= 1024:
        adjusted_iterations = max(10, test_config.test_iter // 100)
        adjusted_warmup = max(5, test_config.warm_up_iter // 20)
    elif qlen >= 256:
        adjusted_iterations = max(50, test_config.test_iter // 20)
        adjusted_warmup = max(10, test_config.warm_up_iter // 10)
    elif qlen >= 64:
        adjusted_iterations = max(100, test_config.test_iter // 10)
        adjusted_warmup = max(20, test_config.warm_up_iter // 5)
    elif qlen >= 16:
        adjusted_iterations = max(200, test_config.test_iter // 5)
        adjusted_warmup = max(40, test_config.warm_up_iter // 2)
    
    logger.info(f"Testing KTransformers MoE: quant={quant_mode}, qlen={qlen}, threads={thread_config.thread_count}, "
                f"iterations={adjusted_iterations} (warmup={adjusted_warmup})")
    
    # Set thread count for this test
    os.environ['OMP_NUM_THREADS'] = str(thread_config.thread_count)
    
    try:
        with torch.inference_mode():
            # Setup worker config with consistent threads per NUMA
            worker_config = kt_kernel_ext.WorkerPoolConfig()
            worker_config.subpool_count = sys_config.numa_count
            worker_config.subpool_numa_map = list(range(sys_config.numa_count))
            worker_config.subpool_thread_count = [thread_config.threads_per_numa] * sys_config.numa_count
            CPUInfer = kt_kernel_ext.CPUInfer(worker_config)
        
            # Create MoE layers
            moes = []
            gate_projs = []
            up_projs = []
            down_projs = []
            
            logger.debug(f"Creating {test_config.layer_num} MoE layers...")
            for i in range(test_config.layer_num):
                gate_proj = torch.randn((test_config.expert_num, test_config.intermediate_size, test_config.hidden_size), 
                                      dtype=torch.float32).contiguous()
                up_proj = torch.randn((test_config.expert_num, test_config.intermediate_size, test_config.hidden_size), 
                                    dtype=torch.float32).contiguous()
                down_proj = torch.randn((test_config.expert_num, test_config.hidden_size, test_config.intermediate_size), 
                                      dtype=torch.float32).contiguous()
            
                config = kt_kernel_ext.moe.MOEConfig(
                    test_config.expert_num, test_config.num_experts_per_tok, 
                    test_config.hidden_size, test_config.intermediate_size)
                config.max_len = test_config.max_len
                config.gate_proj = gate_proj.data_ptr()
                config.up_proj = up_proj.data_ptr()
                config.down_proj = down_proj.data_ptr()
                config.pool = CPUInfer.backend_
            
                if quant_mode == "bf16":
                    moe = kt_kernel_ext.moe.AMXBF16_MOE(config)
                elif quant_mode == "int8":
                    moe = kt_kernel_ext.moe.AMXInt8_MOE(config)
                elif quant_mode == "int4":
                    moe = kt_kernel_ext.moe.AMXInt4_MOE(config)
                else:
                    raise ValueError(f"Unsupported quantization mode: {quant_mode}")
                
                CPUInfer.submit(moe.load_weights_task())
                CPUInfer.sync()
                gate_projs.append(gate_proj)
                up_projs.append(up_proj)
                down_projs.append(down_proj)
                moes.append(moe)
        
            # Prepare test data
            logger.debug("Preparing test data...")
            gen_iter = 1000
            expert_ids = torch.rand(gen_iter * qlen, test_config.expert_num).argsort(dim=-1)[
                :, :test_config.num_experts_per_tok
            ].reshape(gen_iter, qlen * test_config.num_experts_per_tok).contiguous()
            
            weights = torch.rand((gen_iter, qlen, test_config.num_experts_per_tok), 
                               dtype=torch.float32).contiguous()
            input_tensor = torch.randn((test_config.layer_num, qlen, test_config.hidden_size), 
                                     dtype=torch.bfloat16).contiguous()
            output_tensor = torch.empty((test_config.layer_num, qlen, test_config.hidden_size), 
                                      dtype=torch.bfloat16).contiguous()
            bsz_tensor = torch.tensor([qlen], dtype=torch.int32)
        
            # Warmup
            logger.debug(f"Running {adjusted_warmup} warmup iterations...")
            for i in range(adjusted_warmup):
                layer_idx = i % test_config.layer_num
                gen_idx = i % gen_iter
                CPUInfer.submit(
                    moes[layer_idx].forward_task(
                        bsz_tensor.data_ptr(),
                        test_config.num_experts_per_tok,
                        expert_ids[gen_idx].data_ptr(),
                        weights[gen_idx].data_ptr(),
                        input_tensor[layer_idx].data_ptr(),
                        output_tensor[layer_idx].data_ptr(),
                        False,
                    )
                )
                CPUInfer.sync()
        
            # Benchmark
            logger.debug(f"Running {adjusted_iterations} benchmark iterations...")
            start = time.perf_counter()
            for i in range(adjusted_iterations):
                layer_idx = i % test_config.layer_num
                gen_idx = i % gen_iter
                CPUInfer.submit(
                    moes[layer_idx].forward_task(
                        bsz_tensor.data_ptr(),
                        test_config.num_experts_per_tok,
                        expert_ids[gen_idx].data_ptr(),
                        weights[gen_idx].data_ptr(),
                        input_tensor[layer_idx].data_ptr(),
                        output_tensor[layer_idx].data_ptr(),
                        False,
                    )
                )
                CPUInfer.sync()
            end = time.perf_counter()
        
            # Calculate metrics
            total_time = end - start
            time_per_iter_us = total_time / adjusted_iterations * 1e6
            
            # Bytes per element based on quantization
            bytes_per_elem = {
                "bf16": 2.0,
                "int8": 1.0,
                "int4": 0.5
            }.get(quant_mode, 2.0)
            
            # Memory bandwidth calculation (GB/s)
            memory_per_iter = (
                test_config.hidden_size * test_config.intermediate_size * 3 * 
                test_config.num_experts_per_tok * 
                (1/8 * test_config.expert_num * (1-(31/32)**qlen)) * bytes_per_elem
            )
            bandwidth_gbs = memory_per_iter * adjusted_iterations / total_time / 1e9
            
            # FLOPS calculation (TFLOPS)
            flops_per_iter = (
                test_config.hidden_size * test_config.intermediate_size * qlen * 3 * 
                test_config.num_experts_per_tok * 2
            )
            tflops = flops_per_iter * adjusted_iterations / total_time / 1e12
            
            logger.info(f"Results - Time: {total_time:.4f}s, Per-iter: {time_per_iter_us:.2f}μs, "
                       f"BW: {bandwidth_gbs:.2f} GB/s, TFLOPS: {tflops:.2f}")
            
            return BenchmarkResult(
                implementation="KTransformers",
                quant_mode=quant_mode,
                qlen=qlen,
                thread_count=thread_config.thread_count,
                total_time=total_time,
                time_per_iter_us=time_per_iter_us,
                bandwidth_gbs=bandwidth_gbs,
                tflops=tflops,
                iterations=adjusted_iterations
            )
            
    except Exception as e:
        logger.error(f"KTransformers benchmark failed: {e}", exc_info=True)
        return None

def run_sgl_int4_with_numactl(test_config: TestConfig, qlen: int, 
                             thread_config: ThreadConfig) -> Optional[BenchmarkResult]:
    """Run SGL INT4 benchmark with numactl in subprocess"""
    if not SGL_INT4_AVAILABLE:
        logger.error("SGL INT4 not available, skipping benchmark")
        return None
    
    # Calculate SGL intermediate size (divided by NUMA nodes)
    sgl_intermediate_size = test_config.intermediate_size // sys_config.numa_count
    
    # Adjust iterations based on qlen to maintain reasonable runtime
    adjusted_iterations = test_config.test_iter
    adjusted_warmup = test_config.warm_up_iter
    if qlen >= 1024:
        adjusted_iterations = max(10, test_config.test_iter // 100)
        adjusted_warmup = max(5, test_config.warm_up_iter // 20)
    elif qlen >= 256:
        adjusted_iterations = max(50, test_config.test_iter // 20)
        adjusted_warmup = max(10, test_config.warm_up_iter // 10)
    elif qlen >= 64:
        adjusted_iterations = max(100, test_config.test_iter // 10)
        adjusted_warmup = max(20, test_config.warm_up_iter // 5)
    elif qlen >= 16:
        adjusted_iterations = max(200, test_config.test_iter // 5)
        adjusted_warmup = max(40, test_config.warm_up_iter // 2)
    
    logger.info(f"Testing SGL INT4: qlen={qlen}, iterations={adjusted_iterations} (warmup={adjusted_warmup}), "
                f"threads per NUMA: {thread_config.sgl_thread_count}")
    
    script_content = f'''
import sys
sys.path.insert(0, '/home/xwy/Projects/sglang-jianan')
sys.path.insert(0, '/home/xwy/Projects/sglang-jianan/test')

import os
import torch
import numpy as np
import sgl_kernel
from srt.cpu.utils import autoawq_to_int4pack
import time

torch.manual_seed(1111)
M, N, K, E, topk = {qlen}, {sgl_intermediate_size}, {test_config.hidden_size}, {test_config.expert_num}, {test_config.num_experts_per_tok}
layer_num = {test_config.layer_num}
group_size = 128
kernel = torch.ops.sgl_kernel

# Prepare int4 data
dtype = torch.bfloat16
device = "cpu"

# Generate input activations for all layers
input_tensors = [torch.rand(M, K, dtype=dtype, device=device) / np.sqrt(K) for _ in range(layer_num)]

# Generate weights and pack for each layer
all_awq_w13_weight_pack = []
all_awq_w13_zero_pack = []
all_awq_w13_scales_pack = []
all_awq_w2_weight_pack = []
all_awq_w2_zero_pack = []
all_awq_w2_scales_pack = []

# Generate expert routing scores (different for each iteration)
gen_iter = 1000
all_topk_weights = []
all_topk_ids = []

for gen_idx in range(gen_iter):
    score = torch.rand(M, E, dtype=dtype, device=device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    all_topk_weights.append(topk_weight)
    all_topk_ids.append(topk_ids.to(torch.int32))

print("Creating " + str(layer_num) + " MoE layers...")
for layer_idx in range(layer_num):
    # Generate INT4 quantized weights for each expert
    # w1: gate and up projection (K -> 2*N)
    awq_w13_weight = torch.randint(-127, 128, (E, K, 2 * N // 8), device=device).to(torch.int)
    awq_w13_zero = torch.randint(0, 10, (E, K // group_size, 2 * N // 8), device=device).to(torch.int)
    awq_w13_scales = torch.rand(E, K // group_size, 2 * N, dtype=dtype, device=device)
    
    # w2: down projection (N -> K)  
    awq_w2_weight = torch.randint(-127, 128, (E, N, K // 8), device=device).to(torch.int)
    awq_w2_zero = torch.randint(0, 10, (E, N // group_size, K // 8), device=device).to(torch.int)
    awq_w2_scales = torch.rand(E, N // group_size, K, dtype=dtype, device=device)
    
    # Pack weights for optimized kernel
    awq_w13_weight_pack = []
    awq_w13_zero_pack = []
    awq_w13_scales_pack = []
    awq_w2_weight_pack = []
    awq_w2_zero_pack = []
    awq_w2_scales_pack = []
    
    for i in range(E):
        packed_weight_13, packed_zero_13, packed_scales_13 = autoawq_to_int4pack(
            awq_w13_weight[i], awq_w13_zero[i], awq_w13_scales[i], False
        )
        awq_w13_weight_pack.append(packed_weight_13)
        awq_w13_zero_pack.append(packed_zero_13)
        awq_w13_scales_pack.append(packed_scales_13)
        
        packed_weight_2, packed_zero_2, packed_scales_2 = autoawq_to_int4pack(
            awq_w2_weight[i], awq_w2_zero[i], awq_w2_scales[i], False
        )
        awq_w2_weight_pack.append(packed_weight_2)
        awq_w2_zero_pack.append(packed_zero_2)
        awq_w2_scales_pack.append(packed_scales_2)
    
    all_awq_w13_weight_pack.append(torch.stack(awq_w13_weight_pack).detach())
    all_awq_w13_zero_pack.append(torch.stack(awq_w13_zero_pack).detach())
    all_awq_w13_scales_pack.append(torch.stack(awq_w13_scales_pack).detach())
    all_awq_w2_weight_pack.append(torch.stack(awq_w2_weight_pack).detach())
    all_awq_w2_zero_pack.append(torch.stack(awq_w2_zero_pack).detach())
    all_awq_w2_scales_pack.append(torch.stack(awq_w2_scales_pack).detach())

# Warmup
print("Running " + str({adjusted_warmup}) + " warmup iterations...")
for i in range({adjusted_warmup}):
    layer_idx = i % layer_num
    gen_idx = i % gen_iter
    out = kernel.fused_experts_cpu(
        input_tensors[layer_idx],
        all_awq_w13_weight_pack[layer_idx],
        all_awq_w2_weight_pack[layer_idx],
        all_topk_weights[gen_idx],
        all_topk_ids[gen_idx],
        False,  # inplace
        False,  # use_int8_w8a8
        False,  # use_fp8_w8a16
        True,   # use_int4_w4a16
        all_awq_w13_scales_pack[layer_idx],
        all_awq_w2_scales_pack[layer_idx],
        all_awq_w13_zero_pack[layer_idx],
        all_awq_w2_zero_pack[layer_idx],
        None,   # block_size
        None,   # a1_scale
        None,   # a2_scale
        True,   # is_vnni
    )

# Benchmark
print("Running " + str({adjusted_iterations}) + " benchmark iterations...")
start = time.perf_counter()
for i in range({adjusted_iterations}):
    layer_idx = i % layer_num
    gen_idx = i % gen_iter
    out = kernel.fused_experts_cpu(
        input_tensors[layer_idx],
        all_awq_w13_weight_pack[layer_idx],
        all_awq_w2_weight_pack[layer_idx],
        all_topk_weights[gen_idx],
        all_topk_ids[gen_idx],
        False,
        False,
        False,
        True,
        all_awq_w13_scales_pack[layer_idx],
        all_awq_w2_scales_pack[layer_idx],
        all_awq_w13_zero_pack[layer_idx],
        all_awq_w2_zero_pack[layer_idx],
        None,
        None,
        None,
        True,
    )
end = time.perf_counter()

total_time = end - start
time_per_iter_us = total_time / {adjusted_iterations} * 1e6

# Calculate performance metrics for int4
bytes_per_elem = 0.5  # int4
memory_per_iter = (
    {test_config.hidden_size} * {sgl_intermediate_size} * 3 * {test_config.num_experts_per_tok} * 
    (1/8 * {test_config.expert_num} * (1-(31/32)**{qlen})) * bytes_per_elem
)
bandwidth_gbs = memory_per_iter * {adjusted_iterations} / total_time / 1e9

# FLOPS calculation 
flops_per_iter = {test_config.hidden_size} * {sgl_intermediate_size} * {qlen} * 3 * {test_config.num_experts_per_tok} * 2
tflops = flops_per_iter * {adjusted_iterations} / total_time / 1e12

print(f"SGL_RESULT:{{total_time}},{{time_per_iter_us}},{{bandwidth_gbs}},{{tflops}}")
'''
    
    # Create temporary script in sglang-jianan directory
    sglang_path = "/home/xwy/Projects/sglang-jianan"
    temp_script = f"{sglang_path}/temp_sgl_int4_bench_{os.getpid()}_{qlen}.py"
    
    try:
        with open(temp_script, 'w') as f:
            f.write(script_content)
        
        # Setup environment
        env = os.environ.copy()
        env['MALLOC_CONF'] = env_config.malloc_conf
        if os.path.exists(env_config.jemalloc_path):
            env['LD_PRELOAD'] = env_config.jemalloc_path
        env['OMP_NUM_THREADS'] = str(thread_config.sgl_thread_count)
        
        # Run with numactl from the sglang-jianan directory
        cmd = f"cd {sglang_path} && {thread_config.numa_prefix} python3 {temp_script}"
        logger.debug(f"Running SGL INT4 command: {cmd}")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, timeout=300)
        
        if result.returncode == 0:
            # Parse result
            for line in result.stdout.split('\n'):
                if line.startswith('SGL_RESULT:'):
                    parts = line.replace('SGL_RESULT:', '').split(',')
                    if len(parts) >= 4:
                        try:
                            total_time = float(parts[0])
                            time_per_iter_us = float(parts[1])
                            bandwidth_gbs = float(parts[2])
                            tflops = float(parts[3])
                            
                            logger.info(f"SGL INT4 Results - Time: {total_time:.4f}s, Per-iter: {time_per_iter_us:.2f}μs, "
                                       f"BW: {bandwidth_gbs:.2f} GB/s, TFLOPS: {tflops:.2f}")
                            
                            return BenchmarkResult(
                                implementation="SGL",
                                quant_mode="int4",
                                qlen=qlen,
                                thread_count=thread_config.thread_count,
                                total_time=total_time,
                                time_per_iter_us=time_per_iter_us,
                                bandwidth_gbs=bandwidth_gbs,
                                tflops=tflops,
                                iterations=adjusted_iterations
                            )
                        except ValueError as e:
                            logger.error(f"Failed to parse SGL INT4 results: {e}")
        else:
            logger.error(f"SGL INT4 subprocess failed with code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error("SGL INT4 benchmark timed out")
    except Exception as e:
        logger.error(f"SGL INT4 benchmark error: {e}", exc_info=True)
    finally:
        # Clean up
        if os.path.exists(temp_script):
            try:
                os.remove(temp_script)
            except:
                pass
    
    return None

def run_sgl_with_numactl(test_config: TestConfig, qlen: int, 
                        thread_config: ThreadConfig) -> Optional[BenchmarkResult]:
    """Run SGL benchmark with numactl in subprocess"""
    if not SGL_AVAILABLE:
        logger.error("SGL not available, skipping benchmark")
        return None
    
    # Calculate SGL intermediate size (divided by NUMA nodes)
    sgl_intermediate_size = test_config.intermediate_size // sys_config.numa_count
    
    # Adjust iterations based on qlen to maintain reasonable runtime
    adjusted_iterations = test_config.test_iter
    adjusted_warmup = test_config.warm_up_iter
    if qlen >= 1024:
        adjusted_iterations = max(10, test_config.test_iter // 100)
        adjusted_warmup = max(5, test_config.warm_up_iter // 20)
    elif qlen >= 256:
        adjusted_iterations = max(50, test_config.test_iter // 20)
        adjusted_warmup = max(10, test_config.warm_up_iter // 10)
    elif qlen >= 64:
        adjusted_iterations = max(100, test_config.test_iter // 10)
        adjusted_warmup = max(20, test_config.warm_up_iter // 5)
    elif qlen >= 16:
        adjusted_iterations = max(200, test_config.test_iter // 5)
        adjusted_warmup = max(40, test_config.warm_up_iter // 2)
    
    logger.info(f"Testing SGL INT8: qlen={qlen}, iterations={adjusted_iterations} (warmup={adjusted_warmup}), "
                f"threads per NUMA: {thread_config.sgl_thread_count}")
    
    script_content = f'''
import sys
sys.path.insert(0, "/home/xwy/Projects/sgl-cpu-tests")

import os
import torch
from sgl_kernel.common_ops import fused_experts_cpu as fused_experts
from sgl_kernel.common_ops import convert_weight_packed
import time

torch.manual_seed(1111)
M, N, K, E, topk = {qlen}, {sgl_intermediate_size}, {test_config.hidden_size}, {test_config.expert_num}, {test_config.num_experts_per_tok}
layer_num = {test_config.layer_num}

# Generate expert routing scores (different for each iteration)
gen_iter = 1000
all_topk_weights = []
all_topk_ids = []

for gen_idx in range(gen_iter):
    score = torch.randn(M, E).to(dtype=torch.bfloat16)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    all_topk_weights.append(topk_weight)
    all_topk_ids.append(topk_ids.to(torch.int32))

prepack = True
inplace = True
use_int4_w4a16 = False

# Create multiple layers
print("Creating " + str(layer_num) + " MoE layers...")
inputs = []
packed_w1s_int8 = []
packed_w2s_int8 = []
w1_s_list = []
w2_s_list = []

for layer_idx in range(layer_num):
    input_tensor = torch.randn(M, K).to(dtype=torch.bfloat16)
    
    # int8 weights
    w1_int8 = torch.randn(E, 2 * N, K).to(dtype=torch.int8)
    w2_int8 = torch.randn(E, K, N).to(dtype=torch.int8)
    packed_w1_int8 = convert_weight_packed(w1_int8)
    packed_w2_int8 = convert_weight_packed(w2_int8)
    w1_s = torch.rand(E, 2 * N)
    w2_s = torch.rand(E, K)
    
    inputs.append(input_tensor)
    packed_w1s_int8.append(packed_w1_int8)
    packed_w2s_int8.append(packed_w2_int8)
    w1_s_list.append(w1_s)
    w2_s_list.append(w2_s)

# Warmup
print("Running " + str({adjusted_warmup}) + " warmup iterations...")
for i in range({adjusted_warmup}):
    layer_idx = i % layer_num
    gen_idx = i % gen_iter
    fused_experts(inputs[layer_idx], packed_w1s_int8[layer_idx], packed_w2s_int8[layer_idx], 
                 all_topk_weights[gen_idx], all_topk_ids[gen_idx],
                 inplace, True, False, use_int4_w4a16, w1_s_list[layer_idx], w2_s_list[layer_idx], 
                 None, None, None, None, None, prepack)

# Benchmark
print("Running " + str({adjusted_iterations}) + " benchmark iterations...")
start = time.perf_counter()
for i in range({adjusted_iterations}):
    layer_idx = i % layer_num
    gen_idx = i % gen_iter
    fused_experts(inputs[layer_idx], packed_w1s_int8[layer_idx], packed_w2s_int8[layer_idx], 
                 all_topk_weights[gen_idx], all_topk_ids[gen_idx],
                 inplace, True, False, use_int4_w4a16, w1_s_list[layer_idx], w2_s_list[layer_idx], 
                 None, None, None, None, None, prepack)
end = time.perf_counter()

total_time = end - start
time_per_iter_us = total_time / {adjusted_iterations} * 1e6

# Calculate performance metrics for int8
bytes_per_elem = 1.0  # int8
memory_per_iter = (
    {test_config.hidden_size} * {sgl_intermediate_size} * 3 * {test_config.num_experts_per_tok} * 
    (1/8 * {test_config.expert_num} * (1-(31/32)**{qlen})) * bytes_per_elem
)
bandwidth_gbs = memory_per_iter * {adjusted_iterations} / total_time / 1e9

# FLOPS calculation 
flops_per_iter = {test_config.hidden_size} * {sgl_intermediate_size} * {qlen} * 3 * {test_config.num_experts_per_tok} * 2
tflops = flops_per_iter * {adjusted_iterations} / total_time / 1e12

print(f"SGL_RESULT:{{total_time}},{{time_per_iter_us}},{{bandwidth_gbs}},{{tflops}}")
'''
    
    # Create temporary script
    temp_script = f"/tmp/sgl_bench_{os.getpid()}_{qlen}.py"
    
    try:
        with open(temp_script, 'w') as f:
            f.write(script_content)
        
        # Setup environment
        env = os.environ.copy()
        env['MALLOC_CONF'] = env_config.malloc_conf
        if os.path.exists(env_config.jemalloc_path):
            env['LD_PRELOAD'] = env_config.jemalloc_path
        env['OMP_NUM_THREADS'] = str(thread_config.sgl_thread_count)
        
        # Run with numactl
        cmd = f"{thread_config.numa_prefix} python3 {temp_script}"
        logger.debug(f"Running SGL command: {cmd}")
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env, timeout=300)
        
        if result.returncode == 0:
            # Parse result
            for line in result.stdout.split('\n'):
                if line.startswith('SGL_RESULT:'):
                    parts = line.replace('SGL_RESULT:', '').split(',')
                    if len(parts) >= 4:
                        try:
                            total_time = float(parts[0])
                            time_per_iter_us = float(parts[1])
                            bandwidth_gbs = float(parts[2])
                            tflops = float(parts[3])
                            
                            logger.info(f"SGL Results - Time: {total_time:.4f}s, Per-iter: {time_per_iter_us:.2f}μs, "
                                       f"BW: {bandwidth_gbs:.2f} GB/s, TFLOPS: {tflops:.2f}")
                            
                            return BenchmarkResult(
                                implementation="SGL",
                                quant_mode="int8",
                                qlen=qlen,
                                thread_count=thread_config.thread_count,
                                total_time=total_time,
                                time_per_iter_us=time_per_iter_us,
                                bandwidth_gbs=bandwidth_gbs,
                                tflops=tflops,
                                iterations=adjusted_iterations
                            )
                        except ValueError as e:
                            logger.error(f"Failed to parse SGL results: {e}")
        else:
            logger.error(f"SGL subprocess failed with code {result.returncode}: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logger.error("SGL benchmark timed out")
    except Exception as e:
        logger.error(f"SGL benchmark error: {e}", exc_info=True)
    finally:
        # Clean up
        if os.path.exists(temp_script):
            try:
                os.remove(temp_script)
            except:
                pass
    
    return None

def save_results(results: List[BenchmarkResult], test_config: TestConfig, filename: str = None) -> str:
    """Save benchmark results to JSON file"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"moe_comparison_{timestamp}.json"
    
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "test_configuration": asdict(test_config),
        "system_info": get_system_info(),
        "results": [r.to_dict() for r in results],
        "summary": {
            "total_benchmarks": len(results),
            "implementations_tested": list(set(r.implementation for r in results)),
            "quantization_modes": list(set(r.quant_mode for r in results)),
            "qlen_values_tested": sorted(set(r.qlen for r in results)),
            "thread_counts_tested": sorted(set(r.thread_count for r in results))
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"Results saved to: {filename}")
    return filename

def print_summary_table(results: List[BenchmarkResult]):
    """Print formatted summary table of results"""
    if not results:
        return
    
    print("\n" + "=" * 100)
    print("PERFORMANCE SUMMARY")
    print("=" * 100)
    print(f"{'Implementation':<15} {'Quant':<6} {'Threads':<8} {'QLen':<8} {'Time(μs)':<12} {'BW(GB/s)':<12} {'TFLOPS':<10} {'Speedup':<10}")
    print("-" * 100)
    
    # Group by configuration for better comparison
    baseline_times = {}
    
    for result in sorted(results, key=lambda r: (r.thread_count, r.qlen, r.implementation, r.quant_mode)):
        key = (result.thread_count, result.qlen)
        
        if key not in baseline_times:
            baseline_times[key] = result.time_per_iter_us
            speedup = "1.00x"
        else:
            speedup = f"{baseline_times[key]/result.time_per_iter_us:.2f}x"
        
        print(f"{result.implementation:<15} {result.quant_mode:<6} {result.thread_count:<8} "
              f"{result.qlen:<8} {result.time_per_iter_us:<12.2f} {result.bandwidth_gbs:<12.2f} "
              f"{result.tflops:<10.2f} {speedup:<10}")

def main():
    parser = argparse.ArgumentParser(description="Compare MoE performance between KTransformers and SGL")
    parser.add_argument("--qlen", type=int, nargs="+", help="Sequence lengths to test")
    parser.add_argument("--threads", type=int, nargs="+", help="Thread counts to test")
    parser.add_argument("--iterations", type=int, help="Number of test iterations")
    parser.add_argument("--warmup", type=int, help="Number of warmup iterations")
    parser.add_argument("--output", type=str, help="Output filename for results")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory for checkpoint files")
    parser.add_argument("--no-checkpoint", action="store_true", help="Disable checkpoint saving")
    parser.add_argument("--framework", choices=["all", "ktransformers", "sgl"], default="all",
                        help="Framework to test (default: all)")
    parser.add_argument("--precision", choices=["all", "int8", "int4"], default="all",
                        help="Precision to test (default: all)")
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create test configuration
    test_config = TestConfig()
    if args.qlen:
        test_config.qlen_values = args.qlen
    if args.threads:
        test_config.thread_count_values = args.threads
    if args.iterations:
        test_config.test_iter = args.iterations
    if args.warmup:
        test_config.warm_up_iter = args.warmup
    
    # Determine which frameworks and precisions to test
    test_ktransformers = args.framework in ["all", "ktransformers"] and KTRANSFORMERS_AVAILABLE
    test_sgl = args.framework in ["all", "sgl"] and (SGL_AVAILABLE or SGL_INT4_AVAILABLE)
    
    # Determine which precisions to test
    test_precisions = []
    if args.precision == "all":
        test_precisions = ["int8", "int4"]
    else:
        test_precisions = [args.precision]
    
    # Print configuration
    logger.info("MoE Performance Comparison")
    logger.info("=" * 60)
    logger.info(f"System configuration:")
    logger.info(f"  CPU cores: {sys_config.cpu_cores}")
    logger.info(f"  NUMA nodes: {sys_config.numa_count}")
    logger.info(f"Test parameters:")
    logger.info(f"  Expert count: {test_config.expert_num}")
    logger.info(f"  Hidden size: {test_config.hidden_size}")
    logger.info(f"  Intermediate size: {test_config.intermediate_size}")
    logger.info(f"  Experts per token: {test_config.num_experts_per_tok}")
    logger.info(f"  Test iterations: {test_config.test_iter}")
    logger.info(f"  Warmup iterations: {test_config.warm_up_iter}")
    logger.info(f"Testing configurations:")
    logger.info(f"  QLEN values: {test_config.qlen_values}")
    logger.info(f"  Thread counts: {test_config.thread_count_values}")
    logger.info(f"  Frameworks: {args.framework}")
    logger.info(f"  Precisions: {args.precision}")
    logger.info(f"  Total configs: {test_config.total_configurations}")
    print()
    
    # Check availability
    if not KTRANSFORMERS_AVAILABLE and not SGL_AVAILABLE:
        logger.error("Neither KTransformers nor SGL is available. Cannot run benchmarks.")
        return 1
    
    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(args.checkpoint_dir) if not args.no_checkpoint else None
    
    # Load checkpoint if resuming
    checkpoint_state = None
    completed_configs = set()
    all_results = []
    start_time = datetime.now().isoformat()
    
    if args.resume and checkpoint_mgr:
        checkpoint_state = checkpoint_mgr.load_checkpoint()
        if checkpoint_state:
            # Verify configuration matches
            if (checkpoint_state.test_config.qlen_values != test_config.qlen_values or
                checkpoint_state.test_config.thread_count_values != test_config.thread_count_values):
                logger.warning("Checkpoint configuration doesn't match current configuration")
                response = input("Continue with checkpoint anyway? (y/n): ")
                if response.lower() != 'y':
                    logger.info("Starting fresh run")
                    checkpoint_state = None
            
            if checkpoint_state:
                all_results = checkpoint_state.results
                completed_configs = set(checkpoint_state.completed_configs)
                start_time = checkpoint_state.start_time
                logger.info(f"Resuming from checkpoint with {len(all_results)} results")
    
    # Create checkpoint state if not loaded
    if not checkpoint_state and checkpoint_mgr:
        checkpoint_state = CheckpointState(
            test_config=test_config,
            completed_configs=[],
            results=[],
            start_time=start_time,
            last_update=start_time
        )
    
    config_count = 0
    total_configs_to_run = 0
    
    # Calculate total configs to run
    for thread_count in test_config.thread_count_values:
        for qlen in test_config.qlen_values:
            if test_ktransformers:
                for quant_mode in test_precisions:
                    if (thread_count, qlen, "KTransformers", quant_mode) not in completed_configs:
                        total_configs_to_run += 1
            if test_sgl:
                if "int8" in test_precisions and SGL_AVAILABLE:
                    if (thread_count, qlen, "SGL", "int8") not in completed_configs:
                        total_configs_to_run += 1
                if "int4" in test_precisions and SGL_INT4_AVAILABLE:
                    if (thread_count, qlen, "SGL", "int4") not in completed_configs:
                        total_configs_to_run += 1
    
    logger.info(f"Total configurations to run: {total_configs_to_run}")
    
    # Test all combinations
    for thread_count in test_config.thread_count_values:
        thread_config = ThreadConfig.from_thread_count(thread_count, sys_config.numa_count, sys_config.cpu_cores)
        logger.info(f"\nThread Configuration: {thread_count} total ({thread_config.threads_per_numa} per NUMA)")
        
        for qlen in test_config.qlen_values:
            # Check for interrupt
            if checkpoint_mgr and checkpoint_mgr.interrupted:
                logger.warning("Interrupt detected, saving checkpoint and exiting...")
                if checkpoint_state:
                    checkpoint_state.results = all_results
                    checkpoint_state.completed_configs = list(completed_configs)
                    checkpoint_mgr.save_checkpoint(checkpoint_state)
                return 2
            
            logger.info(f"\n--- Configuration: threads={thread_count}, qlen={qlen} ---")
            
            # Test KTransformers
            if test_ktransformers:
                for quant_mode in test_precisions:
                    config_key = (thread_count, qlen, "KTransformers", quant_mode)
                    if config_key in completed_configs:
                        logger.info(f"Skipping already completed: KTransformers-{quant_mode}")
                        continue
                    
                    config_count += 1
                    logger.info(f"Progress: {config_count}/{total_configs_to_run}")
                    
                    result = bench_ktransformers_moe(test_config, quant_mode, qlen, thread_config)
                    if result:
                        all_results.append(result)
                        completed_configs.add(config_key)
                        
                        # Save checkpoint after each successful test
                        if checkpoint_mgr and checkpoint_state:
                            checkpoint_state.results = all_results
                            checkpoint_state.completed_configs = list(completed_configs)
                            checkpoint_mgr.save_checkpoint(checkpoint_state)
            
            # Test SGL int8
            if test_sgl and "int8" in test_precisions and SGL_AVAILABLE:
                config_key = (thread_count, qlen, "SGL", "int8")
                if config_key in completed_configs:
                    logger.info("Skipping already completed: SGL-int8")
                    continue
                
                config_count += 1
                logger.info(f"Progress: {config_count}/{total_configs_to_run}")
                
                logger.info(f"Testing SGL MoE (int8): qlen={qlen}, threads={thread_count}")
                sgl_intermediate = test_config.intermediate_size // sys_config.numa_count
                sgl_threads_per_numa = thread_config.sgl_thread_count
                logger.info(f"Using NUMA TP: intermediate_size {test_config.intermediate_size} -> "
                           f"{sgl_intermediate} (/{sys_config.numa_count}), threads per NUMA: {sgl_threads_per_numa}")
                
                result = run_sgl_with_numactl(test_config, qlen, thread_config)
                if result:
                    all_results.append(result)
                    completed_configs.add(config_key)
                    
                    # Save checkpoint after each successful test
                    if checkpoint_mgr and checkpoint_state:
                        checkpoint_state.results = all_results
                        checkpoint_state.completed_configs = list(completed_configs)
                        checkpoint_mgr.save_checkpoint(checkpoint_state)
            
            # Test SGL int4
            if test_sgl and "int4" in test_precisions and SGL_INT4_AVAILABLE:
                config_key = (thread_count, qlen, "SGL", "int4")
                if config_key in completed_configs:
                    logger.info("Skipping already completed: SGL-int4")
                    continue
                
                config_count += 1
                logger.info(f"Progress: {config_count}/{total_configs_to_run}")
                
                logger.info(f"Testing SGL MoE (int4): qlen={qlen}, threads={thread_count}")
                sgl_intermediate = test_config.intermediate_size // sys_config.numa_count
                sgl_threads_per_numa = thread_config.sgl_thread_count
                logger.info(f"Using NUMA TP: intermediate_size {test_config.intermediate_size} -> "
                           f"{sgl_intermediate} (/{sys_config.numa_count}), threads per NUMA: {sgl_threads_per_numa}")
                
                result = run_sgl_int4_with_numactl(test_config, qlen, thread_config)
                if result:
                    all_results.append(result)
                    completed_configs.add(config_key)
                    
                    # Save checkpoint after each successful test
                    if checkpoint_mgr and checkpoint_state:
                        checkpoint_state.results = all_results
                        checkpoint_state.completed_configs = list(completed_configs)
                        checkpoint_mgr.save_checkpoint(checkpoint_state)
    
    # Final summary
    if all_results:
        print_summary_table(all_results)
        
        # Save results
        output_file = save_results(all_results, test_config, args.output)
        
        print(f"\nTotal benchmarks completed: {len(all_results)}")
        print(f"Results saved to: {output_file}")
        
        # Clear checkpoint on successful completion
        if checkpoint_mgr and config_count == total_configs_to_run:
            checkpoint_mgr.clear_checkpoint()
            logger.info("All tests completed successfully, checkpoint cleared")
        elif checkpoint_mgr and config_count < total_configs_to_run:
            logger.warning(f"Only {config_count}/{total_configs_to_run} configurations completed")
            logger.info("Checkpoint preserved for resuming")
        
        # Print best performers per configuration
        print("\nBest performers by configuration:")
        from itertools import groupby
        
        sorted_results = sorted(all_results, key=lambda r: (r.qlen, r.thread_count, r.time_per_iter_us))
        for key, group in groupby(sorted_results, key=lambda r: (r.qlen, r.thread_count)):
            qlen, threads = key
            best = next(group)
            print(f"  QLen={qlen}, Threads={threads}: {best.implementation}-{best.quant_mode} "
                  f"({best.time_per_iter_us:.2f}μs, {best.tflops:.2f} TFLOPS)")
    else:
        logger.error("No successful benchmarks completed.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())