from __future__ import annotations

from enum import IntEnum, auto
from typing import Optional, Union, List
import torch

class GPUVendor(IntEnum):
    NVIDIA = auto()
    AMD = auto()
    MooreThreads = auto()
    MetaX = auto()
    MUSA = auto()
    Unknown = auto()

class DeviceManager:
    """
    Device manager that provides a unified interface for handling different GPU vendors
    """
    def __init__(self):
        self.gpu_vendor = self._detect_gpu_vendor()
        self.available_devices = self._get_available_devices()
    
    def _detect_gpu_vendor(self) -> GPUVendor:
        """Detect GPU vendor type"""
        if not torch.cuda.is_available():
            # Check MUSA availability (assuming a musa module exists)
            try:
                import musa
                if musa.is_available():
                    return GPUVendor.MUSA
            except (ImportError, AttributeError):
                pass
            
            return GPUVendor.Unknown
        
        device_name = torch.cuda.get_device_name(0).lower()
        
        if any(name in device_name for name in ["nvidia", "geforce", "quadro", "tesla", "titan", "rtx", "gtx"]):
            return GPUVendor.NVIDIA
        elif any(name in device_name for name in ["amd", "radeon", "rx", "vega", "instinct", "firepro", "mi"]):
            return GPUVendor.AMD
        elif any(name in device_name for name in ["mthreads", "moore", "mtt"]):
            return GPUVendor.MooreThreads
        elif any(name in device_name for name in ["metax", "meta"]):
            return GPUVendor.MetaX
        elif "musa" in device_name:
            return GPUVendor.MUSA
        
        # Backend check
        try:
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                return GPUVendor.AMD
            elif hasattr(torch.version, 'cuda') and torch.version.cuda is not None:
                return GPUVendor.NVIDIA
        except:
            pass
            
        return GPUVendor.Unknown
    
    def _get_available_devices(self) -> List[int]:
        """Get list of available device indices"""
        devices = []
        
        if self.gpu_vendor == GPUVendor.NVIDIA or self.gpu_vendor == GPUVendor.AMD:
            devices = list(range(torch.cuda.device_count()))
        elif self.gpu_vendor == GPUVendor.MUSA:
            try:
                import musa
                devices = list(range(musa.device_count()))
            except (ImportError, AttributeError):
                pass
            
        return devices
    
    def get_device_str(self, device_id: Union[int, str]) -> str:
        """
        Get device string for the given device ID
        
        Args:
            device_id: Device index (0, 1, 2, etc.), -1 for CPU, or "cpu" string
            
        Returns:
            Device string representation (e.g., "cuda:0", "musa:1", "cpu")
        """
        if device_id == -1 or device_id == "cpu":
            return "cpu"
            
        if isinstance(device_id, int):
            if self.gpu_vendor == GPUVendor.NVIDIA or self.gpu_vendor == GPUVendor.AMD:
                if device_id < torch.cuda.device_count():
                    return f"cuda:{device_id}"
            elif self.gpu_vendor == GPUVendor.MUSA:
                try:
                    import musa
                    if device_id < musa.device_count():
                        return f"musa:{device_id}"
                except (ImportError, AttributeError):
                    pass
        
        return "cpu"
    
    def to_torch_device(self, device_id: Union[int, str] = 0) -> torch.device:
        """
        Convert device ID to torch.device object
        
        Args:
            device_id: Device index (0, 1, 2, etc.), -1 for CPU, or "cpu" string
            
        Returns:
            torch.device object
        """
        device_str = self.get_device_str(device_id)
        
        # Handle MUSA device
        if device_str.startswith("musa:"):
            try:
                import musa
                index = int(device_str.split(":")[-1])
                return musa.device(index)
            except (ImportError, ValueError, AttributeError):
                return torch.device("cpu")
        
        # Standard PyTorch device
        return torch.device(device_str)
    
    def move_tensor_to_device(self, tensor: torch.Tensor, device_id: Union[int, str] = 0) -> torch.Tensor:
        """
        Move tensor to specified device
        
        Args:
            tensor: PyTorch tensor to move
            device_id: Device index (0, 1, 2, etc.), -1 for CPU, or "cpu" string
            
        Returns:
            Tensor moved to the specified device
        """
        device = self.to_torch_device(device_id)
        return tensor.to(device)
    
    def is_available(self, index: int = 0) -> bool:
        """
        Check if device at specified index is available
        
        Args:
            index: Device index to check
            
        Returns:
            True if the device is available, False otherwise
        """
        if index < 0:
            return True  # CPU is always available
            
        return index in self.available_devices
    
    def get_all_devices(self) -> List[int]:
        """
        Get all available device indices
        
        Returns:
            List of available device indices (0, 1, 2, etc.)
        """
        return self.available_devices

# Create global device manager instance
device_manager = DeviceManager()

# Convenience functions
def get_device(device_id: Union[int, str] = 0) -> torch.device:
    """
    Get torch.device object for the specified device ID
    
    Args:
        device_id: Device index (0, 1, 2, etc.), -1 for CPU, or "cpu" string
        
    Returns:
        torch.device object
    """
    return device_manager.to_torch_device(device_id)

def to_device(tensor: torch.Tensor, device_id: Union[int, str] = 0) -> torch.Tensor:
    """
    Move tensor to specified device
    
    Args:
        tensor: PyTorch tensor to move
        device_id: Device index (0, 1, 2, etc.), -1 for CPU, or "cpu" string
        
    Returns:
        Tensor moved to the specified device
    """
    return device_manager.move_tensor_to_device(tensor, device_id)

# Get devices
cpu_device = get_device(-1)        # CPU using index -1
cpu_device2 = get_device("cpu")    # CPU using string "cpu"
gpu0 = get_device(0)               # First GPU

# Move tensors
x = torch.randn(3, 3)
x_gpu = to_device(x, 0)            # Move to first GPU
x_cpu1 = to_device(x, -1)          # Move to CPU using index -1
x_cpu2 = to_device(x, "cpu")       # Move to CPU using string "cpu"