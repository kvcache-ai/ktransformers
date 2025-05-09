from abc import ABC, abstractmethod
import os
import torch
import numpy as np
from safetensors import safe_open
from typing import Dict, Any, Optional, Union

class ModelLoader(ABC):
    """
    Abstract base class for model loaders.
    Defines the interface that all model loaders must implement.
    """
    
    @abstractmethod
    def load_tensor(self, name: str, device: str = "cpu") -> torch.Tensor:
        """
        Load a tensor by name.
        
        Args:
            name: Name of the tensor to load
            device: Device to load the tensor to
            
        Returns:
            The loaded tensor
        """
        pass
    
    @classmethod
    @abstractmethod
    def supports_format(cls, path: str) -> bool:
        """
        Check if this loader supports the given path format.
        
        Args:
            path: Path to check
            
        Returns:
            True if this loader supports the given path, False otherwise
        """
        pass


class SafeTensorLoader(ModelLoader):
    """
    Loader for SafeTensor format models.
    """
    
    def __init__(self, path: str):
        """
        Initialize the SafeTensor loader.
        
        Args:
            path: Path to the model directory or file
        """
        self.tensor_file_map = {}  # Maps tensor names to file paths
        self.file_handle_map = {}  # Maps file names to file handles
        self._load_tensor_file_map(path)
    
    def _load_tensor_file_map(self, path: str) -> None:
        """
        Load the tensor file map from the given path.
        
        Args:
            path: Path to the model directory or file
        """
        # Normalize path to directory
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
        if os.path.isfile(path):
            folder_path = os.path.dirname(path)
        else:
            folder_path = path

        found_safetensor = False
        for root, _, files in os.walk(folder_path):
            files = sorted(files)
            for file in files:
                if file.endswith(".safetensors"):
                    found_safetensor = True
                    file_path = os.path.join(root, file)
                    if file not in self.file_handle_map:
                        try:
                            handle = safe_open(file_path, framework="pt")
                            self.file_handle_map[file] = handle
                        except Exception as e:
                            print(f"Error opening Safetensor file {file_path}: {e}")
                            continue

                    f = self.file_handle_map.get(file)
                    if f is None:
                        continue
                    try:
                        for key in f.keys():
                            self.tensor_file_map[key] = file
                    except Exception as e:
                        print(f"Error reading Safetensor file {file_path}: {e}")

        if not found_safetensor:
            # Not raising an error here allows for the factory to try other loaders
            print(f"No Safetensor files found in {folder_path}")
    
    def load_tensor(self, name: str, device: str = "cpu") -> torch.Tensor:
        """
        Load a tensor by name.
        
        Args:
            name: Name of the tensor to load
            device: Device to load the tensor to
            
        Returns:
            The loaded tensor
        """
        if name not in self.tensor_file_map:
            raise KeyError(f"Key {name} not found in Safetensor files")
        file = self.tensor_file_map[name]
        f = self.file_handle_map.get(file)
        if f is None:
            raise FileNotFoundError(f"File {file} not found in Safetensor files")
        tensor = f.get_tensor(name)
        return tensor.to(device)
    
    def load_dequantized_tensor(self, name: str, device: str = "cpu") -> torch.Tensor:
        """
        Load and dequantize a tensor.
        
        Args:
            name: Name of the tensor to load
            device: Device to load the tensor to
            
        Returns:
            The dequantized tensor
        """
        if name not in self.tensor_file_map:
            raise KeyError(f"Key {name} not found in Safetensor files")
        file = self.tensor_file_map[name]
        f = self.file_handle_map.get(file)
        if f is None:
            raise FileNotFoundError(f"File {file} not found in Safetensor files")
        tensor = f.get_tensor(name).to(device)
        if name.endswith(".weight"):
            if name[:-7] + ".weight_scale_inv" in self.tensor_file_map:
                weight_scale_inv = f.get_tensor(name[:-7] + ".weight_scale_inv").to(device)
                # Assuming weight_dequant function is imported
                from ktransformers.ktransformers_ext.triton.fp8gemm import weight_dequant
                tensor = weight_dequant(tensor, weight_scale_inv)
        return tensor.to(device)
    
    def close_all_handles(self) -> None:
        """
        Close all file handles.
        """
        for handle in self.file_handle_map.values():
            handle.close()
        self.file_handle_map.clear()

    @classmethod
    def supports_format(cls, path: str) -> bool:
        """
        Check if this loader supports the given path format.
        
        Args:
            path: Path to check
            
        Returns:
            True if safetensor files are found in the path, False otherwise
        """
        # Normalize path to directory
        if not os.path.exists(path):
            return False
        if os.path.isfile(path):
            if path.endswith(".safetensors"):
                return True
            folder_path = os.path.dirname(path)
        else:
            folder_path = path
            
        # Check if any safetensor files exist in the folder
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".safetensors"):
                    return True
        return False


class GGUFLoader(ModelLoader):
    """
    Loader for GGUF format models.
    """
    
    def __init__(self, path: str):
        """
        Initialize the GGUF loader.
        
        Args:
            path: Path to the model directory or file
        """
        # Check if path exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"GGUF dir not found: {path}")
        if os.path.isfile(path):
            self.gguf_path = os.path.dirname(path)
        else:
            self.gguf_path = path
            
        self.tensor_info = {}  # Stores tensor metadata
        self.tensor_file_map = {}  # Maps tensor names to file paths
        self.file_data_map = {}  # Maps file paths to memory-mapped data
        self.gguf_file_meta = {}  # Stores GGUF metadata
        
        # For compatibility with the factory pattern
        self.safetensor_loader = None
        
        # Scan all GGUF files in the directory
        found_gguf = False
        for root, _, files in os.walk(self.gguf_path):
            for file in files:
                if file.endswith(".gguf"):
                    found_gguf = True
                    file_path = os.path.join(root, file)
                    with open(file_path, "rb") as f:
                        self._load_gguf(f)
                        if file_path not in self.file_data_map:
                            self.file_data_map[file_path] = np.memmap(file_path, mode='r')
        
        if not found_gguf:
            raise FileNotFoundError(f"Cannot find any .gguf files in: {self.gguf_path}")
    
    def _load_gguf(self, f) -> None:
        """
        Load GGUF file metadata and tensor info.
        
        Args:
            f: File handle of the GGUF file
        """
        # Implementation should follow the original GGUFLoader._load_gguf
        # This is a simplified version for illustration
        f.seek(0)
        assert f.read(4) == b'GGUF'
        
        # Read header
        values = struct.unpack("<IQQ", f.read(4+8+8))
        version, n_tensors, n_kv = values
        if version != 3:
            warnings.warn(f"Version {version} has never been tested, might not work")

        # Read key-value pairs
        info = {}
        for _ in range(n_kv):
            name = self._read_value(f, 8)  # DATA_TYPES["string"]
            data_type = struct.unpack("<I", f.read(4))[0]
            info[name] = self._read_value(f, data_type)

        # Read tensor info
        tensor_info = {}
        for _ in range(n_tensors):
            name = self._read_value(f, 8)  # DATA_TYPES["string"]
            shape_len = self._read_value(f, 4)  # DATA_TYPES["uint32"]
            shape = [self._read_value(f, 10) for _ in range(shape_len)]  # DATA_TYPES["uint64"]
            ggml_type = self._read_value(f, 4)  # DATA_TYPES["uint32"]
            offset = self._read_value(f, 10)  # DATA_TYPES["uint64"]
            
            # Additional tensor metadata would be calculated here
            # For brevity, we're omitting the detailed tensor metadata calculation
            tensor_info[name] = {
                "ggml_type": ggml_type,
                "shape": shape,
                "offset": offset,
                # ... other tensor metadata
            }
            
        start = f.tell()
        alignment = info.get("general.alignment", 32)
        
        # Calculate actual file offsets
        for t in tensor_info.values():
            offset = start + t["offset"]
            offset += (alignment - offset % alignment) % alignment
            t["offset"] = offset
            
        # Update file maps
        for name in tensor_info:
            self.tensor_file_map[name] = f.name
            
        self.tensor_info.update(tensor_info)
        self.gguf_file_meta.update(info)
    
    def _read_value(self, f, data_type) -> Any:
        """
        Read a value from the file according to its data type.
        
        Args:
            f: File handle
            data_type: Type of data to read
            
        Returns:
            The read value
        """
        # Simplified implementation
        # In a complete implementation, this would handle all data types
        if data_type == 8:  # DATA_TYPES["string"]
            length = struct.unpack("<Q", f.read(8))[0]
            return f.read(length).decode("utf-8")
        elif data_type == 4:  # DATA_TYPES["uint32"]
            return struct.unpack("<I", f.read(4))[0]
        elif data_type == 10:  # DATA_TYPES["uint64"]
            return struct.unpack("<Q", f.read(8))[0]
        # ... handling for other data types
        return None
    
    def load_tensor(self, name: str, device: str = "cpu") -> torch.Tensor:
        """
        Load a tensor by name.
        
        Args:
            name: Name of the tensor to load
            device: Device to load the tensor to
            
        Returns:
            The loaded tensor
        """
        # This should call load_gguf_tensor with the appropriate parameters
        return self.load_gguf_tensor(name, device)
    
    def load_gguf_tensor(self, name: str, device: str = "cpu", target_dtype = None) -> torch.Tensor:
        """
        Load a GGUF tensor by name.
        
        Args:
            name: Name of the tensor to load
            device: Device to load the tensor to
            target_dtype: Target data type for the tensor
            
        Returns:
            The loaded tensor
        """
        # Implementation would follow the original GGUFLoader.load_gguf_tensor
        # This is a placeholder for illustration
        if name not in self.tensor_info:
            raise KeyError(f"Tensor {name} not found")
            
        # Actual implementation would dequantize the tensor data
        # and return a torch.Tensor
        return torch.zeros(1, device=device)  # Placeholder
    
    @classmethod
    def supports_format(cls, path: str) -> bool:
        """
        Check if this loader supports the given path format.
        
        Args:
            path: Path to check
            
        Returns:
            True if GGUF files are found in the path, False otherwise
        """
        # Normalize path to directory
        if not os.path.exists(path):
            return False
        if os.path.isfile(path):
            return path.endswith(".gguf")
        
        # Check if any GGUF files exist in the folder
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(".gguf"):
                    return True
        return False