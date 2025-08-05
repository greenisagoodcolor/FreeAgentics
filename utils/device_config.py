"""Device configuration and tensor management for CPU/GPU agnostic operations.

This module provides a unified interface for numerical operations that work
consistently across CPU and GPU environments, with automatic fallback to CPU
when CUDA is unavailable.
"""

import logging
import os
from typing import Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Global device configuration
_FORCE_CPU = os.environ.get("CUDA_VISIBLE_DEVICES") == ""
_DEVICE_INITIALIZED = False
_TORCH_AVAILABLE = False
_CUDA_AVAILABLE = False

try:
    import torch

    _TORCH_AVAILABLE = True
    # Only check CUDA availability if not forced to CPU mode
    if not _FORCE_CPU:
        try:
            _CUDA_AVAILABLE = torch.cuda.is_available()
        except Exception as e:
            logger.warning(f"CUDA check failed: {e}, defaulting to CPU-only mode")
            _CUDA_AVAILABLE = False
    else:
        _CUDA_AVAILABLE = False
    logger.info(
        f"PyTorch device configuration: CUDA available={_CUDA_AVAILABLE}, Force CPU={_FORCE_CPU}"
    )
except ImportError:
    logger.info("PyTorch not available, using CPU-only mode")


class DeviceConfig:
    """Configuration for device-agnostic tensor operations."""

    def __init__(self, force_cpu: bool = False):
        """Initialize device configuration.

        Args:
            force_cpu: If True, forces CPU-only mode regardless of CUDA availability
        """
        self.force_cpu = force_cpu or _FORCE_CPU
        self.torch_available = _TORCH_AVAILABLE
        self.cuda_available = _CUDA_AVAILABLE and not self.force_cpu

        # Set default device
        if self.torch_available:
            self.device = torch.device(
                "cpu" if self.force_cpu else ("cuda" if self.cuda_available else "cpu")
            )
        else:
            self.device = "cpu"

        logger.debug(
            f"DeviceConfig initialized: device={self.device}, torch_available={self.torch_available}"
        )

    @property
    def is_cpu(self) -> bool:
        """Check if using CPU device."""
        return str(self.device) == "cpu"

    @property
    def is_cuda(self) -> bool:
        """Check if using CUDA device."""
        return self.torch_available and "cuda" in str(self.device)


# Global device configuration instance
device_config = DeviceConfig()


class TensorAdapter:
    """Adapter for device-agnostic tensor operations."""

    def __init__(self, config: Optional[DeviceConfig] = None):
        """Initialize tensor adapter with device configuration.

        Args:
            config: Device configuration (uses global config if None)
        """
        self.config = config or device_config

    def to_tensor(
        self, array: Union[np.ndarray, list, float, int], dtype: Optional[Any] = None
    ) -> Any:
        """Convert array-like to appropriate tensor type.

        Args:
            array: Input array or scalar
            dtype: Target data type (numpy or torch dtype)

        Returns:
            Tensor on appropriate device
        """
        # Convert to numpy first for consistency
        np_array = np.asarray(array)

        if self.config.torch_available and not self.config.force_cpu:
            # Convert to PyTorch tensor
            if dtype is None:
                tensor = torch.from_numpy(np_array)
            else:
                tensor = torch.from_numpy(np_array).to(dtype=dtype)

            return tensor.to(self.config.device)
        else:
            # Return numpy array with specified dtype
            if dtype is not None:
                # Map torch dtypes to numpy if necessary
                if hasattr(dtype, "numpy"):
                    dtype = dtype.numpy()
                return np_array.astype(dtype)
            return np_array

    def to_numpy(self, tensor: Any) -> np.ndarray:
        """Convert tensor to numpy array.

        Args:
            tensor: Input tensor (torch or numpy)

        Returns:
            NumPy array
        """
        if self.config.torch_available and hasattr(tensor, "cpu"):
            # PyTorch tensor
            return tensor.cpu().detach().numpy()
        elif hasattr(tensor, "numpy"):
            # Convert to numpy if method exists
            return tensor.numpy()
        else:
            # Already numpy or array-like
            return np.asarray(tensor)

    def ensure_device(self, tensor: Any) -> Any:
        """Ensure tensor is on correct device.

        Args:
            tensor: Input tensor

        Returns:
            Tensor on correct device
        """
        if self.config.torch_available and hasattr(tensor, "to"):
            return tensor.to(self.config.device)
        return tensor

    def create_zeros(self, shape: tuple, dtype: Optional[Any] = None) -> Any:
        """Create zeros tensor on appropriate device.

        Args:
            shape: Tensor shape
            dtype: Data type

        Returns:
            Zeros tensor
        """
        if self.config.torch_available:
            return torch.zeros(shape, dtype=dtype, device=self.config.device)
        else:
            return np.zeros(shape, dtype=dtype)

    def create_ones(self, shape: tuple, dtype: Optional[Any] = None) -> Any:
        """Create ones tensor on appropriate device.

        Args:
            shape: Tensor shape
            dtype: Data type

        Returns:
            Ones tensor
        """
        if self.config.torch_available:
            return torch.ones(shape, dtype=dtype, device=self.config.device)
        else:
            return np.ones(shape, dtype=dtype)


# Global tensor adapter instance
tensor_adapter = TensorAdapter()


def safe_tensor_to_python(tensor: Any, target_type: type = float) -> Union[int, float, str, bool]:
    """Safely convert tensor/array to Python primitive.

    Args:
        tensor: Input tensor, array, or scalar
        target_type: Target Python type

    Returns:
        Python primitive value

    Raises:
        ValueError: If conversion fails
    """
    try:
        # Convert to numpy first
        np_array = tensor_adapter.to_numpy(tensor)

        # Handle scalar case
        if np_array.ndim == 0:
            return target_type(np_array.item())

        # Handle array case
        if np_array.size == 1:
            return target_type(np_array.flat[0])

        # Multi-element array - take first element with warning
        logger.warning(
            f"Converting multi-element array {np_array.shape} to scalar, using first element"
        )
        return target_type(np_array.flat[0])

    except Exception as e:
        raise ValueError(f"Failed to convert {type(tensor)} to {target_type.__name__}: {e}")


def get_device_info() -> dict:
    """Get comprehensive device information for debugging.

    Returns:
        Dictionary with device information
    """
    info = {
        "force_cpu": device_config.force_cpu,
        "torch_available": device_config.torch_available,
        "cuda_available": device_config.cuda_available,
        "device": str(device_config.device),
        "numpy_version": np.__version__,
    }

    if device_config.torch_available:
        import torch

        info.update(
            {
                "torch_version": torch.__version__,
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "current_device": torch.cuda.current_device()
                if torch.cuda.is_available() and torch.cuda.device_count() > 0
                else None,
            }
        )

    return info


def set_cpu_only_mode(enabled: bool = True) -> None:
    """Force CPU-only mode for testing.

    Args:
        enabled: If True, forces CPU-only mode
    """
    global device_config, tensor_adapter
    device_config = DeviceConfig(force_cpu=enabled)
    tensor_adapter = TensorAdapter(device_config)
    logger.info(
        f"Device configuration updated: CPU-only mode {'enabled' if enabled else 'disabled'}"
    )
