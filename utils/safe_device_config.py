"""Safe device configuration that avoids PyTorch CUDA issues during testing.

This module provides a minimal interface for device configuration that doesn't
trigger PyTorch CUDA initialization crashes in test environments.
"""

import logging
import os
from typing import Any, Union

import numpy as np

logger = logging.getLogger(__name__)

# Simple CPU-only configuration for tests
_IS_TEST_MODE = os.environ.get("TESTING") == "true" or os.environ.get("CUDA_VISIBLE_DEVICES") == ""


def is_test_mode() -> bool:
    """Check if running in test mode (CPU-only)."""
    return _IS_TEST_MODE


def safe_tensor_to_python(value: Any, target_type: type = float) -> Union[int, float, str, bool]:
    """Safely convert tensor/array to Python primitive without PyTorch dependencies.
    
    Args:
        value: Input tensor, array, or scalar
        target_type: Target Python type
        
    Returns:
        Python primitive value
        
    Raises:
        ValueError: If conversion fails
    """
    try:
        # Handle numpy arrays and scalars
        if hasattr(value, 'item'):
            # Numpy scalar or 0-d array
            return target_type(value.item())
        elif hasattr(value, '__getitem__') and hasattr(value, '__len__'):
            # Array-like object
            np_array = np.asarray(value)
            if np_array.size == 0:
                raise ValueError(f"Empty array cannot be converted to {target_type.__name__}")
            elif np_array.size == 1:
                return target_type(np_array.flat[0])
            else:
                # Multi-element array - take first element with warning
                logger.warning(f"Converting multi-element array {np_array.shape} to scalar, using first element")
                return target_type(np_array.flat[0])
        else:
            # Direct conversion for primitives
            return target_type(value)
            
    except Exception as e:
        raise ValueError(f"Failed to convert {type(value)} to {target_type.__name__}: {e}")


def ensure_numpy_array(value: Any) -> np.ndarray:
    """Safely convert value to numpy array without PyTorch dependencies.
    
    Args:
        value: Input value
        
    Returns:
        NumPy array
    """
    return np.asarray(value)


def get_safe_device_info() -> dict:
    """Get safe device information without triggering CUDA initialization.
    
    Returns:
        Dictionary with basic device information
    """
    return {
        "test_mode": _IS_TEST_MODE,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        "numpy_version": np.__version__,
        "device": "cpu" if _IS_TEST_MODE else "auto",
    }