"""
Pytest configuration for FreeAgentics test suite.

This module provides fixtures and configuration for testing both PyTorch
and PyMDP components with proper graceful degradation when dependencies
are not available.
"""

import os
import sys
import warnings
from pathlib import Path

import pytest

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Dependency availability checks
TORCH_AVAILABLE = False
PYMDP_AVAILABLE = False
TORCH_GEOMETRIC_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    # Test basic PyTorch functionality
    _ = torch.tensor([1.0, 2.0])
except (ImportError, RuntimeError) as e:
    warnings.warn(f"PyTorch not available for testing: {e}", stacklevel=2)
    torch = None
    nn = None

try:
    import torch_geometric
    TORCH_GEOMETRIC_AVAILABLE = TORCH_AVAILABLE  # Requires PyTorch
except (ImportError, RuntimeError) as e:
    if TORCH_AVAILABLE:
        warnings.warn(
            f"PyTorch Geometric not available for testing: {e}", stacklevel=2
        )
    torch_geometric = None

try:
    from pymdp import Agent as PyMDPAgent
    from pymdp import utils as pymdp_utils
    from pymdp.maths import entropy, kl_divergence
    PYMDP_AVAILABLE = True
    # Test basic PyMDP functionality
    _ = pymdp_utils.random_A_matrix([2, 2])
except (ImportError, RuntimeError) as e:
    warnings.warn(f"PyMDP not available for testing: {e}", stacklevel=2)
    PyMDPAgent = None
    pymdp_utils = None
    entropy = None
    kl_divergence = None

# Environment setup
os.environ.setdefault("TESTING", "1")
os.environ.setdefault("LOG_LEVEL", "WARNING")


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "pytorch: mark test as requiring PyTorch"
    )
    config.addinivalue_line(
        "markers", "pymdp: mark test as requiring PyMDP"
    )
    config.addinivalue_line(
        "markers", "gnn: mark test as requiring Graph Neural Networks"
    )
    config.addinivalue_line(
        "markers", "core: mark test as core functionality (no ML dependencies)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle dependency requirements."""
    for item in items:
        # Auto-mark tests based on their module/function names
        if "torch" in str(item.fspath).lower() or "torch" in item.name.lower():
            item.add_marker(pytest.mark.pytorch)
        
        if "pymdp" in str(item.fspath).lower() or "pymdp" in item.name.lower():
            item.add_marker(pytest.mark.pymdp)
            
        if "gnn" in str(item.fspath).lower() or "gnn" in item.name.lower():
            item.add_marker(pytest.mark.gnn)
            
        # Skip tests based on availability
        if item.get_closest_marker("pytorch") and not TORCH_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="PyTorch not available"))
            
        if item.get_closest_marker("pymdp") and not PYMDP_AVAILABLE:
            item.add_marker(pytest.mark.skip(reason="PyMDP not available"))
            
        if (item.get_closest_marker("gnn") and 
            not TORCH_GEOMETRIC_AVAILABLE):
            item.add_marker(
                pytest.mark.skip(reason="PyTorch Geometric not available")
            )


@pytest.fixture(scope="session")
def torch_available():
    """Fixture to check if PyTorch is available."""
    return TORCH_AVAILABLE


@pytest.fixture(scope="session")
def pymdp_available():
    """Fixture to check if PyMDP is available."""
    return PYMDP_AVAILABLE


@pytest.fixture(scope="session")
def gnn_available():
    """Fixture to check if Graph Neural Networks are available."""
    return TORCH_GEOMETRIC_AVAILABLE


@pytest.fixture
def torch_device():
    """Fixture providing appropriate torch device."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
    return torch.device("cpu")  # Use CPU for consistent testing


@pytest.fixture
def sample_tensor():
    """Fixture providing a sample PyTorch tensor."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
    return torch.randn(10, 5)


@pytest.fixture
def sample_pymdp_matrices():
    """Fixture providing sample PyMDP matrices."""
    if not PYMDP_AVAILABLE:
        pytest.skip("PyMDP not available")
    
    return {
        "A": pymdp_utils.random_A_matrix([3, 2]),  # 3 observations, 2 states
        "B": pymdp_utils.random_B_matrix([2, 2]),  # 2 states, 2 actions
        "C": pymdp_utils.obj_array_uniform([3]),   # 3 observations
        "D": pymdp_utils.obj_array_uniform([2]),   # 2 states
    }


@pytest.fixture
def mock_torch_when_unavailable():
    """Fixture that provides mock torch functionality when PyTorch is unavailable."""
    if TORCH_AVAILABLE:
        return torch
    
    # Create minimal mock for testing import paths
    class MockTorch:
        def __init__(self):
            self.nn = MockNN()
            
        def tensor(self, data):
            import numpy as np
            return np.array(data)
            
        def randn(self, *shape):
            import numpy as np
            return np.random.randn(*shape)
    
    class MockNN:
        def Module(self):
            pass
            
    return MockTorch()


@pytest.fixture
def test_data_dir():
    """Fixture providing path to test data directory."""
    return project_root / "tests" / "data"


@pytest.fixture
def temp_test_dir(tmp_path):
    """Fixture providing temporary directory for test files."""
    return tmp_path


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Automatically suppress known warnings during testing."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", message=".*torch.*", category=UserWarning)


@pytest.fixture
def coverage_test_env():
    """Fixture that sets up environment variables for coverage testing."""
    original_env = os.environ.copy()
    
    # Set environment variables for coverage testing
    os.environ["COVERAGE_TESTING"] = "1"
    os.environ["SKIP_PYTORCH_INIT"] = "1"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Helper functions for test utilities
def requires_torch(test_func):
    """Decorator to skip tests that require PyTorch."""
    return pytest.mark.skipif(
        not TORCH_AVAILABLE, 
        reason="PyTorch not available"
    )(test_func)


def requires_pymdp(test_func):
    """Decorator to skip tests that require PyMDP."""
    return pytest.mark.skipif(
        not PYMDP_AVAILABLE, 
        reason="PyMDP not available"
    )(test_func)


def requires_gnn(test_func):
    """Decorator to skip tests that require Graph Neural Networks."""
    return pytest.mark.skipif(
        not TORCH_GEOMETRIC_AVAILABLE, 
        reason="PyTorch Geometric not available"
    )(test_func)


# Export availability flags for use in tests
__all__ = [
    "TORCH_AVAILABLE",
    "PYMDP_AVAILABLE", 
    "TORCH_GEOMETRIC_AVAILABLE",
    "requires_torch",
    "requires_pymdp",
    "requires_gnn",
]
