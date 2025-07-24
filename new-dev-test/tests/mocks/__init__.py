"""Test mocks for CI environments."""

import sys
from unittest.mock import MagicMock

# Mock heavy ML dependencies for CI
if "pymdp" not in sys.modules:
    from . import pymdp_mock

    sys.modules["pymdp"] = pymdp_mock
    sys.modules["pymdp.Agent"] = pymdp_mock.Agent
    sys.modules["pymdp.agent"] = pymdp_mock  # Add agent submodule
    sys.modules["pymdp.utils"] = pymdp_mock.utils

# Mock torch_geometric if not available
if "torch_geometric" not in sys.modules:
    torch_geometric_mock = MagicMock()
    sys.modules["torch_geometric"] = torch_geometric_mock
    sys.modules["torch_geometric.nn"] = MagicMock()
    sys.modules["torch_geometric.data"] = MagicMock()

# Mock inferactively_pymdp if not available
if "inferactively_pymdp" not in sys.modules:
    inferactively_mock = MagicMock()
    sys.modules["inferactively_pymdp"] = inferactively_mock
