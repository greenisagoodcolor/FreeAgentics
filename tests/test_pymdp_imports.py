"""
Comprehensive test suite for PyMDP import validation.
Tests ALL PyMDP imports across the codebase for correctness.
This follows TDD approach - these tests should FAIL initially (RED phase).
"""

import inspect
from pathlib import Path

import pytest


class TestPyMDPImports:
    """Test suite to validate all PyMDP imports are correct."""

    def test_pymdp_package_available(self):
        """Test that the correct pymdp package is available."""
        try:
            import pymdp

            # Verify this is inferactively-pymdp, not the old pymdp
            assert hasattr(pymdp, "agent"), "pymdp package should have 'agent' module"
            assert hasattr(pymdp, "utils"), "pymdp package should have 'utils' module"
        except ImportError as e:
            pytest.fail(f"PyMDP package not available: {e}")

    def test_agent_import_correct_path(self):
        """Test that Agent class can be imported from pymdp.agent."""
        try:
            from pymdp.agent import Agent

            assert inspect.isclass(Agent), "Agent should be a class"
            assert Agent.__module__ == "pymdp.agent", (
                f"Agent module should be 'pymdp.agent', got {Agent.__module__}"
            )
        except ImportError as e:
            pytest.fail(f"Cannot import Agent from pymdp.agent: {e}")

    def test_utils_import_correct_path(self):
        """Test that utils can be imported from pymdp."""
        try:
            from pymdp import utils

            assert hasattr(utils, "random_A_matrix"), "utils should have random_A_matrix function"
        except ImportError as e:
            pytest.fail(f"Cannot import utils from pymdp: {e}")

    def test_categorical_import_path(self):
        """Test Categorical import - Categorical doesn't exist in pymdp v0.0.7.1."""
        # Direct import should fail
        with pytest.raises(ImportError):
            pass

        # Alternative locations should also fail
        with pytest.raises(ImportError):
            pass

        with pytest.raises(ImportError):
            pass

        # This is correct behavior - Categorical is not available in this version

    def test_no_direct_pymdp_agent_import(self):
        """Test that 'from pymdp import Agent' should fail."""
        with pytest.raises(ImportError):
            pass

    def test_inference_active_pymdp_integration_imports(self):
        """Test imports in inference/active/pymdp_integration.py are correct."""
        # After fixes, this should import successfully
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from inference.active.pymdp_integration import PyMDPIntegration

        # Should import successfully after fixes
        assert PyMDPIntegration is not None

    def test_agents_base_agent_pymdp_imports(self):
        """Test imports in agents/base_agent.py are correct."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from agents.base_agent import ActiveInferenceAgent

        # Should import successfully after fixes
        assert ActiveInferenceAgent is not None

    def test_agents_resource_collector_pymdp_imports(self):
        """Test imports in agents/resource_collector.py are correct."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from agents.resource_collector import ResourceCollectorAgent

        # Should import successfully after fixes
        assert ResourceCollectorAgent is not None

    def test_agents_coalition_coordinator_pymdp_imports(self):
        """Test imports in agents/coalition_coordinator.py are correct."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from agents.coalition_coordinator import CoalitionCoordinatorAgent

        # Should import successfully after fixes
        assert CoalitionCoordinatorAgent is not None

    def test_agents_enhanced_active_inference_agent_imports(self):
        """Test imports in agents/enhanced_active_inference_agent.py are correct."""
        import sys

        sys.path.insert(0, str(Path(__file__).parent.parent))
        from agents.enhanced_active_inference_agent import (
            EnhancedActiveInferenceAgent,
        )

        # Should import successfully after fixes
        assert EnhancedActiveInferenceAgent is not None

    def test_no_graceful_fallbacks_allowed(self):
        """Test that there are no try/except blocks hiding PyMDP import failures."""
        # This test will scan for PyMDP import patterns with try/except
        # and fail if graceful fallbacks are found around PyMDP imports

        pymdp_files = [
            "inference/active/pymdp_integration.py",
            "agents/base_agent.py",
            "agents/resource_collector.py",
            "agents/coalition_coordinator.py",
            "agents/enhanced_active_inference_agent.py",
        ]

        project_root = Path(__file__).parent.parent

        for file_path in pymdp_files:
            full_path = project_root / file_path
            if full_path.exists():
                with open(full_path) as f:
                    content = f.read()

                # Check for PyMDP import wrapped in try/except blocks
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if "try:" in line:
                        pass
                    elif "except ImportError:" in line or "except ModuleNotFoundError:" in line:
                        # Look back in try block for PyMDP imports
                        for j in range(max(0, i - 10), i):
                            if "from pymdp" in lines[j] or "import pymdp" in lines[j]:
                                pytest.fail(
                                    f"Found PyMDP import in try/except block at line {j + 1} in {file_path}. "
                                    f"Hard failures required for PyMDP imports!"
                                )
                    elif "except" not in line and line.strip() and not line.strip().startswith("#"):
                        pass

    def test_all_pymdp_imports_use_correct_paths(self):
        """Test that all files use correct PyMDP import paths."""
        # After fixes, should not find any incorrect patterns

        incorrect_patterns = [
            "from pymdp import Agent",  # Should be: from pymdp.agent import Agent
            "from pymdp import Categorical",  # Categorical doesn't exist in pymdp v0.0.7.1
        ]

        project_root = Path(__file__).parent.parent
        found_issues = []

        # Scan all Python files for incorrect patterns
        for py_file in project_root.rglob("*.py"):
            if "test_pymdp_imports.py" in str(py_file):
                continue  # Skip this test file

            try:
                with open(py_file) as f:
                    content = f.read()

                for pattern in incorrect_patterns:
                    if pattern in content:
                        found_issues.append(
                            f"Found incorrect import pattern '{pattern}' in {py_file}"
                        )
            except (UnicodeDecodeError, PermissionError):
                continue  # Skip binary files or permission issues

        # After fixes, this should be empty
        if found_issues:
            pytest.fail("Found incorrect PyMDP import patterns:\n" + "\n".join(found_issues))

    def test_pymdp_version_compatibility(self):
        """Test that the installed PyMDP version is compatible."""
        import pymdp

        # Check that we're using inferactively-pymdp (has certain characteristics)
        expected_modules = ["agent", "utils", "maths"]

        for module_name in expected_modules:
            try:
                getattr(pymdp, module_name)
            except AttributeError:
                pytest.fail(
                    f"PyMDP missing expected module: {module_name}. "
                    f"Ensure inferactively-pymdp is installed, not old pymdp package."
                )
