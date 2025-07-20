#!/usr/bin/env python3
"""
CRITICAL MISSION: PyMDP API Comprehensive Import Testing Suite

Following strict TDD RED-GREEN-REFACTOR methodology as specified in CLAUDE.MD.
This test suite creates FAILING tests for ALL PyMDP import statements in the codebase.

Task 1.1: Research PyMDP API changes and create failing import tests
Requirements:
- Tests must fail initially (RED phase)
- Cover ALL import statements in codebase
- Test direct imports, nested imports, circular dependencies
- No graceful fallbacks - imports must succeed or crash
- Document all discovered API changes

MUST RUN: pytest test_pymdp_imports_comprehensive.py -v --tb=short

Author: Agent 1 (Task 1.1)
Date: 2025-07-13
"""

import ast
import importlib
import inspect
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Add project root to path for testing
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class PyMDPImportScanner:
    """Scans codebase for ALL PyMDP import statements."""

    def __init__(self):
        self.found_imports = {}
        self.import_patterns = []

    def scan_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Extract all PyMDP imports from a Python file."""
        imports = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(file_path))

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if "pymdp" in alias.name:
                            imports.append(
                                {
                                    "type": "import",
                                    "module": alias.name,
                                    "name": alias.asname or alias.name,
                                    "line": node.lineno,
                                    "file": file_path,
                                }
                            )

                elif isinstance(node, ast.ImportFrom):
                    if node.module and "pymdp" in node.module:
                        for alias in node.names:
                            imports.append(
                                {
                                    "type": "from_import",
                                    "module": node.module,
                                    "name": alias.name,
                                    "asname": alias.asname,
                                    "line": node.lineno,
                                    "file": file_path,
                                }
                            )
        except (SyntaxError, UnicodeDecodeError):
            pass  # Skip files with syntax errors or encoding issues

        return imports

    def scan_codebase(self) -> Dict[str, List[Dict[str, Any]]]:
        """Scan entire codebase for PyMDP imports."""
        results = {}

        # Scan all Python files in project
        for py_file in PROJECT_ROOT.rglob("*.py"):
            # Skip test files and this file itself
            if (
                py_file.name.startswith("test_")
                or py_file.name == Path(__file__).name
                or ".git" in str(py_file)
                or "__pycache__" in str(py_file)
            ):
                continue

            imports = self.scan_file(py_file)
            if imports:
                results[str(py_file)] = imports

        return results


class TestPyMDPImportsComprehensive:
    """
    COMPREHENSIVE PyMDP Import Testing Suite.

    These tests are designed to FAIL initially (RED phase of TDD).
    Each test validates specific import patterns and API behaviors.
    """

    @classmethod
    def setup_class(cls):
        """Setup class-wide resources."""
        cls.scanner = PyMDPImportScanner()
        cls.all_imports = cls.scanner.scan_codebase()

        # Expected import patterns based on codebase analysis
        cls.expected_patterns = [
            "from pymdp.agent import Agent as PyMDPAgent",
            "from pymdp import utils",
            "import pymdp",
            "from pymdp.maths import softmax",
            "from pymdp.control import construct_policies",
            "from pymdp.learning import update_obs_model",
            "from pymdp.env import Environment",
        ]

    def test_all_imports_discovered(self):
        """TEST: Verify we found ALL PyMDP imports in codebase."""
        print(f"\n=== DISCOVERED PYMDP IMPORTS ===")

        total_imports = 0
        for file_path, imports in self.all_imports.items():
            print(f"\nFile: {file_path}")
            for imp in imports:
                total_imports += 1
                if imp["type"] == "import":
                    print(f"  Line {imp['line']}: import {imp['module']}")
                else:
                    asname_part = f" as {imp['asname']}" if imp["asname"] else ""
                    print(
                        f"  Line {imp['line']}: from {imp['module']} import {imp['name']}{asname_part}"
                    )

        # This SHOULD FAIL if we haven't scanned comprehensively enough
        assert total_imports >= 25, f"Expected at least 25 PyMDP imports, found {total_imports}"
        print(f"\n✓ Total PyMDP imports discovered: {total_imports}")

    def test_pymdp_agent_import_direct_failure(self):
        """TEST: Direct Agent import from pymdp root should FAIL."""
        # This test should PASS (import should fail)
        with pytest.raises(ImportError, match="cannot import name 'Agent'"):
            from pymdp import Agent  

    def test_pymdp_categorical_import_all_paths_fail(self):
        """TEST: Categorical import should FAIL from all locations."""
        # These should all fail - Categorical doesn't exist in current PyMDP

        with pytest.raises(ImportError):
            from pymdp import Categorical  

        with pytest.raises(ImportError):
            from pymdp.maths import Categorical  

        with pytest.raises(ImportError):
            from pymdp.agent import Categorical  

        with pytest.raises(ImportError):
            from pymdp.utils import Categorical  

    def test_pymdp_core_imports_must_succeed(self):
        """TEST: Core PyMDP imports MUST work or tests fail hard."""
        # These imports should succeed in current environment

        try:
            import pymdp

            assert hasattr(pymdp, "agent"), "pymdp.agent module missing"
            assert hasattr(pymdp, "utils"), "pymdp.utils module missing"
        except ImportError as e:
            pytest.fail(f"CRITICAL: PyMDP core import failed: {e}")

        try:
            from pymdp.agent import Agent

            assert inspect.isclass(Agent), "Agent is not a class"
            assert Agent.__module__ == "pymdp.agent", f"Wrong module: {Agent.__module__}"
        except ImportError as e:
            pytest.fail(f"CRITICAL: PyMDP Agent import failed: {e}")

        try:
            from pymdp import utils

            # Check for expected utils functions
            expected_utils = ["random_A_matrix", "random_B_matrix", "sample"]
            for util_func in expected_utils:
                if not hasattr(utils, util_func):
                    pytest.fail(f"PyMDP utils missing expected function: {util_func}")
        except ImportError as e:
            pytest.fail(f"CRITICAL: PyMDP utils import failed: {e}")

    def test_pymdp_advanced_modules_availability(self):
        """TEST: Advanced PyMDP modules may fail - testing API coverage."""

        # Test control module
        try:
            from pymdp import control

            assert hasattr(
                control, "update_posterior_policies"
            ), "control.update_posterior_policies missing"
        except (ImportError, AttributeError) as e:
            # This failure indicates API change
            pytest.fail(f"PyMDP control module API changed: {e}")

        # Test maths module
        try:
            from pymdp import maths

            assert hasattr(maths, "softmax"), "maths.softmax missing"
            assert hasattr(maths, "norm_dist"), "maths.norm_dist missing"
        except (ImportError, AttributeError) as e:
            pytest.fail(f"PyMDP maths module API changed: {e}")

        # Test learning module
        try:
            from pymdp import learning

            assert hasattr(learning, "update_obs_model"), "learning.update_obs_model missing"
        except (ImportError, AttributeError) as e:
            pytest.fail(f"PyMDP learning module API changed: {e}")

    def test_all_codebase_imports_execute_successfully(self):
        """TEST: ALL discovered imports in codebase must execute without error."""
        failed_imports = []

        for file_path, imports in self.all_imports.items():
            for imp in imports:
                try:
                    if imp["type"] == "import":
                        # Test: import pymdp
                        importlib.import_module(imp["module"])
                    else:
                        # Test: from pymdp.agent import Agent
                        module = importlib.import_module(imp["module"])
                        if not hasattr(module, imp["name"]):
                            failed_imports.append(
                                f"{file_path}:{imp['line']} - {imp['module']}.{imp['name']} not found"
                            )
                except ImportError as e:
                    failed_imports.append(
                        f"{file_path}:{imp['line']} - {imp['module']} import failed: {e}"
                    )

        if failed_imports:
            failure_msg = "CRITICAL: Found failing PyMDP imports in codebase:\n" + "\n".join(
                failed_imports
            )
            pytest.fail(failure_msg)

    def test_pymdp_agent_class_api_methods(self):
        """TEST: PyMDP Agent class must have expected API methods."""
        try:
            from pymdp.agent import Agent

            agent_instance = None  # We'll create a minimal instance if needed

            # Expected Agent methods based on documentation
            expected_methods = [
                "infer_states",
                "infer_policies",
                "sample_action",
                "update_A",
                "update_B",
                "update_D",
                "reset",
                "step_time",
            ]

            for method_name in expected_methods:
                if not hasattr(Agent, method_name):
                    pytest.fail(f"PyMDP Agent missing required method: {method_name}")

                # Verify it's actually callable
                method = getattr(Agent, method_name)
                if not callable(method):
                    pytest.fail(f"PyMDP Agent.{method_name} is not callable")

        except ImportError as e:
            pytest.fail(f"CRITICAL: Cannot test Agent API - import failed: {e}")

    def test_pymdp_utils_expected_functions(self):
        """TEST: PyMDP utils must have expected utility functions."""
        try:
            from pymdp import utils

            # Expected utils functions based on codebase usage
            expected_functions = [
                "random_A_matrix",
                "random_B_matrix",
                "sample",
                "norm_dist",
                "onehot",
            ]

            missing_functions = []
            for func_name in expected_functions:
                if not hasattr(utils, func_name):
                    missing_functions.append(func_name)

            if missing_functions:
                pytest.fail(f"PyMDP utils missing functions: {missing_functions}")

        except ImportError as e:
            pytest.fail(f"CRITICAL: Cannot test utils API - import failed: {e}")

    def test_pymdp_import_circular_dependencies(self):
        """TEST: Check for circular import issues in PyMDP."""

        # Test importing in different orders to catch circular imports
        import_sequences = [
            ["pymdp", "pymdp.agent", "pymdp.utils"],
            ["pymdp.agent", "pymdp", "pymdp.utils"],
            ["pymdp.utils", "pymdp.agent", "pymdp"],
        ]

        for sequence in import_sequences:
            # Clear module cache for clean test
            for module_name in sequence:
                if module_name in sys.modules:
                    del sys.modules[module_name]

            try:
                for module_name in sequence:
                    importlib.import_module(module_name)
            except ImportError as e:
                pytest.fail(f"Circular import detected in sequence {sequence}: {e}")

    def test_no_graceful_import_fallbacks_in_codebase(self):
        """TEST: Verify NO graceful fallbacks exist around PyMDP imports."""

        # This test scans for try/except blocks around PyMDP imports
        # These are forbidden - imports must succeed or crash hard

        graceful_fallbacks_found = []

        for file_path, imports in self.all_imports.items():
            try:
                with open(file_path, "r") as f:
                    content = f.read()

                # Look for try/except patterns around PyMDP imports
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    line_stripped = line.strip()

                    # Check if this line has a PyMDP import
                    if "from pymdp" in line_stripped or "import pymdp" in line_stripped:
                        # Look backwards for try: statements
                        for j in range(max(0, i - 5), i):
                            if "try:" in lines[j]:
                                # Look forwards for except: statements
                                for k in range(i + 1, min(len(lines), i + 10)):
                                    if "except" in lines[k] and (
                                        "ImportError" in lines[k]
                                        or "ModuleNotFoundError" in lines[k]
                                    ):
                                        graceful_fallbacks_found.append(
                                            f"{file_path}:{i+1} - PyMDP import in try/except block"
                                        )
                                        break
                                break

            except (UnicodeDecodeError, FileNotFoundError):
                continue

        if graceful_fallbacks_found:
            failure_msg = (
                "CRITICAL: Found graceful fallbacks around PyMDP imports (FORBIDDEN):\n"
                + "\n".join(graceful_fallbacks_found)
            )
            pytest.fail(failure_msg)

    def test_pymdp_package_installation_validation(self):
        """TEST: Validate correct PyMDP package is installed."""
        try:
            import pymdp

            # Check package metadata if available
            if hasattr(pymdp, "__version__"):
                version = pymdp.__version__
                print(f"PyMDP version: {version}")

                # Validate version format
                if not version or not isinstance(version, str):
                    pytest.fail(f"Invalid PyMDP version format: {version}")

            # Check for inferactively-pymdp specific attributes
            if hasattr(pymdp, "__file__"):
                file_path = pymdp.__file__
                if "inferactively-pymdp" not in file_path and "pymdp" not in file_path:
                    pytest.fail(f"Unexpected PyMDP installation path: {file_path}")

            # Verify key modules exist
            required_modules = ["agent", "utils", "maths", "control", "learning"]
            missing_modules = []

            for module_name in required_modules:
                if not hasattr(pymdp, module_name):
                    missing_modules.append(module_name)

            if missing_modules:
                pytest.fail(f"PyMDP installation missing modules: {missing_modules}")

        except ImportError as e:
            pytest.fail(f"CRITICAL: PyMDP package not properly installed: {e}")

    def test_generate_comprehensive_import_report(self):
        """TEST: Generate comprehensive report of all PyMDP usage patterns."""

        # This test always fails to generate the report
        report_lines = [
            "=== COMPREHENSIVE PYMDP IMPORT ANALYSIS REPORT ===",
            f"Scan Date: 2025-07-13",
            f"Total Files Scanned: {len(self.all_imports)}",
            "",
        ]

        # File by file analysis
        total_imports = 0
        for file_path, imports in self.all_imports.items():
            if imports:
                report_lines.append(f"FILE: {file_path}")
                for imp in imports:
                    total_imports += 1
                    if imp["type"] == "import":
                        report_lines.append(f"  Line {imp['line']}: import {imp['module']}")
                    else:
                        asname = f" as {imp['asname']}" if imp["asname"] else ""
                        report_lines.append(
                            f"  Line {imp['line']}: from {imp['module']} import {imp['name']}{asname}"
                        )
                report_lines.append("")

        # Summary statistics
        report_lines.extend(
            [
                "=== SUMMARY STATISTICS ===",
                f"Total PyMDP imports found: {total_imports}",
                f"Files with PyMDP imports: {len(self.all_imports)}",
                "",
                "=== IMPORT PATTERNS ANALYSIS ===",
            ]
        )

        # Pattern analysis
        patterns = {}
        for file_path, imports in self.all_imports.items():
            for imp in imports:
                if imp["type"] == "import":
                    pattern = f"import {imp['module']}"
                else:
                    pattern = f"from {imp['module']} import {imp['name']}"
                patterns[pattern] = patterns.get(pattern, 0) + 1

        for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"  {count:2d}x {pattern}")

        report_content = "\n".join(report_lines)

        # Write report to file
        report_file = PROJECT_ROOT / "PYMDP_IMPORT_ANALYSIS_REPORT.md"
        with open(report_file, "w") as f:
            f.write(report_content)

        # Always fail to show report in test output
        pytest.fail(f"Generated PyMDP import analysis report:\n{report_content}")


if __name__ == "__main__":
    # Run the comprehensive test suite
    print("=== PYMDP COMPREHENSIVE IMPORT TESTING SUITE ===")
    print("Following TDD RED-GREEN-REFACTOR methodology")
    print("These tests are designed to FAIL initially (RED phase)")
    print()

    # Run with pytest
    exit_code = pytest.main(
        [__file__, "-v", "--tb=short", "--no-header", "-x"]  # Stop on first failure
    )

    sys.exit(exit_code)
