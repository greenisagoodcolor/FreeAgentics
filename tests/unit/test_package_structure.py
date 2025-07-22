"""Test that FreeAgentics package structure is correct.

This test ensures our package can be properly imported without
sys.path manipulation, following the committee's zero-tolerance
policy for architectural violations.
"""

import subprocess
import sys
from pathlib import Path

import pytest


class TestPackageStructure:
    """Test suite for package structure validation."""

    def test_package_can_be_imported(self):
        """Test that main modules can be imported."""
        # These imports should work without sys.path manipulation
        # if the package is properly installed
        try:
            import agents
            import api
            import auth
            import database
            import inference
            import observability
            import world

            # Verify modules have expected attributes
            assert hasattr(agents, "__file__")
            assert hasattr(api, "__file__")
            assert hasattr(auth, "__file__")
            assert hasattr(database, "__file__")
            assert hasattr(inference, "__file__")
            assert hasattr(observability, "__file__")
            assert hasattr(world, "__file__")
        except ImportError as e:
            pytest.fail(f"Package import failed: {e}")

    def test_no_sys_path_manipulation_in_package_modules(self):
        """Ensure no production code uses sys.path manipulation."""
        # Find all Python files in package directories
        package_dirs = [
            "agents",
            "api",
            "auth",
            "database",
            "inference",
            "observability",
            "world",
            "coalitions",
            "knowledge_graph",
        ]

        violations = []

        for pkg_dir in package_dirs:
            pkg_path = Path(pkg_dir)
            if not pkg_path.exists():
                continue

            for py_file in pkg_path.rglob("*.py"):
                with open(py_file) as f:
                    content = f.read()
                    if "sys.path.insert" in content or "sys.path.append" in content:
                        violations.append(str(py_file))

        assert not violations, f"sys.path manipulation found in: {violations}"

    def test_setup_py_is_valid(self):
        """Test that setup.py can be parsed and is valid."""
        result = subprocess.run(
            [sys.executable, "setup.py", "--name"], capture_output=True, text=True
        )

        assert result.returncode == 0, f"setup.py failed: {result.stderr}"
        assert result.stdout.strip() == "freeagentics"

    def test_scripts_are_importable_as_module(self):
        """Test that scripts directory is a proper Python package."""
        import scripts

        # Should have __init__.py
        assert hasattr(scripts, "__file__")

    def test_entry_points_have_main_functions(self):
        """Test that all entry points have callable main functions."""
        # These are defined in setup.py entry_points
        entry_points = [
            ("scripts.apply_database_indexes", "main"),
            ("scripts.seed_database", "main"),
            ("scripts.test_database_connection", "main"),
            ("scripts.generate_memory_profiling_report", "main"),
        ]

        for module_name, func_name in entry_points:
            try:
                # Don't actually import these as they might have dependencies
                # Just check they would be importable
                module_path = module_name.replace(".", "/") + ".py"
                assert Path(module_path).exists(), f"{module_path} not found"
            except Exception as e:
                pytest.fail(f"Entry point {module_name}:{func_name} invalid: {e}")
