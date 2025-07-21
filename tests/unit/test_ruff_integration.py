"""Test suite for Ruff linting integration.

This module tests the Ruff linting configuration and integration
to ensure code quality standards are maintained across the project.

Following TDD principles, these tests are written before the implementation.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest


class TestRuffConfiguration:
    """Test cases for Ruff configuration validation."""

    def test_ruff_config_exists(self):
        """Test that Ruff configuration file exists."""
        config_path = Path("pyproject.toml")
        assert config_path.exists(), "pyproject.toml configuration file should exist"

    def test_ruff_installed(self):
        """Test that Ruff is installed and available."""
        result = subprocess.run(
            ["python", "-m", "pip", "show", "ruff"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "Ruff should be installed"

    def test_ruff_executable(self):
        """Test that Ruff can be executed."""
        result = subprocess.run(
            ["python", "-m", "ruff", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, "Ruff should be executable"
        assert "ruff" in result.stdout.lower(), "Ruff version should be displayed"


class TestRuffRules:
    """Test cases for Ruff rule configuration."""

    def test_ruff_rules_configured(self):
        """Test that Ruff rules are properly configured in pyproject.toml."""
        import toml

        config_path = Path("pyproject.toml")
        if not config_path.exists():
            assert False, "Test bypass removed - must fix underlying issue"

        config = toml.load(config_path)
        assert "tool" in config, "pyproject.toml should have [tool] section"
        assert "ruff" in config.get("tool", {}), (
            "pyproject.toml should have [tool.ruff] section"
        )

        ruff_config = config["tool"]["ruff"]
        assert "line-length" in ruff_config, "Line length should be configured"
        assert ruff_config["line-length"] == 100, (
            "Line length should be 100 to match Black"
        )

    def test_ruff_select_rules(self):
        """Test that appropriate rules are selected."""
        import toml

        config_path = Path("pyproject.toml")
        if not config_path.exists():
            assert False, "Test bypass removed - must fix underlying issue"

        config = toml.load(config_path)
        ruff_lint_config = config.get("tool", {}).get("ruff", {}).get("lint", {})

        selected_rules = ruff_lint_config.get("select", [])
        # Should include essential rule sets
        assert "E" in selected_rules, "Should include pycodestyle errors"
        assert "F" in selected_rules, "Should include Pyflakes"
        assert "I" in selected_rules, "Should include isort"
        assert "UP" in selected_rules, "Should include pyupgrade"

    def test_ruff_ignore_rules(self):
        """Test that appropriate rules are ignored for compatibility."""
        import toml

        config_path = Path("pyproject.toml")
        if not config_path.exists():
            assert False, "Test bypass removed - must fix underlying issue"

        config = toml.load(config_path)
        ruff_lint_config = config.get("tool", {}).get("ruff", {}).get("lint", {})

        ignored_rules = ruff_lint_config.get("ignore", [])
        # Should ignore rules that conflict with Black
        assert "E203" in ignored_rules, "Should ignore E203 (conflicts with Black)"


class TestRuffExecution:
    """Test cases for Ruff execution and output."""

    def test_ruff_check_clean_file(self):
        """Test that Ruff passes on a clean Python file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''"""Clean Python module."""


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b
'''
            )
            f.flush()

            # Use isolated config to test basic functionality
            result = subprocess.run(
                ["python", "-m", "ruff", "check", f.name, "--isolated"],
                capture_output=True,
                text=True,
            )

            os.unlink(f.name)
            assert result.returncode == 0, (
                f"Ruff should pass on clean file: {result.stderr}"
            )

    def test_ruff_check_problematic_file(self):
        """Test that Ruff fails on a problematic Python file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """import os
import sys
import os  # duplicate import

def bad_function( ):
    unused_var = 42
    return None
"""
            )
            f.flush()

            result = subprocess.run(
                ["python", "-m", "ruff", "check", f.name, "--isolated"],
                capture_output=True,
                text=True,
            )

            os.unlink(f.name)
            assert result.returncode != 0, "Ruff should fail on problematic file"
            assert "F401" in result.stdout or "F811" in result.stdout, (
                "Should detect import issues"
            )

    def test_ruff_format_check(self):
        """Test that Ruff format check works."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """def poorly_formatted(  x,y   ):
    return x+y
"""
            )
            f.flush()

            result = subprocess.run(
                ["python", "-m", "ruff", "format", "--check", f.name],
                capture_output=True,
                text=True,
            )

            os.unlink(f.name)
            assert result.returncode != 0, (
                "Ruff format should fail on poorly formatted file"
            )


class TestRuffIntegration:
    """Test cases for Ruff integration with project tools."""

    def test_makefile_ruff_command(self):
        """Test that Makefile includes Ruff command."""
        makefile_path = Path("Makefile")
        if not makefile_path.exists():
            assert False, "Test bypass removed - must fix underlying issue"

        content = makefile_path.read_text()
        assert "ruff" in content.lower(), "Makefile should include Ruff commands"

    def test_precommit_ruff_hook(self):
        """Test that pre-commit configuration includes Ruff."""
        import yaml

        config_path = Path(".pre-commit-config.yaml")
        if not config_path.exists():
            assert False, "Test bypass removed - must fix underlying issue"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        repos = config.get("repos", [])
        ruff_configured = any("ruff" in repo.get("repo", "").lower() for repo in repos)
        assert ruff_configured, "Pre-commit should include Ruff configuration"

    def test_github_actions_ruff(self):
        """Test that GitHub Actions CI includes Ruff checks."""
        ci_path = Path(".github/workflows/ci.yml")
        if not ci_path.exists():
            assert False, "Test bypass removed - must fix underlying issue"

        content = ci_path.read_text()
        assert "ruff" in content.lower(), "CI should include Ruff checks"


class TestRuffCompatibility:
    """Test cases for Ruff compatibility with existing tools."""

    def test_ruff_black_compatibility(self):
        """Test that Ruff is configured to be compatible with Black."""
        import toml

        config_path = Path("pyproject.toml")
        if not config_path.exists():
            assert False, "Test bypass removed - must fix underlying issue"

        config = toml.load(config_path)
        ruff_config = config.get("tool", {}).get("ruff", {})

        # Check line length matches Black
        assert ruff_config.get("line-length") == 100, (
            "Ruff line length should match Black"
        )

        # Check that conflicting rules are ignored
        ignored = ruff_config.get("lint", {}).get("ignore", [])
        assert "E203" in ignored, "Should ignore E203 for Black compatibility"

    def test_ruff_isort_compatibility(self):
        """Test that Ruff import sorting is compatible with isort configuration."""
        import toml

        config_path = Path("pyproject.toml")
        if not config_path.exists():
            assert False, "Test bypass removed - must fix underlying issue"

        config = toml.load(config_path)
        ruff_isort = (
            config.get("tool", {}).get("ruff", {}).get("lint", {}).get("isort", {})
        )

        # Should be configured to be compatible with Black (combine-as-imports)
        assert ruff_isort.get("combine-as-imports") is True, (
            "Should use combine-as-imports for Black compatibility"
        )


class TestRuffPerformance:
    """Test cases for Ruff performance requirements."""

    def test_ruff_speed_on_codebase(self):
        """Test that Ruff can lint the entire codebase quickly."""
        import time

        start_time = time.time()
        subprocess.run(
            ["python", "-m", "ruff", "check", ".", "--quiet"],
            capture_output=True,
            text=True,
        )
        end_time = time.time()

        duration = end_time - start_time
        # Ruff should be very fast, even on large codebases
        assert duration < 10.0, (
            f"Ruff should complete in under 10 seconds, took {duration:.2f}s"
        )


class TestRuffOutput:
    """Test cases for Ruff output formatting."""

    def test_ruff_json_output(self):
        """Test that Ruff can output results in JSON format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("import unused_module\n")
            f.flush()

            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "ruff",
                    "check",
                    "--output-format",
                    "json",
                    f.name,
                    "--isolated",
                ],
                capture_output=True,
                text=True,
            )

            os.unlink(f.name)

            # Should be valid JSON
            try:
                violations = json.loads(result.stdout)
                assert isinstance(violations, list), "JSON output should be a list"
            except json.JSONDecodeError:
                pytest.fail("Ruff JSON output should be valid JSON")

    def test_ruff_github_format(self):
        """Test that Ruff can output GitHub Actions annotations."""
        result = subprocess.run(
            [
                "python",
                "-m",
                "ruff",
                "check",
                "--output-format",
                "github",
                ".",
            ],
            capture_output=True,
            text=True,
        )

        # GitHub format should use :: syntax
        if result.stdout:
            assert "::" in result.stdout, "GitHub format should use :: annotations"
