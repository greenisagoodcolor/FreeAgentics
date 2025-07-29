"""Smoke test for development environment setup."""

import subprocess
import os
import pytest
from pathlib import Path


class TestDevSmoke:
    """Test that development environment works from clean state."""

    def test_makefile_no_virtual_env_errors(self):
        """Test that Makefile doesn't have unbound VIRTUAL_ENV errors."""
        # Ensure we're in project root
        project_root = Path(__file__).parent.parent

        # Clear VIRTUAL_ENV to simulate fresh shell
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)

        # Test make help (safe, non-interactive)
        result = subprocess.run(
            ["make", "help"],
            env=env,
            capture_output=True,
            text=True,
            cwd=str(project_root),
            timeout=5,
        )

        # Should succeed without bash errors
        assert result.returncode == 0, f"make help failed: {result.stderr}"
        assert "unbound variable" not in result.stderr
        assert "FreeAgentics Multi-Agent AI Platform" in result.stdout

    def test_make_dev_starts_without_venv(self):
        """Test that 'make dev' can start without manual venv activation."""
        # This would be too heavy for unit tests, but we can at least check
        # that the command doesn't immediately fail with bash errors

        project_root = Path(__file__).parent.parent
        os.chdir(project_root)

        # Clear VIRTUAL_ENV
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)

        # Just check if make dev --dry-run works (won't actually start servers)
        result = subprocess.run(
            ["make", "-n", "dev"], env=env, capture_output=True, text=True  # -n is dry run
        )

        # Should not have bash errors about unbound variables
        assert "unbound variable" not in result.stderr
        assert result.returncode == 0
