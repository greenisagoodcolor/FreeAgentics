#!/usr/bin/env python3
"""
Standalone test runner for security encryption and SOAR features.
Runs without requiring the full application stack.
"""

import os
import sys
from pathlib import Path

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


# Mock dependencies that aren't available
class MockApp:
    def __init__(self):
        pass


sys.modules["api.main"] = MockApp()

# Run specific test file
if __name__ == "__main__":
    # Run our encryption and SOAR tests
    test_file = "tests/security/test_encryption_soar.py"

    # Install required packages first
    os.system("pip install -q moto hvac pyyaml numpy")

    # Run tests
    exit_code = pytest.main(
        [
            test_file,
            "-v",
            "--tb=short",
            "-x",
            "--disable-warnings",
        ]  # Stop on first failure
    )

    sys.exit(exit_code)
