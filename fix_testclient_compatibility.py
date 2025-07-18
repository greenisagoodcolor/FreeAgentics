#!/usr/bin/env python3
"""
Fix TestClient compatibility issues across all test files.

This script replaces all imports of TestClient from fastapi.testclient
with imports from the compatibility wrapper.
"""

import glob
import os
import re
import subprocess
from pathlib import Path


def find_test_files():
    """Find all test files that import TestClient from fastapi.testclient."""
    test_files = []

    # Find all Python files in tests directory
    for root, dirs, files in os.walk("tests"):
        for file in files:
            if file.endswith(".py"):
                test_files.append(os.path.join(root, file))

    return test_files


def fix_testclient_imports(filepath):
    """Fix TestClient imports in a single file."""
    with open(filepath, "r") as f:
        content = f.read()

    # Check if file imports TestClient from fastapi.testclient
    if "from fastapi.testclient import TestClient" in content:
        print(f"Fixing TestClient import in {filepath}")

        # Replace the import
        content = content.replace(
            "from fastapi.testclient import TestClient",
            "from tests.utils.test_client_compat import TestClient",
        )

        # Write the updated content back
        with open(filepath, "w") as f:
            f.write(content)

        return True

    return False


def main():
    """Main function to fix all TestClient imports."""
    test_files = find_test_files()
    fixed_files = []

    for test_file in test_files:
        if fix_testclient_imports(test_file):
            fixed_files.append(test_file)

    print(f"\nFixed {len(fixed_files)} files:")
    for file in fixed_files:
        print(f"  - {file}")

    if fixed_files:
        print(
            "\nAll TestClient imports have been updated to use the compatibility wrapper."
        )
        print(
            "You can now run tests without the httpx 0.28.1+ compatibility issues."
        )
    else:
        print("No files needed to be fixed.")


if __name__ == "__main__":
    main()
