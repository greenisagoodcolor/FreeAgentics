#!/usr/bin/env python3
"""Script to fix whitespace issues in Python files."""

import os
import re
import sys


def fix_whitespace_issues(file_path):
    """Fix whitespace issues in a Python file."""
    try:
        with open(file_path, "r") as f:
            content = f.read()

        original_content = content

        # Fix trailing whitespace (W291)
        content = re.sub(r"[ \t]+$", "", content, flags=re.MULTILINE)

        # Fix blank lines with whitespace (W293)
        content = re.sub(r"^[ \t]+$", "", content, flags=re.MULTILINE)

        # Ensure file ends with newline (W292)
        if content and not content.endswith("\n"):
            content += "\n"

        # Only write if content changed
        if content != original_content:
            with open(file_path, "w") as f:
                f.write(content)
            print(f"Fixed whitespace in {file_path}")
            return True

        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Fix whitespace issues in all Python files."""
    base_dir = "/home/green/FreeAgentics"

    # Directories to exclude
    exclude_dirs = {"venv", "__pycache__", ".git", "node_modules", ".pytest_cache"}

    fixed_count = 0

    for root, dirs, files in os.walk(base_dir):
        # Remove excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                if fix_whitespace_issues(file_path):
                    fixed_count += 1

    print(f"Fixed whitespace issues in {fixed_count} files")


if __name__ == "__main__":
    main()
