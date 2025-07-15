#!/usr/bin/env python3
"""Script to fix critical linting issues automatically."""

import os
import re
import subprocess
from pathlib import Path


def fix_unused_imports():
    """Remove unused imports from Python files."""
    print("Fixing unused imports...")

    # Run autoflake to remove unused imports
    cmd = [
        "autoflake",
        "--in-place",
        "--remove-all-unused-imports",
        "--remove-unused-variables",
        "--recursive",
        "--exclude=.venv,venv,migrations,__pycache__,*.pyc",
        ".",
    ]

    try:
        subprocess.run(cmd, check=True)
        print("✓ Fixed unused imports")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error fixing unused imports: {e}")
    except FileNotFoundError:
        print("✗ autoflake not found. Installing...")
        subprocess.run(["pip", "install", "autoflake"], check=True)
        subprocess.run(cmd, check=True)


def fix_line_length_issues():
    """Fix line length issues in Python files."""
    print("Fixing line length issues...")

    # Black should handle most line length issues
    cmd = ["black", "--line-length", "88", "."]

    try:
        subprocess.run(cmd, check=True)
        print("✓ Fixed line length issues")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error fixing line length: {e}")


def fix_import_order():
    """Fix import order using isort."""
    print("Fixing import order...")

    cmd = ["isort", "--profile", "black", "."]

    try:
        subprocess.run(cmd, check=True)
        print("✓ Fixed import order")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error fixing import order: {e}")


def fix_docstring_issues():
    """Add missing docstrings to __init__ methods."""
    print("Fixing missing docstrings...")

    count = 0
    for path in Path(".").rglob("*.py"):
        if any(skip in str(path) for skip in [".venv", "venv", "__pycache__", "migrations"]):
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            # Simple regex to find __init__ methods without docstrings
            pattern = r'(\s*)def __init__\(self[^)]*\):\n(?!\s*""")'

            def add_docstring(match):
                indent = match.group(1)
                return f'{match.group(0)}{indent}    """Initialize the instance."""\n'

            new_content, n = re.subn(pattern, add_docstring, content)

            if n > 0:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                count += n

        except Exception as e:
            print(f"Error processing {path}: {e}")

    print(f"✓ Added {count} missing docstrings")


def fix_whitespace_issues():
    """Fix trailing whitespace and blank line issues."""
    print("Fixing whitespace issues...")

    count = 0
    for path in Path(".").rglob("*.py"):
        if any(skip in str(path) for skip in [".venv", "venv", "__pycache__", "migrations"]):
            continue

        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Fix trailing whitespace
            new_lines = [line.rstrip() + "\n" if line.strip() else "\n" for line in lines]

            # Fix multiple blank lines
            final_lines = []
            blank_count = 0
            for line in new_lines:
                if line.strip() == "":
                    blank_count += 1
                    if blank_count <= 2:
                        final_lines.append(line)
                else:
                    blank_count = 0
                    final_lines.append(line)

            # Ensure file ends with newline
            if final_lines and not final_lines[-1].endswith("\n"):
                final_lines[-1] += "\n"

            # Remove trailing blank lines
            while final_lines and final_lines[-1].strip() == "":
                final_lines.pop()
            if final_lines:
                final_lines.append("\n")

            if lines != final_lines:
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(final_lines)
                count += 1

        except Exception as e:
            print(f"Error processing {path}: {e}")

    print(f"✓ Fixed whitespace in {count} files")


def main():
    """Run all automatic fixes."""
    print("Running automatic lint fixes...\n")

    # Change to project root
    os.chdir("/home/green/FreeAgentics")

    # Run fixes in order
    fix_unused_imports()
    fix_line_length_issues()
    fix_import_order()
    fix_docstring_issues()
    fix_whitespace_issues()

    print("\n✓ Automatic fixes complete!")
    print("Note: Some issues may require manual intervention.")


if __name__ == "__main__":
    main()
