#!/usr/bin/env python3
import os
import re


def check_file_for_any_error(filepath):
    """Check if a file uses 'Any' without importing it properly."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if file uses 'Any' in type annotations
    any_pattern = r"(?::\s*Any\b|-> Any\b|\[Any\]|Any\[|\bAny,|\s+Any\s*=)"
    uses_any = re.search(any_pattern, content)

    if not uses_any:
        return False

    # Check if 'Any' is imported
    import_patterns = [
        r"from typing import.*\bAny\b",
        r"from typing import.*\*",
        r"import typing\b",
        r"from typing_extensions import.*\bAny\b",
    ]

    has_import = any(re.search(pattern, content) for pattern in import_patterns)

    return not has_import


# Find all test files
test_files = []
for root, dirs, files in os.walk("."):
    # Skip hidden directories and common non-test directories
    dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["venv", "env", "__pycache__"]]

    for file in files:
        if "test" in file and file.endswith(".py"):
            test_files.append(os.path.join(root, file))

# Check each test file
files_with_errors = []
for filepath in test_files:
    try:
        if check_file_for_any_error(filepath):
            files_with_errors.append(filepath)
    except Exception:
        pass

# Print results
print(f"Found {len(files_with_errors)} test files with 'Any' not imported:")
for i, filepath in enumerate(files_with_errors[:10], 1):  # Show first 10
    print(f"{i}. {filepath}")

if len(files_with_errors) > 10:
    print(f"... and {len(files_with_errors) - 10} more files")
