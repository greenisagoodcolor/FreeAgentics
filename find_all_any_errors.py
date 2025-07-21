#!/usr/bin/env python3
import os
import re


def check_file_for_any_error(filepath):
    """Check if a file uses 'Any' without importing it properly."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except:
        return False

    # Check if file uses 'Any' in type annotations
    any_patterns = [
        r":\s*Any\b",  # : Any
        r"->\s*Any\b",  # -> Any
        r"\[Any\]",  # [Any]
        r"Any\[",  # Any[
        r"\bDict\[.*,\s*Any\]",  # Dict[str, Any]
        r"\bList\[Any\]",  # List[Any]
        r"\bOptional\[Any\]",  # Optional[Any]
        r"\bUnion\[.*Any.*\]",  # Union[..., Any, ...]
    ]

    uses_any = any(re.search(pattern, content) for pattern in any_patterns)

    if not uses_any:
        return False

    # Check if 'Any' is imported
    import_patterns = [
        r"from typing import.*\bAny\b",
        r"from typing import.*\*",
        r"import typing\b.*as",
        r"import typing\b",
        r"from typing_extensions import.*\bAny\b",
    ]

    has_import = any(re.search(pattern, content, re.MULTILINE) for pattern in import_patterns)

    # Also check if typing.Any is used with full qualification
    if re.search(r"typing\.Any", content):
        has_import = True

    return not has_import


# Find all Python files
python_files = []
for root, dirs, files in os.walk("."):
    # Skip hidden directories and common non-source directories
    dirs[:] = [
        d
        for d in dirs
        if not d.startswith(".") and d not in ["venv", "env", "__pycache__", "node_modules"]
    ]

    for file in files:
        if file.endswith(".py"):
            python_files.append(os.path.join(root, file))

# Check each Python file
files_with_errors = []
for filepath in python_files:
    try:
        if check_file_for_any_error(filepath):
            # Get line numbers where Any is used
            with open(filepath, "r") as f:
                lines = f.readlines()
            any_lines = []
            for i, line in enumerate(lines, 1):
                if re.search(r":\s*Any\b|->\s*Any\b|\[Any\]|Any\[", line):
                    any_lines.append(i)
            if any_lines:
                files_with_errors.append((filepath, any_lines))
    except Exception:
        pass

# Print results
print(f"Found {len(files_with_errors)} Python files with 'Any' not imported:\n")
for i, (filepath, line_nums) in enumerate(files_with_errors[:10], 1):
    print(f"{i}. {filepath}")
    print(f"   Lines with Any: {line_nums[:5]}{'...' if len(line_nums) > 5 else ''}")
    print()

if len(files_with_errors) > 10:
    print(f"... and {len(files_with_errors) - 10} more files")
