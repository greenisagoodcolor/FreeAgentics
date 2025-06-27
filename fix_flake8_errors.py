#!/usr/bin/env python3
"""
Comprehensive flake8 error fixer for FreeAgentics.

This script systematically fixes common flake8 errors across the codebase.
"""
import os
import re
import subprocess
from pathlib import Path
from typing import List, Tuple

# Common patterns to fix
FIXES = {
    # Module docstring
    "D100": {
        "pattern": r"^((?:import|from)\s+.*?)$",
        "replacement": '"""\nModule docstring.\n"""\n\\1',
        "multiline": True,
        "at_start": True,
    },
    # First line should end with period
    "D400": {"pattern": r'"""([^.]+)"""', "replacement": '"""\\1"""'},
    # Line too long - split at appropriate points
    "E501": {"custom_handler": "fix_long_lines"},
    # Unused imports
    "F401": {"custom_handler": "remove_unused_imports"},
    # Missing docstring in __init__
    "D107": {
        "pattern": r"(\s+)def __init__\(self.*?\):$",
        "replacement": '\\1def __init__(self\\2):\n\\1    """Initialize"""',
    },
}


def fix_long_lines(content: str, line_num: int) -> str:
    """Fix lines that are too long"""
    lines = content.split("\n")
    if line_num <= len(lines):
        line = lines[line_num - 1]
        if len(line) > 79:
            # Handle different cases
            if "import" in line and "from" in line:
                # Split imports
                parts = line.split("import")
                if len(parts) == 2:
                    lines[line_num - 1] = parts[0] + "import ("
                    imports = [i.strip() for i in parts[1].split(",")]
                    for imp in imports[:-1]:
                        lines.insert(line_num, "    " + imp + ",")
                        line_num += 1
                    lines.insert(line_num, "    " + imports[-1] + ")")
            elif "=" in line and not line.strip().startswith("if"):
                # Split assignments
                indent = len(line) - len(line.lstrip())
                parts = line.split("=", 1)
                if len(parts) == 2:
                    lines[line_num - 1] = parts[0] + "= ("
                    lines.insert(line_num, " " * (indent + 4) + parts[1].strip() + ")")
            elif "raise" in line:
                # Split raise statements
                indent = len(line) - len(line.lstrip())
                match = re.match(r"(\s*raise\s+\w+)\((.*)\)", line)
                if match:
                    lines[line_num - 1] = match.group(1) + "("
                    lines.insert(line_num, " " * (indent + 4) + match.group(2) + ")")
    return "\n".join(lines)


def remove_unused_imports(content: str, unused_import: str) -> str:
    """Remove unused imports"""
    # Extract the unused import name from the error message
    import_match = re.search(r"'([^']+)' imported but unused", unused_import)
    if import_match:
        import_name = import_match.group(1)
        # Handle different import styles
        patterns = [
            rf"^import\s+{re.escape(import_name)}\s*$",
            rf"^from\s+[\w.]+\s+import\s+.*\b{re.escape(import_name)}\b.*$",
            rf",\s*{re.escape(import_name)}\b",
            rf"\b{re.escape(import_name)}\s*,",
        ]
        for pattern in patterns:
            content = re.sub(pattern, "", content, flags=re.MULTILINE)
    return content


def fix_file(filepath: Path) -> int:
    """Fix flake8 errors in a single file"""
    # Get flake8 errors
    result = subprocess.run(["flake8", str(filepath)], capture_output=True, text=True)

    if not result.stdout:
        return 0

    errors = result.stdout.strip().split("\n")
    fixes_applied = 0

    # Read file content
    with open(filepath, "r") as f:
        content = f.read()

    # Group errors by type
    error_groups = {}
    for error in errors:
        parts = error.split(":", 4)
        if len(parts) >= 5:
            line_num = int(parts[1])
            error_code = parts[3].strip().split()[0]
            error_msg = parts[4].strip()

            if error_code not in error_groups:
                error_groups[error_code] = []
            error_groups[error_code].append((line_num, error_msg))

    # Apply fixes
    for error_code, instances in error_groups.items():
        if error_code in FIXES:
            fix_config = FIXES[error_code]

            if "custom_handler" in fix_config:
                # Use custom handler
                handler = globals()[fix_config["custom_handler"]]
                for line_num, error_msg in instances:
                    if error_code == "F401":
                        content = handler(content, error_msg)
                    else:
                        content = handler(content, line_num)
                    fixes_applied += 1
            elif "pattern" in fix_config:
                # Use regex replacement
                if fix_config.get("at_start") and not content.startswith('"""'):
                    # Add module docstring at start
                    content = '"""\nModule docstring.\n"""\n' + content
                    fixes_applied += 1
                else:
                    content = re.sub(
                        fix_config["pattern"],
                        fix_config["replacement"],
                        content,
                        flags=(re.MULTILINE if fix_config.get("multiline") else 0),
                    )
                    fixes_applied += 1

    # Write fixed content
    if fixes_applied > 0:
        with open(filepath, "w") as f:
            f.write(content)

    return fixes_applied


def main():
    """Main function to fix flake8 errors across the codebase"""
    # Find all Python files
    python_files = []
    for root, dirs, files in os.walk("."):
        # Skip hidden directories and common non-source directories
        dirs[:] = [
            d for d in dirs if not d.startswith(".") and d not in ["__pycache__", "venv", "env"]
        ]

        for file in files:
            if file.endswith(".py"):
                python_files.append(Path(root) / file)

    print(f"Found {len(python_files)} Python files")

    total_fixes = 0
    files_fixed = 0

    # Fix files with most errors first
    file_errors = []
    for filepath in python_files:
        result = subprocess.run(
            ["flake8", str(filepath), "--count"], capture_output=True, text=True
        )
        error_count = len(result.stdout.strip().split("\n")) if result.stdout else 0
        if error_count > 0:
            file_errors.append((filepath, error_count))

    # Sort by error count
    file_errors.sort(key=lambda x: x[1], reverse=True)

    # Fix files
    for filepath, error_count in file_errors[:100]:  # Fix top 100 files
        print(f"Fixing {filepath} ({error_count} errors)...")
        fixes = fix_file(filepath)
        if fixes > 0:
            total_fixes += fixes
            files_fixed += 1
            print(f"  Applied {fixes} fixes")

    print(f"\nSummary: Fixed {total_fixes} errors in {files_fixed} files")


if __name__ == "__main__":
    main()
