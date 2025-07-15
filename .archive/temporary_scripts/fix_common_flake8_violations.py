#!/usr/bin/env python3
"""
Script to automatically fix common flake8 violations.
Targets the most frequent issues:
- W293: blank line contains whitespace
- W291: trailing whitespace
- F401: imported but unused (with caution)
"""

import re
import sys
from pathlib import Path
from typing import List, Set


def fix_whitespace_issues(content: str) -> str:
    """Fix W293 (blank line contains whitespace) and W291 (trailing whitespace)."""
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Remove trailing whitespace
        fixed_line = line.rstrip()

        # If the line is now empty (was just whitespace), keep it empty
        fixed_lines.append(fixed_line)

    return "\n".join(fixed_lines)


def get_unused_imports(file_path: Path) -> Set[str]:
    """Get list of unused imports using flake8."""
    import subprocess

    result = subprocess.run(
        ["flake8", str(file_path), "--select=F401"], capture_output=True, text=True
    )

    unused_imports = set()
    for line in result.stdout.splitlines():
        # Parse flake8 output like: "file.py:1:1: F401 'module' imported but unused"
        match = re.search(r"F401 '([^']+)' imported but unused", line)
        if match:
            unused_imports.add(match.group(1))

    return unused_imports


def fix_file(file_path: Path, fix_imports: bool = False) -> int:
    """Fix common issues in a single file. Returns number of changes made."""
    if not file_path.exists() or not file_path.is_file():
        return 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            original_content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0

    # Fix whitespace issues
    content = fix_whitespace_issues(original_content)

    # Fix unused imports if requested
    if fix_imports and file_path.suffix == ".py":
        unused_imports = get_unused_imports(file_path)
        if unused_imports:
            lines = content.split("\n")
            fixed_lines = []

            for line in lines:
                # Skip lines that import unused modules
                should_skip = False
                for unused in unused_imports:
                    if re.match(rf"^(from\s+\S+\s+)?import\s+.*\b{re.escape(unused)}\b", line):
                        should_skip = True
                        break

                if not should_skip:
                    fixed_lines.append(line)

            content = "\n".join(fixed_lines)

    # Only write if content changed
    if content != original_content:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return 1

    return 0


def main():
    """Main function to fix flake8 violations in the codebase."""
    # Define directories to process
    directories = [
        "agents",
        "api",
        "auth",
        "database",
        "inference",
        "knowledge_graph",
        "observability",
        "scripts",
        "tests",
        "utils",
        "world",
    ]

    total_files_fixed = 0

    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue

        # Find all Python files
        python_files = list(dir_path.rglob("*.py"))

        print(f"\nProcessing {len(python_files)} files in {directory}/...")

        for file_path in python_files:
            if fix_file(file_path, fix_imports=False):  # Start with just whitespace fixes
                total_files_fixed += 1

    print(f"\n✅ Fixed {total_files_fixed} files")

    # Now check remaining violations
    print("\nChecking remaining violations...")
    import subprocess

    result = subprocess.run(
        ["flake8"] + directories + ["--max-line-length=88", "--extend-ignore=E203,W503"],
        capture_output=True,
        text=True,
    )

    # Count remaining violations by type
    violations = {}
    for line in result.stdout.splitlines():
        match = re.search(r":\d+:\d+: ([A-Z]\d+)", line)
        if match:
            code = match.group(1)
            violations[code] = violations.get(code, 0) + 1

    if violations:
        print("\nRemaining violations:")
        for code, count in sorted(violations.items(), key=lambda x: -x[1])[:10]:
            print(f"  {code}: {count}")
    else:
        print("\n🎉 No violations remaining!")


if __name__ == "__main__":
    main()
