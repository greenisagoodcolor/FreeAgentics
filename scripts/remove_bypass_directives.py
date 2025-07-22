#!/usr/bin/env python3
"""Remove all bypass directives from the codebase."""

import re
import sys
from pathlib import Path
from typing import List

# Patterns to remove or fix
PATTERNS_TO_REMOVE = [
    # Skip patterns in various formats
    (r"\[skip\s+ci\]", ""),
    (r"\[ci\s+skip\]", ""),
    (r"", ""),
    (r"--skip=[\w,]+", ""),
    (r"skip:\s*\[.*?\].*?#.*?Skip.*?$", ""),
    (r"\|\|\s*true", ""),  # Remove  bypass
    (r"continue-on-error:\s*true", "continue-on-error: false"),
    (r"allow_failure:\s*true", "allow_failure: false"),
    (r"fail_ci_if_error:\s*false", "fail_ci_if_error: true"),
    # Type ignore patterns
    (r"#\s*type:\s*ignore.*$", ""),
    (r"#\s*noqa:?\s*[\w\d,]*", ""),
    (r"#\s*ruff:\s*noqa.*$", ""),
    (r"#\s*mypy:\s*ignore.*$", ""),
    # Test skip patterns
    (r"@pytest\.mark\.skip\(.*?\)", ""),
    (r"@pytest\.mark\.skipif\(.*?\)", ""),
    (r"@pytest\.mark\.xfail.*$", ""),
    # ESLint patterns
    (r"//\s*eslint-disable.*$", ""),
    (r"/\*\s*eslint-disable.*?\*/", ""),
    (r"//\s*@ts-ignore.*$", ""),
    (r"/\*\s*@ts-ignore.*?\*/", ""),
    # Coverage threshold fix
    (r"fail_under\s*=\s*\d+", "fail_under = 80"),
    # Skip library check
    (r'"skipLibCheck":\s*true', '"skipLibCheck": false'),
]


def should_process_file(file_path: Path) -> bool:
    """Check if file should be processed."""
    # Skip certain directories
    skip_dirs = {
        ".git",
        "node_modules",
        "venv",
        "__pycache__",
        ".pytest_cache",
        "htmlcov",
    }
    if any(part in skip_dirs for part in file_path.parts):
        return False

    # Process specific file types
    extensions = {
        ".py",
        ".yml",
        ".yaml",
        ".ts",
        ".tsx",
        ".js",
        ".jsx",
        ".json",
        ".toml",
        ".ini",
        ".cfg",
    }
    return file_path.suffix in extensions or file_path.name in {
        ".flake8",
        "Makefile",
        "Jenkinsfile",
    }


def remove_bypass_directives(file_path: Path) -> List[str]:
    """Remove bypass directives from a file."""
    changes = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            original_content = content
    except Exception as e:
        return [f"Error reading {file_path}: {e}"]

    # Apply patterns
    for pattern, replacement in PATTERNS_TO_REMOVE:
        new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE | re.IGNORECASE)
        if new_content != content:
            changes.append(f"Removed pattern '{pattern}' from {file_path}")
            content = new_content

    # Write back if changed
    if content != original_content:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            changes.append(f"Updated {file_path}")
        except Exception as e:
            changes.append(f"Error writing {file_path}: {e}")

    return changes


def main():
    """Main function."""
    root_dir = Path("/home/green/FreeAgentics")
    all_changes = []

    print("Scanning for bypass directives...")

    # Process all files
    for file_path in root_dir.rglob("*"):
        if file_path.is_file() and should_process_file(file_path):
            changes = remove_bypass_directives(file_path)
            all_changes.extend(changes)

    # Print summary
    print(f"\nTotal changes made: {len(all_changes)}")
    for change in all_changes[:50]:  # Show first 50 changes
        print(f"  - {change}")

    if len(all_changes) > 50:
        print(f"  ... and {len(all_changes) - 50} more changes")

    return 0 if all_changes else 1


if __name__ == "__main__":
    sys.exit(main())
