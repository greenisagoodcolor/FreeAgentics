#!/usr/bin/env python3
"""Fix common linting errors automatically."""

import os
import re
import sys
from pathlib import Path


def fix_bare_except(content):
    """Replace bare except with except Exception."""
    return re.sub(r"\bexcept\s*:", "except Exception:", content)


def fix_fstring_no_placeholder(content):
    """Convert f-strings without placeholders to regular strings."""
    lines = content.split("\n")
    new_lines = []

    for line in lines:
        # Match f-strings without placeholders
        if match := re.search(r'(\s*)(f)(["\']{1,3})([^"\']*?)\3', line):
            indent, f, quotes, string_content = match.groups()
            # Check if there are no placeholders
            if "{" not in string_content or "}" not in string_content:
                # Remove the 'f' prefix
                new_line = line.replace(f"{f}{quotes}", quotes, 1)
                new_lines.append(new_line)
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def fix_unused_variables(content):
    """Comment out or use underscore for unused variables."""
    lines = content.split("\n")
    new_lines = []

    for line in lines:
        # Skip if it's already using underscore
        if " _ =" in line or "_ =" in line:
            new_lines.append(line)
            continue

        # For except clauses, replace with underscore
        if match := re.match(r"(\s*)except\s+(\w+)\s+as\s+(\w+)\s*:", line):
            indent, exception_type, var_name = match.groups()
            new_lines.append(f"{indent}except {exception_type} as _:")
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def process_file(filepath):
    """Process a single file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        original_content = content

        # Apply fixes
        content = fix_bare_except(content)
        content = fix_fstring_no_placeholder(content)
        content = fix_unused_variables(content)

        # Only write if changed
        if content != original_content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Fixed: {filepath}")
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    """Main function."""
    fixed_count = 0

    # Find all Python files
    for filepath in Path(".").rglob("*.py"):
        # Skip excluded directories
        if any(
            part in str(filepath)
            for part in [
                "venv/",
                ".venv/",
                "node_modules/",
                "web/",
                ".git/",
                "__pycache__/",
                ".pytest_cache/",
                ".mypy_cache/",
            ]
        ):
            continue

        if process_file(filepath):
            fixed_count += 1

    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()
