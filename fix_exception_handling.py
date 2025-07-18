#!/usr/bin/env python3
"""Fix exception handling issues."""

import os
import re
from pathlib import Path


def fix_exception_handlers(content):
    """Fix exception handlers that use the exception variable."""
    lines = content.split("\n")
    new_lines = []

    for i, line in enumerate(lines):
        # Check if this is an except clause with underscore
        if match := re.match(r"(\s*)except\s+(\w+)\s+as\s+_\s*:", line):
            indent, exception_type = match.groups()

            # Look ahead to see if 'e' is used in the exception block
            e_used = False
            j = i + 1
            while j < len(lines) and (
                not lines[j].strip() or lines[j].startswith(indent + " ")
            ):
                if re.search(r"\be\b", lines[j]):
                    e_used = True
                    break
                j += 1

            if e_used:
                # Revert to using 'e' instead of '_'
                new_lines.append(f"{indent}except {exception_type} as e:")
            else:
                new_lines.append(line)
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
        content = fix_exception_handlers(content)

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
