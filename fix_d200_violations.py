#!/usr/bin/env python3
"""Script to fix D200 violations (one-line docstrings should fit on one line)."""

import re
import sys
from pathlib import Path


def fix_d200_violations(filepath):
    """Fix D200 violations in a Python file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    modified = False
    while i < len(lines):
        # Check for docstring pattern that spans 3 lines
        if i + 2 < len(lines):
            line1 = lines[i]
            line2 = lines[i + 1]
            line3 = lines[i + 2]

            # Pattern: '"""' on first line, content on second, '"""' on third
            indent_match = re.match(r'^(\s*)', line1)
            if indent_match:
                indent = indent_match.group(1)
                if (
                    line1.strip() == '"""'
                    and line3.strip() == '"""'
                    and line2.strip()
                    and not '\n\n' in line2
                ):  # Single line content
                    # Check if it can fit on one line
                    content = line2.strip()
                    one_line = f'{indent}"""{content}"""\n'
                    if (
                        len(one_line.rstrip()) <= 100
                    ):  # Within line length limit
                        lines[i] = one_line
                        lines.pop(i + 1)
                        lines.pop(i + 1)
                        modified = True
                        continue
        i += 1

    if modified:
        with open(filepath, 'w') as f:
            f.writelines(lines)
        return True
    return False


def main():
    """Main function to process files."""
    if len(sys.argv) < 2:
        print("Usage: python fix_d200_violations.py <file1> [file2] ...")
        sys.exit(1)

    for filepath in sys.argv[1:]:
        path = Path(filepath)
        if path.exists() and path.suffix == '.py':
            if fix_d200_violations(path):
                print(f"Fixed D200 violations in {path}")
            else:
                print(f"No D200 violations found in {path}")
        else:
            print(f"Skipping {path} (not a Python file or doesn't exist)")


if __name__ == "__main__":
    main()
