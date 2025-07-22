#!/usr/bin/env python3
"""
Script to automatically fix common flake8 violations.
"""

import argparse
import os
import re
from typing import List, Tuple


def fix_line_too_long(line: str, max_length: int = 79) -> List[str]:
    """Fix E501: line too long."""
    if len(line) <= max_length:
        return [line]

    # Handle imports specially
    if line.strip().startswith(("import ", "from ")):
        # For imports, try to break at commas
        if "," in line:
            parts = line.split(",")
            result = [parts[0] + ","]
            for part in parts[1:-1]:
                result.append("    " + part.strip() + ",")
            result.append("    " + parts[-1].strip())
            return result
        else:
            # Single long import - use parentheses
            match = re.match(r"^(\s*from\s+\S+\s+import\s+)(.+)$", line)
            if match:
                return [
                    match.group(1) + "(",
                    "    " + match.group(2).strip(),
                    ")",
                ]

    # For other lines, try to break at logical points
    # This is a simple implementation - real code might need manual adjustment
    len(line) - len(line.lstrip())
    if "(" in line and ")" in line:
        # Try to break at commas within parentheses
        return [line]  # Complex - needs manual fix

    return [line]  # Needs manual fix


def fix_unused_imports(content: str) -> str:
    """Fix F401: imported but unused."""
    lines = content.split("\n")
    used_names = set()

    # Collect all used names (simple approach)
    for line in lines:
        if not line.strip().startswith(("import ", "from ")):
            # Extract potential identifier usage
            words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", line)
            used_names.update(words)

    # Filter imports
    new_lines = []
    for line in lines:
        if line.strip().startswith("import ") or line.strip().startswith("from "):
            # Extract imported names
            imported = []
            if line.strip().startswith("import "):
                parts = line.strip()[7:].split(" as ")
                name = parts[-1] if len(parts) > 1 else parts[0]
                imported.append(name.split(".")[-1])
            elif " import " in line:
                import_part = line.split(" import ")[-1]
                # Handle multiple imports
                for item in import_part.split(","):
                    item = item.strip()
                    if " as " in item:
                        imported.append(item.split(" as ")[-1].strip())
                    else:
                        imported.append(item.strip("()"))

            # Check if any imported name is used
            if any(name in used_names for name in imported):
                new_lines.append(line)
            else:
                # Comment out unused import
                new_lines.append(f"# {line}  # UNUSED")
        else:
            new_lines.append(line)

    return "\n".join(new_lines)


def fix_blank_lines(content: str) -> str:
    """Fix E302, E303, E305: blank line issues."""
    lines = content.split("\n")
    new_lines: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check for class or function definition
        if line.strip().startswith(("def ", "class ", "async def ")):
            # Ensure 2 blank lines before (unless at start or after decorator)
            if i > 0 and new_lines:
                # Count existing blank lines
                blank_count = 0
                j = len(new_lines) - 1
                while j >= 0 and not new_lines[j].strip():
                    blank_count += 1
                    j -= 1

                # Check if previous line is decorator
                is_decorator = j >= 0 and new_lines[j].strip().startswith("@")

                if not is_decorator:
                    # Ensure exactly 2 blank lines
                    while blank_count < 2:
                        new_lines.append("")
                        blank_count += 1
                    while blank_count > 2:
                        new_lines.pop()
                        blank_count -= 1

        new_lines.append(line)
        i += 1

    # Remove trailing blank lines
    while new_lines and not new_lines[-1].strip():
        new_lines.pop()

    # Ensure newline at end
    if new_lines and new_lines[-1].strip():
        new_lines.append("")

    return "\n".join(new_lines)


def fix_whitespace(content: str) -> str:
    """Fix W291, W292, W293: whitespace issues."""
    lines = content.split("\n")
    new_lines = []

    for line in lines:
        # Remove trailing whitespace
        line = line.rstrip()
        # Fix blank lines with whitespace
        if not line.strip():
            line = ""
        new_lines.append(line)

    # Ensure newline at end of file
    if new_lines and new_lines[-1]:
        new_lines.append("")

    return "\n".join(new_lines)


def fix_file(filepath: str, auto_fix: bool = True) -> Tuple[str, List[str]]:
    """Fix flake8 violations in a file."""
    if not os.path.exists(filepath):
        return filepath, [f"File not found: {filepath}"]

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return filepath, [f"Error reading file: {e}"]

    original_content = content
    issues = []

    # Apply fixes in order
    try:
        # Fix whitespace issues first
        content = fix_whitespace(content)

        # Fix blank line issues
        content = fix_blank_lines(content)

        # Fix unused imports
        content = fix_unused_imports(content)

        # Skip autopep8 if not available
        # Additional manual fixes can be added here

        # Save if changed
        if content != original_content:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            issues.append("Fixed and saved")
        else:
            issues.append("No changes needed")

    except Exception as e:
        issues.append(f"Error fixing file: {e}")

    return filepath, issues


def main():
    parser = argparse.ArgumentParser(description="Fix flake8 violations")
    parser.add_argument("paths", nargs="+", help="Files or directories to fix")
    parser.add_argument("--no-auto", action="store_true", help="Disable autopep8")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fixed")

    args = parser.parse_args()

    files_to_fix = []
    for path in args.paths:
        if os.path.isfile(path):
            if path.endswith(".py"):
                files_to_fix.append(path)
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                if "__pycache__" in root or ".git" in root:
                    continue
                for file in files:
                    if file.endswith(".py"):
                        files_to_fix.append(os.path.join(root, file))

    print(f"Found {len(files_to_fix)} Python files to check")

    for filepath in sorted(files_to_fix):
        if args.dry_run:
            print(f"Would fix: {filepath}")
        else:
            filepath, issues = fix_file(filepath, auto_fix=not args.no_auto)
            print(f"{filepath}: {', '.join(issues)}")


if __name__ == "__main__":
    main()
