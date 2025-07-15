#!/usr/bin/env python3
"""
Script to fix E501 (line too long) violations in Python files.
"""

import argparse
import os
import re
from typing import List, Tuple


def split_long_string(line: str, max_length: int = 79) -> List[str]:
    """Split a long string literal into multiple lines."""
    # Find string delimiters
    string_match = re.search(r'(["\'])(.+?)\1', line)
    if not string_match:
        return [line]

    quote, content = string_match.groups()
    prefix = line[: string_match.start()]
    suffix = line[string_match.end() :]

    # If it's a docstring or multiline string, leave it
    if quote * 3 in line:
        return [line]

    # Calculate available space
    indent = len(prefix) - len(prefix.lstrip())
    available = max_length - indent - 4  # Leave room for quotes and continuation

    if len(content) <= available:
        return [line]

    # Split the string
    parts = []
    while content:
        # Try to split at a space
        split_point = available
        space_idx = content[:split_point].rfind(" ")
        if space_idx > available * 0.5:  # Only split at space if it's reasonable
            split_point = space_idx + 1

        parts.append(content[:split_point])
        content = content[split_point:]

    # Reconstruct lines
    lines = []
    for i, part in enumerate(parts):
        if i == 0:
            lines.append(f"{prefix}{quote}{part}{quote}")
        else:
            lines.append(f"{' ' * indent}{quote}{part}{quote}")

    # Add suffix to last line
    if suffix.strip():
        lines[-1] += suffix

    return lines


def split_function_call(line: str, max_length: int = 79) -> List[str]:
    """Split a long function call into multiple lines."""
    # Find function call pattern
    match = re.match(r"^(\s*)(.+?)\((.*)\)(.*)$", line)
    if not match:
        return [line]

    indent_str, func_name, args, suffix = match.groups()
    len(indent_str)

    # If no arguments or already formatted, skip
    if not args.strip() or "\n" in line:
        return [line]

    # Split arguments
    arg_parts = []
    current_arg = ""
    paren_depth = 0
    in_string = False
    string_char = None

    for char in args:
        if not in_string:
            if char in "\"'":
                in_string = True
                string_char = char
            elif char == "(":
                paren_depth += 1
            elif char == ")":
                paren_depth -= 1
            elif char == "," and paren_depth == 0:
                arg_parts.append(current_arg.strip())
                current_arg = ""
                continue
        elif char == string_char and args[args.index(char) - 1] != "\\":
            in_string = False

        current_arg += char

    if current_arg.strip():
        arg_parts.append(current_arg.strip())

    # Reconstruct with proper formatting
    lines = [f"{indent_str}{func_name}("]
    for i, arg in enumerate(arg_parts):
        if i < len(arg_parts) - 1:
            lines.append(f"{indent_str}    {arg},")
        else:
            lines.append(f"{indent_str}    {arg}")
    lines.append(f"{indent_str}){suffix}")

    return lines


def split_import_statement(line: str, max_length: int = 79) -> List[str]:
    """Split a long import statement."""
    if " import " in line:
        # from X import Y, Z, ...
        match = re.match(r"^(\s*from\s+\S+\s+import\s+)(.+)$", line)
        if match:
            prefix, imports = match.groups()
            import_list = [imp.strip() for imp in imports.split(",")]

            if len(import_list) > 1:
                lines = [prefix + "("]
                for imp in import_list[:-1]:
                    lines.append(f"    {imp},")
                lines.append(f"    {import_list[-1]}")
                lines.append(")")
                return lines

    return [line]


def fix_long_lines_in_file(filepath: str, max_length: int = 79) -> Tuple[int, int]:
    """Fix long lines in a Python file."""
    if not os.path.exists(filepath):
        return 0, 0

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    fixed_count = 0

    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")

        if len(line) > max_length:
            # Skip if it's a URL or similar
            if "http://" in line or "https://" in line:
                new_lines.append(line)
            # Try different splitting strategies
            elif " import " in line and line.strip().startswith(("from ", "import ")):
                split = split_import_statement(line, max_length)
                new_lines.extend(split)
                fixed_count += 1 if len(split) > 1 else 0
            elif "(" in line and ")" in line and "=" not in line[: line.find("(")]:
                split = split_function_call(line, max_length)
                new_lines.extend(split)
                fixed_count += 1 if len(split) > 1 else 0
            elif '"' in line or "'" in line:
                split = split_long_string(line, max_length)
                new_lines.extend(split)
                fixed_count += 1 if len(split) > 1 else 0
            else:
                # Generic splitting at operators
                if " and " in line:
                    parts = line.split(" and ", 1)
                    new_lines.append(parts[0] + " and")
                    new_lines.append(" " * (len(line) - len(line.lstrip()) + 4) + parts[1])
                    fixed_count += 1
                elif " or " in line:
                    parts = line.split(" or ", 1)
                    new_lines.append(parts[0] + " or")
                    new_lines.append(" " * (len(line) - len(line.lstrip()) + 4) + parts[1])
                    fixed_count += 1
                else:
                    new_lines.append(line)
        else:
            new_lines.append(line)

        i += 1

    # Count remaining long lines
    remaining = sum(1 for line in new_lines if len(line) > max_length)

    # Write back
    with open(filepath, "w", encoding="utf-8") as f:
        for line in new_lines:
            f.write(line + "\n")

    return fixed_count, remaining


def main():
    parser = argparse.ArgumentParser(description="Fix E501 line length violations")
    parser.add_argument("paths", nargs="+", help="Files or directories to fix")
    parser.add_argument("--max-length", type=int, default=79, help="Maximum line length")

    args = parser.parse_args()

    # Collect Python files
    files_to_fix = []
    for path in args.paths:
        if os.path.isfile(path) and path.endswith(".py"):
            files_to_fix.append(path)
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                if "__pycache__" not in root:
                    for file in files:
                        if file.endswith(".py"):
                            files_to_fix.append(os.path.join(root, file))

    print(f"Processing {len(files_to_fix)} files...")

    total_fixed = 0
    total_remaining = 0

    for filepath in sorted(files_to_fix):
        fixed, remaining = fix_long_lines_in_file(filepath, args.max_length)
        if fixed > 0:
            print(f"  {filepath}: Fixed {fixed} long lines, {remaining} remaining")
            total_fixed += fixed
            total_remaining += remaining

    print(f"\nTotal: Fixed {total_fixed} long lines, {total_remaining} remaining")


if __name__ == "__main__":
    main()
