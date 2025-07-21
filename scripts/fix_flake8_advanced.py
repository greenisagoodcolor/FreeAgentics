#!/usr/bin/env python3
"""
Advanced script to fix flake8 violations systematically.
"""

import argparse
import os
import re
import subprocess
from typing import List


def run_flake8(filepath: str) -> List[str]:
    """Run flake8 on a file and return violations."""
    try:
        result = subprocess.run(
            [
                "flake8",
                filepath,
                "--format=%(path)s:%(row)d:%(col)d: %(code)s %(text)s",
            ],
            capture_output=True,
            text=True,
        )
        return result.stdout.strip().split("\n") if result.stdout else []
    except Exception as e:
        print(f"Error running flake8: {e}")
        return []


def fix_line_too_long(lines: List[str], line_num: int) -> List[str]:
    """Fix E501: line too long."""
    if line_num >= len(lines):
        return lines

    line = lines[line_num]
    if len(line.rstrip()) <= 79:
        return lines

    # Handle different types of long lines
    stripped = line.strip()
    indent = len(line) - len(line.lstrip())
    indent_str = " " * indent

    # Long strings
    if '"' in line or "'" in line:
        # Try to split at natural break points
        if " and " in line:
            parts = line.split(" and ", 1)
            lines[line_num] = parts[0] + " and"
            lines.insert(line_num + 1, indent_str + "    " + parts[1].lstrip())
        elif " or " in line:
            parts = line.split(" or ", 1)
            lines[line_num] = parts[0] + " or"
            lines.insert(line_num + 1, indent_str + "    " + parts[1].lstrip())
        elif "," in line and not line.strip().startswith(("def ", "class ")):
            # Split at last comma that keeps line under 79 chars
            best_split = -1
            for i, char in enumerate(line):
                if char == "," and i < 79:
                    best_split = i
            if best_split > 0:
                lines[line_num] = line[: best_split + 1]
                lines.insert(
                    line_num + 1,
                    indent_str + "    " + line[best_split + 1 :].lstrip(),
                )

    # Function definitions
    elif stripped.startswith(("def ", "async def ")):
        # Split parameters
        match = re.match(r"^(\s*(?:async\s+)?def\s+\w+)\((.*)\)(.*)$", line)
        if match:
            prefix, params, suffix = match.groups()
            if "," in params:
                param_list = params.split(",")
                lines[line_num] = prefix + "("
                for i, param in enumerate(param_list[:-1]):
                    lines.insert(
                        line_num + 1 + i,
                        indent_str + "        " + param.strip() + ",",
                    )
                lines.insert(
                    line_num + 1 + len(param_list) - 1,
                    indent_str + "        " + param_list[-1].strip() + ")" + suffix,
                )

    # Long conditionals
    elif any(op in line for op in [" and ", " or ", " if ", " else "]):
        # Use parentheses for line continuation
        if not line.rstrip().endswith("\\"):
            if " if " in line and " else " in line:  # Ternary
                parts = line.split(" if ", 1)
                if len(parts) == 2:
                    lines[line_num] = parts[0] + " if ("
                    rest = parts[1]
                    if " else " in rest:
                        cond, else_part = rest.split(" else ", 1)
                        lines.insert(line_num + 1, indent_str + "    " + cond.strip())
                        lines.insert(
                            line_num + 2,
                            indent_str + ") else " + else_part.strip(),
                        )

    return lines


def fix_unused_imports(lines: List[str]) -> List[str]:
    """Fix F401: imported but unused."""
    # Collect all identifiers used in the file
    used_names = set()
    import_lines = {}

    for i, line in enumerate(lines):
        if line.strip().startswith(("import ", "from ")):
            # Parse imports
            if " import " in line:
                parts = line.split(" import ")
                if len(parts) == 2:
                    imports = parts[1].strip()
                    # Handle multiple imports
                    for imp in imports.split(","):
                        imp = imp.strip()
                        if " as " in imp:
                            name = imp.split(" as ")[1].strip()
                        else:
                            name = imp.strip("()")
                        if name not in import_lines:
                            import_lines[name] = i
            elif line.strip().startswith("import "):
                name = line.strip()[7:].split(".")[0]
                if " as " in line:
                    name = line.split(" as ")[1].strip()
                import_lines[name] = i
        else:
            # Collect used names
            # Simple regex to find identifiers
            words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", line)
            used_names.update(words)

    # Remove unused imports
    lines_to_remove = set()
    for name, line_idx in import_lines.items():
        if name not in used_names and line_idx < len(lines):
            # Check if it's a multi-import line
            line = lines[line_idx]
            if "," in line and " import " in line:
                # Remove just this import
                parts = line.split(" import ")
                import_list: list[str] = parts[1].split(",")
                new_imports: list[str] = []
                for imp in import_list:
                    imp_name = imp.strip()
                    if " as " in imp_name:
                        imp_name = imp_name.split(" as ")[1].strip()
                    if imp_name != name:
                        new_imports.append(imp.strip())
                if new_imports:
                    lines[line_idx] = parts[0] + " import " + ", ".join(new_imports)
                else:
                    lines_to_remove.add(line_idx)
            else:
                lines_to_remove.add(line_idx)

    # Remove lines in reverse order
    for idx in sorted(lines_to_remove, reverse=True):
        if idx < len(lines):
            lines.pop(idx)

    return lines


def fix_blank_lines(lines: List[str]) -> List[str]:
    """Fix E302, E303, E305: blank line issues."""
    new_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Remove extra blank lines (E303)
        if not line.strip():
            blank_count = 0
            j = i
            while j < len(lines) and not lines[j].strip():
                blank_count += 1
                j += 1

            # Keep at most 2 blank lines
            if blank_count > 2:
                i = j - 1
                new_lines.extend(["", ""])
            else:
                new_lines.append(line)

        # Check for function/class definitions
        elif line.strip().startswith(("def ", "class ", "async def ")):
            # Count preceding blank lines
            blank_before = 0
            j = len(new_lines) - 1
            while j >= 0 and not new_lines[j].strip():
                blank_before += 1
                j -= 1

            # Check if preceded by decorator
            is_decorated = j >= 0 and new_lines[j].strip().startswith("@")

            # Ensure proper spacing
            if not is_decorated and j >= 0:
                # Need 2 blank lines before top-level definitions
                if len(line) - len(line.lstrip()) == 0:  # Top-level
                    while blank_before < 2:
                        new_lines.append("")
                        blank_before += 1
                    while blank_before > 2:
                        new_lines.pop()
                        blank_before -= 1

            new_lines.append(line)
        else:
            new_lines.append(line)

        i += 1

    return new_lines


def fix_indentation_errors(lines: List[str], line_num: int, col: int) -> List[str]:
    """Fix E128: continuation line under-indented."""
    if line_num >= len(lines) or line_num == 0:
        return lines

    current_line = lines[line_num]
    prev_line = lines[line_num - 1]

    # Find the opening bracket
    open_brackets = {"(": ")", "[": "]", "{": "}"}
    for bracket, close in open_brackets.items():
        if bracket in prev_line:
            # Find position of opening bracket
            bracket_pos = prev_line.rfind(bracket)
            if bracket_pos >= 0:
                # Align with opening bracket
                expected_indent = bracket_pos + 1
                current_indent = len(current_line) - len(current_line.lstrip())

                if current_indent < expected_indent:
                    # Fix indentation
                    lines[line_num] = " " * expected_indent + current_line.lstrip()
                break

    return lines


def fix_missing_placeholders(lines: List[str], line_num: int) -> List[str]:
    """Fix F541: f-string is missing placeholders."""
    if line_num >= len(lines):
        return lines

    line = lines[line_num]
    # Remove f prefix from strings without placeholders
    lines[line_num] = re.sub(
        r'\bf"([^"]*)"',
        lambda m: '"' + str(m.group(1)) + '"' if "{" not in str(m.group(1)) else m.group(0),
        line,
    )
    lines[line_num] = re.sub(
        r"\bf'([^']*)'",
        lambda m: "'" + str(m.group(1)) + "'" if "{" not in str(m.group(1)) else m.group(0),
        lines[line_num],
    )

    return lines


def fix_file_comprehensive(filepath: str) -> bool:
    """Fix all flake8 violations in a file."""
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return False

    # Read file
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Strip trailing newlines but preserve structure
    lines = [line.rstrip("\n") for line in lines]

    # Get current violations
    violations = run_flake8(filepath)
    if not violations or violations == [""]:
        print(f"{filepath}: No violations found")
        return True

    print(f"{filepath}: Found {len(violations)} violations")

    # Parse violations and fix
    for violation in violations:
        if not violation:
            continue

        match = re.match(r"^[^:]+:(\d+):(\d+):\s*([A-Z]\d+)\s+(.*)$", violation)
        if not match:
            continue

        line_num = int(match.group(1)) - 1  # 0-based
        col = int(match.group(2))
        code = match.group(3)
        match.group(4)

        # Apply fixes based on error code
        if code == "E501":  # Line too long
            lines = fix_line_too_long(lines, line_num)
        elif code == "F401":  # Unused import
            lines = fix_unused_imports(lines)
        elif code in ["E302", "E303", "E305"]:  # Blank lines
            lines = fix_blank_lines(lines)
        elif code == "E128":  # Indentation
            lines = fix_indentation_errors(lines, line_num, col)
        elif code == "F541":  # f-string missing placeholders
            lines = fix_missing_placeholders(lines, line_num)

    # Clean up trailing whitespace and ensure newline at end
    lines = [line.rstrip() for line in lines]
    while lines and not lines[-1]:
        lines.pop()
    if lines and lines[-1]:
        lines.append("")

    # Write back
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Check if we fixed everything
    remaining = run_flake8(filepath)
    remaining_count = len([v for v in remaining if v])

    print(f"{filepath}: Fixed, {remaining_count} violations remaining")
    return remaining_count == 0


def main():
    parser = argparse.ArgumentParser(description="Fix flake8 violations comprehensively")
    parser.add_argument("paths", nargs="+", help="Files or directories to fix")
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum fix iterations per file",
    )

    args = parser.parse_args()

    # Collect Python files
    files_to_fix = []
    for path in args.paths:
        if os.path.isfile(path) and path.endswith(".py"):
            files_to_fix.append(path)
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                if "__pycache__" in root or ".git" in root:
                    continue
                for file in files:
                    if file.endswith(".py"):
                        files_to_fix.append(os.path.join(root, file))

    print(f"Processing {len(files_to_fix)} files...")

    success_count = 0
    for filepath in sorted(files_to_fix):
        # Try fixing multiple times as some fixes enable others
        for iteration in range(args.max_iterations):
            if fix_file_comprehensive(filepath):
                success_count += 1
                break

    print(f"\nFixed {success_count}/{len(files_to_fix)} files completely")


if __name__ == "__main__":
    main()
