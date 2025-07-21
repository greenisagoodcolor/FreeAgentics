#!/usr / bin / env python3
"""Comprehensive quality fix script."""

import re
import subprocess
from pathlib import Path
from typing import List, Tuple


def fix_unused_variables(content: str) -> str:
    """Fix F841 unused variable errors."""
    lines = content.split("\n")
    new_lines = []

    for line in lines:
        # Skip if already using underscore
        if " _ =" in line or "_ =" in line:
            new_lines.append(line)
            continue

        # Pattern for unused variable assignments
        match = re.match(r"^(\s*)([\w_]+)\s*=\s*(.+)$", line)
        if match and not line.strip().startswith("#"):
            indent, var_name, value = match.groups()
            # Check if it looks like an intentionally unused variable
            if var_name not in [
                "self",
                "cls",
            ] and not value.strip().startswith("("):
                # Check if variable is used in next few lines
                var_used = False
                for i in range(
                    len(new_lines), min(len(new_lines) + 10, len(lines))
                ):
                    if i < len(lines) and re.search(
                        rf"\b{var_name}\b", lines[i]
                    ):
                        var_used = True
                        break

                if not var_used and "logger" not in var_name:
                    # Comment out the line instead of using underscore
                    new_lines.append(
                        f"{indent}# {var_name} = {value}  # Unused variable"
                    )
                    continue

        new_lines.append(line)

    return "\n".join(new_lines)


def fix_undefined_names(content: str) -> str:
    """Fix F821 undefined name errors."""
    # Common undefined names and their imports
    common_imports = {
        "List": "from typing import List",
        "Dict": "from typing import Dict",
        "Any": "from typing import Any",
        "Optional": "from typing import Optional",
        "Union": "from typing import Union",
        "Tuple": "from typing import Tuple",
        "TYPE_CHECKING": "from typing import TYPE_CHECKING",
    }

    lines = content.split("\n")
    imports_to_add = set()

    # Find undefined names
    for line in lines:
        for name, import_stmt in common_imports.items():
            if re.search(rf"\b{name}\b", line) and import_stmt not in content:
                imports_to_add.add(import_stmt)

    if imports_to_add:
        # Find where to insert imports
        import_index = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                import_index = i + 1
            elif (
                import_index > 0
                and line.strip()
                and not line.startswith("import")
                and not line.startswith("from")
            ):
                break

        # Insert imports
        for imp in sorted(imports_to_add):
            lines.insert(import_index, imp)
            import_index += 1

    return "\n".join(lines)


def fix_whitespace_issues(content: str) -> str:
    """Fix E226 missing whitespace around operators."""
    # Fix common whitespace issues
    content = re.sub(
        r"(\w)(\+|\-|\*|\/|\%|\=\=|\!\=|\<\=|\>\=)(\w)", r"\1 \2 \3", content
    )
    content = re.sub(r"(\w)(\+|\-|\*|\/|\%)=(\w)", r"\1 \2= \3", content)
    return content


def get_flake8_errors(filepath: Path) -> List[Tuple[int, str, str]]:
    """Get flake8 errors for a file."""
    try:
        result = subprocess.run(
            ["flake8", "--config=.flake8.minimal", str(filepath)],
            capture_output=True,
            text=True,
        )
        errors = []
        for line in result.stdout.split("\n"):
            if line:
                parts = line.split(":")
                if len(parts) >= 4:
                    line_num = int(parts[1])
                    parts[2]
                    error_msg = ":".join(parts[3:]).strip()
                    error_code = error_msg.split()[0] if error_msg else ""
                    errors.append((line_num, error_code, error_msg))
        return errors
    except Exception:
        return []


def fix_file(filepath: Path) -> bool:
    """Fix a single file."""
    try:
        # Get current errors
        errors = get_flake8_errors(filepath)
        if not errors:
            return False

        with open(filepath, "r", encoding="utf - 8") as f:
            content = f.read()

        original_content = content

        # Count error types
        error_counts = {}
        for _, code, _ in errors:
            error_counts[code] = error_counts.get(code, 0) + 1

        # Apply fixes based on error types
        if "F841" in error_counts:
            content = fix_unused_variables(content)

        if "F821" in error_counts:
            content = fix_undefined_names(content)

        if "E226" in error_counts:
            content = fix_whitespace_issues(content)

        # Only write if changed
        if content != original_content:
            with open(filepath, "w", encoding="utf - 8") as f:
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
    total_errors_before = 0
    total_errors_after = 0

    # Get initial error count
    result = subprocess.run(
        ["flake8", "--config=.flake8.minimal", "--count", "--quiet", "."],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        total_errors_before = (
            int(result.stdout.strip()) if result.stdout.strip() else 0
        )

    # Find all Python files
    python_files = []
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
        python_files.append(filepath)

    print(f"Found {len(python_files)} Python files to check")

    # Process files in batches
    batch_size = 50
    for i in range(0, len(python_files), batch_size):
        batch = python_files[i : i + batch_size]
        print(
            f"\nProcessing batch {i//batch_size + 1}/{(len(python_files) + batch_size - 1)//batch_size}"
        )

        for filepath in batch:
            if fix_file(filepath):
                fixed_count += 1

    # Get final error count
    result = subprocess.run(
        ["flake8", "--config=.flake8.minimal", "--count", "--quiet", "."],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        total_errors_after = (
            int(result.stdout.strip()) if result.stdout.strip() else 0
        )

    print(f"\n{'='*60}")
    print(f"Fixed {fixed_count} files")
    print(f"Errors before: {total_errors_before}")
    print(f"Errors after: {total_errors_after}")
    print(f"Errors fixed: {total_errors_before - total_errors_after}")


if __name__ == "__main__":
    main()
