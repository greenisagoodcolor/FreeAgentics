#!/usr/bin/env python3
"""
Batch script to systematically fix flake8 violations across the codebase.
"""

import argparse
import json
import os
import subprocess
from typing import Dict, List


def get_violation_stats(directory: str) -> Dict[str, int]:
    """Get violation statistics for a directory."""
    cmd = [
        "flake8",
        directory,
        "--exclude=__pycache__,*.pyc",
        "--format=%(code)s",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        violations = result.stdout.strip().split("\n") if result.stdout else []

        # Count by error code
        stats = {}
        for code in violations:
            if code:
                stats[code] = stats.get(code, 0) + 1

        return stats
    except Exception as e:
        print(f"Error getting stats for {directory}: {e}")
        return {}


def fix_imports_with_isort(directory: str) -> bool:
    """Fix import ordering using isort."""
    try:
        # Check if isort is available
        subprocess.run(["isort", "--version"], capture_output=True, check=True)

        # Run isort
        cmd = ["isort", directory, "--profile", "black", "--line-length", "79"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ Fixed import ordering in {directory}")
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"⚠ isort not available, skipping import fixes")

    return False


def fix_with_black(directory: str) -> bool:
    """Format code using black with flake8-compatible settings."""
    try:
        # Check if black is available
        subprocess.run(["black", "--version"], capture_output=True, check=True)

        # Run black with 79 char line length
        cmd = ["black", directory, "--line-length", "79", "--skip-string-normalization"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ Formatted code with black in {directory}")
            return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"⚠ black not available, skipping formatting")

    return False


def remove_unused_imports(file_path: str) -> int:
    """Remove unused imports from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Get unused imports
        cmd = ["flake8", file_path, "--select=F401", "--format=%(row)d:%(code)s:%(text)s"]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if not result.stdout:
            return 0

        # Parse unused imports
        unused_count = 0
        lines_to_remove = set()

        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split(":", 2)
                if len(parts) >= 3:
                    line_num = int(parts[0]) - 1
                    if "imported but unused" in parts[2]:
                        lines_to_remove.add(line_num)
                        unused_count += 1

        # Remove unused import lines
        if lines_to_remove:
            new_lines = []
            for i, line in enumerate(lines):
                if i not in lines_to_remove:
                    new_lines.append(line)

            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

        return unused_count
    except Exception as e:
        print(f"Error removing unused imports from {file_path}: {e}")
        return 0


def fix_line_length_violations(file_path: str) -> int:
    """Fix line length violations in a file."""
    fixed_count = 0

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = []
        for i, line in enumerate(lines):
            if len(line.rstrip()) > 79:
                # Try to break long lines intelligently
                stripped = line.strip()
                indent = len(line) - len(line.lstrip())

                # Long comments
                if stripped.startswith("#"):
                    # Break at word boundaries
                    words = stripped[1:].strip().split()
                    current_line = "#"
                    for word in words:
                        if len(current_line) + len(word) + 1 <= 78:
                            current_line += " " + word
                        else:
                            new_lines.append(" " * indent + current_line + "\n")
                            current_line = "#" + " " + word
                            fixed_count += 1
                    new_lines.append(" " * indent + current_line + "\n")

                # Long strings
                elif '"' in line or "'" in line:
                    # For now, keep as is - manual fix needed
                    new_lines.append(line)

                # Function calls with many parameters
                elif "(" in line and ")" in line and "," in line:
                    # Try to break at commas
                    prefix = line[: line.index("(") + 1]
                    suffix = line[line.rindex(")") :]
                    params = line[line.index("(") + 1 : line.rindex(")")].split(",")

                    if len(params) > 1:
                        new_lines.append(prefix + "\n")
                        for j, param in enumerate(params):
                            if j < len(params) - 1:
                                new_lines.append(" " * (indent + 4) + param.strip() + ",\n")
                            else:
                                new_lines.append(" " * (indent + 4) + param.strip() + "\n")
                        new_lines.append(" " * indent + suffix)
                        fixed_count += 1
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        if fixed_count > 0:
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

        return fixed_count
    except Exception as e:
        print(f"Error fixing line length in {file_path}: {e}")
        return 0


def process_directory(directory: str, fix_types: List[str]) -> Dict[str, int]:
    """Process a directory and fix violations."""
    print(f"\n📁 Processing {directory}...")

    # Get initial stats
    before_stats = get_violation_stats(directory)
    total_before = sum(before_stats.values())

    if total_before == 0:
        print(f"  ✓ No violations found!")
        return {"fixed": 0, "remaining": 0}

    print(f"  Found {total_before} violations")
    print(
        f"  Top violations: {dict(sorted(before_stats.items(), key=lambda x: x[1], reverse=True)[:5])}"
    )

    # Apply fixes based on options
    if "imports" in fix_types:
        fix_imports_with_isort(directory)

    if "format" in fix_types:
        fix_with_black(directory)

    if "unused" in fix_types or "length" in fix_types:
        # Process individual files
        py_files = []
        for root, _, files in os.walk(directory):
            if "__pycache__" not in root:
                for file in files:
                    if file.endswith(".py"):
                        py_files.append(os.path.join(root, file))

        print(f"  Processing {len(py_files)} Python files...")

        for file_path in py_files:
            if "unused" in fix_types:
                removed = remove_unused_imports(file_path)
                if removed > 0:
                    print(f"    Removed {removed} unused imports from {file_path}")

            if "length" in fix_types:
                fixed = fix_line_length_violations(file_path)
                if fixed > 0:
                    print(f"    Fixed {fixed} line length violations in {file_path}")

    # Get final stats
    after_stats = get_violation_stats(directory)
    total_after = sum(after_stats.values())

    fixed = total_before - total_after
    print(f"  ✓ Fixed {fixed} violations ({total_before} → {total_after})")

    return {
        "directory": directory,
        "before": total_before,
        "after": total_after,
        "fixed": fixed,
        "remaining": total_after,
        "stats": after_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch fix flake8 violations")
    parser.add_argument(
        "directories",
        nargs="*",
        default=["agents", "api", "database", "inference", "knowledge_graph"],
        help="Directories to process",
    )
    parser.add_argument("--fix-imports", action="store_true", help="Fix import ordering")
    parser.add_argument("--fix-format", action="store_true", help="Format with black")
    parser.add_argument("--fix-unused", action="store_true", help="Remove unused imports")
    parser.add_argument("--fix-length", action="store_true", help="Fix line length violations")
    parser.add_argument("--fix-all", action="store_true", help="Apply all fixes")
    parser.add_argument("--report", default="flake8_fix_report.json", help="Output report file")

    args = parser.parse_args()

    # Determine which fixes to apply
    fix_types = []
    if args.fix_all:
        fix_types = ["imports", "format", "unused", "length"]
    else:
        if args.fix_imports:
            fix_types.append("imports")
        if args.fix_format:
            fix_types.append("format")
        if args.fix_unused:
            fix_types.append("unused")
        if args.fix_length:
            fix_types.append("length")

    if not fix_types:
        # Default: fix unused imports and basic issues
        fix_types = ["unused"]

    print(f"🔧 Flake8 Batch Fixer")
    print(f"Directories: {args.directories}")
    print(f"Fix types: {fix_types}")

    # Process each directory
    results = []
    for directory in args.directories:
        if os.path.exists(directory):
            result = process_directory(directory, fix_types)
            results.append(result)

    # Summary
    print("\n📊 Summary:")
    total_fixed = sum(r["fixed"] for r in results)
    total_remaining = sum(r["remaining"] for r in results)

    print(f"Total violations fixed: {total_fixed}")
    print(f"Total violations remaining: {total_remaining}")

    # Save report
    with open(args.report, "w") as f:
        json.dump(
            {
                "results": results,
                "summary": {
                    "total_fixed": total_fixed,
                    "total_remaining": total_remaining,
                    "fix_types": fix_types,
                },
            },
            f,
            indent=2,
        )

    print(f"\n✓ Report saved to {args.report}")


if __name__ == "__main__":
    main()
