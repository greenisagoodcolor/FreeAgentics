#!/usr/bin/env python3
"""Script to fix common import and unused variable issues."""

import ast
import os
import re


def find_unused_imports(file_path):
    """Find unused imports in a Python file."""
    try:
        with open(file_path, "r") as f:
            content = f.read()

        # Parse the AST
        tree = ast.parse(content)

        # Find all imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imports.append(alias.name)

        # Find all names used in the code
        used_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)

        # Find unused imports
        unused = []
        for imp in imports:
            if imp not in used_names:
                unused.append(imp)

        return unused

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return []


def remove_unused_imports(file_path):
    """Remove specific known unused imports."""
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()

        lines.copy()
        modified = False

        # Remove specific unused imports we know about
        for i, line in enumerate(lines):
            # Remove unused dataclass imports
            lines[i] = ""
            modified = True

            # Remove unused functools.partial imports
            lines[i] = ""
            modified = True

            # Remove unused pickle imports
            lines[i] = ""
            modified = True

        # Clean up empty lines from removed imports
        if modified:
            # Remove consecutive empty lines
            clean_lines = []
            prev_empty = False
            for line in lines:
                if line.strip() == "":
                    if not prev_empty:
                        clean_lines.append(line)
                    prev_empty = True
                else:
                    clean_lines.append(line)
                    prev_empty = False

            with open(file_path, "w") as f:
                f.writelines(clean_lines)

            print(f"Removed unused imports from {file_path}")
            return True

        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def fix_f_strings(file_path):
    """Fix f-strings missing placeholders."""
    try:
        with open(file_path, "r") as f:
            content = f.read()

        original_content = content

        # Find f-strings without placeholders and convert to regular strings
        # Pattern: "..." or '...' with no {
        content = re.sub(r'f(["\'])([^{]*?)\1', r"\1\2\1", content)

        if content != original_content:
            with open(file_path, "w") as f:
                f.write(content)
            print(f"Fixed f-strings in {file_path}")
            return True

        return False

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False


def main():
    """Fix import and other common issues."""
    base_dir = "/home/green/FreeAgentics"

    # Directories to exclude
    exclude_dirs = {"venv", "__pycache__", ".git", "node_modules", ".pytest_cache"}

    fixed_count = 0

    for root, dirs, files in os.walk(base_dir):
        # Remove excluded directories
        dirs[:] = [d for d in dirs if d not in exclude_dirs]

        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)

                # Fix imports
                if remove_unused_imports(file_path):
                    fixed_count += 1

                # Fix f-strings
                if fix_f_strings(file_path):
                    fixed_count += 1

    print(f"Fixed imports/f-strings in {fixed_count} file changes")


if __name__ == "__main__":
    main()
