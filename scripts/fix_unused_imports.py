#!/usr/bin/env python3
"""
Fix unused imports in Python files systematically.

This script removes unused imports while preserving:
- Type annotation imports
- __all__ exports
- Imports used in string literals (for dynamic imports)
"""

import ast
import subprocess
from pathlib import Path
from typing import List, Set, Tuple


def get_used_names(tree: ast.AST) -> Set[str]:
    """Extract all names used in the code."""
    used_names = set()

    class NameVisitor(ast.NodeVisitor):
        def visit_Name(self, node):
            used_names.add(node.id)
            self.generic_visit(node)

        def visit_Attribute(self, node):
            # For x.y.z, we want to capture 'x'
            while isinstance(node, ast.Attribute):
                node = node.value
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            self.generic_visit(node)

        def visit_Str(self, node):
            # Check string literals for dynamic imports
            if "." in node.s:
                parts = node.s.split(".")
                used_names.add(parts[0])
            self.generic_visit(node)

        def visit_Constant(self, node):
            # Python 3.8+ uses Constant instead of Str
            if isinstance(node.value, str) and "." in node.value:
                parts = node.value.split(".")
                used_names.add(parts[0])
            self.generic_visit(node)

    visitor = NameVisitor()
    visitor.visit(tree)
    return used_names


def get_imported_names(tree: ast.AST) -> List[Tuple[str, ast.Import | ast.ImportFrom, str]]:
    """Extract all imported names with their import statements."""
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports.append((name, node, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.names[0].name == "*":
                # Skip star imports
                continue
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name
                imports.append((name, node, alias.name))

    return imports


def get_type_annotation_names(tree: ast.AST) -> Set[str]:
    """Extract names used in type annotations."""
    annotation_names = set()

    class AnnotationVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            # Check return type annotation
            if node.returns:
                self._extract_annotation_names(node.returns)
            # Check argument annotations
            for arg in node.args.args:
                if arg.annotation:
                    self._extract_annotation_names(arg.annotation)
            self.generic_visit(node)

        def visit_AnnAssign(self, node):
            if node.annotation:
                self._extract_annotation_names(node.annotation)
            self.generic_visit(node)

        def _extract_annotation_names(self, annotation):
            if isinstance(annotation, ast.Name):
                annotation_names.add(annotation.id)
            elif isinstance(annotation, ast.Attribute):
                # For x.y.z, we want 'x'
                node = annotation
                while isinstance(node, ast.Attribute):
                    node = node.value
                if isinstance(node, ast.Name):
                    annotation_names.add(node.id)
            elif isinstance(annotation, (ast.Subscript, ast.List, ast.Tuple)):
                # Recursively handle complex annotations like List[str],
                # Tuple[int, ...]
                for node in ast.walk(annotation):
                    if isinstance(node, ast.Name):
                        annotation_names.add(node.id)
            elif hasattr(annotation, "elts"):
                # Handle Union types
                for elt in annotation.elts:
                    self._extract_annotation_names(elt)

    visitor = AnnotationVisitor()
    visitor.visit(tree)
    return annotation_names


def is_import_used(
    name: str,
    used_names: Set[str],
    annotation_names: Set[str],
    exports: Set[str],
    is_test_file: bool,
) -> bool:
    """Check if an import is used anywhere in the file."""
    # Always keep certain imports
    if name in {
        "TYPE_CHECKING",
        "annotations",
        "absolute_import",
        "print_function",
        "division",
        "unicode_literals",
    }:
        return True

    # Keep pytest fixtures in test files
    if is_test_file and name in {
        "pytest",
        "unittest",
        "mock",
        "Mock",
        "patch",
        "AsyncMock",
        "MagicMock",
    }:
        return True

    # Check if used in code, annotations, or exports
    return name in used_names or name in annotation_names or name in exports


def get_exports(tree: ast.AST) -> Set[str]:
    """Get names that are exported via __all__."""
    exports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    if isinstance(node.value, ast.List):
                        for elt in node.value.elts:
                            if isinstance(elt, (ast.Str, ast.Constant)):
                                value = elt.s if hasattr(elt, "s") else elt.value
                                if isinstance(value, str):
                                    exports.add(value)

    return exports


def fix_file(filepath: Path) -> int:
    """Fix unused imports in a single file. Returns number of imports removed."""
    try:
        content = filepath.read_text()
        tree = ast.parse(content)
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return 0

    # Get all the necessary information
    used_names = get_used_names(tree)
    annotation_names = get_type_annotation_names(tree)
    exports = get_exports(tree)
    imports = get_imported_names(tree)
    is_test_file = "test" in filepath.name

    # Track which imports to remove
    unused_imports = []
    for name, import_node, original_name in imports:
        if not is_import_used(name, used_names, annotation_names, exports, is_test_file):
            unused_imports.append((name, import_node, original_name))

    if not unused_imports:
        return 0

    # Create new content without unused imports
    lines = content.splitlines()
    new_lines = []
    removed_count = 0

    for i, line in enumerate(lines):
        should_keep = True

        # Check if this line contains an unused import
        for name, import_node, original_name in unused_imports:
            if isinstance(import_node, ast.Import):
                if f"import {original_name}" in line:
                    if original_name == name:  # No alias
                        should_keep = False
                    elif f"as {name}" in line:  # With alias
                        should_keep = False
            elif isinstance(import_node, ast.ImportFrom):
                if (
                    f"from {
                        import_node.module}"
                    in line
                    and original_name in line
                ):
                    # Check if this is the only import on the line
                    if line.count(",") == 0:
                        should_keep = False
                    else:
                        # Remove just this import from the line
                        if f", {original_name}" in line:
                            line = line.replace(f", {original_name}", "")
                        elif f"{original_name}, " in line:
                            line = line.replace(f"{original_name}, ", "")
                        elif f"{original_name} as {name}, " in line:
                            line = line.replace(f"{original_name} as {name}, ", "")
                        elif f", {original_name} as {name}" in line:
                            line = line.replace(f", {original_name} as {name}", "")

        if should_keep:
            new_lines.append(line)
        else:
            removed_count += 1

    # Write back the file
    if removed_count > 0:
        filepath.write_text("\n".join(new_lines) + "\n")
        print(f"Fixed {filepath}: removed {removed_count} unused imports")

    return removed_count


def main():
    """Main function to fix unused imports in the project."""
    # Get all Python files
    project_root = Path(__file__).parent.parent
    python_files = []

    for pattern in [
        "agents/**/*.py",
        "api/**/*.py",
        "world/**/*.py",
        "inference/**/*.py",
        "coalitions/**/*.py",
        "tests/**/*.py",
    ]:
        python_files.extend(project_root.glob(pattern))

    # Sort by directory to process related files together
    python_files.sort()

    total_removed = 0
    files_fixed = 0

    for filepath in python_files:
        if "__pycache__" in str(filepath):
            continue

        removed = fix_file(filepath)
        if removed > 0:
            total_removed += removed
            files_fixed += 1

    print(f"\nSummary: Fixed {files_fixed} files, removed {total_removed} unused imports")

    # Run flake8 to check remaining F401 issues
    print("\nRunning flake8 to check remaining unused imports...")
    result = subprocess.run(
        ["flake8", "--select=F401", "--exclude=venv,node_modules,.git", "."],
        capture_output=True,
        text=True,
    )

    remaining = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
    print(f"Remaining F401 issues: {remaining}")


if __name__ == "__main__":
    main()
