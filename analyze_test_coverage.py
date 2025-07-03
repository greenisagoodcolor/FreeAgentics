#!/usr/bin/env python3
"""Analyze test coverage for core modules."""

import os
from collections import defaultdict
from pathlib import Path


def find_test_files(project_root):
    """Find all test files in the project."""
    test_files = set()
    test_patterns = ["test_", "_test.py", "tests.py"]

    for root, dirs, files in os.walk(project_root):
        if "venv" in root or "__pycache__" in root:
            continue
        for file in files:
            if any(pattern in file for pattern in test_patterns) and file.endswith(".py"):
                # Extract the base name being tested
                if file.startswith("test_"):
                    base_name = file[5:-3]  # Remove 'test_' prefix and '.py'
                elif file.endswith("_test.py"):
                    base_name = file[:-8]  # Remove '_test.py' suffix
                else:
                    base_name = file[:-3]  # Remove '.py' suffix
                test_files.add(base_name)

                # Also check for specific module tests
                full_path = os.path.join(root, file)
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    # Look for imports that might indicate what's being tested
                    if "from " in content or "import " in content:
                        test_files.add(file[:-3])

    return test_files


def find_source_modules(project_root, module_dirs):
    """Find all source modules in specified directories."""
    modules = defaultdict(list)

    for module_dir in module_dirs:
        dir_path = os.path.join(project_root, module_dir)
        if not os.path.exists(dir_path):
            continue

        for root, dirs, files in os.walk(dir_path):
            if "__pycache__" in root or "test" in root:
                continue
            for file in files:
                if file.endswith(".py") and not file.startswith("test_"):
                    relative_path = os.path.relpath(os.path.join(root, file), project_root)
                    module_name = file[:-3]  # Remove .py extension
                    modules[module_dir].append(
                        {
                            "path": relative_path,
                            "name": module_name,
                            "full_module": relative_path[:-3].replace("/", "."),
                        }
                    )

    return modules


def analyze_coverage(project_root):
    """Analyze test coverage for core modules."""
    # Core module directories
    module_dirs = ["api", "agents", "infrastructure", "inference", "coalitions"]

    # Find all test files
    test_files = find_test_files(project_root)

    # Find all source modules
    source_modules = find_source_modules(project_root, module_dirs)

    # Analyze coverage
    results = defaultdict(lambda: {"total": 0, "tested": 0, "missing": []})

    for module_dir, modules in source_modules.items():
        for module in modules:
            results[module_dir]["total"] += 1

            # Check if module has a test
            module_name = module["name"]
            has_test = False

            # Check various test naming conventions
            if module_name in test_files:
                has_test = True
            elif f"{module_name}_test" in test_files:
                has_test = True
            elif module_name == "__init__":
                # Check for module-level tests
                parent_dir = os.path.basename(os.path.dirname(module["path"]))
                if parent_dir in test_files or f"test_{parent_dir}" in test_files:
                    has_test = True

            if has_test:
                results[module_dir]["tested"] += 1
            else:
                results[module_dir]["missing"].append(module["path"])

    return results


def main():
    """Main function."""
    project_root = "/Users/matthewmoroney/builds/FreeAgentics"
    results = analyze_coverage(project_root)

    print("=" * 80)
    print("TEST COVERAGE ANALYSIS FOR CORE MODULES")
    print("=" * 80)
    print()

    total_modules = 0
    total_tested = 0

    for module_dir in ["api", "agents", "infrastructure", "inference", "coalitions"]:
        if module_dir in results:
            data = results[module_dir]
            coverage = (data["tested"] / data["total"] * 100) if data["total"] > 0 else 0

            print(f"\n{module_dir.upper()} MODULES")
            print("-" * 40)
            print(f"Total modules: {data['total']}")
            print(f"Tested modules: {data['tested']}")
            print(f"Coverage: {coverage:.1f}%")

            total_modules += data["total"]
            total_tested += data["tested"]

            if data["missing"]:
                print(f"\nModules without tests ({len(data['missing'])}):")
                for module in sorted(data["missing"]):
                    print(f"  - {module}")

    # Overall summary
    overall_coverage = (total_tested / total_modules * 100) if total_modules > 0 else 0
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total modules across all core areas: {total_modules}")
    print(f"Total modules with tests: {total_tested}")
    print(f"Overall test coverage: {overall_coverage:.1f}%")
    print(f"Modules missing tests: {total_modules - total_tested}")


if __name__ == "__main__":
    main()
