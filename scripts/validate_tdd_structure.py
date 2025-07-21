#!/usr/bin/env python3
"""
TDD Structure Validation Script

This script validates that the codebase follows TDD principles from CLAUDE.MD:
1. Every production module has corresponding tests
2. Test coverage is comprehensive
3. No orphaned test files
4. Proper test organization
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set


class TDDStructureValidator:
    """Validates TDD structure compliance."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.production_modules: Set[str] = set()
        self.test_modules: Set[str] = set()
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def find_production_modules(self) -> Set[str]:
        """Find all production Python modules."""
        production_dirs = [
            "agents",
            "api",
            "auth",
            "coalitions",
            "database",
            "inference",
            "knowledge_graph",
            "observability",
            "world",
        ]

        modules = set()
        for dir_name in production_dirs:
            dir_path = self.project_root / dir_name
            if dir_path.exists():
                for py_file in dir_path.rglob("*.py"):
                    if not py_file.name.startswith("__"):
                        # Convert to module path
                        rel_path = py_file.relative_to(self.project_root)
                        module_path = str(rel_path.with_suffix("")).replace("/", ".")
                        modules.add(module_path)

        return modules

    def find_test_modules(self) -> Set[str]:
        """Find all test modules."""
        test_dir = self.project_root / "tests"
        modules = set()

        if test_dir.exists():
            for py_file in test_dir.rglob("test_*.py"):
                rel_path = py_file.relative_to(test_dir)
                module_path = str(rel_path.with_suffix("")).replace("/", ".")
                modules.add(module_path)

            for py_file in test_dir.rglob("*_test.py"):
                rel_path = py_file.relative_to(test_dir)
                module_path = str(rel_path.with_suffix("")).replace("/", ".")
                modules.add(module_path)

        return modules

    def check_test_coverage_existence(self) -> List[str]:
        """Check if all production modules have corresponding tests."""
        missing_tests = []

        for prod_module in self.production_modules:
            # Convert production module to expected test module name
            module_parts = prod_module.split(".")

            # Look for various test naming patterns
            possible_test_names = [
                f"test_{module_parts[-1]}",  # test_module_name.py
                f"unit.test_{module_parts[-1]}",  # tests/unit/test_module_name.py
                f"integration.test_{module_parts[-1]}",  # tests/integration/test_module_name.py
                f"{module_parts[0]}.test_{module_parts[-1]}",  # tests/agents/test_module_name.py
            ]

            # Also check for module-specific test directories
            if len(module_parts) > 1:
                possible_test_names.extend(
                    [
                        f"{module_parts[0]}.{module_parts[1]}.test_{module_parts[-1]}",
                        f"unit.test_{'.'.join(module_parts)}".replace(".", "_"),
                    ]
                )

            has_test = False
            for test_name in possible_test_names:
                if any(test_name in test_mod for test_mod in self.test_modules):
                    has_test = True
                    break

            if not has_test:
                missing_tests.append(prod_module)

        return missing_tests

    def check_orphaned_tests(self) -> List[str]:
        """Check for test files that don't correspond to production modules."""
        orphaned = []

        for test_module in self.test_modules:
            # Extract the module being tested
            if test_module.startswith("test_"):
                tested_module = test_module[5:]  # Remove "test_" prefix
            elif test_module.endswith("_test"):
                tested_module = test_module[:-5]  # Remove "_test" suffix
            else:
                continue

            # Check if the tested module exists in production
            module_exists = False
            for prod_module in self.production_modules:
                if tested_module in prod_module or prod_module.endswith(tested_module):
                    module_exists = True
                    break

            if not module_exists:
                orphaned.append(test_module)

        return orphaned

    def check_test_structure(self) -> Dict[str, List[str]]:
        """Check test directory structure follows TDD best practices."""
        structure_issues: Dict[str, List[str]] = {
            "missing_unit_tests": [],
            "missing_integration_tests": [],
            "poor_organization": [],
        }

        test_dir = self.project_root / "tests"
        if not test_dir.exists():
            structure_issues["poor_organization"].append("No tests directory found")
            return structure_issues

        # Check for proper test categorization
        required_dirs = ["unit", "integration", "e2e"]
        for req_dir in required_dirs:
            if not (test_dir / req_dir).exists():
                structure_issues["poor_organization"].append(
                    f"Missing {req_dir} test directory"
                )

        return structure_issues

    def analyze_test_quality(self) -> Dict[str, List[str]]:
        """Analyze test quality metrics."""
        quality_issues: Dict[str, List[str]] = {
            "empty_tests": [],
            "no_assertions": [],
            "poor_naming": [],
        }

        test_dir = self.project_root / "tests"
        if not test_dir.exists():
            return quality_issues

        for test_file in test_dir.rglob("test_*.py"):
            try:
                with open(test_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Parse AST to analyze test functions
                tree = ast.parse(content)

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith(
                        "test_"
                    ):
                        # Check if test has assertions
                        has_assertion = False
                        for child in ast.walk(node):
                            if isinstance(child, ast.Assert) or (
                                isinstance(child, ast.Call)
                                and isinstance(child.func, ast.Attribute)
                                and child.func.attr.startswith("assert")
                            ):
                                has_assertion = True
                                break

                        if not has_assertion:
                            quality_issues["no_assertions"].append(
                                f"{test_file}::{node.name}"
                            )

                        # Check naming convention
                        if len(node.name) < 10:  # Very short test names
                            quality_issues["poor_naming"].append(
                                f"{test_file}::{node.name}"
                            )

            except Exception as e:
                self.warnings.append(f"Could not analyze {test_file}: {e}")

        return quality_issues

    def validate(self) -> bool:
        """Run complete TDD structure validation."""
        print("🔍 Starting TDD structure validation...")

        # Find modules
        self.production_modules = self.find_production_modules()
        self.test_modules = self.find_test_modules()

        print(f"Found {len(self.production_modules)} production modules")
        print(f"Found {len(self.test_modules)} test modules")

        # Check test coverage existence
        missing_tests = self.check_test_coverage_existence()
        if missing_tests:
            self.errors.extend(
                [
                    f"Missing tests for production module: {module}"
                    for module in missing_tests
                ]
            )

        # Check for orphaned tests
        orphaned_tests = self.check_orphaned_tests()
        if orphaned_tests:
            self.warnings.extend(
                [
                    f"Orphaned test (no corresponding production module): {test}"
                    for test in orphaned_tests
                ]
            )

        # Check test structure
        structure_issues = self.check_test_structure()
        for category, issues in structure_issues.items():
            if issues:
                self.warnings.extend(
                    [f"Structure issue ({category}): {issue}" for issue in issues]
                )

        # Analyze test quality
        quality_issues = self.analyze_test_quality()
        for category, issues in quality_issues.items():
            if issues and category != "poor_naming":  # Treat poor naming as warning
                self.errors.extend(
                    [f"Quality issue ({category}): {issue}" for issue in issues]
                )
            elif issues:
                self.warnings.extend(
                    [f"Quality issue ({category}): {issue}" for issue in issues]
                )

        return len(self.errors) == 0

    def report(self) -> None:
        """Generate validation report."""
        print("\n" + "=" * 50)
        print("TDD STRUCTURE VALIDATION REPORT")
        print("=" * 50)

        if not self.errors and not self.warnings:
            print("✅ TDD structure validation PASSED")
            print("All production modules have corresponding tests")
            print("Test organization follows TDD best practices")
            return

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"   • {error}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")

        print("\n📊 SUMMARY:")
        print(f"   Production modules: {len(self.production_modules)}")
        print(f"   Test modules: {len(self.test_modules)}")
        print(f"   Errors: {len(self.errors)}")
        print(f"   Warnings: {len(self.warnings)}")

        if self.errors:
            print("\n❌ TDD structure validation FAILED")
            print("Fix all errors to ensure TDD compliance")
        else:
            print("\n✅ TDD structure validation PASSED (with warnings)")


def main():
    """Main entry point."""
    validator = TDDStructureValidator()

    try:
        success = validator.validate()
        validator.report()

        if not success:
            sys.exit(1)

    except Exception as e:
        print(f"❌ TDD structure validation failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
