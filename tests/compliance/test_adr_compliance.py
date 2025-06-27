"""Compliance Tests for Architectural Decision Records (ADRs).

Expert Committee Mandate: Robert C. Martin, Kent Beck, Rich Hickey, Conor Heins
Validates architectural compliance as specified in ADR documents.

Key compliance areas:
- Dependency rules (ADR-003)
- Testing strategy (ADR-007)
- Directory structure (ADR-002)
- Migration patterns (ADR-001)
"""

import ast
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest


class TestProjectStructureCompliance:
    """Test compliance with ADR-002: Canonical Directory Structure."""

    def test_required_directories_exist(self):
        """Verify all required directories from ADR-002 exist."""
        required_dirs = [
            "agents/",
            "inference/",
            "coalitions/",
            "world/",
            "knowledge/",
            "infrastructure/",
            "api/",
            "web/",
            "tests/",
            "docs/",
            "config/",
        ]

        for dir_path in required_dirs:
            assert os.path.exists(dir_path), f"Required directory missing: {dir_path}"

    def test_agents_module_structure(self):
        """Verify agents module follows ADR-002 structure."""
        agent_dirs = [
            "agents/base/",
            "agents/explorer/",
            "agents/scholar/",
            "agents/guardian/",
            "agents/merchant/",
            "agents/templates/",
            "agents/testing/",
            "agents/core/",
            "agents/active_inference/",
        ]

        for dir_path in agent_dirs:
            assert os.path.exists(dir_path), f"Agent directory missing: {dir_path}"

    def test_inference_module_structure(self):
        """Verify inference module follows ADR-002 structure."""
        inference_dirs = [
            "inference/engine/",
            "inference/gnn/",
            "inference/llm/",
            "inference/algorithms/",
        ]

        for dir_path in inference_dirs:
            assert os.path.exists(dir_path), f"Inference directory missing: {dir_path}"

    def test_test_directory_structure(self):
        """Verify comprehensive testing structure per ADR-007."""
        test_dirs = [
            "tests/unit/",
            "tests/integration/",
            "tests/property/",  # ADR-007 mandate
            "tests/behavior/",  # ADR-007 mandate
            "tests/security/",
            "tests/chaos/",  # ADR-007 mandate
            "tests/contract/",
            "tests/compliance/",
            "tests/features/",
        ]

        for dir_path in test_dirs:
            assert os.path.exists(dir_path), f"Test directory missing: {dir_path}"

    def test_documentation_structure(self):
        """Verify documentation follows ADR structure."""
        doc_dirs = [
            "docs/adr/",
            "docs/api/",
            "docs/architecture/",
            "docs/guides/",
            "docs/tutorials/",
        ]

        for dir_path in doc_dirs:
            assert os.path.exists(dir_path), f"Documentation directory missing: {dir_path}"


class TestDependencyCompliance:
    """Test compliance with ADR-003: Dependency Rules."""

    def test_circular_dependency_prohibition(self):
        """Verify no circular dependencies exist between modules."""
        modules = self._get_python_modules()
        dependency_graph = self._build_dependency_graph(modules)
        cycles = self._detect_cycles(dependency_graph)

        assert len(cycles) == 0, f"Circular dependencies detected: {cycles}"

    def test_layer_dependency_rules(self):
        """Test that layers only depend on lower layers (ADR-003)."""
        violations = []

        # Infrastructure should not depend on business logic
        infrastructure_files = self._get_files_in_module("infrastructure/")
        for file_path in infrastructure_files:
            imports = self._get_imports(file_path)
            for imp in imports:
                if any(layer in imp for layer in ["agents.", "coalitions.", "inference.engine"]):
                    violations.append(f"{file_path} imports {imp}")

        # Core agents should not depend on specific agent types
        base_agent_files = self._get_files_in_module("agents/base/")
        for file_path in base_agent_files:
            imports = self._get_imports(file_path)
            for imp in imports:
                if any(
                    agent_type in imp
                    for agent_type in ["explorer.", "scholar.", "guardian.", "merchant."]
                ):
                    violations.append(f"{file_path} imports specific agent: {imp}")

        assert len(violations) == 0, f"Layer dependency violations: {violations}"

    def test_external_dependency_compliance(self):
        """Test that external dependencies are properly managed."""
        # Check requirements files exist and are properly structured
        requirements_files = [
            "requirements.txt",
            "requirements-dev.txt",
            "web/package.json",
        ]

        for req_file in requirements_files:
            if os.path.exists(req_file):
                self._validate_requirements_file(req_file)

    def _get_python_modules(self) -> List[str]:
        """Get all Python modules in the project."""
        modules = []
        for root, dirs, files in os.walk("."):
            if any(exclude in root for exclude in [".git", "__pycache__", ".venv", "venv"]):
                continue
            for file in files:
                if file.endswith(".py"):
                    modules.append(os.path.join(root, file))
        return modules

    def _get_files_in_module(self, module_path: str) -> List[str]:
        """Get all Python files in a specific module."""
        files = []
        if os.path.exists(module_path):
            for root, dirs, filenames in os.walk(module_path):
                for filename in filenames:
                    if filename.endswith(".py"):
                        files.append(os.path.join(root, filename))
        return files

    def _get_imports(self, file_path: str) -> List[str]:
        """Extract import statements from a Python file."""
        imports = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
        except (SyntaxError, UnicodeDecodeError):
            pass  # Skip files that can't be parsed
        return imports

    def _build_dependency_graph(self, modules: List[str]) -> Dict[str, Set[str]]:
        """Build a dependency graph between modules."""
        graph = {}
        for module in modules:
            module_name = self._get_module_name(module)
            imports = self._get_imports(module)
            graph[module_name] = set(
                self._get_module_name(imp) for imp in imports if self._is_internal_import(imp)
            )
        return graph

    def _get_module_name(self, file_path: str) -> str:
        """Convert file path to module name."""
        return file_path.replace("/", ".").replace("\\", ".").replace(".py", "")

    def _is_internal_import(self, import_name: str) -> bool:
        """Check if an import is internal to the project."""
        internal_modules = [
            "agents",
            "inference",
            "coalitions",
            "world",
            "knowledge",
            "infrastructure",
            "api",
        ]
        return any(import_name.startswith(module) for module in internal_modules)

    def _detect_cycles(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Detect circular dependencies in the graph."""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node, path):
            if node in rec_stack:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                dfs(neighbor, path.copy())

            rec_stack.remove(node)

        for node in graph:
            if node not in visited:
                dfs(node, [])

        return cycles

    def _validate_requirements_file(self, file_path: str):
        """Validate that requirements file is properly structured."""
        if file_path.endswith("package.json"):
            with open(file_path, "r") as f:
                data = json.load(f)
                assert "dependencies" in data or "devDependencies" in data
        else:
            with open(file_path, "r") as f:
                lines = f.readlines()
                # Basic validation - should have actual packages
                package_lines = [
                    line for line in lines if line.strip() and not line.startswith("#")
                ]
                assert len(package_lines) > 0, f"Requirements file {file_path} appears empty"


class TestTestingStrategyCompliance:
    """Test compliance with ADR-007: Testing Strategy."""

    def test_property_based_testing_implemented(self):
        """Verify property-based testing is implemented (ADR-007 mandate)."""
        property_test_files = list(Path("tests/property").glob("*.py"))
        assert len(property_test_files) > 0, "Property-based tests missing (ADR-007 violation)"

        # Verify hypothesis usage
        has_hypothesis = False
        for test_file in property_test_files:
            content = test_file.read_text()
            if "hypothesis" in content and "@given" in content:
                has_hypothesis = True
                break

        assert has_hypothesis, "Hypothesis library not used in property tests (ADR-007 violation)"

    def test_behavior_driven_testing_implemented(self):
        """Verify BDD testing is implemented (ADR-007 mandate)."""
        behavior_test_files = list(Path("tests/behavior").glob("*.py"))
        feature_files = list(Path("tests/features").glob("*.feature"))

        assert len(behavior_test_files) > 0, "BDD tests missing (ADR-007 violation)"
        assert len(feature_files) > 0, "Gherkin feature files missing (ADR-007 violation)"

        # Verify pytest-bdd usage
        has_pytest_bdd = False
        for test_file in behavior_test_files:
            content = test_file.read_text()
            if "pytest_bdd" in content or "scenarios" in content:
                has_pytest_bdd = True
                break

        assert has_pytest_bdd, "pytest-bdd not used in behavior tests (ADR-007 violation)"

    def test_chaos_engineering_implemented(self):
        """Verify chaos engineering is implemented (ADR-007 mandate)."""
        chaos_test_files = list(Path("tests/chaos").glob("*.py"))
        assert len(chaos_test_files) > 0, "Chaos engineering tests missing (ADR-007 violation)"

        # Verify failure injection patterns
        has_failure_injection = False
        for test_file in chaos_test_files:
            content = test_file.read_text()
            if any(
                pattern in content
                for pattern in ["inject", "failure", "timeout", "exception", "chaos"]
            ):
                has_failure_injection = True
                break

        assert has_failure_injection, "Failure injection not implemented (ADR-007 violation)"

    def test_security_testing_implemented(self):
        """Verify security testing is implemented."""
        security_test_files = list(Path("tests/security").glob("*.py"))
        assert len(security_test_files) > 0, "Security tests missing"

        # Verify OWASP coverage
        has_owasp_tests = False
        for test_file in security_test_files:
            content = test_file.read_text()
            if any(
                pattern in content
                for pattern in ["sql injection", "xss", "csrf", "authentication", "owasp"]
            ):
                has_owasp_tests = True
                break

        assert has_owasp_tests, "OWASP security tests not implemented"

    def test_contract_testing_implemented(self):
        """Verify API contract testing is implemented."""
        contract_test_files = list(Path("tests/contract").glob("*.py"))
        assert len(contract_test_files) > 0, "Contract tests missing"

    def test_mathematical_invariant_coverage(self):
        """Verify mathematical invariants are tested (Active Inference requirement)."""
        property_files = list(Path("tests/property").glob("*.py"))
        invariant_patterns = [
            "free_energy",
            "belief",
            "entropy",
            "precision",
            "utility",
            "resource",
            "conservation",
        ]

        has_invariant_tests = False
        for test_file in property_files:
            content = test_file.read_text().lower()
            if any(pattern in content for pattern in invariant_patterns):
                has_invariant_tests = True
                break

        assert has_invariant_tests, "Mathematical invariant tests missing for Active Inference"


class TestNamingConventionCompliance:
    """Test compliance with naming conventions and standards."""

    def test_python_naming_conventions(self):
        """Test Python files follow PEP 8 naming conventions."""
        violations = []

        for root, dirs, files in os.walk("."):
            if any(
                exclude in root
                for exclude in [".git", "__pycache__", ".venv", "venv", "node_modules"]
            ):
                continue

            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)

                    # Check file naming (snake_case)
                    if not re.match(r"^[a-z_][a-z0-9_]*\.py$", file) and file != "__init__.py":
                        violations.append(f"File naming violation: {file_path}")

                    # Check for proper __init__.py files in packages
                    if os.path.isdir(os.path.dirname(file_path)):
                        init_file = os.path.join(os.path.dirname(file_path), "__init__.py")
                        if not os.path.exists(init_file) and any(
                            f.endswith(".py") for f in os.listdir(os.path.dirname(file_path))
                        ):
                            violations.append(
                                f"Missing __init__.py in package: {os.path.dirname(file_path)}"
                            )

        assert (
            len(violations) == 0
        ), f"Naming convention violations: {violations[:10]}"  # Show first 10

    def test_directory_naming_conventions(self):
        """Test directories follow naming conventions."""
        violations = []

        for root, dirs, files in os.walk("."):
            if any(
                exclude in root
                for exclude in [".git", "__pycache__", ".venv", "venv", "node_modules"]
            ):
                continue

            for dir_name in dirs:
                # Check directory naming (snake_case for Python modules)
                if not re.match(r"^[a-z_][a-z0-9_]*$", dir_name) and not re.match(
                    r"^[A-Z][a-zA-Z0-9]*$", dir_name
                ):
                    # Allow some exceptions
                    exceptions = [
                        ".git",
                        "__pycache__",
                        ".venv",
                        "venv",
                        "node_modules",
                        ".next",
                        ".pytest_cache",
                        ".mypy_cache",
                    ]
                    if dir_name not in exceptions:
                        violations.append(
                            f"Directory naming violation: {os.path.join(root, dir_name)}"
                        )

        assert len(violations) == 0, f"Directory naming violations: {violations[:10]}"


class TestConfigurationCompliance:
    """Test configuration and environment compliance."""

    def test_configuration_files_exist(self):
        """Test that required configuration files exist."""
        required_configs = [
            "config/.flake8",
            "commitlint.config.js",
            "web/package.json",
            "Makefile",
            ".gitignore",
        ]

        for config_file in required_configs:
            assert os.path.exists(config_file), f"Required config file missing: {config_file}"

    def test_gitignore_completeness(self):
        """Test that .gitignore covers common patterns."""
        with open(".gitignore", "r") as f:
            gitignore_content = f.read()

        required_patterns = [
            "__pycache__",
            "*.pyc",
            ".env",
            "node_modules",
            ".DS_Store",
            "venv",
            ".venv",
        ]

        for pattern in required_patterns:
            assert pattern in gitignore_content, f"Missing .gitignore pattern: {pattern}"

    def test_makefile_test_targets(self):
        """Test that Makefile includes all required test targets."""
        with open("Makefile", "r") as f:
            makefile_content = f.read()

        required_targets = [
            "test-property",
            "test-behavior",
            "test-security",
            "test-chaos",
            "test-contract",
            "test-compliance",
            "test-comprehensive",
        ]

        for target in required_targets:
            assert f"{target}:" in makefile_content, f"Missing Makefile target: {target}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
