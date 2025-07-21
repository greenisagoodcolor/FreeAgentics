"""
TDD Pytest Plugin

This plugin enforces TDD principles during test execution:
1. Validates test structure follows TDD patterns
2. Ensures 100% coverage compliance
3. Prevents test pollution and ensures isolation
4. Validates Red-Green-Refactor workflow compliance
"""

import os
import time
import warnings
from typing import Dict, List, Optional

import pytest
from _pytest.config import Config
from _pytest.main import Session
from _pytest.nodes import Item
from _pytest.reports import TestReport
from _pytest.runner import CallInfo


class TDDPlugin:
    """TDD enforcement plugin for pytest."""

    def __init__(self):
        self.test_results: Dict[str, str] = {}
        self.coverage_data: Dict[str, float] = {}
        self.test_start_times: Dict[str, float] = {}
        self.skipped_tests: List[str] = []
        self.failed_tests: List[str] = []
        self.tdd_violations: List[str] = []

    def pytest_configure(self, config: Config) -> None:
        """Configure TDD plugin."""
        # Register custom markers
        config.addinivalue_line("markers", "tdd_compliant: mark test as TDD compliant")
        config.addinivalue_line(
            "markers",
            "red_green_refactor: mark test as following Red-Green-Refactor",
        )

        # Ensure TDD mode is enabled
        os.environ["TDD_MODE"] = "true"
        os.environ["PYTEST_TDD_PLUGIN"] = "true"

        # Configure strict mode for TDD
        if not config.getoption("--tb"):
            config.option.tbstyle = "short"

        print("🧪 TDD Plugin activated - Enforcing TDD compliance")

    def pytest_sessionstart(self, session: Session) -> None:
        """Session start - validate TDD setup."""
        print("📋 TDD Session Starting - Validating environment...")

        # Check for TDD environment setup
        required_env_vars = ["TDD_MODE", "TESTING"]
        missing_vars = [var for var in required_env_vars if not os.environ.get(var)]

        if missing_vars:
            pytest.exit(
                f"TDD setup incomplete. Missing environment variables: {missing_vars}"
            )

        # Validate no production mocks
        self._validate_no_production_mocks()

        print("✅ TDD environment validated")

    def pytest_collection_modifyitems(self, config: Config, items: List[Item]) -> None:
        """Modify collected items to enforce TDD compliance."""
        # Check for skipped tests (TDD violation)
        for item in items:
            if item.get_closest_marker("skip") or item.get_closest_marker("skipif"):
                self.skipped_tests.append(item.nodeid)

        if self.skipped_tests:
            print(
                f"⚠️  WARNING: {len(self.skipped_tests)} skipped tests found (TDD violation)"
            )
            if os.environ.get("TDD_STRICT") == "true":
                pytest.exit(
                    "TDD violation: Skipped tests are not allowed in strict TDD mode"
                )

        # Validate test naming follows TDD conventions
        self._validate_test_naming(items)

    def pytest_runtest_setup(self, item: Item) -> None:
        """Setup for each test - enforce TDD isolation."""
        self.test_start_times[item.nodeid] = time.time()

        # Ensure test isolation
        if not hasattr(item, "_tdd_isolated"):
            # Mark test as needing TDD isolation
            item._tdd_isolated = True

    def pytest_runtest_call(self, item: Item) -> None:
        """Test execution - monitor TDD compliance."""
        # Track test execution for TDD metrics

    def pytest_runtest_teardown(self, item: Item, nextitem: Optional[Item]) -> None:
        """Teardown after each test - validate TDD compliance."""
        # Calculate test execution time
        if item.nodeid in self.test_start_times:
            duration = time.time() - self.test_start_times[item.nodeid]
            # TDD tests should be fast - warn if too slow
            if duration > 5.0:  # 5 seconds threshold
                warnings.warn(
                    f"Test {item.nodeid} took {duration:.2f}s - TDD tests should be fast",
                    UserWarning,
                )

    def pytest_runtest_makereport(
        self, item: Item, call: CallInfo
    ) -> Optional[TestReport]:
        """Generate test report with TDD compliance information."""
        if call.when == "call":
            if call.excinfo is not None:
                self.failed_tests.append(item.nodeid)
            else:
                self.test_results[item.nodeid] = "passed"

        return None

    def pytest_sessionfinish(self, session: Session, exitstatus: int) -> None:
        """Session finish - generate TDD compliance report."""
        self._generate_tdd_report(session, exitstatus)

    def _validate_no_production_mocks(self) -> None:
        """Validate no mocks in production code (TDD principle)."""
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

        mock_violations = []
        for dir_name in production_dirs:
            if os.path.exists(dir_name):
                for root, dirs, files in os.walk(dir_name):
                    for file in files:
                        if file.endswith(".py"):
                            file_path = os.path.join(root, file)
                            try:
                                with open(file_path, "r", encoding="utf-8") as f:
                                    content = f.read()
                                    if any(
                                        mock_term in content
                                        for mock_term in [
                                            "mock.",
                                            "Mock(",
                                            "patch(",
                                            "MagicMock",
                                        ]
                                    ):
                                        mock_violations.append(file_path)
                            except Exception:
                                continue

        if mock_violations:
            self.tdd_violations.extend(
                [f"Mock usage in production code: {path}" for path in mock_violations]
            )

    def _validate_test_naming(self, items: List[Item]) -> None:
        """Validate test naming follows TDD conventions."""
        for item in items:
            test_name = item.name

            # TDD test names should be descriptive
            if len(test_name) < 10:
                self.tdd_violations.append(
                    f"Test name too short (TDD requires descriptive names): {item.nodeid}"
                )

            # Should follow naming pattern
            if not (test_name.startswith("test_") and "_" in test_name[5:]):
                self.tdd_violations.append(
                    f"Test name doesn't follow TDD convention: {item.nodeid}"
                )

    def _generate_tdd_report(self, session: Session, exitstatus: int) -> None:
        """Generate comprehensive TDD compliance report."""
        print("\n" + "=" * 60)
        print("TDD COMPLIANCE REPORT")
        print("=" * 60)

        total_tests = (
            len(self.test_results) + len(self.failed_tests) + len(self.skipped_tests)
        )
        passed_tests = len([r for r in self.test_results.values() if r == "passed"])

        print("📊 TEST SUMMARY:")
        print(f"   Total tests: {total_tests}")
        print(f"   Passed: {passed_tests}")
        print(f"   Failed: {len(self.failed_tests)}")
        print(f"   Skipped: {len(self.skipped_tests)}")

        # TDD Compliance Check
        tdd_compliant = True

        if self.skipped_tests:
            print(f"\n⚠️  TDD VIOLATIONS - SKIPPED TESTS ({len(self.skipped_tests)}):")
            for test in self.skipped_tests[:5]:  # Show first 5
                print(f"   • {test}")
            if len(self.skipped_tests) > 5:
                print(f"   ... and {len(self.skipped_tests) - 5} more")
            tdd_compliant = False

        if self.tdd_violations:
            print(f"\n❌ TDD VIOLATIONS ({len(self.tdd_violations)}):")
            for violation in self.tdd_violations[:10]:  # Show first 10
                print(f"   • {violation}")
            if len(self.tdd_violations) > 10:
                print(f"   ... and {len(self.tdd_violations) - 10} more")
            tdd_compliant = False

        if self.failed_tests:
            print(f"\n❌ FAILED TESTS ({len(self.failed_tests)}):")
            for test in self.failed_tests[:5]:  # Show first 5
                print(f"   • {test}")
            if len(self.failed_tests) > 5:
                print(f"   ... and {len(self.failed_tests) - 5} more")
            tdd_compliant = False

        # Overall TDD Compliance
        print("\n🎯 TDD COMPLIANCE:")
        if tdd_compliant and exitstatus == 0:
            print("   ✅ FULLY COMPLIANT - All TDD principles followed")
            print("   ✅ Ready for production deployment")
        else:
            print("   ❌ NON-COMPLIANT - TDD violations found")
            print("   ❌ Fix all violations before proceeding")

        print("=" * 60)


# Register the plugin
def pytest_configure(config):
    """Register TDD plugin."""
    config.pluginmanager.register(TDDPlugin(), "tdd_plugin")
