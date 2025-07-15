"""Custom pytest plugin for automatic test metrics collection."""

import os
import time
import uuid
from datetime import datetime
from typing import Dict, Optional

import pytest
from _pytest.config import Config
from _pytest.main import Session
from _pytest.nodes import Item
from _pytest.reports import TestReport

from .test_metrics_collector import TestMetricsCollector, TestStatus


class MetricsPlugin:
    """Pytest plugin for collecting test metrics."""

    def __init__(self, config: Config):
        self.config = config
        self.collector = TestMetricsCollector()
        self.session_start_time: Optional[float] = None
        self.test_run_id = (
            f"pytest_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )
        self.environment = os.environ.get("TEST_ENVIRONMENT", "development")

    def pytest_sessionstart(self, session: Session):
        """Called after the Session object has been created."""
        self.session_start_time = time.time()
        self.collector.start_test_suite(self.test_run_id, self.environment)

        # Print metrics info
        print(f"\\n=== Test Metrics Collection Started ===")
        print(f"Test Run ID: {self.test_run_id}")
        print(f"Environment: {self.environment}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def pytest_sessionfinish(self, session: Session, exitstatus: int):
        """Called after whole test run finished."""
        suite_metrics = self.collector.end_test_suite()

        # Print summary
        print(f"\\n=== Test Metrics Collection Finished ===")
        print(f"Total Tests: {suite_metrics.total_tests}")
        print(f"Passed: {suite_metrics.passed_tests}")
        print(f"Failed: {suite_metrics.failed_tests}")
        print(f"Skipped: {suite_metrics.skipped_tests}")
        print(f"Errors: {suite_metrics.error_tests}")
        print(f"Duration: {suite_metrics.total_duration:.2f}s")

        # Generate reports if enabled
        if self.config.getoption("--generate-reports", False):
            self.collector.generate_metrics_report()
            self.collector.export_metrics_json()
            print(f"Reports generated in tests/reporting/")

        # Show flaky tests if any
        flaky_tests = self.collector.get_flaky_tests()
        if flaky_tests:
            print(f"\\n⚠️  Flaky Tests Detected ({len(flaky_tests)}):")
            for test in flaky_tests[:5]:  # Show top 5
                print(f"  - {test['test_name']} ({test['flaky_percentage']:.1f}% flaky)")

        # Show slow tests if any
        slow_tests = self.collector.get_slow_tests()
        if slow_tests:
            print(f"\\n🐌 Slow Tests Detected ({len(slow_tests)}):")
            for test in slow_tests[:5]:  # Show top 5
                print(f"  - {test['test_name']} ({test['avg_duration']:.2f}s avg)")

    def pytest_runtest_setup(self, item: Item):
        """Called to perform the setup phase for a test item."""
        test_id = self._get_test_id(item)
        self.collector.start_test(test_id)

    def pytest_runtest_teardown(self, item: Item, nextitem: Optional[Item]):
        """Called to perform the teardown phase for a test item."""
        # Test teardown is handled in pytest_runtest_logreport

    def pytest_runtest_logreport(self, report: TestReport):
        """Called after a test report is created."""
        if report.when == "call":  # Only process the main test execution
            test_id = self._get_test_id_from_report(report)
            test_name = report.nodeid.split("::")[-1]
            test_file = report.nodeid.split("::")[0]

            # Map pytest outcomes to our status enum
            status_mapping = {
                "passed": TestStatus.PASSED,
                "failed": TestStatus.FAILED,
                "skipped": TestStatus.SKIPPED,
                "error": TestStatus.ERROR,
            }

            status = status_mapping.get(report.outcome, TestStatus.ERROR)

            # Extract error information
            error_message = None
            stack_trace = None

            if report.failed:
                if hasattr(report, "longrepr") and report.longrepr:
                    if hasattr(report.longrepr, "reprcrash"):
                        error_message = str(report.longrepr.reprcrash.message)
                    if hasattr(report.longrepr, "reprtraceback"):
                        stack_trace = str(report.longrepr.reprtraceback)
                    else:
                        error_message = str(report.longrepr)

            self.collector.end_test(
                test_id=test_id,
                test_name=test_name,
                test_file=test_file,
                status=status,
                error_message=error_message,
                stack_trace=stack_trace,
            )

    def _get_test_id(self, item: Item) -> str:
        """Generate a unique test ID from a test item."""
        return f"{item.nodeid}::{item.name}"

    def _get_test_id_from_report(self, report: TestReport) -> str:
        """Generate a unique test ID from a test report."""
        return f"{report.nodeid}::{report.nodeid.split('::')[-1]}"


def pytest_addoption(parser):
    """Add command line options for metrics collection."""
    group = parser.getgroup("metrics", "Test metrics collection options")
    group.addoption(
        "--generate-reports",
        action="store_true",
        default=False,
        help="Generate HTML and JSON reports after test run",
    )
    group.addoption(
        "--metrics-db",
        action="store",
        default="tests/reporting/test_metrics.db",
        help="Database path for storing test metrics",
    )
    group.addoption(
        "--environment",
        action="store",
        default="development",
        help="Test environment name for metrics tracking",
    )


def pytest_configure(config: Config):
    """Configure the plugin."""
    # Only enable if not explicitly disabled
    if not config.getoption("--no-metrics", False):
        plugin = MetricsPlugin(config)
        config.pluginmanager.register(plugin, "test_metrics")


def pytest_unconfigure(config: Config):
    """Cleanup when pytest is unconfigured."""
    plugin = config.pluginmanager.get_plugin("test_metrics")
    if plugin:
        config.pluginmanager.unregister(plugin)


# Add additional option for disabling metrics
def pytest_addoption(parser):
    """Add command line options for metrics collection."""
    group = parser.getgroup("metrics", "Test metrics collection options")
    group.addoption(
        "--generate-reports",
        action="store_true",
        default=False,
        help="Generate HTML and JSON reports after test run",
    )
    group.addoption(
        "--metrics-db",
        action="store",
        default="tests/reporting/test_metrics.db",
        help="Database path for storing test metrics",
    )
    group.addoption(
        "--environment",
        action="store",
        default="development",
        help="Test environment name for metrics tracking",
    )
    group.addoption(
        "--no-metrics", action="store_true", default=False, help="Disable test metrics collection"
    )


class CoverageIntegration:
    """Integration with coverage.py for enhanced reporting."""

    def __init__(self, metrics_collector: TestMetricsCollector):
        self.metrics_collector = metrics_collector

    def pytest_configure(self, config: Config):
        """Configure coverage integration."""
        # Check if coverage is available
        try:
            import coverage

            self.coverage = coverage.Coverage()
            self.coverage.start()
        except ImportError:
            self.coverage = None

    def pytest_sessionfinish(self, session: Session, exitstatus: int):
        """Finish coverage collection."""
        if self.coverage:
            self.coverage.stop()
            self.coverage.save()

            # Generate coverage report
            from .coverage_analyzer import CoverageAnalyzer

            analyzer = CoverageAnalyzer()
            analyzer.generate_coverage_report_html()
            analyzer.export_coverage_json()


# Hook for custom test result formatting
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom summary information to terminal output."""
    if hasattr(config, "pluginmanager"):
        plugin = config.pluginmanager.get_plugin("test_metrics")
        if plugin and hasattr(plugin, "collector"):
            # Get quick stats
            flaky_tests = plugin.collector.get_flaky_tests()
            slow_tests = plugin.collector.get_slow_tests()

            if flaky_tests or slow_tests:
                terminalreporter.write_sep("=", "Test Quality Summary")

                if flaky_tests:
                    terminalreporter.write_line(
                        f"⚠️  {len(flaky_tests)} flaky tests detected - "
                        f"see tests/reporting/metrics_report.html for details"
                    )

                if slow_tests:
                    terminalreporter.write_line(
                        f"🐌 {len(slow_tests)} slow tests detected - " f"consider optimization"
                    )

                terminalreporter.write_line(
                    f"📊 Full metrics report: tests/reporting/metrics_report.html"
                )


# Hook for custom markers
def pytest_configure(config):
    """Register custom markers for test categorization."""
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "flaky: mark test as potentially flaky")
    config.addinivalue_line("markers", "critical: mark test as critical for system functionality")
    config.addinivalue_line("markers", "performance: mark test as performance test")


def pytest_collection_modifyitems(config, items):
    """Modify collected test items based on markers."""
    for item in items:
        # Add automatic markers based on test characteristics
        if "slow" in item.nodeid.lower() or "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)

        if "flaky" in item.nodeid.lower() or "intermittent" in item.nodeid.lower():
            item.add_marker(pytest.mark.flaky)

        if "critical" in item.nodeid.lower() or "smoke" in item.nodeid.lower():
            item.add_marker(pytest.mark.critical)


class TestRetryPlugin:
    """Plugin for automatic test retry on failure."""

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries
        self.retry_counts: Dict[str, int] = {}

    def pytest_runtest_protocol(self, item, nextitem):
        """Handle test execution with retry logic."""
        test_id = item.nodeid

        if test_id not in self.retry_counts:
            self.retry_counts[test_id] = 0

        # Check if test has flaky marker
        if item.get_closest_marker("flaky"):
            max_retries = self.max_retries
        else:
            max_retries = 1

        for retry in range(max_retries):
            try:
                # Run the test
                return pytest.main.runtestprotocol(item, nextitem, log=True)
            except Exception:
                self.retry_counts[test_id] += 1
                if retry < max_retries - 1:
                    print(f"\\n⚠️  Test {test_id} failed, retrying ({retry + 1}/{max_retries})...")
                    continue
                else:
                    raise

        return False


# Integration with external reporting tools
def pytest_configure(config):
    """Configure integration with external reporting tools."""
    # Check for Allure reporter
    if config.getoption("--alluredir", None):
        try:
            import allure

            # Add custom properties to Allure reports
            allure.dynamic.label("framework", "pytest")
            allure.dynamic.label("suite", "freeagentics")
        except ImportError:
            pass

    # Check for JUnit XML output
    if config.getoption("--junitxml", None):
        # JUnit XML will be automatically generated by pytest
        pass
