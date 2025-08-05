"""
E2E Test Runner Implementation
==============================

Minimal test orchestration with proper error handling and health checking
following production-first principles.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .reporting import TestResultCollector, TestResult

logger = logging.getLogger(__name__)


class TestExecutionError(Exception):
    """Exception raised during test execution."""

    pass


class E2ETestRunner:
    """
    Production-grade test runner with health checking, parallel execution,
    and comprehensive error handling.
    """

    def __init__(
        self,
        max_workers: int = 4,
        timeout: int = 300,
        output_dir: Optional[Union[str, Path]] = None,
        fail_fast: bool = False,
    ):
        """
        Initialize E2E test runner.

        Args:
            max_workers: Maximum parallel test workers
            timeout: Default test timeout in seconds
            output_dir: Directory for test outputs
            fail_fast: Stop on first failure
        """
        self.max_workers = max_workers
        self.timeout = timeout
        self.fail_fast = fail_fast

        self.collector = TestResultCollector(output_dir)
        self.test_registry: Dict[str, Callable] = {}
        self._health_checks: List[Callable[[], bool]] = []
        self._setup_hooks: List[Callable] = []
        self._teardown_hooks: List[Callable] = []

        # Runtime state
        self._running = False
        self._cancelled = False
        self._executor: Optional[ThreadPoolExecutor] = None

        logger.info(f"Initialized E2ETestRunner (workers: {max_workers}, timeout: {timeout}s)")

    def add_health_check(self, check_func: Callable[[], bool], name: str = "") -> None:
        """
        Add health check function.

        Args:
            check_func: Function that returns True if healthy
            name: Optional name for the health check
        """

        def named_check():
            try:
                result = check_func()
                check_name = name or check_func.__name__
                logger.debug(f"Health check '{check_name}': {'PASS' if result else 'FAIL'}")
                return result
            except Exception as e:
                check_name = name or check_func.__name__
                logger.error(f"Health check '{check_name}' error: {e}")
                return False

        self._health_checks.append(named_check)
        logger.info(f"Added health check: {name or check_func.__name__}")

    def add_setup_hook(self, setup_func: Callable) -> None:
        """Add setup hook function."""
        self._setup_hooks.append(setup_func)
        logger.info(f"Added setup hook: {setup_func.__name__}")

    def add_teardown_hook(self, teardown_func: Callable) -> None:
        """Add teardown hook function."""
        self._teardown_hooks.append(teardown_func)
        logger.info(f"Added teardown hook: {teardown_func.__name__}")

    def register_test(self, test_func: Callable, name: Optional[str] = None) -> None:
        """
        Register a test function.

        Args:
            test_func: Test function to register
            name: Optional test name (uses function name if None)
        """
        test_name = name or test_func.__name__
        self.test_registry[test_name] = test_func
        logger.info(f"Registered test: {test_name}")

    def run_health_checks(self) -> bool:
        """
        Run all health checks.

        Returns:
            True if all checks pass, False otherwise
        """
        if not self._health_checks:
            logger.info("No health checks configured - assuming healthy")
            return True

        logger.info(f"Running {len(self._health_checks)} health checks...")

        failed_checks = []
        for i, check in enumerate(self._health_checks):
            try:
                if not check():
                    failed_checks.append(f"check_{i}")
            except Exception as e:
                logger.error(f"Health check {i} exception: {e}")
                failed_checks.append(f"check_{i}")

        if failed_checks:
            logger.error(f"Health checks failed: {', '.join(failed_checks)}")
            return False

        logger.info("All health checks passed")
        return True

    def _run_setup_hooks(self) -> bool:
        """Run all setup hooks."""
        logger.info(f"Running {len(self._setup_hooks)} setup hooks...")

        for hook in self._setup_hooks:
            try:
                hook()
                logger.debug(f"Setup hook '{hook.__name__}' completed")
            except Exception as e:
                logger.error(f"Setup hook '{hook.__name__}' failed: {e}")
                return False

        logger.info("All setup hooks completed")
        return True

    def _run_teardown_hooks(self) -> None:
        """Run all teardown hooks (best effort)."""
        logger.info(f"Running {len(self._teardown_hooks)} teardown hooks...")

        for hook in self._teardown_hooks:
            try:
                hook()
                logger.debug(f"Teardown hook '{hook.__name__}' completed")
            except Exception as e:
                logger.error(f"Teardown hook '{hook.__name__}' failed: {e}")
                # Continue with other teardown hooks

        logger.info("Teardown hooks completed")

    def _execute_test(self, test_name: str, test_func: Callable) -> TestResult:
        """
        Execute a single test with proper error handling and timing.

        Args:
            test_name: Name of the test
            test_func: Test function to execute

        Returns:
            Test result
        """
        start_time = time.time()

        try:
            logger.info(f"Executing test: {test_name}")

            # Execute test with timeout
            if asyncio.iscoroutinefunction(test_func):
                # Handle async test functions
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        asyncio.wait_for(test_func(), timeout=self.timeout)
                    )
                finally:
                    loop.close()
            else:
                # Handle sync test functions
                result = test_func()

            duration = time.time() - start_time

            # Determine test status based on result
            if result is False:
                logger.warning(f"Test {test_name} returned False")
                return TestResult(
                    test_name=test_name,
                    status="failed",
                    duration=duration,
                    error_message="Test returned False",
                )
            else:
                logger.info(f"Test {test_name} passed ({duration:.2f}s)")
                return TestResult(test_name=test_name, status="passed", duration=duration)

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            error_msg = f"Test timed out after {self.timeout}s"
            logger.error(f"Test {test_name} timed out")
            return TestResult(
                test_name=test_name, status="failed", duration=duration, error_message=error_msg
            )

        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            stack_trace = None

            # Get stack trace for debugging
            import traceback

            stack_trace = traceback.format_exc()

            logger.error(f"Test {test_name} failed with exception: {error_msg}")
            return TestResult(
                test_name=test_name,
                status="error",
                duration=duration,
                error_message=error_msg,
                stack_trace=stack_trace,
            )

    def run_tests(
        self, test_names: Optional[List[str]] = None, suite_name: str = "e2e_tests"
    ) -> Dict[str, Any]:
        """
        Run specified tests or all registered tests.

        Args:
            test_names: List of test names to run (None = run all)
            suite_name: Name for the test suite

        Returns:
            Test execution summary
        """
        if self._running:
            raise TestExecutionError("Runner is already executing tests")

        self._running = True
        self._cancelled = False

        try:
            # Determine which tests to run
            if test_names is None:
                tests_to_run = list(self.test_registry.items())
            else:
                tests_to_run = []
                for name in test_names:
                    if name in self.test_registry:
                        tests_to_run.append((name, self.test_registry[name]))
                    else:
                        logger.warning(f"Test '{name}' not found in registry")

            if not tests_to_run:
                logger.warning("No tests to run")
                return {"summary": self.collector.get_summary()}

            logger.info(f"Running {len(tests_to_run)} tests...")

            # Run health checks
            if not self.run_health_checks():
                raise TestExecutionError("Health checks failed - aborting test run")

            # Run setup hooks
            if not self._run_setup_hooks():
                raise TestExecutionError("Setup hooks failed - aborting test run")

            # Start test suite collection
            self.collector.start_suite(
                suite_name,
                {
                    "total_tests": len(tests_to_run),
                    "max_workers": self.max_workers,
                    "timeout": self.timeout,
                },
            )

            # Execute tests
            if self.max_workers == 1:
                # Sequential execution
                self._run_tests_sequential(tests_to_run)
            else:
                # Parallel execution
                self._run_tests_parallel(tests_to_run)

            # Finish suite
            self.collector.finish_suite()

            # Print summary
            self.collector.print_summary()

            return {"summary": self.collector.get_summary()}

        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            raise TestExecutionError(f"Test execution failed: {e}")

        finally:
            self._run_teardown_hooks()
            self._running = False
            if self._executor:
                self._executor.shutdown(wait=True)
                self._executor = None

    def _run_tests_sequential(self, tests_to_run: List[tuple]) -> None:
        """Run tests sequentially."""
        logger.info("Running tests sequentially")

        for test_name, test_func in tests_to_run:
            if self._cancelled:
                logger.info("Test execution cancelled")
                break

            result = self._execute_test(test_name, test_func)
            self.collector.add_result(result)

            if self.fail_fast and result.status in ["failed", "error"]:
                logger.info(f"Fail-fast enabled, stopping after {test_name} failure")
                break

    def _run_tests_parallel(self, tests_to_run: List[tuple]) -> None:
        """Run tests in parallel."""
        logger.info(f"Running tests in parallel (workers: {self.max_workers})")

        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)

        try:
            # Submit all tests
            future_to_test = {
                self._executor.submit(self._execute_test, test_name, test_func): test_name
                for test_name, test_func in tests_to_run
            }

            # Collect results as they complete
            for future in as_completed(future_to_test):
                if self._cancelled:
                    logger.info("Test execution cancelled")
                    break

                test_name = future_to_test[future]
                try:
                    result = future.result()
                    self.collector.add_result(result)

                    if self.fail_fast and result.status in ["failed", "error"]:
                        logger.info(
                            f"Fail-fast enabled, cancelling remaining tests after {test_name} failure"
                        )
                        self._cancelled = True
                        # Cancel remaining futures
                        for f in future_to_test:
                            if not f.done():
                                f.cancel()
                        break

                except Exception as e:
                    logger.error(f"Failed to get result for {test_name}: {e}")
                    self.collector.add_test_error(test_name, str(e))

        finally:
            if self._executor:
                self._executor.shutdown(wait=True)
                self._executor = None

    def cancel_execution(self) -> None:
        """Cancel test execution."""
        logger.info("Cancelling test execution")
        self._cancelled = True

        if self._executor:
            self._executor.shutdown(wait=False)

    def save_results(self, filename: Optional[str] = None) -> Path:
        """Save test results to file."""
        return self.collector.save_results(filename)

    def get_results(self) -> TestResultCollector:
        """Get the test result collector."""
        return self.collector


# Convenience functions for basic test execution
def run_single_test(
    test_func: Callable, test_name: Optional[str] = None, timeout: int = 60
) -> TestResult:
    """
    Run a single test function.

    Args:
        test_func: Test function to run
        test_name: Optional test name
        timeout: Test timeout in seconds

    Returns:
        Test result
    """
    runner = E2ETestRunner(max_workers=1, timeout=timeout)
    runner.register_test(test_func, test_name)

    test_name = test_name or test_func.__name__
    summary = runner.run_tests([test_name])

    # Return the single result
    suite = runner.collector.suites[0] if runner.collector.suites else None
    if suite and suite.results:
        return suite.results[0]

    # Fallback result if something went wrong
    return TestResult(
        test_name=test_name, status="error", error_message="Failed to retrieve test result"
    )


def run_test_suite(
    test_funcs: Dict[str, Callable],
    suite_name: str = "test_suite",
    max_workers: int = 4,
    timeout: int = 300,
    fail_fast: bool = False,
) -> Dict[str, Any]:
    """
    Run a suite of test functions.

    Args:
        test_funcs: Dictionary of test name -> test function
        suite_name: Name for the test suite
        max_workers: Maximum parallel workers
        timeout: Test timeout in seconds
        fail_fast: Stop on first failure

    Returns:
        Test execution summary
    """
    runner = E2ETestRunner(max_workers=max_workers, timeout=timeout, fail_fast=fail_fast)

    # Register all tests
    for name, func in test_funcs.items():
        runner.register_test(func, name)

    return runner.run_tests(suite_name=suite_name)


# Export main classes and functions
__all__ = ["E2ETestRunner", "TestExecutionError", "run_single_test", "run_test_suite"]
