"""
Base E2E Test Class
==================

Base class for all E2E tests providing common functionality:
- Setup and teardown
- Browser management
- Test data management
- Assertions
- Reporting
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import Mock

import httpx
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from .config import E2ETestConfig, get_config
from .data_manager import TestDataManager
from .utils import E2ETestUtils

logger = logging.getLogger(__name__)


class BaseE2ETest(ABC):
    """Base class for all E2E tests."""

    def __init__(self, config: Optional[E2ETestConfig] = None):
        self.config = config or get_config()
        self.data_manager = TestDataManager(self.config)
        self.utils = E2ETestUtils(self.config)
        self.driver = None
        self.session = None
        self.test_start_time = None
        self.test_results = []

    async def setup(self):
        """Setup method called before each test."""
        self.test_start_time = time.time()
        logger.info(f"Starting E2E test: {self.__class__.__name__}")

        # Validate configuration
        issues = self.config.validate()
        if issues:
            raise ValueError(f"Configuration issues: {', '.join(issues)}")

        # Initialize driver
        await self._setup_driver()

        # Setup test database
        await self._setup_test_database()

        # Setup test data
        await self._setup_test_data()

        # Wait for services to be ready
        await self._wait_for_services()

    async def teardown(self):
        """Teardown method called after each test."""
        test_duration = (
            time.time() - self.test_start_time if self.test_start_time else 0
        )
        logger.info(
            f"Finished E2E test: {self.__class__.__name__} in {test_duration:.2f}s"
        )

        # Cleanup driver
        await self._cleanup_driver()

        # Cleanup test database
        await self._cleanup_test_database()

        # Cleanup test data
        await self._cleanup_test_data()

    async def _setup_driver(self):
        """Setup browser driver."""
        driver_type = self.config.browser_config.browser_type

        if driver_type == "playwright":
            from .drivers.playwright_driver import PlaywrightDriver

            self.driver = PlaywrightDriver(self.config)
        elif driver_type == "selenium":
            from .drivers.selenium_driver import SeleniumDriver

            self.driver = SeleniumDriver(self.config)
        else:
            # Default to mock driver for testing
            self.driver = Mock()
            logger.warning(f"Using mock driver for {driver_type}")

        if hasattr(self.driver, "start"):
            await self.driver.start()

    async def _cleanup_driver(self):
        """Cleanup browser driver."""
        if self.driver and hasattr(self.driver, "stop"):
            await self.driver.stop()

    async def _setup_test_database(self):
        """Setup test database."""
        if not self.config.test_db_url:
            return

        try:
            engine = create_engine(self.config.test_db_url)
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            self.session = SessionLocal()

            # Create test tables if needed
            await self._create_test_tables()

        except Exception as e:
            logger.error(f"Failed to setup test database: {e}")
            # Continue without database for basic tests

    async def _cleanup_test_database(self):
        """Cleanup test database."""
        if self.session:
            try:
                if self.config.cleanup_db_after_tests:
                    await self._clean_test_tables()
                self.session.close()
            except Exception as e:
                logger.error(f"Failed to cleanup test database: {e}")

    @abstractmethod
    async def _create_test_tables(self):
        """Create test-specific database tables."""
        # This would be implemented with actual database schema

    @abstractmethod
    async def _clean_test_tables(self):
        """Clean test-specific database tables."""
        # This would be implemented with actual database cleanup

    async def _setup_test_data(self):
        """Setup test data."""
        await self.data_manager.setup()

    async def _cleanup_test_data(self):
        """Cleanup test data."""
        await self.data_manager.cleanup()

    async def _wait_for_services(self):
        """Wait for required services to be ready."""
        for service in self.config.required_services:
            await self._wait_for_service(service)

    async def _wait_for_service(self, service: str):
        """Wait for a specific service to be ready."""
        max_retries = 30
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                if service == "backend":
                    async with httpx.AsyncClient() as client:
                        response = await client.get(f"{self.config.base_url}/health")
                        if response.status_code == 200:
                            logger.info(f"Service {service} is ready")
                            return

                elif service == "frontend":
                    async with httpx.AsyncClient() as client:
                        response = await client.get(self.config.frontend_url)
                        if response.status_code == 200:
                            logger.info(f"Service {service} is ready")
                            return

                elif service == "postgres":
                    # Check database connection
                    if self.session:
                        result = self.session.execute(text("SELECT 1"))
                        if result.fetchone():
                            logger.info(f"Service {service} is ready")
                            return

                elif service == "redis":
                    # Redis check would go here
                    logger.info(f"Service {service} check skipped")
                    return

            except Exception as e:
                logger.debug(
                    f"Service {service} not ready (attempt {attempt + 1}): {e}"
                )

            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)

        logger.warning(f"Service {service} not ready after {max_retries} attempts")

    # Helper methods for common test operations

    async def navigate_to(self, url: str):
        """Navigate to a URL."""
        if self.driver and hasattr(self.driver, "navigate"):
            await self.driver.navigate(url)

    async def click_element(self, selector: str):
        """Click an element by selector."""
        if self.driver and hasattr(self.driver, "click"):
            await self.driver.click(selector)

    async def fill_input(self, selector: str, value: str):
        """Fill an input field."""
        if self.driver and hasattr(self.driver, "fill"):
            await self.driver.fill(selector, value)

    async def get_text(self, selector: str) -> str:
        """Get text from an element."""
        if self.driver and hasattr(self.driver, "get_text"):
            return await self.driver.get_text(selector)
        return ""

    async def wait_for_element(self, selector: str, timeout: float = 10.0):
        """Wait for an element to be visible."""
        if self.driver and hasattr(self.driver, "wait_for_element"):
            await self.driver.wait_for_element(selector, timeout)

    async def take_screenshot(self, filename: str):
        """Take a screenshot."""
        if self.driver and hasattr(self.driver, "screenshot"):
            path = self.config.get_screenshot_path(filename)
            await self.driver.screenshot(path)

    async def api_request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """Make an API request."""
        url = f"{self.config.api_url}{endpoint}"
        async with httpx.AsyncClient() as client:
            response = await client.request(method, url, **kwargs)
            return response

    @abstractmethod
    async def websocket_connect(self, endpoint: str):
        """Connect to WebSocket endpoint."""
        # WebSocket connection logic would go here

    # Assertion helpers

    @abstractmethod
    def assert_element_visible(self, selector: str):
        """Assert that an element is visible."""
        # This would check element visibility

    @abstractmethod
    def assert_element_contains_text(self, selector: str, text: str):
        """Assert that an element contains specific text."""
        # This would check element text content

    @abstractmethod
    def assert_page_title(self, expected_title: str):
        """Assert page title."""
        # This would check page title

    @abstractmethod
    def assert_url_contains(self, url_fragment: str):
        """Assert that current URL contains a fragment."""
        # This would check current URL

    def assert_api_response_ok(self, response: httpx.Response):
        """Assert that API response is OK."""
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

    def assert_api_response_json(
        self, response: httpx.Response, expected_keys: List[str]
    ):
        """Assert that API response contains expected JSON keys."""
        data = response.json()
        for key in expected_keys:
            assert key in data, f"Expected key '{key}' not found in response"

    # Performance testing helpers

    async def measure_page_load_time(self, url: str) -> float:
        """Measure page load time."""
        start_time = time.time()
        await self.navigate_to(url)
        # Wait for page to be fully loaded
        await asyncio.sleep(0.5)
        return time.time() - start_time

    async def measure_api_response_time(self, endpoint: str) -> float:
        """Measure API response time."""
        start_time = time.time()
        await self.api_request("GET", endpoint)
        return time.time() - start_time

    # Test data helpers

    def get_test_user(self) -> Dict[str, str]:
        """Get test user credentials."""
        return self.config.test_user

    def get_admin_user(self) -> Dict[str, str]:
        """Get admin user credentials."""
        return self.config.admin_user

    def create_test_data(self, data_type: str) -> Dict[str, Any]:
        """Create test data of specified type."""
        return self.data_manager.create_test_data(data_type)

    # Abstract methods to be implemented by subclasses

    @abstractmethod
    async def run_test(self):
        """Main test method to be implemented by subclasses."""

    # Context manager support

    @asynccontextmanager
    async def test_context(self) -> AsyncGenerator[None, None]:
        """Context manager for test execution."""
        try:
            await self.setup()
            yield
        finally:
            await self.teardown()

    async def run_with_context(self):
        """Run test with proper setup and teardown."""
        async with self.test_context():
            await self.run_test()


class SmokeTest(BaseE2ETest):
    """Base class for smoke tests."""

    @abstractmethod
    async def run_smoke_test(self):
        """Run smoke test."""

    async def run_test(self):
        """Run smoke test implementation."""
        await self.run_smoke_test()


class IntegrationTest(BaseE2ETest):
    """Base class for integration tests."""

    @abstractmethod
    async def run_integration_test(self):
        """Run integration test."""

    async def run_test(self):
        """Run integration test implementation."""
        await self.run_integration_test()


class PerformanceTest(BaseE2ETest):
    """Base class for performance tests."""

    @abstractmethod
    async def run_performance_test(self):
        """Run performance test."""

    async def run_test(self):
        """Run performance test implementation."""
        if self.config.performance_enabled:
            await self.run_performance_test()
        else:
            logger.info("Performance testing disabled, skipping")


class SecurityTest(BaseE2ETest):
    """Base class for security tests."""

    @abstractmethod
    async def run_security_test(self):
        """Run security test."""

    async def run_test(self):
        """Run security test implementation."""
        await self.run_security_test()
