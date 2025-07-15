"""
E2E Test Utilities
==================

Common utility functions for E2E tests:
- URL helpers
- Wait helpers
- Assertion helpers
- Data validation
- Screenshot helpers
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import httpx
from PIL import Image

from .config import E2ETestConfig

logger = logging.getLogger(__name__)


class E2ETestUtils:
    """Utility functions for E2E tests"""

    def __init__(self, config: E2ETestConfig):
        self.config = config

    # URL helpers

    def build_url(self, path: str, base_url: Optional[str] = None) -> str:
        """Build full URL from path"""
        base = base_url or self.config.base_url
        return urljoin(base, path)

    def build_api_url(self, endpoint: str) -> str:
        """Build API URL from endpoint"""
        return urljoin(self.config.api_url, endpoint)

    def build_frontend_url(self, path: str) -> str:
        """Build frontend URL from path"""
        return urljoin(self.config.frontend_url, path)

    def build_ws_url(self, endpoint: str) -> str:
        """Build WebSocket URL from endpoint"""
        return urljoin(self.config.ws_url, endpoint)

    def extract_path_from_url(self, url: str) -> str:
        """Extract path from URL"""
        parsed = urlparse(url)
        return parsed.path

    def extract_query_params(self, url: str) -> Dict[str, str]:
        """Extract query parameters from URL"""
        from urllib.parse import parse_qs

        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        return {k: v[0] if len(v) == 1 else v for k, v in params.items()}

    # Wait helpers

    async def wait_for_condition(
        self,
        condition: Callable[[], bool],
        timeout: float = 10.0,
        interval: float = 0.5,
        error_message: str = "Condition not met",
    ) -> bool:
        """Wait for a condition to be true"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if await self._call_condition(condition):
                return True
            await asyncio.sleep(interval)

        raise TimeoutError(error_message)

    async def _call_condition(self, condition: Callable[[], bool]) -> bool:
        """Call condition function, handling both sync and async"""
        if asyncio.iscoroutinefunction(condition):
            return await condition()
        else:
            return condition()

    async def wait_for_element_text(
        self, driver, selector: str, expected_text: str, timeout: float = 10.0
    ) -> bool:
        """Wait for element to contain expected text"""

        async def check_text():
            try:
                text = await driver.get_text(selector)
                return expected_text in text
            except:
                return False

        return await self.wait_for_condition(
            check_text,
            timeout,
            error_message=f"Element {selector} did not contain text '{expected_text}'",
        )

    async def wait_for_element_visible(self, driver, selector: str, timeout: float = 10.0) -> bool:
        """Wait for element to be visible"""

        async def check_visible():
            try:
                return await driver.is_element_visible(selector)
            except:
                return False

        return await self.wait_for_condition(
            check_visible, timeout, error_message=f"Element {selector} not visible"
        )

    async def wait_for_url_change(self, driver, initial_url: str, timeout: float = 10.0) -> bool:
        """Wait for URL to change from initial URL"""

        async def check_url():
            try:
                current_url = await driver.get_current_url()
                return current_url != initial_url
            except:
                return False

        return await self.wait_for_condition(
            check_url, timeout, error_message=f"URL did not change from {initial_url}"
        )

    async def wait_for_page_load(self, driver, timeout: float = 10.0) -> bool:
        """Wait for page to be fully loaded"""

        async def check_loaded():
            try:
                ready_state = await driver.execute_script("return document.readyState")
                return ready_state == "complete"
            except:
                return False

        return await self.wait_for_condition(
            check_loaded, timeout, error_message="Page not fully loaded"
        )

    # API helpers

    async def make_api_request(
        self,
        method: str,
        endpoint: str,
        headers: Optional[Dict[str, str]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> httpx.Response:
        """Make API request with common settings"""
        url = self.build_api_url(endpoint)

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(
                method=method, url=url, headers=headers, json=json_data, params=params
            )

            logger.debug(f"API {method} {url} -> {response.status_code}")
            return response

    async def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user and return tokens"""
        try:
            response = await self.make_api_request(
                "POST", "/auth/login", json_data={"username": username, "password": password}
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Authentication failed: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None

    async def create_authenticated_headers(self, username: str, password: str) -> Dict[str, str]:
        """Create headers with authentication token"""
        auth_data = await self.authenticate_user(username, password)
        if auth_data and "access_token" in auth_data:
            return {
                "Authorization": f"Bearer {auth_data['access_token']}",
                "Content-Type": "application/json",
            }
        return {"Content-Type": "application/json"}

    # Screenshot helpers

    def generate_screenshot_filename(self, test_name: str, suffix: str = "") -> str:
        """Generate screenshot filename"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_test_name = "".join(c for c in test_name if c.isalnum() or c in "_-")
        filename = f"{safe_test_name}_{timestamp}"
        if suffix:
            filename += f"_{suffix}"
        return f"{filename}.png"

    async def take_screenshot_on_failure(
        self, driver, test_name: str, error: Exception
    ) -> Optional[str]:
        """Take screenshot on test failure"""
        if not self.config.screenshot_on_failure:
            return None

        try:
            filename = self.generate_screenshot_filename(test_name, "failure")
            path = self.config.get_screenshot_path(filename)
            await driver.screenshot(path)
            logger.info(f"Failure screenshot saved: {path}")
            return path

        except Exception as e:
            logger.error(f"Failed to take screenshot: {e}")
            return None

    def compare_screenshots(
        self, reference_path: str, current_path: str, threshold: float = 0.95
    ) -> bool:
        """Compare two screenshots for visual regression testing"""
        try:
            # Open images
            ref_image = Image.open(reference_path)
            cur_image = Image.open(current_path)

            # Ensure same size
            if ref_image.size != cur_image.size:
                cur_image = cur_image.resize(ref_image.size)

            # Convert to same mode
            if ref_image.mode != cur_image.mode:
                cur_image = cur_image.convert(ref_image.mode)

            # Calculate similarity
            similarity = self._calculate_image_similarity(ref_image, cur_image)

            logger.debug(f"Screenshot similarity: {similarity:.2f}")
            return similarity >= threshold

        except Exception as e:
            logger.error(f"Screenshot comparison failed: {e}")
            return False

    def _calculate_image_similarity(self, img1: Image.Image, img2: Image.Image) -> float:
        """Calculate image similarity using histogram comparison"""
        # Convert to RGB if needed
        if img1.mode != "RGB":
            img1 = img1.convert("RGB")
        if img2.mode != "RGB":
            img2 = img2.convert("RGB")

        # Get histograms
        hist1 = img1.histogram()
        hist2 = img2.histogram()

        # Calculate correlation coefficient
        sum1 = sum(hist1)
        sum2 = sum(hist2)

        if sum1 == 0 or sum2 == 0:
            return 0.0

        # Normalize histograms
        hist1 = [h / sum1 for h in hist1]
        hist2 = [h / sum2 for h in hist2]

        # Calculate correlation
        correlation = sum(h1 * h2 for h1, h2 in zip(hist1, hist2))
        return correlation

    # Data validation helpers

    def validate_json_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Validate JSON data against schema"""
        try:
            import jsonschema

            jsonschema.validate(data, schema)
            return True
        except ImportError:
            logger.warning("jsonschema not available, skipping validation")
            return True
        except Exception as e:
            logger.error(f"JSON schema validation failed: {e}")
            return False

    def validate_api_response(
        self,
        response: httpx.Response,
        expected_status: int = 200,
        expected_keys: Optional[List[str]] = None,
    ) -> bool:
        """Validate API response"""
        # Check status code
        if response.status_code != expected_status:
            logger.error(f"Expected status {expected_status}, got {response.status_code}")
            return False

        # Check JSON content
        try:
            data = response.json()
        except Exception as e:
            logger.error(f"Response is not valid JSON: {e}")
            return False

        # Check expected keys
        if expected_keys:
            for key in expected_keys:
                if key not in data:
                    logger.error(f"Expected key '{key}' not found in response")
                    return False

        return True

    def validate_websocket_message(
        self, message: Dict[str, Any], expected_type: Optional[str] = None
    ) -> bool:
        """Validate WebSocket message format"""
        # Check basic structure
        if not isinstance(message, dict):
            logger.error("WebSocket message is not a dictionary")
            return False

        # Check required fields
        required_fields = ["type", "data", "timestamp"]
        for field in required_fields:
            if field not in message:
                logger.error(f"Required field '{field}' missing from WebSocket message")
                return False

        # Check message type
        if expected_type and message["type"] != expected_type:
            logger.error(f"Expected message type '{expected_type}', got '{message['type']}'")
            return False

        return True

    # Performance helpers

    async def measure_performance(self, operation: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Measure performance of an operation"""
        start_time = time.time()
        start_memory = self._get_memory_usage()

        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)

            end_time = time.time()
            end_memory = self._get_memory_usage()

            return {
                "duration": end_time - start_time,
                "memory_delta": end_memory - start_memory,
                "success": True,
                "result": result,
            }

        except Exception as e:
            end_time = time.time()
            end_memory = self._get_memory_usage()

            return {
                "duration": end_time - start_time,
                "memory_delta": end_memory - start_memory,
                "success": False,
                "error": str(e),
            }

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            import psutil

            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0

    # File helpers

    def create_test_file(self, content: str, filename: str) -> str:
        """Create test file with content"""
        file_path = os.path.join(self.config.test_data_path, filename)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as f:
            f.write(content)

        return file_path

    def read_test_file(self, filename: str) -> str:
        """Read test file content"""
        file_path = os.path.join(self.config.test_data_path, filename)

        with open(file_path, "r") as f:
            return f.read()

    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        hash_sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()

    # Assertion helpers

    def assert_element_text_contains(
        self, actual_text: str, expected_text: str, ignore_case: bool = False
    ):
        """Assert that element text contains expected text"""
        if ignore_case:
            actual_text = actual_text.lower()
            expected_text = expected_text.lower()

        assert (
            expected_text in actual_text
        ), f"Expected text '{expected_text}' not found in '{actual_text}'"

    def assert_url_path_equals(self, actual_url: str, expected_path: str):
        """Assert that URL path equals expected path"""
        actual_path = self.extract_path_from_url(actual_url)
        assert actual_path == expected_path, f"Expected path '{expected_path}', got '{actual_path}'"

    def assert_response_time_under(self, duration: float, threshold: float):
        """Assert that response time is under threshold"""
        assert duration < threshold, f"Response time {duration:.2f}s exceeds threshold {threshold}s"

    def assert_memory_usage_under(self, memory_mb: float, threshold: float):
        """Assert that memory usage is under threshold"""
        assert (
            memory_mb < threshold
        ), f"Memory usage {memory_mb:.2f}MB exceeds threshold {threshold}MB"

    # Debugging helpers

    def log_browser_state(self, driver):
        """Log current browser state for debugging"""
        try:
            logger.debug(f"Current URL: {driver.get_current_url()}")
            logger.debug(f"Page title: {driver.get_page_title()}")
            logger.debug(f"Window size: {driver.get_viewport_size()}")
        except Exception as e:
            logger.error(f"Failed to log browser state: {e}")

    def log_api_response(self, response: httpx.Response):
        """Log API response for debugging"""
        logger.debug(f"API Response: {response.status_code}")
        logger.debug(f"Headers: {dict(response.headers)}")
        try:
            logger.debug(f"Body: {response.json()}")
        except:
            logger.debug(f"Body: {response.text}")

    def save_debug_info(self, test_name: str, data: Dict[str, Any]):
        """Save debug information to file"""
        debug_file = os.path.join(
            self.config.report_path, f"debug_{test_name}_{int(time.time())}.json"
        )

        with open(debug_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Debug info saved: {debug_file}")

    # Retry helpers

    async def retry_operation(
        self,
        operation: Callable,
        max_retries: int = 3,
        delay: float = 1.0,
        backoff_factor: float = 2.0,
        *args,
        **kwargs,
    ) -> Any:
        """Retry operation with exponential backoff"""
        last_exception = None

        for attempt in range(max_retries):
            try:
                if asyncio.iscoroutinefunction(operation):
                    return await operation(*args, **kwargs)
                else:
                    return operation(*args, **kwargs)

            except Exception as e:
                last_exception = e
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

                if attempt < max_retries - 1:
                    await asyncio.sleep(delay * (backoff_factor**attempt))

        raise last_exception
