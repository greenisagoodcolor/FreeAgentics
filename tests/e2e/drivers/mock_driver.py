"""
Mock Browser Driver
==================

Mock driver for testing E2E framework itself without real browser.
Useful for unit testing and development.
"""

import asyncio
import logging
from typing import Any, Dict, List

from ..config import E2ETestConfig
from .base import BrowserDriver

logger = logging.getLogger(__name__)


class MockDriver(BrowserDriver):
    """Mock browser driver for testing"""

    def __init__(self, config: E2ETestConfig):
        super().__init__(config)
        self.current_url = ""
        self.page_title = "Mock Page"
        self.elements = {}
        self.network_logs = []
        self.console_logs = []
        self.cookies = []
        self.local_storage = {}
        self.screenshots = []
        self.viewport_size = (1920, 1080)
        self.is_started = False

    async def start(self):
        """Start the mock browser"""
        self.is_started = True
        logger.info("Mock browser started")

    async def stop(self):
        """Stop the mock browser"""
        self.is_started = False
        logger.info("Mock browser stopped")

    async def navigate(self, url: str):
        """Navigate to a URL"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        self.current_url = url
        logger.debug(f"Mock navigated to: {url}")

    async def click(self, selector: str):
        """Click an element"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        logger.debug(f"Mock clicked element: {selector}")

        # Simulate click delay
        await asyncio.sleep(0.1)

    async def fill(self, selector: str, value: str):
        """Fill an input field"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        self.elements[selector] = value
        logger.debug(f"Mock filled input {selector} with: {value}")

    async def get_text(self, selector: str) -> str:
        """Get text from an element"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        return self.elements.get(selector, f"Mock text for {selector}")

    async def wait_for_element(self, selector: str, timeout: float = 10.0):
        """Wait for an element to be visible"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        # Simulate wait
        await asyncio.sleep(0.1)
        logger.debug(f"Mock element visible: {selector}")

    async def screenshot(self, path: str):
        """Take a screenshot"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        self.screenshots.append(path)
        logger.debug(f"Mock screenshot saved: {path}")

    async def get_page_title(self) -> str:
        """Get page title"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        return self.page_title

    async def get_current_url(self) -> str:
        """Get current URL"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        return self.current_url

    async def execute_script(self, script: str) -> Any:
        """Execute JavaScript"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        # Mock script execution
        if "return" in script:
            return {"mock": "result"}
        return None

    async def get_element_attribute(
        self, selector: str, attribute: str
    ) -> str:
        """Get element attribute"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        return f"mock_{attribute}_value"

    async def is_element_visible(self, selector: str) -> bool:
        """Check if element is visible"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        return True

    async def scroll_to_element(self, selector: str):
        """Scroll to an element"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        logger.debug(f"Mock scrolled to element: {selector}")

    async def hover(self, selector: str):
        """Hover over an element"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        logger.debug(f"Mock hovered over element: {selector}")

    async def double_click(self, selector: str):
        """Double click an element"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        logger.debug(f"Mock double clicked element: {selector}")

    async def right_click(self, selector: str):
        """Right click an element"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        logger.debug(f"Mock right clicked element: {selector}")

    async def select_option(self, selector: str, value: str):
        """Select option from dropdown"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        self.elements[selector] = value
        logger.debug(f"Mock selected option {value} in {selector}")

    async def upload_file(self, selector: str, file_path: str):
        """Upload file to input"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        self.elements[selector] = file_path
        logger.debug(f"Mock uploaded file {file_path} to {selector}")

    async def switch_to_frame(self, frame_selector: str):
        """Switch to iframe"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        logger.debug(f"Mock switched to frame: {frame_selector}")

    async def switch_to_default_content(self):
        """Switch back to default content"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        logger.debug("Mock switched to default content")

    async def open_new_tab(self, url: str):
        """Open new tab"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        logger.debug(f"Mock opened new tab: {url}")

    async def close_current_tab(self):
        """Close current tab"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        logger.debug("Mock closed current tab")

    async def switch_to_tab(self, tab_index: int):
        """Switch to specific tab"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        logger.debug(f"Mock switched to tab: {tab_index}")

    async def get_cookies(self) -> List[Dict[str, Any]]:
        """Get all cookies"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        return self.cookies

    async def set_cookie(self, cookie: Dict[str, Any]):
        """Set a cookie"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        self.cookies.append(cookie)

    async def delete_cookie(self, name: str):
        """Delete a cookie"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        self.cookies = [c for c in self.cookies if c.get("name") != name]

    async def clear_cookies(self):
        """Clear all cookies"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        self.cookies = []

    async def get_local_storage(self, key: str) -> str:
        """Get local storage item"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        return self.local_storage.get(key, "")

    async def set_local_storage(self, key: str, value: str):
        """Set local storage item"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        self.local_storage[key] = value

    async def clear_local_storage(self):
        """Clear local storage"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        self.local_storage = {}

    async def wait_for_navigation(self, timeout: float = 10.0):
        """Wait for navigation to complete"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        # Simulate navigation wait
        await asyncio.sleep(0.1)

    async def wait_for_load_state(
        self, state: str = "load", timeout: float = 10.0
    ):
        """Wait for specific load state"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        # Simulate load state wait
        await asyncio.sleep(0.1)

    async def get_network_logs(self) -> List[Dict[str, Any]]:
        """Get network request logs"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        return self.network_logs

    async def get_console_logs(self) -> List[Dict[str, Any]]:
        """Get console logs"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        return self.console_logs

    async def set_viewport_size(self, width: int, height: int):
        """Set viewport size"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        self.viewport_size = (width, height)

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.is_started:
            raise RuntimeError("Browser not started")

        return {
            "load_time": 500,
            "dom_content_loaded": 300,
            "first_paint": 200,
            "memory_used": 50000000,
            "memory_total": 100000000,
            "network_requests": 10,
        }

    # Mock-specific methods

    def set_page_title(self, title: str):
        """Set mock page title"""
        self.page_title = title

    def add_network_log(self, log_entry: Dict[str, Any]):
        """Add network log entry"""
        self.network_logs.append(log_entry)

    def add_console_log(self, log_entry: Dict[str, Any]):
        """Add console log entry"""
        self.console_logs.append(log_entry)

    def set_element_text(self, selector: str, text: str):
        """Set element text for mocking"""
        self.elements[selector] = text

    def get_screenshots(self) -> List[str]:
        """Get list of screenshot paths"""
        return self.screenshots

    def get_viewport_size(self) -> tuple:
        """Get viewport size"""
        return self.viewport_size
