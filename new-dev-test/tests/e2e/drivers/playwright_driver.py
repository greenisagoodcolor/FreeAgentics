"""
Playwright Browser Driver
=========================

Playwright-based browser driver for E2E testing.
Provides high-level interface for browser automation.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

from ..config import E2ETestConfig
from .base import BrowserDriver

logger = logging.getLogger(__name__)

try:
    from playwright.async_api import Browser, BrowserContext, Page, async_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    async_playwright = None
    Browser = None
    BrowserContext = None
    Page = None


class PlaywrightDriver(BrowserDriver):
    """Playwright-based browser driver"""

    def __init__(self, config: E2ETestConfig):
        super().__init__(config)
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.network_logs = []
        self.console_logs = []

        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError("Playwright is not available. Install with: pip install playwright")

    async def start(self):
        """Start the Playwright browser"""
        try:
            self.playwright = await async_playwright().start()

            # Get browser type
            browser_type = self.config.browser_config.browser_type
            if browser_type == "chrome" or browser_type == "chromium":
                browser_launcher = self.playwright.chromium
            elif browser_type == "firefox":
                browser_launcher = self.playwright.firefox
            elif browser_type == "safari":
                browser_launcher = self.playwright.webkit
            else:
                browser_launcher = self.playwright.chromium

            # Launch browser
            self.browser = await browser_launcher.launch(
                headless=self.config.browser_config.headless,
                slow_mo=self.config.browser_config.slow_mo,
                args=self.config.get_browser_args(),
            )

            # Create context
            self.context = await self.browser.new_context(
                viewport={
                    "width": self.config.browser_config.window_size[0],
                    "height": self.config.browser_config.window_size[1],
                }
            )

            # Create page
            self.page = await self.context.new_page()

            # Set up event listeners
            await self._setup_event_listeners()

            # Set default timeout
            self.page.set_default_timeout(self.config.browser_config.timeout * 1000)

            logger.info(f"Playwright browser started: {browser_type}")

        except Exception as e:
            logger.error(f"Failed to start Playwright browser: {e}")
            raise

    async def stop(self):
        """Stop the Playwright browser"""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()

            logger.info("Playwright browser stopped")

        except Exception as e:
            logger.error(f"Failed to stop Playwright browser: {e}")

    async def _setup_event_listeners(self):
        """Set up event listeners for logging"""
        if not self.page:
            return

        # Network request logging
        self.page.on("request", self._on_request)
        self.page.on("response", self._on_response)

        # Console logging
        self.page.on("console", self._on_console)

        # Error logging
        self.page.on("pageerror", self._on_page_error)

    def _on_request(self, request):
        """Handle network request"""
        self.network_logs.append(
            {
                "type": "request",
                "url": request.url,
                "method": request.method,
                "headers": request.headers,
                "timestamp": asyncio.get_event_loop().time(),
            }
        )

    def _on_response(self, response):
        """Handle network response"""
        self.network_logs.append(
            {
                "type": "response",
                "url": response.url,
                "status": response.status,
                "headers": response.headers,
                "timestamp": asyncio.get_event_loop().time(),
            }
        )

    def _on_console(self, message):
        """Handle console message"""
        self.console_logs.append(
            {
                "type": message.type,
                "text": message.text,
                "timestamp": asyncio.get_event_loop().time(),
            }
        )

    def _on_page_error(self, error):
        """Handle page error"""
        logger.error(f"Page error: {error}")
        self.console_logs.append(
            {
                "type": "error",
                "text": str(error),
                "timestamp": asyncio.get_event_loop().time(),
            }
        )

    async def navigate(self, url: str):
        """Navigate to a URL"""
        if not self.page:
            raise RuntimeError("Browser not started")

        await self.page.goto(url, wait_until="load")
        logger.debug(f"Navigated to: {url}")

    async def click(self, selector: str):
        """Click an element"""
        if not self.page:
            raise RuntimeError("Browser not started")

        await self.page.click(selector)
        logger.debug(f"Clicked element: {selector}")

    async def fill(self, selector: str, value: str):
        """Fill an input field"""
        if not self.page:
            raise RuntimeError("Browser not started")

        await self.page.fill(selector, value)
        logger.debug(f"Filled input {selector} with: {value}")

    async def get_text(self, selector: str) -> str:
        """Get text from an element"""
        if not self.page:
            raise RuntimeError("Browser not started")

        return await self.page.text_content(selector) or ""

    async def wait_for_element(self, selector: str, timeout: float = 10.0):
        """Wait for an element to be visible"""
        if not self.page:
            raise RuntimeError("Browser not started")

        await self.page.wait_for_selector(selector, timeout=timeout * 1000)
        logger.debug(f"Element visible: {selector}")

    async def screenshot(self, path: str):
        """Take a screenshot"""
        if not self.page:
            raise RuntimeError("Browser not started")

        await self.page.screenshot(path=path)
        logger.debug(f"Screenshot saved: {path}")

    async def get_page_title(self) -> str:
        """Get page title"""
        if not self.page:
            raise RuntimeError("Browser not started")

        return await self.page.title()

    async def get_current_url(self) -> str:
        """Get current URL"""
        if not self.page:
            raise RuntimeError("Browser not started")

        return self.page.url

    async def execute_script(self, script: str) -> Any:
        """Execute JavaScript"""
        if not self.page:
            raise RuntimeError("Browser not started")

        return await self.page.evaluate(script)

    async def get_element_attribute(self, selector: str, attribute: str) -> str:
        """Get element attribute"""
        if not self.page:
            raise RuntimeError("Browser not started")

        return await self.page.get_attribute(selector, attribute) or ""

    async def is_element_visible(self, selector: str) -> bool:
        """Check if element is visible"""
        if not self.page:
            raise RuntimeError("Browser not started")

        try:
            await self.page.wait_for_selector(selector, timeout=1000)
            return True
        except Exception:
            return False

    async def scroll_to_element(self, selector: str):
        """Scroll to an element"""
        if not self.page:
            raise RuntimeError("Browser not started")

        await self.page.locator(selector).scroll_into_view_if_needed()

    async def hover(self, selector: str):
        """Hover over an element"""
        if not self.page:
            raise RuntimeError("Browser not started")

        await self.page.hover(selector)

    async def double_click(self, selector: str):
        """Double click an element"""
        if not self.page:
            raise RuntimeError("Browser not started")

        await self.page.dblclick(selector)

    async def right_click(self, selector: str):
        """Right click an element"""
        if not self.page:
            raise RuntimeError("Browser not started")

        await self.page.click(selector, button="right")

    async def select_option(self, selector: str, value: str):
        """Select option from dropdown"""
        if not self.page:
            raise RuntimeError("Browser not started")

        await self.page.select_option(selector, value)

    async def upload_file(self, selector: str, file_path: str):
        """Upload file to input"""
        if not self.page:
            raise RuntimeError("Browser not started")

        await self.page.set_input_files(selector, file_path)

    async def switch_to_frame(self, frame_selector: str):
        """Switch to iframe"""
        if not self.page:
            raise RuntimeError("Browser not started")

        frame = self.page.frame_locator(frame_selector)
        # Playwright handles frames differently - return frame locator
        return frame

    async def switch_to_default_content(self):
        """Switch back to default content"""
        # Playwright handles this automatically

    async def open_new_tab(self, url: str):
        """Open new tab"""
        if not self.context:
            raise RuntimeError("Browser not started")

        new_page = await self.context.new_page()
        await new_page.goto(url)
        return new_page

    async def close_current_tab(self):
        """Close current tab"""
        if not self.page:
            raise RuntimeError("Browser not started")

        await self.page.close()

    async def switch_to_tab(self, tab_index: int):
        """Switch to specific tab"""
        if not self.context:
            raise RuntimeError("Browser not started")

        pages = self.context.pages
        if 0 <= tab_index < len(pages):
            self.page = pages[tab_index]
            await self.page.bring_to_front()

    async def get_cookies(self) -> List[Dict[str, Any]]:
        """Get all cookies"""
        if not self.context:
            raise RuntimeError("Browser not started")

        return await self.context.cookies()

    async def set_cookie(self, cookie: Dict[str, Any]):
        """Set a cookie"""
        if not self.context:
            raise RuntimeError("Browser not started")

        await self.context.add_cookies([cookie])

    async def delete_cookie(self, name: str):
        """Delete a cookie"""
        if not self.page:
            raise RuntimeError("Browser not started")

        await self.page.evaluate(
            f"document.cookie = '{name}=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/;'"
        )

    async def clear_cookies(self):
        """Clear all cookies"""
        if not self.context:
            raise RuntimeError("Browser not started")

        await self.context.clear_cookies()

    async def get_local_storage(self, key: str) -> str:
        """Get local storage item"""
        if not self.page:
            raise RuntimeError("Browser not started")

        return await self.page.evaluate(f"localStorage.getItem('{key}')")

    async def set_local_storage(self, key: str, value: str):
        """Set local storage item"""
        if not self.page:
            raise RuntimeError("Browser not started")

        await self.page.evaluate(f"localStorage.setItem('{key}', '{value}')")

    async def clear_local_storage(self):
        """Clear local storage"""
        if not self.page:
            raise RuntimeError("Browser not started")

        await self.page.evaluate("localStorage.clear()")

    async def wait_for_navigation(self, timeout: float = 10.0):
        """Wait for navigation to complete"""
        if not self.page:
            raise RuntimeError("Browser not started")

        await self.page.wait_for_load_state("networkidle", timeout=timeout * 1000)

    async def wait_for_load_state(self, state: str = "load", timeout: float = 10.0):
        """Wait for specific load state"""
        if not self.page:
            raise RuntimeError("Browser not started")

        await self.page.wait_for_load_state(state, timeout=timeout * 1000)

    async def get_network_logs(self) -> List[Dict[str, Any]]:
        """Get network request logs"""
        return self.network_logs

    async def get_console_logs(self) -> List[Dict[str, Any]]:
        """Get console logs"""
        return self.console_logs

    async def set_viewport_size(self, width: int, height: int):
        """Set viewport size"""
        if not self.page:
            raise RuntimeError("Browser not started")

        await self.page.set_viewport_size({"width": width, "height": height})

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.page:
            raise RuntimeError("Browser not started")

        # Get basic performance metrics
        metrics = await self.page.evaluate(
            """
            () => {
                const timing = performance.timing;
                const navigation = performance.getEntriesByType('navigation')[0];
                return {
                    load_time: timing.loadEventEnd - timing.navigationStart,
                    dom_content_loaded: timing.domContentLoadedEventEnd - timing.navigationStart,
                    first_paint: navigation ? navigation.loadEventEnd - navigation.loadEventStart : 0,
                    memory_used: performance.memory ? performance.memory.usedJSHeapSize : 0,
                    memory_total: performance.memory ? performance.memory.totalJSHeapSize : 0,
                    network_requests: performance.getEntriesByType('resource').length
                };
            }
        """
        )

        return metrics
