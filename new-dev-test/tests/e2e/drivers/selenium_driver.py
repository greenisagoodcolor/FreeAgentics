"""
Selenium Browser Driver
=======================

Selenium-based browser driver for E2E testing.
Legacy support for existing Selenium-based tests.
"""

import logging
from typing import Any, Dict, List

from ..config import E2ETestConfig
from .base import BrowserDriver

logger = logging.getLogger(__name__)

try:
    from selenium import webdriver
    from selenium.common.exceptions import NoSuchElementException, TimeoutException
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.by import By
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import Select, WebDriverWait

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    webdriver = None
    By = None
    WebDriverWait = None
    EC = None
    ActionChains = None
    Select = None
    ChromeOptions = None
    FirefoxOptions = None
    EdgeOptions = None
    TimeoutException = None
    NoSuchElementException = None


class SeleniumDriver(BrowserDriver):
    """Selenium-based browser driver"""

    def __init__(self, config: E2ETestConfig):
        super().__init__(config)
        self.driver = None
        self.wait = None
        self.network_logs = []
        self.console_logs = []

        if not SELENIUM_AVAILABLE:
            raise ImportError("Selenium is not available. Install with: pip install selenium")

    async def start(self):
        """Start the Selenium browser"""
        try:
            browser_type = self.config.browser_config.browser_type

            if browser_type == "chrome" or browser_type == "chromium":
                options = ChromeOptions()
                self._setup_chrome_options(options)
                self.driver = webdriver.Chrome(options=options)

            elif browser_type == "firefox":
                options = FirefoxOptions()
                self._setup_firefox_options(options)
                self.driver = webdriver.Firefox(options=options)

            elif browser_type == "edge":
                options = EdgeOptions()
                self._setup_edge_options(options)
                self.driver = webdriver.Edge(options=options)

            else:
                # Default to Chrome
                options = ChromeOptions()
                self._setup_chrome_options(options)
                self.driver = webdriver.Chrome(options=options)

            # Set window size
            width, height = self.config.browser_config.window_size
            self.driver.set_window_size(width, height)

            # Set implicit wait
            self.driver.implicitly_wait(self.config.browser_config.timeout)

            # Create WebDriverWait instance
            self.wait = WebDriverWait(self.driver, self.config.browser_config.timeout)

            # Enable logging
            self._enable_logging()

            logger.info(f"Selenium browser started: {browser_type}")

        except Exception as e:
            logger.error(f"Failed to start Selenium browser: {e}")
            raise

    async def stop(self):
        """Stop the Selenium browser"""
        try:
            if self.driver:
                self.driver.quit()
            logger.info("Selenium browser stopped")

        except Exception as e:
            logger.error(f"Failed to stop Selenium browser: {e}")

    def _setup_chrome_options(self, options):
        """Setup Chrome options"""
        if self.config.browser_config.headless:
            options.add_argument("--headless")

        for arg in self.config.get_browser_args():
            options.add_argument(arg)

        # Enable logging
        options.add_argument("--enable-logging")
        options.add_argument("--v=1")

        # Set preferences
        for key, value in self.config.browser_config.preferences.items():
            options.add_experimental_option("prefs", {key: value})

    def _setup_firefox_options(self, options):
        """Setup Firefox options"""
        if self.config.browser_config.headless:
            options.add_argument("--headless")

        for arg in self.config.get_browser_args():
            options.add_argument(arg)

    def _setup_edge_options(self, options):
        """Setup Edge options"""
        if self.config.browser_config.headless:
            options.add_argument("--headless")

        for arg in self.config.get_browser_args():
            options.add_argument(arg)

    def _enable_logging(self):
        """Enable browser logging"""
        # Note: Selenium logging capabilities are limited compared to Playwright
        try:
            # Enable performance logging for Chrome
            if hasattr(self.driver, "log"):
                self.driver.get_log("performance")
        except Exception as e:
            logger.debug(f"Could not enable performance logging: {e}")

    async def navigate(self, url: str):
        """Navigate to a URL"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        self.driver.get(url)
        logger.debug(f"Navigated to: {url}")

    async def click(self, selector: str):
        """Click an element"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        element = self._find_element(selector)
        element.click()
        logger.debug(f"Clicked element: {selector}")

    async def fill(self, selector: str, value: str):
        """Fill an input field"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        element = self._find_element(selector)
        element.clear()
        element.send_keys(value)
        logger.debug(f"Filled input {selector} with: {value}")

    async def get_text(self, selector: str) -> str:
        """Get text from an element"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        element = self._find_element(selector)
        return element.text

    async def wait_for_element(self, selector: str, timeout: float = 10.0):
        """Wait for an element to be visible"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        wait = WebDriverWait(self.driver, timeout)
        by, value = self._parse_selector(selector)
        wait.until(EC.visibility_of_element_located((by, value)))
        logger.debug(f"Element visible: {selector}")

    async def screenshot(self, path: str):
        """Take a screenshot"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        self.driver.save_screenshot(path)
        logger.debug(f"Screenshot saved: {path}")

    async def get_page_title(self) -> str:
        """Get page title"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        return self.driver.title

    async def get_current_url(self) -> str:
        """Get current URL"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        return self.driver.current_url

    async def execute_script(self, script: str) -> Any:
        """Execute JavaScript"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        return self.driver.execute_script(script)

    async def get_element_attribute(self, selector: str, attribute: str) -> str:
        """Get element attribute"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        element = self._find_element(selector)
        return element.get_attribute(attribute) or ""

    async def is_element_visible(self, selector: str) -> bool:
        """Check if element is visible"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        try:
            element = self._find_element(selector)
            return element.is_displayed()
        except (NoSuchElementException, TimeoutException):
            return False

    async def scroll_to_element(self, selector: str):
        """Scroll to an element"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        element = self._find_element(selector)
        self.driver.execute_script("arguments[0].scrollIntoView();", element)

    async def hover(self, selector: str):
        """Hover over an element"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        element = self._find_element(selector)
        ActionChains(self.driver).move_to_element(element).perform()

    async def double_click(self, selector: str):
        """Double click an element"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        element = self._find_element(selector)
        ActionChains(self.driver).double_click(element).perform()

    async def right_click(self, selector: str):
        """Right click an element"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        element = self._find_element(selector)
        ActionChains(self.driver).context_click(element).perform()

    async def select_option(self, selector: str, value: str):
        """Select option from dropdown"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        element = self._find_element(selector)
        select = Select(element)
        select.select_by_value(value)

    async def upload_file(self, selector: str, file_path: str):
        """Upload file to input"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        element = self._find_element(selector)
        element.send_keys(file_path)

    async def switch_to_frame(self, frame_selector: str):
        """Switch to iframe"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        frame = self._find_element(frame_selector)
        self.driver.switch_to.frame(frame)

    async def switch_to_default_content(self):
        """Switch back to default content"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        self.driver.switch_to.default_content()

    async def open_new_tab(self, url: str):
        """Open new tab"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        self.driver.execute_script(f"window.open('{url}', '_blank');")
        self.driver.switch_to.window(self.driver.window_handles[-1])

    async def close_current_tab(self):
        """Close current tab"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        self.driver.close()

    async def switch_to_tab(self, tab_index: int):
        """Switch to specific tab"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        handles = self.driver.window_handles
        if 0 <= tab_index < len(handles):
            self.driver.switch_to.window(handles[tab_index])

    async def get_cookies(self) -> List[Dict[str, Any]]:
        """Get all cookies"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        return self.driver.get_cookies()

    async def set_cookie(self, cookie: Dict[str, Any]):
        """Set a cookie"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        self.driver.add_cookie(cookie)

    async def delete_cookie(self, name: str):
        """Delete a cookie"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        self.driver.delete_cookie(name)

    async def clear_cookies(self):
        """Clear all cookies"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        self.driver.delete_all_cookies()

    async def get_local_storage(self, key: str) -> str:
        """Get local storage item"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        return self.driver.execute_script(f"return localStorage.getItem('{key}');")

    async def set_local_storage(self, key: str, value: str):
        """Set local storage item"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        self.driver.execute_script(f"localStorage.setItem('{key}', '{value}');")

    async def clear_local_storage(self):
        """Clear local storage"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        self.driver.execute_script("localStorage.clear();")

    async def wait_for_navigation(self, timeout: float = 10.0):
        """Wait for navigation to complete"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        # Wait for document ready state
        wait = WebDriverWait(self.driver, timeout)
        wait.until(lambda driver: driver.execute_script("return document.readyState") == "complete")

    async def wait_for_load_state(self, state: str = "load", timeout: float = 10.0):
        """Wait for specific load state"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        # Selenium doesn't have direct load state support like Playwright
        # So we'll wait for document ready state
        await self.wait_for_navigation(timeout)

    async def get_network_logs(self) -> List[Dict[str, Any]]:
        """Get network request logs"""
        # Selenium has limited network logging capabilities
        return self.network_logs

    async def get_console_logs(self) -> List[Dict[str, Any]]:
        """Get console logs"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        try:
            logs = self.driver.get_log("browser")
            return [
                {
                    "type": log["level"],
                    "text": log["message"],
                    "timestamp": log["timestamp"],
                }
                for log in logs
            ]
        except Exception as e:
            logger.debug(f"Could not get console logs: {e}")
            return self.console_logs

    async def set_viewport_size(self, width: int, height: int):
        """Set viewport size"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        self.driver.set_window_size(width, height)

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.driver:
            raise RuntimeError("Browser not started")

        # Get basic performance metrics via JavaScript
        metrics = self.driver.execute_script(
            """
            var timing = performance.timing;
            var navigation = performance.getEntriesByType('navigation')[0];
            return {
                load_time: timing.loadEventEnd - timing.navigationStart,
                dom_content_loaded: timing.domContentLoadedEventEnd - timing.navigationStart,
                first_paint: navigation ? navigation.loadEventEnd - navigation.loadEventStart : 0,
                memory_used: performance.memory ? performance.memory.usedJSHeapSize : 0,
                memory_total: performance.memory ? performance.memory.totalJSHeapSize : 0,
                network_requests: performance.getEntriesByType('resource').length
            };
        """
        )

        return metrics

    def _find_element(self, selector: str):
        """Find element by selector"""
        by, value = self._parse_selector(selector)
        return self.driver.find_element(by, value)

    def _parse_selector(self, selector: str) -> tuple:
        """Parse selector string to By type and value"""
        if selector.startswith("#"):
            return By.ID, selector[1:]
        elif selector.startswith("."):
            return By.CLASS_NAME, selector[1:]
        elif selector.startswith("["):
            # Attribute selector
            return By.CSS_SELECTOR, selector
        elif selector.startswith("//"):
            return By.XPATH, selector
        else:
            # Default to CSS selector
            return By.CSS_SELECTOR, selector
