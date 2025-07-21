"""
Base Browser Driver Interface
=============================

Abstract base class for browser drivers providing common interface
for different browser automation tools.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..config import E2ETestConfig


class BrowserDriver(ABC):
    """Abstract base class for browser drivers"""

    def __init__(self, config: E2ETestConfig):
        self.config = config
        self.browser = None
        self.page = None
        self.context = None

    @abstractmethod
    async def start(self):
        """Start the browser driver"""

    @abstractmethod
    async def stop(self):
        """Stop the browser driver"""

    @abstractmethod
    async def navigate(self, url: str):
        """Navigate to a URL"""

    @abstractmethod
    async def click(self, selector: str):
        """Click an element"""

    @abstractmethod
    async def fill(self, selector: str, value: str):
        """Fill an input field"""

    @abstractmethod
    async def get_text(self, selector: str) -> str:
        """Get text from an element"""

    @abstractmethod
    async def wait_for_element(self, selector: str, timeout: float = 10.0):
        """Wait for an element to be visible"""

    @abstractmethod
    async def screenshot(self, path: str):
        """Take a screenshot"""

    @abstractmethod
    async def get_page_title(self) -> str:
        """Get page title"""

    @abstractmethod
    async def get_current_url(self) -> str:
        """Get current URL"""

    @abstractmethod
    async def execute_script(self, script: str) -> Any:
        """Execute JavaScript"""

    @abstractmethod
    async def get_element_attribute(self, selector: str, attribute: str) -> str:
        """Get element attribute"""

    @abstractmethod
    async def is_element_visible(self, selector: str) -> bool:
        """Check if element is visible"""

    @abstractmethod
    async def scroll_to_element(self, selector: str):
        """Scroll to an element"""

    @abstractmethod
    async def hover(self, selector: str):
        """Hover over an element"""

    @abstractmethod
    async def double_click(self, selector: str):
        """Double click an element"""

    @abstractmethod
    async def right_click(self, selector: str):
        """Right click an element"""

    @abstractmethod
    async def select_option(self, selector: str, value: str):
        """Select option from dropdown"""

    @abstractmethod
    async def upload_file(self, selector: str, file_path: str):
        """Upload file to input"""

    @abstractmethod
    async def switch_to_frame(self, frame_selector: str):
        """Switch to iframe"""

    @abstractmethod
    async def switch_to_default_content(self):
        """Switch back to default content"""

    @abstractmethod
    async def open_new_tab(self, url: str):
        """Open new tab"""

    @abstractmethod
    async def close_current_tab(self):
        """Close current tab"""

    @abstractmethod
    async def switch_to_tab(self, tab_index: int):
        """Switch to specific tab"""

    @abstractmethod
    async def get_cookies(self) -> List[Dict[str, Any]]:
        """Get all cookies"""

    @abstractmethod
    async def set_cookie(self, cookie: Dict[str, Any]):
        """Set a cookie"""

    @abstractmethod
    async def delete_cookie(self, name: str):
        """Delete a cookie"""

    @abstractmethod
    async def clear_cookies(self):
        """Clear all cookies"""

    @abstractmethod
    async def get_local_storage(self, key: str) -> str:
        """Get local storage item"""

    @abstractmethod
    async def set_local_storage(self, key: str, value: str):
        """Set local storage item"""

    @abstractmethod
    async def clear_local_storage(self):
        """Clear local storage"""

    @abstractmethod
    async def wait_for_navigation(self, timeout: float = 10.0):
        """Wait for navigation to complete"""

    @abstractmethod
    async def wait_for_load_state(self, state: str = "load", timeout: float = 10.0):
        """Wait for specific load state"""

    @abstractmethod
    async def get_network_logs(self) -> List[Dict[str, Any]]:
        """Get network request logs"""

    @abstractmethod
    async def get_console_logs(self) -> List[Dict[str, Any]]:
        """Get console logs"""

    @abstractmethod
    async def set_viewport_size(self, width: int, height: int):
        """Set viewport size"""

    @abstractmethod
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
