"""
E2E Test Browser Drivers
========================

Browser automation drivers for E2E testing:
- Playwright driver (recommended)
- Selenium driver (legacy support)
- Mock driver (for testing)
"""

__all__ = [
    "BrowserDriver",
    "PlaywrightDriver",
    "SeleniumDriver",
    "MockDriver",
    "get_driver",
]

from .base import BrowserDriver
from .mock_driver import MockDriver

# Optional imports based on availability
try:
    from .playwright_driver import PlaywrightDriver
except ImportError:
    PlaywrightDriver = None

try:
    from .selenium_driver import SeleniumDriver
except ImportError:
    SeleniumDriver = None


def get_driver(driver_type: str, config):
    """Get driver instance based on type"""
    if driver_type == "playwright" and PlaywrightDriver:
        return PlaywrightDriver(config)
    elif driver_type == "selenium" and SeleniumDriver:
        return SeleniumDriver(config)
    elif driver_type == "mock":
        return MockDriver(config)
    else:
        raise ValueError(f"Unsupported driver type: {driver_type}")
