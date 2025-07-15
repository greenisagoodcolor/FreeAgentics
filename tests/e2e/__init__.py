"""
E2E Testing Framework for FreeAgentics
=====================================

Comprehensive end-to-end testing framework supporting:
- Browser automation (Playwright, Selenium)
- API testing integration
- Real-time WebSocket testing
- Multi-service testing
- Test data management
- Visual regression testing
- Performance testing
- Cross-browser compatibility

Framework Components:
- Browser automation drivers
- Test utilities and helpers
- Test data factories
- Page object models
- Test configuration management
- Reporting and analytics
"""

__version__ = "1.0.0"
__all__ = [
    "BaseE2ETest",
    "PlaywrightDriver",
    "SeleniumDriver",
    "E2ETestConfig",
    "TestDataManager",
    "PageObjectBase",
    "E2ETestRunner",
    "TestResultCollector",
]

# Core imports
from .base import BaseE2ETest
from .config import E2ETestConfig
from .data_manager import TestDataManager
from .page_objects import PageObjectBase
from .reporting import TestResultCollector
from .runners import E2ETestRunner

# Driver imports
try:
    from .drivers.playwright_driver import PlaywrightDriver
except ImportError:
    PlaywrightDriver = None

try:
    from .drivers.selenium_driver import SeleniumDriver
except ImportError:
    SeleniumDriver = None

# Framework status
FRAMEWORK_STATUS = {
    "playwright_available": PlaywrightDriver is not None,
    "selenium_available": SeleniumDriver is not None,
    "api_testing_enabled": True,
    "websocket_testing_enabled": True,
    "visual_regression_enabled": True,
    "performance_testing_enabled": True,
}
