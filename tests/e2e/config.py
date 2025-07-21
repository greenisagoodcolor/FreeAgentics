"""
E2E Test Configuration Management
================================

Centralized configuration for E2E testing including:
- Browser settings
- Environment configurations
- Test data settings
- Timeout configurations
- Reporting configurations
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class BrowserType(str, Enum):
    """Supported browser types for E2E testing"""

    CHROME = "chrome"
    FIREFOX = "firefox"
    SAFARI = "safari"
    EDGE = "edge"
    CHROMIUM = "chromium"


class TestEnvironment(str, Enum):
    """Test environment configurations"""

    LOCAL = "local"
    DOCKER = "docker"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class BrowserConfig:
    """Browser-specific configuration"""

    browser_type: BrowserType = BrowserType.CHROME
    headless: bool = True
    window_size: tuple = (1920, 1080)
    timeout: int = 30
    slow_mo: int = 0  # Milliseconds delay between actions
    args: List[str] = field(
        default_factory=lambda: [
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-web-security",
            "--disable-features=VizDisplayCompositor",
        ]
    )
    preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class E2ETestConfig:
    """Main E2E test configuration"""

    # Environment settings
    environment: TestEnvironment = TestEnvironment.LOCAL
    base_url: str = "http://localhost:8000"
    frontend_url: str = "http://localhost:3000"
    api_url: str = "http://localhost:8000/api/v1"
    ws_url: str = "ws://localhost:8000/ws"

    # Browser configuration
    browser_config: BrowserConfig = field(default_factory=BrowserConfig)
    parallel_browsers: int = 1
    retry_attempts: int = 3

    # Test data management
    test_data_path: str = "tests/e2e/data"
    fixture_path: str = "tests/e2e/fixtures"
    screenshots_path: str = "tests/e2e/screenshots"
    videos_path: str = "tests/e2e/videos"

    # Database configuration
    test_db_url: str = "postgresql://test:test@localhost:5432/test_freeagentics"
    cleanup_db_after_tests: bool = True

    # Test execution settings
    max_test_duration: int = 300  # 5 minutes
    screenshot_on_failure: bool = True
    video_recording: bool = False
    visual_regression_enabled: bool = True

    # Reporting configuration
    report_path: str = "tests/e2e/reports"
    junit_xml_path: str = "tests/e2e/reports/junit.xml"
    html_report_path: str = "tests/e2e/reports/report.html"

    # Performance testing
    performance_enabled: bool = True
    performance_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "page_load_time": 3.0,
            "api_response_time": 1.0,
            "websocket_connection_time": 0.5,
            "memory_usage_mb": 500.0,
            "cpu_usage_percent": 80.0,
        }
    )

    # Authentication settings
    admin_user: Dict[str, str] = field(
        default_factory=lambda: {
            "username": "admin",
            "password": "admin123",
            "email": "admin@test.com",
        }
    )

    test_user: Dict[str, str] = field(
        default_factory=lambda: {
            "username": "testuser",
            "password": "test123",
            "email": "test@test.com",
        }
    )

    # Service availability checks
    required_services: List[str] = field(
        default_factory=lambda: ["backend", "frontend", "postgres", "redis"]
    )

    # Test categories
    test_categories: List[str] = field(
        default_factory=lambda: [
            "smoke",
            "regression",
            "integration",
            "performance",
            "security",
            "accessibility",
            "cross_browser",
        ]
    )

    @classmethod
    def from_environment(cls) -> "E2ETestConfig":
        """Create configuration from environment variables"""
        config = cls()

        # Override with environment variables
        if os.getenv("E2E_ENVIRONMENT"):
            config.environment = TestEnvironment(os.getenv("E2E_ENVIRONMENT"))

        if os.getenv("E2E_BASE_URL"):
            config.base_url = os.getenv("E2E_BASE_URL")

        if os.getenv("E2E_FRONTEND_URL"):
            config.frontend_url = os.getenv("E2E_FRONTEND_URL")

        if os.getenv("E2E_API_URL"):
            config.api_url = os.getenv("E2E_API_URL")

        if os.getenv("E2E_WS_URL"):
            config.ws_url = os.getenv("E2E_WS_URL")

        # Browser configuration
        if os.getenv("E2E_BROWSER"):
            config.browser_config.browser_type = BrowserType(os.getenv("E2E_BROWSER"))

        if os.getenv("E2E_HEADLESS"):
            config.browser_config.headless = os.getenv("E2E_HEADLESS").lower() == "true"

        if os.getenv("E2E_PARALLEL_BROWSERS"):
            config.parallel_browsers = int(os.getenv("E2E_PARALLEL_BROWSERS"))

        # Test database
        if os.getenv("E2E_TEST_DB_URL"):
            config.test_db_url = os.getenv("E2E_TEST_DB_URL")

        # Performance settings
        if os.getenv("E2E_PERFORMANCE_ENABLED"):
            config.performance_enabled = (
                os.getenv("E2E_PERFORMANCE_ENABLED").lower() == "true"
            )

        return config

    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []

        # Check required directories exist
        required_dirs = [
            self.test_data_path,
            self.fixture_path,
            self.screenshots_path,
            self.videos_path,
            self.report_path,
        ]

        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                except OSError as e:
                    issues.append(f"Cannot create directory {dir_path}: {e}")

        # Validate URLs
        if not self.base_url.startswith(("http://", "https://")):
            issues.append("base_url must start with http:// or https://")

        if not self.frontend_url.startswith(("http://", "https://")):
            issues.append("frontend_url must start with http:// or https://")

        if not self.api_url.startswith(("http://", "https://")):
            issues.append("api_url must start with http:// or https://")

        if not self.ws_url.startswith(("ws://", "wss://")):
            issues.append("ws_url must start with ws:// or wss://")

        # Validate timeouts
        if self.max_test_duration <= 0:
            issues.append("max_test_duration must be positive")

        if self.browser_config.timeout <= 0:
            issues.append("browser timeout must be positive")

        # Validate parallel configuration
        if self.parallel_browsers < 1:
            issues.append("parallel_browsers must be at least 1")

        return issues

    def get_browser_args(self) -> List[str]:
        """Get browser arguments based on environment"""
        args = self.browser_config.args.copy()

        if self.environment == TestEnvironment.DOCKER:
            # Docker-specific arguments
            args.extend(
                [
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--remote-debugging-port=9222",
                ]
            )

        return args

    def get_test_data_path(self, filename: str) -> str:
        """Get full path to test data file"""
        return os.path.join(self.test_data_path, filename)

    def get_fixture_path(self, filename: str) -> str:
        """Get full path to fixture file"""
        return os.path.join(self.fixture_path, filename)

    def get_screenshot_path(self, filename: str) -> str:
        """Get full path to screenshot file"""
        return os.path.join(self.screenshots_path, filename)

    def get_video_path(self, filename: str) -> str:
        """Get full path to video file"""
        return os.path.join(self.videos_path, filename)


# Global configuration instance
_config: Optional[E2ETestConfig] = None


def get_config() -> E2ETestConfig:
    """Get or create global E2E test configuration"""
    global _config
    if _config is None:
        _config = E2ETestConfig.from_environment()
    return _config


def set_config(config: E2ETestConfig) -> None:
    """Set global E2E test configuration"""
    global _config
    _config = config


def reset_config() -> None:
    """Reset global configuration to None"""
    global _config
    _config = None
