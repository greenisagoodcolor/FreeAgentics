"""Centralized development environment detection and configuration.

This module provides a single source of truth for environment detection,
replacing scattered os.getenv("DATABASE_URL") checks throughout the codebase.
"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class EnvironmentType(str, Enum):
    """Supported environment types."""

    DEVELOPMENT = "dev"
    TEST = "test"


@dataclass
class EnvironmentConfig:
    """Environment-specific configuration."""

    type: EnvironmentType
    debug: bool
    database_required: bool
    auth_required: bool
    rate_limiting_enabled: bool
    websocket_endpoint: str
    observability_enabled: bool

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.type == EnvironmentType.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return False  # No production mode for external dev test

    @property
    def is_test(self) -> bool:
        """Check if running in test mode."""
        return self.type == EnvironmentType.TEST


class DevelopmentEnvironment:
    """Centralized environment detection and configuration manager."""

    _instance: Optional["DevelopmentEnvironment"] = None
    _config: Optional[EnvironmentConfig] = None

    def __new__(cls) -> "DevelopmentEnvironment":
        """Singleton pattern to ensure consistent environment detection."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize environment detection."""
        if self._config is None:
            self._config = self._detect_environment()
            logger.info(f"🌍 Environment detected: {self._config.type}")

    def _detect_environment(self) -> EnvironmentConfig:
        """Detect current environment and return appropriate configuration."""
        # Check test environment
        if os.getenv("PYTEST_CURRENT_TEST") or "pytest" in os.getenv("_", ""):
            return self._test_config()

        # Always default to development for external dev test
        return self._development_config()

    def _development_config(self) -> EnvironmentConfig:
        """Development environment configuration."""
        return EnvironmentConfig(
            type=EnvironmentType.DEVELOPMENT,
            debug=True,
            database_required=False,
            auth_required=False,
            rate_limiting_enabled=False,
            websocket_endpoint="/api/v1/ws/connections",
            observability_enabled=True,
        )

    def _test_config(self) -> EnvironmentConfig:
        """Test environment configuration."""
        return EnvironmentConfig(
            type=EnvironmentType.TEST,
            debug=False,
            database_required=False,
            auth_required=False,
            rate_limiting_enabled=False,
            websocket_endpoint="/api/v1/ws/test",
            observability_enabled=False,
        )

    @property
    def config(self) -> EnvironmentConfig:
        """Get current environment configuration."""
        return self._config

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self._config.is_development

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self._config.is_production

    @property
    def is_test(self) -> bool:
        """Check if running in test mode."""
        return self._config.is_test

    def require_development(self) -> None:
        """Raise error if not in development mode."""
        if not self.is_development:
            raise RuntimeError(
                f"This operation requires development mode, but environment is {self._config.type}"
            )


# Global instance
environment = DevelopmentEnvironment()


# Convenience functions for backward compatibility
def is_development() -> bool:
    """Check if running in development mode."""
    return environment.is_development


def is_production() -> bool:
    """Check if running in production mode."""
    return False  # No production mode for external dev test


def is_test() -> bool:
    """Check if running in test mode."""
    return environment.is_test


def get_websocket_endpoint() -> str:
    """Get the appropriate WebSocket endpoint for current environment."""
    return environment.config.websocket_endpoint


def should_enable_rate_limiting() -> bool:
    """Check if rate limiting should be enabled."""
    return environment.config.rate_limiting_enabled


def should_require_auth() -> bool:
    """Check if authentication should be required."""
    return environment.config.auth_required


def get_environment_info() -> dict:
    """Get environment information for debugging."""
    config = environment.config
    return {
        "type": config.type,
        "debug": config.debug,
        "database_required": config.database_required,
        "auth_required": config.auth_required,
        "rate_limiting_enabled": config.rate_limiting_enabled,
        "websocket_endpoint": config.websocket_endpoint,
        "observability_enabled": config.observability_enabled,
    }
