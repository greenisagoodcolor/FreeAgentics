"""Rate Limiting Configuration.

This module provides configuration management for rate limiting and DDoS protection
across different environments (development, staging, production).
"""

import os
from typing import Dict, Optional

from dataclasses import dataclass
from typing import Optional


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10
    window_size: int = 60
    block_duration: int = 300
    ddos_threshold: int = 1000
    ddos_block_duration: int = 3600


class RateLimitingConfig:
    """Configuration manager for rate limiting settings."""

    def __init__(self):
        self.environment = os.getenv("ENVIRONMENT", "development").lower()
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.enabled = os.getenv("RATE_LIMITING_ENABLED", "true").lower() == "true"

    def get_config_for_environment(self) -> Dict[str, RateLimitConfig]:
        """Get rate limiting configuration for current environment."""
        if self.environment == "production":
            return self._get_production_config()
        elif self.environment == "staging":
            return self._get_staging_config()
        else:
            return self._get_development_config()

    def _get_production_config(self) -> Dict[str, RateLimitConfig]:
        """Production rate limiting configuration - strict limits."""
        return {
            "auth": RateLimitConfig(
                requests_per_minute=3,
                requests_per_hour=50,
                burst_limit=2,
                block_duration=900,  # 15 minutes
                ddos_threshold=500,
                ddos_block_duration=7200,  # 2 hours
            ),
            "api": RateLimitConfig(
                requests_per_minute=60,
                requests_per_hour=1000,
                burst_limit=10,
                block_duration=600,  # 10 minutes
                ddos_threshold=1000,
                ddos_block_duration=3600,  # 1 hour
            ),
            "websocket": RateLimitConfig(
                requests_per_minute=100,
                requests_per_hour=2000,
                burst_limit=20,
                block_duration=300,  # 5 minutes
                ddos_threshold=2000,
                ddos_block_duration=1800,  # 30 minutes
            ),
            "static": RateLimitConfig(
                requests_per_minute=200,
                requests_per_hour=5000,
                burst_limit=50,
                block_duration=120,  # 2 minutes
                ddos_threshold=5000,
                ddos_block_duration=900,  # 15 minutes
            ),
        }

    def _get_staging_config(self) -> Dict[str, RateLimitConfig]:
        """Staging rate limiting configuration - moderate limits."""
        return {
            "auth": RateLimitConfig(
                requests_per_minute=5,
                requests_per_hour=100,
                burst_limit=3,
                block_duration=600,  # 10 minutes
                ddos_threshold=800,
                ddos_block_duration=3600,  # 1 hour
            ),
            "api": RateLimitConfig(
                requests_per_minute=100,
                requests_per_hour=2000,
                burst_limit=20,
                block_duration=300,  # 5 minutes
                ddos_threshold=1500,
                ddos_block_duration=1800,  # 30 minutes
            ),
            "websocket": RateLimitConfig(
                requests_per_minute=200,
                requests_per_hour=5000,
                burst_limit=50,
                block_duration=180,  # 3 minutes
                ddos_threshold=3000,
                ddos_block_duration=900,  # 15 minutes
            ),
            "static": RateLimitConfig(
                requests_per_minute=300,
                requests_per_hour=10000,
                burst_limit=100,
                block_duration=60,  # 1 minute
                ddos_threshold=8000,
                ddos_block_duration=600,  # 10 minutes
            ),
        }

    def _get_development_config(self) -> Dict[str, RateLimitConfig]:
        """Development rate limiting configuration - lenient limits."""
        return {
            "auth": RateLimitConfig(
                requests_per_minute=10,
                requests_per_hour=200,
                burst_limit=5,
                block_duration=300,  # 5 minutes
                ddos_threshold=1000,
                ddos_block_duration=1800,  # 30 minutes
            ),
            "api": RateLimitConfig(
                requests_per_minute=200,
                requests_per_hour=5000,
                burst_limit=50,
                block_duration=180,  # 3 minutes
                ddos_threshold=2000,
                ddos_block_duration=900,  # 15 minutes
            ),
            "websocket": RateLimitConfig(
                requests_per_minute=500,
                requests_per_hour=10000,
                burst_limit=100,
                block_duration=60,  # 1 minute
                ddos_threshold=5000,
                ddos_block_duration=600,  # 10 minutes
            ),
            "static": RateLimitConfig(
                requests_per_minute=1000,
                requests_per_hour=50000,
                burst_limit=200,
                block_duration=30,  # 30 seconds
                ddos_threshold=10000,
                ddos_block_duration=300,  # 5 minutes
            ),
        }

    def get_custom_config(self) -> Optional[Dict[str, RateLimitConfig]]:
        """Get custom rate limiting configuration from environment variables."""
        if not os.getenv("RATE_LIMITING_CUSTOM_CONFIG"):
            return None

        try:
            # Custom configuration via environment variables
            return {
                "auth": RateLimitConfig(
                    requests_per_minute=int(os.getenv("RATE_LIMIT_AUTH_PER_MINUTE", "5")),
                    requests_per_hour=int(os.getenv("RATE_LIMIT_AUTH_PER_HOUR", "100")),
                    burst_limit=int(os.getenv("RATE_LIMIT_AUTH_BURST", "3")),
                    block_duration=int(os.getenv("RATE_LIMIT_AUTH_BLOCK", "600")),
                    ddos_threshold=int(os.getenv("RATE_LIMIT_AUTH_DDOS_THRESHOLD", "500")),
                    ddos_block_duration=int(os.getenv("RATE_LIMIT_AUTH_DDOS_BLOCK", "3600")),
                ),
                "api": RateLimitConfig(
                    requests_per_minute=int(os.getenv("RATE_LIMIT_API_PER_MINUTE", "100")),
                    requests_per_hour=int(os.getenv("RATE_LIMIT_API_PER_HOUR", "2000")),
                    burst_limit=int(os.getenv("RATE_LIMIT_API_BURST", "20")),
                    block_duration=int(os.getenv("RATE_LIMIT_API_BLOCK", "300")),
                    ddos_threshold=int(os.getenv("RATE_LIMIT_API_DDOS_THRESHOLD", "1000")),
                    ddos_block_duration=int(os.getenv("RATE_LIMIT_API_DDOS_BLOCK", "3600")),
                ),
                "websocket": RateLimitConfig(
                    requests_per_minute=int(os.getenv("RATE_LIMIT_WS_PER_MINUTE", "200")),
                    requests_per_hour=int(os.getenv("RATE_LIMIT_WS_PER_HOUR", "5000")),
                    burst_limit=int(os.getenv("RATE_LIMIT_WS_BURST", "50")),
                    block_duration=int(os.getenv("RATE_LIMIT_WS_BLOCK", "180")),
                    ddos_threshold=int(os.getenv("RATE_LIMIT_WS_DDOS_THRESHOLD", "2000")),
                    ddos_block_duration=int(os.getenv("RATE_LIMIT_WS_DDOS_BLOCK", "1800")),
                ),
                "static": RateLimitConfig(
                    requests_per_minute=int(os.getenv("RATE_LIMIT_STATIC_PER_MINUTE", "200")),
                    requests_per_hour=int(os.getenv("RATE_LIMIT_STATIC_PER_HOUR", "10000")),
                    burst_limit=int(os.getenv("RATE_LIMIT_STATIC_BURST", "100")),
                    block_duration=int(os.getenv("RATE_LIMIT_STATIC_BLOCK", "60")),
                    ddos_threshold=int(os.getenv("RATE_LIMIT_STATIC_DDOS_THRESHOLD", "5000")),
                    ddos_block_duration=int(os.getenv("RATE_LIMIT_STATIC_DDOS_BLOCK", "900")),
                ),
            }
        except ValueError as e:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(f"Invalid custom rate limiting configuration: {e}")
            return None

    def get_active_config(self) -> Dict[str, RateLimitConfig]:
        """Get the active rate limiting configuration."""
        # Try custom config first
        custom_config = self.get_custom_config()
        if custom_config:
            return custom_config

        # Fall back to environment config
        return self.get_config_for_environment()

    def get_redis_config(self) -> Dict[str, any]:
        """Get Redis configuration for rate limiting."""
        return {
            "url": self.redis_url,
            "max_connections": int(os.getenv("REDIS_MAX_CONNECTIONS", "20")),
            "retry_on_timeout": os.getenv("REDIS_RETRY_ON_TIMEOUT", "true").lower() == "true",
            "socket_keepalive": os.getenv("REDIS_SOCKET_KEEPALIVE", "true").lower() == "true",
            "health_check_interval": int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30")),
            "socket_connect_timeout": int(os.getenv("REDIS_CONNECT_TIMEOUT", "5")),
            "socket_timeout": int(os.getenv("REDIS_SOCKET_TIMEOUT", "5")),
        }

    def get_monitoring_config(self) -> Dict[str, any]:
        """Get monitoring configuration for rate limiting."""
        return {
            "enable_metrics": os.getenv("RATE_LIMITING_METRICS_ENABLED", "true").lower() == "true",
            "metrics_interval": int(os.getenv("RATE_LIMITING_METRICS_INTERVAL", "60")),
            "alert_threshold": float(os.getenv("RATE_LIMITING_ALERT_THRESHOLD", "0.8")),
            "log_level": os.getenv("RATE_LIMITING_LOG_LEVEL", "INFO").upper(),
        }

    def is_enabled(self) -> bool:
        """Check if rate limiting is enabled."""
        return self.enabled

    def get_whitelist(self) -> list:
        """Get IP whitelist for rate limiting bypass."""
        whitelist_str = os.getenv("RATE_LIMITING_WHITELIST", "")
        if not whitelist_str:
            return []

        return [ip.strip() for ip in whitelist_str.split(",") if ip.strip()]

    def get_user_whitelist(self) -> list:
        """Get user whitelist for rate limiting bypass."""
        whitelist_str = os.getenv("RATE_LIMITING_USER_WHITELIST", "")
        if not whitelist_str:
            return []

        return [user.strip() for user in whitelist_str.split(",") if user.strip()]


# Global configuration instance
rate_limiting_config = RateLimitingConfig()


def get_rate_limiting_config() -> RateLimitingConfig:
    """Get the global rate limiting configuration."""
    return rate_limiting_config
