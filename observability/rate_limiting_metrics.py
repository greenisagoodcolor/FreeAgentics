"""
Rate Limiting Metrics for Prometheus monitoring.

This module provides custom metrics for monitoring rate limiting effectiveness,
DDoS protection, and security events.
"""

from prometheus_client import Counter, Gauge, Histogram, Info
from typing import Dict, Optional
import time


# Rate limiting metrics
rate_limit_requests_total = Counter(
    'freeagentics_rate_limit_requests_total',
    'Total number of requests processed by rate limiter',
    ['endpoint', 'method', 'result']  # result: allowed, limited, blocked
)

rate_limit_violations_total = Counter(
    'freeagentics_rate_limit_violations_total',
    'Total number of rate limit violations',
    ['endpoint', 'identifier_type', 'reason']  # identifier_type: ip, user
)

rate_limit_blocks_total = Counter(
    'freeagentics_rate_limit_blocks_total',
    'Total number of IP/user blocks',
    ['block_type', 'reason']  # block_type: ip, user; reason: rate_limit, ddos, suspicious
)

# DDoS protection metrics
ddos_attacks_detected_total = Counter(
    'freeagentics_ddos_attacks_detected_total',
    'Total number of detected DDoS attacks',
    ['attack_type', 'source_ip']
)

suspicious_patterns_detected_total = Counter(
    'freeagentics_suspicious_patterns_detected_total',
    'Total number of detected suspicious patterns',
    ['pattern_type']  # rapid_404, rapid_errors, path_scanning, large_requests
)

# Current state metrics
active_blocks_gauge = Gauge(
    'freeagentics_active_blocks_current',
    'Current number of active IP/user blocks',
    ['block_type']
)

rate_limit_remaining_gauge = Gauge(
    'freeagentics_rate_limit_remaining',
    'Remaining requests in rate limit window',
    ['endpoint', 'identifier']
)

# Performance metrics
rate_limit_check_duration = Histogram(
    'freeagentics_rate_limit_check_duration_seconds',
    'Time spent checking rate limits',
    ['algorithm']  # sliding_window, token_bucket
)

redis_operation_duration = Histogram(
    'freeagentics_rate_limit_redis_operation_duration_seconds',
    'Time spent on Redis operations for rate limiting',
    ['operation']  # get, set, incr, expire
)

# Configuration info
rate_limit_config_info = Info(
    'freeagentics_rate_limit_config',
    'Rate limiting configuration information'
)


class RateLimitingMetrics:
    """Helper class for recording rate limiting metrics."""
    
    @staticmethod
    def record_request(endpoint: str, method: str, result: str):
        """Record a rate limit check result."""
        rate_limit_requests_total.labels(
            endpoint=endpoint,
            method=method,
            result=result
        ).inc()
    
    @staticmethod
    def record_violation(endpoint: str, identifier_type: str, reason: str):
        """Record a rate limit violation."""
        rate_limit_violations_total.labels(
            endpoint=endpoint,
            identifier_type=identifier_type,
            reason=reason
        ).inc()
    
    @staticmethod
    def record_block(block_type: str, reason: str):
        """Record an IP/user block."""
        rate_limit_blocks_total.labels(
            block_type=block_type,
            reason=reason
        ).inc()
        
        # Update active blocks gauge
        active_blocks_gauge.labels(block_type=block_type).inc()
    
    @staticmethod
    def record_unblock(block_type: str):
        """Record an IP/user unblock."""
        active_blocks_gauge.labels(block_type=block_type).dec()
    
    @staticmethod
    def record_ddos_attack(attack_type: str, source_ip: str):
        """Record a detected DDoS attack."""
        ddos_attacks_detected_total.labels(
            attack_type=attack_type,
            source_ip=source_ip
        ).inc()
    
    @staticmethod
    def record_suspicious_pattern(pattern_type: str):
        """Record a detected suspicious pattern."""
        suspicious_patterns_detected_total.labels(
            pattern_type=pattern_type
        ).inc()
    
    @staticmethod
    def update_remaining_requests(endpoint: str, identifier: str, remaining: int):
        """Update the remaining requests gauge."""
        rate_limit_remaining_gauge.labels(
            endpoint=endpoint,
            identifier=identifier
        ).set(remaining)
    
    @staticmethod
    def time_rate_limit_check(algorithm: str):
        """Context manager for timing rate limit checks."""
        return rate_limit_check_duration.labels(algorithm=algorithm).time()
    
    @staticmethod
    def time_redis_operation(operation: str):
        """Context manager for timing Redis operations."""
        return redis_operation_duration.labels(operation=operation).time()
    
    @staticmethod
    def update_config_info(config: Dict[str, any]):
        """Update rate limiting configuration info."""
        rate_limit_config_info.info({
            'redis_enabled': str(config.get('redis_enabled', False)),
            'default_algorithm': config.get('default_algorithm', 'sliding_window'),
            'ddos_protection_enabled': str(config.get('ddos_protection_enabled', True)),
            'environment': config.get('environment', 'unknown'),
        })


class RateLimitMetricsMiddleware:
    """Middleware to automatically collect rate limiting metrics."""
    
    def __init__(self, app):
        self.app = app
        self.metrics = RateLimitingMetrics()
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope["path"]
            method = scope["method"]
            
            # Start timing
            start_time = time.time()
            
            # Track response status
            status_code = None
            
            async def send_wrapper(message):
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                    
                    # Check for rate limit headers
                    headers = dict(message.get("headers", []))
                    if b"x-ratelimit-remaining" in headers:
                        remaining = int(headers[b"x-ratelimit-remaining"])
                        # Extract identifier from request somehow
                        identifier = "unknown"  # This would need proper implementation
                        self.metrics.update_remaining_requests(path, identifier, remaining)
                
                await send(message)
            
            try:
                await self.app(scope, receive, send_wrapper)
                
                # Record metrics based on status
                if status_code == 429:
                    self.metrics.record_request(path, method, "limited")
                    self.metrics.record_violation(path, "unknown", "rate_limit_exceeded")
                elif status_code == 403:
                    self.metrics.record_request(path, method, "blocked")
                else:
                    self.metrics.record_request(path, method, "allowed")
                    
            except Exception as e:
                self.metrics.record_request(path, method, "error")
                raise
        else:
            await self.app(scope, receive, send)


# Export metrics instance
rate_limiting_metrics = RateLimitingMetrics()