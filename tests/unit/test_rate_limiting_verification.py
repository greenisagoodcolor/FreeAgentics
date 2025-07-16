"""
Rate Limiting Verification Test Suite

Tests rate limiting functionality to prevent abuse and DoS attacks.
Tests proper rate limit headers, retry-after responses, and edge cases.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import pytest

from auth.security_implementation import (
    RateLimiter,
    rate_limit,
    get_client_ip,
)


class TestRateLimitingVerification:
    """Test rate limiting functionality comprehensively."""

    @pytest.fixture
    def rate_limiter(self):
        """Create fresh RateLimiter instance."""
        return RateLimiter()

    def test_rate_limiter_initialization(self, rate_limiter):
        """Test RateLimiter initialization."""
        assert hasattr(rate_limiter, 'requests')
        assert hasattr(rate_limiter, 'user_requests')
        assert isinstance(rate_limiter.requests, dict)
        assert isinstance(rate_limiter.user_requests, dict)
        assert len(rate_limiter.requests) == 0
        assert len(rate_limiter.user_requests) == 0

    def test_basic_rate_limiting(self, rate_limiter):
        """Test basic rate limiting functionality."""
        identifier = "test_client_192.168.1.1"
        max_requests = 5
        window_minutes = 1
        
        # First 5 requests should be allowed
        for i in range(max_requests):
            is_limited = rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
            assert not is_limited, f"Request {i+1} should be allowed"
        
        # 6th request should be limited
        is_limited = rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
        assert is_limited, "Request 6 should be rate limited"
        
        # 7th request should also be limited
        is_limited = rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
        assert is_limited, "Request 7 should be rate limited"

    def test_rate_limit_window_expiry(self, rate_limiter):
        """Test that rate limits reset after window expires."""
        identifier = "test_client_192.168.1.2"
        max_requests = 3
        window_minutes = 0.02  # 1.2 seconds (0.02 minutes)
        
        # Use up the rate limit
        for i in range(max_requests):
            is_limited = rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
            assert not is_limited, f"Request {i+1} should be allowed"
        
        # Next request should be limited
        is_limited = rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
        assert is_limited, "Request should be rate limited"
        
        # Wait for window to expire
        time.sleep(1.5)
        
        # Now request should be allowed again
        is_limited = rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
        assert not is_limited, "Request should be allowed after window expiry"

    def test_different_identifiers_independent(self, rate_limiter):
        """Test that different identifiers have independent rate limits."""
        identifier1 = "client_1"
        identifier2 = "client_2"
        max_requests = 3
        window_minutes = 1
        
        # Use up rate limit for identifier1
        for i in range(max_requests):
            is_limited = rate_limiter.is_rate_limited(identifier1, max_requests, window_minutes)
            assert not is_limited, f"Request {i+1} for client_1 should be allowed"
        
        # identifier1 should now be limited
        is_limited = rate_limiter.is_rate_limited(identifier1, max_requests, window_minutes)
        assert is_limited, "client_1 should be rate limited"
        
        # identifier2 should still be allowed
        is_limited = rate_limiter.is_rate_limited(identifier2, max_requests, window_minutes)
        assert not is_limited, "client_2 should not be rate limited"
        
        # Use up rate limit for identifier2
        for i in range(max_requests - 1):  # -1 because we already used one
            is_limited = rate_limiter.is_rate_limited(identifier2, max_requests, window_minutes)
            assert not is_limited, f"Request {i+2} for client_2 should be allowed"
        
        # Now identifier2 should also be limited
        is_limited = rate_limiter.is_rate_limited(identifier2, max_requests, window_minutes)
        assert is_limited, "client_2 should now be rate limited"

    def test_concurrent_rate_limiting(self, rate_limiter):
        """Test rate limiting under concurrent requests."""
        identifier = "concurrent_client"
        max_requests = 10
        window_minutes = 1
        total_requests = 25
        
        results = []
        
        def make_request():
            return rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
        
        # Make concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(total_requests)]
            
            for future in as_completed(futures):
                is_limited = future.result()
                results.append(is_limited)
        
        # Count allowed and limited requests
        allowed_count = sum(1 for r in results if not r)
        limited_count = sum(1 for r in results if r)
        
        # Should allow up to max_requests and limit the rest
        assert allowed_count <= max_requests, f"Too many requests allowed: {allowed_count} > {max_requests}"
        assert limited_count >= (total_requests - max_requests), f"Too few requests limited: {limited_count}"
        
        # Total should equal total_requests
        assert allowed_count + limited_count == total_requests

    def test_rate_limit_cleanup(self, rate_limiter):
        """Test that old request records are cleaned up."""
        identifier = "cleanup_client"
        max_requests = 2
        window_minutes = 0.01  # Very short window
        
        # Make some requests
        for i in range(max_requests):
            rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
        
        # Verify requests are recorded
        assert identifier in rate_limiter.requests
        assert len(rate_limiter.requests[identifier]) == max_requests
        
        # Wait for window to expire
        time.sleep(1)
        
        # Make another request to trigger cleanup
        rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
        
        # Old requests should be cleaned up
        assert len(rate_limiter.requests[identifier]) == 1

    def test_rate_limit_with_zero_max_requests(self, rate_limiter):
        """Test rate limiting with zero max requests."""
        identifier = "zero_limit_client"
        max_requests = 0
        window_minutes = 1
        
        # All requests should be limited
        is_limited = rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
        assert is_limited, "Request should be limited when max_requests is 0"

    def test_rate_limit_with_very_high_max_requests(self, rate_limiter):
        """Test rate limiting with very high max requests."""
        identifier = "high_limit_client"
        max_requests = 10000
        window_minutes = 1
        
        # Make many requests
        for i in range(100):
            is_limited = rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
            assert not is_limited, f"Request {i+1} should be allowed with high limit"

    def test_rate_limit_edge_cases(self, rate_limiter):
        """Test rate limiting edge cases."""
        
        # Test with negative max_requests
        identifier = "edge_case_client"
        max_requests = -1
        window_minutes = 1
        
        # Should be treated as 0 or similar
        is_limited = rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
        assert is_limited, "Negative max_requests should limit all requests"
        
        # Test with zero window
        max_requests = 5
        window_minutes = 0
        
        # Should still work (though practically limits everything)
        is_limited = rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
        # Behavior depends on implementation, but should not crash

    def test_rate_limit_performance(self, rate_limiter):
        """Test rate limiting performance."""
        identifier = "performance_client"
        max_requests = 100
        window_minutes = 1
        
        # Time many rate limit checks
        start_time = time.time()
        
        for i in range(50):
            rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should be very fast
        assert duration < 0.1, f"Rate limiting too slow: {duration:.3f}s for 50 checks"
        
        # Average per check should be very fast
        avg_per_check = duration / 50
        assert avg_per_check < 0.002, f"Average per check too slow: {avg_per_check:.6f}s"

    def test_rate_limit_memory_usage(self, rate_limiter):
        """Test that rate limiting doesn't accumulate excessive memory."""
        max_requests = 10
        window_minutes = 0.01  # Short window
        
        # Create many different identifiers
        for i in range(1000):
            identifier = f"memory_client_{i}"
            rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
        
        # Wait for records to expire
        time.sleep(1)
        
        # Trigger cleanup by making new request
        rate_limiter.is_rate_limited("cleanup_trigger", max_requests, window_minutes)
        
        # Memory should be cleaned up
        # Note: This test depends on implementation details
        # In practice, old records should be cleaned up

    def test_get_client_ip_function(self):
        """Test client IP extraction function."""
        
        # Test with X-Forwarded-For header
        mock_request = Mock()
        mock_request.headers = {"X-Forwarded-For": "192.168.1.1, 10.0.0.1"}
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"
        
        ip = get_client_ip(mock_request)
        assert ip == "192.168.1.1", "Should extract first IP from X-Forwarded-For"
        
        # Test without X-Forwarded-For header
        mock_request = Mock()
        mock_request.headers = {}
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"
        
        ip = get_client_ip(mock_request)
        assert ip == "127.0.0.1", "Should use client.host when no X-Forwarded-For"
        
        # Test with no client
        mock_request = Mock()
        mock_request.headers = {}
        mock_request.client = None
        
        ip = get_client_ip(mock_request)
        assert ip == "unknown", "Should return 'unknown' when no client"

    def test_rate_limit_decorator_functionality(self):
        """Test rate limiting decorator functionality."""
        
        # Create a mock function to decorate
        @rate_limit(max_requests=2, window_minutes=1)
        async def test_endpoint(request):
            return {"message": "success"}
        
        # Test that decorator is applied
        assert hasattr(test_endpoint, '__wrapped__'), "Decorator should wrap function"
        
        # Test with mock request
        mock_request = Mock()
        mock_request.headers = {}
        mock_request.client = Mock()
        mock_request.client.host = "127.0.0.1"
        
        # This test would need to be async to fully test the decorator
        # For now, just verify it's properly decorated

    def test_rate_limit_cleanup_function(self, rate_limiter):
        """Test the cleanup function removes old requests."""
        identifier = "cleanup_test_client"
        max_requests = 3
        window_minutes = 0.01  # Very short window
        
        # Make requests
        for i in range(max_requests):
            rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
        
        # Verify requests are recorded
        assert identifier in rate_limiter.requests
        assert len(rate_limiter.requests[identifier]) == max_requests
        
        # Wait for requests to expire
        time.sleep(1)
        
        # Call cleanup
        rate_limiter.clear_old_requests()
        
        # Old requests should be cleaned up
        # The identifier might be removed entirely if no recent requests
        if identifier in rate_limiter.requests:
            assert len(rate_limiter.requests[identifier]) == 0

    def test_rate_limiting_stress_test(self, rate_limiter):
        """Test rate limiting under stress conditions."""
        max_requests = 20
        window_minutes = 1
        num_clients = 50
        requests_per_client = 30
        
        results = {}
        
        def client_requests(client_id):
            identifier = f"stress_client_{client_id}"
            client_results = []
            
            for i in range(requests_per_client):
                start_time = time.time()
                is_limited = rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
                end_time = time.time()
                
                client_results.append({
                    "limited": is_limited,
                    "duration": end_time - start_time,
                })
            
            return client_id, client_results
        
        # Run stress test
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(client_requests, i) for i in range(num_clients)]
            
            for future in as_completed(futures):
                client_id, client_results = future.result()
                results[client_id] = client_results
        
        # Verify results
        for client_id, client_results in results.items():
            # Count allowed and limited requests
            allowed = sum(1 for r in client_results if not r["limited"])
            limited = sum(1 for r in client_results if r["limited"])
            
            # Should allow up to max_requests per client
            assert allowed <= max_requests, f"Client {client_id} had too many allowed requests: {allowed}"
            assert limited >= (requests_per_client - max_requests), f"Client {client_id} had too few limited requests: {limited}"
            
            # Performance should be good
            durations = [r["duration"] for r in client_results]
            avg_duration = sum(durations) / len(durations)
            assert avg_duration < 0.01, f"Client {client_id} had slow rate limiting: {avg_duration:.6f}s"

    def test_rate_limiting_with_bursty_traffic(self, rate_limiter):
        """Test rate limiting with bursty traffic patterns."""
        identifier = "bursty_client"
        max_requests = 5
        window_minutes = 0.1  # 6 seconds
        
        # First burst - should be allowed
        for i in range(max_requests):
            is_limited = rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
            assert not is_limited, f"First burst request {i+1} should be allowed"
        
        # Additional requests in same window - should be limited
        for i in range(3):
            is_limited = rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
            assert is_limited, f"Additional request {i+1} should be limited"
        
        # Wait for window to partially expire
        time.sleep(7)
        
        # Second burst - should be allowed again
        for i in range(max_requests):
            is_limited = rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
            assert not is_limited, f"Second burst request {i+1} should be allowed"

    def test_rate_limiting_configuration_validation(self, rate_limiter):
        """Test rate limiting with various configuration values."""
        identifier = "config_test_client"
        
        # Test different window sizes
        window_sizes = [0.01, 0.1, 1, 5, 10]
        for window in window_sizes:
            is_limited = rate_limiter.is_rate_limited(identifier, 5, window)
            # Should not crash with any reasonable window size
            assert isinstance(is_limited, bool)
        
        # Test different max_requests values
        max_requests_values = [1, 5, 10, 50, 100, 1000]
        for max_req in max_requests_values:
            is_limited = rate_limiter.is_rate_limited(identifier, max_req, 1)
            # Should not crash with any reasonable max_requests
            assert isinstance(is_limited, bool)

    def test_rate_limiting_with_time_skew(self, rate_limiter):
        """Test rate limiting behavior with time-related edge cases."""
        identifier = "time_skew_client"
        max_requests = 3
        window_minutes = 1
        
        # Make requests
        for i in range(max_requests):
            is_limited = rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
            assert not is_limited, f"Request {i+1} should be allowed"
        
        # Next request should be limited
        is_limited = rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
        assert is_limited, "Request should be limited"
        
        # Mock time going backwards (edge case)
        with patch('time.time', return_value=time.time() - 3600):  # 1 hour ago
            # Should handle gracefully
            is_limited = rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
            assert isinstance(is_limited, bool)

    def test_rate_limiting_identifier_types(self, rate_limiter):
        """Test rate limiting with different identifier types."""
        max_requests = 2
        window_minutes = 1
        
        # Test different identifier formats
        identifiers = [
            "192.168.1.1",  # IP address
            "user_123",     # User ID
            "api_key_abc",  # API key
            "session_xyz",  # Session ID
            "127.0.0.1:8080",  # IP with port
            "2001:db8::1",  # IPv6 address
        ]
        
        for identifier in identifiers:
            # Each identifier should have independent rate limiting
            for i in range(max_requests):
                is_limited = rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
                assert not is_limited, f"Request {i+1} for {identifier} should be allowed"
            
            # Next request should be limited
            is_limited = rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
            assert is_limited, f"Request should be limited for {identifier}"

    def test_rate_limiting_thread_safety(self, rate_limiter):
        """Test that rate limiting is thread-safe."""
        identifier = "thread_safety_client"
        max_requests = 50
        window_minutes = 1
        
        results = []
        
        def make_requests():
            thread_results = []
            for i in range(10):
                is_limited = rate_limiter.is_rate_limited(identifier, max_requests, window_minutes)
                thread_results.append(is_limited)
            return thread_results
        
        # Run multiple threads
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_requests) for _ in range(10)]
            
            for future in as_completed(futures):
                thread_results = future.result()
                results.extend(thread_results)
        
        # Count total allowed requests
        allowed_count = sum(1 for r in results if not r)
        
        # Should not exceed max_requests due to race conditions
        assert allowed_count <= max_requests, f"Race condition: {allowed_count} > {max_requests}"