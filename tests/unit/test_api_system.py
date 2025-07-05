"""
Test suite for System API endpoints.

Tests the FastAPI system endpoints including metrics and health checks.
"""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from api.v1.system import router
from main import app

# Create test client
client = TestClient(app)


class TestSystemMetrics:
    """Test system metrics endpoints."""

    def test_get_metrics(self):
        """Test getting system metrics."""
        response = client.get("/api/v1/system/metrics")

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "timestamp" in data
        assert "cpu_usage" in data
        assert "memory_usage" in data
        assert "active_agents" in data
        assert "total_inferences" in data
        assert "avg_response_time" in data
        assert "api_calls_per_minute" in data

        # Check data types
        assert isinstance(data["cpu_usage"], (int, float))
        assert isinstance(data["memory_usage"], (int, float))
        assert isinstance(data["active_agents"], int)
        assert isinstance(data["total_inferences"], int)
        assert isinstance(data["avg_response_time"], (int, float))
        assert isinstance(data["api_calls_per_minute"], int)


class TestSystemHealth:
    """Test system health endpoints."""

    def test_health_check_services(self):
        """Test health check services endpoint."""
        response = client.get("/api/v1/system/health/services")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

        # Check that we have some services reported
        assert len(data) > 0

        # Check structure of services
        for service in data:
            assert "service" in service
            assert "status" in service
            assert "last_check" in service
            assert "details" in service

            # Check valid status values
            assert service["status"] in ["healthy", "degraded", "unhealthy"]

            # Check service names are not empty
            assert len(service["service"]) > 0
