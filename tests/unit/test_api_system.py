"""
Test suite for System API endpoints.

Tests the FastAPI system endpoints including metrics and health checks.
"""

import os

os.environ["REDIS_ENABLED"] = "false"
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite:///./test.db"


from main import app
from tests.test_client_compat import TestClient

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

    def test_system_info(self):
        """Test system info endpoint."""
        response = client.get("/api/v1/system/info")

        assert response.status_code == 200
        data = response.json()

        # Check main structure
        assert "platform" in data
        assert "version" in data
        assert "environment" in data
        assert "capabilities" in data
        assert "hardware" in data

        # Check capabilities structure
        capabilities = data["capabilities"]
        assert "active_inference" in capabilities
        assert "graph_neural_networks" in capabilities
        assert "coalition_formation" in capabilities
        assert "llm_integration" in capabilities

        # Check hardware structure
        hardware = data["hardware"]
        assert "cpu_cores" in hardware
        assert "total_memory" in hardware
        assert "python_version" in hardware
        assert "cuda_available" in hardware

    def test_recent_logs(self):
        """Test recent logs endpoint."""
        response = client.get("/api/v1/system/logs/recent")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

        # Test with limit parameter
        response = client.get("/api/v1/system/logs/recent?limit=10")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 10

        # Check log structure
        if len(data) > 0:
            log = data[0]
            assert "timestamp" in log
            assert "level" in log
            assert "service" in log
            assert "message" in log

    def test_prometheus_metrics(self):
        """Test Prometheus metrics endpoint."""
        response = client.get("/api/v1/system/metrics/prometheus")

        # Should return 503 since prometheus is not available in test env
        assert response.status_code == 503
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
        assert "Prometheus metrics not available" in response.text

    def test_health_metrics(self):
        """Test health metrics endpoint."""
        response = client.get("/api/v1/system/metrics/health")

        assert response.status_code == 200
        data = response.json()

        # Check required fields
        assert "timestamp" in data
        assert "status" in data

        # In test environment, prometheus is not available
        # so status should be degraded
        assert data["status"] in ["degraded", "unhealthy"]
        assert "error" in data or "metrics" in data
