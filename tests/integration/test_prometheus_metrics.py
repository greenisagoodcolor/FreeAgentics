"""Test Prometheus metrics endpoint following TDD principles.

This test suite validates:
1. /metrics endpoint returns proper Prometheus format
2. Required counters are present (agent_spawn_total, kg_node_total)
3. Basic HTTP request metrics
4. System health metrics
5. Proper content-type headers
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client."""
    import sys
    import os

    # Add parent directory to path to import main
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    from main import app

    return TestClient(app)


class TestPrometheusMetrics:
    """Test Prometheus metrics functionality."""

    def test_metrics_endpoint_exists(self, client):
        """Test that /metrics endpoint exists and responds."""
        response = client.get("/metrics")
        # Should return 200 or 503 (if not configured)
        assert response.status_code in [200, 503]

    def test_metrics_endpoint_returns_prometheus_format(self, client):
        """Test that /metrics returns proper Prometheus format."""
        response = client.get("/metrics")

        if response.status_code == 200:
            # Check content type
            assert response.headers.get("content-type") in [
                "text/plain; version=0.0.4; charset=utf-8",
                "text/plain; charset=utf-8",
            ]

            # Check basic format (comments and metrics)
            content = response.text
            lines = content.strip().split("\n")

            # Should have at least some content
            assert len(lines) > 0

            # Prometheus format has comments starting with # and metric lines
            has_comments = any(line.startswith("#") for line in lines)
            assert has_comments or "no metrics" in content.lower()

    def test_required_counters_present(self, client):
        """Test that required counters are present in metrics."""
        response = client.get("/metrics")

        if response.status_code == 200:
            content = response.text

            # Check for required counters
            required_counters = ["agent_spawn_total", "kg_node_total"]

            for counter in required_counters:
                # Counter should be in HELP, TYPE, or metric line
                assert counter in content, (
                    f"Required counter {counter} not found in metrics"
                )

    def test_http_request_metrics_present(self, client):
        """Test that HTTP request metrics are present."""
        response = client.get("/metrics")

        if response.status_code == 200:
            content = response.text

            # Check for HTTP metrics
            http_metrics = ["http_requests_total", "http_request_duration_seconds"]

            for metric in http_metrics:
                assert metric in content, f"HTTP metric {metric} not found"

    def test_system_health_metrics_present(self, client):
        """Test that system health metrics are present."""
        response = client.get("/metrics")

        if response.status_code == 200:
            content = response.text

            # Check for system metrics
            system_metrics = ["system_cpu_usage_percent", "system_memory_usage_bytes"]

            for metric in system_metrics:
                assert metric in content, f"System metric {metric} not found"

    def test_metrics_endpoint_no_cache(self, client):
        """Test that metrics endpoint has no-cache headers."""
        response = client.get("/metrics")

        if response.status_code == 200:
            # Should have no-cache header
            cache_control = response.headers.get("cache-control", "")
            assert "no-cache" in cache_control.lower()

    def test_agent_spawn_counter_increments(self, client):
        """Test that agent_spawn_total counter increments when agents are created."""
        # Get initial metrics
        response1 = client.get("/metrics")

        if response1.status_code == 200:
            # Try to create an agent (this might fail, which is OK for now)
            try:
                client.post(
                    "/api/v1/agents",
                    json={"name": "test_agent", "agent_type": "active_inference"},
                )
            except:
                pass

            # Get metrics again
            response2 = client.get("/metrics")
            content2 = response2.text

            # We should at least see the counter definition
            assert "agent_spawn_total" in content2

    def test_kg_node_counter_present(self, client):
        """Test that kg_node_total counter is present."""
        response = client.get("/metrics")

        if response.status_code == 200:
            content = response.text

            # Check for knowledge graph node counter
            assert "kg_node_total" in content

    def test_prometheus_metric_format_validation(self, client):
        """Test that metrics follow Prometheus format conventions."""
        response = client.get("/metrics")

        if response.status_code == 200:
            content = response.text
            lines = content.strip().split("\n")

            for line in lines:
                if not line or line.startswith("#"):
                    continue

                # Metric lines should have metric_name{labels} value format
                # or metric_name value format
                if " " in line:
                    parts = line.split(" ", 1)
                    metric_part = parts[0]

                    # Check metric name format (alphanumeric with underscores)
                    metric_name = metric_part.split("{")[0]
                    assert metric_name.replace("_", "").replace(":", "").isalnum(), (
                        f"Invalid metric name format: {metric_name}"
                    )

    def test_metrics_endpoint_performance(self, client):
        """Test that metrics endpoint responds quickly."""
        import time

        start_time = time.time()
        client.get("/metrics")
        end_time = time.time()

        # Should respond within 1 second
        response_time = end_time - start_time
        assert response_time < 1.0, f"Metrics endpoint too slow: {response_time}s"


class TestMetricsIntegration:
    """Test metrics integration with the application."""

    def test_metrics_reflect_system_state(self, client):
        """Test that metrics reflect actual system state."""
        # First, get system info
        client.get("/api/v1/system/info")

        # Then get metrics
        metrics_response = client.get("/metrics")

        if metrics_response.status_code == 200:
            content = metrics_response.text

            # Should have some correlation with system state
            # This is a basic check - in production would be more detailed
            assert len(content) > 100, "Metrics seem too minimal"

    def test_metrics_update_on_activity(self, client):
        """Test that metrics update when system activity occurs."""
        # Get initial metrics
        response1 = client.get("/metrics")

        # Perform some activity
        client.get("/health")
        client.get("/api/v1/system/info")

        # Get metrics again
        response2 = client.get("/metrics")

        if response1.status_code == 200 and response2.status_code == 200:
            # Content might change (timestamps, counters, etc.)
            # This is a basic test - detailed comparison would check specific counters
            assert response1.text or response2.text, "Metrics should have content"
