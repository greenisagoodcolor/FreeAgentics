"""Test metrics endpoint returns prometheus-style metrics."""

from fastapi.testclient import TestClient

from api.main import app


class TestMetricsEndpoint:
    """Test the /metrics endpoint functionality."""

    def test_metrics_endpoint_returns_200(self):
        """Test that /metrics endpoint returns 200 OK."""
        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_endpoint_returns_plain_text(self):
        """Test that /metrics endpoint returns plain text prometheus format."""
        client = TestClient(app)
        response = client.get("/metrics")

        # Check content type is prometheus text format
        content_type = response.headers.get("content-type", "")
        assert "text/plain" in content_type or "prometheus" in content_type

        # Check response body is text
        assert isinstance(response.text, str)
        assert len(response.text) > 0

    def test_metrics_endpoint_contains_counters(self):
        """Test that /metrics endpoint contains expected counter metrics."""
        client = TestClient(app)
        response = client.get("/metrics")

        # Check for required counters per CLAUDE.md requirements
        assert "agent_spawn_total" in response.text
        assert "kg_node_total" in response.text

        # Check prometheus format (HELP and TYPE lines)
        assert "# HELP" in response.text
        assert "# TYPE" in response.text

    def test_metrics_endpoint_no_cache(self):
        """Test that /metrics endpoint has no-cache headers."""
        client = TestClient(app)
        response = client.get("/metrics")

        cache_control = response.headers.get("cache-control", "")
        assert "no-cache" in cache_control
        assert "no-store" in cache_control
        assert "must-revalidate" in cache_control
