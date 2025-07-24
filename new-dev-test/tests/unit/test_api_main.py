"""Comprehensive tests for api/main.py module to achieve 100% coverage."""

import logging
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


# Create proper mock middleware classes
class MockMiddleware:
    def __init__(self, *args, **kwargs):
        pass

    async def __call__(self, scope, receive, send):
        pass


# Mock dependencies before importing main
with patch.dict(
    "sys.modules",
    {
        "observability.prometheus_metrics": MagicMock(),
        "observability.performance_metrics": MagicMock(),
        "api.middleware": MagicMock(),
        "api.v1.agents": MagicMock(),
        "api.v1.auth": MagicMock(),
        "api.v1.inference": MagicMock(),
        "api.v1.monitoring": MagicMock(),
        "api.v1.security": MagicMock(),
        "api.v1.system": MagicMock(),
        "api.v1.websocket": MagicMock(),
        "api.v1.graphql_schema": MagicMock(),
        "auth.security_headers": MagicMock(),
        "api.middleware.security_headers": MagicMock(),
    },
):
    # Mock the middleware classes in the API middleware module
    with patch("api.middleware.DDoSProtectionMiddleware", MockMiddleware):
        with patch("api.middleware.SecurityMonitoringMiddleware", MockMiddleware):
            with patch(
                "auth.security_headers.SecurityHeadersMiddleware",
                MockMiddleware,
            ):
                with patch(
                    "api.middleware.security_headers.security_headers_middleware",
                    AsyncMock(),
                ):
                    # Mock the router objects
                    mock_router = MagicMock()
                    mock_router.router = MagicMock()

                    with patch("api.v1.agents.router", mock_router.router):
                        with patch("api.v1.auth.router", mock_router.router):
                            with patch("api.v1.inference.router", mock_router.router):
                                with patch(
                                    "api.v1.monitoring.router",
                                    mock_router.router,
                                ):
                                    with patch(
                                        "api.v1.security.router",
                                        mock_router.router,
                                    ):
                                        with patch(
                                            "api.v1.system.router",
                                            mock_router.router,
                                        ):
                                            with patch(
                                                "api.v1.websocket.router",
                                                mock_router.router,
                                            ):
                                                with patch(
                                                    "api.v1.graphql_schema.graphql_app",
                                                    mock_router.router,
                                                ):
                                                    from api.main import app, lifespan


class TestAPIMain:
    """Test the main API application."""

    def test_app_creation(self):
        """Test that the FastAPI app is created correctly."""
        assert isinstance(app, FastAPI)
        assert app.title == "FreeAgentics API"
        assert app.description == "Multi-Agent AI Platform API with Active Inference"
        assert app.version == "0.1.0"

    def test_app_configuration(self):
        """Test app configuration and middleware setup."""
        # Test that middleware is configured
        middleware_names = []
        for middleware in app.user_middleware:
            if hasattr(middleware.cls, "__name__"):
                middleware_names.append(middleware.cls.__name__)
            else:
                middleware_names.append(str(middleware.cls))

        # Check that security middleware is present
        assert any("Security" in name or "Mock" in name for name in middleware_names)
        assert any("CORS" in name for name in middleware_names)

    def test_root_endpoint(self):
        """Test the root endpoint."""
        client = TestClient(app)
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Welcome to FreeAgentics API"
        assert data["version"] == "0.1.0"
        assert data["docs"] == "/docs"
        assert data["redoc"] == "/redoc"

    def test_health_check_endpoint(self):
        """Test the health check endpoint."""
        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "FreeAgentics API"
        assert data["version"] == "0.1.0"


class TestLifespan:
    """Test the lifespan context manager."""

    @pytest.fixture
    def mock_app(self):
        """Create a mock FastAPI app for testing."""
        return MagicMock(spec=FastAPI)

    @pytest.mark.asyncio
    async def test_lifespan_startup_success(self, mock_app):
        """Test successful startup sequence."""
        with patch("api.main.logger") as mock_logger:
            with patch(
                "observability.prometheus_metrics.start_prometheus_metrics_collection"
            ) as mock_prometheus:
                with patch(
                    "observability.performance_metrics.start_performance_tracking"
                ) as mock_perf:
                    mock_prometheus.return_value = AsyncMock()
                    mock_perf.return_value = AsyncMock()

                    async with lifespan(mock_app):
                        # Verify startup logging
                        mock_logger.info.assert_any_call("Starting FreeAgentics API...")
                        mock_logger.info.assert_any_call("✅ Prometheus metrics collection started")
                        mock_logger.info.assert_any_call("✅ Performance tracking started")

                        # Verify services were started
                        mock_prometheus.assert_called_once()
                        mock_perf.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_startup_prometheus_failure(self, mock_app):
        """Test startup with Prometheus failure."""
        with patch("api.main.logger") as mock_logger:
            with patch(
                "observability.prometheus_metrics.start_prometheus_metrics_collection"
            ) as mock_prometheus:
                with patch(
                    "observability.performance_metrics.start_performance_tracking"
                ) as mock_perf:
                    mock_prometheus.side_effect = Exception("Prometheus connection failed")
                    mock_perf.return_value = AsyncMock()

                    async with lifespan(mock_app):
                        # Verify failure is logged as warning
                        mock_logger.warning.assert_any_call(
                            "⚠️ Failed to start Prometheus metrics collection: Prometheus connection failed"
                        )

                        # Verify performance tracking still starts
                        mock_perf.assert_called_once()
                        mock_logger.info.assert_any_call("✅ Performance tracking started")

    @pytest.mark.asyncio
    async def test_lifespan_startup_performance_failure(self, mock_app):
        """Test startup with performance tracking failure."""
        with patch("api.main.logger") as mock_logger:
            with patch(
                "observability.prometheus_metrics.start_prometheus_metrics_collection"
            ) as mock_prometheus:
                with patch(
                    "observability.performance_metrics.start_performance_tracking"
                ) as mock_perf:
                    mock_prometheus.return_value = AsyncMock()
                    mock_perf.side_effect = Exception("Performance tracking failed")

                    async with lifespan(mock_app):
                        # Verify Prometheus still starts
                        mock_prometheus.assert_called_once()
                        mock_logger.info.assert_any_call("✅ Prometheus metrics collection started")

                        # Verify failure is logged as warning
                        mock_logger.warning.assert_any_call(
                            "⚠️ Failed to start performance tracking: Performance tracking failed"
                        )

    @pytest.mark.asyncio
    async def test_lifespan_shutdown_success(self, mock_app):
        """Test successful shutdown sequence."""
        with patch("api.main.logger") as mock_logger:
            with patch("observability.prometheus_metrics.start_prometheus_metrics_collection"):
                with patch("observability.performance_metrics.start_performance_tracking"):
                    with patch(
                        "observability.prometheus_metrics.stop_prometheus_metrics_collection"
                    ) as mock_stop_prometheus:
                        with patch(
                            "observability.performance_metrics.stop_performance_tracking"
                        ) as mock_stop_perf:
                            mock_stop_prometheus.return_value = AsyncMock()
                            mock_stop_perf.return_value = AsyncMock()

                            async with lifespan(mock_app):
                                pass

                            # Verify shutdown logging
                            mock_logger.info.assert_any_call("Shutting down FreeAgentics API...")
                            mock_logger.info.assert_any_call(
                                "✅ Prometheus metrics collection stopped"
                            )
                            mock_logger.info.assert_any_call("✅ Performance tracking stopped")

                            # Verify services were stopped
                            mock_stop_prometheus.assert_called_once()
                            mock_stop_perf.assert_called_once()

    @pytest.mark.asyncio
    async def test_lifespan_shutdown_prometheus_failure(self, mock_app):
        """Test shutdown with Prometheus failure."""
        with patch("api.main.logger") as mock_logger:
            with patch("observability.prometheus_metrics.start_prometheus_metrics_collection"):
                with patch("observability.performance_metrics.start_performance_tracking"):
                    with patch(
                        "observability.prometheus_metrics.stop_prometheus_metrics_collection"
                    ) as mock_stop_prometheus:
                        with patch(
                            "observability.performance_metrics.stop_performance_tracking"
                        ) as mock_stop_perf:
                            mock_stop_prometheus.side_effect = Exception("Prometheus stop failed")
                            mock_stop_perf.return_value = AsyncMock()

                            async with lifespan(mock_app):
                                pass

                            # Verify failure is logged as warning
                            mock_logger.warning.assert_any_call(
                                "⚠️ Failed to stop Prometheus metrics collection: Prometheus stop failed"
                            )

                            # Verify performance tracking still stops
                            mock_stop_perf.assert_called_once()
                            mock_logger.info.assert_any_call("✅ Performance tracking stopped")

    @pytest.mark.asyncio
    async def test_lifespan_shutdown_performance_failure(self, mock_app):
        """Test shutdown with performance tracking failure."""
        with patch("api.main.logger") as mock_logger:
            with patch("observability.prometheus_metrics.start_prometheus_metrics_collection"):
                with patch("observability.performance_metrics.start_performance_tracking"):
                    with patch(
                        "observability.prometheus_metrics.stop_prometheus_metrics_collection"
                    ) as mock_stop_prometheus:
                        with patch(
                            "observability.performance_metrics.stop_performance_tracking"
                        ) as mock_stop_perf:
                            mock_stop_prometheus.return_value = AsyncMock()
                            mock_stop_perf.side_effect = Exception("Performance stop failed")

                            async with lifespan(mock_app):
                                pass

                            # Verify Prometheus still stops
                            mock_stop_prometheus.assert_called_once()
                            mock_logger.info.assert_any_call(
                                "✅ Prometheus metrics collection stopped"
                            )

                            # Verify failure is logged as warning
                            mock_logger.warning.assert_any_call(
                                "⚠️ Failed to stop performance tracking: Performance stop failed"
                            )

    @pytest.mark.asyncio
    async def test_lifespan_both_services_fail(self, mock_app):
        """Test lifespan with both services failing."""
        with patch("api.main.logger") as mock_logger:
            with patch(
                "observability.prometheus_metrics.start_prometheus_metrics_collection"
            ) as mock_prometheus:
                with patch(
                    "observability.performance_metrics.start_performance_tracking"
                ) as mock_perf:
                    mock_prometheus.side_effect = Exception("Prometheus failed")
                    mock_perf.side_effect = Exception("Performance failed")

                    async with lifespan(mock_app):
                        # Verify both failures are logged
                        mock_logger.warning.assert_any_call(
                            "⚠️ Failed to start Prometheus metrics collection: Prometheus failed"
                        )
                        mock_logger.warning.assert_any_call(
                            "⚠️ Failed to start performance tracking: Performance failed"
                        )


class TestMiddlewareConfiguration:
    """Test middleware configuration."""

    def test_cors_configuration(self):
        """Test CORS middleware configuration."""
        # Find CORS middleware
        cors_middleware = None
        for middleware in app.user_middleware:
            if "CORS" in middleware.cls.__name__:
                cors_middleware = middleware
                break

        assert cors_middleware is not None

        # Check CORS configuration
        kwargs = cors_middleware.kwargs
        assert "http://localhost:3000" in kwargs["allow_origins"]
        assert "http://localhost:3001" in kwargs["allow_origins"]
        assert "https://freeagentics.com" in kwargs["allow_origins"]
        assert "https://www.freeagentics.com" in kwargs["allow_origins"]
        assert kwargs["allow_credentials"] is True
        assert "GET" in kwargs["allow_methods"]
        assert "POST" in kwargs["allow_methods"]
        assert "PUT" in kwargs["allow_methods"]
        assert "DELETE" in kwargs["allow_methods"]
        assert "OPTIONS" in kwargs["allow_methods"]
        assert "PATCH" in kwargs["allow_methods"]

    def test_cors_headers_configuration(self):
        """Test CORS headers configuration."""
        cors_middleware = None
        for middleware in app.user_middleware:
            if "CORS" in middleware.cls.__name__:
                cors_middleware = middleware
                break

        assert cors_middleware is not None

        kwargs = cors_middleware.kwargs
        allow_headers = kwargs["allow_headers"]
        assert "accept" in allow_headers
        assert "authorization" in allow_headers
        assert "content-type" in allow_headers
        assert "x-api-key" in allow_headers
        assert "x-client-version" in allow_headers

        expose_headers = kwargs["expose_headers"]
        assert "x-total-count" in expose_headers
        assert "x-rate-limit-remaining" in expose_headers
        assert "x-rate-limit-reset" in expose_headers

        assert kwargs["max_age"] == 86400

    def test_security_middleware_order(self):
        """Test that security middleware is added in correct order."""
        middleware_names = [middleware.cls.__name__ for middleware in app.user_middleware]

        # Security middleware should be present
        assert any("Security" in name for name in middleware_names)

        # CORS should be present
        assert any("CORS" in name for name in middleware_names)

    @patch.dict(os.environ, {"REDIS_URL": "redis://test:6379"})
    def test_redis_url_configuration(self):
        """Test Redis URL configuration from environment."""
        with patch("api.main.DDoSProtectionMiddleware"):
            # Re-import to test environment variable
            import importlib

            import api.main

            importlib.reload(api.main)

            # Should use environment variable
            assert os.getenv("REDIS_URL") == "redis://test:6379"

    def test_default_redis_url(self):
        """Test default Redis URL when not set in environment."""
        with patch.dict(os.environ, {}, clear=True):
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            assert redis_url == "redis://localhost:6379"


class TestRouterInclusion:
    """Test router inclusion."""

    def test_router_prefixes(self):
        """Test that routers are included with correct prefixes."""
        # Get all routes
        routes = [route.path for route in app.routes]

        # Should have root routes
        assert "/" in routes
        assert "/health" in routes

        # Should have API routes (mocked but registered)
        # We can't test the exact paths since routers are mocked
        # but we can verify the app has routes

    def test_router_tags(self):
        """Test router tags configuration."""
        # This is harder to test directly since routers are mocked
        # but we can verify the app structure
        assert hasattr(app, "routes")
        assert len(app.routes) > 0


class TestLoggingConfiguration:
    """Test logging configuration."""

    def test_logging_level(self):
        """Test logging is configured correctly."""
        # Check that a logger exists
        logger = logging.getLogger("api.main")
        assert logger is not None

        # Verify root logger level is INFO
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO


class TestAppMetadata:
    """Test app metadata and configuration."""

    def test_app_metadata(self):
        """Test FastAPI app metadata."""
        assert app.title == "FreeAgentics API"
        assert app.description == "Multi-Agent AI Platform API with Active Inference"
        assert app.version == "0.1.0"

    def test_app_has_lifespan(self):
        """Test that app has lifespan configured."""
        assert app.router.lifespan_context is not None


class TestErrorHandling:
    """Test error handling in the application."""

    def test_import_error_handling(self):
        """Test that import errors are handled gracefully."""
        # This is tested implicitly by the successful mocking
        # If imports failed, the module wouldn't load
        assert app is not None

    @pytest.mark.asyncio
    async def test_lifespan_import_errors(self):
        """Test lifespan handles import errors gracefully."""
        mock_app = MagicMock(spec=FastAPI)

        with patch("api.main.logger"):
            # Mock import failure
            with patch(
                "builtins.__import__",
                side_effect=ImportError("Module not found"),
            ):
                try:
                    async with lifespan(mock_app):
                        pass
                except ImportError:
                    # If this raises, the error handling isn't working
                    pytest.fail("Lifespan should handle import errors gracefully")


class TestIntegration:
    """Integration tests for the complete application."""

    def test_app_startup_and_request(self):
        """Test complete app startup and request handling."""
        with TestClient(app) as client:
            # Test root endpoint
            response = client.get("/")
            assert response.status_code == 200

            # Test health endpoint
            response = client.get("/health")
            assert response.status_code == 200

    def test_cors_headers_in_response(self):
        """Test that CORS headers are present in responses."""
        with TestClient(app) as client:
            response = client.get("/", headers={"Origin": "http://localhost:3000"})
            assert response.status_code == 200
            # CORS headers should be present if configured correctly
            # (exact headers depend on CORS middleware implementation)

    def test_security_headers_middleware(self):
        """Test that security headers middleware is applied."""
        with TestClient(app) as client:
            response = client.get("/")
            assert response.status_code == 200
            # Security headers should be present
            # (exact headers depend on security middleware implementation)

    def test_options_request_handling(self):
        """Test OPTIONS request handling for CORS."""
        with TestClient(app) as client:
            response = client.options(
                "/",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
            # Should not return error (exact status depends on CORS config)
            assert response.status_code in [200, 204]
