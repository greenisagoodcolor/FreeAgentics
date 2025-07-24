"""
Test health check endpoint following TDD principles.

RED: Write failing tests first
GREEN: Implement minimal code to pass
REFACTOR: Improve code while keeping tests passing
"""

import time

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.main import app
from database.session import get_db


class TestHealthEndpoint:
    """Test suite for /health endpoint following TDD principles."""

    def test_health_endpoint_returns_200_when_database_connected(self):
        """Test that /health returns 200 OK when database is connected."""
        client = TestClient(app)

        # Execute request to /health endpoint
        start_time = time.time()
        response = client.get("/health")
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        # Assert response status code
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"

        # Assert response JSON structure
        data = response.json()
        assert "status" in data, "Response should contain 'status' field"
        assert data["status"] == "healthy", f"Expected status 'healthy', got {data['status']}"

        assert "db" in data, "Response should contain 'db' field"
        assert data["db"] == "connected", f"Expected db 'connected', got {data['db']}"

        # Assert response time is under 100ms
        assert response_time < 100, f"Response took {response_time:.2f}ms, expected < 100ms"

    def test_health_endpoint_returns_503_when_database_disconnected(self):
        """Test that /health returns 503 Service Unavailable when database is down."""
        client = TestClient(app)

        # Mock a database failure by overriding the dependency
        def get_broken_db():
            # Create a session that will fail
            engine = create_engine("postgresql://invalid:invalid@invalid:5432/invalid")
            SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            db = SessionLocal()
            try:
                yield db
            finally:
                db.close()

        # Override the dependency
        app.dependency_overrides[get_db] = get_broken_db

        try:
            # Execute request to /health endpoint
            response = client.get("/health")

            # Assert response status code
            assert response.status_code == 503, f"Expected 503, got {response.status_code}"

            # Assert response JSON structure
            data = response.json()
            assert "status" in data, "Response should contain 'status' field"
            assert (
                data["status"] == "unhealthy"
            ), f"Expected status 'unhealthy', got {data['status']}"

            assert "db" in data, "Response should contain 'db' field"
            assert data["db"] == "disconnected", f"Expected db 'disconnected', got {data['db']}"

            assert "error" in data, "Response should contain 'error' field when unhealthy"

        finally:
            # Clean up: remove the override
            app.dependency_overrides.pop(get_db, None)

    def test_health_endpoint_performs_real_database_query(self):
        """Test that /health endpoint performs a real SELECT 1 query."""
        client = TestClient(app)

        # Track if SELECT 1 was executed
        query_executed = False
        original_execute = None

        def get_tracking_db():
            """Get a database session that tracks SELECT 1 execution."""
            from database.session import SessionLocal

            db = SessionLocal()

            # Store original execute method
            nonlocal original_execute
            original_execute = db.execute

            # Wrap execute to track SELECT 1
            def tracked_execute(statement, *args, **kwargs):
                nonlocal query_executed
                # Check if this is our SELECT 1 query
                if hasattr(statement, "text"):
                    query_text = str(statement.text).strip().upper()
                    if query_text == "SELECT 1" or query_text == "SELECT 1;":
                        query_executed = True
                return original_execute(statement, *args, **kwargs)

            db.execute = tracked_execute

            try:
                yield db
            finally:
                db.close()

        # Override the dependency
        app.dependency_overrides[get_db] = get_tracking_db

        try:
            # Execute request to /health endpoint
            response = client.get("/health")

            # Assert SELECT 1 was executed
            assert query_executed, "Health endpoint should execute 'SELECT 1' query"

            # Also verify successful response
            assert response.status_code == 200

        finally:
            # Clean up: remove the override
            app.dependency_overrides.pop(get_db, None)

    def test_health_endpoint_response_time_under_100ms(self):
        """Test that /health endpoint responds in under 100ms consistently."""
        client = TestClient(app)

        # Test multiple times to ensure consistency
        response_times = []

        for _ in range(10):
            start_time = time.time()
            response = client.get("/health")
            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            assert response.status_code == 200
            response_times.append(response_time)

        # All response times should be under 100ms
        for i, rt in enumerate(response_times):
            assert rt < 100, f"Request {i + 1} took {rt:.2f}ms, expected < 100ms"

        # Average should also be well under 100ms
        avg_time = sum(response_times) / len(response_times)
        assert avg_time < 50, f"Average response time {avg_time:.2f}ms should be well under 100ms"

    def test_health_endpoint_uses_fastapi_exception_handlers(self):
        """Test that health endpoint doesn't use try/except, relies on FastAPI handlers."""
        # This test will verify the implementation doesn't catch exceptions
        # We'll check this by looking at the source code in the implementation
        # For now, this serves as a reminder of the requirement
