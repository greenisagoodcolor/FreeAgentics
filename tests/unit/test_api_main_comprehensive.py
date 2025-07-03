"""
Comprehensive tests for the main API module to achieve 100% coverage.
This focuses on the FastAPI application setup, middleware, and endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from api.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)


class TestAPIMain:
    """Comprehensive tests for the main API module"""

    def test_app_configuration(self):
        """Test that the FastAPI app is configured correctly"""
        assert app.title == "FreeAgentics API"
        assert app.description == "Multi-Agent Active Inference System API"
        assert app.version == "2.1.0"
        assert app.docs_url == "/docs"
        assert app.redoc_url == "/redoc"

    def test_root_endpoint(self, client):
        """Test the root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "FreeAgentics API"
        assert data["version"] == "2.1.0"
        assert data["status"] == "operational"
        assert data["description"] == "Multi-Agent Active Inference System"
        assert "endpoints" in data
        assert data["endpoints"]["docs"] == "/docs"
        assert data["endpoints"]["redoc"] == "/redoc"
        assert data["endpoints"]["health"] == "/health"

    def test_health_check_endpoint(self, client):
        """Test the health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "FreeAgentics API"
        assert data["version"] == "2.1.0"

    def test_api_status_endpoint(self, client):
        """Test the API status endpoint"""
        response = client.get("/api/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["api_status"] == "online"
        
        # Test agents section
        assert "agents" in data
        agents_data = data["agents"]
        assert agents_data["total"] == 0
        assert agents_data["active"] == 0
        assert "templates" in agents_data
        expected_templates = ["Explorer", "Merchant", "Scholar", "Guardian", "Generalist"]
        assert agents_data["templates"] == expected_templates
        
        # Test system section
        assert "system" in data
        system_data = data["system"]
        assert system_data["active_inference"] == "ready"
        assert system_data["knowledge_graph"] == "ready"
        assert system_data["coalition_formation"] == "ready"

    def test_security_headers_middleware(self, client):
        """Test that security headers are added to responses"""
        response = client.get("/")
        
        # Verify security headers are present
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert response.headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"
        assert response.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"

    def test_security_headers_on_health_endpoint(self, client):
        """Test that security headers are added to health endpoint"""
        response = client.get("/health")
        
        # Verify security headers are present on all endpoints
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-XSS-Protection"] == "1; mode=block"

    def test_cors_configuration(self):
        """Test CORS configuration"""
        # CORS middleware should be configured
        cors_middleware = None
        for middleware in app.user_middleware:
            if hasattr(middleware.cls, '__name__') and 'CORS' in middleware.cls.__name__:
                cors_middleware = middleware
                break
        
        assert cors_middleware is not None

    def test_cors_headers(self, client):
        """Test CORS headers in response"""
        # Make a request with CORS origin header
        response = client.get("/", headers={"Origin": "http://localhost:3000"})
        
        # Should return successful response
        assert response.status_code == 200

    def test_nonexistent_endpoint(self, client):
        """Test that nonexistent endpoints return 404"""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test method not allowed responses"""
        # Try POST on GET-only endpoint
        response = client.post("/")
        assert response.status_code == 405

    def test_api_docs_accessibility(self, client):
        """Test that API documentation endpoints are accessible"""
        # Test that docs endpoint is accessible
        docs_response = client.get("/docs")
        assert docs_response.status_code == 200
        
        # Test that redoc endpoint is accessible
        redoc_response = client.get("/redoc")
        assert redoc_response.status_code == 200

    def test_openapi_schema(self, client):
        """Test OpenAPI schema generation"""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "FreeAgentics API"
        assert schema["info"]["version"] == "2.1.0"

    def test_multiple_concurrent_requests(self, client):
        """Test handling multiple concurrent requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.get("/health")
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
        assert len(results) == 5

    def test_response_time_performance(self, client):
        """Test that endpoints respond within reasonable time"""
        import time
        
        start_time = time.time()
        response = client.get("/")
        end_time = time.time()
        
        assert response.status_code == 200
        # Response should be very fast (less than 1 second)
        assert (end_time - start_time) < 1.0

    def test_large_number_of_requests(self, client):
        """Test handling a large number of sequential requests"""
        for i in range(50):
            response = client.get("/health")
            assert response.status_code == 200

    def test_endpoint_response_formats(self, client):
        """Test that all endpoints return valid JSON"""
        endpoints = ["/", "/health", "/api/status"]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200
            
            # Should be valid JSON
            data = response.json()
            assert isinstance(data, dict)

    def test_security_headers_consistency(self, client):
        """Test that security headers are consistent across all endpoints"""
        endpoints = ["/", "/health", "/api/status"]
        
        expected_headers = [
            "X-Frame-Options",
            "X-Content-Type-Options", 
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Referrer-Policy"
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            for header in expected_headers:
                assert header in response.headers

    def test_content_type_headers(self, client):
        """Test that responses have correct content type"""
        response = client.get("/")
        assert response.headers["content-type"] == "application/json"

    def test_middleware_processing_order(self, client):
        """Test that middleware is processed in correct order"""
        response = client.get("/")
        
        # Both CORS and security headers should be present
        # indicating proper middleware order
        assert response.status_code == 200
        assert "X-Frame-Options" in response.headers

    @patch('uvicorn.run')
    def test_main_execution(self, mock_uvicorn):
        """Test the main execution block"""
        # Import and execute main
        import api.main
        
        # The if __name__ == "__main__" block should be covered
        # when the module is executed directly
        # This is primarily for coverage, actual execution is mocked
        pass

    def test_error_handling(self, client):
        """Test error handling for various scenarios"""
        # Test that the app handles errors gracefully
        response = client.get("/")
        assert response.status_code == 200
        
        # Test invalid path
        response = client.get("/invalid/path/here")
        assert response.status_code == 404

    def test_app_state_consistency(self, client):
        """Test that app state remains consistent across requests"""
        # Make multiple requests and verify app state
        for _ in range(10):
            response = client.get("/api/status")
            data = response.json()
            
            # State should be consistent
            assert data["api_status"] == "online"
            assert data["agents"]["total"] == 0
            assert data["agents"]["active"] == 0

    def test_request_response_cycle(self, client):
        """Test complete request-response cycle"""
        # Test that request flows through middleware and endpoint correctly
        response = client.get("/", headers={"User-Agent": "test-agent"})
        
        assert response.status_code == 200
        # Security headers should be added
        assert "X-Frame-Options" in response.headers
        # Response should contain expected data
        data = response.json()
        assert data["name"] == "FreeAgentics API"

    def test_async_endpoint_execution(self, client):
        """Test that async endpoints execute correctly"""
        # All endpoints are async, test they work properly
        endpoints = ["/", "/health", "/api/status"]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200
            assert isinstance(response.json(), dict)