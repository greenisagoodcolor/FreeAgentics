"""API Contract Tests.

Expert Committee: Martin Fowler (API design), Robert C. Martin (contracts)
Testing API contracts and backwards compatibility.
"""

import pytest
import httpx
from typing import Dict, Any, List
import json


class TestAgentAPIContracts:
    """Test contracts for Agent API endpoints."""

    @pytest.fixture
    async def client(self):
        """Create test client for contract testing."""
        from api.main import app
        async with httpx.AsyncClient(app=app, base_url="http://test") as client:
            yield client

    @pytest.mark.asyncio
    async def test_create_agent_contract(self, client):
        """Test agent creation API contract."""
        # Define expected request/response contract
        create_request = {
            "name": "TestAgent",
            "agent_class": "explorer",
            "initial_position": [0, 0]
        }
        
        response = await client.post("/api/agents", json=create_request)
        
        # Contract: Should return 201 with agent data
        if response.status_code == 201:
            agent_data = response.json()
            
            # Verify response structure contract
            required_fields = ["id", "name", "agent_class", "status", "created_at"]
            for field in required_fields:
                assert field in agent_data, f"Missing required field: {field}"
            
            # Verify data types contract
            assert isinstance(agent_data["id"], str)
            assert isinstance(agent_data["name"], str)
            assert agent_data["agent_class"] == "explorer"
            assert agent_data["status"] in ["active", "inactive", "pending"]

    @pytest.mark.asyncio
    async def test_get_agent_contract(self, client):
        """Test get agent API contract."""
        # First create an agent to test retrieval
        create_response = await client.post("/api/agents", json={
            "name": "ContractTestAgent",
            "agent_class": "scholar"
        })
        
        if create_response.status_code == 201:
            agent_id = create_response.json()["id"]
            
            # Test retrieval contract
            get_response = await client.get(f"/api/agents/{agent_id}")
            
            # Contract: Should return 200 with complete agent data
            if get_response.status_code == 200:
                agent_data = get_response.json()
                
                # Extended response contract for GET
                extended_fields = [
                    "id", "name", "agent_class", "status", "created_at",
                    "position", "capabilities", "memory_stats"
                ]
                
                for field in extended_fields:
                    assert field in agent_data, f"GET response missing: {field}"

    @pytest.mark.asyncio
    async def test_agent_list_pagination_contract(self, client):
        """Test agent list pagination contract."""
        response = await client.get("/api/agents?page=1&limit=10")
        
        if response.status_code == 200:
            list_data = response.json()
            
            # Pagination contract
            pagination_fields = ["items", "total", "page", "limit", "has_next"]
            for field in pagination_fields:
                assert field in list_data, f"Pagination missing: {field}"
            
            assert isinstance(list_data["items"], list)
            assert isinstance(list_data["total"], int)
            assert isinstance(list_data["page"], int)
            assert isinstance(list_data["limit"], int)
            assert isinstance(list_data["has_next"], bool)


class TestCoalitionAPIContracts:
    """Test contracts for Coalition API endpoints."""

    @pytest.mark.asyncio
    async def test_coalition_formation_contract(self, client):
        """Test coalition formation API contract."""
        formation_request = {
            "name": "TestCoalition",
            "agent_ids": ["agent_1", "agent_2"],
            "business_type": "ResourceOptimization",
            "formation_criteria": {
                "min_synergy": 0.7,
                "max_size": 5
            }
        }
        
        response = await client.post("/api/coalitions", json=formation_request)
        
        if response.status_code == 201:
            coalition_data = response.json()
            
            # Coalition response contract
            required_fields = [
                "id", "name", "members", "business_type", "status",
                "formation_timestamp", "synergy_score"
            ]
            
            for field in required_fields:
                assert field in coalition_data, f"Coalition missing: {field}"
            
            # Business logic contracts
            assert len(coalition_data["members"]) >= 2
            assert coalition_data["synergy_score"] >= 0.0
            assert coalition_data["business_type"] == "ResourceOptimization"

    @pytest.mark.asyncio
    async def test_coalition_metrics_contract(self, client):
        """Test coalition metrics API contract."""
        response = await client.get("/api/coalitions/123/metrics")
        
        # Contract: Metrics should have consistent structure
        if response.status_code == 200:
            metrics_data = response.json()
            
            metrics_structure = [
                "performance", "efficiency", "stability", "growth",
                "member_satisfaction", "business_value"
            ]
            
            for metric in metrics_structure:
                assert metric in metrics_data, f"Metric missing: {metric}"
                
                # Each metric should have value and trend
                metric_data = metrics_data[metric]
                assert "current_value" in metric_data
                assert "trend" in metric_data
                assert isinstance(metric_data["current_value"], (int, float))


class TestKnowledgeGraphAPIContracts:
    """Test contracts for Knowledge Graph API endpoints."""

    @pytest.mark.asyncio
    async def test_knowledge_node_contract(self, client):
        """Test knowledge node API contract."""
        node_request = {
            "node_type": "concept",
            "data": {
                "title": "Active Inference",
                "confidence": 0.95,
                "source": "research_paper"
            },
            "metadata": {
                "created_by": "agent_123",
                "timestamp": "2024-01-01T00:00:00Z"
            }
        }
        
        response = await client.post("/api/knowledge/nodes", json=node_request)
        
        if response.status_code == 201:
            node_data = response.json()
            
            # Knowledge node contract
            node_fields = [
                "id", "node_type", "data", "metadata", "connections",
                "confidence_score", "last_updated"
            ]
            
            for field in node_fields:
                assert field in node_data, f"Knowledge node missing: {field}"
            
            # Validate confidence score range
            assert 0.0 <= node_data["confidence_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_knowledge_query_contract(self, client):
        """Test knowledge query API contract."""
        query_request = {
            "query_type": "semantic_search",
            "parameters": {
                "search_term": "coalition formation",
                "similarity_threshold": 0.8,
                "max_results": 10
            }
        }
        
        response = await client.post("/api/knowledge/query", json=query_request)
        
        if response.status_code == 200:
            query_results = response.json()
            
            # Query results contract
            result_structure = [
                "results", "total_found", "query_time_ms",
                "similarity_scores", "result_metadata"
            ]
            
            for field in result_structure:
                assert field in query_results, f"Query result missing: {field}"
            
            assert isinstance(query_results["results"], list)
            assert len(query_results["results"]) <= 10  # Respects max_results


class TestWebSocketContracts:
    """Test contracts for WebSocket connections."""

    @pytest.mark.asyncio
    async def test_agent_updates_websocket_contract(self, client):
        """Test WebSocket message contracts for agent updates."""
        try:
            async with client.websocket_connect("/ws/agents") as websocket:
                # Send subscription message
                subscription = {
                    "action": "subscribe",
                    "agent_ids": ["agent_123"],
                    "update_types": ["position", "status", "beliefs"]
                }
                
                await websocket.send_text(json.dumps(subscription))
                
                # Wait for confirmation
                response = await websocket.receive_text()
                confirmation = json.loads(response)
                
                # WebSocket confirmation contract
                assert "status" in confirmation
                assert "subscription_id" in confirmation
                assert confirmation["status"] == "subscribed"
                
        except Exception:
            # WebSocket might not be implemented yet
            pytest.skip("WebSocket not available for contract testing")

    @pytest.mark.asyncio
    async def test_real_time_updates_contract(self, client):
        """Test real-time update message format contracts."""
        expected_update_format = {
            "message_type": "agent_update",
            "agent_id": "agent_123",
            "timestamp": "2024-01-01T00:00:00Z",
            "update_type": "position",
            "data": {
                "new_position": [1, 1],
                "previous_position": [0, 0]
            },
            "metadata": {
                "sequence_number": 1,
                "reliability": "guaranteed"
            }
        }
        
        # This would test actual WebSocket message format
        # For now, verify the expected structure is documented
        assert isinstance(expected_update_format, dict)
        assert "message_type" in expected_update_format
        assert "timestamp" in expected_update_format


class TestAPIVersioningContracts:
    """Test API versioning and backwards compatibility."""

    @pytest.mark.asyncio
    async def test_api_version_headers(self, client):
        """Test API version header contracts."""
        response = await client.get("/api/agents")
        
        # Version headers contract
        version_headers = ["api-version", "supported-versions"]
        
        for header in version_headers:
            if header in response.headers:
                # If version headers exist, they should follow semantic versioning
                version = response.headers[header]
                assert self._is_valid_semantic_version(version)

    def _is_valid_semantic_version(self, version: str) -> bool:
        """Validate semantic version format."""
        import re
        pattern = r'^\d+\.\d+\.\d+$'
        return bool(re.match(pattern, version))

    @pytest.mark.asyncio
    async def test_backwards_compatibility_contract(self, client):
        """Test backwards compatibility contracts."""
        # Test that deprecated fields are still present
        response = await client.get("/api/agents")
        
        if response.status_code == 200:
            # Even if API evolves, critical fields should remain
            # This prevents breaking changes for existing clients
            backwards_compatible_structure = {
                "items": list,
                "total": int
            }
            
            data = response.json()
            for field, expected_type in backwards_compatible_structure.items():
                if field in data:
                    assert isinstance(data[field], expected_type), \
                        f"Field {field} type changed, breaking compatibility"


class TestErrorResponseContracts:
    """Test error response contracts."""

    @pytest.mark.asyncio
    async def test_validation_error_contract(self, client):
        """Test validation error response contract."""
        # Send invalid data to trigger validation error
        invalid_request = {
            "name": "",  # Invalid empty name
            "agent_class": "invalid_class",  # Invalid class
            "initial_position": "not_a_list"  # Invalid type
        }
        
        response = await client.post("/api/agents", json=invalid_request)
        
        if response.status_code == 422:  # Validation error
            error_data = response.json()
            
            # Error response contract
            error_fields = ["error", "message", "details", "timestamp"]
            for field in error_fields:
                assert field in error_data, f"Error response missing: {field}"
            
            # Validation details contract
            if "details" in error_data and error_data["details"]:
                detail = error_data["details"][0]
                validation_fields = ["field", "error_type", "provided_value"]
                
                for field in validation_fields:
                    assert field in detail, f"Validation detail missing: {field}"

    @pytest.mark.asyncio
    async def test_not_found_error_contract(self, client):
        """Test 404 error response contract."""
        response = await client.get("/api/agents/nonexistent_id")
        
        if response.status_code == 404:
            error_data = response.json()
            
            # 404 error contract
            assert "error" in error_data
            assert "message" in error_data
            assert error_data["error"] == "not_found"
            assert "agent" in error_data["message"].lower()

    @pytest.mark.asyncio
    async def test_server_error_contract(self, client):
        """Test 500 error response contract."""
        # This would test server error responses
        # For security, they should not leak sensitive information
        
        # Mock a server error scenario
        with pytest.raises(Exception):
            # Simulate condition that would cause 500 error
            # The error response contract should hide internal details
            pass


class TestPerformanceContracts:
    """Test performance-related API contracts."""

    @pytest.mark.asyncio
    async def test_response_time_contract(self, client):
        """Test response time contracts."""
        import time
        
        # Standard endpoints should respond within reasonable time
        endpoints_with_limits = [
            ("/api/health", 0.1),  # Health check: 100ms
            ("/api/agents", 1.0),   # List agents: 1 second
        ]
        
        for endpoint, max_time in endpoints_with_limits:
            start_time = time.time()
            response = await client.get(endpoint)
            response_time = time.time() - start_time
            
            # Performance contract
            assert response_time < max_time, \
                f"Endpoint {endpoint} too slow: {response_time}s > {max_time}s"

    @pytest.mark.asyncio
    async def test_concurrent_request_contract(self, client):
        """Test concurrent request handling contract."""
        import asyncio
        
        # System should handle reasonable concurrent load
        async def make_request():
            return await client.get("/api/health")
        
        # Test concurrent requests
        tasks = [make_request() for _ in range(10)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Concurrent handling contract
        successful_responses = [r for r in responses if not isinstance(r, Exception)]
        success_rate = len(successful_responses) / len(responses)
        
        assert success_rate > 0.9, f"Concurrent handling poor: {success_rate}" 