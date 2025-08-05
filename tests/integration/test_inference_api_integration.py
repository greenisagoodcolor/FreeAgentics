"""Integration test for Inference API with real PyMDP agents."""

import numpy as np
from fastapi.testclient import TestClient

from api.main import app


class TestInferenceAPIIntegration:
    """Integration tests for the inference API endpoints."""

    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)

        # Sample GMN specification for testing
        B_matrix = np.zeros((4, 4, 4))
        for action in range(4):
            B_matrix[:, :, action] = np.eye(4)  # Identity transitions

        self.sample_agent_spec = {
            "num_states": [4],
            "num_obs": [4],
            "num_actions": [4],
            "A": [np.eye(4).tolist()],  # Convert to list for JSON serialization
            "B": [B_matrix.tolist()],
            "C": [np.array([1.0, 0.0, 0.0, 0.0]).tolist()],  # Preferences
            "D": [np.ones(4).tolist()],  # Uniform prior (will be normalized)
        }

    def test_run_inference_endpoint(self):
        """Test the main inference endpoint."""

        # Override authentication dependency
        def mock_auth():
            return {"user_id": "test_user"}

        from auth.security_implementation import get_current_user

        app.dependency_overrides[get_current_user] = mock_auth

        # Make inference request
        request_data = {
            "agent_spec": self.sample_agent_spec,
            "observation": [0],
            "planning_horizon": 1,
            "timeout_ms": 5000,
        }

        response = self.client.post("/api/v1/inference/run_inference", json=request_data)

        # Assert successful response
        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "action" in data
        assert "beliefs" in data
        assert "free_energy" in data
        assert "confidence" in data
        assert "metadata" in data

        # Check that action was selected
        assert data["action"] is not None

        # Check beliefs are present
        assert isinstance(data["beliefs"], dict)
        assert "states" in data["beliefs"]

        # Check confidence is reasonable
        assert 0.0 <= data["confidence"] <= 1.0

        # Check metadata contains timing info
        assert "inference_time_ms" in data["metadata"]

        # Cleanup
        app.dependency_overrides.clear()

    def test_batch_inference_endpoint(self):
        """Test batch inference endpoint."""

        # Override authentication dependency
        def mock_auth():
            return {"user_id": "test_user"}

        from auth.security_implementation import get_current_user

        app.dependency_overrides[get_current_user] = mock_auth

        # Make batch inference request
        request_data = {
            "agent_spec": self.sample_agent_spec,
            "observations": [[0], [1], [2]],
            "timeout_ms": 5000,
        }

        response = self.client.post("/api/v1/inference/batch_inference", json=request_data)

        # Assert successful response
        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "results" in data
        assert "metadata" in data

        # Check results for each observation
        assert len(data["results"]) == 3

        for result in data["results"]:
            assert "action" in result
            assert "beliefs" in result
            assert "confidence" in result

        # Check metadata
        assert data["metadata"]["total_observations"] == 3

        # Cleanup
        app.dependency_overrides.clear()

    def test_invalid_observation_error_handling(self):
        """Test error handling for invalid observations."""

        # Override authentication dependency
        def mock_auth():
            return {"user_id": "test_user"}

        from auth.security_implementation import get_current_user

        app.dependency_overrides[get_current_user] = mock_auth

        # Make request with invalid observation
        request_data = {
            "agent_spec": self.sample_agent_spec,
            "observation": [10],  # Invalid - out of bounds
            "timeout_ms": 5000,
        }

        response = self.client.post("/api/v1/inference/run_inference", json=request_data)

        # Should return error
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "invalid observation" in data["detail"].lower()

        # Cleanup
        app.dependency_overrides.clear()

    def test_invalid_agent_spec_error_handling(self):
        """Test error handling for invalid agent specifications."""

        # Override authentication dependency
        def mock_auth():
            return {"user_id": "test_user"}

        from auth.security_implementation import get_current_user

        app.dependency_overrides[get_current_user] = mock_auth

        # Make request with invalid agent spec
        request_data = {
            "agent_spec": {"invalid": "spec"},  # Missing required fields
            "observation": [0],
            "timeout_ms": 5000,
        }

        response = self.client.post("/api/v1/inference/run_inference", json=request_data)

        # Should return error
        assert response.status_code == 400

        # Cleanup
        app.dependency_overrides.clear()

    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/api/v1/inference/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["service"] == "inference"
        assert "pymdp_available" in data
        assert "total_inferences" in data

    def test_metrics_endpoint(self):
        """Test metrics endpoint."""

        # Override authentication dependency
        def mock_auth():
            return {"user_id": "test_user"}

        from auth.security_implementation import get_current_user

        app.dependency_overrides[get_current_user] = mock_auth

        response = self.client.get("/api/v1/inference/metrics")

        assert response.status_code == 200
        data = response.json()

        assert "inference_engine" in data
        assert "agent_factory" in data

        # Check engine metrics structure
        engine_metrics = data["inference_engine"]
        assert "inferences_completed" in engine_metrics
        assert "success_rate" in engine_metrics

        # Check factory metrics structure
        factory_metrics = data["agent_factory"]
        assert "agents_created" in factory_metrics

        # Cleanup
        app.dependency_overrides.clear()

    def test_end_to_end_inference_flow(self):
        """Test complete end-to-end inference flow."""

        # Override authentication dependency
        def mock_auth():
            return {"user_id": "test_user"}

        from auth.security_implementation import get_current_user

        app.dependency_overrides[get_current_user] = mock_auth

        # 1. Check initial health
        health_response = self.client.get("/api/v1/inference/health")
        assert health_response.status_code == 200
        initial_health = health_response.json()
        initial_inferences = initial_health["total_inferences"]

        # 2. Run inference
        request_data = {
            "agent_spec": self.sample_agent_spec,
            "observation": [1],
            "planning_horizon": 2,
            "timeout_ms": 5000,
        }

        inference_response = self.client.post("/api/v1/inference/run_inference", json=request_data)
        assert inference_response.status_code == 200
        inference_data = inference_response.json()

        # 3. Verify inference results
        assert inference_data["action"] is not None
        assert inference_data["confidence"] > 0.0
        assert "planning_horizon" in inference_data["metadata"]
        assert inference_data["metadata"]["planning_horizon"] == 2

        # 4. Check updated health metrics
        final_health_response = self.client.get("/api/v1/inference/health")
        assert final_health_response.status_code == 200
        final_health = final_health_response.json()
        final_inferences = final_health["total_inferences"]

        # Should have incremented inference count
        assert final_inferences > initial_inferences

        # 5. Get detailed metrics
        metrics_response = self.client.get("/api/v1/inference/metrics")
        assert metrics_response.status_code == 200
        metrics_data = metrics_response.json()

        # Verify metrics are updated
        assert metrics_data["inference_engine"]["inferences_completed"] > 0
        assert metrics_data["agent_factory"]["agents_created"] > 0

        # Cleanup
        app.dependency_overrides.clear()
