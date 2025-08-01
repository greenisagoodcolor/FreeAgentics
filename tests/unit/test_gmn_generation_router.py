"""Test suite for GMN generation router."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from database.gmn_versioned_models import GMNVersionedSpecification
from services.gmn_generator import GMNGenerator


class TestGMNGenerationRouter:
    """Test the GMN generation API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_generator(self):
        """Create mock GMN generator."""
        generator = AsyncMock(spec=GMNGenerator)
        return generator

    @pytest.fixture
    def mock_repository(self):
        """Create mock GMN repository."""
        repository = MagicMock()
        return repository

    @pytest.fixture
    def sample_gmn_spec(self):
        """Sample GMN specification for testing."""
        return """
        node state s1 {
            type: discrete
            size: 4
            description: "Agent position"
        }
        node action a1 {
            type: discrete
            size: 3
            description: "Movement actions"
        }
        node observation o1 {
            type: discrete
            size: 5
            description: "Sensor readings"
        }
        node transition T1 {
            from: [s1, a1]
            to: s1
            stochastic: true
        }
        node emission E1 {
            from: s1
            to: o1
        }
        """

    @pytest.fixture
    def sample_stored_spec(self):
        """Sample stored GMN specification."""
        spec_id = uuid.uuid4()
        agent_id = uuid.uuid4()

        spec = MagicMock(spec=GMNVersionedSpecification)
        spec.id = spec_id
        spec.agent_id = agent_id
        spec.version_number = 1
        spec.node_count = 5
        spec.edge_count = 2
        spec.complexity_score = 0.4
        spec.specification_checksum = "abc123"

        return spec

    @patch("api.v1.routers.gmn_generation.get_gmn_generator")
    @patch("api.v1.routers.gmn_generation.get_gmn_repository")
    def test_generate_gmn_success(
        self,
        mock_repo_dep,
        mock_gen_dep,
        client,
        mock_generator,
        mock_repository,
        sample_gmn_spec,
        sample_stored_spec,
    ):
        """Test successful GMN generation."""
        # Setup mocks
        mock_gen_dep.return_value = mock_generator
        mock_repo_dep.return_value = mock_repository

        mock_generator.prompt_to_gmn.return_value = sample_gmn_spec
        mock_generator.validate_gmn.return_value = (True, [])
        mock_generator.suggest_improvements.return_value = ["Add preferences"]

        mock_repository.create_gmn_specification_versioned.return_value = sample_stored_spec

        # Test request
        request_data = {
            "prompt": "Create an explorer agent that navigates a grid world",
            "agent_type": "explorer",
            "name": "GridExplorer",
        }

        response = client.post("/api/v1/gmn/generate", json=request_data)

        # Assertions
        assert response.status_code == 201
        data = response.json()

        assert data["gmn_specification"] == sample_gmn_spec
        assert data["specification_id"] == str(sample_stored_spec.id)
        assert data["agent_id"] == str(sample_stored_spec.agent_id)
        assert data["version_number"] == 1
        assert data["validation_status"] == "warning"  # Valid but has suggestions
        assert data["suggestions"] == ["Add preferences"]
        assert data["metadata"]["node_count"] == 5

        # Verify service calls
        mock_generator.prompt_to_gmn.assert_called_once()
        mock_generator.validate_gmn.assert_called_once_with(sample_gmn_spec)
        mock_repository.create_gmn_specification_versioned.assert_called_once()

    def test_generate_gmn_invalid_agent_type(self, client):
        """Test validation of invalid agent type."""
        request_data = {"prompt": "Create an agent", "agent_type": "invalid_type"}

        response = client.post("/api/v1/gmn/generate", json=request_data)

        assert response.status_code == 422
        assert "Agent type must be one of" in response.json()["detail"][0]["msg"]

    def test_generate_gmn_empty_prompt(self, client):
        """Test validation of empty prompt."""
        request_data = {"prompt": "   ", "agent_type": "general"}

        response = client.post("/api/v1/gmn/generate", json=request_data)

        assert response.status_code == 422
        assert "Prompt cannot be empty" in response.json()["detail"][0]["msg"]

    def test_generate_gmn_prompt_too_short(self, client):
        """Test validation of too short prompt."""
        request_data = {"prompt": "short", "agent_type": "general"}

        response = client.post("/api/v1/gmn/generate", json=request_data)

        assert response.status_code == 422

    @patch("api.v1.routers.gmn_generation.get_gmn_generator")
    @patch("api.v1.routers.gmn_generation.get_gmn_repository")
    def test_generate_gmn_with_existing_agent_id(
        self,
        mock_repo_dep,
        mock_gen_dep,
        client,
        mock_generator,
        mock_repository,
        sample_gmn_spec,
        sample_stored_spec,
    ):
        """Test GMN generation with existing agent ID."""
        # Setup mocks
        mock_gen_dep.return_value = mock_generator
        mock_repo_dep.return_value = mock_repository

        mock_generator.prompt_to_gmn.return_value = sample_gmn_spec
        mock_generator.validate_gmn.return_value = (True, [])
        mock_generator.suggest_improvements.return_value = []

        # Update sample spec to have version 2
        sample_stored_spec.version_number = 2
        mock_repository.create_gmn_specification_versioned.return_value = sample_stored_spec

        # Test request with existing agent ID
        agent_id = str(uuid.uuid4())
        request_data = {
            "prompt": "Update the explorer agent with better navigation",
            "agent_type": "explorer",
            "agent_id": agent_id,
        }

        response = client.post("/api/v1/gmn/generate", json=request_data)

        # Assertions
        assert response.status_code == 201
        data = response.json()

        assert data["version_number"] == 2
        assert data["validation_status"] == "valid"  # No suggestions

    @patch("api.v1.routers.gmn_generation.get_gmn_generator")
    def test_validate_gmn_success(self, mock_gen_dep, client, mock_generator, sample_gmn_spec):
        """Test successful GMN validation."""
        # Setup mocks
        mock_gen_dep.return_value = mock_generator

        mock_generator.validate_gmn.return_value = (True, [])
        mock_generator.suggest_improvements.return_value = ["Add initial state distribution"]

        # Test request
        request_data = {"gmn_specification": sample_gmn_spec}

        response = client.post("/api/v1/gmn/validate", json=request_data)

        # Assertions
        assert response.status_code == 200
        data = response.json()

        assert data["is_valid"] is True
        assert data["errors"] == []
        assert data["suggestions"] == ["Add initial state distribution"]

    @patch("api.v1.routers.gmn_generation.get_gmn_generator")
    def test_validate_gmn_with_errors(self, mock_gen_dep, client, mock_generator):
        """Test GMN validation with errors."""
        # Setup mocks
        mock_gen_dep.return_value = mock_generator

        mock_generator.validate_gmn.return_value = (False, ["Missing required node types"])
        mock_generator.suggest_improvements.return_value = ["Add state nodes"]

        # Test request
        request_data = {"gmn_specification": "invalid gmn spec"}

        response = client.post("/api/v1/gmn/validate", json=request_data)

        # Assertions
        assert response.status_code == 200
        data = response.json()

        assert data["is_valid"] is False
        assert data["errors"] == ["Missing required node types"]
        assert data["suggestions"] == ["Add state nodes"]

    @patch("api.v1.routers.gmn_generation.get_gmn_generator")
    def test_refine_gmn_success(self, mock_gen_dep, client, mock_generator, sample_gmn_spec):
        """Test successful GMN refinement."""
        # Setup mocks
        mock_gen_dep.return_value = mock_generator

        refined_spec = sample_gmn_spec + "\nnode preference C1 { state: s1 }"
        mock_generator.refine_gmn.return_value = refined_spec
        mock_generator.validate_gmn.return_value = (True, [])
        mock_generator.suggest_improvements.return_value = []

        # Test request
        request_data = {
            "gmn_specification": sample_gmn_spec,
            "feedback": "Add preference nodes to define agent goals",
        }

        response = client.post("/api/v1/gmn/refine", json=request_data)

        # Assertions
        assert response.status_code == 200
        data = response.json()

        assert "preference C1" in data["gmn_specification"]
        assert data["specification_id"] == ""  # Not stored
        assert data["validation_status"] == "valid"
        assert data["metadata"]["refined"] is True

    def test_refine_gmn_empty_feedback(self, client, sample_gmn_spec):
        """Test refinement with empty feedback."""
        request_data = {"gmn_specification": sample_gmn_spec, "feedback": ""}

        response = client.post("/api/v1/gmn/refine", json=request_data)

        assert response.status_code == 422

    @patch("api.v1.routers.gmn_generation.get_gmn_generator")
    @patch("api.v1.routers.gmn_generation.get_gmn_repository")
    def test_generate_gmn_service_error(self, mock_repo_dep, mock_gen_dep, client, mock_generator):
        """Test handling of service errors during generation."""
        # Setup mocks
        mock_gen_dep.return_value = mock_generator
        mock_repo_dep.return_value = MagicMock()

        mock_generator.prompt_to_gmn.side_effect = Exception("LLM service unavailable")

        # Test request
        request_data = {"prompt": "Create an agent", "agent_type": "general"}

        response = client.post("/api/v1/gmn/generate", json=request_data)

        # Assertions
        assert response.status_code == 500
        assert "GMN generation failed" in response.json()["detail"]

    def test_parse_gmn_basic_functionality(self):
        """Test the basic GMN parsing utility function."""
        from api.v1.routers.gmn_generation import _parse_gmn_basic

        gmn_spec = """
        node state s1 {
            type: discrete
            size: 4
        }
        node action a1 {
            type: discrete
            size: 3
        }
        node transition T1 {
            from: [s1, a1]
            to: s1
        }
        """

        result = _parse_gmn_basic(gmn_spec)

        assert len(result["nodes"]) == 3
        assert len(result["edges"]) == 1

        # Check node parsing
        node_names = [node["name"] for node in result["nodes"]]
        assert "s1" in node_names
        assert "a1" in node_names
        assert "T1" in node_names

        # Check edge parsing
        edge = result["edges"][0]
        assert edge["from_node"] == "T1"
        assert edge["type"] == "from"

    @patch("api.v1.routers.gmn_generation.get_gmn_generator")
    @patch("api.v1.routers.gmn_generation.get_gmn_repository")
    def test_generate_gmn_with_constraints(
        self,
        mock_repo_dep,
        mock_gen_dep,
        client,
        mock_generator,
        mock_repository,
        sample_gmn_spec,
        sample_stored_spec,
    ):
        """Test GMN generation with custom constraints."""
        # Setup mocks
        mock_gen_dep.return_value = mock_generator
        mock_repo_dep.return_value = mock_repository

        mock_generator.prompt_to_gmn.return_value = sample_gmn_spec
        mock_generator.validate_gmn.return_value = (True, [])
        mock_generator.suggest_improvements.return_value = []

        mock_repository.create_gmn_specification_versioned.return_value = sample_stored_spec

        # Test request with constraints
        request_data = {
            "prompt": "Create a trading agent",
            "agent_type": "trader",
            "constraints": {
                "max_states": 10,
                "require_preferences": True,
                "observation_types": ["price", "volume"],
            },
        }

        response = client.post("/api/v1/gmn/generate", json=request_data)

        # Assertions
        assert response.status_code == 201

        # Verify constraints were passed to generator
        call_args = mock_generator.prompt_to_gmn.call_args
        assert call_args[1]["constraints"]["max_states"] == 10
        assert call_args[1]["constraints"]["require_preferences"] is True

    @patch("api.v1.routers.gmn_generation.get_gmn_generator")
    @patch("api.v1.routers.gmn_generation.get_gmn_repository")
    def test_generate_gmn_storage_failure(
        self, mock_repo_dep, mock_gen_dep, client, mock_generator, mock_repository, sample_gmn_spec
    ):
        """Test handling of storage failures during GMN generation."""
        # Setup mocks
        mock_gen_dep.return_value = mock_generator
        mock_repo_dep.return_value = mock_repository

        mock_generator.prompt_to_gmn.return_value = sample_gmn_spec
        mock_generator.validate_gmn.return_value = (True, [])
        mock_generator.suggest_improvements.return_value = []

        # Repository fails during storage
        mock_repository.create_gmn_specification_versioned.side_effect = Exception("Database error")

        # Test request
        request_data = {"prompt": "Create an agent", "agent_type": "general"}

        response = client.post("/api/v1/gmn/generate", json=request_data)

        # Should return 500 error
        assert response.status_code == 500
        assert "GMN generation failed" in response.json()["detail"]

    def test_request_model_validation_comprehensive(self, client):
        """Test comprehensive validation of request models."""
        # Test prompt too long
        long_prompt = "a" * 2001
        response = client.post(
            "/api/v1/gmn/generate", json={"prompt": long_prompt, "agent_type": "general"}
        )
        assert response.status_code == 422

        # Test invalid name too long
        response = client.post(
            "/api/v1/gmn/generate",
            json={
                "prompt": "Create an agent with a very long name",
                "agent_type": "general",
                "name": "a" * 101,
            },
        )
        assert response.status_code == 422
