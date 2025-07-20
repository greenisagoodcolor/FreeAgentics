"""Tests for GMN API endpoint POST /api/v1/agents/{id}/gmn.

Following TDD principles: These tests define the expected behavior for the GMN API endpoint.
"""

import uuid
from unittest.mock import Mock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from api.main import app
from database.conversation_models import ValidationStatus


class TestGMNAPIEndpoint:
    """Test the POST /api/v1/agents/{id}/gmn endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def agent_id(self):
        """Create test agent ID."""
        return str(uuid.uuid4())

    @pytest.fixture
    def valid_gmn_spec(self):
        """Valid GMN specification for testing."""
        return {
            "name": "Test Exploration Agent",
            "version": "1.0",
            "specification": {
                "nodes": [
                    {"name": "location", "type": "state", "num_states": 9},
                    {
                        "name": "obs_location",
                        "type": "observation",
                        "num_observations": 9,
                    },
                    {"name": "move", "type": "action", "num_actions": 5},
                    {
                        "name": "goal_preference",
                        "type": "preference",
                        "preferred_observation": 8,
                        "preference_strength": 1.0,
                    },
                ],
                "edges": [
                    {
                        "from": "location",
                        "to": "obs_location",
                        "type": "generates",
                    },
                    {
                        "from": "goal_preference",
                        "to": "obs_location",
                        "type": "depends_on",
                    },
                ],
            },
        }

    def test_upload_valid_gmn_specification(
        self, client, agent_id, valid_gmn_spec
    ):
        """Test uploading a valid GMN specification."""
        with patch("database.gmn_repository.GMNRepository") as mock_repo:
            # Mock the database operations
            mock_repo_instance = Mock()
            mock_repo.return_value = mock_repo_instance

            mock_gmn_spec = Mock()
            mock_gmn_spec.id = uuid.uuid4()
            mock_gmn_spec.validation_status = ValidationStatus.VALID
            mock_gmn_spec.name = "Test Exploration Agent"
            mock_gmn_spec.description = None
            mock_gmn_spec.version = "1.0"
            mock_gmn_spec.tags = []
            mock_gmn_spec.spec_metadata = {}
            mock_gmn_spec.is_active = False
            mock_gmn_spec.created_at = None
            mock_gmn_spec.updated_at = None
            mock_repo_instance.create_gmn_specification.return_value = (
                mock_gmn_spec
            )

            response = client.post(
                f"/api/v1/agents/{agent_id}/gmn",
                json=valid_gmn_spec,
                headers={"Content-Type": "application/json"},
            )

            assert response.status_code == status.HTTP_201_CREATED
            response_data = response.json()
            assert "id" in response_data
            assert response_data["validation_status"] == "valid"
            assert "matrices" in response_data

    def test_upload_invalid_gmn_specification(self, client, agent_id):
        """Test uploading an invalid GMN specification."""
        invalid_spec = {
            "name": "Invalid Spec",
            "specification": {
                "nodes": [
                    {"name": "invalid", "type": "invalid_type"}
                ],  # Invalid type
                "edges": [],
            },
        }

        response = client.post(
            f"/api/v1/agents/{agent_id}/gmn",
            json=invalid_spec,
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        response_data = response.json()
        assert "detail" in response_data or "errors" in response_data

    def test_upload_gmn_with_missing_specification(self, client, agent_id):
        """Test uploading GMN without specification field."""
        incomplete_spec = {
            "name": "Missing Spec"
            # Missing "specification" field
        }

        response = client.post(
            f"/api/v1/agents/{agent_id}/gmn",
            json=incomplete_spec,
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        response_data = response.json()
        assert "detail" in response_data

    def test_upload_gmn_with_invalid_agent_id(self, client, valid_gmn_spec):
        """Test uploading GMN with invalid agent ID."""
        invalid_agent_id = "not-a-uuid"

        response = client.post(
            f"/api/v1/agents/{invalid_agent_id}/gmn",
            json=valid_gmn_spec,
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST

    def test_upload_gmn_with_nonexistent_agent(self, client, valid_gmn_spec):
        """Test uploading GMN for nonexistent agent."""
        nonexistent_agent_id = str(uuid.uuid4())

        with patch("database.agent_repository.AgentRepository") as mock_repo:
            mock_repo_instance = Mock()
            mock_repo.return_value = mock_repo_instance
            mock_repo_instance.get_agent.return_value = None  # Agent not found

            response = client.post(
                f"/api/v1/agents/{nonexistent_agent_id}/gmn",
                json=valid_gmn_spec,
                headers={"Content-Type": "application/json"},
            )

            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_upload_gmn_triggers_free_energy_recalculation(
        self, client, agent_id, valid_gmn_spec
    ):
        """Test that uploading GMN triggers free energy recalculation."""
        with (
            patch("database.gmn_repository.GMNRepository") as mock_gmn_repo,
            patch(
                "database.agent_repository.AgentRepository"
            ) as mock_agent_repo,
            patch(
                "agents.free_energy_triggers.FreeEnergyTrigger"
            ) as mock_trigger,
        ):
            # Mock successful operations
            mock_gmn_repo_instance = Mock()
            mock_gmn_repo.return_value = mock_gmn_repo_instance

            mock_agent_repo_instance = Mock()
            mock_agent_repo.return_value = mock_agent_repo_instance
            mock_agent_repo_instance.get_agent.return_value = (
                Mock()
            )  # Agent exists

            mock_gmn_spec = Mock()
            mock_gmn_spec.id = uuid.uuid4()
            mock_gmn_spec.validation_status = ValidationStatus.VALID
            mock_gmn_repo_instance.create_gmn_specification.return_value = (
                mock_gmn_spec
            )

            mock_trigger_instance = Mock()
            mock_trigger.return_value = mock_trigger_instance

            response = client.post(
                f"/api/v1/agents/{agent_id}/gmn",
                json=valid_gmn_spec,
                headers={"Content-Type": "application/json"},
            )

            assert response.status_code == status.HTTP_201_CREATED

            # Verify free energy trigger was called
            mock_trigger_instance.trigger_belief_update.assert_called_once()

    def test_upload_gmn_with_versioning(
        self, client, agent_id, valid_gmn_spec
    ):
        """Test that GMN specifications are properly versioned."""
        # First upload
        with patch("database.gmn_repository.GMNRepository") as mock_repo:
            mock_repo_instance = Mock()
            mock_repo.return_value = mock_repo_instance

            mock_gmn_spec_v1 = Mock()
            mock_gmn_spec_v1.id = uuid.uuid4()
            mock_gmn_spec_v1.version = "1.0"
            mock_gmn_spec_v1.validation_status = ValidationStatus.VALID
            mock_repo_instance.create_gmn_specification.return_value = (
                mock_gmn_spec_v1
            )

            response1 = client.post(
                f"/api/v1/agents/{agent_id}/gmn",
                json=valid_gmn_spec,
                headers={"Content-Type": "application/json"},
            )

            assert response1.status_code == status.HTTP_201_CREATED
            assert response1.json()["version"] == "1.0"

        # Second upload (should increment version)
        updated_spec = valid_gmn_spec.copy()
        updated_spec["version"] = "1.1"

        with patch("database.gmn_repository.GMNRepository") as mock_repo:
            mock_repo_instance = Mock()
            mock_repo.return_value = mock_repo_instance

            mock_gmn_spec_v2 = Mock()
            mock_gmn_spec_v2.id = uuid.uuid4()
            mock_gmn_spec_v2.version = "1.1"
            mock_gmn_spec_v2.validation_status = ValidationStatus.VALID
            mock_repo_instance.create_gmn_specification.return_value = (
                mock_gmn_spec_v2
            )

            response2 = client.post(
                f"/api/v1/agents/{agent_id}/gmn",
                json=updated_spec,
                headers={"Content-Type": "application/json"},
            )

            assert response2.status_code == status.HTTP_201_CREATED
            assert response2.json()["version"] == "1.1"

    def test_upload_gmn_returns_pymdp_matrices(
        self, client, agent_id, valid_gmn_spec
    ):
        """Test that uploaded GMN returns PyMDP matrices."""
        with patch("database.gmn_repository.GMNRepository") as mock_repo:
            mock_repo_instance = Mock()
            mock_repo.return_value = mock_repo_instance

            mock_gmn_spec = Mock()
            mock_gmn_spec.id = uuid.uuid4()
            mock_gmn_spec.validation_status = ValidationStatus.VALID
            mock_repo_instance.create_gmn_specification.return_value = (
                mock_gmn_spec
            )

            response = client.post(
                f"/api/v1/agents/{agent_id}/gmn",
                json=valid_gmn_spec,
                headers={"Content-Type": "application/json"},
            )

            assert response.status_code == status.HTTP_201_CREATED
            response_data = response.json()

            # Check that matrices are returned
            assert "matrices" in response_data
            matrices = response_data["matrices"]
            assert "A" in matrices  # Observation model
            assert "B" in matrices  # Transition model
            assert "C" in matrices  # Preferences
            assert "D" in matrices  # Prior beliefs

    def test_upload_gmn_with_text_format(self, client, agent_id):
        """Test uploading GMN specification in text format."""
        text_spec = {
            "name": "Text Format Test",
            "specification_text": """
                [nodes]
                location: state {num_states: 4}
                obs_location: observation {num_observations: 4}
                move: action {num_actions: 4}

                [edges]
                location -> obs_location: generates
            """,
        }

        with patch("database.gmn_repository.GMNRepository") as mock_repo:
            mock_repo_instance = Mock()
            mock_repo.return_value = mock_repo_instance

            mock_gmn_spec = Mock()
            mock_gmn_spec.id = uuid.uuid4()
            mock_gmn_spec.validation_status = ValidationStatus.VALID
            mock_repo_instance.create_gmn_specification.return_value = (
                mock_gmn_spec
            )

            response = client.post(
                f"/api/v1/agents/{agent_id}/gmn",
                json=text_spec,
                headers={"Content-Type": "application/json"},
            )

            assert response.status_code == status.HTTP_201_CREATED

    def test_upload_gmn_with_metadata(self, client, agent_id, valid_gmn_spec):
        """Test uploading GMN with additional metadata."""
        spec_with_metadata = valid_gmn_spec.copy()
        spec_with_metadata.update(
            {
                "description": "Test exploration agent for grid world",
                "tags": ["exploration", "grid-world", "test"],
                "metadata": {
                    "author": "Test User",
                    "created_for": "Unit Testing",
                    "grid_size": 3,
                },
            }
        )

        with patch("database.gmn_repository.GMNRepository") as mock_repo:
            mock_repo_instance = Mock()
            mock_repo.return_value = mock_repo_instance

            mock_gmn_spec = Mock()
            mock_gmn_spec.id = uuid.uuid4()
            mock_gmn_spec.validation_status = ValidationStatus.VALID
            mock_repo_instance.create_gmn_specification.return_value = (
                mock_gmn_spec
            )

            response = client.post(
                f"/api/v1/agents/{agent_id}/gmn",
                json=spec_with_metadata,
                headers={"Content-Type": "application/json"},
            )

            assert response.status_code == status.HTTP_201_CREATED
            response_data = response.json()
            assert "description" in response_data
            assert "tags" in response_data
            assert "metadata" in response_data

    def test_upload_large_gmn_specification(self, client, agent_id):
        """Test uploading a large GMN specification."""
        large_spec = {
            "name": "Large Grid World",
            "specification": {
                "nodes": [
                    {"name": "location", "type": "state", "num_states": 100},
                    {
                        "name": "obs_location",
                        "type": "observation",
                        "num_observations": 100,
                    },
                    {"name": "move", "type": "action", "num_actions": 8},
                    {
                        "name": "goal_pref",
                        "type": "preference",
                        "preferred_observation": 99,
                        "preference_strength": 2.0,
                    },
                ],
                "edges": [
                    {
                        "from": "location",
                        "to": "obs_location",
                        "type": "generates",
                    },
                    {
                        "from": "goal_pref",
                        "to": "obs_location",
                        "type": "depends_on",
                    },
                ],
            },
        }

        with patch("database.gmn_repository.GMNRepository") as mock_repo:
            mock_repo_instance = Mock()
            mock_repo.return_value = mock_repo_instance

            mock_gmn_spec = Mock()
            mock_gmn_spec.id = uuid.uuid4()
            mock_gmn_spec.validation_status = ValidationStatus.VALID
            mock_repo_instance.create_gmn_specification.return_value = (
                mock_gmn_spec
            )

            response = client.post(
                f"/api/v1/agents/{agent_id}/gmn",
                json=large_spec,
                headers={"Content-Type": "application/json"},
            )

            assert response.status_code == status.HTTP_201_CREATED

    def test_upload_gmn_error_handling(self, client, agent_id, valid_gmn_spec):
        """Test error handling during GMN upload."""
        with patch("database.gmn_repository.GMNRepository") as mock_repo:
            mock_repo_instance = Mock()
            mock_repo.return_value = mock_repo_instance

            # Simulate database error
            mock_repo_instance.create_gmn_specification.side_effect = (
                Exception("Database error")
            )

            response = client.post(
                f"/api/v1/agents/{agent_id}/gmn",
                json=valid_gmn_spec,
                headers={"Content-Type": "application/json"},
            )

            assert (
                response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    def test_upload_gmn_concurrent_requests(
        self, client, agent_id, valid_gmn_spec
    ):
        """Test handling concurrent GMN upload requests."""
        import threading

        results = []

        def upload_gmn():
            with patch("database.gmn_repository.GMNRepository") as mock_repo:
                mock_repo_instance = Mock()
                mock_repo.return_value = mock_repo_instance

                mock_gmn_spec = Mock()
                mock_gmn_spec.id = uuid.uuid4()
                mock_gmn_spec.validation_status = ValidationStatus.VALID
                mock_repo_instance.create_gmn_specification.return_value = (
                    mock_gmn_spec
                )

                response = client.post(
                    f"/api/v1/agents/{agent_id}/gmn",
                    json=valid_gmn_spec,
                    headers={"Content-Type": "application/json"},
                )
                results.append(response.status_code)

        # Start multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=upload_gmn)
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # At least one should succeed (depending on implementation)
        assert len(results) == 3
        # This test mainly checks that concurrent requests don't crash the system


class TestGMNAPIEndpointGET:
    """Test GET endpoints for GMN specifications."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def agent_id(self):
        """Create test agent ID."""
        return str(uuid.uuid4())

    def test_get_active_gmn_specification(self, client, agent_id):
        """Test getting the active GMN specification for an agent."""
        with patch("database.gmn_repository.GMNRepository") as mock_repo:
            mock_repo_instance = Mock()
            mock_repo.return_value = mock_repo_instance

            mock_gmn_spec = Mock()
            mock_gmn_spec.id = uuid.uuid4()
            mock_gmn_spec.name = "Test Spec"
            mock_gmn_spec.version = "1.0"
            mock_gmn_spec.is_active = True
            mock_gmn_spec.validation_status = ValidationStatus.VALID
            mock_repo_instance.get_active_gmn_specification.return_value = (
                mock_gmn_spec
            )

            response = client.get(f"/api/v1/agents/{agent_id}/gmn/active")

            assert response.status_code == status.HTTP_200_OK
            response_data = response.json()
            assert response_data["name"] == "Test Spec"
            assert response_data["version"] == "1.0"
            assert response_data["is_active"] is True

    def test_get_active_gmn_specification_not_found(self, client, agent_id):
        """Test getting active GMN when none exists."""
        with patch("database.gmn_repository.GMNRepository") as mock_repo:
            mock_repo_instance = Mock()
            mock_repo.return_value = mock_repo_instance
            mock_repo_instance.get_active_gmn_specification.return_value = None

            response = client.get(f"/api/v1/agents/{agent_id}/gmn/active")

            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_list_gmn_specifications_for_agent(self, client, agent_id):
        """Test listing all GMN specifications for an agent."""
        with patch("database.gmn_repository.GMNRepository") as mock_repo:
            mock_repo_instance = Mock()
            mock_repo.return_value = mock_repo_instance

            mock_specs = [
                Mock(
                    id=uuid.uuid4(),
                    name="Spec 1",
                    version="1.0",
                    is_active=True,
                ),
                Mock(
                    id=uuid.uuid4(),
                    name="Spec 2",
                    version="1.1",
                    is_active=False,
                ),
            ]
            mock_repo_instance.get_agent_gmn_specifications.return_value = (
                mock_specs
            )

            response = client.get(f"/api/v1/agents/{agent_id}/gmn")

            assert response.status_code == status.HTTP_200_OK
            response_data = response.json()
            assert len(response_data) == 2
            assert response_data[0]["name"] == "Spec 1"
            assert response_data[1]["name"] == "Spec 2"


class TestGMNAPIEndpointPUT:
    """Test PUT endpoints for GMN specification management."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def agent_id(self):
        """Create test agent ID."""
        return str(uuid.uuid4())

    @pytest.fixture
    def spec_id(self):
        """Create test specification ID."""
        return str(uuid.uuid4())

    def test_activate_gmn_specification(self, client, agent_id, spec_id):
        """Test activating a specific GMN specification."""
        with patch("database.gmn_repository.GMNRepository") as mock_repo:
            mock_repo_instance = Mock()
            mock_repo.return_value = mock_repo_instance
            mock_repo_instance.activate_gmn_specification.return_value = True

            response = client.put(
                f"/api/v1/agents/{agent_id}/gmn/{spec_id}/activate"
            )

            assert response.status_code == status.HTTP_200_OK
            response_data = response.json()
            assert (
                response_data["message"]
                == "GMN specification activated successfully"
            )

    def test_deactivate_gmn_specification(self, client, agent_id, spec_id):
        """Test deactivating a specific GMN specification."""
        with patch("database.gmn_repository.GMNRepository") as mock_repo:
            mock_repo_instance = Mock()
            mock_repo.return_value = mock_repo_instance
            mock_repo_instance.deactivate_gmn_specification.return_value = True

            response = client.put(
                f"/api/v1/agents/{agent_id}/gmn/{spec_id}/deactivate"
            )

            assert response.status_code == status.HTTP_200_OK
            response_data = response.json()
            assert (
                response_data["message"]
                == "GMN specification deactivated successfully"
            )


class TestGMNAPIEndpointDELETE:
    """Test DELETE endpoints for GMN specifications."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def agent_id(self):
        """Create test agent ID."""
        return str(uuid.uuid4())

    @pytest.fixture
    def spec_id(self):
        """Create test specification ID."""
        return str(uuid.uuid4())

    def test_delete_gmn_specification(self, client, agent_id, spec_id):
        """Test deleting a GMN specification."""
        with patch("database.gmn_repository.GMNRepository") as mock_repo:
            mock_repo_instance = Mock()
            mock_repo.return_value = mock_repo_instance
            mock_repo_instance.delete_gmn_specification.return_value = True

            response = client.delete(
                f"/api/v1/agents/{agent_id}/gmn/{spec_id}"
            )

            assert response.status_code == status.HTTP_200_OK
            response_data = response.json()
            assert (
                response_data["message"]
                == "GMN specification deleted successfully"
            )

    def test_delete_nonexistent_gmn_specification(
        self, client, agent_id, spec_id
    ):
        """Test deleting a nonexistent GMN specification."""
        with patch("database.gmn_repository.GMNRepository") as mock_repo:
            mock_repo_instance = Mock()
            mock_repo.return_value = mock_repo_instance
            mock_repo_instance.delete_gmn_specification.return_value = False

            response = client.delete(
                f"/api/v1/agents/{agent_id}/gmn/{spec_id}"
            )

            assert response.status_code == status.HTTP_404_NOT_FOUND
