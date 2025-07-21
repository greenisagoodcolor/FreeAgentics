"""Integration test for the complete prompt → agent flow."""

import json
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from inference.llm.provider_interface import GenerationResponse, ProviderStatus


class TestPromptToAgentFlow:
    """Test the complete flow from prompt to agent creation."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        """Mock authentication headers."""
        # In a real test, this would get a valid JWT token
        return {"Authorization": "Bearer test-token"}

    @pytest.fixture
    def mock_llm_response(self):
        """Mock LLM response with valid GMN."""
        gmn_spec = {
            "name": "test_explorer",
            "description": "An agent that explores a grid world",
            "states": ["exploring", "found_target", "avoiding_obstacle"],
            "observations": ["empty", "target", "obstacle", "boundary"],
            "actions": ["move_up", "move_down", "move_left", "move_right", "stay"],
            "parameters": {
                "A": [
                    [0.8, 0.1, 0.1],  # P(empty|exploring)
                    [0.1, 0.8, 0.1],  # P(target|found_target)
                    [0.1, 0.1, 0.8],  # P(obstacle|avoiding_obstacle)
                ],
                "B": [
                    [  # Transitions for move_up
                        [0.7, 0.2, 0.1],
                        [0.1, 0.7, 0.2],
                        [0.2, 0.1, 0.7],
                    ],
                    [  # Transitions for move_down
                        [0.7, 0.2, 0.1],
                        [0.1, 0.7, 0.2],
                        [0.2, 0.1, 0.7],
                    ],
                    [  # Transitions for move_left
                        [0.7, 0.2, 0.1],
                        [0.1, 0.7, 0.2],
                        [0.2, 0.1, 0.7],
                    ],
                    [  # Transitions for move_right
                        [0.7, 0.2, 0.1],
                        [0.1, 0.7, 0.2],
                        [0.2, 0.1, 0.7],
                    ],
                    [  # Transitions for stay
                        [0.9, 0.05, 0.05],
                        [0.05, 0.9, 0.05],
                        [0.05, 0.05, 0.9],
                    ],
                ],
                "C": [[0.1, 0.8, 0.05, 0.05]],  # Prefer finding target
                "D": [[0.8, 0.1, 0.1]],  # Start in exploring state
            },
        }

        return GenerationResponse(
            content=json.dumps(gmn_spec),
            model="gpt-3.5-turbo",
            usage={"input_tokens": 100, "output_tokens": 500, "total_tokens": 600},
            provider_status=ProviderStatus.HEALTHY,
            raw_response={},
        )

    @patch("api.v1.prompts.get_current_user")
    @patch("api.v1.prompts.llm_factory.create_from_config")
    def test_create_agent_from_prompt(
        self, mock_llm_factory, mock_auth, client, mock_llm_response
    ):
        """Test creating an agent from a natural language prompt."""
        # Mock authentication
        mock_auth.return_value = Mock(user_id="test-user", permissions=["CREATE_AGENT"])

        # Mock LLM provider
        mock_provider = Mock()
        mock_provider.generate.return_value = mock_llm_response
        mock_provider.get_provider_type.return_value.value = "openai"

        mock_manager = Mock()
        mock_manager.get_best_available_provider.return_value = mock_provider
        mock_llm_factory.return_value = mock_manager

        # Test request
        response = client.post(
            "/api/v1/prompts",
            json={
                "prompt": "Create an agent that explores a grid world to find hidden rewards",
                "agent_name": "explorer_bot",
                "llm_provider": "openai",
            },
            headers={"Authorization": "Bearer test-token"},
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()

        assert data["agent_name"] == "explorer_bot"
        assert data["status"] == "active"
        assert "agent_id" in data
        assert "gmn_spec" in data
        assert "pymdp_model" in data
        assert data["llm_provider_used"] == "openai"

        # Verify GMN spec structure
        gmn = data["gmn_spec"]
        assert gmn["name"] == "test_explorer"
        assert len(gmn["states"]) == 3
        assert len(gmn["observations"]) == 4
        assert len(gmn["actions"]) == 5

        # Verify PyMDP model was created
        pymdp = data["pymdp_model"]
        assert "A" in pymdp
        assert "B" in pymdp
        assert "C" in pymdp
        assert "D" in pymdp

    @patch("api.v1.prompts.get_current_user")
    @patch("api.v1.prompts.llm_factory.create_from_config")
    def test_retry_on_invalid_gmn(self, mock_llm_factory, mock_auth, client):
        """Test that the system retries when LLM generates invalid GMN."""
        # Mock authentication
        mock_auth.return_value = Mock(user_id="test-user", permissions=["CREATE_AGENT"])

        # First response: invalid GMN (probabilities don't sum to 1)
        invalid_gmn = {
            "name": "bad_agent",
            "states": ["s1", "s2"],
            "observations": ["o1", "o2"],
            "actions": ["a1"],
            "parameters": {
                "A": [[0.5, 0.3]],  # Doesn't sum to 1!
                "B": [[[1.0, 0.0], [0.0, 1.0]]],
                "C": [[0.5, 0.5]],
                "D": [[0.5, 0.5]],
            },
        }

        # Second response: valid GMN
        valid_gmn = {
            "name": "good_agent",
            "states": ["s1", "s2"],
            "observations": ["o1", "o2"],
            "actions": ["a1"],
            "parameters": {
                "A": [[0.7, 0.3]],  # Sums to 1
                "B": [[[1.0, 0.0], [0.0, 1.0]]],
                "C": [[0.5, 0.5]],
                "D": [[0.5, 0.5]],
            },
        }

        mock_provider = Mock()
        mock_provider.generate.side_effect = [
            GenerationResponse(
                content=json.dumps(invalid_gmn),
                model="gpt-3.5-turbo",
                usage={"input_tokens": 100, "output_tokens": 200, "total_tokens": 300},
                provider_status=ProviderStatus.HEALTHY,
                raw_response={},
            ),
            GenerationResponse(
                content=json.dumps(valid_gmn),
                model="gpt-3.5-turbo",
                usage={"input_tokens": 150, "output_tokens": 200, "total_tokens": 350},
                provider_status=ProviderStatus.HEALTHY,
                raw_response={},
            ),
        ]
        mock_provider.get_provider_type.return_value.value = "openai"

        mock_manager = Mock()
        mock_manager.get_best_available_provider.return_value = mock_provider
        mock_llm_factory.return_value = mock_manager

        # Test request with retry enabled
        response = client.post(
            "/api/v1/prompts",
            json={"prompt": "Create a simple agent", "max_retries": 2},
            headers={"Authorization": "Bearer test-token"},
        )

        # Should succeed on retry
        assert response.status_code == 200
        data = response.json()
        assert data["gmn_spec"]["name"] == "good_agent"

        # Verify LLM was called twice
        assert mock_provider.generate.call_count == 2

    def test_get_prompt_examples(self, client):
        """Test getting example prompts."""
        response = client.get("/api/v1/prompts/examples")

        assert response.status_code == 200
        data = response.json()

        assert "examples" in data
        assert len(data["examples"]) > 0

        # Check example structure
        example = data["examples"][0]
        assert "name" in example
        assert "prompt" in example
        assert "description" in example
