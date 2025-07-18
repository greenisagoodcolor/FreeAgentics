"""Test suite for Agent Factory service."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from services.agent_factory import AgentFactory


class TestAgentFactory:
    """Test the agent factory service."""

    @pytest.fixture
    def factory(self):
        """Create agent factory instance."""
        return AgentFactory()

    @pytest.fixture
    def valid_model(self):
        """Create a valid PyMDP model."""
        return {
            "num_states": [4],
            "num_obs": [5],
            "num_controls": [3],
            "planning_horizon": 2,
        }

    @pytest.mark.asyncio
    async def test_validate_model_success(self, factory, valid_model):
        """Test successful model validation."""
        is_valid, errors = await factory.validate_model(valid_model)

        assert is_valid is True
        assert errors == []

    @pytest.mark.asyncio
    async def test_validate_model_missing_fields(self, factory):
        """Test validation with missing required fields."""
        invalid_model = {
            "num_states": [4],
            # Missing num_obs and num_controls
        }

        is_valid, errors = await factory.validate_model(invalid_model)

        assert is_valid is False
        assert any("num_obs" in error for error in errors)
        assert any("num_controls" in error for error in errors)

    @pytest.mark.asyncio
    async def test_validate_model_invalid_dimensions(self, factory):
        """Test validation with invalid dimensions."""
        invalid_models = [
            {"num_states": "not_a_list", "num_obs": [5], "num_controls": [3]},
            {
                "num_states": [0],
                "num_obs": [5],
                "num_controls": [3],
            },  # Zero size
            {
                "num_states": [-1],
                "num_obs": [5],
                "num_controls": [3],
            },  # Negative
            {
                "num_states": [4.5],
                "num_obs": [5],
                "num_controls": [3],
            },  # Non-integer
        ]

        for model in invalid_models:
            is_valid, errors = await factory.validate_model(model)
            assert is_valid is False
            assert len(errors) > 0

    @pytest.mark.asyncio
    async def test_validate_model_with_matrices(self, factory):
        """Test validation with matrix specifications."""
        model_with_matrices = {
            "num_states": [4],
            "num_obs": [5],
            "num_controls": [3],
            "A": np.ones((5, 4)) / 4,  # Valid A matrix
            "B": np.ones((4, 4, 3)) / 4,  # Valid B matrix
            "C": np.zeros(5),  # Valid C matrix
            "D": np.ones(4) / 4,  # Valid D matrix
        }

        is_valid, errors = await factory.validate_model(model_with_matrices)

        assert is_valid is True
        assert errors == []

    @pytest.mark.asyncio
    async def test_validate_a_matrix_wrong_shape(self, factory):
        """Test A matrix validation with wrong shape."""
        model = {
            "num_states": [4],
            "num_obs": [5],
            "num_controls": [3],
            "A": np.ones((3, 4)),  # Wrong shape: should be (5, 4)
        }

        is_valid, errors = await factory.validate_model(model)

        assert is_valid is False
        assert any("A matrix shape" in error for error in errors)

    @pytest.mark.asyncio
    async def test_validate_b_matrix_not_normalized(self, factory):
        """Test B matrix validation for normalization."""
        model = {
            "num_states": [4],
            "num_obs": [5],
            "num_controls": [3],
            "B": np.ones((4, 4, 3)),  # Not normalized
        }

        is_valid, errors = await factory.validate_model(model)

        assert is_valid is False
        assert any("sum to 1" in error for error in errors)

    @pytest.mark.asyncio
    async def test_create_agent_success(self, factory, valid_model):
        """Test successful agent creation."""
        agent_id = "test_agent_123"
        metadata = {"created_by": "test"}

        agent = await factory.create_from_gmn_model(
            valid_model, agent_id, metadata
        )

        assert agent is not None
        assert agent.id == agent_id
        assert agent.metadata == metadata
        assert hasattr(agent, "qs")  # Has beliefs

    @pytest.mark.asyncio
    async def test_create_agent_invalid_model(self, factory):
        """Test agent creation with invalid model."""
        invalid_model = {"num_states": [4]}  # Missing required fields

        with pytest.raises(ValueError, match="Model validation failed"):
            await factory.create_from_gmn_model(invalid_model, "test_agent")

    @pytest.mark.asyncio
    async def test_create_agent_with_custom_parameters(self, factory):
        """Test agent creation with custom parameters."""
        model = {
            "num_states": [4],
            "num_obs": [5],
            "num_controls": [3],
            "planning_horizon": 5,
            "inference_algo": "mmp",
            "policy_len": 3,
        }

        agent = await factory.create_from_gmn_model(model, "test_agent")

        # Check that custom parameters were applied
        assert agent.gmn_model["planning_horizon"] == 5
        assert agent.gmn_model["inference_algo"] == "mmp"
        assert agent.gmn_model["policy_len"] == 3

    def test_create_a_matrix_default(self, factory):
        """Test default A matrix creation."""
        A_matrices = factory._create_A_matrix(None, [5], [4])

        assert len(A_matrices) == 1
        assert A_matrices[0].shape == (5, 4)
        # Check normalization
        assert np.allclose(A_matrices[0].sum(axis=0), 1.0)

    def test_create_a_matrix_identity(self, factory):
        """Test A matrix creation for matching dimensions."""
        A_matrices = factory._create_A_matrix(None, [4], [4])

        assert len(A_matrices) == 1
        # Should be close to identity
        assert A_matrices[0].shape == (4, 4)
        assert np.trace(A_matrices[0]) > 2  # Strong diagonal

    def test_create_b_matrix_cardinal_directions(self, factory):
        """Test B matrix creation for 4 actions (cardinal directions)."""
        B_matrices = factory._create_B_matrix(None, [5], [4])

        assert len(B_matrices) == 1
        assert B_matrices[0].shape == (5, 5, 4)

        # Check normalization for each action
        for a in range(4):
            assert np.allclose(B_matrices[0][:, :, a].sum(axis=0), 1.0)

    def test_create_c_matrix_default(self, factory):
        """Test default C matrix creation."""
        C_matrices = factory._create_C_matrix(None, [5])

        assert len(C_matrices) == 1
        assert C_matrices[0].shape == (5,)
        assert np.allclose(C_matrices[0], 0)  # Neutral preferences

    def test_create_d_matrix_uniform(self, factory):
        """Test uniform initial distribution creation."""
        D_matrices = factory._create_D_matrix(None, [4])

        assert len(D_matrices) == 1
        assert D_matrices[0].shape == (4,)
        assert np.allclose(D_matrices[0], 0.25)  # Uniform over 4 states
        assert np.allclose(D_matrices[0].sum(), 1.0)

    @pytest.mark.asyncio
    async def test_multi_factor_model(self, factory):
        """Test creation of multi-factor agents."""
        multi_factor_model = {
            "num_states": [3, 4],  # Two state factors
            "num_obs": [5, 3],  # Two observation modalities
            "num_controls": [2, 3],  # Two control factors
        }

        is_valid, errors = await factory.validate_model(multi_factor_model)
        assert is_valid is True

        agent = await factory.create_from_gmn_model(
            multi_factor_model, "multi_factor_agent"
        )

        assert agent is not None
        assert len(agent.qs) == 2  # Two belief factors

    @pytest.mark.asyncio
    async def test_agent_creation_error_handling(self, factory):
        """Test error handling during agent creation."""
        model = {"num_states": [4], "num_obs": [5], "num_controls": [3]}

        # Mock Agent class to raise an error
        with patch('services.agent_factory.Agent') as MockAgent:
            MockAgent.side_effect = Exception("Creation failed")

            with pytest.raises(RuntimeError, match="Agent creation failed"):
                await factory.create_from_gmn_model(model, "test_agent")

    def test_validate_c_matrix_optional(self, factory):
        """Test that C matrix is optional."""
        errors = factory._validate_C_matrix(None, [5])
        assert errors == []

    def test_validate_d_matrix_not_normalized(self, factory):
        """Test D matrix validation for normalization."""
        D = np.array([0.1, 0.2, 0.3, 0.1])  # Sums to 0.7, not 1.0
        errors = factory._validate_D_matrix(D, [4])

        assert any("sum to 1" in error for error in errors)
