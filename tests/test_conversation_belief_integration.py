"""
Test PyMDP Belief Integration with Conversations

Tests that conversation messages trigger belief updates which influence agent responses.
Following Kent Beck's TDD approach - this test should initially fail.
"""

from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from api.v1.services.conversation_orchestrator import ConversationOrchestrator
from api.v1.services.pymdp_belief_manager import PyMDPBeliefManager
from inference.active.belief_manager import BeliefStateManager
from inference.active.config import ActiveInferenceConfig


class TestConversationBeliefIntegration:
    """Test suite for conversation-belief integration following TDD principles."""

    @pytest.fixture
    def mock_belief_manager(self):
        """Mock belief manager for testing."""
        manager = Mock(spec=BeliefStateManager)
        manager.update_beliefs.return_value = Mock(
            new_belief_state=Mock(
                entropy=0.5,
                max_confidence=0.7,
                most_likely_state=Mock(return_value=2),
                beliefs=np.array([0.1, 0.2, 0.7]),
            ),
            entropy_change=-0.1,
            kl_divergence=0.2,
        )
        manager.get_current_beliefs.return_value = Mock(
            entropy=0.5, max_confidence=0.7, beliefs=np.array([0.1, 0.2, 0.7])
        )
        return manager

    @pytest.fixture
    def mock_observation_processor(self):
        """Mock observation processor for testing."""
        processor = Mock()
        processor.extract_observation.return_value = 1  # Simple observation
        return processor

    @pytest.fixture
    def mock_repository(self):
        """Mock conversation repository."""
        repository = Mock()
        # Mock conversation with basic structure
        mock_conversation = Mock()
        mock_conversation.turns = []
        mock_conversation.topic = "test topic"
        repository.get_conversation = AsyncMock(return_value=mock_conversation)
        return repository

    @pytest.fixture
    def conversation_orchestrator(self, mock_repository):
        """Create conversation orchestrator with minimal dependencies."""
        # Create orchestrator with mock dependencies
        orchestrator = ConversationOrchestrator(
            repository=mock_repository,
            response_generator=Mock(),
            turn_controller=Mock(),
            event_publisher=Mock(),
            turn_limit_policy=Mock(),
        )
        return orchestrator

    @pytest.mark.asyncio
    async def test_conversation_message_triggers_belief_update(
        self, conversation_orchestrator, mock_belief_manager
    ):
        """Test that incoming conversation messages trigger belief updates."""

        # Arrange
        conversation_id = "test_conv_123"
        agent_id = "test_agent_456"
        message = {
            "content": "I'm feeling uncertain about this topic",
            "role": "user",
            "timestamp": "2025-08-01T10:00:00Z",
        }

        # Mock the belief manager being available
        with patch(
            "api.v1.services.conversation_orchestrator.get_belief_manager"
        ) as mock_get_manager:
            mock_manager = Mock()
            mock_manager.update_beliefs_from_message = AsyncMock(
                return_value={
                    "belief_influenced": True,
                    "belief_status": "success",
                    "belief_metrics": {"entropy": 0.5},
                }
            )
            mock_manager.get_current_belief_context = Mock(
                return_value={
                    "confidence_level": "medium",
                    "belief_context": "I have some insights",
                }
            )
            mock_get_manager.return_value = mock_manager

            # Act - this should trigger belief update
            response = await conversation_orchestrator.process_message(
                conversation_id=conversation_id,
                agent_id=agent_id,
                message=message,
                update_beliefs=True,  # Feature flag for belief integration
            )

            # Assert - belief update should have been called
            mock_manager.update_beliefs_from_message.assert_called_once()

            # Assert - response should include belief influence
            assert "belief_influenced" in response
            assert response["belief_influenced"] is True
            assert "belief_metrics" in response
            assert response["belief_metrics"]["entropy"] == 0.5

    @pytest.mark.asyncio
    async def test_belief_state_influences_response_generation(
        self, conversation_orchestrator, mock_belief_manager
    ):
        """Test that agent beliefs influence the generated response."""

        conversation_id = "test_conv_123"
        agent_id = "test_agent_456"
        message = {"content": "What do you think about this?", "role": "user"}

        # Mock the belief manager with high confidence state
        with patch(
            "api.v1.services.conversation_orchestrator.get_belief_manager"
        ) as mock_get_manager:
            mock_manager = Mock()
            mock_manager.update_beliefs_from_message = AsyncMock(
                return_value={"belief_influenced": True, "belief_status": "success"}
            )
            mock_manager.get_current_belief_context = Mock(
                return_value={
                    "confidence_level": "high",
                    "belief_context": "I'm quite confident in my understanding",
                }
            )
            mock_get_manager.return_value = mock_manager

            # Act
            response = await conversation_orchestrator.process_message(
                conversation_id=conversation_id,
                agent_id=agent_id,
                message=message,
                update_beliefs=True,
            )

            # Assert - response should reflect high confidence
            assert "confidence_level" in response
            assert response["confidence_level"] == "high"
            assert "belief_context" in response
            assert "confident" in response["belief_context"].lower()

    @pytest.mark.asyncio
    async def test_belief_update_failure_does_not_break_conversation(
        self, conversation_orchestrator, mock_belief_manager
    ):
        """Test error resilience - belief failures don't break conversations."""

        conversation_id = "test_conv_123"
        agent_id = "test_agent_456"
        message = {"content": "Hello", "role": "user"}

        # Mock the belief manager to fail
        with patch(
            "api.v1.services.conversation_orchestrator.get_belief_manager"
        ) as mock_get_manager:
            mock_manager = Mock()
            mock_manager.update_beliefs_from_message = AsyncMock(
                side_effect=Exception("PyMDP computation failed")
            )
            mock_get_manager.return_value = mock_manager

            # Act - should not raise exception
            response = await conversation_orchestrator.process_message(
                conversation_id=conversation_id,
                agent_id=agent_id,
                message=message,
                update_beliefs=True,
            )

            # Assert - conversation continues despite belief failure
            assert response is not None
            assert response["belief_influenced"] is False
            assert "error" not in response  # Error handled gracefully
            assert "belief_status" in response
            assert response["belief_status"] == "failed"

    @pytest.mark.asyncio
    async def test_belief_integration_can_be_disabled(
        self, conversation_orchestrator, mock_belief_manager
    ):
        """Test that belief integration can be disabled via feature flag."""

        conversation_id = "test_conv_123"
        agent_id = "test_agent_456"
        message = {"content": "Hello", "role": "user"}

        # Act - belief integration disabled
        response = await conversation_orchestrator.process_message(
            conversation_id=conversation_id,
            agent_id=agent_id,
            message=message,
            update_beliefs=False,  # Disabled
        )

        # Assert - no belief updates called and status is disabled
        assert response["belief_influenced"] is False
        assert response["belief_status"] == "disabled"

    def test_belief_manager_initialization(self):
        """Test PyMDP belief manager can be initialized."""

        # This will fail initially because PyMDPBeliefManager doesn't exist
        config = ActiveInferenceConfig(num_states=3, num_observations=3, planning_horizon=5)

        # This should work once we implement the service
        manager = PyMDPBeliefManager(config=config, agent_id="test_agent")

        assert manager is not None
        assert manager.agent_id == "test_agent"
        assert manager.config == config

    def test_observation_extraction_from_message(self):
        """Test that conversation messages can be converted to PyMDP observations."""
        from api.v1.services.observation_processor import ObservationProcessor

        processor = ObservationProcessor()

        # Test different message types
        uncertain_message = {"content": "I'm not sure about this", "role": "user"}
        confident_message = {"content": "I'm absolutely certain", "role": "user"}

        uncertain_obs = processor.extract_observation(uncertain_message)
        confident_obs = processor.extract_observation(confident_message)

        # Different messages should produce different observations
        assert uncertain_obs != confident_obs
        assert isinstance(uncertain_obs, int)
        assert 0 <= uncertain_obs < 3  # Valid observation range
