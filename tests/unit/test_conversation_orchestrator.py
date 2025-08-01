"""
Test suite for Conversation Orchestrator Architecture

This test suite follows Kent Beck's TDD philosophy with behavior-driven tests
and Martin Fowler's domain modeling approach. Tests are organized to validate
the complete conversation orchestration architecture.
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from api.v1.models.agent_conversation import (
    AgentRole,
    CompletionReason,
    ConversationAggregate,
    ConversationConfig,
    ConversationContext,
    ConversationStatus,
    ConversationTurnDomain,
    TurnStatus,
)
from api.v1.services.conversation_orchestrator import ConversationOrchestrator
from api.v1.services.conversation_policies import ConsensusDetectionPolicy, DefaultTurnLimitPolicy


class TestConversationOrchestrator:
    """Test the core conversation orchestrator behavior."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for orchestrator."""
        repository = AsyncMock()
        response_generator = AsyncMock()
        turn_controller = AsyncMock()
        event_publisher = AsyncMock()
        turn_limit_policy = MagicMock()
        completion_detector = AsyncMock()

        return {
            "repository": repository,
            "response_generator": response_generator,
            "turn_controller": turn_controller,
            "event_publisher": event_publisher,
            "turn_limit_policy": turn_limit_policy,
            "completion_detector": completion_detector,
        }

    @pytest.fixture
    def orchestrator(self, mock_dependencies):
        """Create orchestrator with mock dependencies."""
        return ConversationOrchestrator(**mock_dependencies)

    @pytest.fixture
    def sample_agents(self):
        """Create sample agent roles for testing."""
        return [
            AgentRole(
                agent_id="agent_1",
                name="Alice",
                role="advocate",
                system_prompt="You are an advocate who supports ideas",
                turn_order=1,
            ),
            AgentRole(
                agent_id="agent_2",
                name="Bob",
                role="critic",
                system_prompt="You are a critic who questions ideas",
                turn_order=2,
            ),
        ]

    @pytest.mark.asyncio
    async def test_create_conversation_with_valid_participants(
        self, orchestrator, sample_agents, mock_dependencies
    ):
        """Test creating a conversation with valid participants."""
        # Arrange
        user_id = "user_123"
        topic = "Discuss renewable energy"
        config = ConversationConfig(max_turns=5)

        # Act
        conversation = await orchestrator.create_conversation(
            user_id=user_id,
            topic=topic,
            participants=sample_agents,
            config=config,
        )

        # Assert
        assert conversation.user_id == user_id
        assert conversation.topic == topic
        assert len(conversation.participants) == 2
        assert conversation.status == ConversationStatus.CREATED
        assert conversation.config.max_turns == 5

        # Verify repository was called
        mock_dependencies["repository"].save_conversation.assert_called_once_with(conversation)

        # Verify event was published
        mock_dependencies["event_publisher"].publish_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_conversation_fails_with_no_participants(self, orchestrator):
        """Test that creating a conversation fails with no participants."""
        with pytest.raises(ValueError, match="at least one participant"):
            await orchestrator.create_conversation(
                user_id="user_123",
                topic="Test topic",
                participants=[],
            )

    @pytest.mark.asyncio
    async def test_create_conversation_fails_with_too_many_participants(self, orchestrator):
        """Test that creating a conversation fails with too many participants."""
        # Create 11 agents (exceeds limit of 10)
        too_many_agents = [
            AgentRole(
                agent_id=f"agent_{i}",
                name=f"Agent{i}",
                role="participant",
                system_prompt="System prompt",
            )
            for i in range(11)
        ]

        with pytest.raises(ValueError, match="more than 10 participants"):
            await orchestrator.create_conversation(
                user_id="user_123",
                topic="Test topic",
                participants=too_many_agents,
            )

    @pytest.mark.asyncio
    async def test_start_conversation_changes_status_to_active(
        self, orchestrator, sample_agents, mock_dependencies
    ):
        """Test starting a conversation changes status to ACTIVE."""
        # Arrange
        conversation = ConversationAggregate(
            user_id="user_123",
            topic="Test topic",
            participants=sample_agents,
        )
        mock_dependencies["repository"].get_conversation.return_value = conversation

        # Act
        result = await orchestrator.start_conversation(conversation.conversation_id)

        # Assert
        assert result.status == ConversationStatus.ACTIVE
        assert result.started_at is not None
        mock_dependencies["repository"].save_conversation.assert_called()

    @pytest.mark.asyncio
    async def test_pause_active_conversation(self, orchestrator, sample_agents, mock_dependencies):
        """Test pausing an active conversation."""
        # Arrange
        conversation = ConversationAggregate(
            user_id="user_123",
            topic="Test topic",
            participants=sample_agents,
            status=ConversationStatus.ACTIVE,
            started_at=datetime.now(),
        )
        mock_dependencies["repository"].get_conversation.return_value = conversation

        # Act
        result = await orchestrator.pause_conversation(conversation.conversation_id)

        # Assert
        assert result.status == ConversationStatus.PAUSED
        mock_dependencies["repository"].save_conversation.assert_called()

    @pytest.mark.asyncio
    async def test_resume_paused_conversation(self, orchestrator, sample_agents, mock_dependencies):
        """Test resuming a paused conversation."""
        # Arrange
        conversation = ConversationAggregate(
            user_id="user_123",
            topic="Test topic",
            participants=sample_agents,
            status=ConversationStatus.PAUSED,
            started_at=datetime.now(),
        )
        mock_dependencies["repository"].get_conversation.return_value = conversation

        # Act
        result = await orchestrator.resume_conversation(conversation.conversation_id)

        # Assert
        assert result.status == ConversationStatus.ACTIVE
        mock_dependencies["repository"].save_conversation.assert_called()

    @pytest.mark.asyncio
    async def test_stop_conversation_completes_with_reason(
        self, orchestrator, sample_agents, mock_dependencies
    ):
        """Test stopping a conversation completes it with the specified reason."""
        # Arrange
        conversation = ConversationAggregate(
            user_id="user_123",
            topic="Test topic",
            participants=sample_agents,
            status=ConversationStatus.ACTIVE,
            started_at=datetime.now(),
        )
        mock_dependencies["repository"].get_conversation.return_value = conversation

        # Act
        result = await orchestrator.stop_conversation(
            conversation.conversation_id, CompletionReason.MANUAL_STOP
        )

        # Assert
        assert result.status == ConversationStatus.COMPLETED
        assert result.completion_reason == CompletionReason.MANUAL_STOP
        assert result.completed_at is not None
        mock_dependencies["repository"].save_conversation.assert_called()


class TestConversationAggregate:
    """Test the conversation domain aggregate business rules."""

    @pytest.fixture
    def sample_agents(self):
        """Create sample agent roles."""
        return [
            AgentRole(
                agent_id="agent_1",
                name="Alice",
                role="advocate",
                system_prompt="You are an advocate",
                turn_order=1,
            ),
            AgentRole(
                agent_id="agent_2",
                name="Bob",
                role="critic",
                system_prompt="You are a critic",
                turn_order=2,
            ),
        ]

    def test_conversation_creation_with_valid_data(self, sample_agents):
        """Test creating a conversation with valid data."""
        conversation = ConversationAggregate(
            user_id="user_123",
            topic="Test topic",
            participants=sample_agents,
        )

        assert conversation.user_id == "user_123"
        assert conversation.topic == "Test topic"
        assert len(conversation.participants) == 2
        assert conversation.status == ConversationStatus.CREATED
        assert len(conversation.turns) == 0

    def test_conversation_validates_participants_not_empty(self):
        """Test that conversation validates participants list is not empty."""
        with pytest.raises(ValueError, match="at least 1 participant"):
            ConversationAggregate(
                user_id="user_123",
                topic="Test topic",
                participants=[],
            )

    def test_conversation_validates_participants_not_too_many(self):
        """Test that conversation validates participants list is not too large."""
        too_many_agents = [
            AgentRole(
                agent_id=f"agent_{i}",
                name=f"Agent{i}",
                role="participant",
                system_prompt="System prompt",
            )
            for i in range(11)
        ]

        with pytest.raises(ValueError, match="more than 10 participants"):
            ConversationAggregate(
                user_id="user_123",
                topic="Test topic",
                participants=too_many_agents,
            )

    def test_start_conversation_business_rule(self, sample_agents):
        """Test the business rule for starting conversations."""
        conversation = ConversationAggregate(
            user_id="user_123",
            topic="Test topic",
            participants=sample_agents,
        )

        # Should be able to start from CREATED status
        conversation.start_conversation()
        assert conversation.status == ConversationStatus.ACTIVE
        assert conversation.started_at is not None

        # Should not be able to start again
        with pytest.raises(ValueError, match="Cannot start conversation in status"):
            conversation.start_conversation()

    def test_pause_conversation_business_rule(self, sample_agents):
        """Test the business rule for pausing conversations."""
        conversation = ConversationAggregate(
            user_id="user_123",
            topic="Test topic",
            participants=sample_agents,
            status=ConversationStatus.ACTIVE,
            started_at=datetime.now(),
        )

        # Should be able to pause from ACTIVE status
        conversation.pause_conversation()
        assert conversation.status == ConversationStatus.PAUSED

        # Should not be able to pause from PAUSED status
        with pytest.raises(ValueError, match="Cannot pause conversation in status"):
            conversation.pause_conversation()

    def test_resume_conversation_business_rule(self, sample_agents):
        """Test the business rule for resuming conversations."""
        conversation = ConversationAggregate(
            user_id="user_123",
            topic="Test topic",
            participants=sample_agents,
            status=ConversationStatus.PAUSED,
            started_at=datetime.now(),
        )

        # Should be able to resume from PAUSED status
        conversation.resume_conversation()
        assert conversation.status == ConversationStatus.ACTIVE

        # Should not be able to resume from ACTIVE status
        with pytest.raises(ValueError, match="Cannot resume conversation in status"):
            conversation.resume_conversation()

    def test_add_turn_validates_sequence(self, sample_agents):
        """Test that adding turns validates the sequence number."""
        conversation = ConversationAggregate(
            user_id="user_123",
            topic="Test topic",
            participants=sample_agents,
            status=ConversationStatus.ACTIVE,
            started_at=datetime.now(),
        )

        # First turn should be turn number 1
        turn1 = ConversationTurnDomain(
            turn_number=1,
            agent_id="agent_1",
            agent_name="Alice",
            prompt="Test prompt",
            response="Test response",
            status=TurnStatus.COMPLETED,
        )

        conversation.add_turn(turn1)
        assert len(conversation.turns) == 1

        # Second turn should be turn number 2
        turn2 = ConversationTurnDomain(
            turn_number=2,
            agent_id="agent_2",
            agent_name="Bob",
            prompt="Test prompt",
            response="Test response",
            status=TurnStatus.COMPLETED,
        )

        conversation.add_turn(turn2)
        assert len(conversation.turns) == 2

        # Invalid turn number should fail
        invalid_turn = ConversationTurnDomain(
            turn_number=5,  # Should be 3
            agent_id="agent_1",
            agent_name="Alice",
            prompt="Test prompt",
            response="Test response",
            status=TurnStatus.COMPLETED,
        )

        with pytest.raises(ValueError, match="Invalid turn number"):
            conversation.add_turn(invalid_turn)

    def test_next_agent_round_robin(self, sample_agents):
        """Test that next_agent returns agents in round-robin order."""
        conversation = ConversationAggregate(
            user_id="user_123",
            topic="Test topic",
            participants=sample_agents,
        )

        # First turn should be Alice (turn_order=1)
        next_agent = conversation.next_agent
        assert next_agent.name == "Alice"

        # Add a turn for Alice
        turn1 = ConversationTurnDomain(
            turn_number=1,
            agent_id="agent_1",
            agent_name="Alice",
            prompt="Test prompt",
            response="Test response",
            status=TurnStatus.COMPLETED,
        )
        conversation.status = ConversationStatus.ACTIVE
        conversation.add_turn(turn1)

        # Second turn should be Bob (turn_order=2)
        next_agent = conversation.next_agent
        assert next_agent.name == "Bob"


class TestTurnLimitPolicy:
    """Test turn limit policy implementations."""

    def test_default_policy_respects_turn_limit(self):
        """Test that default policy respects the configured turn limit."""
        policy = DefaultTurnLimitPolicy()

        # Create conversation with max 3 turns
        config = ConversationConfig(max_turns=3)
        conversation = ConversationAggregate(
            user_id="user_123",
            topic="Test topic",
            participants=[
                AgentRole(
                    agent_id="agent_1",
                    name="Alice",
                    role="advocate",
                    system_prompt="System prompt",
                )
            ],
            config=config,
            status=ConversationStatus.ACTIVE,
        )

        # Should continue with fewer than max turns
        assert policy.should_continue(conversation) is True

        # Add 3 turns (at limit)
        for i in range(3):
            turn = ConversationTurnDomain(
                turn_number=i + 1,
                agent_id="agent_1",
                agent_name="Alice",
                prompt="Test prompt",
                response="Test response",
                status=TurnStatus.COMPLETED,
            )
            conversation.add_turn(turn)

        # Should not continue after reaching limit
        assert policy.should_continue(conversation) is False
        assert policy.get_completion_reason(conversation) == CompletionReason.TURN_LIMIT_REACHED

    def test_default_policy_respects_timeout(self):
        """Test that default policy respects conversation timeout."""
        policy = DefaultTurnLimitPolicy()

        # Create conversation with 1-minute timeout
        config = ConversationConfig(conversation_timeout_minutes=1)
        past_time = datetime.now() - timedelta(minutes=2)  # 2 minutes ago

        conversation = ConversationAggregate(
            user_id="user_123",
            topic="Test topic",
            participants=[
                AgentRole(
                    agent_id="agent_1",
                    name="Alice",
                    role="advocate",
                    system_prompt="System prompt",
                )
            ],
            config=config,
            status=ConversationStatus.ACTIVE,
            started_at=past_time,
        )

        # Should not continue due to timeout
        assert policy.should_continue(conversation) is False
        assert policy.get_completion_reason(conversation) == CompletionReason.TIMEOUT


class TestConsensusDetectionPolicy:
    """Test consensus detection policy."""

    @pytest.mark.asyncio
    async def test_detects_consensus_with_agreement_keywords(self):
        """Test consensus detection with agreement keywords."""
        detector = ConsensusDetectionPolicy(consensus_threshold=0.7)

        # Create context with consensus responses
        participants = [
            AgentRole(
                agent_id="1",
                name="Alice",
                role="advocate",
                system_prompt="You are an advocate agent",
            ),
            AgentRole(
                agent_id="2", name="Bob", role="critic", system_prompt="You are a critic agent"
            ),
        ]

        turns = [
            ConversationTurnDomain(
                turn_number=1,
                agent_id="1",
                agent_name="Alice",
                prompt="prompt",
                response="I agree with this approach, it's exactly right",
                status=TurnStatus.COMPLETED,
            ),
            ConversationTurnDomain(
                turn_number=2,
                agent_id="2",
                agent_name="Bob",
                prompt="prompt",
                response="Yes, I concur completely, this is correct",
                status=TurnStatus.COMPLETED,
            ),
        ]

        context = ConversationContext(
            conversation_id="conv_123",
            topic="Test topic",
            participants=participants,
            turn_history=turns,
        )

        # Should detect consensus
        result = await detector.detect_completion(context)
        assert result == CompletionReason.CONSENSUS_REACHED

    @pytest.mark.asyncio
    async def test_does_not_detect_consensus_with_disagreement(self):
        """Test that consensus is not detected with disagreement keywords."""
        detector = ConsensusDetectionPolicy(consensus_threshold=0.7)

        participants = [
            AgentRole(
                agent_id="1",
                name="Alice",
                role="advocate",
                system_prompt="You are an advocate agent",
            ),
            AgentRole(
                agent_id="2", name="Bob", role="critic", system_prompt="You are a critic agent"
            ),
        ]

        turns = [
            ConversationTurnDomain(
                turn_number=1,
                agent_id="1",
                agent_name="Alice",
                prompt="prompt",
                response="I think this is a good approach",
                status=TurnStatus.COMPLETED,
            ),
            ConversationTurnDomain(
                turn_number=2,
                agent_id="2",
                agent_name="Bob",
                prompt="prompt",
                response="I disagree, this is wrong and we should do something different",
                status=TurnStatus.COMPLETED,
            ),
        ]

        context = ConversationContext(
            conversation_id="conv_123",
            topic="Test topic",
            participants=participants,
            turn_history=turns,
        )

        # Should not detect consensus
        result = await detector.detect_completion(context)
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
