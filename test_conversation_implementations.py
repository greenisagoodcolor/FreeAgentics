"""
Comprehensive tests for Conversation Implementation

Following TDD principles with comprehensive coverage of all conversation flow
control components: turn controller, event publisher, policies, and completion detection.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest

from api.v1.models.agent_conversation import (
    AgentRole,
    CompletionReason,
    ConversationAggregate,
    ConversationContext,
    ConversationEvent,
    ConversationTurnDomain,
    TurnStatus,
)
from api.v1.services.conversation_implementations import (
    ConsensusCompletionDetector,
    ConversationEventPublisher,
    ConversationTurnController,
    TurnLimitPolicy,
    create_completion_detector,
    create_event_publisher,
    create_turn_controller,
    create_turn_limit_policy,
)


class TestConversationTurnController:
    """Test suite for ConversationTurnController."""

    @pytest.fixture
    def mock_response_generator(self):
        """Create mock response generator."""
        mock_generator = Mock()
        mock_generator.generate_response = AsyncMock()
        return mock_generator

    @pytest.fixture
    def mock_event_publisher(self):
        """Create mock event publisher."""
        mock_publisher = Mock()
        mock_publisher.publish_event = AsyncMock()
        return mock_publisher

    @pytest.fixture
    def test_agent(self):
        """Create test agent."""
        return AgentRole(
            agent_id="test-agent",
            name="TestAgent",
            role="test participant",
            system_prompt="You participate in test conversations.",
        )

    @pytest.fixture
    def test_context(self, test_agent):
        """Create test conversation context."""
        return ConversationContext(
            conversation_id="test-conversation",
            topic="Test conversation topic",
            participants=[test_agent],
        )

    @pytest.fixture
    def test_turn(self):
        """Create test turn."""
        return ConversationTurnDomain(
            turn_number=1,
            agent_id="test-agent",
            agent_name="TestAgent",
            prompt="Test prompt for turn execution",
        )

    def test_controller_initialization(self, mock_response_generator):
        """Test turn controller initializes correctly."""
        controller = ConversationTurnController(mock_response_generator)

        assert controller.response_generator == mock_response_generator
        assert controller.event_publisher is None
        assert controller.metrics["turns_executed"] == 0
        assert controller.metrics["turns_successful"] == 0

    def test_controller_with_event_publisher(self, mock_response_generator, mock_event_publisher):
        """Test turn controller with event publisher."""
        controller = ConversationTurnController(mock_response_generator, mock_event_publisher)

        assert controller.response_generator == mock_response_generator
        assert controller.event_publisher == mock_event_publisher

    @pytest.mark.asyncio
    async def test_successful_turn_execution(
        self, mock_response_generator, mock_event_publisher, test_agent, test_context, test_turn
    ):
        """Test successful turn execution flow."""
        # Setup mock response
        mock_response_generator.generate_response.return_value = (
            "This is a successful test response."
        )

        controller = ConversationTurnController(mock_response_generator, mock_event_publisher)

        # Execute turn
        result_turn = await controller.execute_turn(test_turn, test_agent, test_context)

        # Verify turn completion
        assert result_turn.status == TurnStatus.COMPLETED
        assert result_turn.response == "This is a successful test response."
        assert result_turn.started_at is not None
        assert result_turn.completed_at is not None

        # Verify response generator called
        mock_response_generator.generate_response.assert_called_once_with(
            agent=test_agent, context=test_context, timeout_seconds=30
        )

        # Verify events published
        assert mock_event_publisher.publish_event.call_count == 2  # start and completion events

        # Verify metrics updated
        metrics = controller.get_metrics()
        assert metrics["turns_executed"] == 1
        assert metrics["turns_successful"] == 1
        assert metrics["turns_failed"] == 0

    @pytest.mark.asyncio
    async def test_turn_execution_timeout(
        self, mock_response_generator, mock_event_publisher, test_agent, test_context, test_turn
    ):
        """Test turn execution timeout handling."""
        # Make response generator timeout
        mock_response_generator.generate_response.side_effect = asyncio.TimeoutError()

        controller = ConversationTurnController(mock_response_generator, mock_event_publisher)

        # Execute turn
        result_turn = await controller.execute_turn(test_turn, test_agent, test_context)

        # Verify turn timeout
        assert result_turn.status == TurnStatus.TIMEOUT
        assert result_turn.error_message == "Turn timed out"
        assert result_turn.response is None

        # Verify metrics updated
        metrics = controller.get_metrics()
        assert metrics["turns_executed"] == 1
        assert metrics["turns_timeout"] == 1
        assert metrics["turns_successful"] == 0

    @pytest.mark.asyncio
    async def test_turn_execution_error(
        self, mock_response_generator, mock_event_publisher, test_agent, test_context, test_turn
    ):
        """Test turn execution error handling."""
        # Make response generator raise error
        mock_response_generator.generate_response.side_effect = Exception(
            "Response generation failed"
        )

        controller = ConversationTurnController(mock_response_generator, mock_event_publisher)

        # Execute turn
        result_turn = await controller.execute_turn(test_turn, test_agent, test_context)

        # Verify turn failure
        assert result_turn.status == TurnStatus.FAILED
        assert "Turn execution failed: Response generation failed" in result_turn.error_message

        # Verify metrics updated
        metrics = controller.get_metrics()
        assert metrics["turns_executed"] == 1
        assert metrics["turns_failed"] == 1
        assert metrics["turns_successful"] == 0

    @pytest.mark.asyncio
    async def test_turn_execution_empty_response(
        self, mock_response_generator, test_agent, test_context, test_turn
    ):
        """Test handling of empty response from generator."""
        # Return empty response
        mock_response_generator.generate_response.return_value = ""

        controller = ConversationTurnController(mock_response_generator)

        # Execute turn
        result_turn = await controller.execute_turn(test_turn, test_agent, test_context)

        # Verify turn failure due to empty response
        assert result_turn.status == TurnStatus.FAILED
        assert "Agent generated empty response" in result_turn.error_message

    def test_turn_input_validation(self, mock_response_generator, test_agent, test_context):
        """Test turn input validation."""
        controller = ConversationTurnController(mock_response_generator)

        # Test invalid turn status
        invalid_turn = ConversationTurnDomain(
            turn_number=1,
            agent_id="test-agent",
            agent_name="TestAgent",
            prompt="Test prompt",
        )
        invalid_turn.status = TurnStatus.COMPLETED  # Not PENDING

        with pytest.raises(ValueError, match="Turn must be PENDING status"):
            asyncio.run(controller.execute_turn(invalid_turn, test_agent, test_context))

        # Test empty agent name
        test_agent.name = ""
        valid_turn = ConversationTurnDomain(
            turn_number=1,
            agent_id="test-agent",
            agent_name="TestAgent",
            prompt="Test prompt",
        )

        with pytest.raises(ValueError, match="Agent name cannot be empty"):
            asyncio.run(controller.execute_turn(valid_turn, test_agent, test_context))

    def test_metrics_calculation(self, mock_response_generator):
        """Test metrics calculation and aggregation."""
        controller = ConversationTurnController(mock_response_generator)

        # Set up test metrics
        controller.metrics.update(
            {
                "turns_executed": 10,
                "turns_successful": 8,
                "turns_failed": 1,
                "turns_timeout": 1,
                "total_response_length": 800,
            }
        )

        metrics = controller.get_metrics()

        assert metrics["success_rate"] == 0.8
        assert metrics["failure_rate"] == 0.1
        assert metrics["timeout_rate"] == 0.1
        assert metrics["avg_response_length"] == 100.0  # 800 / 8 successful


class TestConversationEventPublisher:
    """Test suite for ConversationEventPublisher."""

    def test_event_publisher_initialization(self):
        """Test event publisher initializes correctly."""
        publisher = ConversationEventPublisher()

        assert publisher.enable_detailed_logging is True
        assert publisher.metrics["events_published"] == 0
        assert publisher.metrics["events_successful"] == 0

    def test_event_publisher_with_minimal_logging(self):
        """Test event publisher with minimal logging."""
        publisher = ConversationEventPublisher(enable_detailed_logging=False)

        assert publisher.enable_detailed_logging is False

    @pytest.mark.asyncio
    async def test_successful_event_publishing(self):
        """Test successful event publishing."""
        publisher = ConversationEventPublisher()

        test_event = ConversationEvent(
            conversation_id="test-conv",
            event_type="test_event",
            data={"key": "value"},
        )

        # Publish event
        await publisher.publish_event(test_event)

        # Verify metrics updated
        metrics = publisher.get_metrics()
        assert metrics["events_published"] == 1
        assert metrics["events_successful"] == 1
        assert metrics["events_failed"] == 0
        assert metrics["events_by_type"]["test_event"] == 1

    @pytest.mark.asyncio
    async def test_event_publishing_with_error(self):
        """Test event publishing with simulated error."""
        publisher = ConversationEventPublisher()

        # Patch the _publish_to_logs method to raise an error
        original_method = publisher._publish_to_logs
        publisher._publish_to_logs = Mock(side_effect=Exception("Log publishing failed"))

        test_event = ConversationEvent(
            conversation_id="test-conv",
            event_type="test_event",
            data={"key": "value"},
        )

        # Publish event (should not raise)
        await publisher.publish_event(test_event)

        # Verify metrics show failure
        metrics = publisher.get_metrics()
        assert metrics["events_published"] == 1
        assert metrics["events_successful"] == 0
        assert metrics["events_failed"] == 1

        # Restore original method
        publisher._publish_to_logs = original_method

    def test_event_publisher_metrics(self):
        """Test event publisher metrics calculation."""
        publisher = ConversationEventPublisher()

        # Set up test metrics
        publisher.metrics.update(
            {
                "events_published": 10,
                "events_successful": 9,
                "events_failed": 1,
            }
        )

        metrics = publisher.get_metrics()

        assert metrics["success_rate"] == 0.9
        assert metrics["failure_rate"] == 0.1


class TestTurnLimitPolicy:
    """Test suite for TurnLimitPolicy."""

    @pytest.fixture
    def test_conversation(self):
        """Create test conversation aggregate."""
        agent = AgentRole(
            agent_id="test-agent",
            name="TestAgent",
            role="test participant",
            system_prompt="Test agent",
        )

        conversation = ConversationAggregate(
            user_id="test-user",
            topic="Test conversation",
            participants=[agent],
        )
        conversation.started_at = datetime.now() - timedelta(minutes=5)
        return conversation

    def test_policy_initialization(self):
        """Test turn limit policy initialization."""
        policy = TurnLimitPolicy(
            max_turns=5,
            max_duration_minutes=15,
            enable_quality_limits=True,
        )

        assert policy.max_turns == 5
        assert policy.max_duration_minutes == 15
        assert policy.enable_quality_limits is True
        assert policy.decisions["continue_decisions"] == 0

    def test_should_continue_within_limits(self, test_conversation):
        """Test should_continue when within all limits."""
        policy = TurnLimitPolicy(max_turns=10, max_duration_minutes=60)

        # Add a few turns to the conversation
        for i in range(3):
            turn = ConversationTurnDomain(
                turn_number=i + 1,
                agent_id="test-agent",
                agent_name="TestAgent",
                prompt=f"Turn {i + 1} prompt",
            )
            turn.mark_completed(f"Response {i + 1}")
            test_conversation.turns.append(turn)

        result = policy.should_continue(test_conversation)

        assert result is True
        assert policy.decisions["continue_decisions"] == 1

    def test_should_continue_turn_limit_reached(self, test_conversation):
        """Test should_continue when turn limit is reached."""
        policy = TurnLimitPolicy(max_turns=3, max_duration_minutes=60)

        # Add turns to reach limit
        for i in range(3):
            turn = ConversationTurnDomain(
                turn_number=i + 1,
                agent_id="test-agent",
                agent_name="TestAgent",
                prompt=f"Turn {i + 1} prompt",
            )
            turn.mark_completed(f"Response {i + 1}")
            test_conversation.turns.append(turn)

        result = policy.should_continue(test_conversation)

        assert result is False
        assert policy.decisions["turn_limit_stops"] == 1

    def test_should_continue_time_limit_reached(self, test_conversation):
        """Test should_continue when time limit is reached."""
        policy = TurnLimitPolicy(max_turns=10, max_duration_minutes=2)  # 2 minute limit

        # Set conversation start time to 5 minutes ago (exceeds 2 minute limit)
        test_conversation.started_at = datetime.now() - timedelta(minutes=5)

        result = policy.should_continue(test_conversation)

        assert result is False
        assert policy.decisions["time_limit_stops"] == 1

    def test_should_continue_quality_limit_reached(self, test_conversation):
        """Test should_continue when quality limit is reached."""
        policy = TurnLimitPolicy(
            max_turns=10,
            max_duration_minutes=60,
            enable_quality_limits=True,
            min_quality_threshold=0.5,
        )

        # Add low quality turns (very short responses indicate poor quality)
        for i in range(3):
            turn = ConversationTurnDomain(
                turn_number=i + 1,
                agent_id="test-agent",
                agent_name="TestAgent",
                prompt=f"Turn {i + 1} prompt",
            )
            turn.mark_completed("Bad.")  # Very short, low quality response
            test_conversation.turns.append(turn)

        result = policy.should_continue(test_conversation)

        assert result is False
        assert policy.decisions["quality_limit_stops"] == 1

    def test_get_completion_reason(self, test_conversation):
        """Test get_completion_reason returns correct reasons."""
        policy = TurnLimitPolicy(max_turns=2, max_duration_minutes=60)

        # Test no completion reason initially
        reason = policy.get_completion_reason(test_conversation)
        assert reason is None

        # Add turns to reach limit
        for i in range(2):
            turn = ConversationTurnDomain(
                turn_number=i + 1,
                agent_id="test-agent",
                agent_name="TestAgent",
                prompt=f"Turn {i + 1} prompt",
            )
            turn.mark_completed(f"Response {i + 1}")
            test_conversation.turns.append(turn)

        # Should return turn limit reached
        reason = policy.get_completion_reason(test_conversation)
        assert reason == CompletionReason.TURN_LIMIT_REACHED

    def test_quality_estimation(self):
        """Test turn quality estimation logic."""
        policy = TurnLimitPolicy()

        # Test with good quality turns
        good_turns = []
        for i in range(3):
            turn = ConversationTurnDomain(
                turn_number=i + 1,
                agent_id="test-agent",
                agent_name="TestAgent",
                prompt=f"Turn {i + 1} prompt",
            )
            turn.mark_completed("This is a good quality response with adequate length and content.")
            good_turns.append(turn)

        good_quality = policy._estimate_turn_quality(good_turns)
        assert good_quality > 0.3  # Adjust expectation based on actual implementation

        # Test with poor quality turns
        poor_turns = []
        for i in range(3):
            turn = ConversationTurnDomain(
                turn_number=i + 1,
                agent_id="test-agent",
                agent_name="TestAgent",
                prompt=f"Turn {i + 1} prompt",
            )
            turn.mark_completed("Bad.")  # Very short response
            poor_turns.append(turn)

        poor_quality = policy._estimate_turn_quality(poor_turns)
        assert poor_quality < 0.3


class TestConsensusCompletionDetector:
    """Test suite for ConsensusCompletionDetector."""

    @pytest.fixture
    def test_context(self):
        """Create test conversation context."""
        agent = AgentRole(
            agent_id="test-agent",
            name="TestAgent",
            role="test participant",
            system_prompt="Test agent",
        )

        return ConversationContext(
            conversation_id="test-conversation",
            topic="Test conversation topic",
            participants=[agent],
        )

    def test_detector_initialization(self):
        """Test completion detector initialization."""
        detector = ConsensusCompletionDetector(
            enable_consensus_detection=True,
            consensus_threshold=0.8,
            min_turns_for_consensus=5,
        )

        assert detector.enable_consensus_detection is True
        assert detector.consensus_threshold == 0.8
        assert detector.min_turns_for_consensus == 5
        assert detector.detections["consensus_detected"] == 0

    @pytest.mark.asyncio
    async def test_detect_completion_disabled(self, test_context):
        """Test completion detection when disabled."""
        detector = ConsensusCompletionDetector(enable_consensus_detection=False)

        result = await detector.detect_completion(test_context)

        assert result is None
        assert detector.detections["no_completion_detected"] == 1

    @pytest.mark.asyncio
    async def test_detect_completion_insufficient_turns(self, test_context):
        """Test completion detection with insufficient turns."""
        detector = ConsensusCompletionDetector(min_turns_for_consensus=5)

        # Add only 2 turns (less than minimum)
        for i in range(2):
            turn = ConversationTurnDomain(
                turn_number=i + 1,
                agent_id="test-agent",
                agent_name="TestAgent",
                prompt=f"Turn {i + 1} prompt",
            )
            turn.mark_completed(f"Response {i + 1}")
            test_context.turn_history.append(turn)

        result = await detector.detect_completion(test_context)

        assert result is None
        assert detector.detections["no_completion_detected"] == 1

    @pytest.mark.asyncio
    async def test_detect_consensus_completion(self, test_context):
        """Test consensus completion detection."""
        detector = ConsensusCompletionDetector(
            consensus_threshold=0.6,
            min_turns_for_consensus=3,
        )

        # Add turns with consensus indicators
        consensus_responses = [
            "I completely agree with that analysis.",
            "Yes, that's exactly right. We've reached consensus on this approach.",
            "Absolutely correct. I think we're all agreed on this solution.",
        ]

        for i, response in enumerate(consensus_responses):
            turn = ConversationTurnDomain(
                turn_number=i + 1,
                agent_id="test-agent",
                agent_name="TestAgent",
                prompt=f"Turn {i + 1} prompt",
            )
            turn.mark_completed(response)
            test_context.turn_history.append(turn)

        result = await detector.detect_completion(test_context)

        assert result == CompletionReason.CONSENSUS_REACHED
        assert detector.detections["consensus_detected"] == 1

    @pytest.mark.asyncio
    async def test_detect_task_completion(self, test_context):
        """Test task completion detection."""
        detector = ConsensusCompletionDetector(
            consensus_threshold=0.9,  # High threshold to avoid consensus detection
            min_turns_for_consensus=3,
        )

        # Add turns with task completion indicators
        task_completion_responses = [
            "We have successfully completed all the required tasks.",
            "The project is finished and all objectives have been accomplished.",
            "Everything is done according to the specifications.",
        ]

        for i, response in enumerate(task_completion_responses):
            turn = ConversationTurnDomain(
                turn_number=i + 1,
                agent_id="test-agent",
                agent_name="TestAgent",
                prompt=f"Turn {i + 1} prompt",
            )
            turn.mark_completed(response)
            test_context.turn_history.append(turn)

        result = await detector.detect_completion(test_context)

        assert result == CompletionReason.TASK_COMPLETED
        assert detector.detections["task_completion_detected"] == 1

    @pytest.mark.asyncio
    async def test_no_completion_detected(self, test_context):
        """Test when no completion is detected."""
        detector = ConsensusCompletionDetector(
            consensus_threshold=0.8,
            min_turns_for_consensus=3,
        )

        # Add turns with no completion indicators
        normal_responses = [
            "This is an interesting point to consider.",
            "I have a different perspective on this matter.",
            "Let me explore this idea further.",
        ]

        for i, response in enumerate(normal_responses):
            turn = ConversationTurnDomain(
                turn_number=i + 1,
                agent_id="test-agent",
                agent_name="TestAgent",
                prompt=f"Turn {i + 1} prompt",
            )
            turn.mark_completed(response)
            test_context.turn_history.append(turn)

        result = await detector.detect_completion(test_context)

        assert result is None
        assert detector.detections["no_completion_detected"] == 1

    def test_consensus_analysis(self):
        """Test consensus analysis logic."""
        detector = ConsensusCompletionDetector()

        # Create turns with consensus indicators
        consensus_turns = []
        for i, response in enumerate(
            [
                "I agree with this approach completely.",
                "Yes, that's exactly right.",
                "This analysis is correct and I support it.",
            ]
        ):
            turn = ConversationTurnDomain(
                turn_number=i + 1,
                agent_id="test-agent",
                agent_name="TestAgent",
                prompt=f"Turn {i + 1} prompt",
            )
            turn.mark_completed(response)
            consensus_turns.append(turn)

        consensus_score = detector._analyze_consensus(consensus_turns)
        assert consensus_score == 1.0  # All turns have consensus indicators

        # Create turns without consensus indicators
        no_consensus_turns = []
        for i, response in enumerate(
            [
                "This is a complex issue.",
                "I have doubts about this approach.",
                "We need more information.",
            ]
        ):
            turn = ConversationTurnDomain(
                turn_number=i + 1,
                agent_id="test-agent",
                agent_name="TestAgent",
                prompt=f"Turn {i + 1} prompt",
            )
            turn.mark_completed(response)
            no_consensus_turns.append(turn)

        no_consensus_score = detector._analyze_consensus(no_consensus_turns)
        assert no_consensus_score == 0.0  # No consensus indicators

    def test_task_completion_analysis(self):
        """Test task completion analysis logic."""
        detector = ConsensusCompletionDetector()

        # Create turns with task completion indicators
        completion_turns = []
        for i, response in enumerate(
            [
                "The task has been completed successfully finished.",
                "We have accomplished everything that was required.",
            ]
        ):
            turn = ConversationTurnDomain(
                turn_number=i + 1,
                agent_id="test-agent",
                agent_name="TestAgent",
                prompt=f"Turn {i + 1} prompt",
            )
            turn.mark_completed(response)
            completion_turns.append(turn)

        completion_detected = detector._analyze_task_completion(completion_turns)
        assert completion_detected is True

        # Create turns without completion indicators
        no_completion_turns = []
        for i, response in enumerate(
            [
                "This is still in progress.",
                "We're making good headway on this.",
            ]
        ):
            turn = ConversationTurnDomain(
                turn_number=i + 1,
                agent_id="test-agent",
                agent_name="TestAgent",
                prompt=f"Turn {i + 1} prompt",
            )
            turn.mark_completed(response)
            no_completion_turns.append(turn)

        no_completion_detected = detector._analyze_task_completion(no_completion_turns)
        assert no_completion_detected is False


class TestFactoryFunctions:
    """Test suite for factory functions."""

    def test_create_turn_controller(self):
        """Test turn controller factory function."""
        mock_generator = Mock()
        controller = create_turn_controller(mock_generator)

        assert isinstance(controller, ConversationTurnController)
        assert controller.response_generator == mock_generator
        assert controller.event_publisher is not None

    def test_create_event_publisher(self):
        """Test event publisher factory function."""
        publisher = create_event_publisher()

        assert isinstance(publisher, ConversationEventPublisher)
        assert publisher.enable_detailed_logging is True

    def test_create_turn_limit_policy(self):
        """Test turn limit policy factory function."""
        policy = create_turn_limit_policy(
            max_turns=15,
            max_duration_minutes=45,
            enable_quality_limits=True,
        )

        assert isinstance(policy, TurnLimitPolicy)
        assert policy.max_turns == 15
        assert policy.max_duration_minutes == 45
        assert policy.enable_quality_limits is True

    def test_create_completion_detector(self):
        """Test completion detector factory function."""
        detector = create_completion_detector(enable_consensus=False)

        assert isinstance(detector, ConsensusCompletionDetector)
        assert detector.enable_consensus_detection is False


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
