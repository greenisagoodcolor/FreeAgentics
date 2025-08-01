"""
Tests for ConversationContextManager

This test suite validates the conversation context management system
following TDD principles as advocated by the Nemesis Committee.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from api.v1.models.agent_conversation import (
    AgentRole,
    ConversationAggregate,
    ConversationStatus,
    ConversationTurnDomain,
)
from api.v1.services.conversation_context_manager import (
    ContextWindowConfig,
    ConversationContextManager,
    create_context_manager,
)


class TestConversationContextManager:
    """Test suite for ConversationContextManager."""

    @pytest.fixture
    def sample_agent_roles(self):
        """Create sample agent roles for testing."""
        return [
            AgentRole(
                agent_id="agent-1",
                name="Advocate",
                role="advocate",
                system_prompt="You are an advocate who supports ideas.",
                personality_traits=["supportive", "optimistic"],
                turn_order=1,
            ),
            AgentRole(
                agent_id="agent-2",
                name="Critic",
                role="critic",
                system_prompt="You are a critic who questions ideas.",
                personality_traits=["analytical", "cautious"],
                turn_order=2,
            ),
        ]

    @pytest.fixture
    def sample_turns(self):
        """Create sample conversation turns for testing."""
        turns = []
        for i in range(5):
            turn = ConversationTurnDomain(
                turn_number=i + 1,
                agent_id=f"agent-{(i % 2) + 1}",
                agent_name="Advocate" if i % 2 == 0 else "Critic",
                prompt=f"Turn {i + 1} prompt",
                response=f"Turn {i + 1} response from {'Advocate' if i % 2 == 0 else 'Critic'}",
                created_at=datetime.now() - timedelta(minutes=5 - i),
                completed_at=datetime.now() - timedelta(minutes=4 - i),
            )
            turn.mark_completed(turn.response)
            turns.append(turn)
        return turns

    @pytest.fixture
    def sample_conversation(self, sample_agent_roles, sample_turns):
        """Create sample conversation aggregate for testing."""
        conversation = ConversationAggregate(
            user_id="user-123",
            topic="Test conversation about AI ethics",
            participants=sample_agent_roles,
            turns=sample_turns,
            status=ConversationStatus.ACTIVE,
        )
        return conversation

    @pytest.fixture
    def context_manager(self):
        """Create context manager instance for testing."""
        config = ContextWindowConfig(
            max_turns_in_context=3,
            max_characters_in_context=1000,
            preserve_conversation_start=True,
        )
        return ConversationContextManager(config)

    def test_create_context_with_basic_conversation(self, context_manager, sample_conversation):
        """Test creating context for a basic conversation."""
        # When: Creating context for conversation
        context = context_manager.create_context(sample_conversation)

        # Then: Context should be properly created
        assert context.conversation_id == sample_conversation.conversation_id
        assert context.topic == sample_conversation.topic
        assert len(context.participants) == 2
        assert context.current_turn_number == 6  # len(turns) + 1

        # Should apply windowing (max 3 turns with preserve start = True)
        # With 5 turns and max 3, should have first turn + last 2 turns = 3 total
        assert len(context.turn_history) == 3

        # Should include conversation start
        assert context.turn_history[0].turn_number == 1
        # Should include most recent turns
        assert context.turn_history[-1].turn_number == 5

    def test_context_caching_behavior(self, context_manager, sample_conversation):
        """Test that contexts are properly cached."""
        # When: Creating context twice with same conversation state
        context1 = context_manager.create_context(sample_conversation)
        context2 = context_manager.create_context(sample_conversation)

        # Then: Should return same cached context
        assert context1 is context2

        # When: Adding a new turn and creating context
        new_turn = ConversationTurnDomain(
            turn_number=6,
            agent_id="agent-1",
            agent_name="Advocate",
            prompt="New turn prompt",
            response="New turn response",
        )
        new_turn.mark_completed("New turn response")
        sample_conversation.turns.append(new_turn)

        context3 = context_manager.create_context(sample_conversation)

        # Then: Should create new context due to changed turn count
        assert context3 is not context1
        assert context3.current_turn_number == 7

    def test_context_windowing_with_character_limits(self, sample_conversation):
        """Test context windowing based on character limits."""
        # Given: Context manager with small character limit
        config = ContextWindowConfig(
            max_turns_in_context=10,  # High turn limit
            max_characters_in_context=60,  # Very small character limit to force aggressive windowing
            preserve_conversation_start=True,
        )
        context_manager = ConversationContextManager(config)

        # When: Creating context
        context = context_manager.create_context(sample_conversation)

        # Then: Should limit based on characters, not turn count
        total_chars = sum(len(turn.response or "") for turn in context.turn_history)
        assert total_chars <= 60
        assert len(context.turn_history) >= 1  # Should preserve at least start
        # Should have fewer turns due to character limit
        assert len(context.turn_history) < 5

    def test_context_windowing_without_preserving_start(self, sample_conversation):
        """Test context windowing without preserving conversation start."""
        # Given: Context manager that doesn't preserve start
        config = ContextWindowConfig(
            max_turns_in_context=2,
            preserve_conversation_start=False,
        )
        context_manager = ConversationContextManager(config)

        # When: Creating context
        context = context_manager.create_context(sample_conversation)

        # Then: Should only include most recent turns
        assert len(context.turn_history) == 2
        # Should be turns 4 and 5 (most recent)
        turn_numbers = [turn.turn_number for turn in context.turn_history]
        assert turn_numbers == [4, 5]

    def test_fallback_context_creation_on_error(self, sample_conversation):
        """Test fallback context creation when main creation fails."""
        # Given: Context manager
        context_manager = ConversationContextManager()

        # When: Creating context with invalid conversation (simulate error)
        with patch.object(
            context_manager, "_apply_context_windowing", side_effect=Exception("Test error")
        ):
            context = context_manager.create_context(sample_conversation)

        # Then: Should create fallback context
        assert context.conversation_id == sample_conversation.conversation_id
        assert context.topic == sample_conversation.topic
        assert len(context.turn_history) == 0  # Empty history as fallback
        assert context.current_turn_number == 6

    def test_enrich_context_for_agent(
        self, context_manager, sample_conversation, sample_agent_roles
    ):
        """Test context enrichment for specific agent."""
        # Given: Basic context
        context = context_manager.create_context(sample_conversation)
        agent = sample_agent_roles[0]  # Advocate

        # When: Enriching context for agent
        enriched = context_manager.enrich_context_for_agent(context, agent)

        # Then: Should include enriched information
        assert "agent_prompt" in enriched
        assert "conversation_summary" in enriched
        assert "agent_metadata" in enriched
        assert "context_metadata" in enriched

        # Agent metadata should be correct
        agent_meta = enriched["agent_metadata"]
        assert agent_meta["name"] == "Advocate"
        assert agent_meta["role"] == "advocate"
        assert agent_meta["agent_id"] == "agent-1"

        # Context metadata should be present
        context_meta = enriched["context_metadata"]
        assert "timestamp" in context_meta
        assert context_meta["total_participants"] == 2

    def test_enrich_context_fallback_on_error(
        self, context_manager, sample_conversation, sample_agent_roles
    ):
        """Test context enrichment fallback when enrichment fails."""
        # Given: Context and agent
        context = context_manager.create_context(sample_conversation)
        agent = sample_agent_roles[0]

        # When: Enriching context with mocked context that raises error
        with patch(
            "api.v1.services.conversation_context_manager.ConversationContext.create_agent_prompt",
            side_effect=Exception("Test error"),
        ):
            enriched = context_manager.enrich_context_for_agent(context, agent)

        # Then: Should provide fallback enrichment
        assert "error" in enriched
        assert enriched["error"] == "Context enrichment failed - using fallback"
        assert "agent_prompt" in enriched
        assert enriched["agent_metadata"]["name"] == "Advocate"

    def test_context_serialization_and_deserialization(self, context_manager, sample_conversation):
        """Test context serialization and deserialization."""
        # Given: Context
        original_context = context_manager.create_context(sample_conversation)

        # When: Serializing context
        serialized = context_manager.serialize_context(original_context)

        # Then: Should be valid JSON
        context_dict = json.loads(serialized)
        assert context_dict["conversation_id"] == original_context.conversation_id
        assert context_dict["topic"] == original_context.topic
        assert "serialization_metadata" in context_dict

        # When: Deserializing context
        deserialized_context = context_manager.deserialize_context(serialized)

        # Then: Should match original
        assert deserialized_context.conversation_id == original_context.conversation_id
        assert deserialized_context.topic == original_context.topic
        assert len(deserialized_context.participants) == len(original_context.participants)
        assert len(deserialized_context.turn_history) == len(original_context.turn_history)

    def test_serialization_failure_handling(self, context_manager, sample_conversation):
        """Test handling of serialization failures."""
        # Given: Context with unserializable data
        context = context_manager.create_context(sample_conversation)

        # When: Attempting to serialize with mocked JSON failure
        with patch("json.dumps", side_effect=Exception("Serialization error")):
            with pytest.raises(ValueError, match="Context serialization failed"):
                context_manager.serialize_context(context)

        # Then: Metrics should reflect failure
        metrics = context_manager.get_context_metrics()
        assert metrics["serialization_failures"] > 0

    def test_cache_invalidation(self, context_manager, sample_conversation):
        """Test context cache invalidation."""
        # Given: Cached context
        context1 = context_manager.create_context(sample_conversation)
        conversation_id = sample_conversation.conversation_id

        # When: Invalidating cache
        context_manager.invalidate_context_cache(conversation_id)

        # Then: Next context creation should create new context
        context2 = context_manager.create_context(sample_conversation)
        assert context2 is not context1  # New instance created

    def test_conversation_data_cleanup(self, context_manager, sample_conversation):
        """Test cleanup of conversation data."""
        # Given: Context manager with cached data
        context_manager.create_context(sample_conversation)
        conversation_id = sample_conversation.conversation_id

        # Verify data exists
        assert conversation_id in context_manager._context_cache

        # When: Clearing conversation data
        context_manager.clear_conversation_data(conversation_id)

        # Then: All data should be removed
        assert conversation_id not in context_manager._context_cache
        assert conversation_id not in context_manager._turn_buffers

    def test_context_metrics_collection(self, context_manager, sample_conversation):
        """Test context operation metrics collection."""
        # Given: Initial metrics
        initial_metrics = context_manager.get_context_metrics()

        # When: Performing various context operations
        context_manager.create_context(sample_conversation)  # Cache miss
        context_manager.create_context(sample_conversation)  # Cache hit
        context_manager.serialize_context(context_manager.create_context(sample_conversation))

        # Then: Metrics should be updated
        final_metrics = context_manager.get_context_metrics()

        assert final_metrics["contexts_created"] > initial_metrics["contexts_created"]
        assert final_metrics["cache_hits"] > initial_metrics["cache_hits"]
        assert final_metrics["cache_misses"] > initial_metrics["cache_misses"]
        assert (
            final_metrics["serialization_operations"] > initial_metrics["serialization_operations"]
        )
        assert "cache_hit_rate" in final_metrics
        assert "serialization_success_rate" in final_metrics

    def test_stale_context_cleanup(self, context_manager):
        """Test cleanup of stale contexts."""
        # Given: Context manager with multiple cached contexts
        for i in range(150):  # Exceed max cache size
            mock_conversation = Mock()
            mock_conversation.conversation_id = f"conv-{i}"
            mock_conversation.topic = f"Topic {i}"
            mock_conversation.participants = []
            mock_conversation.turns = []

            # Create minimal context to populate cache
            context_manager._context_cache[f"conv-{i}"] = (Mock(), 0)

        initial_cache_size = len(context_manager._context_cache)

        # When: Cleaning up stale contexts
        cleaned_count = context_manager.cleanup_stale_contexts()

        # Then: Should clean up excess contexts
        assert cleaned_count > 0
        final_cache_size = len(context_manager._context_cache)
        assert final_cache_size < initial_cache_size

    def test_factory_function(self):
        """Test context manager factory function."""
        # When: Creating context manager via factory
        manager = create_context_manager(
            max_turns_in_context=5,
            max_characters_in_context=2000,
            preserve_conversation_start=False,
        )

        # Then: Should create properly configured manager
        assert isinstance(manager, ConversationContextManager)
        assert manager.window_config.max_turns_in_context == 5
        assert manager.window_config.max_characters_in_context == 2000
        assert manager.window_config.preserve_conversation_start is False


class TestContextWindowConfig:
    """Test suite for ContextWindowConfig."""

    def test_default_configuration(self):
        """Test default context window configuration."""
        # When: Creating default config
        config = ContextWindowConfig()

        # Then: Should have expected defaults
        assert config.max_turns_in_context == 10
        assert config.max_characters_in_context == 4000
        assert config.preserve_conversation_start is True
        assert config.importance_scoring_enabled is False

    def test_custom_configuration(self):
        """Test custom context window configuration."""
        # When: Creating custom config
        config = ContextWindowConfig(
            max_turns_in_context=5,
            max_characters_in_context=1500,
            preserve_conversation_start=False,
            importance_scoring_enabled=True,
        )

        # Then: Should have custom values
        assert config.max_turns_in_context == 5
        assert config.max_characters_in_context == 1500
        assert config.preserve_conversation_start is False
        assert config.importance_scoring_enabled is True


class TestContextManagerIntegration:
    """Integration tests for context manager with domain models."""

    def test_integration_with_conversation_aggregate(self):
        """Test integration with ConversationAggregate domain model."""
        # Given: Real conversation aggregate
        participants = [
            AgentRole(
                agent_id="agent-1",
                name="Facilitator",
                role="facilitator",
                system_prompt="You facilitate discussions.",
                turn_order=1,
            )
        ]

        conversation = ConversationAggregate(
            user_id="user-123",
            topic="Integration test conversation",
            participants=participants,
        )

        # Start the conversation first (required by domain rules)
        conversation.start_conversation()

        # Add some turns
        for i in range(3):
            turn = ConversationTurnDomain(
                turn_number=i + 1,
                agent_id="agent-1",
                agent_name="Facilitator",
                prompt=f"Turn {i + 1} prompt",
            )
            turn.mark_completed(f"Response {i + 1}")
            conversation.add_turn(turn)

        # When: Using context manager with real aggregate
        context_manager = ConversationContextManager()
        context = context_manager.create_context(conversation)

        # Then: Should work seamlessly with domain models
        assert context.conversation_id == conversation.conversation_id
        assert len(context.turn_history) == 3
        assert context.current_turn_number == 4

        # Context should work with domain model methods
        agent_prompt = context.create_agent_prompt(participants[0])
        assert "Integration test conversation" in agent_prompt
        assert "Facilitator" in agent_prompt

    def test_context_persistence_workflow(self):
        """Test complete context persistence workflow."""
        # Given: Context manager and conversation
        context_manager = ConversationContextManager()

        participants = [
            AgentRole(
                agent_id="agent-1",
                name="TestAgent",
                role="test",
                system_prompt="Test prompt",
            )
        ]

        conversation = ConversationAggregate(
            user_id="user-123",
            topic="Persistence test",
            participants=participants,
        )

        # When: Creating, serializing, and deserializing context
        original_context = context_manager.create_context(conversation)
        serialized = context_manager.serialize_context(original_context)
        restored_context = context_manager.deserialize_context(serialized)

        # Then: Restored context should be functionally equivalent
        assert restored_context.conversation_id == original_context.conversation_id
        assert restored_context.topic == original_context.topic
        assert len(restored_context.participants) == len(original_context.participants)

        # Should be able to use restored context for operations
        enriched = context_manager.enrich_context_for_agent(restored_context, participants[0])
        assert "agent_prompt" in enriched
        assert "TestAgent" in enriched["agent_prompt"]
