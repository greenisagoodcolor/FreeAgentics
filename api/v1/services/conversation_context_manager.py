"""
Conversation Context Management Service

This module provides comprehensive context management for multi-turn agent conversations,
implementing efficient windowing, serialization, and error handling as recommended by
the Nemesis Committee.

The service consolidates context operations that were previously scattered across multiple
files and properly utilizes the existing ConversationContext domain model.
"""

import json
import logging
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from api.v1.models.agent_conversation import (
    AgentRole,
    ConversationAggregate,
    ConversationContext,
    ConversationTurnDomain,
)

logger = logging.getLogger(__name__)


class ContextWindowConfig:
    """Configuration for context windowing behavior."""

    def __init__(
        self,
        max_turns_in_context: int = 10,
        max_characters_in_context: int = 4000,
        preserve_conversation_start: bool = True,
        importance_scoring_enabled: bool = False,
    ):
        """Initialize context window configuration.

        Args:
            max_turns_in_context: Maximum number of turns to include in context
            max_characters_in_context: Maximum total characters in context
            preserve_conversation_start: Always include first few turns
            importance_scoring_enabled: Use importance scoring for turn selection
        """
        self.max_turns_in_context = max_turns_in_context
        self.max_characters_in_context = max_characters_in_context
        self.preserve_conversation_start = preserve_conversation_start
        self.importance_scoring_enabled = importance_scoring_enabled


class ConversationContextManager:
    """
    Comprehensive conversation context management service.

    This service encapsulates all context-related operations including windowing,
    serialization, enrichment, and prompt generation. It properly utilizes the
    existing ConversationContext domain model while providing efficient caching
    and error handling.
    """

    def __init__(self, window_config: Optional[ContextWindowConfig] = None):
        """Initialize context manager with configuration."""
        self.window_config = window_config or ContextWindowConfig()

        # Efficient context caching to avoid rebuilding unchanged contexts
        self._context_cache: Dict[str, Tuple[ConversationContext, int]] = {}

        # Circular buffer for efficient turn storage per conversation
        self._turn_buffers: Dict[str, deque] = {}

        # Context operation metrics for observability
        self._metrics = {
            "contexts_created": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "serialization_operations": 0,
            "serialization_failures": 0,
            "windowing_operations": 0,
        }

        logger.info(
            f"ConversationContextManager initialized with config: {window_config.__dict__ if window_config else 'default'}"
        )

    def create_context(
        self,
        conversation: ConversationAggregate,
        force_refresh: bool = False,
    ) -> ConversationContext:
        """
        Create or retrieve cached conversation context.

        Args:
            conversation: Current conversation aggregate
            force_refresh: Force context recreation ignoring cache

        Returns:
            ConversationContext with properly windowed turn history

        Raises:
            ValueError: If conversation data is invalid
        """
        conversation_id = conversation.conversation_id
        current_turn_count = len(conversation.turns)

        try:
            # Check cache first (unless forced refresh)
            if not force_refresh and conversation_id in self._context_cache:
                cached_context, cached_turn_count = self._context_cache[conversation_id]

                # Use cached context if turn count hasn't changed
                if cached_turn_count == current_turn_count:
                    self._metrics["cache_hits"] += 1
                    logger.debug(f"Using cached context for conversation {conversation_id}")
                    return cached_context

            self._metrics["cache_misses"] += 1
            self._metrics["contexts_created"] += 1

            # Create new context with windowed turn history
            windowed_turns = self._apply_context_windowing(conversation.turns)

            # Create immutable context using existing domain model
            context = ConversationContext(
                conversation_id=conversation_id,
                topic=conversation.topic,
                participants=conversation.participants,
                turn_history=windowed_turns,
                current_turn_number=current_turn_count + 1,
                context_window_size=self.window_config.max_turns_in_context,
            )

            # Cache the context
            self._context_cache[conversation_id] = (context, current_turn_count)

            # Update turn buffer for efficient future operations
            if conversation_id not in self._turn_buffers:
                self._turn_buffers[conversation_id] = deque(
                    maxlen=self.window_config.max_turns_in_context * 2  # Buffer 2x window size
                )

            # Add new turns to buffer
            for turn in windowed_turns:
                if turn not in self._turn_buffers[conversation_id]:
                    self._turn_buffers[conversation_id].append(turn)

            logger.info(
                f"Created context for conversation {conversation_id} with {len(windowed_turns)} turns"
            )
            return context

        except Exception as e:
            logger.error(f"Failed to create context for conversation {conversation_id}: {e}")
            # Return minimal fallback context
            return self._create_fallback_context(conversation)

    def _apply_context_windowing(
        self, all_turns: List[ConversationTurnDomain]
    ) -> List[ConversationTurnDomain]:
        """
        Apply intelligent context windowing to limit context size.

        Args:
            all_turns: All conversation turns

        Returns:
            Filtered list of turns within context window limits
        """
        self._metrics["windowing_operations"] += 1

        if not all_turns:
            return []

        completed_turns = [turn for turn in all_turns if turn.is_completed]

        # Apply character limit windowing first if configured
        if self.window_config.max_characters_in_context > 0:
            char_limited = self._apply_character_limit_windowing(completed_turns)
            # If character limiting already reduced to acceptable turn count, return it
            if len(char_limited) <= self.window_config.max_turns_in_context:
                return char_limited
            # Otherwise apply turn-based windowing to the character-limited result
            return self._apply_recency_windowing(char_limited)

        # If no character limits, only apply turn-based windowing if needed
        if len(completed_turns) <= self.window_config.max_turns_in_context:
            return completed_turns

        # Apply simple recency-based windowing
        return self._apply_recency_windowing(completed_turns)

    def _apply_character_limit_windowing(
        self, turns: List[ConversationTurnDomain]
    ) -> List[ConversationTurnDomain]:
        """Apply windowing based on character limits."""
        if not turns:
            return []

        result = []
        total_chars = 0
        max_chars = self.window_config.max_characters_in_context

        if self.window_config.preserve_conversation_start and turns:
            # Always include first turn if configured
            first_turn = turns[0]
            first_turn_chars = len(first_turn.response or "")

            # If first turn alone exceeds limit, truncate it but still include
            if first_turn_chars <= max_chars:
                result.append(first_turn)
                total_chars += first_turn_chars
            else:
                # First turn is too big, but still include it (policy decision)
                result.append(first_turn)
                total_chars = max_chars  # Mark as at limit
                logger.debug(
                    f"First turn exceeds character limit ({first_turn_chars} > {max_chars})"
                )
                return result  # Can't add any more turns

        # Add recent turns working backwards until character limit
        start_idx = 1 if (self.window_config.preserve_conversation_start and result) else 0
        remaining_chars = max_chars - total_chars
        recent_turns = []

        for turn in reversed(turns[start_idx:]):
            turn_chars = len(turn.response or "")
            if turn_chars <= remaining_chars:
                recent_turns.append(turn)
                remaining_chars -= turn_chars
            # If this turn would exceed limit, stop adding turns

        # Combine preserved start + recent turns in chronological order
        if not self.window_config.preserve_conversation_start or not result:
            # If not preserving start, use recent turns only
            result = list(reversed(recent_turns))
        else:
            # Add recent turns after the preserved first turn
            result.extend(reversed(recent_turns))

        final_chars = sum(len(turn.response or "") for turn in result)
        logger.debug(
            f"Character-limited windowing: {len(result)} turns, {final_chars} characters (limit: {max_chars})"
        )
        return result

    def _apply_recency_windowing(
        self, turns: List[ConversationTurnDomain]
    ) -> List[ConversationTurnDomain]:
        """Apply simple recency-based windowing."""
        max_turns = self.window_config.max_turns_in_context

        if len(turns) <= max_turns:
            return turns

        # Preserve conversation start if configured
        if self.window_config.preserve_conversation_start and max_turns >= 3:
            # Keep first turn + most recent (max_turns - 1) turns
            first_turn = turns[0]
            recent_turns = turns[-(max_turns - 1) :]
            # Avoid duplicating first turn if it's also in recent turns
            if len(turns) > max_turns - 1:
                return [first_turn] + recent_turns
            else:
                return turns

        # Just keep most recent turns
        return turns[-max_turns:]

    def _create_fallback_context(self, conversation: ConversationAggregate) -> ConversationContext:
        """Create minimal fallback context when main context creation fails."""
        logger.warning(f"Creating fallback context for conversation {conversation.conversation_id}")

        return ConversationContext(
            conversation_id=conversation.conversation_id,
            topic=conversation.topic,
            participants=conversation.participants,
            turn_history=[],  # Empty history as fallback
            current_turn_number=len(conversation.turns) + 1,
            context_window_size=self.window_config.max_turns_in_context,
        )

    def enrich_context_for_agent(
        self, context: ConversationContext, agent: AgentRole
    ) -> Dict[str, Any]:
        """
        Enrich context with agent-specific information.

        Args:
            context: Base conversation context
            agent: Agent for whom to enrich context

        Returns:
            Enriched context dictionary with agent-specific metadata
        """
        try:
            # Generate base agent prompt using domain model
            agent_prompt = context.create_agent_prompt(agent)

            # Add enrichment metadata
            enriched_context = {
                "agent_prompt": agent_prompt,
                "conversation_summary": context.conversation_summary,
                "turn_count": context.current_turn_number,
                "recent_turn_count": len(context.recent_turns),
                "participant_names": [p.name for p in context.participants],
                "agent_metadata": {
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "role": agent.role,
                    "turn_order": agent.turn_order,
                    "personality_traits": agent.personality_traits,
                },
                "context_metadata": {
                    "window_size": context.context_window_size,
                    "total_participants": len(context.participants),
                    "timestamp": datetime.now().isoformat(),
                },
            }

            logger.debug(
                f"Enriched context for agent {agent.name} in conversation {context.conversation_id}"
            )
            return enriched_context

        except Exception as e:
            logger.error(f"Failed to enrich context for agent {agent.name}: {e}")
            # Return minimal enriched context
            return {
                "agent_prompt": f"You are {agent.name}. Please respond to: {context.topic}",
                "conversation_summary": f"Discussion about: {context.topic}",
                "turn_count": context.current_turn_number,
                "agent_metadata": {"name": agent.name, "role": agent.role},
                "error": "Context enrichment failed - using fallback",
            }

    def serialize_context(self, context: ConversationContext) -> str:
        """
        Serialize context for persistence or transmission.

        Args:
            context: Context to serialize

        Returns:
            JSON string representation of context

        Raises:
            ValueError: If serialization fails
        """
        try:
            self._metrics["serialization_operations"] += 1

            # Convert to serializable dictionary
            context_dict = {
                "conversation_id": context.conversation_id,
                "topic": context.topic,
                "current_turn_number": context.current_turn_number,
                "context_window_size": context.context_window_size,
                "participants": [
                    {
                        "agent_id": p.agent_id,
                        "name": p.name,
                        "role": p.role,
                        "system_prompt": p.system_prompt,
                        "personality_traits": p.personality_traits,
                        "turn_order": p.turn_order,
                    }
                    for p in context.participants
                ],
                "turn_history": [
                    {
                        "turn_id": turn.turn_id,
                        "turn_number": turn.turn_number,
                        "agent_id": turn.agent_id,
                        "agent_name": turn.agent_name,
                        "response": turn.response,
                        "status": turn.status.value,
                        "created_at": turn.created_at.isoformat(),
                        "completed_at": turn.completed_at.isoformat()
                        if turn.completed_at
                        else None,
                        "duration_seconds": turn.duration_seconds,
                    }
                    for turn in context.turn_history
                ],
                "serialization_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "version": "1.0",
                },
            }

            serialized = json.dumps(context_dict, indent=2)
            logger.debug(f"Serialized context for conversation {context.conversation_id}")
            return serialized

        except Exception as e:
            self._metrics["serialization_failures"] += 1
            logger.error(
                f"Failed to serialize context for conversation {context.conversation_id}: {e}"
            )
            raise ValueError(f"Context serialization failed: {str(e)}")

    def deserialize_context(self, serialized_context: str) -> ConversationContext:
        """
        Deserialize context from JSON string.

        Args:
            serialized_context: JSON string representation

        Returns:
            Reconstructed ConversationContext object

        Raises:
            ValueError: If deserialization fails
        """
        try:
            context_dict = json.loads(serialized_context)

            # Reconstruct participants
            participants = []
            for p_data in context_dict["participants"]:
                participant = AgentRole(
                    agent_id=p_data["agent_id"],
                    name=p_data["name"],
                    role=p_data["role"],
                    system_prompt=p_data["system_prompt"],
                    personality_traits=p_data.get("personality_traits", []),
                    turn_order=p_data.get("turn_order", 0),
                )
                participants.append(participant)

            # Reconstruct turn history
            turn_history = []
            for turn_data in context_dict["turn_history"]:
                turn = ConversationTurnDomain(
                    turn_id=turn_data["turn_id"],
                    turn_number=turn_data["turn_number"],
                    agent_id=turn_data["agent_id"],
                    agent_name=turn_data["agent_name"],
                    prompt="",  # Prompt not serialized
                    response=turn_data["response"],
                    created_at=datetime.fromisoformat(turn_data["created_at"]),
                    completed_at=datetime.fromisoformat(turn_data["completed_at"])
                    if turn_data["completed_at"]
                    else None,
                )
                # Set status using the status value
                turn.status = turn_data["status"]
                turn_history.append(turn)

            # Reconstruct context
            context = ConversationContext(
                conversation_id=context_dict["conversation_id"],
                topic=context_dict["topic"],
                participants=participants,
                turn_history=turn_history,
                current_turn_number=context_dict["current_turn_number"],
                context_window_size=context_dict["context_window_size"],
            )

            logger.debug(f"Deserialized context for conversation {context.conversation_id}")
            return context

        except Exception as e:
            logger.error(f"Failed to deserialize context: {e}")
            raise ValueError(f"Context deserialization failed: {str(e)}")

    def invalidate_context_cache(self, conversation_id: str) -> None:
        """Invalidate cached context for a conversation."""
        if conversation_id in self._context_cache:
            del self._context_cache[conversation_id]
            logger.debug(f"Invalidated context cache for conversation {conversation_id}")

    def clear_conversation_data(self, conversation_id: str) -> None:
        """Clear all cached data for a conversation."""
        self.invalidate_context_cache(conversation_id)

        if conversation_id in self._turn_buffers:
            del self._turn_buffers[conversation_id]

        logger.info(f"Cleared all context data for conversation {conversation_id}")

    def get_context_metrics(self) -> Dict[str, Any]:
        """Get context operation metrics for observability."""
        return {
            **self._metrics,
            "active_conversations": len(self._context_cache),
            "active_buffers": len(self._turn_buffers),
            "cache_hit_rate": (
                self._metrics["cache_hits"]
                / max(self._metrics["cache_hits"] + self._metrics["cache_misses"], 1)
            ),
            "serialization_success_rate": (
                (
                    self._metrics["serialization_operations"]
                    - self._metrics["serialization_failures"]
                )
                / max(self._metrics["serialization_operations"], 1)
            ),
        }

    def cleanup_stale_contexts(self, max_age_hours: int = 24) -> int:
        """
        Clean up stale context caches and buffers.

        Args:
            max_age_hours: Maximum age for cached contexts

        Returns:
            Number of contexts cleaned up
        """
        # For now, implement simple cleanup based on cache size
        # In production, this would use timestamps to determine staleness

        cleaned_count = 0
        max_cache_size = 100  # Configurable limit

        if len(self._context_cache) > max_cache_size:
            # Remove oldest half of cached contexts
            cache_items = list(self._context_cache.items())
            contexts_to_remove = cache_items[: len(cache_items) // 2]

            for conversation_id, _ in contexts_to_remove:
                self.clear_conversation_data(conversation_id)
                cleaned_count += 1

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} stale contexts")

        return cleaned_count


# Factory function for dependency injection
def create_context_manager(
    max_turns_in_context: int = 10,
    max_characters_in_context: int = 4000,
    preserve_conversation_start: bool = True,
) -> ConversationContextManager:
    """Create context manager with specified configuration."""
    config = ContextWindowConfig(
        max_turns_in_context=max_turns_in_context,
        max_characters_in_context=max_characters_in_context,
        preserve_conversation_start=preserve_conversation_start,
    )
    return ConversationContextManager(config)
