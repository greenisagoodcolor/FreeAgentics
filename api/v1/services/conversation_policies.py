"""
Conversation Policy Implementations

This module provides concrete implementations of conversation policies
for turn limits, completion detection, and other business rules.
"""

import logging
from typing import Optional

from api.v1.models.agent_conversation import (
    CompletionReason,
    ConversationAggregate,
    ConversationContext,
    ICompletionDetector,
    ITurnLimitPolicy,
)

logger = logging.getLogger(__name__)


class DefaultTurnLimitPolicy(ITurnLimitPolicy):
    """Default policy for turn limit management."""

    def should_continue(self, conversation: ConversationAggregate) -> bool:
        """Determine if conversation should continue based on turn limits."""
        if not conversation.is_active:
            return False

        # Check turn limit
        if len(conversation.turns) >= conversation.config.max_turns:
            return False

        # Check conversation timeout
        if (
            conversation.duration_minutes
            and conversation.duration_minutes > conversation.config.conversation_timeout_minutes
        ):
            return False

        return True

    def get_completion_reason(
        self, conversation: ConversationAggregate
    ) -> Optional[CompletionReason]:
        """Get reason for conversation completion."""
        if len(conversation.turns) >= conversation.config.max_turns:
            return CompletionReason.TURN_LIMIT_REACHED

        if (
            conversation.duration_minutes
            and conversation.duration_minutes > conversation.config.conversation_timeout_minutes
        ):
            return CompletionReason.TIMEOUT

        return None


class ConsensusDetectionPolicy(ICompletionDetector):
    """Detects when agents reach consensus in their responses."""

    # Keywords that indicate agreement or consensus
    CONSENSUS_KEYWORDS = [
        "agree",
        "agreed",
        "consensus",
        "same",
        "similar",
        "concur",
        "exactly",
        "precisely",
        "correct",
        "right",
        "yes",
        "indeed",
        "absolutely",
        "definitely",
        "certainly",
        "unanimous",
    ]

    # Keywords that indicate disagreement
    DISAGREEMENT_KEYWORDS = [
        "disagree",
        "wrong",
        "incorrect",
        "no",
        "however",
        "but",
        "although",
        "contrary",
        "oppose",
        "different",
        "alternative",
    ]

    def __init__(self, consensus_threshold: float = 0.7):
        """
        Initialize consensus detector.

        Args:
            consensus_threshold: Minimum ratio of agreeable responses to detect consensus
        """
        self.consensus_threshold = consensus_threshold

    async def detect_completion(self, context: ConversationContext) -> Optional[CompletionReason]:
        """Detect if conversation should be completed due to consensus."""
        if len(context.recent_turns) < len(context.participants):
            # Need at least one turn from each participant
            return None

        # Get the most recent round of responses (one from each participant)
        recent_round = context.recent_turns[-len(context.participants) :]

        if len(recent_round) < 2:
            # Need at least 2 responses to detect consensus
            return None

        # Analyze consensus in the recent round
        consensus_score = self._calculate_consensus_score(recent_round)

        if consensus_score >= self.consensus_threshold:
            logger.info(
                f"Consensus detected with score {consensus_score:.2f} in conversation {context.conversation_id}"
            )
            return CompletionReason.CONSENSUS_REACHED

        return None

    def _calculate_consensus_score(self, turns) -> float:
        """Calculate consensus score based on response content."""
        if not turns:
            return 0.0

        responses = [turn.response.lower() for turn in turns if turn.response]
        if len(responses) < 2:
            return 0.0

        # Count consensus and disagreement indicators
        consensus_counts = []
        disagreement_counts = []

        for response in responses:
            consensus_count = sum(1 for keyword in self.CONSENSUS_KEYWORDS if keyword in response)
            disagreement_count = sum(
                1 for keyword in self.DISAGREEMENT_KEYWORDS if keyword in response
            )

            consensus_counts.append(consensus_count)
            disagreement_counts.append(disagreement_count)

        # Calculate overall consensus score
        total_consensus = sum(consensus_counts)
        total_disagreement = sum(disagreement_counts)
        total_indicators = total_consensus + total_disagreement

        if total_indicators == 0:
            # No clear indicators, check for similar response lengths and patterns
            return self._calculate_similarity_score(responses)

        # Consensus score based on ratio of consensus vs disagreement indicators
        consensus_ratio = total_consensus / total_indicators
        return consensus_ratio

    def _calculate_similarity_score(self, responses) -> float:
        """Calculate similarity score when no explicit consensus indicators are found."""
        if len(responses) < 2:
            return 0.0

        # Simple similarity based on response length consistency
        lengths = [len(response) for response in responses]
        avg_length = sum(lengths) / len(lengths)

        # Calculate coefficient of variation (lower = more similar)
        if avg_length == 0:
            return 0.0

        variance = sum((length - avg_length) ** 2 for length in lengths) / len(lengths)
        std_dev = variance**0.5
        cv = std_dev / avg_length

        # Convert to similarity score (lower CV = higher similarity)
        similarity = max(0.0, 1.0 - cv)

        # Only consider high similarity as potential consensus
        return similarity if similarity > 0.8 else 0.0


class TaskCompletionDetector(ICompletionDetector):
    """Detects when a specific task or goal has been completed."""

    COMPLETION_KEYWORDS = [
        "completed",
        "finished",
        "done",
        "solved",
        "resolved",
        "accomplished",
        "achieved",
        "success",
        "successful",
        "conclusion",
        "final",
        "end",
        "wrap up",
        "summary",
    ]

    async def detect_completion(self, context: ConversationContext) -> Optional[CompletionReason]:
        """Detect if conversation should be completed due to task completion."""
        if not context.recent_turns:
            return None

        # Check the most recent few turns for completion indicators
        recent_responses = [
            turn.response.lower() for turn in context.recent_turns[-3:] if turn.response
        ]

        if not recent_responses:
            return None

        # Count completion indicators in recent responses
        completion_indicators = 0
        total_responses = len(recent_responses)

        for response in recent_responses:
            if any(keyword in response for keyword in self.COMPLETION_KEYWORDS):
                completion_indicators += 1

        # If majority of recent responses indicate completion
        if completion_indicators > total_responses / 2:
            logger.info(f"Task completion detected in conversation {context.conversation_id}")
            return CompletionReason.TASK_COMPLETED

        return None


class CompositeCompletionDetector(ICompletionDetector):
    """Combines multiple completion detectors."""

    def __init__(self, detectors: list[ICompletionDetector]):
        """Initialize with list of completion detectors."""
        self.detectors = detectors

    async def detect_completion(self, context: ConversationContext) -> Optional[CompletionReason]:
        """Check all detectors and return first completion reason found."""
        for detector in self.detectors:
            try:
                reason = await detector.detect_completion(context)
                if reason:
                    return reason
            except Exception as e:
                logger.warning(f"Completion detector {type(detector).__name__} failed: {e}")
                continue

        return None
