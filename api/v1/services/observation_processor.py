"""
Observation Processor for Converting Messages to PyMDP Observations

Converts conversation messages into discrete observations suitable for PyMDP
belief updates. Implements natural language understanding for Active Inference.

Following Nemesis Committee consensus:
- Sindre Sorhus: Small, focused module with single responsibility
- Addy Osmani: Performance-optimized with caching
- Jessica Kerr: Comprehensive observability and metrics
"""

import logging
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ObservationType(Enum):
    """Types of observations extracted from messages."""

    UNCERTAIN = 0
    NEUTRAL = 1
    CONFIDENT = 2


@dataclass
class ObservationExtractionResult:
    """Result of observation extraction with metrics."""

    observation: int
    confidence: float
    extraction_time_ms: float
    features_detected: List[str]
    raw_scores: Dict[str, float]


class ObservationProcessor:
    """
    Processes conversation messages to extract PyMDP observations.

    Uses rule-based NLP and sentiment analysis to convert natural language
    into discrete observations suitable for Active Inference belief updates.
    """

    def __init__(self, cache_size: int = 1000):
        """Initialize observation processor with optional caching.

        Args:
            cache_size: Maximum number of cached extractions
        """
        self._cache = {}
        self._cache_size = cache_size
        self._extraction_count = 0
        self._total_extraction_time = 0.0

        # Compile regex patterns for performance
        self._uncertainty_patterns = [
            re.compile(
                r"\b(uncertain|unsure|confused|don\'t know|not sure|maybe|perhaps|might)\b",
                re.IGNORECASE,
            ),
            re.compile(r"\b(unclear|ambiguous|questionable|doubtful|hesitant)\b", re.IGNORECASE),
            re.compile(r"\?{2,}", re.IGNORECASE),  # Multiple question marks
            re.compile(r"\b(I think|I guess|I suppose|I assume)\b", re.IGNORECASE),
        ]

        self._confidence_patterns = [
            re.compile(
                r"\b(certain|sure|confident|absolutely|definitely|clearly)\b", re.IGNORECASE
            ),
            re.compile(
                r"\b(obvious|evident|undoubtedly|without doubt|no question)\b", re.IGNORECASE
            ),
            re.compile(r"\b(I know|I\'m sure|I\'m certain|I\'m confident)\b", re.IGNORECASE),
            re.compile(r"!{2,}", re.IGNORECASE),  # Multiple exclamation marks
        ]

        self._emotional_patterns = {
            "frustration": re.compile(
                r"\b(frustrated|annoying|irritating|ugh|damn)\b", re.IGNORECASE
            ),
            "excitement": re.compile(
                r"\b(excited|amazing|awesome|great|fantastic)\b", re.IGNORECASE
            ),
            "concern": re.compile(r"\b(worried|concerned|anxious|nervous|afraid)\b", re.IGNORECASE),
        }

        logger.info("Initialized observation processor with pattern matching")

    def extract_observation(
        self, message: Dict[str, Any], conversation_context: Optional[Dict[str, Any]] = None
    ) -> int:
        """Extract PyMDP observation from conversation message.

        Args:
            message: Message with content, role, timestamp, etc.
            conversation_context: Additional context for extraction

        Returns:
            Observation index (0=uncertain, 1=neutral, 2=confident)
        """
        # Create cache key for performance
        content = message.get("content", "")
        role = message.get("role", "")
        cache_key = f"{content[:100]}_{role}"  # Truncate for cache efficiency

        if cache_key in self._cache:
            logger.debug("Using cached observation extraction")
            return self._cache[cache_key].observation

        # Perform extraction
        result = self._extract_with_metrics(message, conversation_context)

        # Cache result if we have space
        if len(self._cache) < self._cache_size:
            self._cache[cache_key] = result

        # Update statistics
        self._extraction_count += 1
        self._total_extraction_time += result.extraction_time_ms

        logger.debug(
            f"Extracted observation {result.observation} with confidence {result.confidence:.3f} "
            f"in {result.extraction_time_ms:.2f}ms"
        )

        return result.observation

    def extract_with_details(
        self, message: Dict[str, Any], conversation_context: Optional[Dict[str, Any]] = None
    ) -> ObservationExtractionResult:
        """Extract observation with detailed metrics and explanations.

        Args:
            message: Message with content, role, timestamp, etc.
            conversation_context: Additional context for extraction

        Returns:
            Detailed extraction result with metrics
        """
        return self._extract_with_metrics(message, conversation_context)

    def _extract_with_metrics(
        self, message: Dict[str, Any], conversation_context: Optional[Dict[str, Any]] = None
    ) -> ObservationExtractionResult:
        """Internal method that performs extraction with full metrics.

        Args:
            message: Message to process
            conversation_context: Additional context

        Returns:
            Detailed extraction result
        """
        start_time = time.time()

        content = message.get("content", "").strip()
        role = message.get("role", "user")

        if not content:
            # Empty content is neutral
            return ObservationExtractionResult(
                observation=ObservationType.NEUTRAL.value,
                confidence=1.0,
                extraction_time_ms=(time.time() - start_time) * 1000,
                features_detected=["empty_content"],
                raw_scores={"uncertainty": 0.0, "confidence": 0.0, "neutral": 1.0},
            )

        # Calculate uncertainty score
        uncertainty_score = self._calculate_uncertainty_score(content)

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(content)

        # Calculate contextual adjustments
        context_adjustment = self._calculate_context_adjustment(message, conversation_context)

        # Adjust scores based on context
        uncertainty_score += context_adjustment.get("uncertainty", 0.0)
        confidence_score += context_adjustment.get("confidence", 0.0)

        # Normalize scores
        total_score = uncertainty_score + confidence_score
        neutral_score = max(0.0, 1.0 - total_score)

        # Determine observation type
        observation, extraction_confidence, features = self._determine_observation(
            uncertainty_score, confidence_score, neutral_score, content
        )

        extraction_time_ms = (time.time() - start_time) * 1000

        return ObservationExtractionResult(
            observation=observation,
            confidence=extraction_confidence,
            extraction_time_ms=extraction_time_ms,
            features_detected=features,
            raw_scores={
                "uncertainty": uncertainty_score,
                "confidence": confidence_score,
                "neutral": neutral_score,
            },
        )

    def _calculate_uncertainty_score(self, content: str) -> float:
        """Calculate uncertainty score from message content.

        Args:
            content: Message content to analyze

        Returns:
            Uncertainty score between 0.0 and 1.0
        """
        score = 0.0

        for pattern in self._uncertainty_patterns:
            matches = pattern.findall(content)
            score += len(matches) * 0.2  # Each match adds to uncertainty

        # Question marks indicate uncertainty
        question_marks = content.count("?")
        score += min(question_marks * 0.1, 0.3)  # Cap at 0.3

        # Length-based uncertainty (very short or very long messages can be uncertain)
        length = len(content.split())
        if length < 3:
            score += 0.1  # Short messages often uncertain
        elif length > 50:
            score += 0.05  # Very long messages might be rambling

        return min(score, 1.0)  # Cap at 1.0

    def _calculate_confidence_score(self, content: str) -> float:
        """Calculate confidence score from message content.

        Args:
            content: Message content to analyze

        Returns:
            Confidence score between 0.0 and 1.0
        """
        score = 0.0

        for pattern in self._confidence_patterns:
            matches = pattern.findall(content)
            score += len(matches) * 0.25  # Each match adds to confidence

        # Exclamation marks indicate confidence
        exclamation_marks = content.count("!")
        score += min(exclamation_marks * 0.1, 0.2)  # Cap at 0.2

        # Definitive statements (no qualifiers) are more confident
        if not any(word in content.lower() for word in ["maybe", "perhaps", "might", "could"]):
            score += 0.1

        # Technical or specific language can indicate confidence
        if any(
            word in content.lower()
            for word in ["specifically", "precisely", "exactly", "definitely"]
        ):
            score += 0.1

        return min(score, 1.0)  # Cap at 1.0

    def _calculate_context_adjustment(
        self, message: Dict[str, Any], conversation_context: Optional[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate contextual adjustments to observation scores.

        Args:
            message: Current message
            conversation_context: Conversation context

        Returns:
            Dictionary with score adjustments
        """
        adjustments = {"uncertainty": 0.0, "confidence": 0.0}

        if not conversation_context:
            return adjustments

        # Role-based adjustments
        role = message.get("role", "")
        if role == "system":
            adjustments["confidence"] += 0.2  # System messages are typically confident
        elif role == "assistant":
            adjustments["confidence"] += 0.1  # Assistant responses tend to be confident

        # Previous message analysis
        previous_messages = conversation_context.get("recent_messages", [])
        if previous_messages:
            last_message = previous_messages[-1]
            last_content = last_message.get("content", "").lower()

            # If previous message was uncertain, current might be clarifying
            if any(word in last_content for word in ["uncertain", "confused", "unclear"]):
                adjustments["confidence"] += 0.1

        # Conversation length adjustment
        turn_count = conversation_context.get("turn_count", 0)
        if turn_count > 10:
            # Long conversations might build confidence
            adjustments["confidence"] += 0.05
        elif turn_count < 3:
            # Early in conversation might be more uncertain
            adjustments["uncertainty"] += 0.05

        return adjustments

    def _determine_observation(
        self, uncertainty_score: float, confidence_score: float, neutral_score: float, content: str
    ) -> Tuple[int, float, List[str]]:
        """Determine final observation type and confidence.

        Args:
            uncertainty_score: Calculated uncertainty score
            confidence_score: Calculated confidence score
            neutral_score: Calculated neutral score
            content: Original message content

        Returns:
            Tuple of (observation_type, confidence, features_detected)
        """
        features = []

        # Determine observation based on highest score
        if confidence_score > uncertainty_score and confidence_score > neutral_score:
            observation = ObservationType.CONFIDENT.value
            extraction_confidence = confidence_score
            features.append("high_confidence_language")
        elif uncertainty_score > neutral_score:
            observation = ObservationType.UNCERTAIN.value
            extraction_confidence = uncertainty_score
            features.append("uncertainty_indicators")
        else:
            observation = ObservationType.NEUTRAL.value
            extraction_confidence = neutral_score
            features.append("neutral_tone")

        # Add detected emotional features
        for emotion, pattern in self._emotional_patterns.items():
            if pattern.search(content):
                features.append(f"emotion_{emotion}")

        # Add structural features
        if "?" in content:
            features.append("contains_questions")
        if "!" in content:
            features.append("contains_exclamations")

        word_count = len(content.split())
        if word_count < 5:
            features.append("short_message")
        elif word_count > 30:
            features.append("long_message")

        return observation, extraction_confidence, features

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring.

        Returns:
            Dictionary with performance metrics
        """
        avg_extraction_time = 0.0
        if self._extraction_count > 0:
            avg_extraction_time = self._total_extraction_time / self._extraction_count

        return {
            "total_extractions": self._extraction_count,
            "cache_size": len(self._cache),
            "cache_hit_ratio": len(self._cache) / max(self._extraction_count, 1),
            "average_extraction_time_ms": avg_extraction_time,
            "total_extraction_time_ms": self._total_extraction_time,
        }

    def clear_cache(self) -> None:
        """Clear the extraction cache."""
        self._cache.clear()
        logger.info("Cleared observation extraction cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for debugging.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_entries": len(self._cache),
            "cache_capacity": self._cache_size,
            "cache_utilization": len(self._cache) / self._cache_size,
        }
