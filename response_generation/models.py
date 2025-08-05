"""Response generation data models and value objects."""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from enum import Enum


class ConfidenceLevel(Enum):
    """Confidence level categories for user-friendly display."""

    VERY_LOW = "very_low"  # 0.0 - 0.2
    LOW = "low"  # 0.2 - 0.4
    MODERATE = "moderate"  # 0.4 - 0.6
    HIGH = "high"  # 0.6 - 0.8
    VERY_HIGH = "very_high"  # 0.8 - 1.0

    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        """Convert numeric confidence score to level."""
        if score < 0.2:
            return cls.VERY_LOW
        elif score < 0.4:
            return cls.LOW
        elif score < 0.6:
            return cls.MODERATE
        elif score < 0.8:
            return cls.HIGH
        else:
            return cls.VERY_HIGH


class ResponseType(Enum):
    """Type of response being generated."""

    STRUCTURED = "structured"  # Basic structured response
    ENHANCED = "enhanced"  # LLM-enhanced narrative response
    STREAMING = "streaming"  # Real-time streaming response
    CACHED = "cached"  # Retrieved from cache


@dataclass
class ActionExplanation:
    """Explanation of the selected action."""

    action: Union[int, str, Any]
    action_label: Optional[str] = None
    rationale: Optional[str] = None
    alternatives_considered: List[str] = field(default_factory=list)
    decision_factors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "action": self.action,
            "action_label": self.action_label,
            "rationale": self.rationale,
            "alternatives_considered": self.alternatives_considered,
            "decision_factors": self.decision_factors,
        }


@dataclass
class BeliefSummary:
    """Summary of agent's belief state."""

    states: Union[List[float], List[List[float]]]
    entropy: float
    most_likely_state: Optional[str] = None
    belief_distribution: Optional[Dict[str, float]] = None
    uncertainty_areas: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "states": self.states,
            "entropy": self.entropy,
            "most_likely_state": self.most_likely_state,
            "belief_distribution": self.belief_distribution,
            "uncertainty_areas": self.uncertainty_areas,
        }


@dataclass
class ConfidenceRating:
    """Confidence assessment with multiple dimensions."""

    overall: float
    action_confidence: float
    belief_confidence: float
    model_confidence: float = 1.0
    factors: Dict[str, float] = field(default_factory=dict)
    level: Optional[ConfidenceLevel] = None

    def __post_init__(self):
        """Initialize computed fields."""
        if self.level is None and self.overall is not None:
            self.level = ConfidenceLevel.from_score(self.overall)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall": self.overall,
            "level": self.level.value,
            "action_confidence": self.action_confidence,
            "belief_confidence": self.belief_confidence,
            "model_confidence": self.model_confidence,
            "factors": self.factors,
        }


@dataclass
class ResponseMetadata:
    """Metadata about response generation process."""

    response_id: str
    generation_time_ms: float
    cached: bool = False
    cache_key: Optional[str] = None
    nlg_enhanced: bool = False
    streaming: bool = False

    # Performance metrics
    formatting_time_ms: float = 0.0
    nlg_time_ms: float = 0.0
    cache_lookup_time_ms: float = 0.0

    # Generation details
    template_used: Optional[str] = None
    fallback_used: bool = False
    errors: List[str] = field(default_factory=list)

    # Timestamp and tracing
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trace_id: Optional[str] = None
    conversation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "response_id": self.response_id,
            "generation_time_ms": self.generation_time_ms,
            "cached": self.cached,
            "cache_key": self.cache_key,
            "nlg_enhanced": self.nlg_enhanced,
            "streaming": self.streaming,
            "formatting_time_ms": self.formatting_time_ms,
            "nlg_time_ms": self.nlg_time_ms,
            "cache_lookup_time_ms": self.cache_lookup_time_ms,
            "template_used": self.template_used,
            "fallback_used": self.fallback_used,
            "errors": self.errors,
            "timestamp": self.timestamp.isoformat(),
            "trace_id": self.trace_id,
            "conversation_id": self.conversation_id,
        }


@dataclass
class ResponseData:
    """Complete response data structure."""

    # Core response content
    message: str
    action_explanation: ActionExplanation
    belief_summary: BeliefSummary
    confidence_rating: ConfidenceRating

    # Optional enrichment data
    knowledge_graph_updates: Optional[Dict[str, Any]] = None
    related_concepts: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)

    # Response metadata
    metadata: ResponseMetadata = field(
        default_factory=lambda: ResponseMetadata(
            response_id=str(int(time.time() * 1000000)),  # Microsecond timestamp
            generation_time_ms=0.0,
        )
    )

    # Response type and formatting
    response_type: ResponseType = ResponseType.STRUCTURED
    format_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "message": self.message,
            "action_explanation": self.action_explanation.to_dict(),
            "belief_summary": self.belief_summary.to_dict(),
            "confidence_rating": self.confidence_rating.to_dict(),
            "knowledge_graph_updates": self.knowledge_graph_updates,
            "related_concepts": self.related_concepts,
            "suggested_actions": self.suggested_actions,
            "metadata": self.metadata.to_dict(),
            "response_type": self.response_type.value,
            "format_version": self.format_version,
        }

    def to_json_serializable(self) -> Dict[str, Any]:
        """Convert to JSON-serializable format for API responses."""
        return self.to_dict()


@dataclass
class ResponseOptions:
    """Options for controlling response generation."""

    # Content options
    include_technical_details: bool = False
    include_alternatives: bool = True
    include_confidence_breakdown: bool = True
    include_knowledge_graph: bool = True

    # Format options
    narrative_style: bool = True
    use_natural_language: bool = True
    max_message_length: int = 500

    # Performance options
    enable_caching: bool = True
    cache_ttl_seconds: int = 300  # 5 minutes
    enable_streaming: bool = False

    # Enhancement options
    enable_llm_enhancement: bool = True
    llm_timeout_ms: int = 3000  # 3 seconds
    fallback_on_llm_failure: bool = True

    # Monitoring options
    trace_id: Optional[str] = None
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "include_technical_details": self.include_technical_details,
            "include_alternatives": self.include_alternatives,
            "include_confidence_breakdown": self.include_confidence_breakdown,
            "include_knowledge_graph": self.include_knowledge_graph,
            "narrative_style": self.narrative_style,
            "use_natural_language": self.use_natural_language,
            "max_message_length": self.max_message_length,
            "enable_caching": self.enable_caching,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "enable_streaming": self.enable_streaming,
            "enable_llm_enhancement": self.enable_llm_enhancement,
            "llm_timeout_ms": self.llm_timeout_ms,
            "fallback_on_llm_failure": self.fallback_on_llm_failure,
            "trace_id": self.trace_id,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
        }
