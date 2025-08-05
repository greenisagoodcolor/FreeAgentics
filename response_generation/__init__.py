"""Response Generation Module for Active Inference Conversations.

This module implements high-performance response generation with natural language
enhancement, caching, streaming, and comprehensive monitoring.
"""

from .generator import ResponseGenerator, ProductionResponseGenerator
from .formatter import ResponseFormatter, StructuredResponseFormatter
from .cache import ResponseCache, InMemoryResponseCache, RedisResponseCache
from .streaming import ResponseStreamer, WebSocketResponseStreamer
from .nlg import NaturalLanguageGenerator, LLMEnhancedGenerator
from .models import (
    ResponseData,
    ActionExplanation,
    BeliefSummary,
    ConfidenceRating,
    ConfidenceLevel,
    ResponseMetadata,
    ResponseOptions,
    ResponseType,
)

__all__ = [
    "ResponseGenerator",
    "ProductionResponseGenerator",
    "ResponseFormatter",
    "StructuredResponseFormatter",
    "ResponseCache",
    "InMemoryResponseCache",
    "RedisResponseCache",
    "ResponseStreamer",
    "WebSocketResponseStreamer",
    "NaturalLanguageGenerator",
    "LLMEnhancedGenerator",
    "ResponseData",
    "ActionExplanation",
    "BeliefSummary",
    "ConfidenceRating",
    "ConfidenceLevel",
    "ResponseMetadata",
    "ResponseOptions",
    "ResponseType",
]
