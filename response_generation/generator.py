"""Response generator implementations with performance optimization."""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from agents.inference_engine import InferenceResult
from observability.prometheus_metrics import PrometheusMetricsCollector

from .models import ResponseData, ResponseOptions, ResponseType, ResponseMetadata
from .formatter import ResponseFormatter, StructuredResponseFormatter
from .cache import ResponseCache, InMemoryResponseCache
from .nlg import NaturalLanguageGenerator, LLMEnhancedGenerator
from .streaming import ResponseStreamer

logger = logging.getLogger(__name__)


class ResponseGenerationError(Exception):
    """Raised when response generation fails."""

    pass


class ResponseGenerator(ABC):
    """Abstract base class for response generators."""

    @abstractmethod
    async def generate_response(
        self,
        inference_result: InferenceResult,
        original_prompt: str,
        options: Optional[ResponseOptions] = None,
    ) -> ResponseData:
        """Generate a response from inference results.

        Args:
            inference_result: Results from PyMDP inference
            original_prompt: Original user prompt
            options: Response generation options

        Returns:
            Generated response data

        Raises:
            ResponseGenerationError: If generation fails
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get response generation metrics."""
        pass


class ProductionResponseGenerator(ResponseGenerator):
    """Production-ready response generator with caching and monitoring.

    This implementation follows the established patterns from the codebase:
    - Factory pattern for component creation
    - Comprehensive error handling and fallbacks
    - Performance monitoring and metrics collection
    - Clean separation of concerns through composition
    """

    def __init__(
        self,
        formatter: Optional[ResponseFormatter] = None,
        cache: Optional[ResponseCache] = None,
        nlg_generator: Optional[NaturalLanguageGenerator] = None,
        streamer: Optional[ResponseStreamer] = None,
        metrics_collector: Optional[PrometheusMetricsCollector] = None,
        enable_monitoring: bool = True,
    ):
        """Initialize the production response generator.

        Args:
            formatter: Response formatter (defaults to StructuredResponseFormatter)
            cache: Response cache (defaults to InMemoryResponseCache)
            nlg_generator: Natural language generator (defaults to LLMEnhancedGenerator)
            streamer: Response streamer (optional)
            metrics_collector: Prometheus metrics collector
            enable_monitoring: Enable performance monitoring
        """
        # Core components with sensible defaults
        self.formatter = formatter or StructuredResponseFormatter()
        self.cache = cache or InMemoryResponseCache(max_size=1000, default_ttl_seconds=300)
        self.nlg_generator = nlg_generator or LLMEnhancedGenerator()
        self.streamer = streamer

        # Monitoring and metrics
        self.enable_monitoring = enable_monitoring
        self.metrics_collector = metrics_collector or PrometheusMetricsCollector()

        # Performance tracking
        self._metrics = {
            "responses_generated": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "nlg_enhancements": 0,
            "nlg_failures": 0,
            "avg_generation_time_ms": 0.0,
            "streaming_responses": 0,
            "fallback_responses": 0,
        }

        logger.info("ProductionResponseGenerator initialized")

    async def generate_response(
        self,
        inference_result: InferenceResult,
        original_prompt: str,
        options: Optional[ResponseOptions] = None,
    ) -> ResponseData:
        """Generate a response from inference results with caching and monitoring."""
        start_time = time.time()
        response_id = str(uuid.uuid4())

        # Use default options if none provided
        if options is None:
            options = ResponseOptions()

        # Create metadata
        metadata = ResponseMetadata(
            response_id=response_id,
            generation_time_ms=0.0,  # Will be updated at the end
            trace_id=options.trace_id,
            conversation_id=options.conversation_id,
        )

        try:
            # Step 1: Check cache if enabled
            cache_key = None
            cached_response = None

            if options.enable_caching:
                cache_start = time.time()
                cache_key = self._generate_cache_key(inference_result, original_prompt, options)
                cached_response = await self.cache.get(cache_key)
                cache_time = (time.time() - cache_start) * 1000
                metadata.cache_lookup_time_ms = cache_time
                metadata.cache_key = cache_key

                if cached_response:
                    # Cache hit - update metadata and return
                    cached_response.metadata.cached = True
                    cached_response.metadata.cache_key = cache_key
                    cached_response.response_type = ResponseType.CACHED

                    self._update_metrics("cache_hits", 1)

                    logger.debug(f"Cache hit for response {response_id}")
                    return cached_response
                else:
                    self._update_metrics("cache_misses", 1)

            # Step 2: Generate structured response using formatter
            format_start = time.time()
            response_data = await self.formatter.format_response(
                inference_result=inference_result,
                original_prompt=original_prompt,
                options=options,
                metadata=metadata,
            )
            format_time = (time.time() - format_start) * 1000
            metadata.formatting_time_ms = format_time

            # Step 3: Enhance with natural language if enabled
            if options.enable_llm_enhancement and options.use_natural_language:
                nlg_start = time.time()

                try:
                    enhanced_message = await self.nlg_generator.enhance_message(
                        response_data=response_data,
                        original_prompt=original_prompt,
                        options=options,
                    )

                    if enhanced_message:
                        response_data.message = enhanced_message
                        response_data.response_type = ResponseType.ENHANCED
                        metadata.nlg_enhanced = True
                        self._update_metrics("nlg_enhancements", 1)

                except Exception as e:
                    logger.warning(f"NLG enhancement failed: {e}")
                    metadata.errors.append(f"NLG enhancement failed: {str(e)}")

                    if not options.fallback_on_llm_failure:
                        raise ResponseGenerationError(f"NLG enhancement failed: {str(e)}")

                    self._update_metrics("nlg_failures", 1)
                    metadata.fallback_used = True

                nlg_time = (time.time() - nlg_start) * 1000
                metadata.nlg_time_ms = nlg_time

            # Step 4: Handle streaming if enabled
            if options.enable_streaming and self.streamer:
                try:
                    await self.streamer.stream_response(response_data, options)
                    response_data.response_type = ResponseType.STREAMING
                    metadata.streaming = True
                    self._update_metrics("streaming_responses", 1)
                except Exception as e:
                    logger.warning(f"Streaming failed: {e}")
                    metadata.errors.append(f"Streaming failed: {str(e)}")

            # Step 5: Cache the response if enabled
            if options.enable_caching and cache_key and not cached_response:
                try:
                    await self.cache.set(
                        key=cache_key,
                        value=response_data,
                        ttl_seconds=options.cache_ttl_seconds,
                    )
                except Exception as e:
                    logger.warning(f"Caching failed: {e}")
                    metadata.errors.append(f"Caching failed: {str(e)}")

            # Step 6: Update final metadata
            total_time = (time.time() - start_time) * 1000
            metadata.generation_time_ms = total_time
            response_data.metadata = metadata

            # Update metrics
            self._update_metrics("responses_generated", 1)
            self._update_avg_time(total_time)

            # Record Prometheus metrics
            if self.enable_monitoring:
                self.metrics_collector.increment_counter(
                    "response_generator_requests_total",
                    {
                        "type": response_data.response_type.value,
                        "cached": str(metadata.cached),
                        "enhanced": str(metadata.nlg_enhanced),
                    },
                )

            logger.info(
                f"Response generated successfully in {total_time:.2f}ms",
                extra={
                    "response_id": response_id,
                    "type": response_data.response_type.value,
                    "cached": metadata.cached,
                    "enhanced": metadata.nlg_enhanced,
                    "trace_id": options.trace_id,
                },
            )

            return response_data

        except Exception as e:
            # Record error metrics
            total_time = (time.time() - start_time) * 1000
            metadata.generation_time_ms = total_time
            metadata.errors.append(str(e))

            if self.enable_monitoring:
                self.metrics_collector.increment_counter(
                    "response_generator_errors_total", {"error_type": type(e).__name__}
                )

            logger.error(
                f"Response generation failed after {total_time:.2f}ms: {e}",
                exc_info=True,
                extra={
                    "response_id": response_id,
                    "trace_id": options.trace_id,
                },
            )

            # Try to generate fallback response
            if options.fallback_on_llm_failure:
                try:
                    fallback_response = await self._generate_fallback_response(
                        inference_result=inference_result,
                        original_prompt=original_prompt,
                        options=options,
                        error=e,
                        metadata=metadata,
                    )
                    self._update_metrics("fallback_responses", 1)
                    return fallback_response
                except Exception as fallback_error:
                    logger.error(f"Fallback response generation failed: {fallback_error}")

            raise ResponseGenerationError(f"Response generation failed: {str(e)}") from e

    def get_metrics(self) -> Dict[str, Any]:
        """Get response generation metrics."""
        metrics = self._metrics.copy()

        # Add computed metrics
        total_requests = metrics["responses_generated"]
        total_cache_requests = metrics["cache_hits"] + metrics["cache_misses"]
        total_nlg_requests = metrics["nlg_enhancements"] + metrics["nlg_failures"]

        if total_cache_requests > 0:
            metrics["cache_hit_rate"] = metrics["cache_hits"] / total_cache_requests
        else:
            metrics["cache_hit_rate"] = 0.0

        if total_nlg_requests > 0:
            metrics["nlg_success_rate"] = metrics["nlg_enhancements"] / total_nlg_requests
        else:
            metrics["nlg_success_rate"] = 0.0

        if total_requests > 0:
            metrics["fallback_rate"] = metrics["fallback_responses"] / total_requests
        else:
            metrics["fallback_rate"] = 0.0

        return metrics

    def _generate_cache_key(
        self,
        inference_result: InferenceResult,
        original_prompt: str,
        options: ResponseOptions,
    ) -> str:
        """Generate cache key for response caching."""
        # Create deterministic key based on inputs
        key_parts = [
            str(inference_result.action),
            str(hash(str(inference_result.beliefs))),
            f"{inference_result.confidence:.3f}",
            str(hash(original_prompt)),
            str(options.narrative_style),
            str(options.use_natural_language),
            str(options.include_technical_details),
        ]

        key = "_".join(key_parts)
        return f"response_cache:{hash(key)}"

    async def _generate_fallback_response(
        self,
        inference_result: InferenceResult,
        original_prompt: str,
        options: ResponseOptions,
        error: Exception,
        metadata: ResponseMetadata,
    ) -> ResponseData:
        """Generate a simple fallback response when normal generation fails."""
        logger.info("Generating fallback response")

        # Create minimal options for fallback
        fallback_options = ResponseOptions(
            include_technical_details=False,
            include_alternatives=False,
            narrative_style=False,
            use_natural_language=False,
            enable_llm_enhancement=False,
            enable_caching=False,
            enable_streaming=False,
        )

        # Use formatter for basic structured response
        response_data = await self.formatter.format_response(
            inference_result=inference_result,
            original_prompt=original_prompt,
            options=fallback_options,
            metadata=metadata,
        )

        # Override message with simple fallback
        try:
            confidence = response_data.confidence_rating.overall
            confidence_text = f"with confidence {confidence:.2f}"
        except (AttributeError, TypeError):
            confidence_text = "using Active Inference principles"

        response_data.message = (
            f"I processed your request '{original_prompt}' and generated a response "
            f"based on Active Inference principles. The system selected an action "
            f"{confidence_text}."
        )

        metadata.fallback_used = True
        metadata.template_used = "fallback_simple"
        response_data.metadata = metadata

        return response_data

    def _update_metrics(self, metric_name: str, value: float) -> None:
        """Update internal metrics."""
        if metric_name in self._metrics:
            self._metrics[metric_name] += value

    def _update_avg_time(self, time_ms: float) -> None:
        """Update average generation time using exponential moving average."""
        current_avg = self._metrics["avg_generation_time_ms"]
        if current_avg == 0.0:
            self._metrics["avg_generation_time_ms"] = time_ms
        else:
            # Use exponential moving average with alpha = 0.1
            alpha = 0.1
            self._metrics["avg_generation_time_ms"] = (alpha * time_ms) + (
                (1 - alpha) * current_avg
            )
