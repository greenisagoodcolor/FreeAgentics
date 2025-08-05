"""Conversation Orchestrator with Comprehensive Error Handling.

This orchestrator coordinates the complete end-to-end conversation flow:
1. User prompt input
2. LLM generates GMN specification
3. GMN parser validates and parses
4. PyMDP agent factory creates agent
5. Inference engine runs agent reasoning
6. Knowledge graph updates from results
7. Response generation back to user

Implements the Pipeline pattern with circuit breakers, comprehensive error
handling, observability, and production-ready resilience patterns.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Core infrastructure imports
from agents.pymdp_agent_factory import PyMDPAgentFactory, PyMDPAgentCreationError
from agents.inference_engine import InferenceEngine, InferenceError, InferenceResult
from inference.active.gmn_parser import GMNParser, GMNValidationError
from inference.llm.provider_factory import create_llm_manager
from inference.llm.provider_interface import GenerationRequest
from knowledge_graph.updater import KnowledgeGraphUpdater
from observability.prometheus_metrics import PrometheusMetricsCollector

# Orchestration imports
from .errors import (
    OrchestrationError,
    ComponentTimeoutError,
    ValidationError,
    PipelineExecutionError,
    create_error_context,
)
from .pipeline import ConversationPipeline, PipelineStep, PipelineContext, StepResult
from .monitoring import HealthChecker, MetricsCollector, HealthStatus

logger = logging.getLogger(__name__)


@dataclass
class OrchestrationRequest:
    """Request for conversation orchestration."""

    prompt: str
    user_id: str
    conversation_id: Optional[str] = None
    trace_id: Optional[str] = None

    # LLM options
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    temperature: float = 0.7

    # Processing options
    enable_pymdp: bool = True
    enable_knowledge_graph: bool = True
    timeout_ms: float = 30000

    # Fallback options
    max_retries: int = 3
    enable_fallbacks: bool = True

    def __post_init__(self):
        """Initialize computed fields."""
        if self.conversation_id is None:
            self.conversation_id = str(uuid.uuid4())
        if self.trace_id is None:
            self.trace_id = str(uuid.uuid4())


@dataclass
class OrchestrationResult:
    """Result of conversation orchestration."""

    request: OrchestrationRequest
    success: bool

    # Generated content
    response: Optional[str] = None
    gmn_spec: Optional[Dict[str, Any]] = None
    inference_result: Optional[InferenceResult] = None

    # Execution metadata
    execution_time_ms: float = 0.0
    steps_completed: List[str] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    # Pipeline information
    pipeline_results: Optional[Dict[str, Any]] = None
    fallbacks_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "conversation_id": self.request.conversation_id,
            "trace_id": self.request.trace_id,
            "success": self.success,
            "response": self.response,
            "execution_time_ms": self.execution_time_ms,
            "steps_completed": self.steps_completed,
            "errors": self.errors,
            "fallbacks_used": self.fallbacks_used,
            "pipeline_results": self.pipeline_results,
            "inference_metadata": self.inference_result.to_dict()
            if self.inference_result
            else None,
        }


# Pipeline Step Implementations


class LLMGenerationStep(PipelineStep):
    """Generate GMN specification using LLM."""

    def __init__(self, llm_manager=None):
        super().__init__(
            name="llm_generation",
            timeout_ms=15000,  # 15 second timeout for LLM calls
            max_retries=3,
            required=True,
        )
        self.llm_manager = llm_manager

    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """Generate GMN spec from user prompt."""
        request_data = context.get_data("request")
        prompt = request_data["prompt"]

        # Create LLM manager if not provided
        if self.llm_manager is None:
            self.llm_manager = create_llm_manager(user_id=context.user_id)

        # Build prompt for GMN generation
        gmn_prompt = self._build_gmn_prompt(prompt)

        generation_request = GenerationRequest(
            messages=[{"role": "user", "content": gmn_prompt}],
            model=request_data.get("llm_model", "gpt-3.5-turbo"),
            temperature=request_data.get("temperature", 0.7),
            max_tokens=1000,
        )

        try:
            response = self.llm_manager.generate_with_fallback(generation_request)
            content = response.content if hasattr(response, "content") else str(response)

            return {
                "gmn_text": content,
                "llm_provider": getattr(response, "provider", "unknown"),
                "llm_model": generation_request.model,
            }

        except Exception as e:
            raise ComponentTimeoutError(
                component="llm_provider",
                timeout_ms=self.timeout_ms,
                context=context,
                cause=e,
            )

    def _build_gmn_prompt(self, user_prompt: str) -> str:
        """Build prompt for GMN generation."""
        return f"""Generate a GMN (Generalized Model Notation) specification for an Active Inference agent based on this prompt:

"{user_prompt}"

Please provide a JSON response with the following structure:
{{
  "nodes": [
    {{"id": "state1", "type": "state", "properties": {{"num_states": 3}}}},
    {{"id": "obs1", "type": "observation", "properties": {{"num_observations": 3}}}},
    {{"id": "action1", "type": "action", "properties": {{"num_actions": 2}}}}
  ],
  "edges": [
    {{"source": "state1", "target": "obs1", "type": "generates"}},
    {{"source": "action1", "target": "state1", "type": "influences"}}
  ],
  "metadata": {{"description": "Brief description of the model"}}
}}

Keep the model simple with 2-4 states, 2-4 observations, and 2-3 actions."""


class GMNParsingStep(PipelineStep):
    """Parse and validate GMN specification."""

    def __init__(self):
        super().__init__(
            name="gmn_parsing",
            timeout_ms=5000,
            max_retries=2,
            required=True,
        )
        self.parser = GMNParser()

    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """Parse GMN text into structured specification."""
        llm_data = context.get_data("llm_generation")
        gmn_text = llm_data["gmn_text"]

        try:
            # Parse GMN specification
            graph = self.parser.parse(gmn_text)

            # Convert to PyMDP model format
            pymdp_spec = self.parser.to_pymdp_model(graph)

            return {
                "gmn_graph": graph,
                "pymdp_spec": pymdp_spec,
                "validation_passed": True,
            }

        except Exception as e:
            if isinstance(e, (ValueError, GMNValidationError)):
                raise ValidationError(
                    field="gmn_specification",
                    value=gmn_text[:200] + "..." if len(gmn_text) > 200 else gmn_text,
                    validation_rule="valid GMN format",
                    context=create_error_context(
                        trace_id=context.trace_id,
                        conversation_id=context.conversation_id,
                        step_name=self.name,
                        component="gmn_parser",
                    ),
                )
            else:
                raise

    async def handle_failure(self, context: PipelineContext, error: Exception) -> bool:
        """Handle GMN parsing failures with potential text cleanup."""
        if isinstance(error, ValidationError):
            # For validation errors, we could try to clean up the GMN text
            # This is a simple retry strategy
            return True
        return super().handle_failure(context, error)


class AgentCreationStep(PipelineStep):
    """Create PyMDP agent from GMN specification."""

    def __init__(self):
        super().__init__(
            name="agent_creation",
            timeout_ms=10000,
            max_retries=2,
            required=True,
        )
        self.factory = PyMDPAgentFactory()

    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """Create PyMDP agent."""
        gmn_data = context.get_data("gmn_parsing")
        pymdp_spec = gmn_data["pymdp_spec"]

        try:
            # Create agent in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            agent = await loop.run_in_executor(None, self.factory.create_agent, pymdp_spec)

            return {
                "agent": agent,
                "agent_metrics": self.factory.get_metrics(),
            }

        except PyMDPAgentCreationError as e:
            raise ComponentTimeoutError(
                component="pymdp_factory",
                timeout_ms=self.timeout_ms,
                context=create_error_context(
                    trace_id=context.trace_id,
                    conversation_id=context.conversation_id,
                    step_name=self.name,
                    component="pymdp_factory",
                ),
                cause=e,
            )


class InferenceStep(PipelineStep):
    """Run PyMDP inference."""

    def __init__(self):
        super().__init__(
            name="inference",
            timeout_ms=8000,
            max_retries=2,
            required=True,
        )
        self.engine = InferenceEngine(max_workers=2, default_timeout_ms=7000)

    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """Run inference."""
        agent_data = context.get_data("agent_creation")
        agent = agent_data["agent"]

        # Generate simple observation for demonstration
        # In production, this would come from the conversation context
        observation = [0]  # Simple initial observation

        try:
            # Run inference
            result = self.engine.run_inference(
                agent=agent,
                observation=observation,
                timeout_ms=self.timeout_ms - 1000,  # Leave buffer for cleanup
            )

            if result is None:
                raise InferenceError("Inference operation was cancelled or timed out")

            return {
                "inference_result": result,
                "engine_metrics": self.engine.get_metrics(),
            }

        except InferenceError as e:
            raise ComponentTimeoutError(
                component="inference_engine",
                timeout_ms=self.timeout_ms,
                context=create_error_context(
                    trace_id=context.trace_id,
                    conversation_id=context.conversation_id,
                    step_name=self.name,
                    component="inference_engine",
                ),
                cause=e,
            )


class KnowledgeGraphUpdateStep(PipelineStep):
    """Update knowledge graph with inference results."""

    def __init__(self):
        super().__init__(
            name="knowledge_graph_update",
            timeout_ms=5000,
            max_retries=2,
            required=False,  # Optional step - conversation can succeed without KG updates
        )
        self.updater = None

    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """Update knowledge graph."""
        inference_data = context.get_data("inference")
        inference_result = inference_data["inference_result"]

        # Lazy initialization of updater
        if self.updater is None:
            self.updater = KnowledgeGraphUpdater()
            await self.updater.start()

        try:
            # Update knowledge graph
            update_result = await self.updater.update_from_inference(
                inference_result=inference_result,
                agent_id=f"orchestrated_{context.conversation_id}",
                conversation_id=context.conversation_id,
                trace_id=context.trace_id,
                force_immediate=True,
            )

            return {
                "update_result": update_result,
                "updater_metrics": self.updater.get_metrics(),
            }

        except Exception as e:
            # For optional steps, we log the error but don't fail the pipeline
            logger.warning(f"Knowledge graph update failed: {e}")
            raise

    async def cleanup(self, context: PipelineContext, result: StepResult) -> None:
        """Cleanup updater resources."""
        if self.updater:
            try:
                await self.updater.stop()
            except Exception as e:
                logger.warning(f"Error stopping knowledge graph updater: {e}")


class ResponseGenerationStep(PipelineStep):
    """Generate final response for user using the ResponseGenerator system."""

    def __init__(self, response_generator=None):
        super().__init__(
            name="response_generation",
            timeout_ms=5000,  # Increased timeout for enhanced response generation
            max_retries=2,
            required=True,
        )
        # Lazy import to avoid circular dependencies
        self.response_generator = response_generator

    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """Generate response using the production ResponseGenerator."""
        inference_data = context.get_data("inference")
        inference_result = inference_data["inference_result"]

        request_data = context.get_data("request")
        original_prompt = request_data["prompt"]

        try:
            # Initialize response generator if not provided
            if self.response_generator is None:
                from response_generation import ProductionResponseGenerator

                self.response_generator = ProductionResponseGenerator(
                    enable_monitoring=True,
                )

            # Create response options from context
            from response_generation import ResponseOptions

            options = ResponseOptions(
                narrative_style=True,
                use_natural_language=True,
                include_technical_details=False,
                include_alternatives=True,
                include_knowledge_graph=True,
                enable_caching=True,
                enable_llm_enhancement=True,
                enable_streaming=False,  # Disable streaming in orchestrator
                trace_id=context.trace_id,
                conversation_id=context.conversation_id,
            )

            # Generate enhanced response
            response_data = await self.response_generator.generate_response(
                inference_result=inference_result,
                original_prompt=original_prompt,
                options=options,
            )

            return {
                "response": response_data.message,
                "response_data": response_data.to_dict(),
                "response_metadata": {
                    "based_on_inference": inference_result is not None,
                    "action": inference_result.action if inference_result else None,
                    "confidence": inference_result.confidence if inference_result else 0.0,
                    "generation_time_ms": response_data.metadata.generation_time_ms,
                    "enhanced": response_data.metadata.nlg_enhanced,
                    "cached": response_data.metadata.cached,
                    "response_type": response_data.response_type.value,
                },
            }

        except Exception as e:
            logger.error(f"Enhanced response generation failed: {e}")

            # Fallback to simple response generation
            if inference_result and inference_result.action is not None:
                action = inference_result.action
                confidence = inference_result.confidence

                response = (
                    f"Based on your prompt '{original_prompt}', the Active Inference agent "
                    f"selected action {action} with confidence {confidence:.2f}. "
                    f"The agent's beliefs were updated through Bayesian inference, "
                    f"demonstrating adaptive decision-making under uncertainty."
                )
            else:
                response = (
                    f"I processed your prompt '{original_prompt}' through an Active Inference "
                    f"framework, but was unable to generate a specific action recommendation. "
                    f"The system successfully created and ran a PyMDP agent for analysis."
                )

            return {
                "response": response,
                "response_metadata": {
                    "based_on_inference": inference_result is not None,
                    "action": inference_result.action if inference_result else None,
                    "confidence": inference_result.confidence if inference_result else 0.0,
                    "fallback_used": True,
                    "error": str(e),
                },
            }


class ConversationOrchestrator:
    """Main orchestrator for conversation flow with comprehensive error handling.

    Coordinates the complete pipeline from user prompt to response generation
    with circuit breakers, fallbacks, monitoring, and production-ready resilience.
    """

    def __init__(
        self,
        enable_monitoring: bool = True,
        enable_health_checks: bool = True,
        prometheus_collector: Optional[PrometheusMetricsCollector] = None,
    ):
        # Core components
        self.pipeline = None
        self._setup_pipeline()

        # Monitoring and health
        self.enable_monitoring = enable_monitoring
        self.metrics_collector = MetricsCollector(prometheus_collector=prometheus_collector)

        if enable_health_checks:
            self.health_checker = HealthChecker()
            self._register_components()
        else:
            self.health_checker = None

        # State
        self.is_running = False

        logger.info("ConversationOrchestrator initialized")

    async def start(self) -> None:
        """Start the orchestrator."""
        if self.is_running:
            return

        self.is_running = True

        if self.health_checker:
            await self.health_checker.start()

        logger.info("ConversationOrchestrator started")

    async def stop(self) -> None:
        """Stop the orchestrator."""
        if not self.is_running:
            return

        self.is_running = False

        if self.health_checker:
            await self.health_checker.stop()

        logger.info("ConversationOrchestrator stopped")

    async def process_conversation(
        self,
        request: OrchestrationRequest,
    ) -> OrchestrationResult:
        """Process a conversation request through the complete pipeline.

        Args:
            request: Orchestration request

        Returns:
            Orchestration result with response and metadata

        Raises:
            OrchestrationError: If processing fails
        """
        if not self.is_running:
            raise OrchestrationError("Orchestrator is not running")

        start_time = time.time()

        # Record execution start
        if self.enable_monitoring:
            self.metrics_collector.record_execution_start(request.conversation_id)

        try:
            # Create pipeline context
            context = PipelineContext(
                trace_id=request.trace_id,
                conversation_id=request.conversation_id,
                user_id=request.user_id,
                metadata={
                    "llm_provider": request.llm_provider,
                    "enable_pymdp": request.enable_pymdp,
                    "enable_knowledge_graph": request.enable_knowledge_graph,
                },
            )

            # Add request data to context
            context.set_data(
                "request",
                {
                    "prompt": request.prompt,
                    "llm_model": request.llm_model,
                    "temperature": request.temperature,
                },
            )

            # Execute pipeline
            pipeline_results = await self.pipeline.execute(context)

            # Extract results
            response_data = context.get_data("response_generation")
            inference_data = context.get_data("inference")
            gmn_data = context.get_data("gmn_parsing")

            execution_time_ms = (time.time() - start_time) * 1000

            # Create result
            result = OrchestrationResult(
                request=request,
                success=pipeline_results["success"],
                response=response_data["response"] if response_data else None,
                gmn_spec=gmn_data.get("pymdp_spec") if gmn_data else None,
                inference_result=inference_data.get("inference_result") if inference_data else None,
                execution_time_ms=execution_time_ms,
                steps_completed=[
                    step["step_name"] for step in pipeline_results["steps"] if step["success"]
                ],
                pipeline_results=pipeline_results,
            )

            # Record success
            if self.enable_monitoring:
                self.metrics_collector.record_execution_end(
                    conversation_id=request.conversation_id,
                    success=True,
                    execution_time_ms=execution_time_ms,
                    retry_count=pipeline_results.get("total_retries", 0),
                )

            logger.info(
                f"Conversation processed successfully in {execution_time_ms:.2f}ms",
                extra={
                    "trace_id": request.trace_id,
                    "conversation_id": request.conversation_id,
                    "steps_completed": len(result.steps_completed),
                },
            )

            return result

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000

            # Handle different error types
            if isinstance(e, (ComponentTimeoutError, ValidationError, PipelineExecutionError)):
                orchestration_error = e
            else:
                # Wrap unexpected errors
                orchestration_error = OrchestrationError(
                    message=f"Unexpected error during conversation processing: {str(e)}",
                    context=create_error_context(
                        trace_id=request.trace_id,
                        conversation_id=request.conversation_id,
                        component="orchestrator",
                        start_time=start_time,
                    ),
                    cause=e,
                )

            # Create error result
            result = OrchestrationResult(
                request=request,
                success=False,
                execution_time_ms=execution_time_ms,
                errors=[orchestration_error.to_dict()],
            )

            # Record failure
            if self.enable_monitoring:
                self.metrics_collector.record_execution_end(
                    conversation_id=request.conversation_id,
                    success=False,
                    execution_time_ms=execution_time_ms,
                    error=orchestration_error,
                )

            logger.error(
                f"Conversation processing failed after {execution_time_ms:.2f}ms",
                exc_info=True,
                extra={
                    "trace_id": request.trace_id,
                    "conversation_id": request.conversation_id,
                    "error_type": type(orchestration_error).__name__,
                },
            )

            # Update component health if applicable
            if self.health_checker and hasattr(orchestration_error, "context"):
                component = orchestration_error.context.component
                if component:
                    self.health_checker.update_component_health(
                        name=component,
                        status=HealthStatus.DEGRADED,
                        message=str(orchestration_error),
                    )

            return result

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        if self.health_checker:
            return self.health_checker.get_overall_health()
        else:
            return {"overall_status": "unknown", "message": "Health checking disabled"}

    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestration metrics."""
        if self.enable_monitoring:
            return self.metrics_collector.get_metrics().to_dict()
        else:
            return {"message": "Monitoring disabled"}

    def _setup_pipeline(self) -> None:
        """Setup the conversation pipeline."""
        steps = [
            LLMGenerationStep(),
            GMNParsingStep(),
            AgentCreationStep(),
            InferenceStep(),
            KnowledgeGraphUpdateStep(),
            ResponseGenerationStep(),
        ]

        self.pipeline = ConversationPipeline(
            steps=steps,
            max_total_retries=15,  # Reasonable budget across all steps
        )

    def _register_components(self) -> None:
        """Register components for health checking."""
        if not self.health_checker:
            return

        components = [
            "llm_provider",
            "gmn_parser",
            "pymdp_factory",
            "inference_engine",
            "knowledge_graph_updater",
        ]

        for component in components:
            self.health_checker.register_component(component)
