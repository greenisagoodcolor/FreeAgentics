"""Conversation Pipeline Architecture.

Implements the Pipeline pattern for conversation orchestration with proper
error boundaries, step isolation, and recovery mechanisms. Each step is
idempotent and can be retried independently.
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from .errors import (
    PipelineExecutionError,
    ComponentTimeoutError,
    create_error_context,
    is_retryable_error,
    get_retry_delay,
)

logger = logging.getLogger(__name__)


class StepStatus(Enum):
    """Status of a pipeline step."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class StepResult:
    """Result of executing a pipeline step."""

    step_name: str
    status: StepStatus
    output: Any = None
    error: Optional[Exception] = None
    execution_time_ms: Optional[float] = None
    attempt_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Check if step completed successfully."""
        return self.status == StepStatus.COMPLETED

    @property
    def failed(self) -> bool:
        """Check if step failed."""
        return self.status == StepStatus.FAILED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "step_name": self.step_name,
            "status": self.status.value,
            "success": self.success,
            "execution_time_ms": self.execution_time_ms,
            "attempt_count": self.attempt_count,
            "metadata": self.metadata,
            "error": str(self.error) if self.error else None,
        }


@dataclass
class PipelineContext:
    """Context shared across pipeline steps."""

    trace_id: str
    conversation_id: str
    user_id: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Accumulated data from steps
    data: Dict[str, Any] = field(default_factory=dict)

    def set_data(self, key: str, value: Any) -> None:
        """Set data from a step."""
        self.data[key] = value

    def get_data(self, key: str, default: Any = None) -> Any:
        """Get data from a previous step."""
        return self.data.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trace_id": self.trace_id,
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "started_at": self.started_at.isoformat(),
            "metadata": self.metadata,
            "data": self.data,
        }


class PipelineStep(ABC):
    """Abstract base class for pipeline steps.

    Each step should be idempotent and handle its own error recovery.
    Steps can access shared context and contribute data for subsequent steps.
    """

    def __init__(
        self,
        name: str,
        timeout_ms: float = 30000,
        max_retries: int = 3,
        required: bool = True,
    ):
        self.name = name
        self.timeout_ms = timeout_ms
        self.max_retries = max_retries
        self.required = required

    @abstractmethod
    async def execute(self, context: PipelineContext) -> Any:
        """Execute the step logic.

        Args:
            context: Pipeline context with shared data

        Returns:
            Step output data

        Raises:
            Exception if step fails
        """
        pass

    async def validate_preconditions(self, context: PipelineContext) -> bool:
        """Validate that preconditions are met for this step.

        Args:
            context: Pipeline context

        Returns:
            True if preconditions are met
        """
        return True

    async def handle_failure(self, context: PipelineContext, error: Exception) -> bool:
        """Handle step failure and determine if retry should be attempted.

        Args:
            context: Pipeline context
            error: The error that occurred

        Returns:
            True if step should be retried
        """
        return is_retryable_error(error)

    async def cleanup(self, context: PipelineContext, result: StepResult) -> None:
        """Cleanup resources after step execution.

        Args:
            context: Pipeline context
            result: Step execution result
        """
        pass


class ConversationPipeline:
    """Pipeline for orchestrating conversation flow.

    Executes steps in sequence with proper error handling, retry logic,
    and state management. Supports both required and optional steps.
    """

    def __init__(
        self,
        steps: List[PipelineStep],
        max_total_retries: int = 10,
        step_timeout_buffer_ms: float = 5000,
    ):
        self.steps = steps
        self.max_total_retries = max_total_retries
        self.step_timeout_buffer_ms = step_timeout_buffer_ms
        self._execution_history: List[StepResult] = []

    async def execute(self, context: PipelineContext) -> Dict[str, Any]:
        """Execute the complete pipeline.

        Args:
            context: Pipeline context

        Returns:
            Pipeline execution results

        Raises:
            PipelineExecutionError: If pipeline execution fails
        """
        start_time = time.time()
        total_retries = 0

        logger.info(f"Starting pipeline execution for conversation {context.conversation_id}")

        try:
            for step_index, step in enumerate(self.steps):
                # Check retry budget
                if total_retries >= self.max_total_retries:
                    raise PipelineExecutionError(
                        step_name=step.name,
                        step_index=step_index,
                        total_steps=len(self.steps),
                        context=create_error_context(
                            trace_id=context.trace_id,
                            conversation_id=context.conversation_id,
                            step_name=step.name,
                            component="pipeline",
                            start_time=start_time,
                            total_retries=total_retries,
                        ),
                    )

                # Execute step with retries
                step_result = await self._execute_step_with_retries(step, step_index, context)

                total_retries += step_result.attempt_count - 1
                self._execution_history.append(step_result)

                # Handle step failure
                if step_result.failed:
                    if step.required:
                        # Required step failed - abort pipeline
                        raise PipelineExecutionError(
                            step_name=step.name,
                            step_index=step_index,
                            total_steps=len(self.steps),
                            context=create_error_context(
                                trace_id=context.trace_id,
                                conversation_id=context.conversation_id,
                                step_name=step.name,
                                component="pipeline",
                                start_time=start_time,
                            ),
                            cause=step_result.error,
                        )
                    else:
                        # Optional step failed - continue
                        logger.warning(
                            f"Optional step {step.name} failed, continuing pipeline",
                            extra={
                                "trace_id": context.trace_id,
                                "step_name": step.name,
                                "error": str(step_result.error),
                            },
                        )
                        continue

                # Store step output in context
                if step_result.output is not None:
                    context.set_data(step.name, step_result.output)

            # Pipeline completed successfully
            execution_time_ms = (time.time() - start_time) * 1000

            logger.info(
                f"Pipeline completed successfully in {execution_time_ms:.2f}ms",
                extra={
                    "trace_id": context.trace_id,
                    "conversation_id": context.conversation_id,
                    "total_steps": len(self.steps),
                    "total_retries": total_retries,
                },
            )

            return {
                "success": True,
                "execution_time_ms": execution_time_ms,
                "total_retries": total_retries,
                "steps": [result.to_dict() for result in self._execution_history],
                "context": context.to_dict(),
            }

        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000

            logger.error(
                f"Pipeline failed after {execution_time_ms:.2f}ms",
                exc_info=True,
                extra={
                    "trace_id": context.trace_id,
                    "conversation_id": context.conversation_id,
                    "total_retries": total_retries,
                    "steps_completed": len(self._execution_history),
                },
            )

            # Add failed result to history
            self._execution_history.append(
                StepResult(
                    step_name="pipeline",
                    status=StepStatus.FAILED,
                    error=e,
                    execution_time_ms=execution_time_ms,
                    metadata={"total_retries": total_retries},
                )
            )

            raise

    async def _execute_step_with_retries(
        self,
        step: PipelineStep,
        step_index: int,
        context: PipelineContext,
    ) -> StepResult:
        """Execute a single step with retry logic."""
        attempt = 0
        last_error = None

        while attempt < step.max_retries:
            attempt += 1
            step_start_time = time.time()

            try:
                # Check preconditions
                if not await step.validate_preconditions(context):
                    return StepResult(
                        step_name=step.name,
                        status=StepStatus.SKIPPED,
                        execution_time_ms=(time.time() - step_start_time) * 1000,
                        attempt_count=attempt,
                        metadata={"reason": "preconditions_not_met"},
                    )

                # Execute step with timeout
                output = await asyncio.wait_for(
                    step.execute(context),
                    timeout=step.timeout_ms / 1000.0,
                )

                # Step succeeded
                execution_time_ms = (time.time() - step_start_time) * 1000

                result = StepResult(
                    step_name=step.name,
                    status=StepStatus.COMPLETED,
                    output=output,
                    execution_time_ms=execution_time_ms,
                    attempt_count=attempt,
                )

                # Cleanup
                await step.cleanup(context, result)

                logger.debug(
                    f"Step {step.name} completed in {execution_time_ms:.2f}ms",
                    extra={
                        "trace_id": context.trace_id,
                        "step_name": step.name,
                        "attempt": attempt,
                    },
                )

                return result

            except asyncio.TimeoutError:
                execution_time_ms = (time.time() - step_start_time) * 1000
                last_error = ComponentTimeoutError(
                    component=step.name,
                    timeout_ms=step.timeout_ms,
                    context=create_error_context(
                        trace_id=context.trace_id,
                        conversation_id=context.conversation_id,
                        step_name=step.name,
                        component=step.name,
                        start_time=step_start_time,
                    ),
                )

            except Exception as e:
                execution_time_ms = (time.time() - step_start_time) * 1000
                last_error = e

            # Check if we should retry
            should_retry = attempt < step.max_retries and await step.handle_failure(
                context, last_error
            )

            if should_retry:
                delay = get_retry_delay(last_error, attempt - 1)

                logger.warning(
                    f"Step {step.name} failed (attempt {attempt}), retrying in {delay}s",
                    extra={
                        "trace_id": context.trace_id,
                        "step_name": step.name,
                        "attempt": attempt,
                        "error": str(last_error),
                        "retry_delay": delay,
                    },
                )

                await asyncio.sleep(delay)
            else:
                break

        # All retries exhausted
        final_execution_time = (time.time() - step_start_time) * 1000

        result = StepResult(
            step_name=step.name,
            status=StepStatus.FAILED,
            error=last_error,
            execution_time_ms=final_execution_time,
            attempt_count=attempt,
            metadata={"max_retries_reached": True},
        )

        # Cleanup even on failure
        await step.cleanup(context, result)

        logger.error(
            f"Step {step.name} failed after {attempt} attempts",
            extra={
                "trace_id": context.trace_id,
                "step_name": step.name,
                "final_error": str(last_error),
                "attempt_count": attempt,
            },
        )

        return result

    def get_execution_history(self) -> List[StepResult]:
        """Get the execution history for this pipeline run."""
        return self._execution_history.copy()

    def reset(self) -> None:
        """Reset pipeline state for reuse."""
        self._execution_history.clear()


@asynccontextmanager
async def pipeline_execution_context(
    trace_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **metadata,
):
    """Async context manager for pipeline execution."""
    context = PipelineContext(
        trace_id=trace_id or str(uuid.uuid4()),
        conversation_id=conversation_id or str(uuid.uuid4()),
        user_id=user_id or "anonymous",
        metadata=metadata,
    )

    try:
        yield context
    finally:
        # Cleanup context resources if needed
        pass
