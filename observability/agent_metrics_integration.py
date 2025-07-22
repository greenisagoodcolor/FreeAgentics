"""Integration of performance metrics with agent inference operations.

This module provides decorators and utilities to automatically collect
performance metrics during agent inference operations without modifying
the core agent logic.
"""

import asyncio
import functools
import logging
import time
from typing import Any, Callable, Dict, Optional

from observability.performance_metrics import (
    record_belief_metric,
    record_inference_metric,
    record_step_metric,
)

logger = logging.getLogger(__name__)


def measure_inference_time(agent_method: Callable) -> Callable:
    """Decorator to measure and record inference time for agent methods.

    This decorator:
    1. Times the execution of the method
    2. Records the timing metrics
    3. Handles both sync and async methods
    4. Preserves the original method's return value

    Args:
        agent_method: The agent method to measure

    Returns:
        Wrapped method that records metrics
    """

    @functools.wraps(agent_method)
    def sync_wrapper(self, *args, **kwargs):
        """Wrapper for synchronous methods."""
        start_time = time.time()
        error_occurred = False
        error_message = None

        try:
            # Execute the original method
            result = agent_method(self, *args, **kwargs)
            return result

        except Exception as e:
            error_occurred = True
            error_message = str(e)
            raise

        finally:
            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            # Get agent_id from self
            agent_id = getattr(self, "agent_id", "unknown")

            # Record metric asynchronously
            try:
                # Create task to record metric without blocking
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(
                        record_inference_metric(
                            agent_id=agent_id,
                            inference_time_ms=execution_time_ms,
                            success=not error_occurred,
                            error=error_message,
                        )
                    )
                else:
                    # If no event loop, just log the metric
                    logger.debug(
                        f"Inference metric - Agent: {agent_id}, "
                        f"Time: {execution_time_ms:.2f}ms, "
                        f"Success: {not error_occurred}"
                    )
            except Exception as e:
                logger.warning(f"Failed to record inference metric: {e}")

    @functools.wraps(agent_method)
    async def async_wrapper(self, *args, **kwargs):
        """Wrapper for asynchronous methods."""
        start_time = time.time()
        error_occurred = False
        error_message = None

        try:
            # Execute the original method
            result = await agent_method(self, *args, **kwargs)
            return result

        except Exception as e:
            error_occurred = True
            error_message = str(e)
            raise

        finally:
            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            # Get agent_id from self
            agent_id = getattr(self, "agent_id", "unknown")

            # Record metric
            try:
                await record_inference_metric(
                    agent_id=agent_id,
                    inference_time_ms=execution_time_ms,
                    success=not error_occurred,
                    error=error_message,
                )
            except Exception as e:
                logger.warning(f"Failed to record inference metric: {e}")

    # Return appropriate wrapper based on method type
    if asyncio.iscoroutinefunction(agent_method):
        return async_wrapper
    else:
        return sync_wrapper


def measure_belief_update(agent_method: Callable) -> Callable:
    """Decorator to measure belief update operations and free energy.

    Args:
        agent_method: The belief update method to measure

    Returns:
        Wrapped method that records belief metrics
    """

    @functools.wraps(agent_method)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()

        # Execute the belief update
        result = agent_method(self, *args, **kwargs)

        # Calculate update time
        update_time_ms = (time.time() - start_time) * 1000

        # Get agent_id
        agent_id = getattr(self, "agent_id", "unknown")

        # Try to get free energy if available
        free_energy = None
        try:
            if hasattr(self, "compute_free_energy"):
                fe_components = self.compute_free_energy()
                if isinstance(fe_components, dict) and "total_free_energy" in fe_components:
                    free_energy = fe_components["total_free_energy"]
        except Exception as e:
            logger.debug(f"Could not compute free energy: {e}")

        # Record metric asynchronously
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(
                    record_belief_metric(
                        agent_id=agent_id,
                        update_time_ms=update_time_ms,
                        free_energy=free_energy,
                    )
                )
            else:
                logger.debug(
                    f"Belief update metric - Agent: {agent_id}, "
                    f"Time: {update_time_ms:.2f}ms, "
                    f"Free Energy: {free_energy}"
                )
        except Exception as e:
            logger.warning(f"Failed to record belief metric: {e}")

        return result

    return wrapper


def measure_agent_step(agent_method: Callable) -> Callable:
    """Decorator to measure complete agent step operations.

    Args:
        agent_method: The step method to measure

    Returns:
        Wrapped method that records step metrics
    """

    @functools.wraps(agent_method)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()

        # Execute the step
        result = agent_method(self, *args, **kwargs)

        # Calculate step time
        step_time_ms = (time.time() - start_time) * 1000

        # Get agent_id
        agent_id = getattr(self, "agent_id", "unknown")

        # Record metric asynchronously
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(
                    record_step_metric(agent_id=agent_id, step_time_ms=step_time_ms)
                )
            else:
                logger.debug(f"Step metric - Agent: {agent_id}, Time: {step_time_ms:.2f}ms")
        except Exception as e:
            logger.warning(f"Failed to record step metric: {e}")

        return result

    return wrapper


class MetricsContext:
    """Context manager for recording metrics during a code block."""

    def __init__(self, agent_id: str, operation: str):
        self.agent_id = agent_id
        self.operation = operation
        self.start_time = None
        self.success = True
        self.error = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Calculate execution time
        execution_time_ms = (time.time() - self.start_time) * 1000

        # Check if error occurred
        if exc_type is not None:
            self.success = False
            self.error = str(exc_val)

        # Record appropriate metric
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                if self.operation == "inference":
                    asyncio.create_task(
                        record_inference_metric(
                            agent_id=self.agent_id,
                            inference_time_ms=execution_time_ms,
                            success=self.success,
                            error=self.error,
                        )
                    )
                elif self.operation == "belief_update":
                    asyncio.create_task(
                        record_belief_metric(
                            agent_id=self.agent_id,
                            update_time_ms=execution_time_ms,
                        )
                    )
                elif self.operation == "step":
                    asyncio.create_task(
                        record_step_metric(
                            agent_id=self.agent_id,
                            step_time_ms=execution_time_ms,
                        )
                    )
            else:
                logger.debug(
                    f"Metrics - Agent: {self.agent_id}, "
                    f"Operation: {self.operation}, "
                    f"Time: {execution_time_ms:.2f}ms, "
                    f"Success: {self.success}"
                )
        except Exception as e:
            logger.warning(f"Failed to record metric: {e}")

        # Don't suppress the exception
        return False


def integrate_metrics_with_agent(agent_class):
    """Class decorator to automatically integrate metrics with an agent class.

    This decorator:
    1. Wraps key agent methods with metrics collection
    2. Preserves original functionality
    3. Adds no overhead when metrics are disabled

    Args:
        agent_class: The agent class to enhance with metrics

    Returns:
        Enhanced agent class
    """
    # Methods to instrument
    methods_to_instrument = {
        "step": measure_agent_step,
        "_step_implementation": measure_inference_time,
        "update_beliefs": measure_belief_update,
        "select_action": measure_inference_time,
        "perceive": measure_inference_time,
    }

    # Apply decorators to existing methods
    for method_name, decorator in methods_to_instrument.items():
        if hasattr(agent_class, method_name):
            original_method = getattr(agent_class, method_name)
            wrapped_method = decorator(original_method)
            setattr(agent_class, method_name, wrapped_method)
            logger.debug(f"Instrumented {agent_class.__name__}.{method_name} with metrics")

    return agent_class


# Helper functions for manual metric recording
async def record_custom_agent_metric(
    agent_id: str,
    metric_name: str,
    value: float,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Record a custom agent metric.

    Args:
        agent_id: ID of the agent
        metric_name: Name of the metric
        value: Metric value
        metadata: Optional metadata
    """
    try:
        from api.v1.monitoring import record_agent_metric

        await record_agent_metric(
            agent_id=agent_id,
            metric=f"custom_{metric_name}",
            value=value,
            metadata=metadata,
        )
    except Exception as e:
        logger.warning(f"Failed to record custom metric: {e}")


def get_metrics_summary(agent) -> Dict[str, Any]:
    """Get a summary of metrics for an agent.

    Args:
        agent: The agent instance

    Returns:
        Dictionary with metrics summary
    """
    try:
        from observability.performance_metrics import performance_tracker

        agent_id = getattr(agent, "agent_id", None)
        if not agent_id:
            return {"error": "No agent_id found"}

        # Get agent metrics from tracker
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Return a future that can be awaited
            future = asyncio.create_task(
                performance_tracker.get_agent_performance_summary(agent_id)
            )
            return {"pending": True, "future": future}
        else:
            # Return basic metrics from agent
            return {
                "agent_id": agent_id,
                "total_steps": getattr(agent, "total_steps", 0),
                "metrics": getattr(agent, "metrics", {}),
            }
    except Exception as e:
        logger.error(f"Failed to get metrics summary: {e}")
        return {"error": str(e)}
