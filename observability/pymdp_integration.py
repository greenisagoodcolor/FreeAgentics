"""PyMDP observability integration for Active Inference monitoring.

Integrates PyMDP inference monitoring with the observability framework,
tracking belief updates, free energy, inference speed, and agent lifecycle.
"""

import asyncio
import logging
import time
from datetime import datetime
from functools import wraps
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from api.v1.monitoring import record_agent_metric, record_system_metric

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

    # Mock monitoring functions
    async def record_agent_metric(agent_id: str, metric: str, value: float, metadata: Dict = None):
        logger.debug(f"MOCK Agent {agent_id} - {metric}: {value}")

    async def record_system_metric(metric: str, value: float, metadata: Dict = None):
        logger.debug(f"MOCK System - {metric}: {value}")


class PyMDPObservabilityIntegrator:
    """Integrates PyMDP Active Inference with observability monitoring."""

    def __init__(self):
        self.inference_metrics = {}
        self.belief_update_history = {}
        self.free_energy_history = {}
        self.agent_lifecycles = {}
        self.performance_baselines = {
            "inference_time_ms": 5.0,  # 5ms baseline
            "belief_entropy": 1.0,  # Reasonable entropy baseline
            "free_energy": 10.0,  # Free energy baseline
        }

    def monitor_inference_performance(self, agent_id: str):
        """Decorator to monitor PyMDP inference performance."""

        def decorator(inference_func):
            @wraps(inference_func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()

                try:
                    # Execute inference
                    result = (
                        await inference_func(*args, **kwargs)
                        if asyncio.iscoroutinefunction(inference_func)
                        else inference_func(*args, **kwargs)
                    )

                    # Calculate inference time
                    inference_time = (time.time() - start_time) * 1000  # Convert to ms

                    # Record performance metrics
                    await self.record_inference_metrics(
                        agent_id,
                        {
                            "inference_time_ms": inference_time,
                            "success": True,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )

                    # Check for performance degradation
                    if (
                        inference_time > self.performance_baselines["inference_time_ms"] * 5
                    ):  # 5x baseline
                        await record_system_metric(
                            "pymdp_performance_degradation",
                            1.0,
                            {
                                "agent_id": agent_id,
                                "inference_time_ms": inference_time,
                                "baseline_ms": self.performance_baselines["inference_time_ms"],
                            },
                        )
                        logger.warning(
                            f"⚠️ Performance degradation detected for agent {agent_id}: {inference_time:.2f}ms"
                        )

                    return result

                except Exception as e:
                    # Record failure metrics
                    inference_time = (time.time() - start_time) * 1000
                    await self.record_inference_metrics(
                        agent_id,
                        {
                            "inference_time_ms": inference_time,
                            "success": False,
                            "error": str(e),
                            "timestamp": datetime.now().isoformat(),
                        },
                    )
                    raise

            return wrapper

        return decorator

    async def record_inference_metrics(self, agent_id: str, metrics: Dict[str, Any]):
        """Record PyMDP inference metrics."""
        if agent_id not in self.inference_metrics:
            self.inference_metrics[agent_id] = []

        self.inference_metrics[agent_id].append(metrics)

        # Record to monitoring system
        await record_agent_metric(
            agent_id,
            "pymdp_inference_time_ms",
            metrics["inference_time_ms"],
            {"success": metrics["success"]},
        )

        if metrics["success"]:
            await record_agent_metric(agent_id, "pymdp_inference_success", 1.0)
        else:
            await record_agent_metric(agent_id, "pymdp_inference_failure", 1.0)
            await record_agent_metric(
                agent_id, "pymdp_inference_error", 1.0, {"error": metrics.get("error", "unknown")}
            )

    async def monitor_belief_update(
        self, agent_id: str, beliefs_before: Dict, beliefs_after: Dict, free_energy: float = None
    ):
        """Monitor PyMDP belief state updates."""
        try:
            # Calculate belief change magnitude
            belief_change = self.calculate_belief_change(beliefs_before, beliefs_after)

            # Calculate belief entropy (if possible)
            belief_entropy = self.calculate_belief_entropy(beliefs_after)

            # Store in history
            if agent_id not in self.belief_update_history:
                self.belief_update_history[agent_id] = []

            belief_update = {
                "timestamp": datetime.now().isoformat(),
                "belief_change_magnitude": belief_change,
                "belief_entropy": belief_entropy,
                "free_energy": free_energy,
                "beliefs_size": len(beliefs_after) if isinstance(beliefs_after, dict) else 0,
            }

            self.belief_update_history[agent_id].append(belief_update)

            # Record metrics
            await record_agent_metric(agent_id, "pymdp_belief_change", belief_change)

            if belief_entropy is not None:
                await record_agent_metric(agent_id, "pymdp_belief_entropy", belief_entropy)

            if free_energy is not None:
                await record_agent_metric(agent_id, "pymdp_free_energy", free_energy)

                # Store free energy history
                if agent_id not in self.free_energy_history:
                    self.free_energy_history[agent_id] = []
                self.free_energy_history[agent_id].append(
                    {"timestamp": datetime.now().isoformat(), "free_energy": free_energy}
                )

                # Check for free energy anomalies
                if len(self.free_energy_history[agent_id]) > 10:
                    recent_fe = [
                        fe["free_energy"] for fe in self.free_energy_history[agent_id][-10:]
                    ]
                    avg_fe = sum(recent_fe) / len(recent_fe)

                    if free_energy > avg_fe * 2:  # 2x average
                        await record_system_metric(
                            "pymdp_free_energy_spike",
                            1.0,
                            {"agent_id": agent_id, "current_fe": free_energy, "average_fe": avg_fe},
                        )
                        logger.warning(
                            f"⚠️ Free energy spike detected for agent {agent_id}: {free_energy:.2f} (avg: {avg_fe:.2f})"
                        )

        except Exception as e:
            logger.error(f"Failed to monitor belief update for agent {agent_id}: {e}")

    def calculate_belief_change(self, beliefs_before: Dict, beliefs_after: Dict) -> float:
        """Calculate magnitude of belief state change."""
        try:
            if not beliefs_before or not beliefs_after:
                return 1.0  # Maximum change if missing beliefs

            # Simple magnitude calculation
            total_change = 0.0
            keys_union = set(beliefs_before.keys()) | set(beliefs_after.keys())

            for key in keys_union:
                before_val = beliefs_before.get(key, 0)
                after_val = beliefs_after.get(key, 0)

                if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
                    total_change += abs(after_val - before_val)
                elif before_val != after_val:
                    total_change += 1.0  # Discrete change

            return total_change / len(keys_union) if keys_union else 0.0

        except Exception as e:
            logger.warning(f"Failed to calculate belief change: {e}")
            return 0.0

    def calculate_belief_entropy(self, beliefs: Dict) -> Optional[float]:
        """Calculate belief state entropy."""
        try:
            import math

            if not beliefs:
                return None

            # Simple entropy calculation for belief values
            entropy = 0.0
            total_values = 0

            for key, value in beliefs.items():
                if isinstance(value, (int, float)) and value > 0:
                    prob = abs(value)  # Use absolute value as probability
                    total_values += prob

            if total_values == 0:
                return 0.0

            # Normalize and calculate entropy
            for key, value in beliefs.items():
                if isinstance(value, (int, float)) and value > 0:
                    prob = abs(value) / total_values
                    if prob > 0:
                        entropy -= prob * math.log2(prob)

            return entropy

        except Exception as e:
            logger.warning(f"Failed to calculate belief entropy: {e}")
            return None

    async def monitor_agent_lifecycle(self, agent_id: str, event: str, metadata: Dict = None):
        """Monitor agent lifecycle events."""
        try:
            timestamp = datetime.now().isoformat()

            if agent_id not in self.agent_lifecycles:
                self.agent_lifecycles[agent_id] = []

            lifecycle_event = {"timestamp": timestamp, "event": event, "metadata": metadata or {}}

            self.agent_lifecycles[agent_id].append(lifecycle_event)

            # Record lifecycle metrics
            await record_agent_metric(agent_id, f"lifecycle_{event}", 1.0, metadata)

            # Special handling for creation/termination
            if event == "created":
                await record_system_metric("agents_created_total", 1.0, {"agent_id": agent_id})
            elif event == "terminated":
                await record_system_metric("agents_terminated_total", 1.0, {"agent_id": agent_id})

                # Calculate agent lifespan
                if agent_id in self.agent_lifecycles:
                    creation_events = [
                        e for e in self.agent_lifecycles[agent_id] if e["event"] == "created"
                    ]
                    if creation_events:
                        creation_time = datetime.fromisoformat(creation_events[0]["timestamp"])
                        termination_time = datetime.fromisoformat(timestamp)
                        lifespan_seconds = (termination_time - creation_time).total_seconds()

                        await record_agent_metric(
                            agent_id, "agent_lifespan_seconds", lifespan_seconds
                        )
                        await record_system_metric("agent_avg_lifespan_seconds", lifespan_seconds)

        except Exception as e:
            logger.error(f"Failed to monitor agent lifecycle for {agent_id}: {e}")

    async def monitor_multi_agent_coordination(
        self, coordination_event: str, participants: List[str], metrics: Dict = None
    ):
        """Monitor multi-agent coordination events."""
        try:
            timestamp = datetime.now().isoformat()

            # Record coordination metrics
            await record_system_metric(
                f"coordination_{coordination_event}",
                1.0,
                {
                    "participants": participants,
                    "participant_count": len(participants),
                    "timestamp": timestamp,
                    "metrics": metrics or {},
                },
            )

            # Record per-agent participation
            for agent_id in participants:
                await record_agent_metric(
                    agent_id,
                    "coordination_participation",
                    1.0,
                    {"event": coordination_event, "participant_count": len(participants)},
                )

        except Exception as e:
            logger.error(f"Failed to monitor coordination event {coordination_event}: {e}")

    async def get_performance_summary(self, agent_id: str = None) -> Dict[str, Any]:
        """Get performance summary for agent or system."""
        try:
            if agent_id:
                # Agent-specific summary
                summary = {
                    "agent_id": agent_id,
                    "inference_count": len(self.inference_metrics.get(agent_id, [])),
                    "belief_updates": len(self.belief_update_history.get(agent_id, [])),
                    "lifecycle_events": len(self.agent_lifecycles.get(agent_id, [])),
                }

                # Calculate averages
                if agent_id in self.inference_metrics:
                    inference_times = [
                        m["inference_time_ms"]
                        for m in self.inference_metrics[agent_id]
                        if m["success"]
                    ]
                    if inference_times:
                        summary["avg_inference_time_ms"] = sum(inference_times) / len(
                            inference_times
                        )
                        summary["max_inference_time_ms"] = max(inference_times)
                        summary["min_inference_time_ms"] = min(inference_times)

                if agent_id in self.free_energy_history:
                    free_energies = [fe["free_energy"] for fe in self.free_energy_history[agent_id]]
                    if free_energies:
                        summary["avg_free_energy"] = sum(free_energies) / len(free_energies)
                        summary["latest_free_energy"] = free_energies[-1]

                return summary
            else:
                # System-wide summary
                total_agents = len(self.agent_lifecycles)
                total_inferences = sum(len(metrics) for metrics in self.inference_metrics.values())
                total_belief_updates = sum(
                    len(history) for history in self.belief_update_history.values()
                )

                return {
                    "total_agents": total_agents,
                    "total_inferences": total_inferences,
                    "total_belief_updates": total_belief_updates,
                    "active_agents": len(self.inference_metrics),
                    "timestamp": datetime.now().isoformat(),
                }

        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {"error": str(e)}


# Global PyMDP observability integrator
pymdp_observer = PyMDPObservabilityIntegrator()


def monitor_pymdp_inference(agent_id: str):
    """Decorator for monitoring PyMDP inference functions."""
    return pymdp_observer.monitor_inference_performance(agent_id)


async def record_belief_update(
    agent_id: str, beliefs_before: Dict, beliefs_after: Dict, free_energy: float = None
):
    """Record a PyMDP belief update event."""
    await pymdp_observer.monitor_belief_update(agent_id, beliefs_before, beliefs_after, free_energy)


async def record_agent_lifecycle_event(agent_id: str, event: str, metadata: Dict = None):
    """Record an agent lifecycle event."""
    await pymdp_observer.monitor_agent_lifecycle(agent_id, event, metadata)


async def record_coordination_event(event: str, participants: List[str], metrics: Dict = None):
    """Record a multi-agent coordination event."""
    await pymdp_observer.monitor_multi_agent_coordination(event, participants, metrics)


async def get_pymdp_performance_summary(agent_id: str = None) -> Dict[str, Any]:
    """Get PyMDP performance summary."""
    return await pymdp_observer.get_performance_summary(agent_id)
