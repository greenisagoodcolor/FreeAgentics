"""Performance metrics for agent coordination and coalition operations.

This module provides specialized metrics collection for multi-agent coordination,
coalition formation, and collaborative activities.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from observability.performance_metrics import performance_tracker

logger = logging.getLogger(__name__)

# Try to import monitoring system
try:
    from api.v1.monitoring import record_agent_metric, record_system_metric

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

    # Mock monitoring functions
    async def record_system_metric(metric: str, value: float, metadata: Dict = None):
        logger.debug(f"MOCK Coordination - {metric}: {value}")

    async def record_agent_metric(agent_id: str, metric: str, value: float, metadata: Dict = None):
        logger.debug(f"MOCK Agent {agent_id} Coordination - {metric}: {value}")


@dataclass
class CoordinationEvent:
    """Represents a coordination event between agents."""

    timestamp: datetime
    event_type: str  # 'coalition_formed', 'task_assigned', 'message_sent', etc.
    coordinator_id: str
    participant_ids: List[str]
    duration_ms: float
    success: bool
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CoalitionMetrics:
    """Metrics for a coalition."""

    coalition_id: str
    formation_time: datetime
    member_count: int
    coordination_efficiency: float
    task_completion_rate: float
    communication_overhead: float
    stability_score: float


class CoordinationMetricsCollector:
    """Collects and analyzes coordination performance metrics."""

    def __init__(self):
        self.coordination_events = defaultdict(list)
        self.coalition_metrics = {}
        self.agent_coordination_stats = defaultdict(
            lambda: {
                "coordinations_initiated": 0,
                "coordinations_participated": 0,
                "successful_coordinations": 0,
                "failed_coordinations": 0,
                "total_coordination_time_ms": 0.0,
                "messages_sent": 0,
                "messages_received": 0,
            }
        )

        # Performance thresholds
        self.thresholds = {
            "coordination_time_ms": 50.0,  # Target coordination time
            "coalition_formation_ms": 200.0,  # Target coalition formation time
            "message_latency_ms": 10.0,  # Target message latency
            "coordination_success_rate": 0.8,  # Target success rate
        }

        logger.info("Initialized coordination metrics collector")

    async def record_coordination_start(
        self, coordinator_id: str, participant_ids: List[str], coordination_type: str
    ) -> str:
        """Record the start of a coordination operation.

        Args:
            coordinator_id: ID of the coordinating agent
            participant_ids: IDs of participating agents
            coordination_type: Type of coordination

        Returns:
            Coordination session ID
        """
        session_id = f"{coordinator_id}_{datetime.now().timestamp()}"

        # Record in performance tracker
        if MONITORING_AVAILABLE:
            await record_agent_metric(
                coordinator_id,
                "coordination_started",
                1.0,
                {
                    "type": coordination_type,
                    "participants": len(participant_ids),
                    "session_id": session_id,
                },
            )

        return session_id

    async def record_coordination_end(
        self,
        session_id: str,
        coordinator_id: str,
        participant_ids: List[str],
        coordination_type: str,
        start_time: float,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record the end of a coordination operation.

        Args:
            session_id: Coordination session ID
            coordinator_id: ID of the coordinating agent
            participant_ids: IDs of participating agents
            coordination_type: Type of coordination
            start_time: Start timestamp
            success: Whether coordination succeeded
            metadata: Additional metadata
        """
        duration_ms = (time.time() - start_time) * 1000

        # Create coordination event
        event = CoordinationEvent(
            timestamp=datetime.now(),
            event_type=coordination_type,
            coordinator_id=coordinator_id,
            participant_ids=participant_ids,
            duration_ms=duration_ms,
            success=success,
            metadata=metadata,
        )

        # Store event
        self.coordination_events[coordinator_id].append(event)

        # Update agent statistics
        self.agent_coordination_stats[coordinator_id]["coordinations_initiated"] += 1
        self.agent_coordination_stats[coordinator_id]["total_coordination_time_ms"] += duration_ms

        if success:
            self.agent_coordination_stats[coordinator_id]["successful_coordinations"] += 1
        else:
            self.agent_coordination_stats[coordinator_id]["failed_coordinations"] += 1

        for participant_id in participant_ids:
            self.agent_coordination_stats[participant_id]["coordinations_participated"] += 1

        # Record metrics
        if MONITORING_AVAILABLE:
            await record_agent_metric(
                coordinator_id,
                "coordination_duration_ms",
                duration_ms,
                {
                    "type": coordination_type,
                    "success": success,
                    "participants": len(participant_ids),
                    "session_id": session_id,
                },
            )

            # Check performance threshold
            if duration_ms > self.thresholds["coordination_time_ms"] * 2:
                logger.warning(
                    f"Slow coordination detected: {duration_ms:.2f}ms "
                    f"(threshold: {self.thresholds['coordination_time_ms']}ms)"
                )

        # Record to main performance tracker
        await performance_tracker.record_agent_step(coordinator_id, duration_ms)

    async def record_coalition_formation(
        self,
        coalition_id: str,
        coordinator_id: str,
        member_ids: List[str],
        formation_time_ms: float,
        success: bool,
    ):
        """Record coalition formation metrics.

        Args:
            coalition_id: Unique coalition ID
            coordinator_id: ID of the coordinating agent
            member_ids: IDs of coalition members
            formation_time_ms: Time taken to form coalition
            success: Whether formation succeeded
        """
        if success:
            # Store coalition metrics
            self.coalition_metrics[coalition_id] = CoalitionMetrics(
                coalition_id=coalition_id,
                formation_time=datetime.now(),
                member_count=len(member_ids),
                coordination_efficiency=1.0,  # Initial efficiency
                task_completion_rate=0.0,  # No tasks completed yet
                communication_overhead=0.0,  # No overhead yet
                stability_score=1.0,  # Initial stability
            )

        # Record metrics
        if MONITORING_AVAILABLE:
            await record_system_metric(
                "coalition_formation_time_ms",
                formation_time_ms,
                {
                    "coalition_id": coalition_id,
                    "coordinator": coordinator_id,
                    "member_count": len(member_ids),
                    "success": success,
                },
            )

            if success:
                await record_system_metric(
                    "active_coalitions",
                    len(self.coalition_metrics),
                    {"new_coalition": coalition_id},
                )

    async def record_coalition_dissolution(
        self, coalition_id: str, reason: str, final_metrics: Optional[Dict[str, float]] = None
    ):
        """Record coalition dissolution.

        Args:
            coalition_id: Coalition being dissolved
            reason: Reason for dissolution
            final_metrics: Final performance metrics
        """
        if coalition_id in self.coalition_metrics:
            coalition = self.coalition_metrics[coalition_id]
            lifetime_seconds = (datetime.now() - coalition.formation_time).total_seconds()

            if MONITORING_AVAILABLE:
                await record_system_metric(
                    "coalition_lifetime_seconds",
                    lifetime_seconds,
                    {
                        "coalition_id": coalition_id,
                        "reason": reason,
                        "final_efficiency": coalition.coordination_efficiency,
                        "final_completion_rate": coalition.task_completion_rate,
                    },
                )

            # Remove from active coalitions
            del self.coalition_metrics[coalition_id]

    async def record_inter_agent_message(
        self,
        sender_id: str,
        receiver_id: str,
        message_type: str,
        message_size_bytes: int,
        latency_ms: float,
    ):
        """Record inter-agent communication metrics.

        Args:
            sender_id: Sending agent ID
            receiver_id: Receiving agent ID
            message_type: Type of message
            message_size_bytes: Size of message
            latency_ms: Message latency
        """
        # Update agent statistics
        self.agent_coordination_stats[sender_id]["messages_sent"] += 1
        self.agent_coordination_stats[receiver_id]["messages_received"] += 1

        # Record metrics
        if MONITORING_AVAILABLE:
            await record_system_metric(
                "inter_agent_message_latency_ms",
                latency_ms,
                {
                    "sender": sender_id,
                    "receiver": receiver_id,
                    "type": message_type,
                    "size_bytes": message_size_bytes,
                },
            )

            # Check latency threshold
            if latency_ms > self.thresholds["message_latency_ms"] * 3:
                logger.warning(
                    f"High message latency: {latency_ms:.2f}ms between "
                    f"{sender_id} and {receiver_id}"
                )

    async def update_coalition_efficiency(
        self, coalition_id: str, task_completed: bool, coordination_overhead_ms: float
    ):
        """Update coalition efficiency metrics.

        Args:
            coalition_id: Coalition ID
            task_completed: Whether a task was completed
            coordination_overhead_ms: Overhead time for coordination
        """
        if coalition_id not in self.coalition_metrics:
            return

        coalition = self.coalition_metrics[coalition_id]

        # Update task completion rate
        if task_completed:
            # Simple moving average
            coalition.task_completion_rate = coalition.task_completion_rate * 0.9 + 0.1
        else:
            coalition.task_completion_rate *= 0.95

        # Update communication overhead
        coalition.communication_overhead = (
            coalition.communication_overhead * 0.9 + coordination_overhead_ms * 0.1
        )

        # Update coordination efficiency based on overhead
        if coordination_overhead_ms < self.thresholds["coordination_time_ms"]:
            coalition.coordination_efficiency = min(1.0, coalition.coordination_efficiency * 1.01)
        else:
            coalition.coordination_efficiency *= 0.98

        # Record metrics
        if MONITORING_AVAILABLE:
            await record_system_metric(
                "coalition_efficiency",
                coalition.coordination_efficiency,
                {
                    "coalition_id": coalition_id,
                    "task_completion_rate": coalition.task_completion_rate,
                    "communication_overhead_ms": coalition.communication_overhead,
                },
            )

    def get_coordination_statistics(self, agent_id: str) -> Dict[str, Any]:
        """Get coordination statistics for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Coordination statistics
        """
        stats = self.agent_coordination_stats[agent_id]

        # Calculate derived metrics
        total_coordinations = stats["coordinations_initiated"] + stats["coordinations_participated"]

        success_rate = 0.0
        if stats["coordinations_initiated"] > 0:
            success_rate = stats["successful_coordinations"] / stats["coordinations_initiated"]

        avg_coordination_time = 0.0
        if stats["coordinations_initiated"] > 0:
            avg_coordination_time = (
                stats["total_coordination_time_ms"] / stats["coordinations_initiated"]
            )

        return {
            "agent_id": agent_id,
            "coordinations_initiated": stats["coordinations_initiated"],
            "coordinations_participated": stats["coordinations_participated"],
            "success_rate": success_rate,
            "avg_coordination_time_ms": avg_coordination_time,
            "messages_sent": stats["messages_sent"],
            "messages_received": stats["messages_received"],
            "total_coordinations": total_coordinations,
        }

    def get_coalition_statistics(self) -> Dict[str, Any]:
        """Get overall coalition statistics.

        Returns:
            Coalition statistics
        """
        if not self.coalition_metrics:
            return {
                "active_coalitions": 0,
                "avg_member_count": 0,
                "avg_efficiency": 0.0,
                "avg_task_completion_rate": 0.0,
            }

        member_counts = [c.member_count for c in self.coalition_metrics.values()]
        efficiencies = [c.coordination_efficiency for c in self.coalition_metrics.values()]
        completion_rates = [c.task_completion_rate for c in self.coalition_metrics.values()]

        return {
            "active_coalitions": len(self.coalition_metrics),
            "avg_member_count": np.mean(member_counts),
            "avg_efficiency": np.mean(efficiencies),
            "avg_task_completion_rate": np.mean(completion_rates),
            "total_members": sum(member_counts),
        }

    def get_system_coordination_report(self) -> Dict[str, Any]:
        """Get comprehensive system coordination report.

        Returns:
            System-wide coordination metrics
        """
        # Aggregate agent statistics
        total_coordinations = sum(
            s["coordinations_initiated"] + s["coordinations_participated"]
            for s in self.agent_coordination_stats.values()
        )

        total_messages = sum(s["messages_sent"] for s in self.agent_coordination_stats.values())

        # Calculate system-wide success rate
        total_initiated = sum(
            s["coordinations_initiated"] for s in self.agent_coordination_stats.values()
        )
        total_successful = sum(
            s["successful_coordinations"] for s in self.agent_coordination_stats.values()
        )

        system_success_rate = 0.0
        if total_initiated > 0:
            system_success_rate = total_successful / total_initiated

        return {
            "total_coordinations": total_coordinations,
            "total_messages": total_messages,
            "system_success_rate": system_success_rate,
            "active_agents": len(self.agent_coordination_stats),
            "coalition_stats": self.get_coalition_statistics(),
            "performance_thresholds": self.thresholds,
        }


# Global coordination metrics collector
coordination_metrics = CoordinationMetricsCollector()


# Helper functions for easy integration
async def record_coordination(
    coordinator_id: str,
    participant_ids: List[str],
    coordination_type: str,
    start_time: float,
    success: bool,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Record a coordination operation.

    Args:
        coordinator_id: ID of coordinating agent
        participant_ids: IDs of participating agents
        coordination_type: Type of coordination
        start_time: Start timestamp
        success: Whether coordination succeeded
        metadata: Additional metadata
    """
    session_id = f"{coordinator_id}_{start_time}"
    await coordination_metrics.record_coordination_end(
        session_id,
        coordinator_id,
        participant_ids,
        coordination_type,
        start_time,
        success,
        metadata,
    )


async def record_coalition_event(
    event_type: str,
    coalition_id: str,
    coordinator_id: str,
    member_ids: List[str],
    duration_ms: float,
    success: bool = True,
):
    """Record a coalition-related event.

    Args:
        event_type: Type of event ('formation', 'dissolution', etc.)
        coalition_id: Coalition ID
        coordinator_id: Coordinator agent ID
        member_ids: Member agent IDs
        duration_ms: Duration of operation
        success: Whether operation succeeded
    """
    if event_type == "formation":
        await coordination_metrics.record_coalition_formation(
            coalition_id, coordinator_id, member_ids, duration_ms, success
        )
    elif event_type == "dissolution":
        await coordination_metrics.record_coalition_dissolution(coalition_id, "requested", None)


def get_agent_coordination_stats(agent_id: str) -> Dict[str, Any]:
    """Get coordination statistics for an agent.

    Args:
        agent_id: Agent ID

    Returns:
        Coordination statistics
    """
    return coordination_metrics.get_coordination_statistics(agent_id)


def get_system_coordination_report() -> Dict[str, Any]:
    """Get system-wide coordination report.

    Returns:
        System coordination metrics
    """
    return coordination_metrics.get_system_coordination_report()
