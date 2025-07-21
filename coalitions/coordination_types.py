"""Type definitions and helpers for coalition coordination."""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class CoordinationStrategy(Enum):
    """Strategy types for coalition coordination."""

    DIRECT = "direct"  # Direct peer-to-peer coordination
    HIERARCHICAL = "hierarchical"  # Hub-based coordination
    HYBRID = "hybrid"  # Mixed strategy based on group size
    ADAPTIVE = "adaptive"  # Dynamic strategy selection


class CoalitionFormationStrategy(Enum):
    """Strategy types for coalition formation."""

    CENTRALIZED = "centralized"  # Central coordinator assigns agents
    DISTRIBUTED = "distributed"  # Agents self-organize
    AUCTION_BASED = "auction_based"  # Market-based coalition formation
    PREFERENCE_BASED = "preference_based"  # Based on agent preferences


class CoordinationStatus(Enum):
    """Status of coordination tasks."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class CoordinationMessage:
    """Message exchanged during coalition coordination."""

    message_id: str
    sender_id: str
    receiver_id: Optional[str] = None  # None for broadcast
    message_type: str = "coordination"
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: int = 0
    requires_response: bool = False
    response_to: Optional[str] = None  # ID of message being responded to


@dataclass
class CoordinationTask:
    """Task for coordinating a group of agents."""

    task_id: str
    agents: List[Any]
    strategy: CoordinationStrategy = CoordinationStrategy.ADAPTIVE
    priority: int = 0
    status: CoordinationStatus = CoordinationStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    timeout_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    @property
    def duration(self) -> Optional[float]:
        """Get task duration if completed or current duration if in progress."""
        if self.completed_at and self.started_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return time.time() - self.started_at
        return None

    @property
    def is_expired(self) -> bool:
        """Check if task has exceeded timeout."""
        if not self.timeout_seconds or not self.started_at:
            return False
        return (time.time() - self.started_at) > self.timeout_seconds

    def start(self) -> None:
        """Mark task as started."""
        self.started_at = time.time()
        self.status = CoordinationStatus.IN_PROGRESS

    def complete(self) -> None:
        """Mark task as completed."""
        self.completed_at = time.time()
        self.status = CoordinationStatus.COMPLETED

    def fail(self, error: str) -> None:
        """Mark task as failed with error message."""
        self.completed_at = time.time()
        self.status = CoordinationStatus.FAILED
        self.error_message = error


@dataclass
class CoordinationResult:
    """Result of a coordination task."""

    task_id: str
    agent_count: int
    successful_pairs: int = 0
    failed_pairs: int = 0
    coordination_time_ms: float = 0.0
    message_count: int = 0
    strategy_used: CoordinationStrategy = CoordinationStrategy.DIRECT
    efficiency_score: float = 0.0
    error_details: List[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate coordination success rate."""
        total = self.successful_pairs + self.failed_pairs
        return self.successful_pairs / total if total > 0 else 0.0

    @property
    def coordination_efficiency(self) -> float:
        """Calculate coordination efficiency (successes per second per agent)."""
        if self.coordination_time_ms == 0 or self.agent_count == 0:
            return 0.0
        time_seconds = self.coordination_time_ms / 1000.0
        return self.successful_pairs / (time_seconds * self.agent_count)


class CoordinationTypeHelper:
    """Helper for coalition coordination type safety and conversions."""

    @staticmethod
    def ensure_agent_list(
        agents: Union[List[Any], Set[Any], Tuple[Any, ...]],
    ) -> List[Any]:
        """Ensure agents is a proper list."""
        if isinstance(agents, list):
            return agents
        elif isinstance(agents, (set, tuple)):
            return list(agents)
        else:
            raise TypeError(f"Agents must be a collection, got {type(agents)}")

    @staticmethod
    def validate_agent_collection(agents: Any) -> bool:
        """Validate that agents is a valid collection."""
        try:
            iter(agents)
            return len(agents) > 0
        except (TypeError, AttributeError):
            return False

    @staticmethod
    def get_coordination_strategy(agent_count: int) -> CoordinationStrategy:
        """Get recommended coordination strategy based on agent count."""
        if agent_count <= 5:
            return CoordinationStrategy.DIRECT
        elif agent_count <= 15:
            return CoordinationStrategy.HIERARCHICAL
        else:
            return CoordinationStrategy.HYBRID

    @staticmethod
    def create_coordination_task(
        task_id: str,
        agents: Union[List[Any], Set[Any], Tuple[Any, ...]],
        strategy: Optional[CoordinationStrategy] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> CoordinationTask:
        """Create a coordination task with proper type validation."""
        agent_list = CoordinationTypeHelper.ensure_agent_list(agents)

        if not CoordinationTypeHelper.validate_agent_collection(agent_list):
            raise ValueError("Invalid agent collection provided")

        if strategy is None:
            strategy = CoordinationTypeHelper.get_coordination_strategy(len(agent_list))

        return CoordinationTask(
            task_id=task_id,
            agents=agent_list,
            strategy=strategy,
            timeout_seconds=timeout,
            **kwargs,
        )

    @staticmethod
    def combine_coordination_results(
        results: List[CoordinationResult],
    ) -> CoordinationResult:
        """Combine multiple coordination results into one aggregate result."""
        if not results:
            return CoordinationResult("empty", 0)

        combined = CoordinationResult(
            task_id="combined",
            agent_count=sum(r.agent_count for r in results),
            successful_pairs=sum(r.successful_pairs for r in results),
            failed_pairs=sum(r.failed_pairs for r in results),
            coordination_time_ms=max(r.coordination_time_ms for r in results),
            message_count=sum(r.message_count for r in results),
            strategy_used=CoordinationStrategy.HYBRID,  # Mixed strategy
        )

        # Combine error details
        for result in results:
            combined.error_details.extend(result.error_details)

        # Calculate weighted efficiency score
        total_time = sum(r.coordination_time_ms for r in results)
        if total_time > 0:
            weighted_efficiency = sum(
                r.efficiency_score * (r.coordination_time_ms / total_time)
                for r in results
                if r.coordination_time_ms > 0
            )
            combined.efficiency_score = weighted_efficiency

        return combined


# Type aliases for better readability
AgentCollection = Union[List[Any], Set[Any], Tuple[Any, ...]]
CoordinationPair = Tuple[Any, Any]
CoordinationBatch = List[CoordinationTask]
