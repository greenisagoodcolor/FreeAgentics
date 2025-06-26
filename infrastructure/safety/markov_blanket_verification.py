"""
Markov Blanket Verification Service

This service monitors agent boundaries using the pymdp-based MarkovBlanket interface
from the /agents boundary system (ADR-002). It provides real-time verification of
agent boundary integrity and violation detection using Active Inference principles.

Integration with Active Inference Engine (ADR-005):
- Uses pymdp for mathematical validation of boundary conditions
- Monitors conditional independence: p(μ,η|s,a) = p(μ|s,a)p(η|s,a)
- Tracks free energy and expected free energy from pymdp agents
- Provides real-time violation alerts and mitigation recommendations
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np

from agents.base.data_model import Agent
from agents.base.markov_blanket import (
    AgentState,
    BoundaryMetrics,
    BoundaryState,
    BoundaryViolationEvent,
    MarkovBlanketFactory,
    PyMDPMarkovBlanket,
    ViolationType,
)

logger = logging.getLogger(__name__)


@dataclass
class VerificationConfig:
    """Configuration for Markov blanket verification service"""

    # Monitoring intervals
    verification_interval: float = 1.0  # seconds
    metrics_collection_interval: float = 5.0  # seconds

    # Violation thresholds
    independence_threshold: float = 0.05
    free_energy_threshold: float = 5.0
    stability_threshold: float = 0.7

    # Service limits
    max_agents_monitored: int = 100
    max_violation_history: int = 1000

    # Alert settings
    enable_real_time_alerts: bool = True
    alert_cooldown_seconds: float = 30.0

    # pymdp settings
    num_states: int = 4
    num_observations: int = 4
    num_actions: int = 4


@dataclass
class AgentMonitoringState:
    """State tracking for individual agent monitoring"""

    agent_id: str
    markov_blanket: PyMDPMarkovBlanket
    last_verification: datetime = field(default_factory=datetime.now)
    last_metrics_collection: datetime = field(default_factory=datetime.now)
    last_alert_time: Optional[datetime] = None

    # Violation tracking
    violation_count: int = 0
    recent_violations: List[BoundaryViolationEvent] = field(default_factory=list)

    # Status
    is_active: bool = True
    monitoring_enabled: bool = True


@dataclass
class SystemMetrics:
    """System-wide metrics for boundary verification"""

    total_agents_monitored: int = 0
    active_violations: int = 0
    total_violations_detected: int = 0

    # Performance metrics
    avg_verification_time: float = 0.0
    max_verification_time: float = 0.0
    verification_errors: int = 0

    # Boundary integrity statistics
    avg_boundary_integrity: float = 1.0
    min_boundary_integrity: float = 1.0
    agents_with_compromised_boundaries: int = 0

    last_update: datetime = field(default_factory=datetime.now)


class MarkovBlanketVerificationService:
    """
    Service for monitoring agent boundaries using pymdp-based Markov blankets.

    This service integrates with the Active Inference engine to provide real-time
    monitoring of agent boundary integrity using validated mathematical frameworks.
    """

    def __init__(self, config: Optional[VerificationConfig] = None) -> None:
        """Initialize the verification service"""
        self.config = config or VerificationConfig()

        # Agent monitoring
        self.monitored_agents: Dict[str, AgentMonitoringState] = {}
        self.agent_states: Dict[str, AgentState] = {}
        self.environment_states: Dict[str, np.ndarray] = {}

        # System state
        self.is_running = False
        self.system_metrics = SystemMetrics()

        # Event handlers
        self.violation_handlers: List[Callable[[BoundaryViolationEvent], None]] = []
        self.metrics_handlers: List[Callable[[str, BoundaryMetrics], None]] = []

        # Background tasks
        self._verification_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None

        logger.info("Initialized Markov blanket verification service with pymdp")

    async def start_monitoring(self) -> None:
        """Start the background monitoring service"""
        if self.is_running:
            logger.warning("Verification service is already running")
            return

        self.is_running = True

        # Start background tasks
        self._verification_task = asyncio.create_task(self._verification_loop())
        self._metrics_task = asyncio.create_task(self._metrics_collection_loop())

        logger.info("Started Markov blanket verification service")

    async def stop_monitoring(self) -> None:
        """Stop the background monitoring service"""
        if not self.is_running:
            return

        self.is_running = False

        # Cancel background tasks
        if self._verification_task:
            self._verification_task.cancel()
            try:
                await self._verification_task
            except asyncio.CancelledError:
                pass

        if self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped Markov blanket verification service")

    def register_agent(
        self,
        agent: Agent,
        agent_state: Optional[AgentState] = None,
        environment_state: Optional[np.ndarray] = None,
    ) -> bool:
        """Register an agent for boundary monitoring"""
        try:
            if len(self.monitored_agents) >= self.config.max_agents_monitored:
                logger.error(
                    f"Cannot register agent {agent.agent_id}: " "Maximum agents limit reached"
                )
                return False

            if agent.agent_id in self.monitored_agents:
                logger.warning(f"Agent {agent.agent_id} is already registered")
                return True

            # Create pymdp-based Markov blanket for the agent
            markov_blanket = MarkovBlanketFactory.create_from_agent(agent)

            # Set up violation handler for this agent
            markov_blanket.set_violation_handler(self._handle_violation)

            # Create monitoring state
            monitoring_state = AgentMonitoringState(
                agent_id=agent.agent_id, markov_blanket=markov_blanket
            )

            # Register the agent
            self.monitored_agents[agent.agent_id] = monitoring_state

            # Store initial states if provided
            if agent_state:
                self.agent_states[agent.agent_id] = agent_state
            if environment_state is not None:
                self.environment_states[agent.agent_id] = environment_state

            # Update system metrics
            self.system_metrics.total_agents_monitored = len(self.monitored_agents)

            logger.info(f"Registered agent {agent.agent_id} for boundary monitoring")
            return True

        except Exception as e:
            logger.error(f"Error registering agent {agent.agent_id}: {e}")
            return False

    def unregister_agent(self, agent_id: str) -> bool:
        """Unregister an agent from boundary monitoring"""
        try:
            if agent_id not in self.monitored_agents:
                logger.warning(f"Agent {agent_id} is not registered")
                return False

            # Remove from monitoring
            del self.monitored_agents[agent_id]

            # Clean up stored states
            self.agent_states.pop(agent_id, None)
            self.environment_states.pop(agent_id, None)

            # Update system metrics
            self.system_metrics.total_agents_monitored = len(self.monitored_agents)

            logger.info(f"Unregistered agent {agent_id} from boundary monitoring")
            return True

        except Exception as e:
            logger.error(f"Error unregistering agent {agent_id}: {e}")
            return False

    def update_agent_state(self, agent_id: str, agent_state: AgentState) -> None:
        """Update agent state for boundary monitoring"""
        if agent_id in self.monitored_agents:
            self.agent_states[agent_id] = agent_state

    def update_environment_state(self, agent_id: str, environment_state: np.ndarray) -> None:
        """Update environment state for boundary monitoring"""
        if agent_id in self.monitored_agents:
            self.environment_states[agent_id] = environment_state

    def get_agent_metrics(self, agent_id: str) -> Optional[BoundaryMetrics]:
        """Get current boundary metrics for an agent"""
        if agent_id not in self.monitored_agents:
            return None

        monitoring_state = self.monitored_agents[agent_id]
        return monitoring_state.markov_blanket.get_metrics()

    def get_agent_boundary_state(self, agent_id: str) -> Optional[BoundaryState]:
        """Get current boundary state for an agent"""
        if agent_id not in self.monitored_agents:
            return None

        monitoring_state = self.monitored_agents[agent_id]
        return monitoring_state.markov_blanket.get_boundary_state()

    def get_system_metrics(self) -> SystemMetrics:
        """Get system-wide monitoring metrics"""
        return self.system_metrics

    def get_violation_history(self, agent_id: Optional[str] = None) -> List[BoundaryViolationEvent]:
        """Get violation history for an agent or all agents"""
        if agent_id:
            if agent_id in self.monitored_agents:
                return self.monitored_agents[agent_id].recent_violations.copy()
            return []

        # Return all violations across all agents
        all_violations = []
        for monitoring_state in self.monitored_agents.values():
            all_violations.extend(monitoring_state.recent_violations)

        # Sort by timestamp
        all_violations.sort(key=lambda v: v.timestamp, reverse=True)
        return all_violations

    def add_violation_handler(self, handler: Callable[[BoundaryViolationEvent], None]) -> None:
        """Add a handler for boundary violation events"""
        self.violation_handlers.append(handler)

    def add_metrics_handler(self, handler: Callable[[str, BoundaryMetrics], None]) -> None:
        """Add a handler for metrics updates"""
        self.metrics_handlers.append(handler)

    async def verify_agent_boundary(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Manually trigger boundary verification for a specific agent"""
        if agent_id not in self.monitored_agents:
            return None

        monitoring_state = self.monitored_agents[agent_id]

        try:
            start_time = datetime.now()

            # Update states if available
            if agent_id in self.agent_states and agent_id in self.environment_states:
                monitoring_state.markov_blanket.update_states(
                    self.agent_states[agent_id], self.environment_states[agent_id]
                )

            # Verify independence using pymdp
            independence_measure, evidence = monitoring_state.markov_blanket.verify_independence()

            # Detect violations
            violations = monitoring_state.markov_blanket.detect_violations()

            # Get current metrics
            metrics = monitoring_state.markov_blanket.get_metrics()

            # Update monitoring state
            monitoring_state.last_verification = datetime.now()

            # Calculate verification time
            verification_time = (datetime.now() - start_time).total_seconds()

            # Update system metrics
            self._update_performance_metrics(verification_time)

            return {
                "agent_id": agent_id,
                "independence_measure": independence_measure,
                "evidence": evidence,
                "violations": [v.__dict__ for v in violations],
                "metrics": metrics.__dict__,
                "verification_time": verification_time,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error verifying boundary for agent {agent_id}: {e}")
            self.system_metrics.verification_errors += 1
            return None

    async def _verification_loop(self) -> None:
        """Background loop for continuous boundary verification"""
        while self.is_running:
            try:
                # Verify all active agents
                for agent_id, monitoring_state in self.monitored_agents.items():
                    if not monitoring_state.is_active or not monitoring_state.monitoring_enabled:
                        continue

                    # Check if verification is due
                    time_since_verification = datetime.now() - monitoring_state.last_verification
                    if time_since_verification.total_seconds() < self.config.verification_interval:
                        continue

                    # Perform verification
                    await self.verify_agent_boundary(agent_id)

                # Sleep until next verification cycle
                await asyncio.sleep(self.config.verification_interval)

            except Exception as e:
                logger.error(f"Error in verification loop: {e}")
                await asyncio.sleep(1.0)  # Brief pause on error

    async def _metrics_collection_loop(self) -> None:
        """Background loop for metrics collection"""
        while self.is_running:
            try:
                # Collect metrics from all agents
                for agent_id, monitoring_state in self.monitored_agents.items():
                    if not monitoring_state.is_active:
                        continue

                    # Check if metrics collection is due
                    time_since_collection = (
                        datetime.now() - monitoring_state.last_metrics_collection
                    )
                    if (
                        time_since_collection.total_seconds()
                        < self.config.metrics_collection_interval
                    ):
                        continue

                    # Get current metrics
                    metrics = monitoring_state.markov_blanket.get_metrics()

                    # Update monitoring state
                    monitoring_state.last_metrics_collection = datetime.now()

                    # Trigger metrics handlers
                    for handler in self.metrics_handlers:
                        try:
                            handler(agent_id, metrics)
                        except Exception as e:
                            logger.error(f"Error in metrics handler: {e}")

                # Update system-wide metrics
                self._update_system_metrics()

                # Sleep until next metrics collection
                await asyncio.sleep(self.config.metrics_collection_interval)

            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
                await asyncio.sleep(1.0)  # Brief pause on error

    def _handle_violation(self, violation: BoundaryViolationEvent) -> None:
        """Handle boundary violation events"""
        try:
            agent_id = violation.agent_id

            if agent_id in self.monitored_agents:
                monitoring_state = self.monitored_agents[agent_id]

                # Add to violation history
                monitoring_state.recent_violations.append(violation)
                monitoring_state.violation_count += 1

                # Maintain violation history limit
                if len(monitoring_state.recent_violations) > self.config.max_violation_history:
                    monitoring_state.recent_violations.pop(0)

                # Check alert cooldown
                now = datetime.now()
                if (
                    monitoring_state.last_alert_time is None
                    or (now - monitoring_state.last_alert_time).total_seconds()
                    >= self.config.alert_cooldown_seconds
                ):

                    monitoring_state.last_alert_time = now

                    # Trigger violation handlers
                    for handler in self.violation_handlers:
                        try:
                            handler(violation)
                        except Exception as e:
                            logger.error(f"Error in violation handler: {e}")

                # Update system metrics
                self.system_metrics.total_violations_detected += 1

                logger.warning(
                    f"Boundary violation for agent {agent_id}: "
                    f"{violation.violation_type.value} "
                    f"(severity: {violation.severity:.2f})"
                )

        except Exception as e:
            logger.error(f"Error handling violation: {e}")

    def _update_performance_metrics(self, verification_time: float) -> None:
        """Update performance metrics"""
        # Update verification time statistics
        if verification_time > self.system_metrics.max_verification_time:
            self.system_metrics.max_verification_time = verification_time

        # Update average (simple moving average for now)
        if self.system_metrics.avg_verification_time == 0.0:
            self.system_metrics.avg_verification_time = verification_time
        else:
            alpha = 0.1  # Smoothing factor
            self.system_metrics.avg_verification_time = (
                alpha * verification_time + (1 - alpha) * self.system_metrics.avg_verification_time
            )

    def _update_system_metrics(self) -> None:
        """Update system-wide metrics"""
        try:
            total_agents = len(self.monitored_agents)
            if total_agents == 0:
                return

            # Calculate boundary integrity statistics
            integrity_scores = []
            compromised_count = 0
            active_violations = 0

            for monitoring_state in self.monitored_agents.values():
                if not monitoring_state.is_active:
                    continue

                metrics = monitoring_state.markov_blanket.get_metrics()
                integrity_scores.append(metrics.boundary_integrity)

                boundary_state = monitoring_state.markov_blanket.get_boundary_state()
                if boundary_state in [BoundaryState.COMPROMISED, BoundaryState.VIOLATED]:
                    compromised_count += 1

                # Count recent violations (last 5 minutes)
                recent_threshold = datetime.now() - timedelta(minutes=5)
                recent_violations = [
                    v for v in monitoring_state.recent_violations if v.timestamp > recent_threshold
                ]
                active_violations += len(recent_violations)

            # Update system metrics
            if integrity_scores:
                self.system_metrics.avg_boundary_integrity = np.mean(integrity_scores)
                self.system_metrics.min_boundary_integrity = np.min(integrity_scores)

            self.system_metrics.agents_with_compromised_boundaries = compromised_count
            self.system_metrics.active_violations = active_violations
            self.system_metrics.last_update = datetime.now()

        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")


# Factory functions for easy service creation
def create_verification_service(
    config: Optional[VerificationConfig] = None,
) -> MarkovBlanketVerificationService:
    """Create a Markov blanket verification service"""
    return MarkovBlanketVerificationService(config)


def create_default_config() -> VerificationConfig:
    """Create default verification configuration"""
    return VerificationConfig()


def create_high_frequency_config() -> VerificationConfig:
    """Create configuration for high-frequency monitoring"""
    return VerificationConfig(
        verification_interval=0.5,
        metrics_collection_interval=2.0,
        independence_threshold=0.03,
        free_energy_threshold=3.0,
    )
