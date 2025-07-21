"""Belief state monitoring for agent observability.

This module provides hooks and utilities for monitoring agent belief states,
tracking belief evolution, and detecting anomalies in belief updates.
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Try to import monitoring system
try:
    from api.v1.monitoring import record_agent_metric

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

    # Mock monitoring function
    async def record_agent_metric(
        agent_id: str,
        metric: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        logger.debug(f"MOCK Belief Metric - Agent {agent_id} - {metric}: {value}")


@dataclass
class BeliefSnapshot:
    """Snapshot of agent beliefs at a point in time."""

    timestamp: datetime
    agent_id: str
    beliefs: Dict[str, Any]
    free_energy: Optional[float] = None
    entropy: Optional[float] = None
    kl_divergence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class BeliefMonitor:
    """Monitor and track agent belief states over time."""

    def __init__(self, agent_id: str, history_size: int = 100):
        """Initialize the belief monitor."""
        self.agent_id = agent_id
        self.history_size = history_size
        self.belief_history: Deque[BeliefSnapshot] = deque(maxlen=history_size)
        self.anomaly_threshold = 2.0  # Standard deviations
        self.monitoring_enabled = True

        # Metrics tracking
        self.total_updates = 0
        self.anomaly_count = 0
        self.last_beliefs: Optional[Union[Dict[str, Any], Any]] = None

        logger.info(f"Initialized belief monitor for agent {agent_id}")

    async def record_belief_update(
        self,
        beliefs: Dict[str, Any],
        free_energy: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> BeliefSnapshot:
        """Record a belief update and compute metrics.

        Args:
            beliefs: Current belief state
            free_energy: Free energy value if available
            metadata: Additional metadata

        Returns:
            BeliefSnapshot with computed metrics
        """
        timestamp = datetime.now()

        # Compute belief metrics
        entropy = self._compute_entropy(beliefs)
        kl_divergence = self._compute_kl_divergence(beliefs, self.last_beliefs)

        # Create snapshot
        snapshot = BeliefSnapshot(
            timestamp=timestamp,
            agent_id=self.agent_id,
            beliefs=beliefs.copy() if isinstance(beliefs, dict) else beliefs,
            free_energy=free_energy,
            entropy=entropy,
            kl_divergence=kl_divergence,
            metadata=metadata,
        )

        # Add to history
        self.belief_history.append(snapshot)
        self.total_updates += 1

        # Check for anomalies
        is_anomaly = await self._check_anomaly(snapshot)
        if is_anomaly:
            self.anomaly_count += 1
            await self._handle_anomaly(snapshot)

        # Record metrics if monitoring enabled
        if self.monitoring_enabled and MONITORING_AVAILABLE:
            await self._record_metrics(snapshot, is_anomaly)

        # Update last beliefs
        self.last_beliefs = beliefs.copy() if isinstance(beliefs, dict) else beliefs

        return snapshot

    def _compute_entropy(self, beliefs: Dict[str, Any]) -> float:
        """Compute entropy of belief distribution.

        Args:
            beliefs: Belief state

        Returns:
            Entropy value
        """
        try:
            # Handle different belief representations
            if isinstance(beliefs, dict):
                # If beliefs contain 'qs' (PyMDP format)
                if "qs" in beliefs and isinstance(beliefs["qs"], list):
                    total_entropy = 0.0
                    for qs in beliefs["qs"]:
                        if isinstance(qs, np.ndarray):
                            # Normalize to ensure valid probability distribution
                            qs_norm = qs / (qs.sum() + 1e-10)
                            # Compute Shannon entropy
                            entropy = -np.sum(qs_norm * np.log(qs_norm + 1e-10))
                            total_entropy += entropy
                    return total_entropy

                # If beliefs are direct probability distributions
                elif all(isinstance(v, (float, int, np.ndarray)) for v in beliefs.values()):
                    values = np.array(list(beliefs.values()))
                    if values.ndim == 1:
                        # Normalize
                        values_norm = values / (values.sum() + 1e-10)
                        # Compute entropy
                        return -np.sum(values_norm * np.log(values_norm + 1e-10))

            # If beliefs are numpy array
            elif isinstance(beliefs, np.ndarray):
                beliefs_norm = beliefs / (beliefs.sum() + 1e-10)
                return -np.sum(beliefs_norm * np.log(beliefs_norm + 1e-10))

            return 0.0

        except Exception as e:
            logger.debug(f"Could not compute entropy: {e}")
            return 0.0

    def _compute_kl_divergence(
        self,
        beliefs_new: Dict[str, Any],
        beliefs_old: Optional[Dict[str, Any]],
    ) -> float:
        """Compute KL divergence between belief updates.

        Args:
            beliefs_new: New belief state
            beliefs_old: Previous belief state

        Returns:
            KL divergence value
        """
        if beliefs_old is None:
            return 0.0

        try:
            # Handle PyMDP format
            if isinstance(beliefs_new, dict) and "qs" in beliefs_new:
                if isinstance(beliefs_old, dict) and "qs" in beliefs_old:
                    total_kl = 0.0
                    for i, (qs_new, qs_old) in enumerate(zip(beliefs_new["qs"], beliefs_old["qs"])):
                        if isinstance(qs_new, np.ndarray) and isinstance(qs_old, np.ndarray):
                            # Normalize distributions
                            p = qs_new / (qs_new.sum() + 1e-10)
                            q = qs_old / (qs_old.sum() + 1e-10)
                            # Compute KL divergence
                            kl = np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))
                            total_kl += kl
                    return total_kl

            # Handle direct numpy arrays
            elif isinstance(beliefs_new, np.ndarray) and isinstance(beliefs_old, np.ndarray):
                p = beliefs_new / (beliefs_new.sum() + 1e-10)
                q = beliefs_old / (beliefs_old.sum() + 1e-10)
                return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))

            return 0.0

        except Exception as e:
            logger.debug(f"Could not compute KL divergence: {e}")
            return 0.0

    async def _check_anomaly(self, snapshot: BeliefSnapshot) -> bool:
        """Check if belief update is anomalous.

        Args:
            snapshot: Current belief snapshot

        Returns:
            True if anomaly detected
        """
        if len(self.belief_history) < 10:
            return False  # Not enough history

        # Get recent KL divergences
        recent_kls = [
            s.kl_divergence for s in list(self.belief_history)[-10:] if s.kl_divergence is not None
        ]

        if not recent_kls or snapshot.kl_divergence is None:
            return False

        # Compute statistics
        mean_kl = np.mean(recent_kls)
        std_kl = np.std(recent_kls)

        # Check if current KL divergence is anomalous
        if std_kl > 0:
            z_score = abs(snapshot.kl_divergence - mean_kl) / std_kl
            return z_score > self.anomaly_threshold

        return False

    async def _handle_anomaly(self, snapshot: BeliefSnapshot) -> None:
        """Handle detected belief anomaly.

        Args:
            snapshot: Anomalous belief snapshot
        """
        logger.warning(
            f"Belief anomaly detected for agent {self.agent_id}: "
            f"KL divergence = {snapshot.kl_divergence:.3f}"
        )

        # Record anomaly event
        if MONITORING_AVAILABLE:
            await record_agent_metric(
                self.agent_id,
                "belief_anomaly",
                1.0,
                {
                    "kl_divergence": snapshot.kl_divergence,
                    "entropy": snapshot.entropy,
                    "timestamp": snapshot.timestamp.isoformat(),
                },
            )

    async def _record_metrics(self, snapshot: BeliefSnapshot, is_anomaly: bool) -> None:
        """Record belief metrics to monitoring system.

        Args:
            snapshot: Belief snapshot
            is_anomaly: Whether this is an anomaly
        """
        try:
            # Record entropy
            if snapshot.entropy is not None:
                await record_agent_metric(
                    self.agent_id,
                    "belief_entropy",
                    snapshot.entropy,
                    {"is_anomaly": is_anomaly},
                )

            # Record KL divergence
            if snapshot.kl_divergence is not None:
                await record_agent_metric(
                    self.agent_id,
                    "belief_kl_divergence",
                    snapshot.kl_divergence,
                    {"is_anomaly": is_anomaly},
                )

            # Record free energy
            if snapshot.free_energy is not None:
                await record_agent_metric(
                    self.agent_id,
                    "belief_free_energy",
                    snapshot.free_energy,
                    {"is_anomaly": is_anomaly},
                )

        except Exception as e:
            logger.error(f"Failed to record belief metrics: {e}")

    def get_belief_history(self, limit: Optional[int] = None) -> List[BeliefSnapshot]:
        """Get belief history.

        Args:
            limit: Maximum number of snapshots to return

        Returns:
            List of belief snapshots
        """
        history = list(self.belief_history)
        if limit:
            history = history[-limit:]
        return history

    def get_belief_statistics(self) -> Dict[str, Any]:
        """Get statistics about belief evolution.

        Returns:
            Dictionary with belief statistics
        """
        if not self.belief_history:
            return {
                "total_updates": 0,
                "anomaly_count": 0,
                "anomaly_rate": 0.0,
            }

        # Compute statistics from history
        entropies = [s.entropy for s in self.belief_history if s.entropy is not None]
        kl_divergences = [
            s.kl_divergence for s in self.belief_history if s.kl_divergence is not None
        ]
        free_energies = [s.free_energy for s in self.belief_history if s.free_energy is not None]

        stats = {
            "total_updates": self.total_updates,
            "anomaly_count": self.anomaly_count,
            "anomaly_rate": (
                self.anomaly_count / self.total_updates if self.total_updates > 0 else 0.0
            ),
            "entropy": {
                "mean": np.mean(entropies) if entropies else 0.0,
                "std": np.std(entropies) if entropies else 0.0,
                "min": np.min(entropies) if entropies else 0.0,
                "max": np.max(entropies) if entropies else 0.0,
            },
            "kl_divergence": {
                "mean": np.mean(kl_divergences) if kl_divergences else 0.0,
                "std": np.std(kl_divergences) if kl_divergences else 0.0,
                "min": np.min(kl_divergences) if kl_divergences else 0.0,
                "max": np.max(kl_divergences) if kl_divergences else 0.0,
            },
            "free_energy": {
                "mean": np.mean(free_energies) if free_energies else 0.0,
                "std": np.std(free_energies) if free_energies else 0.0,
                "min": np.min(free_energies) if free_energies else 0.0,
                "max": np.max(free_energies) if free_energies else 0.0,
            },
        }

        return stats

    def reset(self) -> None:
        """Reset belief monitoring state."""
        self.belief_history.clear()
        self.total_updates = 0
        self.anomaly_count = 0
        self.last_beliefs = None
        logger.info(f"Reset belief monitor for agent {self.agent_id}")


class BeliefMonitoringHooks:
    """Hooks for integrating belief monitoring into agents."""

    def __init__(self) -> None:
        """Initialize the belief monitoring hooks."""
        self.monitors: Dict[str, BeliefMonitor] = {}
        self.enabled = True

    def get_monitor(self, agent_id: str) -> BeliefMonitor:
        """Get or create belief monitor for agent.

        Args:
            agent_id: Agent ID

        Returns:
            BeliefMonitor instance
        """
        if agent_id not in self.monitors:
            self.monitors[agent_id] = BeliefMonitor(agent_id)
        return self.monitors[agent_id]

    async def on_belief_update(
        self,
        agent_id: str,
        beliefs: Dict[str, Any],
        free_energy: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[BeliefSnapshot]:
        """Hook called when agent updates beliefs.

        Args:
            agent_id: Agent ID
            beliefs: Updated beliefs
            free_energy: Free energy if available
            metadata: Additional metadata

        Returns:
            BeliefSnapshot if monitoring enabled
        """
        if not self.enabled:
            return None

        monitor = self.get_monitor(agent_id)
        return await monitor.record_belief_update(beliefs, free_energy, metadata)

    def get_agent_statistics(self, agent_id: str) -> Dict[str, Any]:
        """Get belief statistics for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Statistics dictionary
        """
        if agent_id in self.monitors:
            return self.monitors[agent_id].get_belief_statistics()
        return {"error": f"No monitor found for agent {agent_id}"}

    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get belief statistics for all monitored agents.

        Returns:
            Dictionary mapping agent IDs to their statistics
        """
        return {
            agent_id: monitor.get_belief_statistics() for agent_id, monitor in self.monitors.items()
        }

    def reset_agent_monitor(self, agent_id: str) -> None:
        """Reset monitoring for specific agent.

        Args:
            agent_id: Agent ID
        """
        if agent_id in self.monitors:
            self.monitors[agent_id].reset()

    def reset_all(self) -> None:
        """Reset all belief monitors."""
        for monitor in self.monitors.values():
            monitor.reset()
        logger.info("Reset all belief monitors")


# Global instance for belief monitoring hooks
belief_monitoring_hooks = BeliefMonitoringHooks()


# Helper functions for easy integration
async def monitor_belief_update(
    agent_id: str,
    beliefs: Dict[str, Any],
    free_energy: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[BeliefSnapshot]:
    """Monitor an agent belief update.

    Args:
        agent_id: Agent ID
        beliefs: Updated beliefs
        free_energy: Free energy if available
        metadata: Additional metadata

    Returns:
        BeliefSnapshot if monitoring enabled
    """
    return await belief_monitoring_hooks.on_belief_update(agent_id, beliefs, free_energy, metadata)


def get_belief_statistics(agent_id: str) -> Dict[str, Any]:
    """Get belief statistics for an agent.

    Args:
        agent_id: Agent ID

    Returns:
        Statistics dictionary
    """
    return belief_monitoring_hooks.get_agent_statistics(agent_id)


def get_all_belief_statistics() -> Dict[str, Dict[str, Any]]:
    """Get belief statistics for all agents.

    Returns:
        Dictionary mapping agent IDs to their statistics
    """
    return belief_monitoring_hooks.get_all_statistics()
