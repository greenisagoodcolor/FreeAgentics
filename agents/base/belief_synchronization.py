"""
Belief Synchronization Module for FreeAgentics.

This module handles synchronization of beliefs between agents, enabling
coordinated decision-making and shared understanding of the environment.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class SyncMode(Enum):
    """Synchronization modes for belief sharing"""

    FULL = "full"  # Share complete belief state
    PARTIAL = "partial"  # Share only specific beliefs
    INCREMENTAL = "incremental"  # Share only changes since last sync
    HIERARCHICAL = "hierarchical"  # Share based on agent hierarchy


class SyncProtocol(Enum):
    """Protocols for belief synchronization"""

    BROADCAST = "broadcast"  # One-to-many
    CONSENSUS = "consensus"  # Many-to-many with agreement
    MASTER_SLAVE = "master_slave"  # One authoritative source
    PEER_TO_PEER = "peer_to_peer"  # Direct agent-to-agent


@dataclass
class BeliefMessage:
    """Message containing belief information for synchronization"""

    sender_id: str
    receiver_id: Optional[str] = None  # None for broadcast
    belief_type: str = "general"
    belief_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    sync_mode: SyncMode = SyncMode.FULL
    priority: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncResult:
    """Result of a belief synchronization operation"""

    success: bool
    updated_beliefs: Dict[str, Any] = field(default_factory=dict)
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    sync_duration: float = 0.0
    agents_synced: List[str] = field(default_factory=list)


class BeliefConflictResolver:
    """Handles conflicts when beliefs from different agents disagree"""

    def __init__(self, resolution_strategy: str = "weighted_average"):
        self.resolution_strategy = resolution_strategy
        self.resolution_history: List[Dict[str, Any]] = []

    def resolve(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve belief conflicts based on strategy"""
        if not conflicts:
            return {}

        if self.resolution_strategy == "weighted_average":
            return self._weighted_average_resolution(conflicts)
        elif self.resolution_strategy == "majority_vote":
            return self._majority_vote_resolution(conflicts)
        elif self.resolution_strategy == "trust_based":
            return self._trust_based_resolution(conflicts)
        elif self.resolution_strategy == "recency_based":
            return self._recency_based_resolution(conflicts)
        else:
            # Default to most recent
            return conflicts[-1].get("belief_data", {})

    def _weighted_average_resolution(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve by weighted average of numerical beliefs"""
        resolved = {}
        weights = [c.get("weight", 1.0) for c in conflicts]
        total_weight = sum(weights)

        if total_weight == 0:
            return {}

        # Average numerical values
        for conflict in conflicts:
            belief_data = conflict.get("belief_data", {})
            weight = conflict.get("weight", 1.0) / total_weight

            for key, value in belief_data.items():
                if isinstance(value, (int, float)):
                    resolved[key] = resolved.get(key, 0) + value * weight
                elif key not in resolved:
                    resolved[key] = value

        return resolved

    def _majority_vote_resolution(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve by majority vote for discrete beliefs"""
        from collections import Counter

        resolved = {}

        # Count votes for each belief
        for conflict in conflicts:
            belief_data = conflict.get("belief_data", {})
            for key, value in belief_data.items():
                if key not in resolved:
                    resolved[key] = []
                resolved[key].append(value)

        # Select most common value for each belief
        for key, values in resolved.items():
            if isinstance(values[0], (int, float)):
                resolved[key] = np.mean(values)
            else:
                counter = Counter(str(v) for v in values)
                resolved[key] = counter.most_common(1)[0][0]

        return resolved

    def _trust_based_resolution(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve based on trust levels of agents"""
        if not conflicts:
            return {}

        # Sort by trust level (highest first)
        sorted_conflicts = sorted(conflicts, key=lambda x: x.get("trust_level", 0.5), reverse=True)

        # Take beliefs from most trusted agent
        return sorted_conflicts[0].get("belief_data", {})

    def _recency_based_resolution(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve by taking most recent beliefs"""
        if not conflicts:
            return {}

        # Sort by timestamp (most recent first)
        sorted_conflicts = sorted(
            conflicts, key=lambda x: x.get("timestamp", datetime.min), reverse=True
        )

        return sorted_conflicts[0].get("belief_data", {})


class BeliefSynchronizer:
    """Main class for managing belief synchronization between agents"""

    def __init__(
        self,
        agent_id: str,
        sync_protocol: SyncProtocol = SyncProtocol.PEER_TO_PEER,
        conflict_resolver: Optional[BeliefConflictResolver] = None,
    ):
        self.agent_id = agent_id
        self.sync_protocol = sync_protocol
        self.conflict_resolver = conflict_resolver or BeliefConflictResolver()

        # Synchronization state
        self.last_sync_time: Dict[str, datetime] = {}
        self.sync_history: List[SyncResult] = []
        self.pending_messages: List[BeliefMessage] = []
        self.subscribed_agents: Set[str] = set()

        # Callbacks
        self.on_belief_update: Optional[Callable] = None
        self.on_sync_complete: Optional[Callable] = None

        # Performance metrics
        self.sync_count = 0
        self.conflict_count = 0
        self.total_sync_time = 0.0

    async def synchronize(
        self, target_agents: List[str], beliefs: Dict[str, Any], sync_mode: SyncMode = SyncMode.FULL
    ) -> SyncResult:
        """Synchronize beliefs with target agents"""
        start_time = datetime.now()

        # Create belief messages
        messages = []
        for target_id in target_agents:
            message = BeliefMessage(
                sender_id=self.agent_id,
                receiver_id=target_id,
                belief_data=beliefs,
                sync_mode=sync_mode,
            )
            messages.append(message)

        # Send messages based on protocol
        if self.sync_protocol == SyncProtocol.BROADCAST:
            result = await self._broadcast_sync(messages)
        elif self.sync_protocol == SyncProtocol.CONSENSUS:
            result = await self._consensus_sync(messages)
        elif self.sync_protocol == SyncProtocol.MASTER_SLAVE:
            result = await self._master_slave_sync(messages)
        else:  # PEER_TO_PEER
            result = await self._peer_to_peer_sync(messages)

        # Update metrics
        sync_duration = (datetime.now() - start_time).total_seconds()
        result.sync_duration = sync_duration
        self.sync_count += 1
        self.total_sync_time += sync_duration

        # Update last sync times
        for agent_id in target_agents:
            self.last_sync_time[agent_id] = datetime.now()

        # Add to history
        self.sync_history.append(result)

        # Trigger callback
        if self.on_sync_complete:
            self.on_sync_complete(result)

        return result

    async def _broadcast_sync(self, messages: List[BeliefMessage]) -> SyncResult:
        """Broadcast beliefs to all target agents"""
        # In a real implementation, this would send messages through
        # a communication channel. For now, we simulate success.
        return SyncResult(
            success=True, agents_synced=[msg.receiver_id for msg in messages if msg.receiver_id]
        )

    async def _consensus_sync(self, messages: List[BeliefMessage]) -> SyncResult:
        """Achieve consensus among agents before updating beliefs"""
        # Simulate consensus building
        conflicts = []

        # In real implementation, would gather responses from agents
        # and identify conflicts

        if conflicts:
            resolved_beliefs = self.conflict_resolver.resolve(conflicts)
            self.conflict_count += len(conflicts)

            return SyncResult(
                success=True,
                updated_beliefs=resolved_beliefs,
                conflicts=conflicts,
                agents_synced=[msg.receiver_id for msg in messages if msg.receiver_id],
            )

        return SyncResult(
            success=True, agents_synced=[msg.receiver_id for msg in messages if msg.receiver_id]
        )

    async def _master_slave_sync(self, messages: List[BeliefMessage]) -> SyncResult:
        """Sync with master agent as authoritative source"""
        # In master-slave, beliefs flow in one direction
        return SyncResult(
            success=True, agents_synced=[msg.receiver_id for msg in messages if msg.receiver_id]
        )

    async def _peer_to_peer_sync(self, messages: List[BeliefMessage]) -> SyncResult:
        """Direct peer-to-peer synchronization"""
        # Simulate P2P communication
        return SyncResult(
            success=True, agents_synced=[msg.receiver_id for msg in messages if msg.receiver_id]
        )

    def receive_belief_update(self, message: BeliefMessage) -> None:
        """Handle incoming belief update from another agent"""
        self.pending_messages.append(message)

        # Trigger callback if registered
        if self.on_belief_update:
            self.on_belief_update(message)

    async def process_pending_messages(self) -> Dict[str, Any]:
        """Process all pending belief messages"""
        if not self.pending_messages:
            return {}

        # Group messages by sender
        messages_by_sender: Dict[str, List[BeliefMessage]] = {}
        for msg in self.pending_messages:
            if msg.sender_id not in messages_by_sender:
                messages_by_sender[msg.sender_id] = []
            messages_by_sender[msg.sender_id].append(msg)

        # Process and potentially resolve conflicts
        updated_beliefs = {}
        conflicts = []

        for sender_id, messages in messages_by_sender.items():
            # Get latest message from each sender
            latest_msg = max(messages, key=lambda m: m.timestamp)

            for key, value in latest_msg.belief_data.items():
                if key in updated_beliefs and updated_beliefs[key] != value:
                    # Conflict detected
                    conflicts.append(
                        {
                            "key": key,
                            "agents": [self.agent_id, sender_id],
                            "values": [updated_beliefs[key], value],
                            "belief_data": {key: value},
                            "timestamp": latest_msg.timestamp,
                        }
                    )
                else:
                    updated_beliefs[key] = value

        # Resolve conflicts if any
        if conflicts:
            resolved = self.conflict_resolver.resolve(conflicts)
            updated_beliefs.update(resolved)

        # Clear processed messages
        self.pending_messages.clear()

        return updated_beliefs

    def subscribe_to_agent(self, agent_id: str) -> None:
        """Subscribe to belief updates from another agent"""
        self.subscribed_agents.add(agent_id)

    def unsubscribe_from_agent(self, agent_id: str) -> None:
        """Unsubscribe from belief updates from another agent"""
        self.subscribed_agents.discard(agent_id)

    def get_sync_metrics(self) -> Dict[str, Any]:
        """Get synchronization performance metrics"""
        avg_sync_time = self.total_sync_time / self.sync_count if self.sync_count > 0 else 0

        return {
            "total_syncs": self.sync_count,
            "total_conflicts": self.conflict_count,
            "average_sync_time": avg_sync_time,
            "subscribed_agents": len(self.subscribed_agents),
            "pending_messages": len(self.pending_messages),
        }

    def get_sync_history(self, limit: Optional[int] = None) -> List[SyncResult]:
        """Get synchronization history"""
        if limit:
            return self.sync_history[-limit:]
        return self.sync_history.copy()


class HierarchicalBeliefSynchronizer(BeliefSynchronizer):
    """Hierarchical belief synchronization for multi-level agent organizations"""

    def __init__(
        self,
        agent_id: str,
        hierarchy_level: int = 0,
        parent_id: Optional[str] = None,
        conflict_resolver: Optional[BeliefConflictResolver] = None,
    ):
        super().__init__(
            agent_id,
            SyncProtocol.PEER_TO_PEER,  # Hierarchical uses P2P protocol internally
            conflict_resolver,
        )
        self.hierarchy_level = hierarchy_level
        self.parent_id = parent_id
        self.child_ids: Set[str] = set()

    def add_child(self, child_id: str) -> None:
        """Add a child agent in the hierarchy"""
        self.child_ids.add(child_id)

    def remove_child(self, child_id: str) -> None:
        """Remove a child agent from the hierarchy"""
        self.child_ids.discard(child_id)

    async def synchronize_with_parent(
        self, beliefs: Dict[str, Any], sync_mode: SyncMode = SyncMode.INCREMENTAL
    ) -> SyncResult:
        """Synchronize beliefs with parent agent"""
        if not self.parent_id:
            return SyncResult(success=False)

        return await self.synchronize([self.parent_id], beliefs, sync_mode)

    async def synchronize_with_children(
        self, beliefs: Dict[str, Any], sync_mode: SyncMode = SyncMode.INCREMENTAL
    ) -> SyncResult:
        """Synchronize beliefs with child agents"""
        if not self.child_ids:
            return SyncResult(success=True)

        return await self.synchronize(list(self.child_ids), beliefs, sync_mode)


# Utility functions for belief transformation
def transform_belief_format(
    beliefs: Dict[str, Any], source_format: str, target_format: str
) -> Dict[str, Any]:
    """Transform beliefs between different formats"""
    if source_format == target_format:
        return beliefs.copy()

    # Add transformation logic as needed
    transformed = {}

    if source_format == "raw" and target_format == "normalized":
        # Normalize numerical beliefs to [0, 1]
        for key, value in beliefs.items():
            if isinstance(value, (int, float)):
                # Simple min-max normalization (would need actual min/max in practice)
                transformed[key] = max(0.0, min(1.0, value))
            else:
                transformed[key] = value
    else:
        # Default: pass through
        transformed = beliefs.copy()

    return transformed


def merge_beliefs(
    belief_sets: List[Dict[str, Any]], merge_strategy: str = "union"
) -> Dict[str, Any]:
    """Merge multiple belief sets into one"""
    if not belief_sets:
        return {}

    if len(belief_sets) == 1:
        return belief_sets[0].copy()

    merged = {}

    if merge_strategy == "union":
        # Include all beliefs from all sets
        for beliefs in belief_sets:
            merged.update(beliefs)
    elif merge_strategy == "intersection":
        # Include only beliefs present in all sets
        all_keys = set(belief_sets[0].keys())
        for beliefs in belief_sets[1:]:
            all_keys &= set(beliefs.keys())

        for key in all_keys:
            merged[key] = belief_sets[0][key]  # Take from first set
    elif merge_strategy == "weighted":
        # Weighted average for numerical beliefs
        weights = [1.0] * len(belief_sets)  # Equal weights by default
        total_weight = sum(weights)

        for i, beliefs in enumerate(belief_sets):
            weight = weights[i] / total_weight
            for key, value in beliefs.items():
                if isinstance(value, (int, float)):
                    merged[key] = merged.get(key, 0) + value * weight
                elif key not in merged:
                    merged[key] = value

    return merged
