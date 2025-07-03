"""
Comprehensive tests for Belief Synchronization module
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List

import pytest

from agents.base.belief_synchronization import (  # Enums; Classes; Utility functions
    BeliefConflictResolver,
    BeliefMessage,
    BeliefSynchronizer,
    HierarchicalBeliefSynchronizer,
    SyncMode,
    SyncProtocol,
    SyncResult,
    merge_beliefs,
    transform_belief_format,
)


class TestEnums:
    """Test enumeration types"""

    def test_sync_mode_enum(self):
        """Test SyncMode enum values"""
        assert SyncMode.FULL.value == "full"
        assert SyncMode.PARTIAL.value == "partial"
        assert SyncMode.INCREMENTAL.value == "incremental"
        assert SyncMode.HIERARCHICAL.value == "hierarchical"
        assert len(SyncMode) == 4

    def test_sync_protocol_enum(self):
        """Test SyncProtocol enum values"""
        assert SyncProtocol.BROADCAST.value == "broadcast"
        assert SyncProtocol.CONSENSUS.value == "consensus"
        assert SyncProtocol.MASTER_SLAVE.value == "master_slave"
        assert SyncProtocol.PEER_TO_PEER.value == "peer_to_peer"
        assert len(SyncProtocol) == 4


class TestBeliefMessage:
    """Test BeliefMessage dataclass"""

    def test_belief_message_creation(self):
        """Test creating belief messages"""
        msg = BeliefMessage(
            sender_id="agent_1",
            receiver_id="agent_2",
            belief_type="location",
            belief_data={"x": 10, "y": 20},
        )

        assert msg.sender_id == "agent_1"
        assert msg.receiver_id == "agent_2"
        assert msg.belief_type == "location"
        assert msg.belief_data["x"] == 10
        assert msg.sync_mode == SyncMode.FULL
        assert msg.priority == 0.5
        assert isinstance(msg.timestamp, datetime)

    def test_belief_message_defaults(self):
        """Test default values"""
        msg = BeliefMessage(sender_id="agent_1")

        assert msg.receiver_id is None  # Broadcast
        assert msg.belief_type == "general"
        assert msg.belief_data == {}
        assert msg.metadata == {}


class TestSyncResult:
    """Test SyncResult dataclass"""

    def test_sync_result_creation(self):
        """Test creating sync results"""
        result = SyncResult(
            success=True,
            updated_beliefs={"status": "active"},
            conflicts=[{"key": "position", "values": [(0, 0), (1, 1)]}],
            agents_synced=["agent_1", "agent_2"],
        )

        assert result.success is True
        assert result.updated_beliefs["status"] == "active"
        assert len(result.conflicts) == 1
        assert len(result.agents_synced) == 2
        assert isinstance(result.timestamp, datetime)
        assert result.sync_duration == 0.0

    def test_sync_result_defaults(self):
        """Test default values"""
        result = SyncResult(success=False)

        assert result.success is False
        assert result.updated_beliefs == {}
        assert result.conflicts == []
        assert result.agents_synced == []


class TestBeliefConflictResolver:
    """Test BeliefConflictResolver class"""

    def test_resolver_creation(self):
        """Test creating conflict resolver"""
        resolver = BeliefConflictResolver()
        assert resolver.resolution_strategy == "weighted_average"
        assert resolver.resolution_history == []

        resolver2 = BeliefConflictResolver("majority_vote")
        assert resolver2.resolution_strategy == "majority_vote"

    def test_empty_conflicts(self):
        """Test resolving empty conflicts"""
        resolver = BeliefConflictResolver()
        result = resolver.resolve([])
        assert result == {}

    def test_weighted_average_resolution(self):
        """Test weighted average resolution"""
        resolver = BeliefConflictResolver("weighted_average")

        conflicts = [
            {"belief_data": {"temperature": 20.0}, "weight": 1.0},
            {"belief_data": {"temperature": 30.0}, "weight": 2.0},
        ]

        result = resolver.resolve(conflicts)
        # Weighted average: (20*1 + 30*2) / (1+2) = 80/3 ≈ 26.67
        assert abs(result["temperature"] - 26.67) < 0.01

    def test_majority_vote_resolution(self):
        """Test majority vote resolution"""
        resolver = BeliefConflictResolver("majority_vote")

        conflicts = [
            {"belief_data": {"status": "active"}},
            {"belief_data": {"status": "active"}},
            {"belief_data": {"status": "idle"}},
        ]

        result = resolver.resolve(conflicts)
        assert result["status"] == "active"

    def test_trust_based_resolution(self):
        """Test trust-based resolution"""
        resolver = BeliefConflictResolver("trust_based")

        conflicts = [
            {"belief_data": {"location": "A"}, "trust_level": 0.3},
            {"belief_data": {"location": "B"}, "trust_level": 0.8},
            {"belief_data": {"location": "C"}, "trust_level": 0.5},
        ]

        result = resolver.resolve(conflicts)
        assert result["location"] == "B"  # Highest trust

    def test_recency_based_resolution(self):
        """Test recency-based resolution"""
        resolver = BeliefConflictResolver("recency_based")

        now = datetime.now()
        conflicts = [
            {"belief_data": {"value": 1}, "timestamp": now - timedelta(hours=2)},
            {"belief_data": {"value": 2}, "timestamp": now - timedelta(hours=1)},
            {"belief_data": {"value": 3}, "timestamp": now},
        ]

        result = resolver.resolve(conflicts)
        assert result["value"] == 3  # Most recent

    def test_unknown_strategy(self):
        """Test unknown resolution strategy"""
        resolver = BeliefConflictResolver("unknown_strategy")

        conflicts = [{"belief_data": {"value": 1}}, {"belief_data": {"value": 2}}]

        result = resolver.resolve(conflicts)
        assert result["value"] == 2  # Defaults to last


class TestBeliefSynchronizer:
    """Test BeliefSynchronizer class"""

    @pytest.mark.asyncio
    async def test_synchronizer_creation(self):
        """Test creating belief synchronizer"""
        sync = BeliefSynchronizer("agent_1")

        assert sync.agent_id == "agent_1"
        assert sync.sync_protocol == SyncProtocol.PEER_TO_PEER
        assert isinstance(sync.conflict_resolver, BeliefConflictResolver)
        assert sync.last_sync_time == {}
        assert sync.sync_history == []
        assert sync.pending_messages == []
        assert sync.subscribed_agents == set()
        assert sync.sync_count == 0
        assert sync.conflict_count == 0
        assert sync.total_sync_time == 0.0

    @pytest.mark.asyncio
    async def test_synchronize_broadcast(self):
        """Test broadcast synchronization"""
        sync = BeliefSynchronizer("agent_1", SyncProtocol.BROADCAST)

        result = await sync.synchronize(
            target_agents=["agent_2", "agent_3"], beliefs={"status": "active", "energy": 80}
        )

        assert result.success is True
        assert set(result.agents_synced) == {"agent_2", "agent_3"}
        assert sync.sync_count == 1
        assert "agent_2" in sync.last_sync_time
        assert "agent_3" in sync.last_sync_time
        assert len(sync.sync_history) == 1

    @pytest.mark.asyncio
    async def test_synchronize_consensus(self):
        """Test consensus synchronization"""
        sync = BeliefSynchronizer("agent_1", SyncProtocol.CONSENSUS)

        result = await sync.synchronize(
            target_agents=["agent_2", "agent_3"],
            beliefs={"decision": "explore"},
            sync_mode=SyncMode.PARTIAL,
        )

        assert result.success is True
        assert len(result.agents_synced) == 2

    @pytest.mark.asyncio
    async def test_synchronize_master_slave(self):
        """Test master-slave synchronization"""
        sync = BeliefSynchronizer("master", SyncProtocol.MASTER_SLAVE)

        result = await sync.synchronize(
            target_agents=["slave_1", "slave_2"], beliefs={"command": "gather"}
        )

        assert result.success is True
        assert "slave_1" in result.agents_synced
        assert "slave_2" in result.agents_synced

    @pytest.mark.asyncio
    async def test_synchronize_peer_to_peer(self):
        """Test peer-to-peer synchronization"""
        sync = BeliefSynchronizer("agent_1", SyncProtocol.PEER_TO_PEER)

        result = await sync.synchronize(
            target_agents=["agent_2"],
            beliefs={"shared_goal": "build"},
            sync_mode=SyncMode.INCREMENTAL,
        )

        assert result.success is True
        assert result.agents_synced == ["agent_2"]

    def test_receive_belief_update(self):
        """Test receiving belief updates"""
        sync = BeliefSynchronizer("agent_1")

        msg = BeliefMessage(
            sender_id="agent_2", receiver_id="agent_1", belief_data={"resource": "wood"}
        )

        sync.receive_belief_update(msg)

        assert len(sync.pending_messages) == 1
        assert sync.pending_messages[0] == msg

    @pytest.mark.asyncio
    async def test_process_pending_messages(self):
        """Test processing pending messages"""
        sync = BeliefSynchronizer("agent_1")

        # Add multiple messages
        sync.receive_belief_update(BeliefMessage(sender_id="agent_2", belief_data={"energy": 50}))
        sync.receive_belief_update(
            BeliefMessage(sender_id="agent_3", belief_data={"energy": 60, "health": 90})
        )

        updated_beliefs = await sync.process_pending_messages()

        assert "energy" in updated_beliefs
        assert "health" in updated_beliefs
        assert updated_beliefs["health"] == 90
        assert len(sync.pending_messages) == 0

    @pytest.mark.asyncio
    async def test_process_conflicting_messages(self):
        """Test processing messages with conflicts"""
        resolver = BeliefConflictResolver("weighted_average")
        sync = BeliefSynchronizer("agent_1", conflict_resolver=resolver)

        # Add conflicting messages
        sync.receive_belief_update(BeliefMessage(sender_id="agent_2", belief_data={"position": 10}))
        sync.receive_belief_update(BeliefMessage(sender_id="agent_3", belief_data={"position": 20}))

        # Pre-set agent's own belief to create conflict
        sync.pending_messages.insert(
            0, BeliefMessage(sender_id=sync.agent_id, belief_data={"position": 15})
        )

        updated_beliefs = await sync.process_pending_messages()

        # Should resolve to latest or resolved value
        assert "position" in updated_beliefs

    def test_subscription_management(self):
        """Test agent subscription management"""
        sync = BeliefSynchronizer("agent_1")

        sync.subscribe_to_agent("agent_2")
        sync.subscribe_to_agent("agent_3")

        assert "agent_2" in sync.subscribed_agents
        assert "agent_3" in sync.subscribed_agents
        assert len(sync.subscribed_agents) == 2

        sync.unsubscribe_from_agent("agent_2")

        assert "agent_2" not in sync.subscribed_agents
        assert len(sync.subscribed_agents) == 1

    @pytest.mark.asyncio
    async def test_sync_metrics(self):
        """Test synchronization metrics"""
        sync = BeliefSynchronizer("agent_1")

        # Perform some syncs
        await sync.synchronize(["agent_2"], {"test": 1})
        await sync.synchronize(["agent_3"], {"test": 2})

        sync.subscribe_to_agent("agent_2")
        sync.receive_belief_update(BeliefMessage(sender_id="agent_4"))

        metrics = sync.get_sync_metrics()

        assert metrics["total_syncs"] == 2
        assert metrics["total_conflicts"] == 0
        assert metrics["average_sync_time"] >= 0
        assert metrics["subscribed_agents"] == 1
        assert metrics["pending_messages"] == 1

    @pytest.mark.asyncio
    async def test_sync_history(self):
        """Test synchronization history"""
        sync = BeliefSynchronizer("agent_1")

        # Perform multiple syncs
        for i in range(5):
            await sync.synchronize([f"agent_{i}"], {"count": i})

        # Get full history
        history = sync.get_sync_history()
        assert len(history) == 5

        # Get limited history
        limited_history = sync.get_sync_history(limit=3)
        assert len(limited_history) == 3
        assert limited_history[0].agents_synced == ["agent_2"]  # Third sync

    @pytest.mark.asyncio
    async def test_sync_with_callbacks(self):
        """Test synchronization with callbacks"""
        sync = BeliefSynchronizer("agent_1")

        # Track callback calls
        update_calls = []
        complete_calls = []

        def on_update(msg):
            update_calls.append(msg)

        def on_complete(result):
            complete_calls.append(result)

        sync.on_belief_update = on_update
        sync.on_sync_complete = on_complete

        # Trigger callbacks
        msg = BeliefMessage(sender_id="agent_2", belief_data={"test": 1})
        sync.receive_belief_update(msg)

        result = await sync.synchronize(["agent_3"], {"test": 2})

        assert len(update_calls) == 1
        assert update_calls[0] == msg
        assert len(complete_calls) == 1
        assert complete_calls[0] == result


@pytest.mark.asyncio
class TestHierarchicalBeliefSynchronizer:
    """Test HierarchicalBeliefSynchronizer class"""

    def test_hierarchical_creation(self):
        """Test creating hierarchical synchronizer"""
        sync = HierarchicalBeliefSynchronizer(
            agent_id="leader", hierarchy_level=1, parent_id="commander"
        )

        assert sync.agent_id == "leader"
        assert sync.hierarchy_level == 1
        assert sync.parent_id == "commander"
        assert sync.child_ids == set()
        assert sync.sync_protocol == SyncProtocol.PEER_TO_PEER  # Hierarchical uses P2P internally

    def test_hierarchy_management(self):
        """Test managing hierarchy"""
        sync = HierarchicalBeliefSynchronizer("leader")

        sync.add_child("follower_1")
        sync.add_child("follower_2")

        assert "follower_1" in sync.child_ids
        assert "follower_2" in sync.child_ids
        assert len(sync.child_ids) == 2

        sync.remove_child("follower_1")

        assert "follower_1" not in sync.child_ids
        assert len(sync.child_ids) == 1

    @pytest.mark.asyncio
    async def test_synchronize_with_parent(self):
        """Test synchronizing with parent"""
        sync = HierarchicalBeliefSynchronizer(agent_id="child", parent_id="parent")

        result = await sync.synchronize_with_parent(beliefs={"status": "ready"})

        assert result.success is True
        assert result.agents_synced == ["parent"]

    @pytest.mark.asyncio
    async def test_synchronize_with_parent_no_parent(self):
        """Test synchronizing when no parent exists"""
        sync = HierarchicalBeliefSynchronizer("root")

        result = await sync.synchronize_with_parent(beliefs={"status": "ready"})

        assert result.success is False

    @pytest.mark.asyncio
    async def test_synchronize_with_children(self):
        """Test synchronizing with children"""
        sync = HierarchicalBeliefSynchronizer("parent")

        sync.add_child("child_1")
        sync.add_child("child_2")

        result = await sync.synchronize_with_children(beliefs={"command": "execute"})

        assert result.success is True
        assert set(result.agents_synced) == {"child_1", "child_2"}

    @pytest.mark.asyncio
    async def test_synchronize_with_children_no_children(self):
        """Test synchronizing when no children exist"""
        sync = HierarchicalBeliefSynchronizer("leaf")

        result = await sync.synchronize_with_children(beliefs={"data": "value"})

        assert result.success is True
        assert result.agents_synced == []


class TestUtilityFunctions:
    """Test utility functions"""

    def test_transform_belief_format_same(self):
        """Test transforming beliefs with same format"""
        beliefs = {"energy": 50, "status": "active"}
        result = transform_belief_format(beliefs, "raw", "raw")

        assert result == beliefs
        assert result is not beliefs  # Should be a copy

    def test_transform_belief_format_normalize(self):
        """Test normalizing beliefs"""
        beliefs = {"energy": 150, "health": 0.5, "status": "active"}
        result = transform_belief_format(beliefs, "raw", "normalized")

        assert result["energy"] == 1.0  # Clamped to max
        assert result["health"] == 0.5  # Already in range
        assert result["status"] == "active"  # Non-numeric unchanged

    def test_transform_belief_format_unknown(self):
        """Test unknown format transformation"""
        beliefs = {"test": 123}
        result = transform_belief_format(beliefs, "format1", "format2")

        assert result == beliefs  # Should pass through

    def test_merge_beliefs_empty(self):
        """Test merging empty belief sets"""
        result = merge_beliefs([])
        assert result == {}

    def test_merge_beliefs_single(self):
        """Test merging single belief set"""
        beliefs = {"a": 1, "b": 2}
        result = merge_beliefs([beliefs])

        assert result == beliefs
        assert result is not beliefs  # Should be a copy

    def test_merge_beliefs_union(self):
        """Test union merge strategy"""
        set1 = {"a": 1, "b": 2}
        set2 = {"b": 3, "c": 4}
        set3 = {"d": 5}

        result = merge_beliefs([set1, set2, set3], "union")

        assert result["a"] == 1
        assert result["b"] == 3  # Overwritten by set2
        assert result["c"] == 4
        assert result["d"] == 5

    def test_merge_beliefs_intersection(self):
        """Test intersection merge strategy"""
        set1 = {"a": 1, "b": 2, "c": 3}
        set2 = {"b": 4, "c": 5, "d": 6}
        set3 = {"b": 7, "c": 8, "e": 9}

        result = merge_beliefs([set1, set2, set3], "intersection")

        assert "a" not in result  # Not in all sets
        assert "b" in result  # In all sets
        assert "c" in result  # In all sets
        assert "d" not in result  # Not in all sets
        assert result["b"] == 2  # From first set

    def test_merge_beliefs_weighted(self):
        """Test weighted merge strategy"""
        set1 = {"energy": 100, "status": "active"}
        set2 = {"energy": 50, "health": 80}

        result = merge_beliefs([set1, set2], "weighted")

        # Equal weights: (100 + 50) / 2 = 75
        assert result["energy"] == 75
        assert result["status"] == "active"  # Non-numeric from first
        assert result["health"] == 40  # 80 * 0.5 (only in one set)


class TestComplexScenarios:
    """Test complex synchronization scenarios"""

    @pytest.mark.asyncio
    async def test_multi_agent_consensus(self):
        """Test multi-agent consensus building"""
        # Create multiple synchronizers
        agents = []
        for i in range(3):
            sync = BeliefSynchronizer(f"agent_{i}", SyncProtocol.CONSENSUS)
            agents.append(sync)

        # Each agent synchronizes with others
        for i, sync in enumerate(agents):
            other_agents = [f"agent_{j}" for j in range(3) if j != i]
            await sync.synchronize(other_agents, {"vote": i})

        # Check metrics
        for sync in agents:
            assert sync.sync_count == 1
            assert len(sync.last_sync_time) == 2

    @pytest.mark.asyncio
    async def test_hierarchical_propagation(self):
        """Test belief propagation in hierarchy"""
        # Create hierarchy: commander -> [leader1, leader2] -> followers
        commander = HierarchicalBeliefSynchronizer("commander", hierarchy_level=0)

        leaders = []
        for i in range(2):
            leader = HierarchicalBeliefSynchronizer(
                f"leader_{i}", hierarchy_level=1, parent_id="commander"
            )
            commander.add_child(f"leader_{i}")
            leaders.append(leader)

        # Add followers to leaders
        for i, leader in enumerate(leaders):
            for j in range(2):
                leader.add_child(f"follower_{i}_{j}")

        # Propagate command down hierarchy
        await commander.synchronize_with_children({"command": "advance"})

        # Each leader propagates to their followers
        for leader in leaders:
            await leader.synchronize_with_children({"command": "advance"})

        # Verify propagation
        assert commander.sync_count == 1
        for leader in leaders:
            assert leader.sync_count == 1

    def test_conflict_resolution_chain(self):
        """Test chained conflict resolution"""
        resolver = BeliefConflictResolver("weighted_average")

        # Multiple rounds of conflicts
        round1_conflicts = [
            {"belief_data": {"temp": 20}, "weight": 1},
            {"belief_data": {"temp": 30}, "weight": 1},
        ]

        result1 = resolver.resolve(round1_conflicts)
        assert abs(result1["temp"] - 25) < 0.01

        # Use result in next round
        round2_conflicts = [
            {"belief_data": result1, "weight": 2},
            {"belief_data": {"temp": 35}, "weight": 1},
        ]

        result2 = resolver.resolve(round2_conflicts)
        # (25*2 + 35*1) / 3 = 85/3 ≈ 28.33
        assert abs(result2["temp"] - 28.33) < 0.01
