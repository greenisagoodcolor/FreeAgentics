#!/usr/bin/env python3
"""Tests for agent memory lifecycle management.

Tests for Task 5.5: Design agent memory lifecycle management
"""

import logging
import time
import unittest

from agents.memory_optimization.lifecycle_manager import (
    AgentLifecycleState,
    AgentMemoryLifecycleManager,
    AgentMemoryProfile,
    MemoryUsageSnapshot,
    cleanup_agent_memory,
    get_global_lifecycle_manager,
    get_memory_statistics,
    managed_agent_memory,
    register_agent_memory,
    update_agent_memory_usage,
)


class TestAgentMemoryProfile(unittest.TestCase):
    """Test AgentMemoryProfile functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.agent_id = "test_agent_001"
        self.memory_limit = 50.0

    def test_profile_creation(self):
        """Test profile creation with proper initialization."""
        profile = AgentMemoryProfile(
            agent_id=self.agent_id,
            created_at=time.time(),
            last_accessed=time.time(),
            memory_limit_mb=self.memory_limit,
        )

        self.assertEqual(profile.agent_id, self.agent_id)
        self.assertEqual(profile.memory_limit_mb, self.memory_limit)
        self.assertEqual(profile.state, AgentLifecycleState.INITIALIZING)
        self.assertEqual(profile.current_memory_mb, 0.0)
        self.assertEqual(profile.access_count, 0)
        self.assertIsNotNone(profile.belief_compressor)

        # Check lifecycle event was recorded
        self.assertEqual(len(profile.lifecycle_events), 1)
        self.assertEqual(profile.lifecycle_events[0][1], "created")

    def test_memory_usage_update(self):
        """Test memory usage tracking."""
        profile = AgentMemoryProfile(
            agent_id=self.agent_id,
            created_at=time.time(),
            last_accessed=time.time(),
            memory_limit_mb=self.memory_limit,
        )

        # Update memory usage
        belief_mb = 10.0
        matrix_mb = 5.0
        other_mb = 2.0

        profile.update_memory_usage(belief_mb, matrix_mb, other_mb)

        self.assertEqual(profile.current_memory_mb, 17.0)
        self.assertEqual(profile.peak_memory_mb, 17.0)
        self.assertEqual(profile.access_count, 1)
        self.assertTrue(profile.last_accessed > 0)

        # Check memory snapshot was recorded
        self.assertEqual(len(profile.memory_snapshots), 1)
        snapshot = profile.memory_snapshots[0]
        self.assertEqual(snapshot.belief_memory_mb, belief_mb)
        self.assertEqual(snapshot.matrix_memory_mb, matrix_mb)
        self.assertEqual(snapshot.total_memory_mb, 17.0)

    def test_memory_limit_check(self):
        """Test memory limit checking."""
        profile = AgentMemoryProfile(
            agent_id=self.agent_id,
            created_at=time.time(),
            last_accessed=time.time(),
            memory_limit_mb=20.0,
        )

        # Within limit
        profile.update_memory_usage(10.0, 5.0, 2.0)  # 17MB total
        self.assertTrue(profile.check_memory_limit())

        # Over limit
        profile.update_memory_usage(15.0, 8.0, 3.0)  # 26MB total
        self.assertFalse(profile.check_memory_limit())

    def test_memory_efficiency_calculation(self):
        """Test memory efficiency calculation."""
        profile = AgentMemoryProfile(
            agent_id=self.agent_id,
            created_at=time.time(),
            last_accessed=time.time(),
            memory_limit_mb=100.0,
        )

        # Low usage = high efficiency
        profile.update_memory_usage(10.0, 5.0, 5.0)  # 20MB total
        efficiency = profile.get_memory_efficiency()
        self.assertEqual(efficiency, 0.8)  # 1 - (20/100)

        # High usage = low efficiency
        profile.update_memory_usage(40.0, 30.0, 20.0)  # 90MB total
        efficiency = profile.get_memory_efficiency()
        self.assertAlmostEqual(efficiency, 0.1, places=5)  # 1 - (90/100)


class TestAgentMemoryLifecycleManager(unittest.TestCase):
    """Test AgentMemoryLifecycleManager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = AgentMemoryLifecycleManager(
            global_memory_limit_mb=200.0,
            cleanup_threshold=0.8,
            hibernation_idle_seconds=60.0,
            recycling_idle_seconds=120.0,
        )
        self.addCleanup(self.manager.shutdown)

    def test_manager_initialization(self):
        """Test manager initialization."""
        self.assertEqual(self.manager.global_memory_limit_mb, 200.0)
        self.assertEqual(self.manager.cleanup_threshold, 0.8)
        self.assertEqual(self.manager.hibernation_idle_seconds, 60.0)
        self.assertEqual(self.manager.recycling_idle_seconds, 120.0)
        self.assertEqual(len(self.manager._profiles), 0)

    def test_agent_registration(self):
        """Test agent registration and unregistration."""
        agent_id = "test_agent_001"

        # Register agent
        profile = self.manager.register_agent(agent_id, memory_limit_mb=30.0)

        self.assertEqual(profile.agent_id, agent_id)
        self.assertEqual(profile.memory_limit_mb, 30.0)
        self.assertIn(agent_id, self.manager._profiles)
        self.assertEqual(self.manager.stats["total_agents_created"], 1)

        # Unregister agent
        self.manager.unregister_agent(agent_id)
        self.assertNotIn(agent_id, self.manager._profiles)

    def test_duplicate_registration(self):
        """Test handling of duplicate agent registration."""
        agent_id = "test_agent_001"

        # Register agent twice
        profile1 = self.manager.register_agent(agent_id, memory_limit_mb=30.0)
        profile2 = self.manager.register_agent(agent_id, memory_limit_mb=40.0)

        # Should return the same profile
        self.assertEqual(profile1, profile2)
        self.assertEqual(
            profile1.memory_limit_mb, 30.0
        )  # Original limit preserved

    def test_memory_usage_update(self):
        """Test memory usage updates."""
        agent_id = "test_agent_001"
        profile = self.manager.register_agent(agent_id, memory_limit_mb=50.0)

        # Update within limit
        within_limit = self.manager.update_agent_memory(
            agent_id, 15.0, 10.0, 5.0
        )
        self.assertTrue(within_limit)
        self.assertEqual(profile.current_memory_mb, 30.0)

        # Update over limit
        over_limit = self.manager.update_agent_memory(
            agent_id, 30.0, 20.0, 10.0
        )
        self.assertFalse(over_limit)
        self.assertEqual(profile.current_memory_mb, 60.0)

    def test_memory_context_manager(self):
        """Test agent memory context manager."""
        agent_id = "test_agent_001"
        self.manager.register_agent(agent_id)

        with self.manager.agent_memory_context(agent_id) as profile:
            self.assertEqual(profile.agent_id, agent_id)
            # Check lifecycle events were recorded
            events = [event[1] for event in profile.lifecycle_events]
            self.assertIn("memory_context_enter", events)

        # Check exit event was recorded
        events = [event[1] for event in profile.lifecycle_events]
        self.assertIn("memory_context_exit", events)

    def test_agent_hibernation(self):
        """Test agent hibernation functionality."""
        agent_id = "test_agent_001"
        profile = self.manager.register_agent(agent_id)

        # Hibernating should succeed
        success = self.manager.hibernate_agent(agent_id)
        self.assertTrue(success)
        self.assertEqual(profile.state, AgentLifecycleState.HIBERNATING)

        # Check lifecycle event
        events = [event[1] for event in profile.lifecycle_events]
        self.assertIn("hibernated", events)

    def test_agent_wake_from_hibernation(self):
        """Test waking agent from hibernation."""
        agent_id = "test_agent_001"
        profile = self.manager.register_agent(agent_id)

        # Hibernate then wake
        self.manager.hibernate_agent(agent_id)
        success = self.manager.wake_agent(agent_id)

        self.assertTrue(success)
        self.assertEqual(profile.state, AgentLifecycleState.ACTIVE)

        # Check lifecycle event
        events = [event[1] for event in profile.lifecycle_events]
        self.assertIn("awakened", events)

    def test_agent_recycling(self):
        """Test agent recycling functionality."""
        agent_id = "test_agent_001"
        profile = self.manager.register_agent(agent_id)

        # Set some memory usage
        self.manager.update_agent_memory(agent_id, 10.0, 5.0, 2.0)
        profile.current_memory_mb

        # Recycle agent
        success = self.manager.recycle_agent(agent_id)

        self.assertTrue(success)
        self.assertEqual(profile.state, AgentLifecycleState.RECYCLING)
        self.assertEqual(profile.current_memory_mb, 0.0)
        self.assertEqual(profile.access_count, 0)
        self.assertEqual(len(profile.memory_snapshots), 0)
        self.assertEqual(self.manager.stats["total_agents_recycled"], 1)

        # Check lifecycle event
        events = [event[1] for event in profile.lifecycle_events]
        self.assertIn("recycled", events)

    def test_total_memory_calculation(self):
        """Test total memory usage calculation."""
        # Register multiple agents with different memory usage
        agent1 = "agent_001"
        agent2 = "agent_002"

        self.manager.register_agent(agent1)
        self.manager.register_agent(agent2)

        self.manager.update_agent_memory(agent1, 10.0, 5.0, 2.0)  # 17MB
        self.manager.update_agent_memory(agent2, 15.0, 8.0, 3.0)  # 26MB

        total_memory = self.manager.get_total_memory_usage()
        self.assertEqual(total_memory, 43.0)  # 17 + 26

    def test_memory_pressure_calculation(self):
        """Test memory pressure calculation."""
        # Set global limit to 100MB
        self.manager.global_memory_limit_mb = 100.0

        agent_id = "test_agent_001"
        self.manager.register_agent(agent_id)

        # Low pressure
        self.manager.update_agent_memory(agent_id, 10.0, 5.0, 5.0)  # 20MB
        pressure = self.manager.get_memory_pressure()
        self.assertEqual(pressure, 0.2)  # 20/100

        # High pressure
        self.manager.update_agent_memory(agent_id, 40.0, 30.0, 20.0)  # 90MB
        pressure = self.manager.get_memory_pressure()
        self.assertEqual(pressure, 0.9)  # 90/100

    def test_cleanup_idle_agents(self):
        """Test cleanup of idle agents."""
        # Register agents
        agent1 = "agent_001"
        agent2 = "agent_002"

        profile1 = self.manager.register_agent(agent1)
        profile2 = self.manager.register_agent(agent2)

        # Set different last access times and states
        current_time = time.time()
        profile1.last_accessed = (
            current_time - 70.0
        )  # Hibernation candidate (>60s)
        profile1.state = (
            AgentLifecycleState.ACTIVE
        )  # Must be ACTIVE to be hibernated
        profile2.last_accessed = (
            current_time - 130.0
        )  # Recycling candidate (>120s)
        profile2.state = (
            AgentLifecycleState.HIBERNATING
        )  # HIBERNATING can be recycled

        # Run cleanup
        cleanup_stats = self.manager.cleanup_idle_agents()

        # Check results
        self.assertEqual(cleanup_stats["hibernated"], 1)
        self.assertEqual(cleanup_stats["recycled"], 1)
        self.assertEqual(profile1.state, AgentLifecycleState.HIBERNATING)
        self.assertEqual(profile2.state, AgentLifecycleState.RECYCLING)

    def test_lifecycle_statistics(self):
        """Test lifecycle statistics collection."""
        # Register agents with different states
        agent1 = "agent_001"
        agent2 = "agent_002"
        agent3 = "agent_003"

        profile1 = self.manager.register_agent(agent1)
        profile2 = self.manager.register_agent(agent2)
        profile3 = self.manager.register_agent(agent3)

        # Set different states
        profile1.state = AgentLifecycleState.ACTIVE
        profile2.state = AgentLifecycleState.HIBERNATING
        profile3.state = AgentLifecycleState.RECYCLING

        # Set memory usage
        self.manager.update_agent_memory(agent1, 10.0, 5.0, 2.0)
        self.manager.update_agent_memory(agent2, 8.0, 3.0, 1.0)
        self.manager.update_agent_memory(agent3, 0.0, 0.0, 0.0)

        stats = self.manager.get_lifecycle_statistics()

        # Check global stats
        self.assertEqual(stats["global"]["total_agents"], 3)
        self.assertEqual(
            stats["global"]["total_memory_mb"], 29.0
        )  # 17 + 12 + 0

        # Check state counts
        self.assertEqual(stats["agent_states"]["active"], 1)
        self.assertEqual(stats["agent_states"]["hibernating"], 1)
        self.assertEqual(stats["agent_states"]["recycling"], 1)

        # Check memory by state
        self.assertEqual(stats["memory_by_state"]["active"], 17.0)
        self.assertEqual(stats["memory_by_state"]["hibernating"], 12.0)
        self.assertEqual(stats["memory_by_state"]["recycling"], 0.0)

    def test_force_cleanup(self):
        """Test force cleanup functionality."""
        # Register agents and set high memory usage
        agent1 = "agent_001"
        agent2 = "agent_002"

        self.manager.register_agent(agent1)
        self.manager.register_agent(agent2)

        # Set high memory usage to trigger cleanup
        self.manager.update_agent_memory(agent1, 50.0, 30.0, 20.0)  # 100MB
        self.manager.update_agent_memory(agent2, 40.0, 25.0, 15.0)  # 80MB

        # Force cleanup
        cleanup_stats = self.manager.force_cleanup()

        # Check that cleanup was performed
        self.assertIsInstance(cleanup_stats, dict)
        self.assertIn("gc_collected", cleanup_stats)
        self.assertEqual(self.manager.stats["cleanup_cycles"], 1)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear global instance
        import agents.memory_optimization.lifecycle_manager as lm

        if lm._global_lifecycle_manager is not None:
            lm._global_lifecycle_manager.shutdown()
            lm._global_lifecycle_manager = None

    def tearDown(self):
        """Clean up after tests."""
        # Shutdown global instance
        import agents.memory_optimization.lifecycle_manager as lm

        if lm._global_lifecycle_manager is not None:
            lm._global_lifecycle_manager.shutdown()
            lm._global_lifecycle_manager = None

    def test_global_lifecycle_manager(self):
        """Test global lifecycle manager singleton."""
        manager1 = get_global_lifecycle_manager()
        manager2 = get_global_lifecycle_manager()

        self.assertIs(manager1, manager2)  # Same instance
        self.assertIsInstance(manager1, AgentMemoryLifecycleManager)

    def test_register_agent_memory_convenience(self):
        """Test convenience function for agent registration."""
        agent_id = "test_agent_001"
        profile = register_agent_memory(agent_id, memory_limit_mb=40.0)

        self.assertEqual(profile.agent_id, agent_id)
        self.assertEqual(profile.memory_limit_mb, 40.0)

    def test_update_agent_memory_usage_convenience(self):
        """Test convenience function for memory usage update."""
        agent_id = "test_agent_001"
        register_agent_memory(agent_id)

        within_limit = update_agent_memory_usage(agent_id, 10.0, 5.0, 2.0)
        self.assertTrue(within_limit)

    def test_cleanup_agent_memory_convenience(self):
        """Test convenience function for memory cleanup."""
        agent_id = "test_agent_001"
        register_agent_memory(agent_id)

        cleanup_stats = cleanup_agent_memory()
        self.assertIsInstance(cleanup_stats, dict)

    def test_get_memory_statistics_convenience(self):
        """Test convenience function for memory statistics."""
        agent_id = "test_agent_001"
        register_agent_memory(agent_id)

        stats = get_memory_statistics()
        self.assertIsInstance(stats, dict)
        self.assertIn("global", stats)

    def test_managed_agent_memory_context(self):
        """Test managed agent memory context manager."""
        agent_id = "test_agent_001"

        with managed_agent_memory(agent_id, memory_limit_mb=30.0) as profile:
            self.assertEqual(profile.agent_id, agent_id)
            self.assertEqual(profile.memory_limit_mb, 30.0)

        # Agent should be hibernated after context
        manager = get_global_lifecycle_manager()
        profile = manager.get_agent_profile(agent_id)
        self.assertEqual(profile.state, AgentLifecycleState.HIBERNATING)


class TestMemoryUsageSnapshot(unittest.TestCase):
    """Test MemoryUsageSnapshot functionality."""

    def test_snapshot_creation(self):
        """Test memory usage snapshot creation."""
        snapshot = MemoryUsageSnapshot(
            timestamp=time.time(),
            belief_memory_mb=10.0,
            matrix_memory_mb=5.0,
            total_memory_mb=15.0,
            peak_memory_mb=20.0,
            gc_collections=5,
            state=AgentLifecycleState.ACTIVE,
        )

        self.assertEqual(snapshot.belief_memory_mb, 10.0)
        self.assertEqual(snapshot.matrix_memory_mb, 5.0)
        self.assertEqual(snapshot.total_memory_mb, 15.0)
        self.assertEqual(snapshot.peak_memory_mb, 20.0)
        self.assertEqual(snapshot.gc_collections, 5)
        self.assertEqual(snapshot.state, AgentLifecycleState.ACTIVE)


if __name__ == "__main__":
    # Set up logging for tests
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main()
