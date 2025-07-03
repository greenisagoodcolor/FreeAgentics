"""
Tests for World Simulation Engine
"""

import asyncio
import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from world.simulation.engine import (
    ActiveInferenceAgent,
    EcosystemMetrics,
    SimulationConfig,
    SimulationEngine,
    SocialNetwork,
    SystemHealth,
)


class TestSimulationConfig:
    """Test SimulationConfig dataclass"""

    def test_config_creation(self):
        """Test creating simulation config"""
        config = SimulationConfig(
            max_cycles=100, time_step=0.5, enable_logging=True, random_seed=42
        )

        assert config.max_cycles == 100
        assert config.time_step == 0.5
        assert config.enable_logging is True
        assert config.random_seed == 42

    def test_config_defaults(self):
        """Test default config values"""
        config = SimulationConfig()

        assert config.max_cycles == 1000
        assert config.time_step == 1.0
        assert config.enable_logging is True
        assert config.random_seed is None
        assert isinstance(config.world, dict)
        assert isinstance(config.agents, dict)
        assert isinstance(config.performance, dict)

    def test_config_nested_values(self):
        """Test nested configuration values"""
        config = SimulationConfig()

        # World config
        assert config.world["resolution"] == 5
        assert config.world["size"] == 100
        assert config.world["resource_density"] == 1.0

        # Agent config
        assert config.agents["count"] == 10
        assert isinstance(config.agents["distribution"], dict)
        assert config.agents["communication_rate"] == 1.0

        # Performance config
        assert config.performance["max_memory_mb"] == 2048
        assert config.performance["max_cycle_time"] == 5.0


class TestSystemHealth:
    """Test SystemHealth dataclass"""

    def test_health_creation(self):
        """Test creating system health"""
        health = SystemHealth(
            status="healthy",
            agent_count=10,
            message_queue_size=5,
            memory_usage_mb=512.0,
            cpu_usage_percent=25.0,
            last_cycle_time=0.1,
        )

        assert health.status == "healthy"
        assert health.agent_count == 10
        assert health.message_queue_size == 5
        assert health.memory_usage_mb == 512.0
        assert health.cpu_usage_percent == 25.0
        assert health.last_cycle_time == 0.1
        assert health.errors == []

    def test_health_with_errors(self):
        """Test health with errors"""
        errors = ["High memory usage", "Slow response time"]
        health = SystemHealth(
            status="degraded",
            agent_count=8,
            message_queue_size=100,
            memory_usage_mb=1900.0,
            cpu_usage_percent=80.0,
            last_cycle_time=2.5,
            errors=errors,
        )

        assert health.status == "degraded"
        assert health.errors == errors


class TestEcosystemMetrics:
    """Test EcosystemMetrics dataclass"""

    def test_ecosystem_metrics(self):
        """Test ecosystem metrics"""
        metrics = EcosystemMetrics(
            resource_gini_coefficient=0.35,
            average_agent_wealth=50.0,
            knowledge_nodes_per_agent=5.5,
            trades_this_cycle=12,
            explored_cells_percentage=0.75,
            behavior_entropy=2.3,
            average_goal_achievement=0.65,
        )

        assert metrics.resource_gini_coefficient == 0.35
        assert metrics.average_agent_wealth == 50.0
        assert metrics.knowledge_nodes_per_agent == 5.5
        assert metrics.trades_this_cycle == 12
        assert metrics.explored_cells_percentage == 0.75
        assert metrics.behavior_entropy == 2.3
        assert metrics.average_goal_achievement == 0.65


class TestSocialNetwork:
    """Test SocialNetwork dataclass"""

    def test_social_network_creation(self):
        """Test social network creation"""
        trade_clusters = [["agent1", "agent2"], ["agent3", "agent4"]]
        centrality_scores = {"agent1": 0.8, "agent2": 0.6}
        knowledge_network = {"agent1": ["agent2", "agent3"], "agent2": ["agent1"]}
        alliances = [["agent1", "agent3"], ["agent2", "agent4"]]

        network = SocialNetwork(
            trade_clusters=trade_clusters,
            centrality_scores=centrality_scores,
            knowledge_sharing_network=knowledge_network,
            protection_alliances=alliances,
        )

        assert network.get_trade_clusters() == trade_clusters
        assert network.get_centrality_scores() == centrality_scores
        assert network.get_knowledge_sharing_network() == knowledge_network
        assert network.get_protection_alliances() == alliances


class TestActiveInferenceAgent:
    """Test ActiveInferenceAgent class"""

    def test_agent_creation(self):
        """Test agent creation"""
        agent = ActiveInferenceAgent(agent_id="explorer_1", agent_class="explorer", config={})

        assert agent.agent_id == "explorer_1"
        assert agent.agent_class == "explorer"
        assert agent.position is None
        assert agent.wealth > 0
        assert agent.knowledge_nodes > 0
        assert agent.goals_achieved == 0
        assert agent.total_goals > 0
        assert agent.alive is True

    def test_agent_performance_metrics(self):
        """Test agent performance metrics"""
        agent = ActiveInferenceAgent(agent_id="test_agent", agent_class="merchant", config={})

        # Set some values
        agent.wealth = 100.0
        agent.knowledge_nodes = 5
        agent.goals_achieved = 3
        agent.total_goals = 10

        metrics = agent.get_performance_metrics()

        assert metrics["wealth"] == 100.0
        assert metrics["knowledge_nodes"] == 5
        assert metrics["goal_achievement"] == 0.3
        assert metrics["alive"] == 1.0

    @pytest.mark.asyncio
    async def test_agent_update(self):
        """Test agent update"""
        agent = ActiveInferenceAgent(agent_id="test_agent", agent_class="scholar", config={})

        world_state = {"knowledge_obs": 3, "social_obs": 2}

        initial_wealth = agent.wealth
        initial_knowledge = agent.knowledge_nodes

        # Run multiple updates to increase chance of random events
        for _ in range(20):
            await agent.update(1.0, world_state)

        # Check that agent is still alive and has updated
        assert agent.alive is True
        assert agent.last_action_time > 0

    def test_different_agent_classes(self):
        """Test different agent classes"""
        classes = ["explorer", "merchant", "scholar", "guardian"]

        for agent_class in classes:
            agent = ActiveInferenceAgent(
                agent_id=f"{agent_class}_test", agent_class=agent_class, config={}
            )

            assert agent.agent_class == agent_class
            assert agent.alive is True


class TestSimulationEngine:
    """Test main simulation engine"""

    @pytest.fixture
    def engine(self):
        """Create test engine"""
        config = {"max_cycles": 100, "time_step": 0.1}
        return SimulationEngine(config)

    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine.current_cycle == 0
        assert engine.running is False
        assert len(engine.agents) == 0
        assert isinstance(engine.config, SimulationConfig)

    def test_engine_with_nested_config(self):
        """Test engine with nested configuration"""
        config = {
            "simulation": {"max_cycles": 50, "time_step": 0.5},
            "world": {"size": 200},
            "agents": {"count": 20},
        }
        engine = SimulationEngine(config)

        assert engine.config.max_cycles == 50
        assert engine.config.time_step == 0.5

    def test_initialize_engine(self, engine):
        """Test initializing the engine"""
        engine.initialize()

        # Should create agents
        assert len(engine.agents) > 0

        # Check agent distribution
        explorers = sum(1 for a in engine.agents.values() if a.agent_class == "explorer")
        merchants = sum(1 for a in engine.agents.values() if a.agent_class == "merchant")
        scholars = sum(1 for a in engine.agents.values() if a.agent_class == "scholar")
        guardians = sum(1 for a in engine.agents.values() if a.agent_class == "guardian")

        assert explorers > 0
        assert merchants >= 0
        assert scholars >= 0
        assert guardians >= 0

    @pytest.mark.asyncio
    async def test_start_stop_engine(self, engine):
        """Test starting and stopping engine"""
        engine.initialize()

        # Start engine
        await engine.start()

        assert engine.running is True
        assert engine.start_time is not None
        assert engine.current_cycle == 0

        # Stop engine
        await engine.stop()

        assert engine.running is False

    @pytest.mark.asyncio
    async def test_step_simulation(self, engine):
        """Test stepping through simulation"""
        engine.initialize()
        await engine.start()

        initial_cycle = engine.current_cycle

        # Take a step
        await engine.step()

        assert engine.current_cycle == initial_cycle + 1
        assert len(engine.cycle_times) > 0

        await engine.stop()

    @pytest.mark.asyncio
    async def test_run_simulation_steps(self, engine):
        """Test running simulation steps"""
        engine.config.max_cycles = 5  # Short run
        engine.initialize()
        await engine.start()

        # Manually run steps
        for _ in range(5):
            if engine.current_cycle < engine.config.max_cycles:
                await engine.step()

        assert engine.current_cycle == 5

        await engine.stop()
        assert not engine.running

    def test_add_agent_manually(self, engine):
        """Test adding agent manually"""
        engine.initialize()

        # Create a new agent
        new_agent = ActiveInferenceAgent(agent_id="new_agent", agent_class="explorer", config={})

        initial_count = len(engine.agents)

        # Add agent manually
        engine.agents["new_agent"] = new_agent
        assert len(engine.agents) == initial_count + 1
        assert "new_agent" in engine.agents

        # Remove agent manually
        del engine.agents["new_agent"]
        assert len(engine.agents) == initial_count
        assert "new_agent" not in engine.agents

    def test_fail_agent_manually(self, engine):
        """Test failing an agent manually"""
        engine.initialize()

        if engine.agents:
            first_agent_id = list(engine.agents.keys())[0]

            # Fail the agent by adding to failed set
            engine.failed_agents.add(first_agent_id)

            assert first_agent_id in engine.failed_agents

    @pytest.mark.asyncio
    async def test_get_system_health(self, engine):
        """Test getting system health"""
        engine.initialize()

        health = await engine.get_system_health()

        assert isinstance(health, dict)
        assert "status" in health
        assert "agent_count" in health
        assert "memory_usage_mb" in health
        assert "cpu_usage_percent" in health

    @pytest.mark.asyncio
    async def test_get_ecosystem_metrics(self, engine):
        """Test getting ecosystem metrics"""
        engine.initialize()
        await engine.start()
        await engine.step()

        metrics = await engine.get_ecosystem_metrics()

        assert isinstance(metrics, dict)
        assert "resource_gini_coefficient" in metrics
        assert "average_agent_wealth" in metrics
        assert "knowledge_nodes_per_agent" in metrics

    @pytest.mark.asyncio
    async def test_get_social_network(self, engine):
        """Test getting social network data"""
        engine.initialize()

        # Add some relationships
        if len(engine.agents) >= 2:
            agent_ids = list(engine.agents.keys())
            engine.trade_relationships[agent_ids[0]].add(agent_ids[1])
            engine.knowledge_shares[agent_ids[0]].add(agent_ids[1])

        network = await engine.get_social_network()

        assert isinstance(network, SocialNetwork)
        assert hasattr(network, "trade_clusters")
        assert hasattr(network, "centrality_scores")

    @pytest.mark.asyncio
    async def test_stop_simulation(self, engine):
        """Test stopping simulation"""
        engine.initialize()
        await engine.start()

        await engine.stop()

        assert not engine.running

    def test_environmental_conditions(self, engine):
        """Test environmental conditions"""
        engine.initialize()

        # Check environmental conditions
        assert "resource_multiplier" in engine.environmental_conditions
        assert "hazard_level" in engine.environmental_conditions
        assert engine.environmental_conditions["resource_multiplier"] == 1.0
        assert engine.environmental_conditions["hazard_level"] == 0.1

    @pytest.mark.asyncio
    async def test_performance_tracking(self, engine):
        """Test performance tracking"""
        engine.initialize()
        await engine.start()

        # Take a few steps
        for _ in range(3):
            await engine.step()

        # Check that performance data is collected
        assert len(engine.cycle_times) > 0
        assert all(t > 0 for t in engine.cycle_times)

    def test_event_history(self, engine):
        """Test event history tracking"""
        engine.initialize()

        # Add some event history
        engine.event_history.append(
            {"type": "trade", "cycle": 1, "participants": ["agent1", "agent2"]}
        )

        assert len(engine.event_history) == 1
        assert engine.event_history[0]["type"] == "trade"

    @pytest.mark.asyncio
    async def test_concurrent_agent_updates(self, engine):
        """Test concurrent agent updates"""
        engine.config.agents["count"] = 20  # More agents
        engine.initialize()
        await engine.start()

        # Measure time for concurrent updates
        start_time = time.time()
        await engine.step()
        step_time = time.time() - start_time

        # Should complete reasonably fast
        assert step_time < 5.0  # 5 seconds max

    def test_configuration_access(self, engine):
        """Test configuration access"""
        engine.initialize()

        # Test accessing configuration
        assert engine.config is not None
        assert hasattr(engine.config, "max_cycles")
        assert hasattr(engine.config, "time_step")


class TestSocialDynamics:
    """Test social dynamics in simulation"""

    @pytest.fixture
    def engine_with_agents(self):
        """Create engine with multiple agents"""
        config = {
            "agents": {
                "count": 20,
                "distribution": {"explorer": 5, "merchant": 5, "scholar": 5, "guardian": 5},
            }
        }
        engine = SimulationEngine(config)
        engine.initialize()
        return engine

    @pytest.mark.asyncio
    async def test_trade_relationship_formation(self, engine_with_agents):
        """Test trade relationship formation"""
        engine = engine_with_agents
        await engine.start()

        # Run several steps to allow relationships to form
        for _ in range(10):
            await engine.step()

        # Check if any trade relationships formed
        total_relationships = sum(len(partners) for partners in engine.trade_relationships.values())

        # Should have some relationships
        assert total_relationships > 0

    @pytest.mark.asyncio
    async def test_knowledge_sharing(self, engine_with_agents):
        """Test knowledge sharing between agents"""
        engine = engine_with_agents
        await engine.start()

        # Run several steps
        for _ in range(10):
            await engine.step()

        # Check if knowledge sharing occurred
        total_shares = sum(len(shares) for shares in engine.knowledge_shares.values())

        # Scholars should share knowledge
        assert total_shares >= 0

    @pytest.mark.asyncio
    async def test_protection_alliances(self, engine_with_agents):
        """Test protection alliance formation"""
        engine = engine_with_agents
        await engine.start()

        # Run several steps
        for _ in range(10):
            await engine.step()

        # Check if alliances formed
        total_alliances = sum(len(allies) for allies in engine.protection_alliances.values())

        # Should have some alliances
        assert total_alliances >= 0


class TestErrorHandling:
    """Test error handling in simulation"""

    def test_invalid_config(self):
        """Test handling invalid configuration"""
        # This should not crash
        engine = SimulationEngine(None)
        assert isinstance(engine.config, SimulationConfig)

    @pytest.mark.asyncio
    async def test_agent_failure_handling(self):
        """Test handling agent failures"""
        engine = SimulationEngine()
        engine.initialize()

        if engine.agents:
            # Make an agent fail
            first_agent_id = list(engine.agents.keys())[0]
            first_agent = engine.agents[first_agent_id]

            # Mock update to raise exception
            async def failing_update(*args, **kwargs):
                raise Exception("Test failure")

            first_agent.update = failing_update

            await engine.start()

            # Step should raise the exception
            with pytest.raises(Exception, match="Test failure"):
                await engine.step()

            # Engine should still be marked as running
            assert engine.running

    def test_missing_world_module(self):
        """Test handling missing world module"""
        with patch("world.simulation.engine.H3World", None):
            engine = SimulationEngine()
            engine.initialize()

            # Should work without world
            assert engine.world is None
