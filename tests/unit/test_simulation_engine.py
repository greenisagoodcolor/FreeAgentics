"""
Comprehensive tests for Advanced Async Simulation Engine.

Tests the sophisticated simulation engine that integrates Active Inference agents,
real-time system monitoring, ecosystem dynamics, fault tolerance, and edge deployment
capabilities for the FreeAgentics multi-agent system.
"""

import asyncio
import json
import tempfile
from pathlib import Path
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
    """Test SimulationConfig dataclass functionality."""

    def test_simulation_config_creation(self):
        """Test creating a simulation config with defaults."""
        config = SimulationConfig()

        assert config.max_cycles == 1000
        assert config.time_step == 1.0
        assert config.enable_logging is True
        assert config.random_seed is None
        assert config.world["resolution"] == 5
        assert config.world["size"] == 100
        assert config.world["resource_density"] == 1.0
        assert config.agents["count"] == 10
        assert config.performance["max_memory_mb"] == 2048
        assert config.performance["max_cycle_time"] == 5.0

    def test_simulation_config_custom(self):
        """Test creating simulation config with custom values."""
        config = SimulationConfig(
            max_cycles=500,
            time_step=0.5,
            random_seed=42,
            world={"resolution": 7, "size": 50},
            agents={"count": 20, "distribution": {"explorer": 10, "merchant": 10}},
            performance={"max_memory_mb": 4096, "max_cycle_time": 10.0},
        )

        assert config.max_cycles == 500
        assert config.time_step == 0.5
        assert config.random_seed == 42
        assert config.world["resolution"] == 7
        assert config.agents["count"] == 20
        assert config.performance["max_memory_mb"] == 4096


class TestSystemHealth:
    """Test SystemHealth dataclass functionality."""

    def test_system_health_creation(self):
        """Test creating system health status."""
        health = SystemHealth(
            status="healthy",
            agent_count=10,
            message_queue_size=5,
            memory_usage_mb=512.0,
            cpu_usage_percent=25.5,
            last_cycle_time=0.1,
            errors=["warning: high load"],
        )

        assert health.status == "healthy"
        assert health.agent_count == 10
        assert health.message_queue_size == 5
        assert health.memory_usage_mb == 512.0
        assert health.cpu_usage_percent == 25.5
        assert health.last_cycle_time == 0.1
        assert health.errors == ["warning: high load"]

    def test_system_health_defaults(self):
        """Test system health with default error list."""
        health = SystemHealth(
            status="degraded",
            agent_count=8,
            message_queue_size=15,
            memory_usage_mb=1024.0,
            cpu_usage_percent=75.0,
            last_cycle_time=2.5,
        )

        assert health.errors == []


class TestEcosystemMetrics:
    """Test EcosystemMetrics dataclass functionality."""

    def test_ecosystem_metrics_creation(self):
        """Test creating ecosystem metrics."""
        metrics = EcosystemMetrics(
            resource_gini_coefficient=0.35,
            average_agent_wealth=150.5,
            knowledge_nodes_per_agent=7.2,
            trades_this_cycle=12,
            explored_cells_percentage=45.8,
            behavior_entropy=1.85,
            average_goal_achievement=0.73,
        )

        assert metrics.resource_gini_coefficient == 0.35
        assert metrics.average_agent_wealth == 150.5
        assert metrics.knowledge_nodes_per_agent == 7.2
        assert metrics.trades_this_cycle == 12
        assert metrics.explored_cells_percentage == 45.8
        assert metrics.behavior_entropy == 1.85
        assert metrics.average_goal_achievement == 0.73


class TestSocialNetwork:
    """Test SocialNetwork dataclass functionality."""

    def test_social_network_creation(self):
        """Test creating social network analysis data."""
        network = SocialNetwork(
            trade_clusters=[["agent_1", "agent_2"], ["agent_3", "agent_4", "agent_5"]],
            centrality_scores={"agent_1": 0.8, "agent_2": 0.6, "agent_3": 0.9},
            knowledge_sharing_network={"agent_1": ["agent_2", "agent_3"], "agent_2": ["agent_1"]},
            protection_alliances=[["guardian_1", "agent_1", "agent_2"]],
        )

        assert len(network.trade_clusters) == 2
        assert network.trade_clusters[0] == ["agent_1", "agent_2"]
        assert network.centrality_scores["agent_3"] == 0.9
        assert "agent_2" in network.knowledge_sharing_network["agent_1"]
        assert len(network.protection_alliances) == 1

    def test_social_network_getters(self):
        """Test social network getter methods."""
        network = SocialNetwork(
            trade_clusters=[["a", "b"]],
            centrality_scores={"a": 0.5},
            knowledge_sharing_network={"a": ["b"]},
            protection_alliances=[["guard", "a"]],
        )

        assert network.get_trade_clusters() == [["a", "b"]]
        assert network.get_centrality_scores() == {"a": 0.5}
        assert network.get_knowledge_sharing_network() == {"a": ["b"]}
        assert network.get_protection_alliances() == [["guard", "a"]]


class TestActiveInferenceAgent:
    """Test ActiveInferenceAgent functionality."""

    def test_agent_creation_basic(self):
        """Test creating basic Active Inference agent."""
        agent = ActiveInferenceAgent("test_agent", "explorer", {"communication_rate": 1.0})

        assert agent.agent_id == "test_agent"
        assert agent.agent_class == "explorer"
        assert agent.config == {"communication_rate": 1.0}
        assert agent.alive is True
        assert agent.wealth > 0
        assert agent.knowledge_nodes > 0
        assert agent.goals_achieved == 0
        assert agent.total_goals > 0

    def test_agent_creation_different_classes(self):
        """Test creating agents of different classes."""
        classes = ["explorer", "merchant", "scholar", "guardian"]

        for agent_class in classes:
            agent = ActiveInferenceAgent(f"test_{agent_class}", agent_class, {})
            assert agent.agent_class == agent_class
            assert agent.alive is True

    @patch("world.simulation.engine.PYMDP_AVAILABLE", True)
    @patch("world.simulation.engine.utils")
    @patch("world.simulation.engine.PyMDPAgent")
    def test_agent_pymdp_initialization_explorer(self, mock_pymdp_agent, mock_utils):
        """Test pymdp initialization for explorer agent."""
        mock_utils.random_A_matrix.return_value = "A_matrix"
        mock_utils.random_B_matrix.return_value = "B_matrix"
        mock_utils.obj_array_uniform.return_value = "C_vector"

        ActiveInferenceAgent("explorer_1", "explorer", {})

        # Should call pymdp initialization
        assert mock_utils.random_A_matrix.called
        assert mock_utils.random_B_matrix.called
        assert mock_utils.obj_array_uniform.called
        mock_pymdp_agent.assert_called_once()

    @patch("world.simulation.engine.PYMDP_AVAILABLE", True)
    @patch("world.simulation.engine.utils")
    @patch("world.simulation.engine.PyMDPAgent")
    def test_agent_pymdp_initialization_different_classes(self, mock_pymdp_agent, mock_utils):
        """Test pymdp initialization for different agent classes."""
        mock_utils.random_A_matrix.return_value = "A_matrix"
        mock_utils.random_B_matrix.return_value = "B_matrix"
        mock_utils.obj_array_uniform.return_value = "C_vector"

        classes = ["explorer", "merchant", "scholar", "guardian"]

        for agent_class in classes:
            mock_pymdp_agent.reset_mock()
            ActiveInferenceAgent(f"test_{agent_class}", agent_class, {})

            # Each class should initialize pymdp agent
            mock_pymdp_agent.assert_called_once()

    @patch("world.simulation.engine.PYMDP_AVAILABLE", True)
    @patch("world.simulation.engine.utils")
    @patch("world.simulation.engine.PyMDPAgent")
    def test_agent_pymdp_initialization_failure(self, mock_pymdp_agent, mock_utils):
        """Test handling of pymdp initialization failure."""
        mock_utils.random_A_matrix.side_effect = Exception("PyMDP error")

        agent = ActiveInferenceAgent("test_agent", "explorer", {})

        # Should handle error gracefully
        assert agent.pymdp_agent is None

    @pytest.mark.asyncio
    async def test_agent_update_basic(self):
        """Test basic agent update functionality."""
        agent = ActiveInferenceAgent("test_agent", "explorer", {})
        world_state = {"position_obs": 1, "resource_obs": 2, "market_obs": 0, "inventory_obs": 1}

        agent.wealth
        agent.knowledge_nodes
        agent.goals_achieved

        # Mock random to ensure some progression
        with patch(
            "numpy.random.random", side_effect=[0.05, 0.02]
        ):  # Trigger goal achievement and knowledge growth
            await agent.update(1.0, world_state)

        # Agent should be alive and potentially progressed
        assert agent.alive is True
        assert agent.last_action_time > 0

    @pytest.mark.asyncio
    async def test_agent_update_dead_agent(self):
        """Test update of dead agent."""
        agent = ActiveInferenceAgent("test_agent", "explorer", {})
        agent.alive = False

        await agent.update(1.0, {})

        # Dead agent should not change
        assert agent.alive is False

    @pytest.mark.asyncio
    @patch("world.simulation.engine.PYMDP_AVAILABLE", True)
    def test_agent_update_with_pymdp(self):
        """Test agent update with pymdp integration."""
        agent = ActiveInferenceAgent("test_agent", "explorer", {})

        # Mock pymdp agent
        mock_pymdp = Mock()
        mock_pymdp.infer_states.return_value = "belief_state"
        mock_pymdp.infer_policies.return_value = ("policies", "neg_efe")
        mock_pymdp.sample_action.return_value = [1, 0]
        agent.pymdp_agent = mock_pymdp

        world_state = {"position_obs": 1, "resource_obs": 2}

        # Should call pymdp methods
        asyncio.run(agent.update(1.0, world_state))

        mock_pymdp.infer_states.assert_called_once()
        mock_pymdp.infer_policies.assert_called_once()
        mock_pymdp.sample_action.assert_called_once()

    def test_agent_get_observation_explorer(self):
        """Test observation extraction for explorer agent."""
        agent = ActiveInferenceAgent("explorer_1", "explorer", {})
        world_state = {"position_obs": 3, "resource_obs": 1, "market_obs": 2}

        obs = agent._get_observation(world_state)

        assert obs == [3, 1]

    def test_agent_get_observation_merchant(self):
        """Test observation extraction for merchant agent."""
        agent = ActiveInferenceAgent("merchant_1", "merchant", {})
        world_state = {"market_obs": 2, "inventory_obs": 3, "position_obs": 1}

        obs = agent._get_observation(world_state)

        assert obs == [2, 3]

    def test_agent_get_observation_scholar(self):
        """Test observation extraction for scholar agent."""
        agent = ActiveInferenceAgent("scholar_1", "scholar", {})
        world_state = {"knowledge_obs": 4, "social_obs": 2, "market_obs": 1}

        obs = agent._get_observation(world_state)

        assert obs == [4, 2]

    def test_agent_get_observation_guardian(self):
        """Test observation extraction for guardian agent."""
        agent = ActiveInferenceAgent("guardian_1", "guardian", {})
        world_state = {"threat_obs": 1, "protection_obs": 3, "position_obs": 2}

        obs = agent._get_observation(world_state)

        assert obs == [1, 3]

    def test_agent_get_observation_error(self):
        """Test observation extraction with missing world state."""
        agent = ActiveInferenceAgent("test_agent", "explorer", {})

        obs = agent._get_observation({})

        assert obs == [0, 0]  # Default values

    @pytest.mark.asyncio
    async def test_agent_execute_action(self):
        """Test action execution."""
        agent = ActiveInferenceAgent("test_agent", "explorer", {})

        # Should not raise exception
        await agent._execute_action([1, 0], {"test": "world_state"})

    def test_agent_performance_metrics(self):
        """Test agent performance metrics calculation."""
        agent = ActiveInferenceAgent("test_agent", "explorer", {})
        agent.wealth = 150.5
        agent.knowledge_nodes = 8
        agent.goals_achieved = 3
        agent.total_goals = 10
        agent.alive = True

        metrics = agent.get_performance_metrics()

        assert metrics["wealth"] == 150.5
        assert metrics["knowledge_nodes"] == 8
        assert metrics["goal_achievement"] == 0.3
        assert metrics["alive"] == 1.0

    def test_agent_performance_metrics_no_goals(self):
        """Test performance metrics when total_goals is 0."""
        agent = ActiveInferenceAgent("test_agent", "explorer", {})
        agent.total_goals = 0
        agent.goals_achieved = 1

        metrics = agent.get_performance_metrics()

        # goals_achieved / max(total_goals, 1)
        assert metrics["goal_achievement"] == 1.0


class TestSimulationEngineInitialization:
    """Test SimulationEngine initialization functionality."""

    def test_simulation_engine_creation_default(self):
        """Test creating simulation engine with default config."""
        engine = SimulationEngine()

        assert isinstance(engine.config, SimulationConfig)
        assert engine.current_cycle == 0
        assert engine.running is False
        assert engine.agents == {}
        assert engine.start_time is None
        assert len(engine.cycle_times) == 0
        assert len(engine.event_history) == 0

    def test_simulation_engine_creation_with_config(self):
        """Test creating simulation engine with custom config."""
        config = {"max_cycles": 500, "time_step": 0.5, "agents": {"count": 5}}

        engine = SimulationEngine(config)

        assert engine.config.max_cycles == 500
        assert engine.config.time_step == 0.5
        assert engine.config.agents["count"] == 5

    def test_simulation_engine_creation_nested_simulation_config(self):
        """Test creating engine with nested simulation config."""
        config = {
            "simulation": {"max_cycles": 750, "time_step": 2.0},
            "world": {"resolution": 8},
            "agents": {"count": 15},
        }

        engine = SimulationEngine(config)

        assert engine.config.max_cycles == 750
        assert engine.config.time_step == 2.0
        assert engine.config.world["resolution"] == 8
        assert engine.config.agents["count"] == 15

    def test_simulation_engine_creation_with_simulation_config_object(self):
        """Test creating engine with SimulationConfig object."""
        config = SimulationConfig(max_cycles=300, time_step=1.5)

        engine = SimulationEngine(config)

        assert engine.config.max_cycles == 300
        assert engine.config.time_step == 1.5

    @patch("world.simulation.engine.H3World")
    @patch("world.simulation.engine.MessageSystem")
    def test_simulation_engine_initialize(self, mock_message_system, mock_h3_world):
        """Test simulation engine initialization."""
        engine = SimulationEngine()

        engine.initialize()

        # Should create world and message system
        mock_h3_world.assert_called_once()
        mock_message_system.assert_called_once()

        # Should create agents
        assert len(engine.agents) > 0
        assert engine.current_cycle == 0
        assert engine.running is False

    @patch("world.simulation.engine.H3World", None)
    @patch("world.simulation.engine.MessageSystem", None)
    def test_simulation_engine_initialize_no_dependencies(self):
        """Test initialization when dependencies are not available."""
        engine = SimulationEngine()

        engine.initialize()

        # Should still create agents even without world/message system
        assert len(engine.agents) > 0
        assert engine.world is None
        assert engine.message_system is None

    def test_simulation_engine_initialize_error_handling(self):
        """Test initialization error handling."""
        engine = SimulationEngine()

        # Mock world creation to fail
        with patch(
            "world.simulation.engine.H3World", side_effect=Exception("World creation failed")
        ):
            with pytest.raises(Exception, match="World creation failed"):
                engine.initialize()

    def test_create_agents_default_distribution(self):
        """Test agent creation with default distribution."""
        engine = SimulationEngine()
        engine._create_agents()

        assert len(engine.agents) == 10  # Default count

        # Check agent classes are distributed
        agent_classes = [agent.agent_class for agent in engine.agents.values()]
        assert "explorer" in agent_classes
        assert "merchant" in agent_classes
        assert "scholar" in agent_classes
        assert "guardian" in agent_classes

    def test_create_agents_custom_distribution(self):
        """Test agent creation with custom distribution."""
        config = SimulationConfig(
            agents={"count": 6, "distribution": {"explorer": 3, "merchant": 3}}
        )
        engine = SimulationEngine(config)
        engine._create_agents()

        assert len(engine.agents) == 6

        agent_classes = [agent.agent_class for agent in engine.agents.values()]
        assert agent_classes.count("explorer") >= 2  # Should be roughly half
        assert agent_classes.count("merchant") >= 2  # Should be roughly half

    def test_create_agents_zero_distribution(self):
        """Test agent creation with zero distribution."""
        config = SimulationConfig(agents={"count": 5, "distribution": {}})  # Empty distribution
        engine = SimulationEngine(config)
        engine._create_agents()

        assert len(engine.agents) == 5

        # Should default to explorers
        agent_classes = [agent.agent_class for agent in engine.agents.values()]
        assert all(cls == "explorer" for cls in agent_classes)

    def test_create_agents_partial_count(self):
        """Test agent creation when distribution doesn't sum to full count."""
        config = SimulationConfig(
            agents={
                "count": 10,
                # Only 4 specified, need 10
                "distribution": {"explorer": 2, "merchant": 2},
            }
        )
        engine = SimulationEngine(config)
        engine._create_agents()

        assert len(engine.agents) == 10

        # Should fill remaining with explorers
        agent_classes = [agent.agent_class for agent in engine.agents.values()]
        assert "explorer" in agent_classes
        assert "merchant" in agent_classes


class TestSimulationEngineExecution:
    """Test SimulationEngine execution functionality."""

    @pytest.mark.asyncio
    async def test_simulation_start(self):
        """Test simulation start."""
        engine = SimulationEngine()
        engine.initialize()

        await engine.start()

        assert engine.running is True
        assert engine.start_time is not None
        assert engine.current_cycle == 0

    @pytest.mark.asyncio
    async def test_simulation_stop(self):
        """Test simulation stop."""
        engine = SimulationEngine()
        engine.initialize()
        await engine.start()

        await engine.stop()

        assert engine.running is False

    @pytest.mark.asyncio
    async def test_simulation_step_basic(self):
        """Test basic simulation step."""
        engine = SimulationEngine()
        engine.initialize()
        await engine.start()

        initial_cycle = engine.current_cycle

        await engine.step()

        assert engine.current_cycle == initial_cycle + 1
        assert len(engine.cycle_times) > 0

    @pytest.mark.asyncio
    async def test_simulation_step_not_running(self):
        """Test simulation step when not running."""
        engine = SimulationEngine()
        engine.initialize()
        # Don't start

        initial_cycle = engine.current_cycle

        await engine.step()

        # Should not advance cycle
        assert engine.current_cycle == initial_cycle

    @pytest.mark.asyncio
    async def test_simulation_step_with_agents(self):
        """Test simulation step with agent updates."""
        engine = SimulationEngine()
        engine.initialize()
        await engine.start()

        # Mock agent update
        for agent in engine.agents.values():
            agent.update = AsyncMock()

        await engine.step()

        # All agents should have been updated
        for agent in engine.agents.values():
            agent.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_simulation_step_with_failed_agents(self):
        """Test simulation step with some failed agents."""
        engine = SimulationEngine()
        engine.initialize()
        await engine.start()

        # Mark some agents as failed
        agent_ids = list(engine.agents.keys())
        engine.failed_agents.add(agent_ids[0])

        # Mock agent update
        for agent in engine.agents.values():
            agent.update = AsyncMock()

        await engine.step()

        # Failed agent should not be updated
        failed_agent = engine.agents[agent_ids[0]]
        failed_agent.update.assert_not_called()

        # Other agents should be updated
        for agent_id, agent in engine.agents.items():
            if agent_id != agent_ids[0]:
                agent.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_simulation_step_agent_update_error(self):
        """Test simulation step when agent update returns None."""
        engine = SimulationEngine()
        engine.initialize()
        await engine.start()

        # Mock one agent to return None from update
        agent_ids = list(engine.agents.keys())
        engine.agents[agent_ids[0]].update = Mock(return_value=None)

        # Should not raise exception
        await engine.step()

        assert engine.current_cycle == 1

    @pytest.mark.asyncio
    async def test_simulation_step_error_handling(self):
        """Test simulation step error handling."""
        engine = SimulationEngine()
        engine.initialize()
        await engine.start()

        # Mock agent update to raise error
        for agent in engine.agents.values():
            agent.update = AsyncMock(side_effect=Exception("Agent update failed"))

        with pytest.raises(Exception, match="Agent update failed"):
            await engine.step()


class TestSimulationEngineWorldAndMessages:
    """Test SimulationEngine world and message processing."""

    @pytest.mark.asyncio
    async def test_async_world_update_no_world(self):
        """Test world update when no world is present."""
        engine = SimulationEngine()

        world_state = await engine._async_world_update()

        assert world_state == {}

    @pytest.mark.asyncio
    async def test_async_world_update_with_world(self):
        """Test world update with world present."""
        engine = SimulationEngine()
        engine.world = Mock()  # Mock world object

        world_state = await engine._async_world_update()

        # Should return simulated world state
        assert "position_obs" in world_state
        assert "resource_obs" in world_state
        assert "market_obs" in world_state
        assert "inventory_obs" in world_state
        assert "knowledge_obs" in world_state
        assert "social_obs" in world_state
        assert "threat_obs" in world_state
        assert "protection_obs" in world_state

    @pytest.mark.asyncio
    async def test_process_messages_no_system(self):
        """Test message processing when no message system."""
        engine = SimulationEngine()

        await engine._process_messages()

        # Should not raise exception
        assert len(engine.message_latencies) == 0

    @pytest.mark.asyncio
    async def test_process_messages_with_system_no_agents(self):
        """Test message processing with system but no agents."""
        engine = SimulationEngine()
        engine.message_system = Mock()

        await engine._process_messages()

        # Should add latency but no communication events
        assert len(engine.message_latencies) == 1
        assert len(engine.event_history) == 0

    @pytest.mark.asyncio
    async def test_process_messages_with_agents(self):
        """Test message processing with agents."""
        engine = SimulationEngine()
        engine.message_system = Mock()
        engine.initialize()

        await engine._process_messages()

        # Should add latency and potentially communication events
        assert len(engine.message_latencies) == 1

        # May have communication events if agents were selected
        communication_events = [e for e in engine.event_history if e.get("type") == "communication"]
        assert len(communication_events) <= 1


class TestSimulationEngineSocialNetworks:
    """Test SimulationEngine social network functionality."""

    def test_update_social_networks_no_agents(self):
        """Test social network update with no agents."""
        engine = SimulationEngine()

        engine._update_social_networks()

        assert len(engine.trade_relationships) == 0
        assert len(engine.knowledge_shares) == 0
        assert len(engine.protection_alliances) == 0

    @patch("numpy.random.random")
    @patch("numpy.random.choice")
    def test_update_social_networks_trade_relationships(self, mock_choice, mock_random):
        """Test trade relationship formation."""
        engine = SimulationEngine()
        engine.initialize()

        # Set up agents
        merchant_agent = ActiveInferenceAgent("merchant_1", "merchant", {})
        explorer_agent = ActiveInferenceAgent("explorer_1", "explorer", {})
        engine.agents = {"merchant_1": merchant_agent, "explorer_1": explorer_agent}

        # Mock random to trigger trade relationship
        mock_random.return_value = 0.05  # Less than 0.1 threshold
        mock_choice.side_effect = [merchant_agent, explorer_agent]

        engine._update_social_networks()

        # Should create trade relationship
        assert "explorer_1" in engine.trade_relationships["merchant_1"]
        assert "merchant_1" in engine.trade_relationships["explorer_1"]

    @patch("numpy.random.random")
    @patch("numpy.random.choice")
    @patch("numpy.random.randint")
    def test_update_social_networks_knowledge_sharing(self, mock_randint, mock_choice, mock_random):
        """Test knowledge sharing network formation."""
        engine = SimulationEngine()
        engine.initialize()

        # Set up agents
        scholar_agent = ActiveInferenceAgent("scholar_1", "scholar", {})
        explorer_agent = ActiveInferenceAgent("explorer_1", "explorer", {})
        merchant_agent = ActiveInferenceAgent("merchant_1", "merchant", {})
        engine.agents = {
            "scholar_1": scholar_agent,
            "explorer_1": explorer_agent,
            "merchant_1": merchant_agent,
        }

        # Mock random to trigger knowledge sharing
        mock_random.return_value = 0.2  # Less than 0.25 threshold
        mock_choice.side_effect = [scholar_agent, np.array([explorer_agent, merchant_agent])]
        mock_randint.return_value = 2  # Connect to 2 agents

        engine._update_social_networks()

        # Should create knowledge sharing connections
        assert len(engine.knowledge_shares["scholar_1"]) > 0

    @patch("numpy.random.random")
    @patch("numpy.random.choice")
    @patch("numpy.random.randint")
    def test_update_social_networks_protection_alliances(
        self, mock_randint, mock_choice, mock_random
    ):
        """Test protection alliance formation."""
        engine = SimulationEngine()
        engine.initialize()
        engine.current_cycle = 10

        # Set up agents
        guardian_agent = ActiveInferenceAgent("guardian_1", "guardian", {})
        explorer_agent = ActiveInferenceAgent("explorer_1", "explorer", {})
        engine.agents = {"guardian_1": guardian_agent, "explorer_1": explorer_agent}

        # Mock random to trigger protection alliance
        mock_random.return_value = 0.1  # Less than 0.15 threshold
        mock_choice.side_effect = [guardian_agent, np.array([explorer_agent])]
        mock_randint.return_value = 1  # Form 1 alliance

        engine._update_social_networks()

        # Should create protection alliance
        assert "explorer_1" in engine.protection_alliances["guardian_1"]

        # Should record alliance event
        alliance_events = [e for e in engine.event_history if e.get("type") == "alliance_formed"]
        assert len(alliance_events) > 0
        assert alliance_events[0]["guardian"] == "guardian_1"
        assert alliance_events[0]["protected"] == "explorer_1"
        assert alliance_events[0]["cycle"] == 10

    @patch("numpy.random.random")
    @patch("numpy.random.choice")
    def test_update_social_networks_trade_events(self, mock_choice, mock_random):
        """Test trade event recording."""
        engine = SimulationEngine()
        engine.initialize()
        engine.current_cycle = 5

        # Set up agents
        agent1 = ActiveInferenceAgent("agent_1", "explorer", {})
        agent2 = ActiveInferenceAgent("agent_2", "merchant", {})
        engine.agents = {"agent_1": agent1, "agent_2": agent2}

        # Mock random to trigger trade event
        # Only last call triggers trade event
        mock_random.side_effect = [0.5, 0.5, 0.5, 0.11]
        mock_choice.side_effect = [agent1, agent2]

        engine._update_social_networks()

        # Should record trade event
        trade_events = [e for e in engine.event_history if e.get("type") == "trade"]
        assert len(trade_events) == 1
        assert trade_events[0]["trader1"] == "agent_1"
        assert trade_events[0]["trader2"] == "agent_2"
        assert trade_events[0]["cycle"] == 5

    @patch("numpy.random.random")
    @patch("numpy.random.choice")
    def test_update_social_networks_resource_sharing(self, mock_choice, mock_random):
        """Test resource sharing event recording."""
        engine = SimulationEngine()
        engine.initialize()
        engine.current_cycle = 8

        # Set up agents
        agent1 = ActiveInferenceAgent("agent_1", "explorer", {})
        agent2 = ActiveInferenceAgent("agent_2", "scholar", {})
        engine.agents = {"agent_1": agent1, "agent_2": agent2}

        # Mock random to trigger resource sharing
        # Only last call triggers sharing
        mock_random.side_effect = [0.5, 0.5, 0.5, 0.5, 0.07]
        mock_choice.side_effect = [agent1, agent2]

        engine._update_social_networks()

        # Should record resource sharing event
        sharing_events = [e for e in engine.event_history if e.get("type") == "resource_share"]
        assert len(sharing_events) == 1
        assert sharing_events[0]["sharer"] == "agent_1"
        assert sharing_events[0]["receiver"] == "agent_2"
        assert sharing_events[0]["cycle"] == 8

    def test_record_cycle_events(self):
        """Test cycle event recording."""
        engine = SimulationEngine()
        engine.current_cycle = 15

        # Set up some relationships
        engine.trade_relationships["agent_1"].add("agent_2")
        engine.knowledge_shares["scholar_1"].add("agent_3")
        engine.protection_alliances["guardian_1"].add("agent_4")

        engine._record_cycle_events()

        # Should record events for each type
        trade_events = [e for e in engine.event_history if e.get("type") == "trade"]
        knowledge_events = [e for e in engine.event_history if e.get("type") == "knowledge_share"]
        alliance_events = [e for e in engine.event_history if e.get("type") == "alliance_formed"]

        assert len(trade_events) == 1
        assert len(knowledge_events) == 1
        assert len(alliance_events) == 1

        assert trade_events[0]["cycle"] == 15
        assert knowledge_events[0]["cycle"] == 15
        assert alliance_events[0]["cycle"] == 15

    def test_record_communication(self):
        """Test communication event recording."""
        engine = SimulationEngine()
        engine.current_cycle = 20

        engine._record_communication("sender_1", "receiver_1")

        communication_events = [e for e in engine.event_history if e.get("type") == "communication"]
        assert len(communication_events) == 1
        assert communication_events[0]["sender"] == "sender_1"
        assert communication_events[0]["receiver"] == "receiver_1"
        assert communication_events[0]["cycle"] == 20
        assert "timestamp" in communication_events[0]


class TestSimulationEngineSystemHealth:
    """Test SimulationEngine system health and monitoring functionality."""

    @pytest.mark.asyncio
    async def test_get_system_health_no_agents(self):
        """Test system health with no agents."""
        engine = SimulationEngine()

        health = await engine.get_system_health()

        assert isinstance(health, dict)
        assert health["agent_count"] == 0
        assert health["message_queue_size"] == 0
        assert health["memory_usage_mb"] > 0  # Should track actual memory
        assert health["cpu_usage_percent"] >= 0
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_get_system_health_with_agents(self):
        """Test system health with agents present."""
        engine = SimulationEngine()
        engine.initialize()

        health = await engine.get_system_health()

        assert health["agent_count"] == len(engine.agents) - len(engine.failed_agents)
        assert health["agent_count"] >= 0
        assert health["status"] in ["healthy", "degraded", "critical"]

    @pytest.mark.asyncio
    async def test_get_system_health_with_failed_agents(self):
        """Test system health with failed agents."""
        engine = SimulationEngine()
        engine.initialize()

        # Add failed agents - need more than 50% to trigger error
        agent_ids = list(engine.agents.keys())
        majority = len(agent_ids) // 2 + 1
        for i in range(majority):
            if i < len(agent_ids):
                engine.failed_agents.add(agent_ids[i])

        health = await engine.get_system_health()

        assert len(health["errors"]) > 0
        assert any("High agent failure rate" in error for error in health["errors"])
        assert health["status"] == "critical"

    @pytest.mark.asyncio
    async def test_get_system_health_high_memory(self):
        """Test system health status calculation with high memory usage."""
        engine = SimulationEngine()

        # Mock the process to return high memory usage
        with patch.object(engine.process, "memory_info") as mock_memory:
            mock_memory.return_value.rss = 3000 * 1024 * 1024  # 3GB > 2GB limit

            health = await engine.get_system_health()

            assert health["memory_usage_mb"] > 2048  # Above threshold
            assert health["status"] == "degraded"
            assert any("High memory usage" in error for error in health["errors"])

    @pytest.mark.asyncio
    async def test_get_system_health_slow_cycle(self):
        """Test system health with slow cycle times."""
        engine = SimulationEngine()
        engine.cycle_times.append(6.0)  # Above 5.0 threshold

        health = await engine.get_system_health()

        assert health["last_cycle_time"] > 5.0
        assert len(health["errors"]) > 0
        assert any("Slow cycle" in error for error in health["errors"])
        assert health["status"] == "degraded"


class TestSimulationEngineEcosystemMetrics:
    """Test SimulationEngine ecosystem metrics functionality."""

    @pytest.mark.asyncio
    async def test_get_ecosystem_metrics_no_agents(self):
        """Test ecosystem metrics with no agents."""
        engine = SimulationEngine()

        metrics = await engine.get_ecosystem_metrics()

        assert isinstance(metrics, dict)
        assert metrics["resource_gini_coefficient"] == 0
        assert metrics["average_agent_wealth"] == 0
        assert metrics["knowledge_nodes_per_agent"] == 0
        assert metrics["trades_this_cycle"] == 0
        assert metrics["explored_cells_percentage"] == 0
        assert metrics["behavior_entropy"] == 0
        assert metrics["average_goal_achievement"] == 0

    @pytest.mark.asyncio
    async def test_get_ecosystem_metrics_with_agents(self):
        """Test ecosystem metrics with agents."""
        engine = SimulationEngine()
        engine.initialize()

        # Set some agent properties for testing
        for agent in engine.agents.values():
            agent.wealth = 100.0
            agent.knowledge_nodes = 5
            agent.goals_achieved = 2
            agent.total_goals = 10

        metrics = await engine.get_ecosystem_metrics()

        assert metrics["average_agent_wealth"] == 100.0
        assert metrics["knowledge_nodes_per_agent"] == 5.0
        assert metrics["average_goal_achievement"] == 0.2

    @pytest.mark.asyncio
    async def test_get_ecosystem_metrics_wealth_distribution(self):
        """Test Gini coefficient calculation for wealth distribution."""
        engine = SimulationEngine()
        engine.initialize()

        # Create unequal wealth distribution
        agent_list = list(engine.agents.values())
        if len(agent_list) >= 3:
            agent_list[0].wealth = 1000.0  # Rich agent
            agent_list[1].wealth = 100.0  # Middle agent
            agent_list[2].wealth = 10.0  # Poor agent

        metrics = await engine.get_ecosystem_metrics()

        # With unequal distribution, Gini should be > 0
        assert metrics["resource_gini_coefficient"] > 0.0
        assert metrics["resource_gini_coefficient"] <= 1.0

    @pytest.mark.asyncio
    async def test_get_ecosystem_metrics_behavior_entropy(self):
        """Test behavior entropy calculation."""
        engine = SimulationEngine()
        engine.initialize()

        metrics = await engine.get_ecosystem_metrics()

        # With varied agent classes, entropy should be > 0
        assert metrics["behavior_entropy"] >= 0.0

    @pytest.mark.asyncio
    async def test_get_ecosystem_metrics_trades_counting(self):
        """Test trade counting in ecosystem metrics."""
        engine = SimulationEngine()
        engine.current_cycle = 5

        # Add trade events for recent cycles
        engine.event_history = [
            {"type": "trade", "cycle": 5, "trader1": "agent1", "trader2": "agent2"},
            {"type": "trade", "cycle": 4, "trader1": "agent3", "trader2": "agent4"},
            {"type": "trade", "cycle": 1, "trader1": "agent1", "trader2": "agent3"},  # Old cycle
        ]

        metrics = await engine.get_ecosystem_metrics()

        # Should count trades from recent cycles (within 5 cycles)
        assert metrics["trades_this_cycle"] >= 2


class TestSimulationEngineSocialNetworkAnalysis:
    """Test SimulationEngine social network analysis functionality."""

    @pytest.mark.asyncio
    async def test_get_social_network_no_agents(self):
        """Test social network analysis with no agents."""
        engine = SimulationEngine()

        network = await engine.get_social_network()

        assert isinstance(network, SocialNetwork)
        assert len(network.trade_clusters) == 0
        assert len(network.centrality_scores) == 0
        assert len(network.knowledge_sharing_network) == 0
        assert len(network.protection_alliances) == 0

    @pytest.mark.asyncio
    async def test_get_social_network_with_relationships(self):
        """Test social network with established relationships."""
        engine = SimulationEngine()
        engine.initialize()

        # Set up trade relationships
        agent_ids = list(engine.agents.keys())
        if len(agent_ids) >= 2:
            engine.trade_relationships[agent_ids[0]].add(agent_ids[1])
            engine.trade_relationships[agent_ids[1]].add(agent_ids[0])

        # Set up knowledge sharing
        if len(agent_ids) >= 2:
            engine.knowledge_shares[agent_ids[0]].add(agent_ids[1])

        # Set up protection alliances
        if len(agent_ids) >= 2:
            engine.protection_alliances[agent_ids[0]].add(agent_ids[1])

        network = await engine.get_social_network()

        assert len(network.centrality_scores) > 0
        assert isinstance(network.knowledge_sharing_network, dict)
        assert isinstance(network.protection_alliances, list)

    @pytest.mark.asyncio
    async def test_get_social_network_centrality_calculation(self):
        """Test centrality score calculation."""
        engine = SimulationEngine()
        engine.initialize()

        # Create a star network with first agent at center
        agent_ids = list(engine.agents.keys())
        if len(agent_ids) >= 3:
            central_agent = agent_ids[0]
            for other_agent in agent_ids[1:]:
                engine.trade_relationships[central_agent].add(other_agent)
                engine.trade_relationships[other_agent].add(central_agent)

        network = await engine.get_social_network()

        if len(agent_ids) >= 3:
            # Central agent should have higher centrality
            central_centrality = network.centrality_scores.get(central_agent, 0)
            other_centralities = [network.centrality_scores.get(aid, 0) for aid in agent_ids[1:]]

            assert central_centrality >= max(other_centralities, default=0)

    @pytest.mark.asyncio
    async def test_get_social_network_trade_clusters(self):
        """Test trade cluster identification."""
        engine = SimulationEngine()
        engine.initialize()

        agent_ids = list(engine.agents.keys())
        if len(agent_ids) >= 4:
            # Create two separate trade clusters
            engine.trade_relationships[agent_ids[0]].add(agent_ids[1])
            engine.trade_relationships[agent_ids[1]].add(agent_ids[0])

            engine.trade_relationships[agent_ids[2]].add(agent_ids[3])
            engine.trade_relationships[agent_ids[3]].add(agent_ids[2])

        network = await engine.get_social_network()

        if len(agent_ids) >= 4:
            # Should identify separate clusters
            assert len(network.trade_clusters) >= 1


class TestSimulationEngineExportCapabilities:
    """Test SimulationEngine export and edge deployment capabilities."""

    @pytest.mark.asyncio
    async def test_export_agent_basic(self):
        """Test basic agent export."""
        engine = SimulationEngine()
        engine.initialize()

        agent_ids = list(engine.agents.keys())
        if agent_ids:
            with tempfile.TemporaryDirectory() as temp_dir:
                export_path = Path(temp_dir) / "agent_export"

                success = await engine.export_agent(agent_ids[0], export_path)

                assert success is True
                assert export_path.exists()

                # Check for expected files
                config_file = export_path / "agent_config.json"
                gnn_file = export_path / "gnn_model.json"
                requirements_file = export_path / "requirements.txt"
                readme_file = export_path / "README.md"

                assert config_file.exists()
                assert gnn_file.exists()
                assert requirements_file.exists()
                assert readme_file.exists()

    @pytest.mark.asyncio
    async def test_export_agent_with_config(self):
        """Test agent export with proper config structure."""
        engine = SimulationEngine()
        engine.initialize()

        agent_ids = list(engine.agents.keys())
        if agent_ids:
            with tempfile.TemporaryDirectory() as temp_dir:
                export_path = Path(temp_dir) / "agent_export"

                success = await engine.export_agent(agent_ids[0], export_path)

                assert success is True

                # Read and verify agent config
                config_file = export_path / "agent_config.json"
                with open(config_file) as f:
                    config_data = json.load(f)

                assert "agent_id" in config_data
                assert "agent_class" in config_data
                assert "personality" in config_data
                assert "configuration" in config_data

    @pytest.mark.asyncio
    async def test_export_agent_nonexistent(self):
        """Test export of non-existent agent."""
        engine = SimulationEngine()

        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir) / "agent_export"

            success = await engine.export_agent("nonexistent_agent", export_path)

            assert success is False

    @pytest.mark.asyncio
    async def test_export_deployment_basic(self):
        """Test basic deployment export."""
        engine = SimulationEngine()
        engine.initialize()

        with tempfile.TemporaryDirectory() as temp_dir:
            deployment_path = Path(temp_dir) / "deployment"

            success = await engine.export_deployment(deployment_path)

            assert success is True
            assert deployment_path.exists()

            # Check for expected structure
            assert (deployment_path / "deployment.json").exists()
            assert (deployment_path / "agents").exists()
            assert (deployment_path / "world").exists()
            assert (deployment_path / "configs").exists()

    @pytest.mark.asyncio
    async def test_export_deployment_with_agents(self):
        """Test deployment export includes all agents."""
        engine = SimulationEngine()
        engine.initialize()

        with tempfile.TemporaryDirectory() as temp_dir:
            deployment_path = Path(temp_dir) / "deployment"

            success = await engine.export_deployment(deployment_path)

            assert success is True

            # Check agent exports
            agents_dir = deployment_path / "agents"
            exported_agents = list(agents_dir.iterdir())

            # Should have one directory per agent
            assert len(exported_agents) == len(engine.agents)

    @pytest.mark.asyncio
    async def test_export_deployment_config_structure(self):
        """Test deployment config structure."""
        config = SimulationConfig(max_cycles=500, time_step=0.5)
        engine = SimulationEngine(config)
        engine.initialize()

        with tempfile.TemporaryDirectory() as temp_dir:
            deployment_path = Path(temp_dir) / "deployment"

            success = await engine.export_deployment(deployment_path)

            assert success is True

            # Read and verify deployment config
            config_file = deployment_path / "deployment.json"
            with open(config_file) as f:
                config_data = json.load(f)

            assert "simulation_config" in config_data
            assert "export_time" in config_data
            assert "agent_count" in config_data
            assert "current_cycle" in config_data

            assert config_data["simulation_config"]["max_cycles"] == 500
            assert config_data["simulation_config"]["time_step"] == 0.5


class TestSimulationEngineFaultTolerance:
    """Test SimulationEngine fault tolerance functionality."""

    @pytest.mark.asyncio
    async def test_simulate_agent_failure(self):
        """Test simulating agent failures."""
        engine = SimulationEngine()
        engine.initialize()

        agent_ids = list(engine.agents.keys())
        failing_agent_id = agent_ids[0]

        # Simulate agent failure
        await engine.simulate_agent_failure(failing_agent_id)

        assert failing_agent_id in engine.failed_agents
        assert engine.agents[failing_agent_id].alive is False

        # Check if failure affects system health
        health = await engine.get_system_health()
        assert health["agent_count"] == len(engine.agents) - len(engine.failed_agents)

    @pytest.mark.asyncio
    async def test_simulate_agent_failure_nonexistent(self):
        """Test simulating failure for non-existent agent."""
        engine = SimulationEngine()

        # Should not raise exception
        await engine.simulate_agent_failure("nonexistent_agent")

        # Should not add to failed_agents since agent doesn't exist
        assert "nonexistent_agent" not in engine.failed_agents

    @pytest.mark.asyncio
    async def test_simulate_communication_failure(self):
        """Test simulating communication failures."""
        engine = SimulationEngine()
        engine.current_cycle = 10

        await engine.simulate_communication_failure(5)

        assert len(engine.communication_failures) == 1
        failure = engine.communication_failures[0]
        assert failure["start_cycle"] == 10
        assert failure["duration"] == 5
        assert failure["type"] == "communication"

    @pytest.mark.asyncio
    async def test_simulate_resource_depletion(self):
        """Test simulating resource depletion."""
        engine = SimulationEngine()

        await engine.simulate_resource_depletion(0.3)

        assert engine.environmental_conditions["resource_multiplier"] == 0.7

    def test_survival_rate_calculation(self):
        """Test survival rate calculation."""
        engine = SimulationEngine()
        engine.initialize()

        initial_rate = engine.get_survival_rate()
        assert initial_rate == 1.0  # All agents alive initially

        # Fail some agents
        agent_ids = list(engine.agents.keys())
        if len(agent_ids) >= 2:
            engine.failed_agents.add(agent_ids[0])
            new_rate = engine.get_survival_rate()
            assert new_rate < 1.0

    def test_is_healthy_calculation(self):
        """Test system health status calculation."""
        engine = SimulationEngine()
        engine.initialize()

        assert engine.is_healthy() is True  # Initially healthy

        # Fail majority of agents
        agent_ids = list(engine.agents.keys())
        majority = len(agent_ids) // 2 + 1
        for i in range(majority):
            if i < len(agent_ids):
                engine.failed_agents.add(agent_ids[i])

        assert engine.is_healthy() is False  # Now unhealthy

    @pytest.mark.asyncio
    async def test_get_communication_health(self):
        """Test communication health monitoring."""
        engine = SimulationEngine()
        engine.current_cycle = 10

        # Initially healthy
        health = await engine.get_communication_health()
        assert health["status"] == "healthy"

        # Simulate failure
        await engine.simulate_communication_failure(5)
        health = await engine.get_communication_health()
        assert health["status"] == "failed"

        # Move past failure duration
        engine.current_cycle = 20
        health = await engine.get_communication_health()
        assert health["status"] == "recovered"

    @pytest.mark.asyncio
    async def test_environmental_adaptation(self):
        """Test environmental condition changes and adaptation."""
        engine = SimulationEngine()
        engine.initialize()
        engine.current_cycle = 5

        # Set harsh environmental conditions
        conditions = {"resource_multiplier": 0.5, "hazard_level": 0.6}
        await engine.set_environmental_conditions(conditions)

        assert engine.environmental_conditions["resource_multiplier"] == 0.5
        assert engine.environmental_conditions["hazard_level"] == 0.6

        # Check adaptation history
        assert hasattr(engine, "_env_change_history")
        assert len(engine._env_change_history) == 1

        change = engine._env_change_history[0]
        assert change["cycle"] == 5
        assert "previous_conditions" in change
        assert "new_conditions" in change


class TestSimulationEngineUtilityMethods:
    """Test SimulationEngine utility and state methods."""

    def test_get_agents(self):
        """Test getting list of all agents."""
        engine = SimulationEngine()
        engine.initialize()

        agents = engine.get_agents()

        assert isinstance(agents, list)
        assert len(agents) == len(engine.agents)

        for agent in agents:
            assert hasattr(agent, "agent_id")
            assert hasattr(agent, "agent_class")

    def test_get_agent(self):
        """Test getting specific agent by ID."""
        engine = SimulationEngine()
        engine.initialize()

        agent_ids = list(engine.agents.keys())
        if agent_ids:
            agent = engine.get_agent(agent_ids[0])
            assert agent is not None
            assert agent.agent_id == agent_ids[0]

        # Test non-existent agent
        agent = engine.get_agent("nonexistent")
        assert agent is None

    def test_get_agent_count(self):
        """Test agent count calculation."""
        engine = SimulationEngine()
        engine.initialize()

        initial_count = engine.get_agent_count()
        assert initial_count == len(engine.agents)

        # Fail an agent
        agent_ids = list(engine.agents.keys())
        if agent_ids:
            engine.failed_agents.add(agent_ids[0])
            new_count = engine.get_agent_count()
            assert new_count == initial_count - 1

    def test_get_alive_agent_count(self):
        """Test alive agent count calculation."""
        engine = SimulationEngine()
        engine.initialize()

        initial_count = engine.get_alive_agent_count()
        assert initial_count > 0

        # Kill an agent
        agent_ids = list(engine.agents.keys())
        if agent_ids:
            engine.agents[agent_ids[0]].alive = False
            new_count = engine.get_alive_agent_count()
            assert new_count == initial_count - 1

    def test_get_statistics(self):
        """Test simulation statistics."""
        engine = SimulationEngine()
        engine.initialize()
        engine.current_cycle = 10

        # Add some events
        engine.event_history.extend(
            [
                {"type": "communication", "cycle": 5},
                {"type": "trade", "cycle": 8},
                {"type": "communication", "cycle": 9},
            ]
        )

        stats = engine.get_statistics()

        assert "cycles_completed" in stats
        assert "agents_alive" in stats
        assert "total_messages" in stats
        assert "total_trades" in stats
        assert "knowledge_nodes_created" in stats

        assert stats["cycles_completed"] == 10
        assert stats["total_messages"] == 2  # 2 communication events

    def test_get_event_history(self):
        """Test event history retrieval."""
        engine = SimulationEngine()

        # Add some events
        events = [{"type": "test", "cycle": 1}, {"type": "test", "cycle": 2}]
        engine.event_history.extend(events)

        history = engine.get_event_history()

        assert isinstance(history, list)
        assert len(history) == 2
        assert history[0]["type"] == "test"

    def test_get_events_this_cycle(self):
        """Test getting events from current cycle."""
        engine = SimulationEngine()
        engine.current_cycle = 5

        # Add events from different cycles
        engine.event_history.extend(
            [
                {"type": "test", "cycle": 4},  # Previous cycle
                {"type": "test", "cycle": 3},  # Earlier cycle
                {"type": "test", "cycle": 5},  # Future cycle
            ]
        )

        current_events = engine.get_events_this_cycle()

        # Should return events from completed cycle (4)
        assert len(current_events) == 1
        assert current_events[0]["cycle"] == 4

    @pytest.mark.asyncio
    async def test_get_state_snapshot(self):
        """Test state snapshot generation."""
        engine = SimulationEngine()
        engine.initialize()
        await engine.start()
        engine.current_cycle = 15  # Set after start

        snapshot = await engine.get_state_snapshot()

        assert "cycle" in snapshot
        assert "agents" in snapshot
        assert "alive_agents" in snapshot
        assert "failed_agents" in snapshot
        assert "running" in snapshot

        assert snapshot["cycle"] == 15
        assert snapshot["running"] is True

    @pytest.mark.asyncio
    async def test_get_average_agent_performance(self):
        """Test average agent performance calculation."""
        engine = SimulationEngine()
        engine.initialize()

        # Set some performance metrics
        for agent in engine.agents.values():
            agent.goals_achieved = 5
            agent.total_goals = 10

        performance = await engine.get_average_agent_performance()

        assert isinstance(performance, float)
        assert performance > 0.0
        assert performance <= 1.0  # Performance should be normalized

    @pytest.mark.asyncio
    async def test_get_adaptation_metrics(self):
        """Test adaptation metrics calculation."""
        engine = SimulationEngine()
        engine.initialize()

        metrics = await engine.get_adaptation_metrics()

        assert "total_knowledge_nodes" in metrics
        assert "behavior_entropy" in metrics
        assert "average_goal_achievement" in metrics

        assert isinstance(metrics["total_knowledge_nodes"], (int, float))
        assert isinstance(metrics["behavior_entropy"], float)
        assert isinstance(metrics["average_goal_achievement"], float)

    @pytest.mark.asyncio
    async def test_get_message_system_stats(self):
        """Test message system statistics."""
        engine = SimulationEngine()
        engine.message_latencies = [10, 20, 15, 25]
        engine.event_history = [{"type": "message"}, {"type": "trade"}]

        stats = await engine.get_message_system_stats()

        assert "dropped_count" in stats
        assert "avg_delay" in stats
        assert "queue_size" in stats

        assert stats["avg_delay"] == 17.5  # Average of latencies
        assert stats["queue_size"] == 2  # Length of event history


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
