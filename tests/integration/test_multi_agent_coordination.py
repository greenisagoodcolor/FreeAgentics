"""Comprehensive integration tests for multi-agent coordination scenarios."""

import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agents.base_agent import PYMDP_AVAILABLE, BasicExplorerAgent
from agents.coalition_coordinator import CoalitionCoordinatorAgent
from agents.resource_collector import ResourceCollectorAgent
from database.models import Agent as AgentModel
from database.models import Coalition
from database.session import get_db, init_db
from knowledge_graph.graph_engine import KnowledgeGraph


@pytest.fixture
def test_db():
    """Set up test database."""
    # Use existing database session
    db_gen = get_db()
    session = next(db_gen)
    yield session
    session.close()


@pytest.fixture
def knowledge_graph():
    """Create knowledge graph for tests."""
    return KnowledgeGraph()


@pytest.mark.asyncio
class TestMultiAgentCoordination:
    """Test real multi-agent coordination scenarios."""

    async def test_agent_creation_and_activation(self):
        """Test that different agent types can be created and activated properly."""
        # Create different types of agents
        explorer = BasicExplorerAgent("explorer_1", "Explorer 1", grid_size=5)
        collector = ResourceCollectorAgent(
            "collector_1", "Collector 1", grid_size=5
        )
        coordinator = CoalitionCoordinatorAgent(
            "coord_1", "Coordinator 1", max_agents=5
        )

        agents = [explorer, collector, coordinator]

        # Start all agents
        for agent in agents:
            agent.start()

        # Verify all agents are active
        for agent in agents:
            assert agent.is_active
            assert hasattr(agent, "error_handler")

        # Verify agent types have expected capabilities
        assert hasattr(explorer, "action_map")
        assert hasattr(collector, "resource_memory")
        assert hasattr(coordinator, "known_agents")

    async def test_basic_multi_agent_simulation(self):
        """Test basic multi-agent simulation with real PyMDP operations."""
        # Create a small multi-agent system
        agents = [
            BasicExplorerAgent(f"explorer_{i}", f"Explorer {i}", grid_size=3)
            for i in range(3)
        ]

        # Start all agents
        for agent in agents:
            agent.start()

        # Run simulation steps
        observations = [
            {"position": [i % 3, i // 3], "surroundings": np.zeros((3, 3))}
            for i in range(3)
        ]

        # Execute coordinated steps
        actions = []
        for step in range(5):
            step_actions = []
            for i, agent in enumerate(agents):
                obs = observations[i].copy()
                obs["step"] = step
                action = agent.step(obs)
                step_actions.append(action)

                # Verify action is valid
                assert action in agent.actions

            actions.append(step_actions)

        # Verify all agents completed steps
        for agent in agents:
            assert agent.total_steps == 5
            assert agent.metrics["total_actions"] == 5
            assert agent.metrics["total_observations"] == 5

    async def test_resource_collection_coordination(self):
        """Test coordination between resource collectors and coordinators."""
        # Create resource collection scenario
        collectors = [
            ResourceCollectorAgent(
                f"collector_{i}", f"Collector {i}", grid_size=5
            )
            for i in range(2)
        ]
        coordinator = CoalitionCoordinatorAgent(
            "coord_main", "Main Coordinator", max_agents=5
        )

        # Start all agents
        all_agents = collectors + [coordinator]
        for agent in all_agents:
            agent.start()

        # Simulate resource discovery scenario
        resource_observations = [
            {
                "position": [1, 1],
                "visible_cells": [
                    {
                        "x": 1,
                        "y": 1,
                        "type": "resource",
                        "resource_type": "energy",
                        "amount": 10,
                    }
                ],
                "current_load": 0,
                "visible_agents": [
                    {
                        "id": "collector_1",
                        "position": [2, 1],
                        "status": "active",
                    },
                    {
                        "id": "coord_main",
                        "position": [1, 2],
                        "status": "active",
                    },
                ],
            },
            {
                "position": [2, 1],
                "visible_cells": [{"x": 2, "y": 1, "type": "empty"}],
                "current_load": 0,
                "visible_agents": [
                    {
                        "id": "collector_0",
                        "position": [1, 1],
                        "status": "active",
                    },
                    {
                        "id": "coord_main",
                        "position": [1, 2],
                        "status": "active",
                    },
                ],
            },
        ]

        coord_observation = {
            "position": [1, 2],
            "visible_agents": [
                {
                    "id": "collector_0",
                    "position": [1, 1],
                    "status": "active",
                    "capabilities": ["collect"],
                },
                {
                    "id": "collector_1",
                    "position": [2, 1],
                    "status": "active",
                    "capabilities": ["collect"],
                },
            ],
            "coalition_status": {},
        }

        # Execute coordination step
        collector_actions = []
        for i, collector in enumerate(collectors):
            action = collector.step(resource_observations[i])
            collector_actions.append(action)

        coord_action = coordinator.step(coord_observation)

        # Verify reasonable actions
        assert all(
            action
            in ["up", "down", "left", "right", "collect", "return_to_base"]
            for action in collector_actions
        )
        assert coord_action in [
            "invite",
            "exclude",
            "merge",
            "split",
            "coordinate",
            "dissolve",
        ]

        # Verify resource memory is updated
        assert len(collectors[0].resource_memory) > 0

    async def test_knowledge_graph_integration(self, knowledge_graph):
        """Test integration with knowledge graph during multi-agent operations."""
        # Create agents that will generate knowledge
        agents = [
            BasicExplorerAgent(f"kg_agent_{i}", f"KG Agent {i}", grid_size=3)
            for i in range(2)
        ]

        # Start agents
        for agent in agents:
            agent.start()

        # Create knowledge nodes for agent interactions
        for i, agent in enumerate(agents):
            node = knowledge_graph.create_node(
                node_type="agent",
                content={
                    "agent_id": agent.agent_id,
                    "type": "explorer",
                    "position": [i, 0],
                },
                metadata={
                    "created_by": "integration_test",
                    "timestamp": datetime.now().isoformat(),
                },
            )

        # Create interaction edges
        nodes = list(knowledge_graph.nodes.values())
        if len(nodes) >= 2:
            edge = knowledge_graph.create_edge(
                from_node_id=nodes[0].id,
                to_node_id=nodes[1].id,
                edge_type="coordination",
                properties={
                    "interaction_type": "resource_sharing",
                    "strength": 0.8,
                },
            )

        # Simulate agent actions that update knowledge graph
        for step in range(3):
            for i, agent in enumerate(agents):
                observation = {
                    "position": [i, step],
                    "surroundings": np.random.randint(0, 2, (3, 3)),
                    "visible_agents": (
                        [{"id": f"kg_agent_{1-i}", "position": [1 - i, step]}]
                        if step > 0
                        else []
                    ),
                }

                action = agent.step(observation)

                # Update knowledge graph with new information
                if observation.get("visible_agents"):
                    knowledge_graph.update_node(
                        nodes[i].id,
                        content={
                            "last_action": action,
                            "step": step,
                            "observed_agents": len(
                                observation["visible_agents"]
                            ),
                        },
                        metadata={"last_updated": datetime.now().isoformat()},
                    )

        # Verify knowledge graph was updated
        assert len(knowledge_graph.nodes) == 2
        assert len(knowledge_graph.edges) == 1

        # Check that nodes have been updated
        updated_nodes = [
            node
            for node in knowledge_graph.nodes.values()
            if "last_action" in node.content
        ]
        assert len(updated_nodes) == 2

    async def test_coalition_formation_scenario(self):
        """Test realistic coalition formation scenario."""
        # Create mixed agent types for coalition formation
        explorers = [
            BasicExplorerAgent(f"exp_{i}", f"Explorer {i}", grid_size=4)
            for i in range(2)
        ]
        collectors = [
            ResourceCollectorAgent(f"col_{i}", f"Collector {i}", grid_size=4)
            for i in range(2)
        ]
        coordinator = CoalitionCoordinatorAgent(
            "coalition_coord", "Coalition Coordinator", max_agents=6
        )

        all_agents = explorers + collectors + [coordinator]

        # Start all agents
        for agent in all_agents:
            agent.start()

        # Simulate coalition formation scenario
        # Step 1: Agents discover each other
        visibility_map = {
            "exp_0": ["exp_1", "col_0"],
            "exp_1": ["exp_0", "col_1"],
            "col_0": ["exp_0", "coalition_coord"],
            "col_1": ["exp_1", "coalition_coord"],
            "coalition_coord": ["col_0", "col_1"],
        }

        coordination_steps = []

        for step in range(3):
            step_actions = {}

            # Each agent acts based on what they can see
            for agent in all_agents:
                visible_agents = []
                if agent.agent_id in visibility_map:
                    for visible_id in visibility_map[agent.agent_id]:
                        visible_agents.append(
                            {
                                "id": visible_id,
                                "position": [step, step % 2],
                                "status": "active",
                                "capabilities": ["explore"]
                                if "exp" in visible_id
                                else ["collect"],
                            }
                        )

                if isinstance(agent, CoalitionCoordinatorAgent):
                    observation = {
                        "visible_agents": visible_agents,
                        "coalition_status": {},
                    }
                else:
                    observation = {
                        "position": [step % 4, (step + 1) % 4],
                        "surroundings": np.zeros((3, 3)),
                    }
                    if visible_agents:
                        observation["visible_agents"] = visible_agents

                action = agent.step(observation)
                step_actions[agent.agent_id] = action

            coordination_steps.append(step_actions)

        # Verify coordination occurred
        assert len(coordination_steps) == 3

        # Check that coordinator took meaningful actions
        coord_actions = [
            step["coalition_coord"] for step in coordination_steps
        ]
        assert all(
            action
            in [
                "invite",
                "exclude",
                "merge",
                "split",
                "coordinate",
                "dissolve",
            ]
            for action in coord_actions
        )

        # Verify agents have knowledge of each other
        assert len(coordinator.known_agents) > 0

    async def test_error_resilience_in_coordination(self):
        """Test that multi-agent coordination is resilient to individual agent failures."""
        # Create agents with some that will fail
        reliable_agents = [
            BasicExplorerAgent(f"reliable_{i}", f"Reliable {i}", grid_size=3)
            for i in range(2)
        ]
        failing_agent = BasicExplorerAgent(
            "failing_agent", "Failing Agent", grid_size=3
        )

        # Start all agents
        all_agents = reliable_agents + [failing_agent]
        for agent in all_agents:
            agent.start()

        # Make one agent fail consistently
        mock_pymdp = MagicMock()
        mock_pymdp.infer_policies.side_effect = Exception(
            "Simulated PyMDP failure"
        )
        failing_agent.pymdp_agent = mock_pymdp

        # Run coordination with mixed success/failure
        coordination_results = []

        for step in range(5):
            step_results = {}

            for agent in all_agents:
                observation = {
                    "position": [step % 3, (step + 1) % 3],
                    "surroundings": np.zeros((3, 3)),
                    "step": step,
                }

                try:
                    action = agent.step(observation)
                    step_results[agent.agent_id] = {
                        "action": action,
                        "status": "success",
                    }
                except Exception as e:
                    step_results[agent.agent_id] = {
                        "action": "stay",
                        "status": "failed",
                        "error": str(e),
                    }

            coordination_results.append(step_results)

        # Verify that reliable agents continued working
        for step_result in coordination_results:
            for reliable_agent in reliable_agents:
                assert (
                    step_result[reliable_agent.agent_id]["status"] == "success"
                )

        # Verify that failing agent used fallbacks (didn't crash the system)
        failing_results = [
            result["failing_agent"] for result in coordination_results
        ]
        assert all(
            result["action"] in ["up", "down", "left", "right", "stay"]
            for result in failing_results
        )

        # Check error tracking
        assert len(failing_agent.error_handler.error_history) > 0

    async def test_performance_under_load(self):
        """Test system performance with multiple agents under load."""
        # Create a larger number of agents for load testing
        num_agents = 10
        agents = [
            BasicExplorerAgent(
                f"load_agent_{i}", f"Load Agent {i}", grid_size=5
            )
            for i in range(num_agents)
        ]

        # Start all agents
        for agent in agents:
            agent.start()

        # Measure performance
        start_time = datetime.now()

        # Run intensive coordination scenario
        num_steps = 20
        for step in range(num_steps):
            # All agents act simultaneously
            for agent in agents:
                observation = {
                    "position": [step % 5, (step * 2) % 5],
                    "surroundings": np.random.randint(0, 3, (3, 3)),
                    "visible_agents": (
                        [
                            {
                                "id": f"load_agent_{(i + step) % num_agents}",
                                "position": [i, step],
                            }
                            for i in range(
                                min(3, num_agents)
                            )  # Each agent sees up to 3 others
                        ]
                        if step % 2 == 0
                        else []
                    ),
                }

                action = agent.step(observation)
                assert action in agent.actions

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Performance assertions
        total_operations = num_agents * num_steps
        ops_per_second = total_operations / total_time

        # Realistic performance expectation based on actual measurements
        # PyMDP Active Inference is computationally intensive
        assert (
            ops_per_second > 5
        ), f"Performance too slow: {ops_per_second:.2f} ops/sec"

        # Verify all agents completed all steps
        for agent in agents:
            assert agent.total_steps == num_steps

        print(
            f"Performance test: {ops_per_second:.2f} agent operations/second"
        )


@pytest.mark.asyncio
class TestDatabaseIntegration:
    """Test database integration in multi-agent scenarios."""

    async def test_agent_persistence(self, test_db):
        """Test that agent states are properly persisted."""
        # Create agent
        agent = BasicExplorerAgent(
            "persist_test", "Persistence Test", grid_size=3
        )
        agent.start()

        # Create database entry
        agent_model = AgentModel(
            id=agent.agent_id,
            name=agent.name,
            agent_type="basic_explorer",
            config=json.dumps(agent.config),
            status="active",
            metrics=json.dumps(agent.metrics),
        )

        test_db.add(agent_model)
        test_db.commit()

        # Verify persistence
        retrieved = (
            test_db.query(AgentModel)
            .filter(AgentModel.id == agent.agent_id)
            .first()
        )
        assert retrieved is not None
        assert retrieved.name == agent.name
        assert retrieved.status == "active"

    async def test_coalition_persistence(self, test_db):
        """Test that coalition data is properly persisted."""
        # Create coalition
        coalition = Coalition(
            name="Test Coalition",
            description="Integration test coalition",
            objectives=json.dumps({"efficiency": 0.8, "coverage": 0.9}),
            performance_score=0.75,
            cohesion_score=0.85,
        )

        test_db.add(coalition)
        test_db.commit()

        # Verify persistence
        retrieved = (
            test_db.query(Coalition)
            .filter(Coalition.name == "Test Coalition")
            .first()
        )
        assert retrieved is not None
        assert retrieved.performance_score == 0.75
        assert retrieved.cohesion_score == 0.85


@pytest.mark.asyncio
class TestKnowledgeGraphIntegration:
    """Test knowledge graph integration in multi-agent scenarios."""

    async def test_knowledge_node_creation_during_coordination(
        self, knowledge_graph, test_db
    ):
        """Test that knowledge nodes are created during agent coordination."""
        # Create agents
        agents = [
            BasicExplorerAgent(f"kg_test_{i}", f"KG Test {i}", grid_size=3)
            for i in range(2)
        ]

        # Start agents
        for agent in agents:
            agent.start()

        # Simulate coordination that generates knowledge
        for step in range(3):
            for i, agent in enumerate(agents):
                observation = {
                    "position": [i, step],
                    "surroundings": np.ones((3, 3)) * step,
                    "discovered_area": step * 0.3,
                }

                action = agent.step(observation)

                # Create knowledge nodes based on discoveries
                if step > 0:  # After first step, agents have made discoveries
                    node = knowledge_graph.create_node(
                        node_type="discovery",
                        content={
                            "agent_id": agent.agent_id,
                            "action": action,
                            "position": observation["position"],
                            "discovered_area": observation["discovered_area"],
                        },
                        metadata={
                            "step": step,
                            "timestamp": datetime.now().isoformat(),
                        },
                    )

        # Verify knowledge graph has been populated
        assert (
            len(knowledge_graph.nodes) >= 4
        )  # At least 2 agents * 2 steps with discoveries

        # Verify nodes contain meaningful data
        discovery_nodes = [
            node
            for node in knowledge_graph.nodes.values()
            if node.node_type == "discovery"
        ]
        assert len(discovery_nodes) >= 4

        for node in discovery_nodes:
            assert "agent_id" in node.content
            assert "action" in node.content
            assert "position" in node.content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
