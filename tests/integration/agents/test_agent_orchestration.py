"""Integration tests for agent orchestration systems"""

import math
import time

import numpy as np
import pytest

from agents.base.data_model import (
    Action,
    ActionType,
    AgentCapability,
    AgentStatus,
    Experience,
    Position,
)
from agents.base.decision_making import DecisionSystem
from agents.base.interaction import (
    InteractionRequest,
    InteractionSystem,
    InteractionType,
    MessageType,
    ResourceType,
)
from agents.base.memory import MemorySystem, MemoryType
from agents.base.movement import CollisionSystem, MovementController, PathfindingGrid
from agents.base.perception import PerceptionSystem, Stimulus, StimulusType
from agents.base.persistence import AgentPersistence
from agents.base.state_manager import AgentStateManager
from agents.testing.agent_test_framework import (
    AgentFactory,
    AgentTestOrchestrator,
    AgentTestScenario,
    BehaviorValidator,
    PerformanceBenchmark,
    SimulationEnvironment,
)


class TestAgentIntegration:
    """Integration tests for agent systems"""

    def test_full_agent_lifecycle(self) -> None:
        """Test complete agent lifecycle from creation to persistence."""
        agent = AgentFactory.create_basic_agent("test_agent")
        assert agent is not None
        assert agent.status == AgentStatus.IDLE
        # Create state manager
        state_manager = AgentStateManager()
        state_manager.register_agent(agent)
        # Create required components for MovementController
        collision_system = CollisionSystem()
        pathfinding_grid = PathfindingGrid(100, 100)  # Create a 100x100 grid
        # Create movement controller
        movement_controller = MovementController(state_manager, collision_system, pathfinding_grid)
        movement_controller.register_agent(agent)
        # Create perception system
        perception_system = PerceptionSystem(state_manager)
        perception_system.register_agent(agent)
        # Create decision system with all required components
        decision_system = DecisionSystem(state_manager, perception_system, movement_controller)
        decision_system.register_agent(agent)
        # Create memory system
        memory_system = MemorySystem(agent.agent_id, agent.resources.memory_capacity)
        # Test status update
        state_manager.update_agent_status(agent.agent_id, AgentStatus.MOVING)
        assert agent.status == AgentStatus.MOVING
        # Test movement
        target = Position(10, 10, 0)
        movement_controller.set_destination(agent.agent_id, target)
        movement_controller.update(0.1)
        assert agent.position.x != 0 or agent.position.y != 0
        # Test perception
        # First get the agent's current position
        agent_position = agent.position
        # Create a stimulus very close to the agent
        stimulus = Stimulus(
            stimulus_id="object1",
            stimulus_type=StimulusType.OBJECT,
            position=Position(agent_position.x + 1.0, agent_position.y + 1.0, agent_position.z),
            intensity=1.0,
            radius=10.0,  # Make it have a large radius
            metadata={"type": "resource", "is_resource": True},
        )
        perception_system.add_stimulus(stimulus)
        # Make sure the agent has perception capabilities
        perception_caps = perception_system.perception_capabilities[agent.agent_id]
        perception_caps.visual_range = 100.0
        perception_caps.field_of_view = math.pi * 2
        percepts = perception_system.perceive(agent.agent_id)
        assert len(percepts) > 0
        # Test decision making
        action = decision_system.make_decision(agent.agent_id)
        assert action is not None
        # Test memory
        memory_system.working_memory.add("test_key", 1.0)
        assert "test_key" in memory_system.working_memory.items
        # Test persistence
        persistence = AgentPersistence()
        agent_data = persistence._serialize_agent(agent)
        assert agent_data is not None
        # Skip deserialization since we don't have a database connection
        # loaded_agent = persistence._deserialize_agent(agent_data)
        # assert loaded_agent.agent_id == agent.agent_id
        # assert loaded_agent.position.x == agent.position.x

    def test_multi_agent_interaction(self) -> None:
        """Test interaction between multiple agents."""
        agent1 = AgentFactory.create_social_agent("agent1")
        agent2 = AgentFactory.create_social_agent("agent2")
        agent3 = AgentFactory.create_resource_agent("agent3")
        # Create interaction system
        interaction_system = InteractionSystem()
        interaction_system.register_agent(agent1)
        interaction_system.register_agent(agent2)
        interaction_system.register_agent(agent3)
        # Create interaction request
        request = InteractionRequest(
            initiator_id=agent1.agent_id,
            target_id=agent2.agent_id,
            interaction_type=InteractionType.COMMUNICATION,
            parameters={
                "message_type": MessageType.GREETING,
                "content": {"greeting": "hello"},
            },
        )
        # Initiate interaction
        interaction_id = interaction_system.initiate_interaction(request)
        assert interaction_id is not None
        # Process interaction
        result = interaction_system.process_interaction(interaction_id)
        assert result.success
        # Test resource exchange
        exchange_request = InteractionRequest(
            initiator_id=agent3.agent_id,
            target_id=agent1.agent_id,
            interaction_type=InteractionType.RESOURCE_EXCHANGE,
            parameters={"resource_type": ResourceType.ENERGY, "amount": 10.0},
        )
        exchange_id = interaction_system.initiate_interaction(exchange_request)
        exchange_result = interaction_system.process_interaction(exchange_id)
        assert exchange_result.success
        # Test memory system
        memory_system = MemorySystem(agent1.agent_id)
        memory_system.store_memory({"interaction_id": interaction_id}, MemoryType.EPISODIC)
        memory_system.consolidate_memories()
        # Test persistence
        persistence = AgentPersistence()
        _ = persistence._serialize_agent(agent1)
        retrieved = memory_system.retrieve_memories({"memory_type": MemoryType.EPISODIC}, 10)
        assert len(retrieved) > 0

    def test_environment_simulation(self) -> None:
        """Test full environment simulation with multiple agents"""
        environment = SimulationEnvironment(bounds=(-50, -50, 50, 50), time_scale=1.0)
        # Create a state manager for the perception system
        state_manager = AgentStateManager()
        environment.perception_system = PerceptionSystem(state_manager)
        for i in range(5):
            agent = AgentFactory.create_basic_agent(f"agent_{i}")
            environment.add_agent(agent)
            # Register agents with the state manager
            state_manager.register_agent(agent)
        environment.add_resource(Position(10, 10, 0), "energy", 100)
        environment.add_resource(Position(-20, 15, 0), "materials", 50)
        environment.add_obstacle(Position(0, 0, 0), 5.0)
        for _ in range(100):
            environment.step(0.1)
        assert environment.current_time > 0
        metrics = environment.get_metrics()
        assert metrics["agent_count"] == 5
        assert metrics["resource_count"] == 2

    def test_behavior_validation(self) -> None:
        """Test behavior validation system."""
        agent = AgentFactory.create_basic_agent("validator_test")
        validator = BehaviorValidator()
        history = []
        positions = [
            Position(0, 0, 0),
            Position(1, 0, 0),
            Position(2, 1, 0),
            Position(3, 2, 0),
        ]
        for i, pos in enumerate(positions):
            history.append({"timestamp": i * 0.1, "position": pos, "action": "move"})
        success, error = validator.validate("movement_coherence", agent, history)
        assert success
        history.append({"timestamp": len(history) * 0.1, "position": Position(100, 100, 0)})
        success, error = validator.validate("movement_coherence", agent, history)
        assert not success
        assert "Impossible speed" in error

    def test_performance_benchmarking(self) -> None:
        """Test performance benchmarking system"""
        benchmark = PerformanceBenchmark()
        with benchmark.measure("agent_creation"):
            for i in range(100):
                AgentFactory.create_basic_agent(f"perf_agent_{i}")
        agent = AgentFactory.create_basic_agent("benchmark_agent")
        state_manager = AgentStateManager()
        state_manager.register_agent(agent)
        with benchmark.measure("state_update"):
            for _ in range(100):
                state_manager.update_agent_status(agent.agent_id, AgentStatus.MOVING)
                state_manager.update_agent_status(agent.agent_id, AgentStatus.IDLE)
        report = benchmark.get_report()
        assert "agent_creation" in report
        assert "state_update" in report
        assert report["agent_creation"]["count"] == 1
        assert report["state_update"]["count"] == 1

    def test_scenario_orchestration(self) -> None:
        """Test scenario orchestration system"""
        orchestrator = AgentTestOrchestrator()
        scenario = AgentTestScenario(
            name="Test Scenario",
            description="Basic test scenario",
            duration=5.0,
            agent_configs=[
                {"type": "basic", "id": "test1"},
                {"type": "resource", "id": "test2"},
                {"type": "social", "id": "test3"},
            ],
            environment_config={"bounds": (-20, -20, 20, 20), "time_scale": 10.0},
            success_criteria={"all_agents_active": True},
            metrics_to_track=["position", "energy", "status"],
        )
        orchestrator.add_scenario(scenario)
        results = orchestrator.run_all_scenarios()
        assert len(results) == 1
        result = results[0]
        assert result.scenario_name == "Test Scenario"
        assert result.success is not None
        assert result.end_time > result.start_time
        report = orchestrator.generate_report()
        assert report["summary"]["total_scenarios"] == 1

    def test_agent_memory_persistence(self) -> None:
        """Test memory system persistence across agent reload."""
        agent = AgentFactory.create_basic_agent("memory_test")
        memory_system = MemorySystem(agent.agent_id)
        # Create an experience
        action = Action(action_type=ActionType.EXPLORE)
        # Create an experience
        experience = Experience(
            state={"location": "home"},
            action=action,
            outcome={"found": "resource"},
            reward=10.0,
            next_state={"location": "resource_site"},
        )
        # Store the experience
        memory_system.store_experience(experience)
        # Store a memory directly
        memory_system.store_memory(
            content={"discovery": "valuable resource"},
            memory_type=MemoryType.EPISODIC,
            importance=0.8,
        )
        # Verify memory was stored
        memories = memory_system.retrieve_memories({"memory_type": MemoryType.EPISODIC}, 10)
        assert len(memories) > 0
        # Test persistence (simplified since we don't have a real database)
        memory_summary = memory_system.get_memory_summary()
        assert memory_summary["total_memories"] > 0

    def test_perception_to_decision_pipeline(self) -> None:
        """Test the full pipeline from perception to decision."""
        agent = AgentFactory.create_basic_agent("pipeline_test")
        agent.capabilities.add(AgentCapability.COMMUNICATION)
        # Create state manager
        state_manager = AgentStateManager()
        state_manager.register_agent(agent)
        # Create perception system
        perception_system = PerceptionSystem(state_manager)
        perception_system.register_agent(agent)
        # Create collision system and pathfinding grid
        collision_system = CollisionSystem()
        pathfinding_grid = PathfindingGrid(100, 100)
        # Create movement controller
        movement_controller = MovementController(state_manager, collision_system, pathfinding_grid)
        movement_controller.register_agent(agent)
        # Create decision system
        decision_system = DecisionSystem(state_manager, perception_system, movement_controller)
        decision_system.register_agent(agent)
        # Create a stimulus
        stimulus = Stimulus(
            stimulus_id="test_object",
            stimulus_type=StimulusType.OBJECT,
            position=Position(agent.position.x + 2.0, agent.position.y + 2.0, 0),
            intensity=1.0,
            radius=5.0,
            metadata={"type": "resource", "is_resource": True},
        )
        # Add stimulus to perception system
        perception_system.add_stimulus(stimulus)
        # Test perception
        percepts = perception_system.perceive(agent.agent_id)
        assert len(percepts) > 0
        # Test decision making
        action = decision_system.make_decision(agent.agent_id)
        assert action is not None
        # Test action execution
        result = decision_system.execute_action(agent.agent_id, action)
        assert result is not False

    def test_movement_with_obstacles(self) -> None:
        """Test movement system with obstacle avoidance."""
        agent = AgentFactory.create_basic_agent("obstacle_test")
        # Create state manager
        state_manager = AgentStateManager()
        state_manager.register_agent(agent)
        # Create collision system and pathfinding grid
        collision_system = CollisionSystem()
        pathfinding_grid = PathfindingGrid(100, 100)
        # Add obstacles to collision system
        obstacle_pos = Position(5, 5, 0)
        collision_system.add_static_obstacle(obstacle_pos, 2.0)
        # Create movement controller
        movement_controller = MovementController(state_manager, collision_system, pathfinding_grid)
        movement_controller.register_agent(agent)
        # Set destination on other side of obstacle
        target = Position(10, 10, 0)
        movement_controller.set_destination(agent.agent_id, target)
        # Update movement several times
        for _ in range(10):
            movement_controller.update(0.1)
        # Check that agent has moved
        assert agent.position.x != 0 or agent.position.y != 0
        # Check that agent didn't move through obstacle
        distance_to_obstacle = agent.position.distance_to(obstacle_pos)
        assert distance_to_obstacle > 2.0  # Obstacle radius


class TestAgentStressTests:
    """Stress tests for agent systems"""

    def test_many_agents_simulation(self) -> None:
        """Test simulation with many agents"""
        environment = SimulationEnvironment(bounds=(-200, -200, 200, 200), time_scale=1.0)
        # Create a state manager for the perception system
        state_manager = AgentStateManager()
        environment.perception_system = PerceptionSystem(state_manager)
        agents = []
        for i in range(100):
            agent_type = ["basic", "resource", "social"][i % 3]
            if agent_type == "basic":
                agent = AgentFactory.create_basic_agent(f"stress_{i}")
            elif agent_type == "resource":
                agent = AgentFactory.create_resource_agent(f"stress_{i}")
            else:
                agent = AgentFactory.create_social_agent(f"stress_{i}")
            environment.add_agent(agent)
            # Register agents with the state manager
            state_manager.register_agent(agent)
            agents.append(agent)
        for i in range(50):
            pos = Position(np.random.uniform(-190, 190), np.random.uniform(-190, 190), 0)
            environment.add_resource(pos, "energy", np.random.uniform(10, 100))
        start_time = time.time()
        for _ in range(10):
            environment.step(0.1)
        duration = time.time() - start_time
        assert duration < 5.0
        for agent in agents:
            assert agent.status != AgentStatus.ERROR

    def test_memory_system_capacity(self) -> None:
        """Test memory system with large amounts of data."""
        memory_system = MemorySystem("capacity_test")
        # Create an action for experiences
        action = Action(action_type=ActionType.EXPLORE)
        # Store many experiences
        for i in range(100):
            experience = Experience(
                state={"index": i, "data": f"state_{i}" * 10},
                action=action,
                outcome={"result": f"outcome_{i}"},
                reward=float(i % 10),
                next_state={"index": i + 1},
            )
            memory_system.store_experience(experience)
            # Also store direct memories
            memory_system.store_memory(
                content={"data": f"memory_{i}"},
                memory_type=MemoryType.EPISODIC,
                importance=float((i % 5) / 5),
            )
        # Test memory retrieval
        memories = memory_system.retrieve_memories({"memory_type": MemoryType.EPISODIC}, 10)
        assert len(memories) > 0
        # Test memory statistics
        memory_summary = memory_system.get_memory_summary()
        assert memory_summary["total_memories"] >= 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
