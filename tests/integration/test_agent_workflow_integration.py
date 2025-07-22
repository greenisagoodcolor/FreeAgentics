"""
Integration tests for agent workflow functionality
"""

from unittest.mock import MagicMock, patch

import pytest

# Mock complex dependencies
mock_modules = {
    "pymdp": MagicMock(),
    "pymdp.utils": MagicMock(),
    "pymdp.agent": MagicMock(),
    "sqlalchemy": MagicMock(),
    "sqlalchemy.orm": MagicMock(),
    "redis": MagicMock(),
    "database": MagicMock(),
    "database.session": MagicMock(),
    "database.models": MagicMock(),
    "observability": MagicMock(),
    "observability.belief_monitoring": MagicMock(),
}

with patch.dict("sys.modules", mock_modules):
    from agents.agent_manager import AgentManager
    from agents.base_agent import ActiveInferenceAgent
    from agents.error_handling import ErrorHandler
    from services.agent_factory import AgentFactory


class TestAgentWorkflowIntegration:
    """Test integration between agent components."""

    def setup_method(self):
        """Set up test environment."""
        self.agent_manager = AgentManager()
        self.error_handler = ErrorHandler()
        self.agent_factory = AgentFactory()

    def test_agent_creation_workflow(self):
        """Test complete agent creation workflow."""

        # Mock agent creation
        def mock_create_agent(agent_id, name, config):
            class MockAgent(ActiveInferenceAgent):
                def perceive(self, observation):
                    self.last_observation = observation

                def update_beliefs(self):
                    self.beliefs = {"test": 0.5}

                def select_action(self):
                    return 0

            return MockAgent(agent_id, name, config)

        # Patch factory method
        with patch.object(self.agent_factory, "create_agent", side_effect=mock_create_agent):
            # Create agent
            agent = self.agent_factory.create_agent(
                agent_id="test_agent_001",
                name="Test Agent",
                config={"use_pymdp": False},
            )

            # Verify agent was created
            assert agent.agent_id == "test_agent_001"
            assert agent.name == "Test Agent"
            assert agent.config["use_pymdp"] is False
            assert agent.is_active is False

    def test_agent_lifecycle_workflow(self):
        """Test complete agent lifecycle workflow."""

        # Create test agent
        class LifecycleAgent(ActiveInferenceAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.lifecycle_events = []

            def perceive(self, observation):
                self.lifecycle_events.append(("perceive", observation))
                self.last_observation = observation

            def update_beliefs(self):
                self.lifecycle_events.append(("update_beliefs", None))
                self.beliefs = {"updated": True}

            def select_action(self):
                action = len(self.lifecycle_events) % 2
                self.lifecycle_events.append(("select_action", action))
                return action

        agent = LifecycleAgent(
            agent_id="lifecycle_agent",
            name="Lifecycle Agent",
            config={"use_pymdp": False},
        )

        # Test lifecycle: activate -> perceive -> update -> act -> deactivate

        # 1. Activate
        agent.is_active = True
        assert agent.is_active is True

        # 2. Perceive
        observation = {"sensor": "value1"}
        agent.perceive(observation)
        assert agent.last_observation == observation

        # 3. Update beliefs
        agent.update_beliefs()
        assert agent.beliefs["updated"] is True

        # 4. Select action
        action = agent.select_action()
        assert isinstance(action, int)

        # 5. Verify lifecycle events
        assert len(agent.lifecycle_events) == 3
        assert agent.lifecycle_events[0][0] == "perceive"
        assert agent.lifecycle_events[1][0] == "update_beliefs"
        assert agent.lifecycle_events[2][0] == "select_action"

        # 6. Deactivate
        agent.is_active = False
        assert agent.is_active is False

    def test_agent_error_handling_integration(self):
        """Test agent error handling integration."""

        # Create agent that can fail
        class ErrorProneAgent(ActiveInferenceAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.error_count = 0
                self.success_count = 0

            def perceive(self, observation):
                if "error" in observation:
                    self.error_count += 1
                    raise ValueError("Perception error")
                else:
                    self.success_count += 1
                    self.last_observation = observation

            def update_beliefs(self):
                self.beliefs = {
                    "errors": self.error_count,
                    "successes": self.success_count,
                }

            def select_action(self):
                return 0

        agent = ErrorProneAgent(
            agent_id="error_agent",
            name="Error Agent",
            config={"use_pymdp": False},
        )

        # Test successful perception
        agent.perceive({"sensor": "normal"})
        assert agent.success_count == 1
        assert agent.error_count == 0

        # Test error handling
        with pytest.raises(ValueError):
            agent.perceive({"sensor": "error"})

        assert agent.error_count == 1
        assert agent.success_count == 1

        # Verify beliefs reflect error state
        agent.update_beliefs()
        assert agent.beliefs["errors"] == 1
        assert agent.beliefs["successes"] == 1

    def test_multi_agent_coordination(self):
        """Test coordination between multiple agents."""

        # Create coordinating agents
        class CoordinatingAgent(ActiveInferenceAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.messages = []
                self.coordination_state = "idle"

            def perceive(self, observation):
                if "message" in observation:
                    self.messages.append(observation["message"])
                    self.coordination_state = "processing"
                self.last_observation = observation

            def update_beliefs(self):
                self.beliefs = {
                    "messages_received": len(self.messages),
                    "coordination_state": self.coordination_state,
                }

            def select_action(self):
                if self.coordination_state == "processing":
                    self.coordination_state = "responding"
                    return 1  # Respond action
                return 0  # Idle action

        # Create multiple agents
        agent1 = CoordinatingAgent("agent1", "Agent 1", {"use_pymdp": False})
        agent2 = CoordinatingAgent("agent2", "Agent 2", {"use_pymdp": False})

        # Test coordination
        # Agent 1 receives message
        agent1.perceive({"message": "Hello from external"})
        agent1.update_beliefs()
        action1 = agent1.select_action()

        # Agent 2 receives message from Agent 1
        agent2.perceive({"message": f"Response from {agent1.agent_id}"})
        agent2.update_beliefs()
        action2 = agent2.select_action()

        # Verify coordination
        assert len(agent1.messages) == 1
        assert len(agent2.messages) == 1
        assert agent1.beliefs["messages_received"] == 1
        assert agent2.beliefs["messages_received"] == 1
        assert action1 == 1  # Respond action
        assert action2 == 1  # Respond action

    def test_agent_performance_monitoring(self):
        """Test agent performance monitoring integration."""

        # Create agent with performance tracking
        class PerformanceAgent(ActiveInferenceAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.performance_metrics = {
                    "perception_time": 0.0,
                    "belief_update_time": 0.0,
                    "action_selection_time": 0.0,
                    "total_operations": 0,
                }

            def perceive(self, observation):
                import time

                start_time = time.time()

                # Simulate perception work
                self.last_observation = observation

                self.performance_metrics["perception_time"] += time.time() - start_time
                self.performance_metrics["total_operations"] += 1

            def update_beliefs(self):
                import time

                start_time = time.time()

                # Simulate belief update work
                self.beliefs = {"performance": "tracked"}

                self.performance_metrics["belief_update_time"] += time.time() - start_time

            def select_action(self):
                import time

                start_time = time.time()

                # Simulate action selection work
                action = 0

                self.performance_metrics["action_selection_time"] += time.time() - start_time
                return action

        agent = PerformanceAgent(
            agent_id="perf_agent",
            name="Performance Agent",
            config={"use_pymdp": False},
        )

        # Run agent operations
        for i in range(5):
            agent.perceive({"step": i})
            agent.update_beliefs()
            agent.select_action()

        # Verify performance tracking
        assert agent.performance_metrics["total_operations"] == 5
        assert agent.performance_metrics["perception_time"] > 0
        assert agent.performance_metrics["belief_update_time"] > 0
        assert agent.performance_metrics["action_selection_time"] > 0

    def test_agent_state_persistence(self):
        """Test agent state persistence across operations."""

        # Create agent with persistent state
        class PersistentAgent(ActiveInferenceAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.persistent_state = {
                    "observation_history": [],
                    "belief_history": [],
                    "action_history": [],
                    "step_count": 0,
                }

            def perceive(self, observation):
                self.persistent_state["observation_history"].append(observation)
                self.last_observation = observation

            def update_beliefs(self):
                belief_state = {
                    "step": self.persistent_state["step_count"],
                    "confidence": 0.5 + (self.persistent_state["step_count"] * 0.1),
                }
                self.beliefs = belief_state
                self.persistent_state["belief_history"].append(belief_state)

            def select_action(self):
                action = self.persistent_state["step_count"] % 3
                self.persistent_state["action_history"].append(action)
                self.persistent_state["step_count"] += 1
                return action

        agent = PersistentAgent(
            agent_id="persistent_agent",
            name="Persistent Agent",
            config={"use_pymdp": False},
        )

        # Run multiple steps
        for i in range(3):
            agent.perceive({"step": i, "data": f"data_{i}"})
            agent.update_beliefs()
            agent.select_action()

        # Verify state persistence
        assert len(agent.persistent_state["observation_history"]) == 3
        assert len(agent.persistent_state["belief_history"]) == 3
        assert len(agent.persistent_state["action_history"]) == 3
        assert agent.persistent_state["step_count"] == 3

        # Verify belief evolution
        beliefs = agent.persistent_state["belief_history"]
        assert beliefs[0]["confidence"] == 0.5
        assert beliefs[1]["confidence"] == 0.6
        assert beliefs[2]["confidence"] == 0.7

    def test_agent_configuration_flexibility(self):
        """Test agent configuration flexibility."""
        # Test different configurations
        configs = [
            {"use_pymdp": False, "mode": "test"},
            {"use_pymdp": False, "mode": "production", "precision": 0.01},
            {"use_pymdp": False, "mode": "debug", "logging": True},
        ]

        agents = []

        for i, config in enumerate(configs):

            class ConfigurableAgent(ActiveInferenceAgent):
                def perceive(self, observation):
                    self.last_observation = observation

                def update_beliefs(self):
                    self.beliefs = {"config_mode": self.config.get("mode", "default")}

                def select_action(self):
                    return 0

            agent = ConfigurableAgent(
                agent_id=f"config_agent_{i}",
                name=f"Configurable Agent {i}",
                config=config,
            )
            agents.append(agent)

        # Test each agent's configuration
        for i, agent in enumerate(agents):
            agent.update_beliefs()

            assert agent.config == configs[i]
            assert agent.beliefs["config_mode"] == configs[i]["mode"]

        # Verify agents are independent
        agents[0].config["test_value"] = "modified"
        assert "test_value" not in agents[1].config
        assert "test_value" not in agents[2].config

    def test_agent_resource_management(self):
        """Test agent resource management."""

        # Create resource-aware agent
        class ResourceAgent(ActiveInferenceAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.resource_usage = {"memory": 0, "cpu": 0, "operations": 0}

            def perceive(self, observation):
                # Simulate resource usage
                self.resource_usage["memory"] += len(str(observation))
                self.resource_usage["operations"] += 1
                self.last_observation = observation

            def update_beliefs(self):
                # Simulate CPU usage
                self.resource_usage["cpu"] += 1
                self.beliefs = {"resource_efficient": True}

            def select_action(self):
                return 0

            def cleanup_resources(self):
                """Clean up resources."""
                self.resource_usage["memory"] = max(0, self.resource_usage["memory"] - 10)

        agent = ResourceAgent(
            agent_id="resource_agent",
            name="Resource Agent",
            config={"use_pymdp": False},
        )

        # Use resources
        for i in range(5):
            agent.perceive({"data": f"observation_{i}"})
            agent.update_beliefs()
            agent.select_action()

        # Verify resource tracking
        assert agent.resource_usage["operations"] == 5
        assert agent.resource_usage["cpu"] == 5
        assert agent.resource_usage["memory"] > 0

        # Test cleanup
        initial_memory = agent.resource_usage["memory"]
        agent.cleanup_resources()
        assert agent.resource_usage["memory"] < initial_memory
