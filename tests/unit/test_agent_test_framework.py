"""
Comprehensive tests for Agent Test Framework.

Tests the testing utilities for agent systems including test scenarios,
agent factories, behavior validation, simulation environment, and
performance benchmarking capabilities.
"""

import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from agents.base.data_model import (
    AgentCapability,
    AgentPersonality,
    AgentResources,
    Orientation,
    Position,
)
from agents.testing.agent_test_framework import (
    AgentFactory,
    AgentTestMetrics,
    AgentTestScenario,
    BehaviorValidator,
    PerformanceBenchmark,
    SimulationEnvironment,
)


class TestAgentTestScenario:
    """Test AgentTestScenario dataclass."""

    def test_scenario_creation(self):
        """Test creating test scenario with all fields."""
        agent_configs = [{"id": "agent1", "type": "basic"},
                         {"id": "agent2", "type": "resource"}]
        environment_config = {
            "bounds": (
                0, 0, 100, 100), "resources": {
                "energy": 1000.0}}
        success_criteria = {
            "min_survival_rate": 0.8,
            "min_cooperation_score": 0.5}
        metrics = ["agent_health", "resource_collection", "cooperation_events"]

        scenario = AgentTestScenario(
            name="test_scenario",
            description="Test scenario for cooperative resource gathering",
            duration=100.0,
            agent_configs=agent_configs,
            environment_config=environment_config,
            success_criteria=success_criteria,
            metrics_to_track=metrics,
        )

        assert scenario.name == "test_scenario"
        assert scenario.description == "Test scenario for cooperative resource gathering"
        assert scenario.duration == 100.0
        assert scenario.agent_configs == agent_configs
        assert scenario.environment_config == environment_config
        assert scenario.success_criteria == success_criteria
        assert scenario.metrics_to_track == metrics


class TestAgentTestMetrics:
    """Test AgentTestMetrics dataclass."""

    def test_metrics_creation(self):
        """Test creating test metrics with all fields."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=100)
        agent_metrics = {
            "agent1": {"health": 80.0, "resources_collected": 150.0},
            "agent2": {"health": 90.0, "resources_collected": 200.0},
        }
        environment_metrics = {
            "total_resources": 500.0,
            "resource_depletion_rate": 0.1}
        performance_metrics = {"avg_fps": 60.0, "memory_usage_mb": 128.0}

        metrics = AgentTestMetrics(
            scenario_name="test_scenario",
            start_time=start_time,
            end_time=end_time,
            agent_metrics=agent_metrics,
            environment_metrics=environment_metrics,
            performance_metrics=performance_metrics,
            success=True,
            failure_reason=None,
        )

        assert metrics.scenario_name == "test_scenario"
        assert metrics.start_time == start_time
        assert metrics.end_time == end_time
        assert metrics.agent_metrics == agent_metrics
        assert metrics.environment_metrics == environment_metrics
        assert metrics.performance_metrics == performance_metrics
        assert metrics.success is True
        assert metrics.failure_reason is None

    def test_metrics_defaults(self):
        """Test default values for optional fields."""
        metrics = AgentTestMetrics(
            scenario_name="test",
            start_time=datetime.now())

        assert metrics.end_time is None
        assert metrics.agent_metrics == {}
        assert metrics.environment_metrics == {}
        assert metrics.performance_metrics == {}
        assert metrics.success is None
        assert metrics.failure_reason is None


class TestAgentFactory:
    """Test AgentFactory class."""

    @patch("agents.testing.agent_test_framework.random.uniform")
    @patch("agents.testing.agent_test_framework.random.random")
    def test_create_basic_agent_defaults(self, mock_random, mock_uniform):
        """Test creating basic agent with default values."""
        # Mock random values
        mock_uniform.side_effect = [50.0, -25.0]  # x, y positions
        mock_random.side_effect = [
            0.8, 0.7, 0.6, 0.5, 0.4]  # personality traits

        agent = AgentFactory.create_basic_agent("test_agent")

        assert agent.agent_id == "test_agent"
        assert agent.name == "TestAgent_test_agent"
        assert agent.position.x == 50.0
        assert agent.position.y == -25.0
        assert agent.position.z == 0.0
        assert agent.orientation == Orientation(0, 0, 0, 1)

        # Check personality
        assert agent.personality.openness == 0.8
        assert agent.personality.conscientiousness == 0.7
        assert agent.personality.extraversion == 0.6
        assert agent.personality.agreeableness == 0.5
        assert agent.personality.neuroticism == 0.4

        # Check capabilities
        assert AgentCapability.MOVEMENT in agent.capabilities
        assert AgentCapability.PERCEPTION in agent.capabilities

        # Check resources
        assert agent.resources.energy == 100.0
        assert agent.resources.health == 100.0
        assert agent.resources.memory_capacity == 100

    def test_create_basic_agent_custom(self):
        """Test creating basic agent with custom values."""
        position = Position(10.0, 20.0, 5.0)
        personality = AgentPersonality(
            openness=0.9,
            conscientiousness=0.8,
            extraversion=0.7,
            agreeableness=0.6,
            neuroticism=0.3,
        )
        capabilities = {
            AgentCapability.MOVEMENT,
            AgentCapability.COMMUNICATION}

        agent = AgentFactory.create_basic_agent(
            "custom_agent",
            position=position,
            personality=personality,
            capabilities=capabilities)

        assert agent.agent_id == "custom_agent"
        assert agent.position == position
        assert agent.personality == personality
        assert agent.capabilities == capabilities

    def test_create_resource_agent_defaults(self):
        """Test creating resource agent with default resource types."""
        with patch.object(AgentFactory, "create_basic_agent") as mock_create:
            # Create a mock basic agent
            basic_agent = Mock()
            basic_agent.name = "TestAgent_resource1"
            basic_agent.position = Position(0, 0, 0)
            basic_agent.orientation = Orientation(0, 0, 0, 1)
            basic_agent.personality = AgentPersonality()
            basic_agent.capabilities = {AgentCapability.MOVEMENT}
            basic_agent.resources = AgentResources()
            mock_create.return_value = basic_agent

            resource_agent = AgentFactory.create_resource_agent("resource1")

            assert resource_agent.agent_id == "resource1"
            assert resource_agent.managed_resources == {
                "energy": 100.0, "materials": 100.0}
            assert AgentCapability.RESOURCE_MANAGEMENT in resource_agent.capabilities

    def test_create_resource_agent_custom(self):
        """Test creating resource agent with custom resource types."""
        with patch.object(AgentFactory, "create_basic_agent") as mock_create:
            # Create a mock basic agent
            basic_agent = Mock()
            basic_agent.name = "TestAgent_resource2"
            basic_agent.position = Position(0, 0, 0)
            basic_agent.orientation = Orientation(0, 0, 0, 1)
            basic_agent.personality = AgentPersonality()
            basic_agent.capabilities = {AgentCapability.MOVEMENT}
            basic_agent.resources = AgentResources()
            mock_create.return_value = basic_agent

            resource_types = ["food", "water", "tools"]
            resource_agent = AgentFactory.create_resource_agent(
                "resource2", resource_types)

            expected_resources = {
                "food": 100.0,
                "water": 100.0,
                "tools": 100.0}
            assert resource_agent.managed_resources == expected_resources

    def test_create_social_agent(self):
        """Test creating social agent."""
        with patch.object(AgentFactory, "create_basic_agent") as mock_create:
            # Create a mock basic agent
            basic_agent = Mock()
            basic_agent.name = "TestAgent_social1"
            basic_agent.position = Position(0, 0, 0)
            basic_agent.orientation = Orientation(0, 0, 0, 1)
            basic_agent.personality = AgentPersonality()
            basic_agent.capabilities = {AgentCapability.MOVEMENT}
            basic_agent.resources = AgentResources()
            mock_create.return_value = basic_agent

            social_agent = AgentFactory.create_social_agent("social1")

            assert social_agent.agent_id == "social1"
            assert AgentCapability.COMMUNICATION in social_agent.capabilities


class TestSimulationEnvironment:
    """Test SimulationEnvironment class."""

    def setup_method(self):
        """Set up test simulation environment."""
        self.bounds = (0.0, 0.0, 100.0, 100.0)

        # Mock all the dependencies
        with patch(
            "agents.testing.agent_test_framework.AgentStateManager"
        ) as mock_state_manager_class:
            with patch(
                "agents.testing.agent_test_framework.PerceptionSystem"
            ) as mock_perception_class:
                with patch(
                    "agents.testing.agent_test_framework.InteractionSystem"
                ) as mock_interaction_class:
                    # Create mock instances
                    self.mock_state_manager = Mock()
                    self.mock_perception = Mock()
                    self.mock_interaction = Mock()

                    # Configure mock classes to return mock instances
                    mock_state_manager_class.return_value = self.mock_state_manager
                    mock_perception_class.return_value = self.mock_perception
                    mock_interaction_class.return_value = self.mock_interaction

                    self.env = SimulationEnvironment(
                        self.bounds, time_scale=2.0)

    def test_environment_initialization(self):
        """Test environment initialization."""
        assert self.env.bounds == self.bounds
        assert self.env.time_scale == 2.0
        assert self.env.agents == {}
        assert self.env.resources == {}
        assert self.env.obstacles == []
        assert self.env.current_time == 0.0
        assert self.env.events == []
        assert isinstance(self.env.state_managers, dict)
        assert isinstance(self.env.movement_controllers, dict)

    @patch("agents.testing.agent_test_framework.AgentStateManager")
    @patch("agents.testing.agent_test_framework.CollisionSystem")
    @patch("agents.testing.agent_test_framework.PathfindingGrid")
    @patch("agents.testing.agent_test_framework.MovementController")
    @patch("agents.testing.agent_test_framework.DecisionSystem")
    @patch("agents.testing.agent_test_framework.MemorySystem")
    def test_add_agent(
        self,
        mock_memory_class,
        mock_decision_class,
        mock_movement_class,
        mock_pathfinding_class,
        mock_collision_class,
        mock_state_manager_class,
    ):
        """Test adding agent to environment."""
        # Create mocks
        mock_agent_state_manager = Mock()
        mock_collision = Mock()
        mock_pathfinding = Mock()
        mock_movement = Mock()
        mock_decision = Mock()
        mock_memory = Mock()

        # Configure mock classes
        mock_state_manager_class.return_value = mock_agent_state_manager
        mock_collision_class.return_value = mock_collision
        mock_pathfinding_class.return_value = mock_pathfinding
        mock_movement_class.return_value = mock_movement
        mock_decision_class.return_value = mock_decision
        mock_memory_class.return_value = mock_memory

        agent = AgentFactory.create_basic_agent("test_agent")

        self.env.add_agent(agent)

        assert "test_agent" in self.env.agents
        assert self.env.agents["test_agent"] == agent
        assert "test_agent" in self.env.state_managers
        assert "test_agent" in self.env.movement_controllers
        assert "test_agent" in self.env.decision_systems
        assert "test_agent" in self.env.memory_systems

        # Verify registrations
        self.mock_state_manager.register_agent.assert_called_once_with(agent)
        self.mock_perception.register_agent.assert_called_once_with(agent)
        mock_movement.register_agent.assert_called_once_with(agent)
        mock_decision.register_agent.assert_called_once_with(agent)

    def test_add_resource(self):
        """Test adding resource to environment."""
        position = Position(50.0, 50.0, 0.0)

        self.env.add_resource(position, "energy", 100.0)

        assert position in self.env.resources
        assert self.env.resources[position]["energy"] == 100.0

        # Add another resource at same position
        self.env.add_resource(position, "materials", 50.0)

        assert self.env.resources[position]["energy"] == 100.0
        assert self.env.resources[position]["materials"] == 50.0

    def test_get_metrics(self):
        """Test getting environment metrics."""
        # Add resources
        self.env.add_resource(Position(10, 10, 0), "energy", 100.0)
        self.env.add_resource(Position(20, 20, 0), "materials", 50.0)
        self.env.add_resource(Position(20, 20, 0), "food", 30.0)

        # Add some events
        self.env.current_time = 10.0
        self.env.events = [{"type": "test"} for _ in range(5)]

        # Add fake agents
        self.env.agents = {"agent1": Mock(), "agent2": Mock()}

        metrics = self.env.get_metrics()

        assert metrics["time"] == 10.0
        assert metrics["agent_count"] == 2
        assert metrics["resource_count"] == 3  # 1 + 2 resources
        assert metrics["total_resources"] == 180.0  # 100 + 50 + 30
        assert metrics["events"] == 5


class TestBehaviorValidator:
    """Test BehaviorValidator class."""

    def setup_method(self):
        """Set up test behavior validator."""
        self.validator = BehaviorValidator()
        self.agent = AgentFactory.create_basic_agent("test_agent")

    def test_validator_initialization(self):
        """Test validator initialization with default validators."""
        assert "movement_coherence" in self.validator.validators
        assert "decision_consistency" in self.validator.validators
        assert "resource_efficiency" in self.validator.validators
        assert "social_cooperation" in self.validator.validators

    def test_validate_unknown_behavior(self):
        """Test validating unknown behavior type."""
        success, error = self.validator.validate(
            "unknown_behavior", self.agent, [])

        assert success is False
        assert error == "Unknown behavior type: unknown_behavior"

    def test_validate_movement_coherence_insufficient_data(self):
        """Test movement coherence with insufficient history."""
        history = [{"position": Position(0, 0, 0), "timestamp": 0}]

        success, error = self.validator.validate(
            "movement_coherence", self.agent, history)

        assert success is True
        assert error is None

    def test_validate_movement_coherence_valid(self):
        """Test movement coherence with valid movement."""
        history = [
            {"position": Position(0, 0, 0), "timestamp": 0},
            {"position": Position(1, 1, 0), "timestamp": 1},
            {"position": Position(2, 2, 0), "timestamp": 2},
        ]

        success, error = self.validator.validate(
            "movement_coherence", self.agent, history)

        assert success is True
        assert error is None

    def test_validate_movement_coherence_impossible_speed(self):
        """Test movement coherence detecting impossible speed."""
        # Create positions with distance_to method
        pos1 = Mock()
        pos2 = Mock()
        # 200 units distance from pos1
        pos2.distance_to = Mock(return_value=200.0)

        history = [
            {"position": pos1, "timestamp": 0},
            # 200 units in 1 second = 200 units/s
            {"position": pos2, "timestamp": 1},
        ]

        success, error = self.validator.validate(
            "movement_coherence", self.agent, history)

        assert success is False
        assert "Impossible speed detected" in error
        assert "200" in error

    def test_validate_decision_consistency(self):
        """Test decision consistency validation (placeholder)."""
        history = [{"decision": "move", "goal": "explore"}]

        success, error = self.validator.validate(
            "decision_consistency", self.agent, history)

        assert success is True
        assert error is None


class TestPerformanceBenchmark:
    """Test PerformanceBenchmark class."""

    def setup_method(self):
        """Set up test performance benchmark."""
        self.benchmark = PerformanceBenchmark()

    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        assert self.benchmark.results == {}

    def test_measure_context_manager(self):
        """Test measure context manager."""
        with self.benchmark.measure("test_operation"):
            time.sleep(0.01)  # Simulate some work

        assert "test_operation" in self.benchmark.results
        assert len(self.benchmark.results["test_operation"]) == 1
        assert self.benchmark.results["test_operation"][0] > 0.01

    def test_multiple_measurements(self):
        """Test multiple measurements of same operation."""
        for i in range(3):
            with self.benchmark.measure("repeated_op"):
                time.sleep(0.001)

        assert len(self.benchmark.results["repeated_op"]) == 3
        for duration in self.benchmark.results["repeated_op"]:
            assert duration > 0

    def test_get_statistics_empty(self):
        """Test getting statistics for non-existent operation."""
        stats = self.benchmark.get_statistics("non_existent")
        assert stats == {}

    def test_get_statistics_single_measurement(self):
        """Test getting statistics with single measurement."""
        with self.benchmark.measure("single_op"):
            pass

        stats = self.benchmark.get_statistics("single_op")
        assert stats["count"] == 1
        assert stats["mean"] == stats["median"] == stats["min"] == stats["max"]
        assert stats["stdev"] == 0.0

    def test_get_statistics_multiple_measurements(self):
        """Test getting statistics with multiple measurements."""
        # Add known durations manually for predictable results
        self.benchmark.results["test_op"] = [0.1, 0.2, 0.3]

        stats = self.benchmark.get_statistics("test_op")
        assert stats["count"] == 3
        assert stats["mean"] == 0.2
        assert stats["median"] == 0.2
        assert stats["min"] == 0.1
        assert stats["max"] == 0.3
        assert stats["stdev"] > 0

    def test_get_report(self):
        """Test getting full performance report."""
        # Add test data
        self.benchmark.results["op1"] = [0.1, 0.2]
        self.benchmark.results["op2"] = [0.3]

        report = self.benchmark.get_report()

        assert "op1" in report
        assert "op2" in report
        assert report["op1"]["count"] == 2
        assert report["op2"]["count"] == 1
