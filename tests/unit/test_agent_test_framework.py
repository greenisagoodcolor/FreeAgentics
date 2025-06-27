"""
Module for FreeAgentics Active Inference implementation.
"""

import time

import pytest

from agents.base.data_model import AgentCapability, AgentStatus, Position
from agents.testing.agent_test_framework import (
    AgentFactory,
    AgentTestMetrics,
    AgentTestOrchestrator,
    AgentTestScenario,
    BehaviorValidator,
    PerformanceBenchmark,
    SimulationEnvironment,
    create_basic_test_scenarios,
)


class TestAgentFactory:
    ."""Test AgentFactory functionality."""

    def test_create_basic_agent(self) -> None:
        """Test basic agent creation"""
        agent = AgentFactory.create_basic_agent("test_agent")
        assert agent is not None
        assert agent.agent_id == "test_agent"
        assert agent.name == "TestAgent_test_agent"
        assert agent.status == AgentStatus.IDLE
        assert AgentCapability.MOVEMENT in agent.capabilities
        assert AgentCapability.PERCEPTION in agent.capabilities
        assert agent.resources.energy == 100.0
        assert agent.resources.health == 100.0

    def test_create_resource_agent(self) -> None:
        """Test resource agent creation"""
        agent = AgentFactory.create_resource_agent(
            "resource_agent", resource_types=["gold", "wood"]
        )
        assert agent is not None
        assert isinstance(agent.managed_resources, list)
        assert "gold" in agent.managed_resources
        assert "wood" in agent.managed_resources
        assert AgentCapability.RESOURCE_MANAGEMENT in agent.capabilities

    def test_create_social_agent(self) -> None:
        """Test social agent creation"""
        agent = AgentFactory.create_social_agent("social_agent")
        assert agent is not None
        assert AgentCapability.COMMUNICATION in agent.capabilities
        assert AgentCapability.SOCIAL_INTERACTION in agent.capabilities


class TestSimulationEnvironment:
    ."""Test SimulationEnvironment functionality."""

    def test_environment_creation(self) -> None:
        """Test environment initialization"""
        env = SimulationEnvironment(bounds=(-100, -100, 100, 100), time_scale=2.0)
        assert env.bounds == (-100, -100, 100, 100)
        assert env.time_scale == 2.0
        assert env.current_time == 0.0
        assert len(env.agents) == 0
        assert len(env.resources) == 0

    def test_add_agent(self) -> None:
        """Test adding agents to environment"""
        env = SimulationEnvironment((-50, -50, 50, 50))
        agent = AgentFactory.create_basic_agent("env_agent")
        env.add_agent(agent)
        assert agent.agent_id in env.agents
        assert agent.agent_id in env.state_managers
        assert agent.agent_id in env.movement_controllers
        assert agent.agent_id in env.decision_systems
        assert agent.agent_id in env.memory_systems

    def test_add_resource(self) -> None:
        """Test adding resources to environment"""
        env = SimulationEnvironment((-50, -50, 50, 50))
        position = Position(10, 20, 0)
        env.add_resource(position, "energy", 50.0)
        assert position in env.resources
        assert env.resources[position]["energy"] == 50.0

    def test_add_obstacle(self) -> None:
        """Test adding obstacles to environment"""
        env = SimulationEnvironment((-50, -50, 50, 50))
        agent = AgentFactory.create_basic_agent("obstacle_test")
        env.add_agent(agent)
        obstacle_pos = Position(0, 0, 0)
        env.add_obstacle(obstacle_pos, 5.0)
        assert len(env.obstacles) == 1
        assert env.obstacles[0] == (obstacle_pos, 5.0)

    def test_simulation_step(self) -> None:
        """Test simulation stepping"""
        env = SimulationEnvironment((-50, -50, 50, 50), time_scale=2.0)
        agent = AgentFactory.create_basic_agent("step_test")
        env.add_agent(agent)
        initial_time = env.current_time
        env.step(1.0)
        assert env.current_time == initial_time + 2.0

    def test_get_metrics(self) -> None:
        """Test environment metrics collection"""
        env = SimulationEnvironment((-50, -50, 50, 50))
        for i in range(3):
            env.add_agent(AgentFactory.create_basic_agent(f"agent_{i}"))
        env.add_resource(Position(10, 10, 0), "energy", 100)
        env.add_resource(Position(20, 20, 0), "materials", 50)
        metrics = env.get_metrics()
        assert metrics["agent_count"] == 3
        assert metrics["resource_count"] == 2
        assert metrics["total_resources"] == 150
        assert metrics["time"] == 0.0


class TestBehaviorValidator:
    ."""Test BehaviorValidator functionality."""

    def test_validator_creation(self) -> None:
        """Test validator initialization"""
        validator = BehaviorValidator()
        assert "movement_coherence" in validator.validators
        assert "decision_consistency" in validator.validators
        assert "resource_efficiency" in validator.validators
        assert "social_cooperation" in validator.validators

    def test_movement_validation_success(self) -> None:
        """Test successful movement validation"""
        validator = BehaviorValidator()
        agent = AgentFactory.create_basic_agent("validator_agent")
        history = [
            {"timestamp": 0.0, "position": Position(0, 0, 0)},
            {"timestamp": 0.1, "position": Position(1, 0, 0)},
            {"timestamp": 0.2, "position": Position(2, 1, 0)},
        ]
        success, error = validator.validate("movement_coherence", agent, history)
        assert success
        assert error is None

    def test_movement_validation_failure(self) -> None:
        """Test failed movement validation (teleportation)"""
        validator = BehaviorValidator()
        agent = AgentFactory.create_basic_agent("validator_agent")
        history = [
            {"timestamp": 0.0, "position": Position(0, 0, 0)},
            {"timestamp": 0.1, "position": Position(200, 200, 0)},
        ]
        success, error = validator.validate("movement_coherence", agent, history)
        assert not success
        assert "Impossible speed" in error

    def test_unknown_behavior_type(self) -> None:
        """Test validation with unknown behavior type"""
        validator = BehaviorValidator()
        agent = AgentFactory.create_basic_agent("validator_agent")
        success, error = validator.validate("unknown_behavior", agent, [])
        assert not success
        assert "Unknown behavior type" in error


class TestPerformanceBenchmark:
    ."""Test PerformanceBenchmark functionality."""

    def test_benchmark_creation(self) -> None:
        ."""Test benchmark initialization."""
        benchmark = PerformanceBenchmark()
        assert benchmark.results == {}

    def test_measure_operation(self) -> None:
        """Test measuring operation performance"""
        benchmark = PerformanceBenchmark()
        with benchmark.measure("test_operation"):
            time.sleep(0.01)
        assert "test_operation" in benchmark.results
        assert len(benchmark.results["test_operation"]) == 1
        assert benchmark.results["test_operation"][0] >= 0.01

    def test_get_statistics(self) -> None:
        """Test getting performance statistics"""
        benchmark = PerformanceBenchmark()
        for _ in range(5):
            with benchmark.measure("multi_operation"):
                time.sleep(0.001)
        stats = benchmark.get_statistics("multi_operation")
        assert stats["count"] == 5
        assert "mean" in stats
        assert "median" in stats
        assert "min" in stats
        assert "max" in stats
        assert "stdev" in stats
        assert stats["min"] >= 0.001

    def test_get_report(self) -> None:
        """Test getting full performance report"""
        benchmark = PerformanceBenchmark()
        with benchmark.measure("op1"):
            pass
        with benchmark.measure("op2"):
            pass
        report = benchmark.get_report()
        assert "op1" in report
        assert "op2" in report
        assert "count" in report["op1"]
        assert "count" in report["op2"]


class TestTestOrchestrator:
    ."""Test TestOrchestrator functionality."""

    def test_orchestrator_creation(self) -> None:
        """Test orchestrator initialization"""
        orchestrator = AgentTestOrchestrator()
        assert orchestrator.scenarios == []
        assert orchestrator.results == []
        assert orchestrator.benchmark is not None
        assert orchestrator.validator is not None

    def test_add_scenario(self) -> None:
        """Test adding scenarios"""
        orchestrator = AgentTestOrchestrator()
        scenario = AgentTestScenario(
            name="Test",
            description="Test scenario",
            duration=1.0,
            agent_configs=[{"type": "basic", "id": "test1"}],
            environment_config={"bounds": (-10, -10, 10, 10)},
            success_criteria={},
            metrics_to_track=[],
        )
        orchestrator.add_scenario(scenario)
        assert len(orchestrator.scenarios) == 1
        assert orchestrator.scenarios[0].name == "Test"

    def test_run_scenario(self) -> None:
        """Test running a single scenario"""
        orchestrator = AgentTestOrchestrator()
        scenario = AgentTestScenario(
            name="Quick Test",
            description="Quick test scenario",
            duration=0.5,
            agent_configs=[
                {"type": "basic", "id": "agent1"},
                {"type": "resource", "id": "agent2"},
            ],
            environment_config={"bounds": (-20, -20, 20, 20), "time_scale": 10.0},
            success_criteria={"test": True},
            metrics_to_track=["position", "energy"],
        )
        result = orchestrator.run_scenario(scenario)
        assert result.scenario_name == "Quick Test"
        assert result.success is not None
        assert result.end_time > result.start_time
        assert len(result.agent_metrics) > 0

    def test_generate_report(self) -> None:
        """Test report generation"""
        orchestrator = AgentTestOrchestrator()
        scenario = AgentTestScenario(
            name="Report Test",
            description="Test for reporting",
            duration=0.1,
            agent_configs=[{"type": "basic", "id": "report_agent"}],
            environment_config={"bounds": (-5, -5, 5, 5)},
            success_criteria={},
            metrics_to_track=[],
        )
        orchestrator.add_scenario(scenario)
        orchestrator.run_all_scenarios()
        report = orchestrator.generate_report()
        assert "summary" in report
        assert "scenarios" in report
        assert "performance" in report
        assert report["summary"]["total_scenarios"] == 1
        assert len(report["scenarios"]) == 1


class TestPredefinedScenarios:
    ."""Test predefined test scenarios."""

    def test_create_basic_test_scenarios(self) -> None:
        """Test creation of basic test scenarios"""
        scenarios = create_basic_test_scenarios()
        assert len(scenarios) == 3
        names = [s.name for s in scenarios]
        assert "Basic Movement" in names
        assert "Resource Collection" in names
        assert "Social Cooperation" in names
        for scenario in scenarios:
            assert scenario.duration > 0
            assert len(scenario.agent_configs) > 0
            assert "bounds" in scenario.environment_config
            assert scenario.success_criteria is not None
            assert len(scenario.metrics_to_track) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
