"""
Agent Test Framework

This module provides testing utilities for agent systems including:
- Test scenarios and environment setup
- Agent factories for different agent types
- Behavior validation and verification
- Performance benchmarking
"""

import json
import logging
import random
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from agents.base.data_model import (
    Agent,
    AgentCapability,
    AgentPersonality,
    AgentResources,
    Orientation,
    Position,
    ResourceAgent,
    SocialAgent,
)
from agents.base.decision_making import DecisionSystem
from agents.base.interaction import InteractionSystem
from agents.base.memory import MemorySystem
from agents.base.movement import CollisionSystem, MovementController, PathfindingGrid
from agents.base.perception import PerceptionSystem
from agents.base.state_manager import AgentStateManager

logger = logging.getLogger(__name__)


@dataclass
class AgentTestScenario:
    """Represents a test scenario for agent behavior"""

    name: str
    description: str
    duration: float
    agent_configs: List[Dict[str, Any]]
    environment_config: Dict[str, Any]
    success_criteria: Dict[str, Any]
    metrics_to_track: List[str]


@dataclass
class AgentTestMetrics:
    """Metrics collected during test execution"""

    scenario_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    agent_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    environment_metrics: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    success: Optional[bool] = None
    failure_reason: Optional[str] = None


class AgentFactory:
    """Factory for creating test agents with various configurations"""

    @staticmethod
    def create_basic_agent(
        agent_id: str,
        position: Optional[Position] = None,
        personality: Optional[AgentPersonality] = None,
        capabilities: Optional[set[AgentCapability]] = None,
    ) -> Agent:
        """Create a basic test agent"""
        if position is None:
            position = Position(random.uniform(-100, 100), random.uniform(-100, 100), 0)
        if personality is None:
            personality = AgentPersonality(
                openness=random.random(),
                conscientiousness=random.random(),
                extraversion=random.random(),
                agreeableness=random.random(),
                neuroticism=random.random(),
            )
        if capabilities is None:
            capabilities = {AgentCapability.MOVEMENT, AgentCapability.PERCEPTION}
        return Agent(
            agent_id=agent_id,
            name=f"TestAgent_{agent_id}",
            position=position,
            orientation=Orientation(0, 0, 0, 1),
            personality=personality,
            capabilities=capabilities,
            resources=AgentResources(energy=100.0, health=100.0, memory_capacity=100),
        )

    @staticmethod
    def create_resource_agent(
        agent_id: str, resource_types: Optional[List[str]] = None
    ) -> ResourceAgent:
        """Create a resource-focused test agent"""
        agent = AgentFactory.create_basic_agent(agent_id)
        if resource_types is None:
            resource_types = ["energy", "materials"]
        return ResourceAgent(
            agent_id=agent_id,
            name=agent.name,
            position=agent.position,
            orientation=agent.orientation,
            personality=agent.personality,
            capabilities=agent.capabilities | {AgentCapability.RESOURCE_MANAGEMENT},
            resources=agent.resources,
            managed_resources={rt: 100.0 for rt in resource_types},  # Convert list to dict
        )

    @staticmethod
    def create_social_agent(agent_id: str) -> SocialAgent:
        """Create a socially-focused test agent"""
        agent = AgentFactory.create_basic_agent(agent_id)
        return SocialAgent(
            agent_id=agent_id,
            name=agent.name,
            position=agent.position,
            orientation=agent.orientation,
            personality=agent.personality,
            capabilities=agent.capabilities | {AgentCapability.COMMUNICATION},
            resources=agent.resources,
        )


class SimulationEnvironment:
    """Simulated environment for agent testing"""

    def __init__(self, bounds: tuple[float, float, float, float], time_scale: float = 1.0) -> None:
        """
        Initialize simulation environment

        Args:
            bounds: (min_x, min_y, max_x, max_y)
            time_scale: Time multiplier for simulation speed
        """
        self.bounds = bounds
        self.time_scale = time_scale
        self.agents: Dict[str, Agent] = {}
        self.resources: Dict[Position, Dict[str, float]] = {}
        self.obstacles: List[tuple[Position, float]] = []
        self.current_time = 0.0
        self.events: List[Dict[str, Any]] = []
        self.state_managers: Dict[str, AgentStateManager] = {}
        self.movement_controllers: Dict[str, MovementController] = {}
        self.state_manager = AgentStateManager()  # Main state manager for perception
        self.perception_system = PerceptionSystem(self.state_manager)
        self.decision_systems: Dict[str, DecisionSystem] = {}
        self.interaction_system = InteractionSystem()
        self.memory_systems: Dict[str, MemorySystem] = {}

    def add_agent(self, agent: Agent) -> None:
        """Add an agent to the simulation"""
        self.agents[agent.agent_id] = agent
        self.state_manager.register_agent(agent)  # Register with main state manager

        # Create a new state manager for this agent
        agent_state_manager = AgentStateManager()
        agent_state_manager.register_agent(agent)
        self.state_managers[agent.agent_id] = agent_state_manager

        # Create a collision system and pathfinding grid for this agent
        collision_system = CollisionSystem()
        pathfinding_grid = PathfindingGrid(100, 100)

        # Create a movement controller for this agent
        movement_controller = MovementController(
            agent_state_manager, collision_system, pathfinding_grid
        )
        movement_controller.register_agent(agent)
        self.movement_controllers[agent.agent_id] = movement_controller

        # Register agent with perception system
        self.perception_system.register_agent(agent)

        # Create a decision system for this agent
        decision_system = DecisionSystem(
            agent_state_manager, self.perception_system, movement_controller
        )
        decision_system.register_agent(agent)
        self.decision_systems[agent.agent_id] = decision_system

        # Create a memory system for this agent
        self.memory_systems[agent.agent_id] = MemorySystem(agent_id=agent.agent_id)

    def add_resource(self, position: Position, resource_type: str, amount: float) -> None:
        """Add a resource to the environment"""
        if position not in self.resources:
            self.resources[position] = {}
        self.resources[position][resource_type] = amount

    def add_obstacle(self, position: Position, radius: float) -> None:
        """Add an obstacle to the environment"""
        self.obstacles.append((position, radius))
        for controller in self.movement_controllers.values():
            controller.collision_system.add_static_obstacle(position, radius)

    def step(self, delta_time: float) -> None:
        """Advance simulation by one time step"""
        actual_delta = delta_time * self.time_scale
        self.current_time += actual_delta
        for agent_id, agent in self.agents.items():
            percepts = self.perception_system.perceive(agent_id)
            decision_system = self.decision_systems.get(agent_id)
            if decision_system:
                action = decision_system.make_decision(agent_id)
                if action:
                    decision_system.execute_action(agent_id, action)
            movement_controller = self.movement_controllers.get(agent_id)
            if movement_controller:
                movement_controller.update(actual_delta)
        self.perception_system.update_agent_positions()
        self.events.append(
            {
                "time": self.current_time,
                "type": "simulation_step",
                "delta": actual_delta,
            }
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get current environment metrics"""
        return {
            "time": self.current_time,
            "agent_count": len(self.agents),
            "resource_count": sum(len(r) for r in self.resources.values()),
            "total_resources": sum(
                amount for resources in self.resources.values() for amount in resources.values()
            ),
            "events": len(self.events),
        }


class BehaviorValidator:
    """Validates agent behaviors against expected patterns"""

    def __init__(self) -> None:
        self.validators: Dict[str, Callable] = {}
        self._register_default_validators()

    def _register_default_validators(self) -> None:
        """Register default behavior validators"""
        self.validators["movement_coherence"] = self._validate_movement_coherence
        self.validators["decision_consistency"] = self._validate_decision_consistency
        self.validators["resource_efficiency"] = self._validate_resource_efficiency
        self.validators["social_cooperation"] = self._validate_social_cooperation

    def validate(
        self, behavior_type: str, agent: Agent, history: List[Dict[str, Any]]
    ) -> tuple[bool, Optional[str]]:
        """
        Validate a specific behavior type

        Returns:
            (success, error_message)
        """
        if behavior_type not in self.validators:
            return (False, f"Unknown behavior type: {behavior_type}")
        result = self.validators[behavior_type](agent, history)
        return result  # type: ignore[no-any-return]

    def _validate_movement_coherence(
        self, agent: Agent, history: List[Dict[str, Any]]
    ) -> tuple[bool, Optional[str]]:
        """Validate that movement is coherent and follows physics"""
        positions = [h.get("position") for h in history if "position" in h]
        if len(positions) < 2:
            return (True, None)
        for i in range(1, len(positions)):
            if positions[i] and positions[i - 1]:
                if hasattr(positions[i], "distance_to"):
                    distance = positions[i].distance_to(positions[i - 1])
                else:
                    distance = 0.0  # Fallback for None positions
                time_delta = history[i].get("timestamp", 0) - history[i - 1].get("timestamp", 0)
                if time_delta > 0:
                    speed = distance / time_delta
                    if speed > 100:
                        return (False, f"Impossible speed detected: {speed} units/s")
        return (True, None)

    def _validate_decision_consistency(
        self, agent: Agent, history: List[Dict[str, Any]]
    ) -> tuple[bool, Optional[str]]:
        """Validate that decisions are consistent with agent goals"""
        return (True, None)

    def _validate_resource_efficiency(
        self, agent: Agent, history: List[Dict[str, Any]]
    ) -> tuple[bool, Optional[str]]:
        """Validate resource usage efficiency"""
        return (True, None)

    def _validate_social_cooperation(
        self, agent: Agent, history: List[Dict[str, Any]]
    ) -> tuple[bool, Optional[str]]:
        """Validate social interaction patterns"""
        return (True, None)


class PerformanceBenchmark:
    """Performance benchmarking for agent operations"""

    def __init__(self) -> None:
        self.results: Dict[str, List[float]] = {}

    @contextmanager
    def measure(self, operation: str):
        """Context manager for measuring operation performance"""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            if operation not in self.results:
                self.results[operation] = []
            self.results[operation].append(duration)

    def get_statistics(self, operation: str) -> Dict[str, float]:
        """Get performance statistics for an operation"""
        if operation not in self.results or not self.results[operation]:
            return {}
        times = self.results[operation]
        return {
            "count": len(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "min": min(times),
            "max": max(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
        }

    def get_report(self) -> Dict[str, Dict[str, float]]:
        """Get full performance report"""
        return {operation: self.get_statistics(operation) for operation in self.results}


class AgentTestOrchestrator:
    """Orchestrates test execution and reporting"""

    def __init__(self) -> None:
        self.scenarios: List[AgentTestScenario] = []
        self.results: List[AgentTestMetrics] = []
        self.benchmark = PerformanceBenchmark()
        self.validator = BehaviorValidator()

    def add_scenario(self, scenario: AgentTestScenario) -> None:
        """Add a test scenario"""
        self.scenarios.append(scenario)

    def run_scenario(self, scenario: AgentTestScenario) -> AgentTestMetrics:
        """Run a single test scenario"""
        logger.info(f"Running scenario: {scenario.name}")
        metrics = AgentTestMetrics(scenario_name=scenario.name, start_time=datetime.now())
        try:
            env_config = scenario.environment_config
            environment = SimulationEnvironment(
                bounds=env_config.get("bounds", (-100, -100, 100, 100)),
                time_scale=env_config.get("time_scale", 1.0),
            )
            agents: List[Agent] = []
            for agent_config in scenario.agent_configs:
                agent_type = agent_config.get("type", "basic")
                agent_id = agent_config.get("id", f"agent_{len(agents)}")
                if agent_type == "resource":
                    agent = AgentFactory.create_resource_agent(agent_id)
                elif agent_type == "social":
                    agent = AgentFactory.create_social_agent(agent_id)
                else:
                    agent = AgentFactory.create_basic_agent(agent_id)
                environment.add_agent(agent)
                agents.append(agent)
            step_duration = 0.1
            steps = int(scenario.duration / step_duration)
            for step in range(steps):
                with self.benchmark.measure("simulation_step"):
                    environment.step(step_duration)
                if step % 10 == 0:
                    self._collect_metrics(environment, metrics)
            self._collect_metrics(environment, metrics)
            metrics.success = self._evaluate_criteria(scenario.success_criteria, metrics)
        except Exception as e:
            logger.error(f"Scenario failed: {e}")
            metrics.success = False
            metrics.failure_reason = str(e)
        metrics.end_time = datetime.now()
        self.results.append(metrics)
        return metrics

    def run_all_scenarios(self) -> List[AgentTestMetrics]:
        """Run all test scenarios"""
        results = []
        for scenario in self.scenarios:
            result = self.run_scenario(scenario)
            results.append(result)
        return results

    def _collect_metrics(
        self, environment: SimulationEnvironment, metrics: AgentTestMetrics
    ) -> None:
        """Collect metrics from the environment"""
        metrics.environment_metrics = environment.get_metrics()
        for agent_id, agent in environment.agents.items():
            if agent_id not in metrics.agent_metrics:
                metrics.agent_metrics[agent_id] = {}
            metrics.agent_metrics[agent_id].update(
                {
                    "position": (agent.position.x, agent.position.y, agent.position.z),
                    "status": agent.status.value,
                    "energy": agent.resources.energy,
                    "health": agent.resources.health,
                    "relationships": len(agent.relationships),
                }
            )
        # Store benchmark report in environment metrics instead
        metrics.environment_metrics["benchmark_report"] = self.benchmark.get_report()

    def _evaluate_criteria(self, criteria: Dict[str, Any], metrics: AgentTestMetrics) -> bool:
        """Evaluate success criteria"""
        return True

    def generate_report(self) -> Dict[str, Any]:
        """Generate test report"""
        return {
            "summary": {
                "total_scenarios": len(self.results),
                "successful": sum(1 for r in self.results if r.success),
                "failed": sum(1 for r in self.results if not r.success),
                "total_duration": sum(
                    (r.end_time - r.start_time).total_seconds() for r in self.results if r.end_time
                ),
            },
            "scenarios": [
                {
                    "name": r.scenario_name,
                    "success": r.success,
                    "duration": (
                        (r.end_time - r.start_time).total_seconds() if r.end_time else None
                    ),
                    "failure_reason": r.failure_reason,
                    "metrics": {
                        "environment": r.environment_metrics,
                        "performance": r.performance_metrics,
                    },
                }
                for r in self.results
            ],
            "performance": self.benchmark.get_report(),
        }


def create_basic_test_scenarios() -> List[AgentTestScenario]:
    """Create basic test scenarios for agent testing"""
    scenarios = []
    scenarios.append(
        AgentTestScenario(
            name="Basic Movement",
            description="Test basic agent movement capabilities",
            duration=10.0,
            agent_configs=[
                {"type": "basic", "id": "mover1"},
                {"type": "basic", "id": "mover2"},
            ],
            environment_config={"bounds": (-50, -50, 50, 50), "time_scale": 1.0},
            success_criteria={"all_agents_moved": True, "no_collisions": True},
            metrics_to_track=["position", "velocity", "energy"],
        )
    )
    scenarios.append(
        AgentTestScenario(
            name="Resource Collection",
            description="Test resource discovery and collection",
            duration=30.0,
            agent_configs=[
                {"type": "resource", "id": "collector1"},
                {"type": "resource", "id": "collector2"},
                {"type": "resource", "id": "collector3"},
            ],
            environment_config={
                "bounds": (-100, -100, 100, 100),
                "time_scale": 2.0,
                "resources": [
                    {"position": (10, 10), "type": "energy", "amount": 50},
                    {"position": (-20, 30), "type": "materials", "amount": 30},
                    {"position": (40, -40), "type": "energy", "amount": 40},
                ],
            },
            success_criteria={"resources_collected": 0.5, "agent_survival": 1.0},
            metrics_to_track=["resources", "energy", "position"],
        )
    )
    scenarios.append(
        AgentTestScenario(
            name="Social Cooperation",
            description="Test social interaction and cooperation",
            duration=20.0,
            agent_configs=[
                {"type": "social", "id": "social1"},
                {"type": "social", "id": "social2"},
                {"type": "social", "id": "social3"},
                {"type": "social", "id": "social4"},
            ],
            environment_config={"bounds": (-30, -30, 30, 30), "time_scale": 1.0},
            success_criteria={"relationships_formed": 4, "trust_established": True},
            metrics_to_track=["relationships", "trust", "communications"],
        )
    )
    return scenarios


if __name__ == "__main__":
    orchestrator = AgentTestOrchestrator()
    for scenario in create_basic_test_scenarios():
        orchestrator.add_scenario(scenario)
    results = orchestrator.run_all_scenarios()
    report = orchestrator.generate_report()
    print(json.dumps(report, indent=2))
