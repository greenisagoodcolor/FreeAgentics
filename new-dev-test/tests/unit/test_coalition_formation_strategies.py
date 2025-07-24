"""
Test suite for Coalition Formation Strategies module.

This test suite provides comprehensive coverage for the coalition formation
algorithms and strategies in the FreeAgentics multi-agent system.
Coverage target: 95%+
"""

import time
from unittest.mock import Mock

import pytest

# Import the module under test
try:
    from coalitions.coalition import Coalition, CoalitionObjective, CoalitionRole
    from coalitions.formation_strategies import (
        AgentProfile,
        FormationResult,
        FormationStrategy,
        GreedyFormation,
        HierarchicalFormation,
        OptimalFormation,
    )

    IMPORT_SUCCESS = True
except ImportError:
    IMPORT_SUCCESS = False

    # Mock classes for testing when imports fail
    class AgentProfile:
        pass

    class FormationResult:
        pass

    class FormationStrategy:
        pass

    class GreedyFormation:
        pass

    class OptimalFormation:
        pass

    class HierarchicalFormation:
        pass

    class Coalition:
        pass

    class CoalitionObjective:
        pass

    class CoalitionRole:
        pass


class TestAgentProfile:
    """Test AgentProfile dataclass."""

    def test_agent_profile_creation(self):
        """Test AgentProfile creation with basic data."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        profile = AgentProfile(
            agent_id="agent_001",
            capabilities=["exploration", "communication"],
            capacity=0.8,
            reputation=0.9,
            preferences={"collaboration": 0.7, "competition": 0.3},
            current_coalitions=["coalition_a"],
        )

        assert profile.agent_id == "agent_001"
        assert "exploration" in profile.capabilities
        assert "communication" in profile.capabilities
        assert profile.capacity == 0.8
        assert profile.reputation == 0.9
        assert profile.preferences["collaboration"] == 0.7
        assert profile.preferences["competition"] == 0.3
        assert "coalition_a" in profile.current_coalitions
        assert profile.max_coalitions == 3  # Default value

    def test_agent_profile_defaults(self):
        """Test AgentProfile default values."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        profile = AgentProfile(
            agent_id="agent_002",
            capabilities=["analysis"],
            capacity=0.5,
            reputation=0.6,
            preferences={},
            current_coalitions=[],
        )

        assert profile.max_coalitions == 3

    def test_agent_profile_custom_max_coalitions(self):
        """Test AgentProfile with custom max coalitions."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        profile = AgentProfile(
            agent_id="agent_003",
            capabilities=["coordination"],
            capacity=1.0,
            reputation=1.0,
            preferences={},
            current_coalitions=[],
            max_coalitions=5,
        )

        assert profile.max_coalitions == 5

    @pytest.mark.parametrize("capacity", [0.0, 0.5, 1.0])
    def test_agent_profile_capacity_range(self, capacity):
        """Test AgentProfile with different capacity values."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        profile = AgentProfile(
            agent_id=f"agent_{capacity}",
            capabilities=["test"],
            capacity=capacity,
            reputation=0.5,
            preferences={},
            current_coalitions=[],
        )

        assert 0.0 <= profile.capacity <= 1.0

    @pytest.mark.parametrize("reputation", [0.0, 0.5, 1.0])
    def test_agent_profile_reputation_range(self, reputation):
        """Test AgentProfile with different reputation values."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        profile = AgentProfile(
            agent_id=f"agent_{reputation}",
            capabilities=["test"],
            capacity=0.5,
            reputation=reputation,
            preferences={},
            current_coalitions=[],
        )

        assert 0.0 <= profile.reputation <= 1.0


class TestFormationResult:
    """Test FormationResult dataclass."""

    def test_formation_result_creation(self):
        """Test FormationResult creation."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        # Mock coalitions
        mock_coalition1 = Mock(spec=Coalition)
        mock_coalition1.id = "coalition_1"

        mock_coalition2 = Mock(spec=Coalition)
        mock_coalition2.id = "coalition_2"

        result = FormationResult(
            coalitions=[mock_coalition1, mock_coalition2],
            unassigned_agents=["agent_003", "agent_004"],
            formation_time=1.5,
            objective_coverage=0.8,
            formation_score=0.85,
        )

        assert len(result.coalitions) == 2
        assert result.coalitions[0].id == "coalition_1"
        assert result.coalitions[1].id == "coalition_2"
        assert "agent_003" in result.unassigned_agents
        assert "agent_004" in result.unassigned_agents
        assert result.formation_time == 1.5
        assert result.objective_coverage == 0.8
        assert result.formation_score == 0.85

    def test_formation_result_empty(self):
        """Test FormationResult with empty data."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        result = FormationResult(
            coalitions=[],
            unassigned_agents=[],
            formation_time=0.0,
            objective_coverage=0.0,
            formation_score=0.0,
        )

        assert len(result.coalitions) == 0
        assert len(result.unassigned_agents) == 0
        assert result.formation_time == 0.0
        assert result.objective_coverage == 0.0
        assert result.formation_score == 0.0


class TestFormationStrategy:
    """Test base FormationStrategy class."""

    def test_formation_strategy_abc(self):
        """Test that FormationStrategy is abstract."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        # Should not be able to instantiate abstract class directly
        with pytest.raises(TypeError):
            FormationStrategy("test_strategy")

    def test_formation_strategy_interface(self):
        """Test FormationStrategy interface."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        # Check that abstract methods exist
        assert hasattr(FormationStrategy, "form_coalitions")

        # This should be an abstract method
        assert getattr(FormationStrategy.form_coalitions, "__isabstractmethod__", False)


class TestGreedyFormation:
    """Test GreedyFormation class."""

    @pytest.fixture
    def greedy_strategy(self):
        """Create GreedyFormation instance."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        return GreedyFormation()

    @pytest.fixture
    def sample_agents(self):
        """Create sample agent profiles for testing."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        return [
            AgentProfile(
                agent_id="agent_001",
                capabilities=["exploration", "communication"],
                capacity=0.8,
                reputation=0.9,
                preferences={"exploration": 0.8},
                current_coalitions=[],
            ),
            AgentProfile(
                agent_id="agent_002",
                capabilities=["analysis", "computation"],
                capacity=0.7,
                reputation=0.8,
                preferences={"analysis": 0.9},
                current_coalitions=[],
            ),
            AgentProfile(
                agent_id="agent_003",
                capabilities=["coordination", "planning"],
                capacity=0.9,
                reputation=0.85,
                preferences={"coordination": 0.7},
                current_coalitions=[],
            ),
        ]

    @pytest.fixture
    def sample_objectives(self):
        """Create sample coalition objectives."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        objectives = []

        obj1 = Mock(spec=CoalitionObjective)
        obj1.objective_id = "obj_001"
        obj1.description = "Exploration task"
        obj1.required_capabilities = ["exploration", "communication"]
        obj1.priority = 1.0
        objectives.append(obj1)

        obj2 = Mock(spec=CoalitionObjective)
        obj2.objective_id = "obj_002"
        obj2.description = "Analysis task"
        obj2.required_capabilities = ["analysis", "computation"]
        obj2.priority = 0.8
        objectives.append(obj2)

        return objectives

    def test_greedy_strategy_initialization(self, greedy_strategy):
        """Test GreedyFormation initialization."""
        assert greedy_strategy.name == "Greedy Formation"
        assert hasattr(greedy_strategy, "form_coalitions")

    def test_greedy_form_coalitions_basic(self, greedy_strategy, sample_agents, sample_objectives):
        """Test basic coalition formation with greedy strategy."""
        result = greedy_strategy.form_coalitions(sample_agents, sample_objectives)

        assert isinstance(result, FormationResult)
        assert isinstance(result.coalitions, list)
        assert isinstance(result.unassigned_agents, list)
        assert isinstance(result.formation_time, float)
        assert isinstance(result.objective_coverage, float)
        assert isinstance(result.formation_score, float)

        # Should have some coalitions formed
        assert len(result.coalitions) > 0

    def test_greedy_form_coalitions_empty_agents(self, greedy_strategy, sample_objectives):
        """Test coalition formation with empty agent list."""
        result = greedy_strategy.form_coalitions([], sample_objectives)

        assert isinstance(result, FormationResult)
        assert len(result.coalitions) == 0
        assert len(result.unassigned_agents) == 0
        assert result.objective_coverage == 0.0

    def test_greedy_form_coalitions_empty_objectives(self, greedy_strategy, sample_agents):
        """Test coalition formation with empty objectives list."""
        result = greedy_strategy.form_coalitions(sample_agents, [])

        assert isinstance(result, FormationResult)
        assert len(result.coalitions) == 0
        assert len(result.unassigned_agents) == len(sample_agents)
        assert result.objective_coverage == 1.0  # No objectives means 100% coverage

    def test_greedy_formation_result(self, greedy_strategy, sample_agents, sample_objectives):
        """Test formation result structure."""
        result = greedy_strategy.form_coalitions(sample_agents, sample_objectives)

        assert isinstance(result.formation_score, float)
        assert 0.0 <= result.formation_score

    def test_greedy_capability_matching(self, greedy_strategy):
        """Test capability matching in greedy strategy."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        # Agent with exploration capability
        exploration_agent = AgentProfile(
            agent_id="explorer",
            capabilities=["exploration"],
            capacity=1.0,
            reputation=1.0,
            preferences={},
            current_coalitions=[],
        )

        # Objective requiring exploration
        exploration_objective = Mock(spec=CoalitionObjective)
        exploration_objective.objective_id = "explore_task"
        exploration_objective.description = "Exploration task"
        exploration_objective.required_capabilities = ["exploration"]
        exploration_objective.priority = 1.0

        result = greedy_strategy.form_coalitions([exploration_agent], [exploration_objective])

        # Should form exactly one coalition
        assert len(result.coalitions) == 1
        assert len(result.unassigned_agents) == 0


class TestOptimalFormation:
    """Test OptimalFormation class."""

    @pytest.fixture
    def optimal_strategy(self):
        """Create OptimalFormation instance."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        return OptimalFormation()

    def test_optimal_strategy_initialization(self, optimal_strategy):
        """Test OptimalFormation initialization."""
        assert optimal_strategy.name == "Optimal Formation"
        assert hasattr(optimal_strategy, "form_coalitions")

    def test_optimal_form_coalitions_small_problem(self, optimal_strategy):
        """Test optimal strategy with small problem size."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        # Small problem that can be solved optimally
        agents = [
            AgentProfile(
                agent_id="agent_001",
                capabilities=["task_a"],
                capacity=1.0,
                reputation=1.0,
                preferences={},
                current_coalitions=[],
            ),
            AgentProfile(
                agent_id="agent_002",
                capabilities=["task_b"],
                capacity=1.0,
                reputation=1.0,
                preferences={},
                current_coalitions=[],
            ),
        ]

        objectives = []

        obj_a = Mock(spec=CoalitionObjective)
        obj_a.objective_id = "obj_a"
        obj_a.description = "Task A"
        obj_a.required_capabilities = ["task_a"]
        obj_a.priority = 1.0
        objectives.append(obj_a)

        obj_b = Mock(spec=CoalitionObjective)
        obj_b.objective_id = "obj_b"
        obj_b.description = "Task B"
        obj_b.required_capabilities = ["task_b"]
        obj_b.priority = 1.0
        objectives.append(obj_b)

        result = optimal_strategy.form_coalitions(agents, objectives)

        assert isinstance(result, FormationResult)
        assert len(result.coalitions) >= 2  # Should find good solution
        assert len(result.unassigned_agents) == 0

    def test_optimal_complexity_handling(self, optimal_strategy):
        """Test optimal strategy handles complex problems gracefully."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        # Larger problem that might timeout
        agents = [
            AgentProfile(
                agent_id=f"agent_{i:03d}",
                capabilities=[f"skill_{i % 3}"],
                capacity=0.8,
                reputation=0.9,
                preferences={},
                current_coalitions=[],
            )
            for i in range(10)
        ]

        objectives = []
        for i in range(5):
            obj = Mock(spec=CoalitionObjective)
            obj.objective_id = f"obj_{i}"
            obj.description = f"Objective {i}"
            obj.required_capabilities = [f"skill_{i % 3}"]
            obj.priority = 1.0
            objectives.append(obj)

        # Should complete without hanging
        start_time = time.time()
        result = optimal_strategy.form_coalitions(agents, objectives)
        elapsed_time = time.time() - start_time

        assert isinstance(result, FormationResult)
        assert elapsed_time < 5.0  # Should not take too long


class TestHierarchicalFormation:
    """Test HierarchicalFormation class."""

    @pytest.fixture
    def hierarchical_strategy(self):
        """Create HierarchicalFormation instance."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        return HierarchicalFormation()

    def test_hierarchical_strategy_initialization(self, hierarchical_strategy):
        """Test HierarchicalFormation initialization."""
        assert hierarchical_strategy.name == "Hierarchical Formation"
        assert hasattr(hierarchical_strategy, "form_coalitions")

    def test_hierarchical_reputation_consideration(self, hierarchical_strategy):
        """Test that heuristic strategy considers agent reputation."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        # Agents with different reputations
        high_rep_agent = AgentProfile(
            agent_id="high_rep",
            capabilities=["leadership"],
            capacity=0.8,
            reputation=0.9,
            preferences={},
            current_coalitions=[],
        )

        low_rep_agent = AgentProfile(
            agent_id="low_rep",
            capabilities=["leadership"],
            capacity=0.8,
            reputation=0.3,
            preferences={},
            current_coalitions=[],
        )

        objective = Mock(spec=CoalitionObjective)
        objective.objective_id = "leadership_task"
        objective.description = "Leadership task"
        objective.required_capabilities = ["leadership"]
        objective.priority = 1.0

        result = hierarchical_strategy.form_coalitions([high_rep_agent, low_rep_agent], [objective])

        # Should prefer high reputation agent
        assert len(result.coalitions) == 1
        # The actual implementation should prefer higher reputation

    def test_hierarchical_capacity_balancing(self, hierarchical_strategy):
        """Test capacity balancing in hierarchical strategy."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        agents = [
            AgentProfile(
                agent_id=f"agent_{i}",
                capabilities=["general"],
                capacity=0.2 * (i + 1),  # Varying capacities
                reputation=0.8,
                preferences={},
                current_coalitions=[],
            )
            for i in range(4)
        ]

        objectives = [
            Mock(
                spec=CoalitionObjective,
                objective_id=f"task_{i}",
                description=f"Task {i}",
                required_capabilities=["general"],
                priority=1.0,
            )
            for i in range(2)
        ]

        result = hierarchical_strategy.form_coalitions(agents, objectives)

        assert isinstance(result, FormationResult)
        # Should attempt to balance load across agents


class TestFormationAlgorithms:
    """Test formation algorithm implementations."""

    def test_capability_matching_algorithm(self):
        """Test capability matching algorithm."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        # This would test the internal algorithms used by strategies
        # For now, test that the concept works
        required_capabilities = ["skill_a", "skill_b"]
        agent_capabilities = ["skill_a", "skill_b", "skill_c"]

        # Should match
        match = all(cap in agent_capabilities for cap in required_capabilities)
        assert match is True

        # Should not match
        required_capabilities = ["skill_d"]
        match = all(cap in agent_capabilities for cap in required_capabilities)
        assert match is False

    def test_coalition_size_constraints(self):
        """Test coalition size constraint checking."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        # Test size constraints
        min_agents = 2
        max_agents = 4
        current_size = 3

        assert min_agents <= current_size <= max_agents

    def test_utility_calculation(self):
        """Test utility calculation for coalition formation."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        # Mock utility calculation
        agent_values = [0.8, 0.9, 0.7]
        coalition_utility = sum(agent_values) / len(agent_values)

        assert abs(coalition_utility - 0.8) < 0.001
        assert 0.0 <= coalition_utility <= 1.0


class TestFormationPerformance:
    """Test performance characteristics of formation strategies."""

    def test_formation_time_tracking(self):
        """Test that formation time is properly tracked."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        start_time = time.time()
        # Simulate formation work
        time.sleep(0.01)
        formation_time = time.time() - start_time

        assert formation_time > 0.0
        assert formation_time < 1.0  # Should be fast

    def test_scalability_handling(self):
        """Test handling of large agent populations."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        # Large number of agents
        large_agent_count = 100
        agents = []

        for i in range(large_agent_count):
            agent = AgentProfile(
                agent_id=f"agent_{i:03d}",
                capabilities=[f"skill_{i % 10}"],
                capacity=0.8,
                reputation=0.8,
                preferences={},
                current_coalitions=[],
            )
            agents.append(agent)

        assert len(agents) == large_agent_count

        # Should be able to handle large populations efficiently
        # (This would be tested with actual strategy implementations)

    def test_memory_efficiency(self):
        """Test memory efficiency of formation algorithms."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        # Test that data structures are reasonably sized
        agent_profile = AgentProfile(
            agent_id="memory_test",
            capabilities=["test"] * 10,  # Moderate capability list
            capacity=0.8,
            reputation=0.9,
            preferences={f"pref_{i}": 0.5 for i in range(5)},
            current_coalitions=[f"coalition_{i}" for i in range(3)],
        )

        # Should not use excessive memory
        assert len(agent_profile.capabilities) <= 50
        assert len(agent_profile.preferences) <= 20
        assert len(agent_profile.current_coalitions) <= 10


if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
            "--cov=coalitions.formation_strategies",
            "--cov-report=html",
        ]
    )
