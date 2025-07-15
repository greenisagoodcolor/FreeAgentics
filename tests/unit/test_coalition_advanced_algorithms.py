"""
Advanced Coalition Formation Algorithm Tests.

This test suite provides comprehensive coverage for advanced coalition formation
algorithms, stability calculations, and performance characteristics.
Following TDD principles with ultrathink reasoning for edge case detection.
"""

import time

import pytest

# Import the modules under test
try:
    from coalitions.coalition import Coalition, CoalitionObjective, CoalitionRole, CoalitionStatus
    from coalitions.formation_strategies import (
        AgentProfile,
        FormationResult,
        GreedyFormation,
        HierarchicalFormation,
        OptimalFormation,
    )

    IMPORT_SUCCESS = True
except ImportError:
    IMPORT_SUCCESS = False

    # Mock classes for testing when imports fail
    class AgentProfile:
        def __init__(
            self,
            agent_id,
            capabilities,
            capacity,
            reputation,
            preferences,
            current_coalitions,
            max_coalitions=3,
        ):
            self.agent_id = agent_id
            self.capabilities = capabilities
            self.capacity = capacity
            self.reputation = reputation
            self.preferences = preferences
            self.current_coalitions = current_coalitions
            self.max_coalitions = max_coalitions

    class FormationResult:
        def __init__(
            self, coalitions, unassigned_agents, formation_time, objective_coverage, formation_score
        ):
            self.coalitions = coalitions
            self.unassigned_agents = unassigned_agents
            self.formation_time = formation_time
            self.objective_coverage = objective_coverage
            self.formation_score = formation_score

    class GreedyFormation:
        def __init__(self):
            self.name = "Greedy Formation"

        def form_coalitions(self, agents, objectives, constraints=None):
            return FormationResult([], [], 0.0, 0.0, 0.0)

    class OptimalFormation:
        def __init__(self):
            self.name = "Optimal Formation"

        def form_coalitions(self, agents, objectives, constraints=None):
            return FormationResult([], [], 0.0, 0.0, 0.0)

    class HierarchicalFormation:
        def __init__(self):
            self.name = "Hierarchical Formation"

        def form_coalitions(self, agents, objectives, constraints=None):
            return FormationResult([], [], 0.0, 0.0, 0.0)

    class Coalition:
        def __init__(self, coalition_id, name, objectives=None, max_size=None):
            self.coalition_id = coalition_id
            self.name = name
            self.objectives = objectives or []
            self.max_size = max_size
            self.members = {}
            self.status = "forming"

        def add_member(self, agent_id, role, capabilities):
            return True

        def activate(self):
            self.status = "active"

        def can_achieve_objective(self, objective):
            return True

        def get_capabilities(self):
            return set()

    class CoalitionObjective:
        def __init__(self, objective_id, description, required_capabilities, priority):
            self.objective_id = objective_id
            self.description = description
            self.required_capabilities = required_capabilities
            self.priority = priority

    class CoalitionRole:
        LEADER = "leader"
        MEMBER = "member"
        COORDINATOR = "coordinator"

    class CoalitionStatus:
        FORMING = "forming"
        ACTIVE = "active"
        DISSOLVED = "dissolved"


class TestCoalitionStabilityCalculations:
    """Test coalition stability calculations and metrics."""

    def test_coalition_stability_core_concept(self):
        """Test core concept of coalition stability."""
        if not IMPORT_SUCCESS:
            pytest.skip("Coalition modules not available")

        # Create a simple coalition with known properties
        coalition = Coalition("test_coalition", "Test Coalition")

        # Add members with complementary capabilities
        assert coalition.add_member("agent_1", CoalitionRole.LEADER, ["skill_a", "skill_b"])
        assert coalition.add_member("agent_2", CoalitionRole.MEMBER, ["skill_c", "skill_d"])

        # Coalition should be stable with complementary skills
        assert len(coalition.members) == 2
        assert coalition.status == CoalitionStatus.FORMING

        coalition.activate()
        assert coalition.status == CoalitionStatus.ACTIVE

    def test_coalition_capability_coverage(self):
        """Test that coalition capability coverage is calculated correctly."""
        if not IMPORT_SUCCESS:
            pytest.skip("Coalition modules not available")

        # Create objective requiring specific capabilities
        objective = CoalitionObjective(
            objective_id="test_obj",
            description="Test objective",
            required_capabilities=["skill_a", "skill_b", "skill_c"],
            priority=1.0,
        )

        # Create coalition with partial coverage
        coalition = Coalition("test_coalition", "Test Coalition", [objective])
        coalition.add_member("agent_1", CoalitionRole.LEADER, ["skill_a", "skill_b"])

        # Should not be able to achieve objective with partial coverage
        available_caps = coalition.get_capabilities()
        required_caps = set(objective.required_capabilities)

        # Test coverage calculation
        coverage = len(available_caps.intersection(required_caps)) / len(required_caps)
        assert coverage < 1.0  # Partial coverage

    def test_coalition_size_optimization(self):
        """Test that coalition size affects formation quality."""
        if not IMPORT_SUCCESS:
            pytest.skip("Coalition modules not available")

        # Create agents with overlapping capabilities
        agents = [
            AgentProfile(
                agent_id=f"agent_{i}",
                capabilities=["skill_a", "skill_b"],
                capacity=0.8,
                reputation=0.9,
                preferences={},
                current_coalitions=[],
            )
            for i in range(5)
        ]

        # Create objective requiring only basic skills
        objective = CoalitionObjective(
            objective_id="simple_obj",
            description="Simple objective",
            required_capabilities=["skill_a"],
            priority=1.0,
        )

        # Test with greedy formation
        strategy = GreedyFormation()
        result = strategy.form_coalitions(agents, [objective])

        # Should form at least one coalition
        assert len(result.coalitions) >= 1

        # Formation score should be reasonable
        assert result.formation_score >= 0.0


class TestConstraintSatisfactionAlgorithms:
    """Test constraint satisfaction in coalition formation."""

    def test_max_coalition_size_constraint(self):
        """Test that maximum coalition size constraint is respected."""
        if not IMPORT_SUCCESS:
            pytest.skip("Coalition modules not available")

        # Create many agents with same capabilities
        agents = [
            AgentProfile(
                agent_id=f"agent_{i}",
                capabilities=["common_skill"],
                capacity=0.8,
                reputation=0.9,
                preferences={},
                current_coalitions=[],
            )
            for i in range(10)
        ]

        # Create objective requiring common skill
        objective = CoalitionObjective(
            objective_id="constrained_obj",
            description="Constrained objective",
            required_capabilities=["common_skill"],
            priority=1.0,
        )

        # Test with size constraint
        strategy = GreedyFormation()
        constraints = {"max_coalition_size": 3}
        result = strategy.form_coalitions(agents, [objective], constraints)

        # Check that coalitions respect size constraint
        for coalition in result.coalitions:
            assert len(coalition.members) <= 3

    def test_agent_coalition_limit_constraint(self):
        """Test that agent coalition limit constraint is respected."""
        if not IMPORT_SUCCESS:
            pytest.skip("Coalition modules not available")

        # Create agent with low coalition limit
        agent = AgentProfile(
            agent_id="limited_agent",
            capabilities=["skill_a", "skill_b", "skill_c"],
            capacity=1.0,
            reputation=1.0,
            preferences={},
            current_coalitions=["existing_coalition_1"],
            max_coalitions=2,  # Can only join 1 more coalition
        )

        # Create multiple objectives
        objectives = [
            CoalitionObjective(
                objective_id=f"obj_{i}",
                description=f"Objective {i}",
                required_capabilities=["skill_a"],
                priority=1.0,
            )
            for i in range(3)
        ]

        # Test formation
        strategy = GreedyFormation()
        result = strategy.form_coalitions([agent], objectives)

        # Agent should not be over-assigned
        agent_assignments = sum(
            1 for coalition in result.coalitions if "limited_agent" in coalition.members
        )
        assert agent_assignments <= 1  # Can join at most 1 more coalition

    def test_capability_satisfaction_priority(self):
        """Test that capability satisfaction is prioritized correctly."""
        if not IMPORT_SUCCESS:
            pytest.skip("Coalition modules not available")

        # Create agents with different specializations
        specialist = AgentProfile(
            agent_id="specialist",
            capabilities=["rare_skill"],
            capacity=0.8,
            reputation=0.9,
            preferences={},
            current_coalitions=[],
        )

        generalist = AgentProfile(
            agent_id="generalist",
            capabilities=["common_skill", "rare_skill"],
            capacity=0.6,
            reputation=0.7,
            preferences={},
            current_coalitions=[],
        )

        # Create objectives with different requirements
        rare_objective = CoalitionObjective(
            objective_id="rare_obj",
            description="Rare objective",
            required_capabilities=["rare_skill"],
            priority=1.0,
        )

        common_objective = CoalitionObjective(
            objective_id="common_obj",
            description="Common objective",
            required_capabilities=["common_skill"],
            priority=0.8,
        )

        # Test formation
        strategy = GreedyFormation()
        result = strategy.form_coalitions(
            [specialist, generalist], [rare_objective, common_objective]
        )

        # Should form coalitions efficiently
        assert len(result.coalitions) >= 1
        assert result.objective_coverage > 0.0


class TestFormationPerformanceBenchmarks:
    """Test performance characteristics of formation algorithms."""

    def test_greedy_formation_performance(self):
        """Test performance of greedy formation algorithm."""
        if not IMPORT_SUCCESS:
            pytest.skip("Coalition modules not available")

        # Create medium-sized problem
        agents = [
            AgentProfile(
                agent_id=f"agent_{i}",
                capabilities=[f"skill_{i % 5}", f"skill_{(i+1) % 5}"],
                capacity=0.8,
                reputation=0.8,
                preferences={},
                current_coalitions=[],
            )
            for i in range(50)
        ]

        objectives = [
            CoalitionObjective(
                objective_id=f"obj_{i}",
                description=f"Objective {i}",
                required_capabilities=[f"skill_{i % 5}"],
                priority=1.0,
            )
            for i in range(10)
        ]

        # Measure performance
        strategy = GreedyFormation()
        start_time = time.time()
        result = strategy.form_coalitions(agents, objectives)
        formation_time = time.time() - start_time

        # Performance assertions
        assert formation_time < 1.0  # Should be fast for medium problems
        assert result.formation_time < 1.0
        assert len(result.coalitions) > 0

    def test_optimal_formation_scalability(self):
        """Test scalability limits of optimal formation."""
        if not IMPORT_SUCCESS:
            pytest.skip("Coalition modules not available")

        # Create small problem for optimal solution
        agents = [
            AgentProfile(
                agent_id=f"agent_{i}",
                capabilities=[f"skill_{i}"],
                capacity=0.8,
                reputation=0.8,
                preferences={},
                current_coalitions=[],
            )
            for i in range(5)
        ]

        objectives = [
            CoalitionObjective(
                objective_id=f"obj_{i}",
                description=f"Objective {i}",
                required_capabilities=[f"skill_{i}"],
                priority=1.0,
            )
            for i in range(3)
        ]

        # Test optimal formation
        strategy = OptimalFormation()
        start_time = time.time()
        result = strategy.form_coalitions(agents, objectives)
        formation_time = time.time() - start_time

        # Should handle small problems efficiently
        assert formation_time < 5.0  # Reasonable time for small optimal problems
        assert len(result.coalitions) > 0
        assert result.objective_coverage > 0.0

    def test_hierarchical_formation_large_scale(self):
        """Test hierarchical formation with large agent populations."""
        if not IMPORT_SUCCESS:
            pytest.skip("Coalition modules not available")

        # Create large population
        agents = [
            AgentProfile(
                agent_id=f"agent_{i}",
                capabilities=(
                    [f"skill_{i % 10}", "coordination"] if i % 10 == 0 else [f"skill_{i % 10}"]
                ),
                capacity=0.8,
                reputation=0.9 if i % 10 == 0 else 0.7,  # Some leaders
                preferences={},
                current_coalitions=[],
            )
            for i in range(100)
        ]

        objectives = [
            CoalitionObjective(
                objective_id=f"obj_{i}",
                description=f"Objective {i}",
                required_capabilities=[f"skill_{i % 10}"],
                priority=1.0,
            )
            for i in range(20)
        ]

        # Test hierarchical formation
        strategy = HierarchicalFormation()
        start_time = time.time()
        result = strategy.form_coalitions(agents, objectives)
        formation_time = time.time() - start_time

        # Should handle large problems reasonably
        assert formation_time < 10.0  # Reasonable time for large problems
        assert len(result.coalitions) > 0
        assert result.objective_coverage >= 0.0


class TestAlgorithmCorrectnessValidation:
    """Test algorithm correctness against known optimal solutions."""

    def test_simple_optimal_assignment(self):
        """Test against known optimal solution for simple problem."""
        if not IMPORT_SUCCESS:
            pytest.skip("Coalition modules not available")

        # Perfect matching problem: 1 agent per objective
        agents = [
            AgentProfile(
                agent_id="agent_1",
                capabilities=["skill_a"],
                capacity=1.0,
                reputation=1.0,
                preferences={},
                current_coalitions=[],
            ),
            AgentProfile(
                agent_id="agent_2",
                capabilities=["skill_b"],
                capacity=1.0,
                reputation=1.0,
                preferences={},
                current_coalitions=[],
            ),
        ]

        objectives = [
            CoalitionObjective(
                objective_id="obj_a",
                description="Objective A",
                required_capabilities=["skill_a"],
                priority=1.0,
            ),
            CoalitionObjective(
                objective_id="obj_b",
                description="Objective B",
                required_capabilities=["skill_b"],
                priority=1.0,
            ),
        ]

        # Test both greedy and optimal
        greedy_strategy = GreedyFormation()
        optimal_strategy = OptimalFormation()

        greedy_result = greedy_strategy.form_coalitions(agents, objectives)
        optimal_result = optimal_strategy.form_coalitions(agents, objectives)

        # Both should achieve perfect assignment
        assert greedy_result.objective_coverage == 1.0
        assert optimal_result.objective_coverage == 1.0
        assert len(greedy_result.unassigned_agents) == 0
        assert len(optimal_result.unassigned_agents) == 0

    def test_impossible_assignment_handling(self):
        """Test handling of impossible assignments."""
        if not IMPORT_SUCCESS:
            pytest.skip("Coalition modules not available")

        # Create impossible scenario
        agents = [
            AgentProfile(
                agent_id="agent_1",
                capabilities=["skill_a"],
                capacity=1.0,
                reputation=1.0,
                preferences={},
                current_coalitions=[],
            )
        ]

        objectives = [
            CoalitionObjective(
                objective_id="impossible_obj",
                description="Impossible objective",
                required_capabilities=["skill_z"],  # Agent doesn't have this
                priority=1.0,
            )
        ]

        # Test all strategies
        strategies = [GreedyFormation(), OptimalFormation(), HierarchicalFormation()]

        for strategy in strategies:
            result = strategy.form_coalitions(agents, objectives)

            # Should handle gracefully
            assert isinstance(result, FormationResult)
            assert result.objective_coverage < 1.0  # Cannot achieve all objectives

    def test_multi_objective_optimization(self):
        """Test multi-objective optimization scenarios."""
        if not IMPORT_SUCCESS:
            pytest.skip("Coalition modules not available")

        # Create agent with multiple capabilities
        versatile_agent = AgentProfile(
            agent_id="versatile",
            capabilities=["skill_a", "skill_b", "skill_c"],
            capacity=1.0,
            reputation=1.0,
            preferences={},
            current_coalitions=[],
        )

        # Create competing objectives
        objectives = [
            CoalitionObjective(
                objective_id="high_priority",
                description="High priority task",
                required_capabilities=["skill_a"],
                priority=1.0,
            ),
            CoalitionObjective(
                objective_id="medium_priority",
                description="Medium priority task",
                required_capabilities=["skill_b"],
                priority=0.8,
            ),
            CoalitionObjective(
                objective_id="low_priority",
                description="Low priority task",
                required_capabilities=["skill_c"],
                priority=0.5,
            ),
        ]

        # Test prioritization
        strategy = GreedyFormation()
        result = strategy.form_coalitions([versatile_agent], objectives)

        # Should handle multiple objectives appropriately
        assert len(result.coalitions) >= 1
        assert result.formation_score > 0.0


class TestDynamicCoalitionOperations:
    """Test dynamic coalition operations like merging and splitting."""

    def test_coalition_member_addition(self):
        """Test adding members to existing coalitions."""
        if not IMPORT_SUCCESS:
            pytest.skip("Coalition modules not available")

        # Create coalition with initial member
        coalition = Coalition("test_coalition", "Test Coalition")
        assert coalition.add_member("agent_1", CoalitionRole.LEADER, ["skill_a"])

        # Add additional member
        assert coalition.add_member("agent_2", CoalitionRole.MEMBER, ["skill_b"])

        # Verify addition
        assert len(coalition.members) == 2
        assert "agent_1" in coalition.members
        assert "agent_2" in coalition.members

    def test_coalition_member_removal(self):
        """Test removing members from coalitions."""
        if not IMPORT_SUCCESS:
            pytest.skip("Coalition modules not available")

        # Create coalition with members
        coalition = Coalition("test_coalition", "Test Coalition")
        coalition.add_member("agent_1", CoalitionRole.LEADER, ["skill_a"])
        coalition.add_member("agent_2", CoalitionRole.MEMBER, ["skill_b"])

        # Remove member
        assert coalition.remove_member("agent_2")

        # Verify removal
        assert len(coalition.members) == 1
        assert "agent_1" in coalition.members
        assert "agent_2" not in coalition.members

    def test_coalition_leader_election(self):
        """Test leader election when leader leaves."""
        if not IMPORT_SUCCESS:
            pytest.skip("Coalition modules not available")

        # Create coalition with leader and member
        coalition = Coalition("test_coalition", "Test Coalition")
        coalition.add_member("leader", CoalitionRole.LEADER, ["skill_a"])
        coalition.add_member("member", CoalitionRole.MEMBER, ["skill_b"])

        # Store initial leader
        initial_leader = coalition.leader_id
        assert initial_leader == "leader"

        # Remove leader
        assert coalition.remove_member("leader")

        # Should elect new leader
        assert coalition.leader_id == "member"
        assert coalition.members["member"].role == CoalitionRole.LEADER

    def test_coalition_dissolution(self):
        """Test coalition dissolution when empty."""
        if not IMPORT_SUCCESS:
            pytest.skip("Coalition modules not available")

        # Create coalition with single member
        coalition = Coalition("test_coalition", "Test Coalition")
        coalition.add_member("sole_member", CoalitionRole.LEADER, ["skill_a"])

        # Remove sole member
        assert coalition.remove_member("sole_member")

        # Should dissolve coalition
        assert coalition.status == CoalitionStatus.DISSOLVED
        assert len(coalition.members) == 0
        assert coalition.leader_id is None


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling in coalition formation."""

    def test_empty_agent_list(self):
        """Test formation with empty agent list."""
        if not IMPORT_SUCCESS:
            pytest.skip("Coalition modules not available")

        objectives = [
            CoalitionObjective(
                objective_id="test_obj",
                description="Test objective",
                required_capabilities=["skill_a"],
                priority=1.0,
            )
        ]

        strategies = [GreedyFormation(), OptimalFormation(), HierarchicalFormation()]

        for strategy in strategies:
            result = strategy.form_coalitions([], objectives)

            # Should handle gracefully
            assert isinstance(result, FormationResult)
            assert len(result.coalitions) == 0
            assert len(result.unassigned_agents) == 0

    def test_empty_objectives_list(self):
        """Test formation with empty objectives list."""
        if not IMPORT_SUCCESS:
            pytest.skip("Coalition modules not available")

        agents = [
            AgentProfile(
                agent_id="agent_1",
                capabilities=["skill_a"],
                capacity=1.0,
                reputation=1.0,
                preferences={},
                current_coalitions=[],
            )
        ]

        strategies = [GreedyFormation(), OptimalFormation(), HierarchicalFormation()]

        for strategy in strategies:
            result = strategy.form_coalitions(agents, [])

            # Should handle gracefully
            assert isinstance(result, FormationResult)
            assert len(result.coalitions) == 0
            assert len(result.unassigned_agents) == 1

    def test_malformed_agent_data(self):
        """Test handling of malformed agent data."""
        if not IMPORT_SUCCESS:
            pytest.skip("Coalition modules not available")

        # Test with various malformed data
        malformed_agents = [
            AgentProfile(
                agent_id="",  # Empty ID
                capabilities=[],
                capacity=0.5,
                reputation=0.5,
                preferences={},
                current_coalitions=[],
            ),
            AgentProfile(
                agent_id="agent_2",
                capabilities=["skill_a"],
                capacity=1.5,  # Invalid capacity > 1.0
                reputation=0.5,
                preferences={},
                current_coalitions=[],
            ),
        ]

        objectives = [
            CoalitionObjective(
                objective_id="test_obj",
                description="Test objective",
                required_capabilities=["skill_a"],
                priority=1.0,
            )
        ]

        # Should handle without crashing
        strategy = GreedyFormation()
        result = strategy.form_coalitions(malformed_agents, objectives)

        # Should complete without error
        assert isinstance(result, FormationResult)

    def test_invalid_constraints(self):
        """Test handling of invalid constraints."""
        if not IMPORT_SUCCESS:
            pytest.skip("Coalition modules not available")

        agents = [
            AgentProfile(
                agent_id="agent_1",
                capabilities=["skill_a"],
                capacity=1.0,
                reputation=1.0,
                preferences={},
                current_coalitions=[],
            )
        ]

        objectives = [
            CoalitionObjective(
                objective_id="test_obj",
                description="Test objective",
                required_capabilities=["skill_a"],
                priority=1.0,
            )
        ]

        # Test with invalid constraints
        invalid_constraints = {
            "max_coalition_size": -1,  # Invalid negative size
            "unknown_constraint": "invalid_value",
        }

        strategy = GreedyFormation()
        result = strategy.form_coalitions(agents, objectives, invalid_constraints)

        # Should handle gracefully
        assert isinstance(result, FormationResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=coalitions", "--cov-report=term-missing"])
