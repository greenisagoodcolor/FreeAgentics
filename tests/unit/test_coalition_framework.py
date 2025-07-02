"""
Module for FreeAgentics Active Inference implementation.
"""

from typing import List

import pytest

from coalitions.coalition.coalition_models import (
    Coalition,
    CoalitionGoal,
    CoalitionGoalStatus,
    CoalitionMember,
    CoalitionRole,
    CoalitionStatus,
)
from coalitions.formation.coalition_formation_algorithms import (
    AgentProfile,
    CoalitionFormationEngine,
    FormationStrategy,
)


class TestCoalitionModels:
    """Test coalition data models"""

    def test_coalition_goal_creation(self) -> None:
        """Test creating and managing coalition goals"""
        goal = CoalitionGoal(
            goal_id="test_goal",
            title="Test Goal",
            description="A test goal for the coalition",
            priority=0.8,
        )
        assert goal.goal_id == "test_goal"
        assert goal.title == "Test Goal"
        assert goal.status == CoalitionGoalStatus.PROPOSED
        assert goal.progress_percentage == 0.0
        assert not goal.is_completed
        assert not goal.is_overdue

    def test_coalition_goal_voting(self) -> None:
        """Test goal voting and consensus"""
        goal = CoalitionGoal("test_goal", "Test Goal", "Description")
        # Add votes
        goal.add_vote("agent1", True)
        goal.add_vote("agent2", True)
        goal.add_vote("agent3", False)
        assert "agent1" in goal.votes_for
        assert "agent2" in goal.votes_for
        assert "agent3" in goal.votes_against
        # Calculate consensus
        consensus = goal.calculate_consensus(3)
        assert consensus == pytest.approx(2 / 3, rel=1e-2)

    def test_coalition_goal_progress(self) -> None:
        """Test goal progress tracking"""
        goal = CoalitionGoal("test_goal", "Test Goal", "Description")
        goal.status = CoalitionGoalStatus.ACCEPTED
        # Update progress
        goal.update_progress(0.5)
        assert goal.current_progress == 0.5
        assert goal.progress_percentage == 50.0
        assert goal.status == CoalitionGoalStatus.IN_PROGRESS
        # Complete goal
        goal.update_progress(0.9)
        assert goal.status == CoalitionGoalStatus.COMPLETED
        assert goal.is_completed

    def test_coalition_member_creation(self) -> None:
        """Test creating and managing coalition members"""
        member = CoalitionMember(
            agent_id="agent1",
            role=CoalitionRole.LEADER,
            capability_contributions={"leadership", "strategy"},
            resource_commitments={"compute": 100.0, "storage": 50.0},
        )
        assert member.agent_id == "agent1"
        assert member.role == CoalitionRole.LEADER
        assert member.is_leader
        assert member.is_active
        assert member.success_rate == 1.0
        assert "leadership" in member.capability_contributions

    def test_coalition_member_performance(self) -> None:
        """Test member performance tracking"""
        member = CoalitionMember("agent1")
        # Complete some goals
        member.complete_goal(True)
        member.complete_goal(True)
        member.complete_goal(False)
        assert member.goals_completed == 2
        assert member.goals_failed == 1
        assert member.success_rate == pytest.approx(2 / 3, rel=1e-2)
        assert member.reliability_score < 1.0  # Should decrease due to failure

    def test_coalition_creation(self) -> None:
        """Test creating and managing coalitions"""
        coalition = Coalition(
            coalition_id="test_coalition",
            name="Test Coalition",
            description="A test coalition",
        )
        assert coalition.coalition_id == "test_coalition"
        assert coalition.name == "Test Coalition"
        assert coalition.status == CoalitionStatus.FORMING
        assert coalition.member_count == 0
        assert not coalition.is_viable  # No members yet

    def test_coalition_membership(self) -> None:
        """Test adding and removing coalition members"""
        coalition = Coalition("test_coalition", "Test Coalition")
        # Add members
        success = coalition.add_member(
            "agent1",
            CoalitionRole.LEADER,
            capabilities={"leadership"},
            resources={"compute": 100},
        )
        assert success
        assert coalition.member_count == 1
        assert coalition.status == CoalitionStatus.FORMING  # Below minimum
        # Add second member
        coalition.add_member("agent2", CoalitionRole.CONTRIBUTOR)
        assert coalition.member_count == 2
        assert coalition.status == CoalitionStatus.ACTIVE  # At minimum
        assert coalition.is_viable
        # Test capabilities and resources
        assert "leadership" in coalition.combined_capabilities
        assert coalition.total_resources["compute"] == 100
        # Remove member
        coalition.remove_member("agent2")
        assert coalition.member_count == 1
        assert coalition.status == CoalitionStatus.DISSOLVING  # Below minimum

    def test_coalition_goals_management(self) -> None:
        """Test managing coalition goals"""
        coalition = Coalition("test_coalition", "Test Coalition")
        coalition.add_member("agent1", CoalitionRole.LEADER)
        coalition.add_member("agent2", CoalitionRole.CONTRIBUTOR)
        # Add goal
        goal = CoalitionGoal("goal1", "Test Goal", "Description")
        coalition.add_goal(goal)
        assert "goal1" in coalition.goals
        assert coalition.primary_goal_id == "goal1"
        # Vote on goal
        coalition.vote_on_goal("goal1", "agent1", True)
        coalition.vote_on_goal("goal1", "agent2", True)
        # Check if goal is accepted (should be with 100% support)
        assert coalition.goals["goal1"].status == CoalitionGoalStatus.ACCEPTED


class TestCoalitionFormation:
    """Test coalition formation algorithms"""

    def create_test_agents(self) -> List[AgentProfile]:
        """Create test agents for formation testing"""
        agents = [
            AgentProfile(
                agent_id="agent1",
                capabilities={"leadership", "strategy", "communication"},
                resources={"compute": 100, "storage": 50},
                beliefs={"cooperation": 0.8, "efficiency": 0.7},
                reliability_score=0.9,
            ),
            AgentProfile(
                agent_id="agent2",
                capabilities={"technical", "analysis", "optimization"},
                resources={"compute": 150, "memory": 100},
                beliefs={"cooperation": 0.6, "innovation": 0.8},
                reliability_score=0.8,
            ),
            AgentProfile(
                agent_id="agent3",
                capabilities={"communication", "coordination", "planning"},
                resources={"storage": 200, "bandwidth": 75},
                beliefs={"efficiency": 0.9, "quality": 0.7},
                reliability_score=0.85,
            ),
            AgentProfile(
                agent_id="agent4",
                capabilities={"technical", "development", "testing"},
                resources={"compute": 80, "memory": 60},
                beliefs={"innovation": 0.9, "quality": 0.8},
                reliability_score=0.7,
            ),
        ]
        return agents

    def test_active_inference_formation(self) -> None:
        """Test active inference coalition formation"""
        engine = CoalitionFormationEngine()
        agents = self.create_test_agents()
        result = engine.form_coalition(
            agents=agents,
            strategy=FormationStrategy.ACTIVE_INFERENCE,
            max_size=3)
        assert result.success
        assert result.coalition is not None
        assert len(result.participants) >= 2
        assert len(result.participants) <= 3
        assert result.strategy_used == FormationStrategy.ACTIVE_INFERENCE
        assert result.formation_time > 0
        # Check coalition properties
        coalition = result.coalition
        assert coalition.status in [
            CoalitionStatus.FORMING,
            CoalitionStatus.ACTIVE]
        assert coalition.leader_id in result.participants

    def test_capability_based_formation(self) -> None:
        """Test capability-based coalition formation"""
        engine = CoalitionFormationEngine()
        agents = self.create_test_agents()
        required_capabilities = {"technical", "communication", "planning"}
        result = engine.form_coalition(
            agents=agents,
            strategy=FormationStrategy.CAPABILITY_BASED,
            max_size=4)
        assert result.success
        assert result.coalition is not None
        # Check if required capabilities are covered
        coalition = result.coalition
        covered_capabilities = coalition.combined_capabilities
        # Should cover most or all required capabilities
        coverage = len(covered_capabilities & required_capabilities) / \
            len(required_capabilities)
        assert coverage > 0.5  # At least 50% coverage

    def test_resource_optimization_formation(self) -> None:
        """Test resource optimization coalition formation"""
        engine = CoalitionFormationEngine()
        agents = self.create_test_agents()
        result = engine.form_coalition(
            agents=agents,
            strategy=FormationStrategy.RESOURCE_OPTIMIZATION,
            max_size=3)
        assert result.success
        assert result.coalition is not None
        # Check resource diversity
        coalition = result.coalition
        total_resources = coalition.total_resources
        assert len(total_resources) > 0  # Should have diverse resources
        # Should have significant resources
        assert sum(total_resources.values()) > 0

    def test_multiple_strategy_formation(self) -> None:
        """Test trying multiple formation strategies"""
        engine = CoalitionFormationEngine()
        agents = self.create_test_agents()
        strategies = [
            FormationStrategy.ACTIVE_INFERENCE,
            FormationStrategy.CAPABILITY_BASED,
            FormationStrategy.RESOURCE_OPTIMIZATION,
        ]
        result = engine.try_multiple_strategies(
            agents=agents, strategies=strategies, max_size=3)
        assert result.success
        assert result.coalition is not None
        assert result.strategy_used in strategies
        # Should select the strategy with the best score
        assert result.score > 0

    def test_formation_with_constraints(self) -> None:
        """Test formation with agent constraints"""
        engine = CoalitionFormationEngine()
        agents = self.create_test_agents()
        # Set constraints on some agents
        agents[1].max_coalitions = 0  # Agent2 unavailable
        agents[2].availability = 0.5  # Agent3 partially available
        result = engine.form_coalition(
            agents=agents, strategy=FormationStrategy.ACTIVE_INFERENCE)
        # Should still succeed with remaining agents
        if result.success:
            assert "agent2" not in result.participants  # Should exclude unavailable agent
        else:
            # If formation fails, it should be due to insufficient viable
            # agents
            assert len(
                [a for a in agents if a.current_coalitions < a.max_coalitions]) < 2


class TestCoalitionIntegration:
    """Test integration between different coalition components"""

    def test_end_to_end_coalition_lifecycle(self) -> None:
        """Test complete coalition lifecycle from formation to dissolution"""
        engine = CoalitionFormationEngine()
        # Create agents
        agents = [
            AgentProfile(
                agent_id=f"agent_{i}",
                capabilities={f"skill_{i}", "general"},
                resources={"compute": 100 + i * 50},
                beliefs={"cooperation": 0.7 + i * 0.1},
                reliability_score=0.8 + i * 0.05,
            )
            for i in range(4)
        ]
        # Form coalition
        result = engine.form_coalition(agents, max_size=3)
        assert result.success
        coalition = result.coalition
        # Add a goal
        goal = CoalitionGoal(
            goal_id="main_goal",
            title="Main Objective",
            description="Primary coalition objective",
            priority=0.9,
        )
        coalition.add_goal(goal)
        # Members vote on goal
        for member_id in coalition.members.keys():
            coalition.vote_on_goal("main_goal", member_id, True)
        # Check goal acceptance
        assert coalition.goals["main_goal"].status == CoalitionGoalStatus.ACCEPTED
        # Progress on goal
        coalition.update_goal_progress("main_goal", 0.5)
        assert coalition.goals["main_goal"].status == CoalitionGoalStatus.IN_PROGRESS
        # Complete goal
        coalition.update_goal_progress("main_goal", 0.9)
        assert coalition.goals["main_goal"].is_completed
        # Test member performance tracking
        for member in coalition.members.values():
            member.complete_goal(True)
            assert member.goals_completed == 1
        # Test resource allocation
        allocation = {"main_goal": {"compute": 200, "storage": 50}}
        coalition.allocate_resources(allocation)
        # Should succeed if coalition has enough resources
        # Get status summary
        summary = coalition.get_status_summary()
        assert summary["coalition_id"] == coalition.coalition_id
        assert summary["member_count"] == len(result.participants)
        assert summary["goals"]["completed"] == 1

    def test_formation_statistics(self) -> None:
        """Test formation statistics tracking"""
        engine = CoalitionFormationEngine()
        agents = [
            AgentProfile(
                agent_id=f"agent_{i}",
                capabilities={f"skill_{i}"},
                resources={"compute": 100},
            )
            for i in range(3)
        ]
        # Perform multiple formations
        for _ in range(3):
            engine.form_coalition(agents)
        stats = engine.get_formation_statistics()
        assert "total_attempts" in stats
        assert "successful_formations" in stats
        assert "success_rate" in stats
        assert stats["total_attempts"] == 3

    def test_coalition_value_distribution(self) -> None:
        """Test coalition value distribution among members"""
        coalition = Coalition("test", "Test Coalition")
        # Add members with different contributions
        coalition.add_member(
            "agent1",
            CoalitionRole.LEADER,
            capabilities={"leadership"},
            resources={"compute": 100},
        )
        coalition.add_member(
            "agent2",
            CoalitionRole.CONTRIBUTOR,
            capabilities={"technical"},
            resources={"storage": 50},
        )
        coalition.add_member(
            "agent3",
            CoalitionRole.SPECIALIST,
            capabilities={"analysis"},
            resources={"memory": 75},
        )
        # Calculate contributions
        contributions = coalition.calculate_member_contributions()
        assert len(contributions) == 3
        # Leader should have higher contribution due to leadership bonus
        assert contributions["agent1"] > contributions["agent2"]
        # Test different distribution models
        total_value = 1000.0
        # Equal distribution
        coalition.value_distribution_model = "equal"
        equal_dist = coalition.distribute_value(total_value)
        assert all(value == pytest.approx(total_value / 3, rel=1e-2)
                   for value in equal_dist.values())
        # Contribution-based distribution
        coalition.value_distribution_model = "contribution-based"
        contrib_dist = coalition.distribute_value(total_value)
        assert sum(
            contrib_dist.values()) == pytest.approx(
            total_value, rel=1e-2)
        # Leader gets more
        assert contrib_dist["agent1"] > contrib_dist["agent2"]


if __name__ == "__main__":
    pytest.main([__file__])
