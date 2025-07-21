"""
Core Coalition Class Tests.

This test suite provides comprehensive coverage for the core Coalition class,
including member management, objective handling, and status transitions.
Following TDD principles with ultrathink reasoning for edge case detection.
"""

from datetime import datetime

import pytest

# Import the modules under test
try:
    from coalitions.coalition import (
        Coalition,
        CoalitionMember,
        CoalitionObjective,
        CoalitionRole,
        CoalitionStatus,
    )

    IMPORT_SUCCESS = True
except ImportError:
    IMPORT_SUCCESS = False

    # Mock classes for testing when imports fail
    class Coalition:
        def __init__(self, coalition_id, name, objectives=None, max_size=None):
            self.coalition_id = coalition_id
            self.name = name
            self.objectives = objectives or []
            self.max_size = max_size
            self.members = {}
            self.status = "forming"
            self.leader_id = None
            self.created_at = datetime.now()
            self.last_modified = datetime.now()
            self.performance_score = 0.0
            self.coordination_efficiency = 0.0
            self.objective_completion_rate = 0.0
            self.communication_history = []
            self.decision_log = []

        def add_member(self, agent_id, role="member", capabilities=None):
            return True

        def remove_member(self, agent_id):
            return True

        def activate(self):
            self.status = "active"

        def get_capabilities(self):
            return set()

        def can_achieve_objective(self, objective):
            return True

    class CoalitionMember:
        def __init__(self, agent_id, role, capabilities=None):
            self.agent_id = agent_id
            self.role = role
            self.capabilities = capabilities or []
            self.contribution_score = 0.0
            self.trust_score = 1.0
            self.active = True
            self.join_time = datetime.now()
            self.last_activity = None

    class CoalitionObjective:
        def __init__(self, objective_id, description, required_capabilities, priority):
            self.objective_id = objective_id
            self.description = description
            self.required_capabilities = required_capabilities
            self.priority = priority
            self.progress = 0.0
            self.completed = False
            self.metadata = {}

    class CoalitionRole:
        LEADER = "leader"
        MEMBER = "member"
        COORDINATOR = "coordinator"
        OBSERVER = "observer"

    class CoalitionStatus:
        FORMING = "forming"
        ACTIVE = "active"
        DISBANDING = "disbanding"
        DISSOLVED = "dissolved"


class TestCoalitionInitialization:
    """Test Coalition class initialization and basic properties."""

    def test_coalition_basic_initialization(self):
        """Test basic coalition initialization."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        coalition = Coalition("test_001", "Test Coalition")

        assert coalition.coalition_id == "test_001"
        assert coalition.name == "Test Coalition"
        assert coalition.objectives == []
        assert coalition.max_size is None
        assert coalition.members == {}
        assert coalition.status == CoalitionStatus.FORMING
        assert coalition.leader_id is None
        assert isinstance(coalition.created_at, datetime)
        assert isinstance(coalition.last_modified, datetime)

    def test_coalition_initialization_with_objectives(self):
        """Test coalition initialization with objectives."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        objectives = [
            CoalitionObjective(
                objective_id="obj_1",
                description="Test objective 1",
                required_capabilities=["skill_a"],
                priority=1.0,
            ),
            CoalitionObjective(
                objective_id="obj_2",
                description="Test objective 2",
                required_capabilities=["skill_b"],
                priority=0.8,
            ),
        ]

        coalition = Coalition("test_002", "Test Coalition", objectives=objectives)

        assert len(coalition.objectives) == 2
        assert coalition.objectives[0].objective_id == "obj_1"
        assert coalition.objectives[1].objective_id == "obj_2"

    def test_coalition_initialization_with_max_size(self):
        """Test coalition initialization with maximum size constraint."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        coalition = Coalition("test_003", "Test Coalition", max_size=5)

        assert coalition.max_size == 5

    def test_coalition_performance_metrics_initialization(self):
        """Test that performance metrics are initialized correctly."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        coalition = Coalition("test_004", "Test Coalition")

        assert coalition.performance_score == 0.0
        assert coalition.coordination_efficiency == 0.0
        assert coalition.objective_completion_rate == 0.0

    def test_coalition_history_initialization(self):
        """Test that communication and decision history are initialized."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        coalition = Coalition("test_005", "Test Coalition")

        assert coalition.communication_history == []
        assert coalition.decision_log == []


class TestCoalitionMemberManagement:
    """Test coalition member management operations."""

    @pytest.fixture
    def basic_coalition(self):
        """Create a basic coalition for testing."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"
        return Coalition("test_coalition", "Test Coalition")

    def test_add_member_basic(self, basic_coalition):
        """Test adding a member to coalition."""
        success = basic_coalition.add_member("agent_001", CoalitionRole.MEMBER, ["skill_a"])

        assert success is True
        assert "agent_001" in basic_coalition.members
        assert basic_coalition.members["agent_001"].agent_id == "agent_001"
        # First member automatically becomes leader
        assert basic_coalition.members["agent_001"].role == CoalitionRole.LEADER
        assert "skill_a" in basic_coalition.members["agent_001"].capabilities

    def test_add_first_member_becomes_leader(self, basic_coalition):
        """Test that first member automatically becomes leader."""
        success = basic_coalition.add_member("agent_001", CoalitionRole.MEMBER, ["skill_a"])

        assert success is True
        assert basic_coalition.leader_id == "agent_001"
        assert basic_coalition.members["agent_001"].role == CoalitionRole.LEADER

    def test_add_member_explicit_leader(self, basic_coalition):
        """Test adding member with explicit leader role."""
        success = basic_coalition.add_member("agent_001", CoalitionRole.LEADER, ["skill_a"])

        assert success is True
        assert basic_coalition.leader_id == "agent_001"
        assert basic_coalition.members["agent_001"].role == CoalitionRole.LEADER

    def test_add_member_max_size_constraint(self):
        """Test that maximum size constraint is enforced."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        coalition = Coalition("test_coalition", "Test Coalition", max_size=2)

        # Add two members successfully
        assert coalition.add_member("agent_001", CoalitionRole.MEMBER, ["skill_a"]) is True
        assert coalition.add_member("agent_002", CoalitionRole.MEMBER, ["skill_b"]) is True

        # Third member should fail
        assert coalition.add_member("agent_003", CoalitionRole.MEMBER, ["skill_c"]) is False
        assert len(coalition.members) == 2

    def test_add_duplicate_member(self, basic_coalition):
        """Test that duplicate member addition fails."""
        # Add member first time
        assert basic_coalition.add_member("agent_001", CoalitionRole.MEMBER, ["skill_a"]) is True

        # Try to add same member again
        assert basic_coalition.add_member("agent_001", CoalitionRole.MEMBER, ["skill_b"]) is False
        assert len(basic_coalition.members) == 1

    def test_remove_member_basic(self, basic_coalition):
        """Test removing a member from coalition."""
        # Add member first
        basic_coalition.add_member("agent_001", CoalitionRole.MEMBER, ["skill_a"])
        basic_coalition.add_member("agent_002", CoalitionRole.MEMBER, ["skill_b"])

        # Remove member
        success = basic_coalition.remove_member("agent_002")

        assert success is True
        assert "agent_002" not in basic_coalition.members
        assert len(basic_coalition.members) == 1

    def test_remove_nonexistent_member(self, basic_coalition):
        """Test removing a member that doesn't exist."""
        success = basic_coalition.remove_member("nonexistent_agent")

        assert success is False

    def test_remove_leader_triggers_election(self, basic_coalition):
        """Test that removing leader triggers new leader election."""
        # Add leader and member
        basic_coalition.add_member("leader", CoalitionRole.LEADER, ["skill_a"])
        basic_coalition.add_member("member", CoalitionRole.MEMBER, ["skill_b"])

        # Set contribution scores for election
        basic_coalition.members["member"].contribution_score = 0.9
        basic_coalition.members["member"].trust_score = 0.8

        # Remove leader
        success = basic_coalition.remove_member("leader")

        assert success is True
        assert basic_coalition.leader_id == "member"
        assert basic_coalition.members["member"].role == CoalitionRole.LEADER

    def test_remove_last_member_dissolves_coalition(self, basic_coalition):
        """Test that removing last member dissolves coalition."""
        # Add single member
        basic_coalition.add_member("sole_member", CoalitionRole.MEMBER, ["skill_a"])

        # Remove sole member
        success = basic_coalition.remove_member("sole_member")

        assert success is True
        assert basic_coalition.status == CoalitionStatus.DISSOLVED
        assert len(basic_coalition.members) == 0
        assert basic_coalition.leader_id is None


class TestCoalitionObjectiveManagement:
    """Test coalition objective management operations."""

    @pytest.fixture
    def coalition_with_members(self):
        """Create a coalition with members for testing."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        coalition = Coalition("test_coalition", "Test Coalition")
        coalition.add_member("agent_001", CoalitionRole.LEADER, ["skill_a", "skill_b"])
        coalition.add_member("agent_002", CoalitionRole.MEMBER, ["skill_c"])
        return coalition

    def test_add_objective_with_sufficient_capabilities(self, coalition_with_members):
        """Test adding objective when coalition has required capabilities."""
        objective = CoalitionObjective(
            objective_id="obj_001",
            description="Test objective",
            required_capabilities=["skill_a", "skill_c"],
            priority=1.0,
        )

        success = coalition_with_members.add_objective(objective)

        assert success is True
        assert len(coalition_with_members.objectives) == 1
        assert coalition_with_members.objectives[0].objective_id == "obj_001"

    def test_add_objective_with_insufficient_capabilities(self, coalition_with_members):
        """Test adding objective when coalition lacks required capabilities."""
        objective = CoalitionObjective(
            objective_id="obj_002",
            description="Impossible objective",
            required_capabilities=["skill_z"],  # Not available
            priority=1.0,
        )

        success = coalition_with_members.add_objective(objective)

        assert success is False
        assert len(coalition_with_members.objectives) == 0

    def test_update_objective_progress(self, coalition_with_members):
        """Test updating objective progress."""
        objective = CoalitionObjective(
            objective_id="obj_003",
            description="Progressive objective",
            required_capabilities=["skill_a"],
            priority=1.0,
        )

        coalition_with_members.add_objective(objective)

        # Update progress
        success = coalition_with_members.update_objective_progress("obj_003", 0.7)

        assert success is True
        assert coalition_with_members.objectives[0].progress == 0.7
        assert coalition_with_members.objectives[0].completed is False

    def test_complete_objective(self, coalition_with_members):
        """Test completing an objective."""
        objective = CoalitionObjective(
            objective_id="obj_004",
            description="Completable objective",
            required_capabilities=["skill_a"],
            priority=1.0,
        )

        coalition_with_members.add_objective(objective)

        # Complete objective
        success = coalition_with_members.update_objective_progress("obj_004", 1.0)

        assert success is True
        assert coalition_with_members.objectives[0].progress == 1.0
        assert coalition_with_members.objectives[0].completed is True

    def test_update_nonexistent_objective(self, coalition_with_members):
        """Test updating progress on nonexistent objective."""
        success = coalition_with_members.update_objective_progress("nonexistent_obj", 0.5)

        assert success is False

    def test_can_achieve_objective(self, coalition_with_members):
        """Test checking if coalition can achieve objective."""
        # Test with achievable objective
        achievable_objective = CoalitionObjective(
            objective_id="achievable",
            description="Achievable objective",
            required_capabilities=["skill_a"],
            priority=1.0,
        )

        assert coalition_with_members.can_achieve_objective(achievable_objective) is True

        # Test with unachievable objective
        unachievable_objective = CoalitionObjective(
            objective_id="unachievable",
            description="Unachievable objective",
            required_capabilities=["skill_z"],
            priority=1.0,
        )

        assert coalition_with_members.can_achieve_objective(unachievable_objective) is False


class TestCoalitionStatusTransitions:
    """Test coalition status transitions and lifecycle."""

    @pytest.fixture
    def coalition_with_members(self):
        """Create a coalition with members for testing."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        coalition = Coalition("test_coalition", "Test Coalition")
        coalition.add_member("agent_001", CoalitionRole.LEADER, ["skill_a"])
        return coalition

    def test_initial_status(self):
        """Test that coalition starts in FORMING status."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        coalition = Coalition("test_coalition", "Test Coalition")

        assert coalition.status == CoalitionStatus.FORMING

    def test_activate_coalition(self, coalition_with_members):
        """Test activating a coalition."""
        coalition_with_members.activate()

        assert coalition_with_members.status == CoalitionStatus.ACTIVE

    def test_activate_empty_coalition(self):
        """Test that empty coalition cannot be activated."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        coalition = Coalition("empty_coalition", "Empty Coalition")
        coalition.activate()

        # Should remain in FORMING status
        assert coalition.status == CoalitionStatus.FORMING

    def test_disband_coalition(self, coalition_with_members):
        """Test disbanding a coalition."""
        coalition_with_members.activate()
        coalition_with_members.disband()

        assert coalition_with_members.status == CoalitionStatus.DISBANDING

    def test_automatic_dissolution_on_empty(self, coalition_with_members):
        """Test automatic dissolution when coalition becomes empty."""
        # Remove the only member
        coalition_with_members.remove_member("agent_001")

        assert coalition_with_members.status == CoalitionStatus.DISSOLVED


class TestCoalitionPerformanceMetrics:
    """Test coalition performance metrics calculation."""

    @pytest.fixture
    def coalition_with_objectives(self):
        """Create a coalition with objectives for testing."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        coalition = Coalition("test_coalition", "Test Coalition")
        coalition.add_member("agent_001", CoalitionRole.LEADER, ["skill_a"])
        coalition.add_member("agent_002", CoalitionRole.MEMBER, ["skill_b"])

        # Add objectives
        obj1 = CoalitionObjective("obj_1", "Objective 1", ["skill_a"], 1.0)
        obj2 = CoalitionObjective("obj_2", "Objective 2", ["skill_b"], 0.8)

        coalition.objectives = [obj1, obj2]
        return coalition

    def test_objective_completion_rate_calculation(self, coalition_with_objectives):
        """Test objective completion rate calculation."""
        # Complete one objective
        coalition_with_objectives.objectives[0].completed = True

        # Update metrics
        coalition_with_objectives._update_performance_metrics()

        assert coalition_with_objectives.objective_completion_rate == 0.5

    def test_coordination_efficiency_calculation(self, coalition_with_objectives):
        """Test coordination efficiency calculation."""
        # All members active
        coalition_with_objectives._update_performance_metrics()

        assert coalition_with_objectives.coordination_efficiency == 1.0

        # One member inactive
        coalition_with_objectives.members["agent_002"].active = False
        coalition_with_objectives._update_performance_metrics()

        assert coalition_with_objectives.coordination_efficiency == 0.5

    def test_performance_score_calculation(self, coalition_with_objectives):
        """Test overall performance score calculation."""
        # Set up metrics
        coalition_with_objectives.objectives[0].completed = True
        coalition_with_objectives._update_performance_metrics()

        # Performance score should be weighted combination
        expected_score = 0.7 * 0.5 + 0.3 * 1.0  # 70% completion + 30% coordination
        assert abs(coalition_with_objectives.performance_score - expected_score) < 0.001


class TestCoalitionCapabilityManagement:
    """Test coalition capability management and queries."""

    @pytest.fixture
    def coalition_with_diverse_members(self):
        """Create a coalition with diverse member capabilities."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        coalition = Coalition("diverse_coalition", "Diverse Coalition")
        coalition.add_member("agent_001", CoalitionRole.LEADER, ["skill_a", "skill_b"])
        coalition.add_member("agent_002", CoalitionRole.MEMBER, ["skill_c", "skill_d"])
        coalition.add_member("agent_003", CoalitionRole.MEMBER, ["skill_b", "skill_e"])
        return coalition

    def test_get_capabilities(self, coalition_with_diverse_members):
        """Test getting all capabilities in coalition."""
        capabilities = coalition_with_diverse_members.get_capabilities()

        expected_capabilities = {
            "skill_a",
            "skill_b",
            "skill_c",
            "skill_d",
            "skill_e",
        }
        assert capabilities == expected_capabilities

    def test_get_capabilities_empty_coalition(self):
        """Test getting capabilities from empty coalition."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        coalition = Coalition("empty_coalition", "Empty Coalition")
        capabilities = coalition.get_capabilities()

        assert capabilities == set()

    def test_get_member_by_role(self, coalition_with_diverse_members):
        """Test getting members by role."""
        leaders = coalition_with_diverse_members.get_member_by_role(CoalitionRole.LEADER)
        members = coalition_with_diverse_members.get_member_by_role(CoalitionRole.MEMBER)

        assert len(leaders) == 1
        assert leaders[0].agent_id == "agent_001"
        assert len(members) == 2

        member_ids = {member.agent_id for member in members}
        assert member_ids == {"agent_002", "agent_003"}

    def test_get_member_by_nonexistent_role(self, coalition_with_diverse_members):
        """Test getting members by nonexistent role."""
        observers = coalition_with_diverse_members.get_member_by_role(CoalitionRole.OBSERVER)

        assert len(observers) == 0


class TestCoalitionCommunication:
    """Test coalition communication and decision logging."""

    @pytest.fixture
    def active_coalition(self):
        """Create an active coalition for testing."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        coalition = Coalition("active_coalition", "Active Coalition")
        coalition.add_member("agent_001", CoalitionRole.LEADER, ["skill_a"])
        coalition.add_member("agent_002", CoalitionRole.MEMBER, ["skill_b"])
        coalition.activate()
        return coalition

    def test_add_communication_broadcast(self, active_coalition):
        """Test adding broadcast communication."""
        active_coalition.add_communication("agent_001", "Hello everyone!")

        assert len(active_coalition.communication_history) == 1

        comm = active_coalition.communication_history[0]
        assert comm["sender_id"] == "agent_001"
        assert comm["message"] == "Hello everyone!"
        assert comm["broadcast"] is True
        assert set(comm["recipients"]) == {"agent_001", "agent_002"}

    def test_add_communication_targeted(self, active_coalition):
        """Test adding targeted communication."""
        active_coalition.add_communication("agent_001", "Private message", ["agent_002"])

        assert len(active_coalition.communication_history) == 1

        comm = active_coalition.communication_history[0]
        assert comm["sender_id"] == "agent_001"
        assert comm["message"] == "Private message"
        assert comm["broadcast"] is False
        assert comm["recipients"] == ["agent_002"]

    def test_communication_updates_activity(self, active_coalition):
        """Test that communication updates sender's last activity."""
        initial_activity = active_coalition.members["agent_001"].last_activity

        active_coalition.add_communication("agent_001", "Test message")

        updated_activity = active_coalition.members["agent_001"].last_activity
        assert updated_activity != initial_activity
        assert isinstance(updated_activity, datetime)

    def test_decision_logging(self, active_coalition):
        """Test decision logging functionality."""
        # Decisions are logged automatically during operations
        initial_decisions = len(active_coalition.decision_log)

        # Add a member (should log decision)
        active_coalition.add_member("agent_003", CoalitionRole.MEMBER, ["skill_c"])

        # Check decision was logged
        assert len(active_coalition.decision_log) > initial_decisions

        # Find the decision about adding member
        add_decision = None
        for decision in active_coalition.decision_log:
            if "Added member agent_003" in decision["decision"]:
                add_decision = decision
                break

        assert add_decision is not None
        assert add_decision["leader_id"] == "agent_001"
        assert add_decision["member_count"] == 3


class TestCoalitionStatusReporting:
    """Test coalition status reporting and information retrieval."""

    @pytest.fixture
    def comprehensive_coalition(self):
        """Create a comprehensive coalition for testing."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        coalition = Coalition("comprehensive_coalition", "Comprehensive Coalition")
        coalition.add_member("agent_001", CoalitionRole.LEADER, ["skill_a", "skill_b"])
        coalition.add_member("agent_002", CoalitionRole.MEMBER, ["skill_c"])

        # Add objectives
        obj1 = CoalitionObjective("obj_1", "Objective 1", ["skill_a"], 1.0)
        obj1.completed = True
        obj2 = CoalitionObjective("obj_2", "Objective 2", ["skill_b"], 0.8)

        coalition.objectives = [obj1, obj2]
        coalition.activate()
        return coalition

    def test_get_status_complete(self, comprehensive_coalition):
        """Test getting complete coalition status."""
        status = comprehensive_coalition.get_status()

        # Check all required fields
        assert status["coalition_id"] == "comprehensive_coalition"
        assert status["name"] == "Comprehensive Coalition"
        assert status["status"] == CoalitionStatus.ACTIVE.value
        assert status["member_count"] == 2
        assert status["leader_id"] == "agent_001"
        assert status["objectives_count"] == 2
        assert status["completed_objectives"] == 1
        assert isinstance(status["performance_score"], float)
        assert isinstance(status["coordination_efficiency"], float)
        assert isinstance(status["objective_completion_rate"], float)
        assert isinstance(status["created_at"], str)
        assert isinstance(status["last_modified"], str)
        assert isinstance(status["capabilities"], list)

    def test_get_status_capabilities_list(self, comprehensive_coalition):
        """Test that status includes capabilities as list."""
        status = comprehensive_coalition.get_status()

        capabilities = set(status["capabilities"])
        expected_capabilities = {"skill_a", "skill_b", "skill_c"}
        assert capabilities == expected_capabilities


class TestCoalitionEdgeCases:
    """Test edge cases and error conditions."""

    def test_coalition_with_none_capabilities(self):
        """Test coalition operations with None capabilities."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        coalition = Coalition("test_coalition", "Test Coalition")

        # Add member with None capabilities
        success = coalition.add_member("agent_001", CoalitionRole.MEMBER, None)

        assert success is True
        assert coalition.members["agent_001"].capabilities == []

    def test_coalition_with_empty_capabilities(self):
        """Test coalition operations with empty capabilities."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        coalition = Coalition("test_coalition", "Test Coalition")

        # Add member with empty capabilities
        success = coalition.add_member("agent_001", CoalitionRole.MEMBER, [])

        assert success is True
        assert coalition.members["agent_001"].capabilities == []

    def test_coalition_progress_bounds(self):
        """Test that objective progress is properly bounded."""
        if not IMPORT_SUCCESS:
            assert False, "Test bypass removed - must fix underlying issue"

        coalition = Coalition("test_coalition", "Test Coalition")
        coalition.add_member("agent_001", CoalitionRole.LEADER, ["skill_a"])

        obj = CoalitionObjective("obj_1", "Test Objective", ["skill_a"], 1.0)
        coalition.objectives = [obj]

        # Test progress > 1.0 is capped
        coalition.update_objective_progress("obj_1", 1.5)
        assert coalition.objectives[0].progress == 1.0

        # Test progress < 0.0 is capped
        coalition.update_objective_progress("obj_1", -0.5)
        assert coalition.objectives[0].progress == 0.0


if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
            "--cov=coalitions.coalition",
            "--cov-report=term-missing",
        ]
    )
