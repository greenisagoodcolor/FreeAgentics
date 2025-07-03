"""
Comprehensive tests for Agent Data Model
"""

from datetime import datetime, timedelta
from typing import Set

import numpy as np
import pytest

from agents.base.data_model import (  # Enums; Core classes
    Action,
    ActionType,
    Agent,
    AgentBuilder,
    AgentCapability,
    AgentClass,
    AgentGoal,
    AgentPersonality,
    AgentResources,
    AgentStatus,
    Experience,
    Orientation,
    PersonalityTraits,
    Position,
    ResourceAgent,
    SocialAgent,
    SocialRelationship,
    SpecializedAgent,
)


class TestEnums:
    """Test enumeration types"""

    def test_agent_status_enum(self):
        """Test AgentStatus enum values"""
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.MOVING.value == "moving"
        assert AgentStatus.INTERACTING.value == "interacting"
        assert AgentStatus.PLANNING.value == "planning"
        assert AgentStatus.LEARNING.value == "learning"
        assert AgentStatus.OFFLINE.value == "offline"
        assert AgentStatus.ERROR.value == "error"
        assert len(AgentStatus) == 7

    def test_agent_class_enum(self):
        """Test AgentClass enum values"""
        assert AgentClass.EXPLORER.value == "explorer"
        assert AgentClass.MERCHANT.value == "merchant"
        assert AgentClass.SCHOLAR.value == "scholar"
        assert AgentClass.GUARDIAN.value == "guardian"
        assert len(AgentClass) == 4

    def test_personality_traits_enum(self):
        """Test PersonalityTraits enum values"""
        assert PersonalityTraits.OPENNESS.value == "openness"
        assert PersonalityTraits.CONSCIENTIOUSNESS.value == "conscientiousness"
        assert PersonalityTraits.EXTRAVERSION.value == "extraversion"
        assert PersonalityTraits.AGREEABLENESS.value == "agreeableness"
        assert PersonalityTraits.NEUROTICISM.value == "neuroticism"
        assert len(PersonalityTraits) == 5

    def test_agent_capability_enum(self):
        """Test AgentCapability enum values"""
        assert AgentCapability.MOVEMENT.value == "movement"
        assert AgentCapability.PERCEPTION.value == "perception"
        assert AgentCapability.COMMUNICATION.value == "communication"
        assert AgentCapability.MEMORY.value == "memory"
        assert AgentCapability.LEARNING.value == "learning"
        assert AgentCapability.PLANNING.value == "planning"
        assert AgentCapability.RESOURCE_MANAGEMENT.value == "resource_management"
        assert AgentCapability.SOCIAL_INTERACTION.value == "social_interaction"
        assert len(AgentCapability) == 8

    def test_action_type_enum(self):
        """Test ActionType enum values"""
        assert ActionType.MOVE.value == "move"
        assert ActionType.COMMUNICATE.value == "communicate"
        assert ActionType.GATHER.value == "gather"
        assert ActionType.EXPLORE.value == "explore"
        assert ActionType.TRADE.value == "trade"
        assert ActionType.LEARN.value == "learn"
        assert ActionType.WAIT.value == "wait"
        assert ActionType.ATTACK.value == "attack"
        assert ActionType.DEFEND.value == "defend"
        assert ActionType.BUILD.value == "build"
        assert len(ActionType) == 10


class TestPosition:
    """Test Position class"""

    def test_position_creation(self):
        """Test position creation"""
        pos = Position(10.5, 20.3, 5.0)
        assert pos.x == 10.5
        assert pos.y == 20.3
        assert pos.z == 5.0

        # Test default z value
        pos2 = Position(1.0, 2.0)
        assert pos2.z == 0.0

    def test_position_to_array(self):
        """Test conversion to numpy array"""
        pos = Position(1.0, 2.0, 3.0)
        arr = pos.to_array()
        np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0]))
        assert arr.dtype == np.float64

    def test_position_distance(self):
        """Test distance calculation"""
        pos1 = Position(0.0, 0.0, 0.0)
        pos2 = Position(3.0, 4.0, 0.0)
        assert pos1.distance_to(pos2) == 5.0

        # Test 3D distance
        pos3 = Position(1.0, 1.0, 1.0)
        pos4 = Position(2.0, 2.0, 2.0)
        expected = np.sqrt(3)
        assert abs(pos3.distance_to(pos4) - expected) < 1e-10

    def test_position_hash_and_equality(self):
        """Test Position hashing and equality"""
        pos1 = Position(1.0, 2.0, 3.0)
        pos2 = Position(1.0, 2.0, 3.0)
        pos3 = Position(1.0, 2.0, 3.1)

        # Test equality
        assert pos1 == pos2
        assert pos1 != pos3
        assert pos1 != "not a position"

        # Test hashing
        assert hash(pos1) == hash(pos2)
        assert hash(pos1) != hash(pos3)

        # Test in set/dict
        position_set = {pos1}
        assert pos2 in position_set
        assert pos3 not in position_set


class TestOrientation:
    """Test Orientation class"""

    def test_orientation_creation(self):
        """Test orientation creation"""
        # Default orientation
        orient1 = Orientation()
        assert orient1.w == 1.0
        assert orient1.x == 0.0
        assert orient1.y == 0.0
        assert orient1.z == 0.0

        # Custom orientation
        orient2 = Orientation(w=0.707, x=0.0, y=0.707, z=0.0)
        assert orient2.w == 0.707
        assert orient2.y == 0.707

    def test_orientation_to_euler(self):
        """Test quaternion to Euler conversion"""
        # Identity quaternion should give zero angles
        orient = Orientation()
        roll, pitch, yaw = orient.to_euler()
        assert abs(roll) < 1e-10
        assert abs(pitch) < 1e-10
        assert abs(yaw) < 1e-10


class TestAction:
    """Test Action class"""

    def test_action_creation(self):
        """Test action creation"""
        action = Action(
            action_type=ActionType.MOVE,
            target_position=Position(10, 20),
            target_agent_id="agent_123",
            parameters={"speed": 5.0},
        )

        assert action.action_type == ActionType.MOVE
        assert action.target_position.x == 10
        assert action.target_position.y == 20
        assert action.target_agent_id == "agent_123"
        assert action.parameters["speed"] == 5.0
        assert isinstance(action.created_at, datetime)
        assert action.duration == 1.0
        assert action.energy_cost == 1.0
        assert action.priority == 0.5

    def test_action_to_dict(self):
        """Test action serialization"""
        action = Action(action_type=ActionType.COMMUNICATE, parameters={"message": "Hello"})

        action_dict = action.to_dict()
        assert action_dict["action_type"] == "communicate"
        assert action_dict["parameters"]["message"] == "Hello"
        assert "created_at" in action_dict
        assert action_dict["target_position"] is None
        assert action_dict["target_agent_id"] is None

    def test_action_can_execute(self):
        """Test action execution check"""
        agent = Agent()
        agent.resources.energy = 50.0

        # Action with low energy cost
        action1 = Action(action_type=ActionType.WAIT, energy_cost=10.0)
        assert action1.can_execute(agent)

        # Action with high energy cost
        action2 = Action(action_type=ActionType.BUILD, energy_cost=100.0)
        assert not action2.can_execute(agent)


class TestAgentPersonality:
    """Test AgentPersonality class"""

    def test_personality_creation(self):
        """Test personality creation with defaults"""
        personality = AgentPersonality()

        # Check default values (all 0.5)
        assert personality.openness == 0.5
        assert personality.conscientiousness == 0.5
        assert personality.extraversion == 0.5
        assert personality.agreeableness == 0.5
        assert personality.neuroticism == 0.5

    def test_personality_custom_values(self):
        """Test personality with custom values"""
        personality = AgentPersonality(
            openness=0.8,
            conscientiousness=0.7,
            extraversion=0.3,
            agreeableness=0.9,
            neuroticism=0.2,
        )

        assert personality.openness == 0.8
        assert personality.conscientiousness == 0.7
        assert personality.extraversion == 0.3
        assert personality.agreeableness == 0.9
        assert personality.neuroticism == 0.2

    def test_personality_to_vector(self):
        """Test personality to vector conversion"""
        personality = AgentPersonality(
            openness=0.8,
            conscientiousness=0.7,
            extraversion=0.6,
            agreeableness=0.5,
            neuroticism=0.4,
        )

        vector = personality.to_vector()
        expected = np.array([0.8, 0.7, 0.6, 0.5, 0.4])
        np.testing.assert_array_equal(vector, expected)

    def test_personality_validate(self):
        """Test personality validation"""
        # Valid personality
        valid_personality = AgentPersonality()
        assert valid_personality.validate()

        # Invalid personality (would need to manually set invalid values)
        invalid_personality = AgentPersonality()
        invalid_personality.openness = 1.5  # Out of range
        assert not invalid_personality.validate()


class TestAgentResources:
    """Test AgentResources class"""

    def test_resources_creation(self):
        """Test resources creation with defaults"""
        resources = AgentResources()

        assert resources.energy == 100.0
        assert resources.health == 100.0
        assert resources.memory_capacity == 100.0
        assert resources.memory_used == 0.0

    def test_resources_custom_values(self):
        """Test resources with custom values"""
        resources = AgentResources(
            energy=70.0, health=90.0, memory_capacity=150.0, memory_used=50.0
        )

        assert resources.energy == 70.0
        assert resources.health == 90.0
        assert resources.memory_capacity == 150.0
        assert resources.memory_used == 50.0

    def test_resources_energy_management(self):
        """Test energy management methods"""
        resources = AgentResources(energy=100.0)

        # Check sufficient energy
        assert resources.has_sufficient_energy(50.0)
        assert not resources.has_sufficient_energy(150.0)

        # Consume energy
        resources.consume_energy(30.0)
        assert resources.energy == 70.0

        # Consume more than available
        resources.consume_energy(100.0)
        assert resources.energy == 0.0

        # Restore energy
        resources.restore_energy(50.0)
        assert resources.energy == 50.0

        # Restore beyond maximum
        resources.restore_energy(100.0)
        assert resources.energy == 100.0


class TestSocialRelationship:
    """Test SocialRelationship class"""

    def test_relationship_creation(self):
        """Test relationship creation"""
        rel = SocialRelationship(
            target_agent_id="agent_123",
            relationship_type="friend",
            trust_level=0.8,
            interaction_count=10,
        )

        assert rel.target_agent_id == "agent_123"
        assert rel.relationship_type == "friend"
        assert rel.trust_level == 0.8
        assert rel.interaction_count == 10
        assert rel.last_interaction is None

    def test_relationship_update_trust(self):
        """Test trust level updates"""
        rel = SocialRelationship(
            target_agent_id="agent_123", relationship_type="neutral", trust_level=0.5
        )

        # Increase trust
        rel.update_trust(0.1)
        assert rel.trust_level == 0.6

        # Decrease trust
        rel.update_trust(-0.2)
        assert abs(rel.trust_level - 0.4) < 1e-10

        # Clamp at boundaries
        rel.update_trust(1.0)
        assert rel.trust_level == 1.0

        rel.update_trust(-2.0)
        assert rel.trust_level == 0.0


class TestAgentGoal:
    """Test AgentGoal class"""

    def test_goal_creation(self):
        """Test goal creation"""
        goal = AgentGoal(
            description="Explore the area",
            priority=0.8,
            target_position=Position(100, 200),
            target_agent_id="agent_456",
            progress=0.3,
        )

        assert goal.goal_id is not None
        assert goal.description == "Explore the area"
        assert goal.priority == 0.8
        assert not goal.completed
        assert goal.progress == 0.3
        assert goal.target_position.x == 100
        assert goal.deadline is None

    def test_goal_unique_ids(self):
        """Test that goals have unique IDs"""
        goals = [AgentGoal(description="test") for _ in range(100)]
        goal_ids = [g.goal_id for g in goals]
        assert len(set(goal_ids)) == 100  # All unique

    def test_goal_expiration(self):
        """Test goal expiration"""
        # Goal without deadline
        goal1 = AgentGoal(description="test")
        assert not goal1.is_expired()

        # Goal with future deadline
        future_deadline = datetime.now() + timedelta(hours=1)
        goal2 = AgentGoal(description="test", deadline=future_deadline)
        assert not goal2.is_expired()

        # Goal with past deadline
        past_deadline = datetime.now() - timedelta(hours=1)
        goal3 = AgentGoal(description="test", deadline=past_deadline)
        assert goal3.is_expired()


class TestAgent:
    """Test main Agent class"""

    def test_agent_creation_minimal(self):
        """Test agent creation with minimal parameters"""
        agent = Agent()

        assert agent.agent_id is not None
        assert agent.name == "Agent"
        assert agent.agent_type == "basic"
        assert agent.status == AgentStatus.IDLE
        assert isinstance(agent.position, Position)
        assert isinstance(agent.orientation, Orientation)
        assert isinstance(agent.created_at, datetime)

    def test_agent_creation_custom(self):
        """Test agent creation with custom parameters"""
        position = Position(50, 50, 10)
        capabilities = {AgentCapability.MOVEMENT, AgentCapability.PERCEPTION}

        agent = Agent(
            agent_id="custom_123",
            name="CustomAgent",
            agent_type="explorer",
            position=position,
            capabilities=capabilities,
        )

        assert agent.agent_id == "custom_123"
        assert agent.name == "CustomAgent"
        assert agent.agent_type == "explorer"
        assert agent.position.x == 50
        assert AgentCapability.MOVEMENT in agent.capabilities

    def test_agent_unique_ids(self):
        """Test that agents have unique IDs by default"""
        agents = [Agent() for _ in range(100)]
        agent_ids = [a.agent_id for a in agents]
        assert len(set(agent_ids)) == 100  # All unique

    def test_agent_status_updates(self):
        """Test agent status updates"""
        agent = Agent()

        agent.update_status(AgentStatus.MOVING)
        assert agent.status == AgentStatus.MOVING
        assert agent.last_updated > agent.created_at

        agent.update_status(AgentStatus.INTERACTING)
        assert agent.status == AgentStatus.INTERACTING

    def test_agent_position_updates(self):
        """Test agent position updates"""
        agent = Agent(position=Position(0, 0))

        new_pos = Position(10, 20)
        agent.update_position(new_pos)

        assert agent.position.x == 10
        assert agent.position.y == 20
        assert agent.last_updated > agent.created_at

    def test_agent_memory_operations(self):
        """Test agent memory operations"""
        agent = Agent()

        # Add to short-term memory
        agent.add_to_memory({"event": "saw_tree", "location": (10, 20)})
        assert len(agent.short_term_memory) == 1
        assert agent.short_term_memory[0]["experience"]["event"] == "saw_tree"
        assert agent.experience_count == 1

        # Add important memory
        agent.add_to_memory({"event": "found_treasure"}, is_important=True)
        assert len(agent.long_term_memory) == 1
        assert agent.long_term_memory[0]["importance"] == True

    def test_agent_goal_management(self):
        """Test agent goal management"""
        agent = Agent()

        # Add goal
        goal = AgentGoal(description="Explore", priority=0.9)
        agent.add_goal(goal)

        assert len(agent.goals) == 1
        assert agent.goals[0].description == "Explore"

        # Goals should be sorted by priority
        goal2 = AgentGoal(description="Gather", priority=0.5)
        goal3 = AgentGoal(description="Build", priority=0.7)

        agent.add_goal(goal2)
        agent.add_goal(goal3)

        assert agent.goals[0].priority == 0.9
        assert agent.goals[1].priority == 0.7
        assert agent.goals[2].priority == 0.5

        # Select next goal
        next_goal = agent.select_next_goal()
        assert next_goal == goal
        assert agent.current_goal == goal

    def test_agent_capability_management(self):
        """Test capability management"""
        agent = Agent()

        # Check default capabilities
        assert agent.has_capability(AgentCapability.MOVEMENT)
        assert agent.has_capability(AgentCapability.PERCEPTION)

        # Add capability
        agent.add_capability(AgentCapability.PLANNING)
        assert agent.has_capability(AgentCapability.PLANNING)

        # Remove capability
        agent.remove_capability(AgentCapability.MOVEMENT)
        assert not agent.has_capability(AgentCapability.MOVEMENT)

    def test_agent_relationships(self):
        """Test relationship management"""
        agent = Agent()

        rel = SocialRelationship("friend_123", "friend", trust_level=0.8)
        agent.add_relationship(rel)

        assert len(agent.relationships) == 1
        assert agent.get_relationship("friend_123") == rel
        assert agent.get_relationship("unknown") is None

    def test_agent_serialization(self):
        """Test agent to_dict serialization"""
        agent = Agent(name="TestAgent", agent_type="explorer", position=Position(10, 20, 5))

        agent_dict = agent.to_dict()

        assert agent_dict["agent_id"] == agent.agent_id
        assert agent_dict["name"] == "TestAgent"
        assert agent_dict["agent_type"] == "explorer"
        assert agent_dict["status"] == "idle"
        assert agent_dict["position"]["x"] == 10
        assert agent_dict["position"]["y"] == 20
        assert agent_dict["position"]["z"] == 5

    def test_agent_from_dict(self):
        """Test agent deserialization"""
        data = {
            "agent_id": "test_123",
            "name": "TestAgent",
            "agent_type": "explorer",
            "position": {"x": 10, "y": 20, "z": 5},
            "status": "moving",
            "capabilities": ["movement", "perception"],
            "personality": {"openness": 0.8, "conscientiousness": 0.7},
            "resources": {"energy": 80.0, "health": 90.0},
        }

        agent = Agent.from_dict(data)

        assert agent.agent_id == "test_123"
        assert agent.name == "TestAgent"
        assert agent.position.x == 10
        assert agent.status == AgentStatus.MOVING
        assert AgentCapability.MOVEMENT in agent.capabilities
        assert agent.personality.openness == 0.8
        assert agent.resources.energy == 80.0


class TestExperience:
    """Test Experience class"""

    def test_experience_creation(self):
        """Test experience creation"""
        state = {"position": (10, 20), "energy": 80}
        action = Action(action_type=ActionType.MOVE)
        outcome = {"new_position": (15, 25), "energy_consumed": 5}
        next_state = {"position": (15, 25), "energy": 75}

        exp = Experience(
            state=state, action=action, outcome=outcome, reward=10.0, next_state=next_state
        )

        assert exp.state == state
        assert exp.action == action
        assert exp.outcome == outcome
        assert exp.reward == 10.0
        assert exp.next_state == next_state
        assert isinstance(exp.timestamp, datetime)


class TestSpecializedAgent:
    """Test SpecializedAgent class"""

    def test_specialized_agent_creation(self):
        """Test specialized agent creation"""
        agent = SpecializedAgent(
            name="SpecialAgent", agent_type="specialist", specialization="research"
        )

        assert agent.name == "SpecialAgent"
        assert agent.specialization == "research"
        assert isinstance(agent.specialized_capabilities, set)

    def test_specialized_capabilities(self):
        """Test specialized capabilities"""
        agent = SpecializedAgent()
        agent.specialized_capabilities.add("fast_learning")
        agent.specialized_capabilities.add("pattern_recognition")

        assert agent.has_specialized_capability("fast_learning")
        assert not agent.has_specialized_capability("unknown_skill")


class TestResourceAgent:
    """Test ResourceAgent class"""

    def test_resource_agent_creation(self):
        """Test resource agent creation"""
        agent = ResourceAgent(name="ResourceCollector")

        assert agent.name == "ResourceCollector"
        assert agent.specialization == "resource_management"
        assert agent.resource_efficiency == 1.0
        assert isinstance(agent.managed_resources, dict)
        assert isinstance(agent.trading_history, list)

        # Check capabilities added in __post_init__
        assert agent.has_capability(AgentCapability.RESOURCE_MANAGEMENT)
        assert agent.has_specialized_capability("trading")
        assert agent.has_specialized_capability("resource_optimization")

    def test_resource_agent_properties(self):
        """Test resource agent specific properties"""
        agent = ResourceAgent()

        # Add managed resources
        agent.managed_resources["wood"] = 100.0
        agent.managed_resources["stone"] = 50.0

        assert agent.managed_resources["wood"] == 100.0
        assert agent.managed_resources["stone"] == 50.0

        # Add trading history
        trade = {"item": "wood", "quantity": 10, "price": 5.0}
        agent.trading_history.append(trade)

        assert len(agent.trading_history) == 1
        assert agent.trading_history[0]["item"] == "wood"


class TestSocialAgent:
    """Test SocialAgent class"""

    def test_social_agent_creation(self):
        """Test social agent creation"""
        agent = SocialAgent(name="SocialButterfly")

        assert agent.name == "SocialButterfly"
        assert agent.specialization == "social_interaction"
        assert agent.influence_radius == 10.0
        assert agent.reputation == 0.5
        assert agent.communication_style == "neutral"
        assert agent.social_network_size == 0

        # Check capabilities added in __post_init__
        assert agent.has_capability(AgentCapability.SOCIAL_INTERACTION)
        assert agent.has_specialized_capability("negotiation")
        assert agent.has_specialized_capability("coalition_formation")
        assert agent.has_specialized_capability("influence")

    def test_social_agent_properties(self):
        """Test social agent specific properties"""
        agent = SocialAgent(
            influence_radius=20.0,
            reputation=0.8,
            communication_style="assertive",
            social_network_size=10,
        )

        assert agent.influence_radius == 20.0
        assert agent.reputation == 0.8
        assert agent.communication_style == "assertive"
        assert agent.social_network_size == 10


class TestComplexInteractions:
    """Test complex interactions between classes"""

    def test_agent_with_relationships(self):
        """Test agent with social relationships"""
        agent = Agent(name="SocialAgent")

        # Add relationships
        rel1 = SocialRelationship("friend_1", "friend", trust_level=0.8)
        rel2 = SocialRelationship("rival_1", "rival", trust_level=0.2)

        agent.add_relationship(rel1)
        agent.add_relationship(rel2)

        assert len(agent.relationships) == 2
        assert agent.relationships["friend_1"].trust_level == 0.8
        assert agent.relationships["rival_1"].relationship_type == "rival"

    def test_agent_action_scenarios(self):
        """Test various action scenarios"""
        agent = Agent(position=Position(0, 0))
        agent.resources.energy = 50.0

        # Test move action
        move_action = Action(
            action_type=ActionType.MOVE, target_position=Position(10, 10), energy_cost=20.0
        )
        assert move_action.can_execute(agent)

        # Test trade action
        trade_action = Action(
            action_type=ActionType.TRADE,
            target_agent_id="merchant_123",
            parameters={"item": "wood", "quantity": 5},
            energy_cost=10.0,
        )
        assert trade_action.can_execute(agent)

        # Test high energy action
        build_action = Action(action_type=ActionType.BUILD, energy_cost=60.0)
        assert not build_action.can_execute(agent)

    def test_specialized_agents_inheritance(self):
        """Test that specialized agents inherit from Agent"""
        resource_agent = ResourceAgent(name="Collector")
        social_agent = SocialAgent(name="Diplomat")

        # Both should have base Agent properties
        assert hasattr(resource_agent, "position")
        assert hasattr(resource_agent, "status")
        assert hasattr(social_agent, "goals")
        assert hasattr(social_agent, "personality")

        # And their specialized properties
        assert hasattr(resource_agent, "resource_efficiency")
        assert hasattr(social_agent, "influence_radius")
