"""
Comprehensive tests for GraphQL schema
"""

import asyncio
import os
import sys
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add mock strawberry to path
sys.path.insert(0, os.path.dirname(__file__))
import mock_strawberry as strawberry
from mock_strawberry import ExecutionResult, GraphQLTestClient

# Skip actual schema import during test collection to avoid type errors
try:
    from api.graphql.schema import (
        Agent,
        AgentAction,
        AgentActionInput,
        Coalition,
        CoalitionActionInput,
        CoalitionGoal,
        CoalitionMetrics,
        CreateAgentInput,
        CreateCoalitionInput,
        GraphQLAgentClass,
        GraphQLAgentStatus,
        GraphQLBiome,
        GraphQLCoalitionRole,
        GraphQLCoalitionStatus,
        GraphQLTerrainType,
        HexCell,
        Mutation,
        PersonalityProfile,
        Position,
        Query,
        SimulationMetrics,
        Subscription,
        UpdateAgentInput,
        WorldState,
        schema,
    )
except (ImportError, TypeError) as e:
    # Mock the schema objects for testing when imports fail
    Agent = Mock()
    AgentAction = Mock()
    AgentActionInput = Mock()
    Coalition = Mock()
    CoalitionActionInput = Mock()
    CoalitionGoal = Mock()
    CoalitionMetrics = Mock()
    CreateAgentInput = Mock()
    CreateCoalitionInput = Mock()
    GraphQLAgentClass = Mock()
    GraphQLAgentStatus = Mock()
    GraphQLBiome = Mock()
    GraphQLCoalitionRole = Mock()
    GraphQLCoalitionStatus = Mock()
    GraphQLTerrainType = Mock()
    HexCell = Mock()
    Mutation = Mock()
    PersonalityProfile = Mock()
    Position = Mock()
    Query = Mock()
    SimulationMetrics = Mock()
    Subscription = Mock()
    UpdateAgentInput = Mock()
    WorldState = Mock()
    schema = Mock()
    print(f"GraphQL schema import failed: {e}. Using mocks for testing.")


class TestGraphQLTypes:
    """Test GraphQL type definitions"""

    def test_position_type(self):
        """Test Position type"""
        pos = Position(x=10.0, y=20.0, z=5.0)
        assert pos.x == 10.0
        assert pos.y == 20.0
        assert pos.z == 5.0
        assert pos.hex_id() is None  # Mock implementation

    def test_personality_profile_type(self):
        """Test PersonalityProfile type"""
        profile = PersonalityProfile(
            openness=0.8,
            conscientiousness=0.7,
            extraversion=0.6,
            agreeableness=0.9,
            neuroticism=0.3,
        )
        assert profile.openness == 0.8
        assert profile.conscientiousness == 0.7
        assert profile.extraversion == 0.6
        assert profile.agreeableness == 0.9
        assert profile.neuroticism == 0.3

    def test_agent_type(self):
        """Test Agent type"""
        agent = Agent(
            id=strawberry.ID("agent_123"),
            name="Test Agent",
            agent_class=GraphQLAgentClass.EXPLORER,
            status=GraphQLAgentStatus.IDLE,
            position=Position(x=0, y=0, z=0),
            energy=100.0,
            health=100.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            personality=PersonalityProfile(
                openness=0.5,
                conscientiousness=0.5,
                extraversion=0.5,
                agreeableness=0.5,
                neuroticism=0.5,
            ),
            capabilities=["movement", "perception"],
            memory_size=1000,
            learning_rate=0.1,
        )

        assert agent.id == strawberry.ID("agent_123")
        assert agent.name == "Test Agent"
        assert agent.agent_class == GraphQLAgentClass.EXPLORER
        assert agent.status == GraphQLAgentStatus.IDLE
        assert agent.energy == 100.0
        assert agent.health == 100.0
        assert len(agent.capabilities) == 2

    def test_coalition_type(self):
        """Test Coalition type"""
        goal = CoalitionGoal(
            id=strawberry.ID("goal_1"),
            title="Test Goal",
            description="A test goal",
            priority=0.8,
            target_value=100.0,
            current_progress=50.0,
            success_threshold=0.8,
            status="in_progress",
            created_at=datetime.utcnow(),
            assigned_members=[strawberry.ID("agent_1")],
            required_capabilities=["exploration"],
            resource_requirements={"energy": 100},
        )

        coalition = Coalition(
            id=strawberry.ID("coalition_123"),
            name="Test Coalition",
            description="A test coalition",
            status=GraphQLCoalitionStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            member_count=5,
            total_resources={"energy": 500},
            shared_goals=[goal],
        )

        assert coalition.id == strawberry.ID("coalition_123")
        assert coalition.name == "Test Coalition"
        assert coalition.status == GraphQLCoalitionStatus.ACTIVE
        assert coalition.member_count == 5
        assert len(coalition.shared_goals) == 1

    def test_hex_cell_type(self):
        """Test HexCell type"""
        cell = HexCell(
            hex_id=strawberry.ID("8928308280fffff"),
            biome=GraphQLBiome.FOREST,
            terrain=GraphQLTerrainType.HILLS,
            elevation=250.0,
            temperature=20.0,
            moisture=60.0,
            resources={"wood": 100, "water": 50},
            visibility_range=5.0,
            movement_cost=1.5,
        )

        assert cell.hex_id == strawberry.ID("8928308280fffff")
        assert cell.biome == GraphQLBiome.FOREST
        assert cell.terrain == GraphQLTerrainType.HILLS
        assert cell.elevation == 250.0
        assert cell.resources["wood"] == 100

    def test_world_state_type(self):
        """Test WorldState type"""
        state = WorldState(
            current_time=datetime.utcnow(),
            total_agents=100,
            total_coalitions=10,
            active_agents=85,
            resource_totals={"energy": 10000, "materials": 5000},
            weather_conditions={"temperature": 20, "humidity": 60},
        )

        assert state.total_agents == 100
        assert state.total_coalitions == 10
        assert state.active_agents == 85
        assert state.resource_totals["energy"] == 10000


class TestGraphQLEnums:
    """Test GraphQL enum definitions"""

    def test_agent_status_enum(self):
        """Test AgentStatus enum values"""
        assert GraphQLAgentStatus.IDLE.value == "idle"
        assert GraphQLAgentStatus.MOVING.value == "moving"
        assert GraphQLAgentStatus.INTERACTING.value == "interacting"
        assert GraphQLAgentStatus.PLANNING.value == "planning"
        assert GraphQLAgentStatus.LEARNING.value == "learning"
        assert GraphQLAgentStatus.OFFLINE.value == "offline"
        assert GraphQLAgentStatus.ERROR.value == "error"

    def test_agent_class_enum(self):
        """Test AgentClass enum values"""
        assert GraphQLAgentClass.EXPLORER.value == "explorer"
        assert GraphQLAgentClass.MERCHANT.value == "merchant"
        assert GraphQLAgentClass.SCHOLAR.value == "scholar"
        assert GraphQLAgentClass.GUARDIAN.value == "guardian"

    def test_coalition_status_enum(self):
        """Test CoalitionStatus enum values"""
        assert GraphQLCoalitionStatus.FORMING.value == "forming"
        assert GraphQLCoalitionStatus.ACTIVE.value == "active"
        assert GraphQLCoalitionStatus.SUSPENDED.value == "suspended"
        assert GraphQLCoalitionStatus.DISSOLVING.value == "dissolving"
        assert GraphQLCoalitionStatus.DISSOLVED.value == "dissolved"

    def test_biome_enum(self):
        """Test Biome enum values"""
        assert GraphQLBiome.FOREST.value == "forest"
        assert GraphQLBiome.DESERT.value == "desert"
        assert GraphQLBiome.OCEAN.value == "ocean"
        assert GraphQLBiome.MOUNTAIN.value == "mountain"

    def test_terrain_type_enum(self):
        """Test TerrainType enum values"""
        assert GraphQLTerrainType.FLAT.value == "flat"
        assert GraphQLTerrainType.HILLS.value == "hills"
        assert GraphQLTerrainType.MOUNTAINS.value == "mountains"
        assert GraphQLTerrainType.WATER.value == "water"


class TestGraphQLInputTypes:
    """Test GraphQL input type definitions"""

    def test_create_agent_input(self):
        """Test CreateAgentInput"""
        input_data = CreateAgentInput(
            name="New Agent",
            agent_class=GraphQLAgentClass.EXPLORER,
            starting_position=Position(x=10, y=20, z=0),
            initial_energy=150.0,
        )

        assert input_data.name == "New Agent"
        assert input_data.agent_class == GraphQLAgentClass.EXPLORER
        assert input_data.starting_position.x == 10
        assert input_data.initial_energy == 150.0

    def test_update_agent_input(self):
        """Test UpdateAgentInput"""
        input_data = UpdateAgentInput(
            agent_id=strawberry.ID("agent_123"),
            name="Updated Name",
            status=GraphQLAgentStatus.MOVING,
            energy=75.0,
        )

        assert input_data.agent_id == strawberry.ID("agent_123")
        assert input_data.name == "Updated Name"
        assert input_data.status == GraphQLAgentStatus.MOVING
        assert input_data.energy == 75.0
        assert input_data.health is None  # Optional field

    def test_agent_action_input(self):
        """Test AgentActionInput"""
        input_data = AgentActionInput(
            agent_id=strawberry.ID("agent_123"),
            action_type="move",
            target_location=Position(x=5, y=5, z=0),
            parameters={"speed": "fast"},
        )

        assert input_data.agent_id == strawberry.ID("agent_123")
        assert input_data.action_type == "move"
        assert input_data.target_location.x == 5
        assert input_data.parameters["speed"] == "fast"

    def test_create_coalition_input(self):
        """Test CreateCoalitionInput"""
        input_data = CreateCoalitionInput(
            name="New Coalition",
            description="A new test coalition",
            founding_members=[strawberry.ID("agent_1"), strawberry.ID("agent_2")],
            initial_goals=["explore", "gather resources"],
        )

        assert input_data.name == "New Coalition"
        assert len(input_data.founding_members) == 2
        assert len(input_data.initial_goals) == 2


@pytest.mark.asyncio
class TestGraphQLQueries:
    """Test GraphQL query operations"""

    async def test_query_agent(self):
        """Test agent query"""
        query = Query()

        # Test None response (not found)
        result = await query.agent(info=Mock(), id=strawberry.ID("nonexistent"))
        assert result is None

    async def test_query_agents(self):
        """Test agents list query"""
        query = Query()

        # Test empty list
        result = await query.agents(info=Mock())
        assert result == []

        # Test with filters
        result = await query.agents(
            info=Mock(),
            limit=50,
            status=GraphQLAgentStatus.IDLE,
            agent_class=GraphQLAgentClass.EXPLORER,
        )
        assert result == []

    async def test_query_coalition(self):
        """Test coalition query"""
        query = Query()

        result = await query.coalition(info=Mock(), id=strawberry.ID("coalition_123"))
        assert result is None

    async def test_query_coalitions(self):
        """Test coalitions list query"""
        query = Query()

        result = await query.coalitions(info=Mock(), limit=20, status=GraphQLCoalitionStatus.ACTIVE)
        assert result == []

    async def test_query_world_state(self):
        """Test world state query"""
        query = Query()

        result = await query.world_state(info=Mock())
        assert isinstance(result, WorldState)
        assert result.total_agents == 0
        assert result.total_coalitions == 0
        assert isinstance(result.current_time, datetime)

    async def test_query_simulation_metrics(self):
        """Test simulation metrics query"""
        query = Query()

        result = await query.simulation_metrics(info=Mock())
        assert isinstance(result, SimulationMetrics)
        assert result.fps == 60.0
        assert result.agent_count == 0
        assert isinstance(result.timestamp, datetime)

    async def test_query_search_agents(self):
        """Test agent search query"""
        query = Query()

        result = await query.search_agents(info=Mock(), query="explorer", limit=10)
        assert result == []

    async def test_query_nearby_agents(self):
        """Test nearby agents query"""
        query = Query()

        result = await query.nearby_agents(
            info=Mock(), position=Position(x=0, y=0, z=0), radius=10.0
        )
        assert result == []


@pytest.mark.asyncio
class TestGraphQLMutations:
    """Test GraphQL mutation operations"""

    async def test_mutation_create_agent(self):
        """Test create agent mutation"""
        mutation = Mutation()

        input_data = CreateAgentInput(
            name="Test Agent", agent_class=GraphQLAgentClass.EXPLORER, initial_energy=100.0
        )

        result = await mutation.create_agent(info=Mock(), input=input_data)
        assert isinstance(result, Agent)
        assert result.name == "Test Agent"
        assert result.agent_class == GraphQLAgentClass.EXPLORER
        assert result.energy == 100.0

    async def test_mutation_update_agent(self):
        """Test update agent mutation"""
        mutation = Mutation()

        input_data = UpdateAgentInput(agent_id=strawberry.ID("agent_123"), name="Updated Agent")

        with pytest.raises(NotImplementedError):
            await mutation.update_agent(info=Mock(), input=input_data)

    async def test_mutation_delete_agent(self):
        """Test delete agent mutation"""
        mutation = Mutation()

        result = await mutation.delete_agent(info=Mock(), id=strawberry.ID("agent_123"))
        assert result is True

    async def test_mutation_perform_agent_action(self):
        """Test perform agent action mutation"""
        mutation = Mutation()

        input_data = AgentActionInput(
            agent_id=strawberry.ID("agent_123"),
            action_type="move",
            target_location=Position(x=5, y=5, z=0),
        )

        result = await mutation.perform_agent_action(info=Mock(), input=input_data)
        assert isinstance(result, AgentAction)
        assert result.agent_id == input_data.agent_id
        assert result.action_type == "move"
        assert result.success is True

    async def test_mutation_create_coalition(self):
        """Test create coalition mutation"""
        mutation = Mutation()

        input_data = CreateCoalitionInput(
            name="Test Coalition",
            description="A test coalition",
            founding_members=[strawberry.ID("agent_1"), strawberry.ID("agent_2")],
        )

        result = await mutation.create_coalition(info=Mock(), input=input_data)
        assert isinstance(result, Coalition)
        assert result.name == "Test Coalition"
        assert result.member_count == 2
        assert result.status == GraphQLCoalitionStatus.FORMING

    async def test_mutation_join_coalition(self):
        """Test join coalition mutation"""
        mutation = Mutation()

        with pytest.raises(NotImplementedError):
            await mutation.join_coalition(
                info=Mock(),
                agent_id=strawberry.ID("agent_123"),
                coalition_id=strawberry.ID("coalition_123"),
            )

    async def test_mutation_leave_coalition(self):
        """Test leave coalition mutation"""
        mutation = Mutation()

        result = await mutation.leave_coalition(
            info=Mock(),
            agent_id=strawberry.ID("agent_123"),
            coalition_id=strawberry.ID("coalition_123"),
        )
        assert result is True

    async def test_mutation_dissolve_coalition(self):
        """Test dissolve coalition mutation"""
        mutation = Mutation()

        result = await mutation.dissolve_coalition(
            info=Mock(), coalition_id=strawberry.ID("coalition_123")
        )
        assert result is True


@pytest.mark.asyncio
class TestGraphQLSubscriptions:
    """Test GraphQL subscription operations"""

    async def test_subscription_agent_updates(self):
        """Test agent updates subscription"""
        subscription = Subscription()

        # Get the async generator
        agent_gen = subscription.agent_updates(info=Mock(), agent_id=strawberry.ID("agent_123"))

        # Get first update
        agent = await agent_gen.__anext__()
        assert isinstance(agent, Agent)
        assert agent.id == strawberry.ID("agent_123")

        # Cleanup
        await agent_gen.aclose()

    async def test_subscription_coalition_updates(self):
        """Test coalition updates subscription"""
        subscription = Subscription()

        # Get the async generator
        coalition_gen = subscription.coalition_updates(
            info=Mock(), coalition_id=strawberry.ID("coalition_123")
        )

        # Get first update
        coalition = await coalition_gen.__anext__()
        assert isinstance(coalition, Coalition)
        assert coalition.id == strawberry.ID("coalition_123")

        # Cleanup
        await coalition_gen.aclose()

    async def test_subscription_world_events(self):
        """Test world events subscription"""
        subscription = Subscription()

        # Get the async generator
        event_gen = subscription.world_events(
            info=Mock(), hex_id="8928308280fffff", event_types=["agent_moved"]
        )

        # Get first event
        event = await event_gen.__anext__()
        assert isinstance(event, dict)
        assert event["event_type"] == "agent_moved"
        assert event["hex_id"] == "8928308280fffff"

        # Cleanup
        await event_gen.aclose()

    async def test_subscription_simulation_metrics_stream(self):
        """Test simulation metrics subscription"""
        subscription = Subscription()

        # Get the async generator
        metrics_gen = subscription.simulation_metrics_stream(info=Mock())

        # Get first metrics update
        metrics = await metrics_gen.__anext__()
        assert isinstance(metrics, SimulationMetrics)
        assert metrics.fps == 60.0
        assert metrics.agent_count == 100

        # Cleanup
        await metrics_gen.aclose()


class TestGraphQLFieldResolvers:
    """Test GraphQL field resolvers"""

    @pytest.mark.asyncio
    async def test_agent_coalition_resolver(self):
        """Test Agent.coalition field resolver"""
        agent = Agent(
            id=strawberry.ID("agent_123"),
            name="Test Agent",
            agent_class=GraphQLAgentClass.EXPLORER,
            status=GraphQLAgentStatus.IDLE,
            position=Position(x=0, y=0, z=0),
            energy=100.0,
            health=100.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            personality=PersonalityProfile(
                openness=0.5,
                conscientiousness=0.5,
                extraversion=0.5,
                agreeableness=0.5,
                neuroticism=0.5,
            ),
            capabilities=[],
            memory_size=1000,
            learning_rate=0.1,
            coalition_id=None,
        )

        # No coalition ID
        result = await agent.coalition(info=Mock())
        assert result is None

        # With coalition ID
        agent.coalition_id = strawberry.ID("coalition_123")
        result = await agent.coalition(info=Mock())
        assert result is None  # Mock implementation returns None

    @pytest.mark.asyncio
    async def test_agent_recent_actions_resolver(self):
        """Test Agent.recent_actions field resolver"""
        agent = Agent(
            id=strawberry.ID("agent_123"),
            name="Test Agent",
            agent_class=GraphQLAgentClass.EXPLORER,
            status=GraphQLAgentStatus.IDLE,
            position=Position(x=0, y=0, z=0),
            energy=100.0,
            health=100.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            personality=PersonalityProfile(
                openness=0.5,
                conscientiousness=0.5,
                extraversion=0.5,
                agreeableness=0.5,
                neuroticism=0.5,
            ),
            capabilities=[],
            memory_size=1000,
            learning_rate=0.1,
        )

        result = await agent.recent_actions(info=Mock(), limit=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_coalition_members_resolver(self):
        """Test Coalition.members field resolver"""
        coalition = Coalition(
            id=strawberry.ID("coalition_123"),
            name="Test Coalition",
            description="A test coalition",
            status=GraphQLCoalitionStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            member_count=5,
            total_resources={},
            shared_goals=[],
        )

        result = await coalition.members(info=Mock())
        assert result == []

    @pytest.mark.asyncio
    async def test_coalition_leader_resolver(self):
        """Test Coalition.leader field resolver"""
        coalition = Coalition(
            id=strawberry.ID("coalition_123"),
            name="Test Coalition",
            description="A test coalition",
            status=GraphQLCoalitionStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            member_count=5,
            total_resources={},
            shared_goals=[],
        )

        result = await coalition.leader(info=Mock())
        assert result is None

    @pytest.mark.asyncio
    async def test_coalition_performance_metrics_resolver(self):
        """Test Coalition.performance_metrics field resolver"""
        coalition = Coalition(
            id=strawberry.ID("coalition_123"),
            name="Test Coalition",
            description="A test coalition",
            status=GraphQLCoalitionStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            member_count=5,
            total_resources={},
            shared_goals=[],
        )

        result = await coalition.performance_metrics(info=Mock())
        assert isinstance(result, CoalitionMetrics)
        assert result.coalition_id == coalition.id
        assert result.goal_completion_rate == 0.0

    @pytest.mark.asyncio
    async def test_hex_cell_agents_present_resolver(self):
        """Test HexCell.agents_present field resolver"""
        cell = HexCell(
            hex_id=strawberry.ID("8928308280fffff"),
            biome=GraphQLBiome.FOREST,
            terrain=GraphQLTerrainType.FLAT,
            elevation=100.0,
            temperature=20.0,
            moisture=50.0,
            resources={},
            visibility_range=5.0,
            movement_cost=1.0,
        )

        result = await cell.agents_present(info=Mock())
        assert result == []

    @pytest.mark.asyncio
    async def test_hex_cell_neighboring_cells_resolver(self):
        """Test HexCell.neighboring_cells field resolver"""
        cell = HexCell(
            hex_id=strawberry.ID("8928308280fffff"),
            biome=GraphQLBiome.FOREST,
            terrain=GraphQLTerrainType.FLAT,
            elevation=100.0,
            temperature=20.0,
            moisture=50.0,
            resources={},
            visibility_range=5.0,
            movement_cost=1.0,
        )

        result = await cell.neighboring_cells(info=Mock())
        assert result == []

    @pytest.mark.asyncio
    async def test_world_state_get_hex_cell_resolver(self):
        """Test WorldState.get_hex_cell field resolver"""
        state = WorldState(
            current_time=datetime.utcnow(),
            total_agents=100,
            total_coalitions=10,
            active_agents=85,
            resource_totals={},
            weather_conditions={},
        )

        result = await state.get_hex_cell(info=Mock(), hex_id="8928308280fffff")
        assert result is None

    @pytest.mark.asyncio
    async def test_world_state_get_region_resolver(self):
        """Test WorldState.get_region field resolver"""
        state = WorldState(
            current_time=datetime.utcnow(),
            total_agents=100,
            total_coalitions=10,
            active_agents=85,
            resource_totals={},
            weather_conditions={},
        )

        result = await state.get_region(info=Mock(), center_hex="8928308280fffff", radius=2)
        assert result == []


class TestGraphQLSchema:
    """Test the GraphQL schema itself"""

    def test_schema_creation(self):
        """Test that schema is created successfully"""
        assert schema is not None
        assert hasattr(schema, "query_type")
        assert hasattr(schema, "mutation_type")
        assert hasattr(schema, "subscription_type")

    def test_schema_query_type(self):
        """Test schema query type"""
        assert schema.query_type is Query

    def test_schema_mutation_type(self):
        """Test schema mutation type"""
        assert schema.mutation_type is Mutation

    def test_schema_subscription_type(self):
        """Test schema subscription type"""
        assert schema.subscription_type is Subscription

    @pytest.mark.asyncio
    async def test_schema_introspection(self):
        """Test schema introspection query"""
        introspection_query = """
            query {
                __schema {
                    queryType {
                        name
                        fields {
                            name
                            type {
                                name
                            }
                        }
                    }
                }
            }
        """

        result = await schema.execute(introspection_query)
        assert result.errors is None
        assert result.data is not None
        assert result.data["__schema"]["queryType"]["name"] == "Query"
        assert len(result.data["__schema"]["queryType"]["fields"]) > 0

    @pytest.mark.asyncio
    async def test_simple_query_execution(self):
        """Test executing a simple query"""
        query = """
            query {
                worldState {
                    totalAgents
                    totalCoalitions
                    activeAgents
                }
            }
        """

        result = await schema.execute(query)
        assert result.errors is None
        assert result.data is not None
        assert result.data["worldState"]["totalAgents"] == 0
        assert result.data["worldState"]["totalCoalitions"] == 0
        assert result.data["worldState"]["activeAgents"] == 0

    @pytest.mark.asyncio
    async def test_mutation_execution(self):
        """Test executing a mutation"""
        mutation = """
            mutation {
                createAgent(input: {
                    name: "Test Agent"
                    agentClass: EXPLORER
                    initialEnergy: 100.0
                }) {
                    id
                    name
                    agentClass
                    energy
                }
            }
        """

        result = await schema.execute(mutation)
        assert result.errors is None
        assert result.data is not None
        assert result.data["createAgent"]["name"] == "Test Agent"
        assert result.data["createAgent"]["agentClass"] == "EXPLORER"
        assert result.data["createAgent"]["energy"] == 100.0

    @pytest.mark.asyncio
    async def test_query_with_variables(self):
        """Test query with variables"""
        query = """
            query GetAgent($id: ID!) {
                agent(id: $id) {
                    id
                    name
                }
            }
        """

        variables = {"id": "agent_123"}
        result = await schema.execute(query, variable_values=variables)
        assert result.errors is None
        assert result.data is not None
        assert result.data["agent"] is None  # Mock returns None

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in schema"""
        # Invalid query
        query = """
            query {
                invalidField
            }
        """

        result = await schema.execute(query)
        assert result.errors is not None
        assert len(result.errors) > 0
        assert result.data is None

    @pytest.mark.asyncio
    async def test_nested_query_execution(self):
        """Test executing a nested query"""
        query = """
            query {
                simulationMetrics {
                    timestamp
                    fps
                    agentCount
                    coalitionCount
                    totalInteractions
                }
            }
        """

        result = await schema.execute(query)
        assert result.errors is None
        assert result.data is not None
        assert result.data["simulationMetrics"]["fps"] == 60.0
        assert result.data["simulationMetrics"]["agentCount"] == 0


class TestGraphQLExports:
    """Test module exports"""

    def test_exports(self):
        """Test that all expected exports are available"""
        from api.graphql.schema import __all__

        expected_exports = [
            "schema",
            "Query",
            "Mutation",
            "Subscription",
            "Agent",
            "Coalition",
            "HexCell",
            "WorldState",
            "CreateAgentInput",
            "CreateCoalitionInput",
            "AgentActionInput",
            "CoalitionActionInput",
        ]

        for export in expected_exports:
            assert export in __all__


class TestImportFallbacks:
    """Test import fallback mechanisms"""

    def test_strawberry_fallback(self):
        """Test that strawberry import fallback works"""
        # This is already being tested by the fact that tests run without strawberry
        from api.graphql.schema import strawberry

        assert hasattr(strawberry, "type")
        assert hasattr(strawberry, "field")
        assert hasattr(strawberry, "enum")
        assert hasattr(strawberry, "mutation")
        assert hasattr(strawberry, "subscription")
        assert hasattr(strawberry, "ID")
        assert hasattr(strawberry, "Schema")

    def test_enum_definitions(self):
        """Test that enum fallbacks are defined"""
        from api.graphql.schema import (
            ActionType,
            AgentCapability,
            AgentClass,
            AgentStatus,
            Biome,
            CoalitionGoalStatus,
            CoalitionRole,
            CoalitionStatus,
            PersonalityTraits,
            TerrainType,
        )

        # Test AgentStatus
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.MOVING.value == "moving"

        # Test AgentClass
        assert AgentClass.EXPLORER.value == "explorer"
        assert AgentClass.MERCHANT.value == "merchant"

        # Test PersonalityTraits
        assert PersonalityTraits.OPENNESS.value == "openness"

        # Test AgentCapability
        assert AgentCapability.MOVEMENT.value == "movement"

        # Test ActionType
        assert ActionType.MOVE.value == "move"

        # Test CoalitionStatus
        assert CoalitionStatus.FORMING.value == "forming"

        # Test CoalitionRole
        assert CoalitionRole.LEADER.value == "leader"

        # Test CoalitionGoalStatus
        assert CoalitionGoalStatus.PROPOSED.value == "proposed"

        # Test Biome
        assert Biome.FOREST.value == "forest"

        # Test TerrainType
        assert TerrainType.FLAT.value == "flat"


class TestComplexFieldResolvers:
    """Test complex field resolver scenarios"""

    @pytest.mark.asyncio
    async def test_position_hex_id_calculation(self):
        """Test Position hex_id method"""
        pos = Position(x=37.7749, y=-122.4194, z=0)  # San Francisco coords
        hex_id = pos.hex_id()
        # Currently returns None in mock implementation
        assert hex_id is None

    @pytest.mark.asyncio
    async def test_world_state_complex_queries(self):
        """Test WorldState with complex region queries"""
        state = WorldState(
            current_time=datetime.utcnow(),
            total_agents=1000,
            total_coalitions=50,
            active_agents=950,
            resource_totals={"energy": 50000, "materials": 25000, "knowledge": 10000},
            weather_conditions={"temperature": 25, "humidity": 60, "wind_speed": 10},
        )

        # Test getting hex cell
        cell = await state.get_hex_cell(info=Mock(), hex_id="8928308280fffff")
        assert cell is None  # Mock implementation

        # Test getting region with different radii
        for radius in [1, 3, 5]:
            region = await state.get_region(
                info=Mock(), center_hex="8928308280fffff", radius=radius
            )
            assert region == []  # Mock implementation

    @pytest.mark.asyncio
    async def test_agent_with_full_capabilities(self):
        """Test agent with all capabilities and coalition"""
        agent = Agent(
            id=strawberry.ID("agent_complex"),
            name="Complex Agent",
            agent_class=GraphQLAgentClass.SCHOLAR,
            status=GraphQLAgentStatus.LEARNING,
            position=Position(x=10, y=20, z=5),
            energy=75.0,
            health=90.0,
            created_at=datetime(2024, 1, 1),
            updated_at=datetime(2024, 1, 2),
            personality=PersonalityProfile(
                openness=0.9,
                conscientiousness=0.8,
                extraversion=0.3,
                agreeableness=0.7,
                neuroticism=0.2,
            ),
            capabilities=[
                "movement",
                "perception",
                "communication",
                "memory",
                "learning",
                "planning",
                "resource_management",
                "social_interaction",
            ],
            memory_size=5000,
            learning_rate=0.15,
            coalition_id=strawberry.ID("coalition_456"),
        )

        # Test all fields
        assert agent.id == strawberry.ID("agent_complex")
        assert agent.agent_class == GraphQLAgentClass.SCHOLAR
        assert agent.status == GraphQLAgentStatus.LEARNING
        assert len(agent.capabilities) == 8
        assert agent.coalition_id == strawberry.ID("coalition_456")

        # Test resolvers
        coalition = await agent.coalition(info=Mock())
        assert coalition is None  # Mock returns None

        actions = await agent.recent_actions(info=Mock(), limit=20)
        assert actions == []  # Mock returns empty list


class TestMutationEdgeCases:
    """Test mutation edge cases and error scenarios"""

    @pytest.mark.asyncio
    async def test_create_agent_with_all_options(self):
        """Test creating agent with all optional parameters"""
        mutation = Mutation()

        personality = PersonalityProfile(
            openness=0.95,
            conscientiousness=0.85,
            extraversion=0.75,
            agreeableness=0.65,
            neuroticism=0.15,
        )

        input_data = CreateAgentInput(
            name="Fully Configured Agent",
            agent_class=GraphQLAgentClass.GUARDIAN,
            starting_position=Position(x=100, y=200, z=50),
            personality_profile=personality,
            initial_energy=200.0,
        )

        result = await mutation.create_agent(info=Mock(), input=input_data)
        assert result.name == "Fully Configured Agent"
        assert result.agent_class == GraphQLAgentClass.GUARDIAN
        assert result.energy == 200.0
        assert result.position.x == 100
        assert result.position.y == 200
        assert result.position.z == 50

    @pytest.mark.asyncio
    async def test_coalition_action_mutation(self):
        """Test coalition action with complex parameters"""
        mutation = Mutation()

        # Test creating coalition with initial goals
        input_data = CreateCoalitionInput(
            name="Advanced Coalition",
            description="A coalition with complex goals",
            founding_members=[
                strawberry.ID("agent_1"),
                strawberry.ID("agent_2"),
                strawberry.ID("agent_3"),
                strawberry.ID("agent_4"),
                strawberry.ID("agent_5"),
            ],
            initial_goals=[
                "explore_new_territory",
                "establish_trade_routes",
                "research_advanced_technology",
                "defend_territory",
                "cultural_exchange",
            ],
        )

        result = await mutation.create_coalition(info=Mock(), input=input_data)
        assert result.name == "Advanced Coalition"
        assert result.member_count == 5
        assert result.status == GraphQLCoalitionStatus.FORMING
