"""
GraphQL Schema for FreeAgentics API.

This module defines the GraphQL schema for the FreeAgentics system,
exposing queries, mutations, and subscriptions for agents, coalitions,
and world state management.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

# Try to import strawberry, fall back to mock for testing
try:
    import strawberry
    from strawberry.scalars import JSON
    from strawberry.types import Info
except ImportError:
    # For testing without strawberry installed
    import os
    import sys

    test_dir = os.path.join(os.path.dirname(__file__), "..", "..", "tests", "unit")
    if test_dir not in sys.path:
        sys.path.insert(0, test_dir)
    try:
        import mock_strawberry as strawberry
        from mock_strawberry import JSON, Info
    except ImportError:
        # If mock is also not available, define minimal types
        class strawberry:
            @staticmethod
            def type(cls):
                return cls

            @staticmethod
            def input(cls):
                return cls

            @staticmethod
            def enum(cls):
                return cls

            @staticmethod
            def field(func):
                return func

            @staticmethod
            def mutation(func):
                return func

            @staticmethod
            def subscription(func):
                return func

            ID = str

            class Schema:
                def __init__(self, **kwargs):
                    pass

        Info = object
        JSON = dict

# Import data models (these will be mocked in tests)
try:
    from agents.base.data_model import (
        ActionType,
        AgentCapability,
        AgentClass,
        AgentStatus,
        PersonalityTraits,
    )
except ImportError:
    # Define minimal enums for testing
    class AgentStatus(Enum):
        IDLE = "idle"
        MOVING = "moving"
        INTERACTING = "interacting"
        PLANNING = "planning"
        LEARNING = "learning"
        OFFLINE = "offline"
        ERROR = "error"

    class AgentClass(Enum):
        EXPLORER = "explorer"
        MERCHANT = "merchant"
        SCHOLAR = "scholar"
        GUARDIAN = "guardian"

    class PersonalityTraits(Enum):
        OPENNESS = "openness"
        CONSCIENTIOUSNESS = "conscientiousness"
        EXTRAVERSION = "extraversion"
        AGREEABLENESS = "agreeableness"
        NEUROTICISM = "neuroticism"

    class AgentCapability(Enum):
        MOVEMENT = "movement"
        PERCEPTION = "perception"
        COMMUNICATION = "communication"
        MEMORY = "memory"
        LEARNING = "learning"
        PLANNING = "planning"
        RESOURCE_MANAGEMENT = "resource_management"
        SOCIAL_INTERACTION = "social_interaction"

    class ActionType(Enum):
        MOVE = "move"
        COMMUNICATE = "communicate"
        GATHER = "gather"
        EXPLORE = "explore"
        TRADE = "trade"
        LEARN = "learn"
        WAIT = "wait"
        ATTACK = "attack"
        DEFEND = "defend"
        BUILD = "build"


try:
    from coalitions.coalition.coalition_models import (
        CoalitionGoalStatus,
        CoalitionRole,
        CoalitionStatus,
    )
except ImportError:

    class CoalitionStatus(Enum):
        FORMING = "forming"
        ACTIVE = "active"
        SUSPENDED = "suspended"
        DISSOLVING = "dissolving"
        DISSOLVED = "dissolved"

    class CoalitionRole(Enum):
        LEADER = "leader"
        COORDINATOR = "coordinator"
        CONTRIBUTOR = "contributor"
        SPECIALIST = "specialist"
        OBSERVER = "observer"

    class CoalitionGoalStatus(Enum):
        PROPOSED = "proposed"
        ACCEPTED = "accepted"
        IN_PROGRESS = "in_progress"
        COMPLETED = "completed"
        FAILED = "failed"
        ABANDONED = "abandoned"


try:
    from world.h3_world import Biome, TerrainType
except ImportError:

    class Biome(Enum):
        ARCTIC = "arctic"
        TUNDRA = "tundra"
        FOREST = "forest"
        GRASSLAND = "grassland"
        DESERT = "desert"
        SAVANNA = "savanna"
        JUNGLE = "jungle"
        MOUNTAIN = "mountain"
        COASTAL = "coastal"
        OCEAN = "ocean"

    class TerrainType(Enum):
        FLAT = "flat"
        HILLS = "hills"
        MOUNTAINS = "mountains"
        WATER = "water"
        MARSH = "marsh"
        SAND = "sand"


# GraphQL Enums
@strawberry.enum
class GraphQLAgentStatus(Enum):
    IDLE = "idle"
    MOVING = "moving"
    INTERACTING = "interacting"
    PLANNING = "planning"
    LEARNING = "learning"
    OFFLINE = "offline"
    ERROR = "error"


@strawberry.enum
class GraphQLAgentClass(Enum):
    EXPLORER = "explorer"
    MERCHANT = "merchant"
    SCHOLAR = "scholar"
    GUARDIAN = "guardian"


@strawberry.enum
class GraphQLCoalitionStatus(Enum):
    FORMING = "forming"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DISSOLVING = "dissolving"
    DISSOLVED = "dissolved"


@strawberry.enum
class GraphQLCoalitionRole(Enum):
    LEADER = "leader"
    COORDINATOR = "coordinator"
    CONTRIBUTOR = "contributor"
    SPECIALIST = "specialist"
    OBSERVER = "observer"


@strawberry.enum
class GraphQLBiome(Enum):
    ARCTIC = "arctic"
    TUNDRA = "tundra"
    FOREST = "forest"
    GRASSLAND = "grassland"
    DESERT = "desert"
    SAVANNA = "savanna"
    JUNGLE = "jungle"
    MOUNTAIN = "mountain"
    COASTAL = "coastal"
    OCEAN = "ocean"


@strawberry.enum
class GraphQLTerrainType(Enum):
    FLAT = "flat"
    HILLS = "hills"
    MOUNTAINS = "mountains"
    WATER = "water"
    MARSH = "marsh"
    SAND = "sand"


# GraphQL Types
@strawberry.type
class Position:
    x: float
    y: float
    z: float = 0.0

    @strawberry.field
    def hex_id(self) -> Optional[str]:
        """Get H3 hex ID for this position"""
        # Implementation would convert coordinates to H3
        return None


@strawberry.type
class PersonalityProfile:
    openness: float
    conscientiousness: float
    extraversion: float
    agreeableness: float
    neuroticism: float


@strawberry.type
class Agent:
    id: strawberry.ID
    name: str
    agent_class: GraphQLAgentClass
    status: GraphQLAgentStatus
    position: Position
    energy: float
    health: float
    created_at: datetime
    updated_at: datetime
    personality: PersonalityProfile
    capabilities: List[str]
    memory_size: int
    learning_rate: float
    coalition_id: Optional[strawberry.ID] = None

    @strawberry.field
    async def coalition(self, info: Info) -> Optional["Coalition"]:
        """Resolve the agent's coalition"""
        if not self.coalition_id:
            return None
        # Would fetch from database/service
        return None

    @strawberry.field
    async def recent_actions(self, info: Info, limit: int = 10) -> List["AgentAction"]:
        """Get recent actions performed by this agent"""
        # Would fetch from event store
        return []


@strawberry.type
class AgentAction:
    id: strawberry.ID
    agent_id: strawberry.ID
    action_type: str
    success: bool
    energy_cost: float
    timestamp: datetime
    data: JSON
    target_location: Optional[Position] = None
    target_agent_id: Optional[strawberry.ID] = None


@strawberry.type
class CoalitionGoal:
    id: strawberry.ID
    title: str
    description: str
    priority: float
    target_value: float
    current_progress: float
    success_threshold: float
    status: str
    created_at: datetime
    assigned_members: List[strawberry.ID]
    required_capabilities: List[str]
    resource_requirements: JSON
    deadline: Optional[datetime] = None


@strawberry.type
class Coalition:
    id: strawberry.ID
    name: str
    description: str
    status: GraphQLCoalitionStatus
    created_at: datetime
    updated_at: datetime
    member_count: int
    total_resources: JSON
    shared_goals: List[CoalitionGoal]

    @strawberry.field
    async def members(self, info: Info) -> List[Agent]:
        """Get all members of this coalition"""
        # Would fetch from database
        return []

    @strawberry.field
    async def leader(self, info: Info) -> Optional[Agent]:
        """Get the coalition leader"""
        # Would fetch leader from members
        return None

    @strawberry.field
    async def performance_metrics(self, info: Info) -> "CoalitionMetrics":
        """Get coalition performance metrics"""
        return CoalitionMetrics(
            coalition_id=self.id,
            goal_completion_rate=0.0,
            resource_efficiency=0.0,
            member_satisfaction=0.0,
            stability_score=0.0,
        )


@strawberry.type
class CoalitionMetrics:
    coalition_id: strawberry.ID
    goal_completion_rate: float
    resource_efficiency: float
    member_satisfaction: float
    stability_score: float


@strawberry.type
class HexCell:
    hex_id: strawberry.ID
    biome: GraphQLBiome
    terrain: GraphQLTerrainType
    elevation: float
    temperature: float
    moisture: float
    resources: JSON
    visibility_range: float
    movement_cost: float

    @strawberry.field
    async def agents_present(self, info: Info) -> List[Agent]:
        """Get agents currently in this hex"""
        # Would query spatial index
        return []

    @strawberry.field
    async def neighboring_cells(self, info: Info) -> List["HexCell"]:
        """Get neighboring hex cells"""
        # Would use H3 library to get neighbors
        return []


@strawberry.type
class WorldState:
    current_time: datetime
    total_agents: int
    total_coalitions: int
    active_agents: int
    resource_totals: JSON
    weather_conditions: JSON

    @strawberry.field
    async def get_hex_cell(self, info: Info, hex_id: str) -> Optional[HexCell]:
        """Get a specific hex cell by ID"""
        # Would fetch from world state
        return None

    @strawberry.field
    async def get_region(self, info: Info, center_hex: str, radius: int = 1) -> List[HexCell]:
        """Get hex cells in a region"""
        # Would use H3 to get region
        return []


@strawberry.type
class SimulationMetrics:
    timestamp: datetime
    fps: float
    agent_count: int
    coalition_count: int
    total_interactions: int
    resource_generation_rate: float
    resource_consumption_rate: float
    average_agent_energy: float
    average_coalition_size: float


# Input Types
@strawberry.input
class CreateAgentInput:
    name: str
    agent_class: GraphQLAgentClass
    starting_position: Optional[Position] = None
    personality_profile: Optional[PersonalityProfile] = None
    initial_energy: float = 100.0


@strawberry.input
class UpdateAgentInput:
    agent_id: strawberry.ID
    name: Optional[str] = None
    status: Optional[GraphQLAgentStatus] = None
    energy: Optional[float] = None
    health: Optional[float] = None


@strawberry.input
class AgentActionInput:
    agent_id: strawberry.ID
    action_type: str
    target_location: Optional[Position] = None
    target_agent_id: Optional[strawberry.ID] = None
    parameters: Optional[JSON] = None


@strawberry.input
class CreateCoalitionInput:
    name: str
    description: str
    founding_members: List[strawberry.ID]
    initial_goals: Optional[List[str]] = None


@strawberry.input
class CoalitionActionInput:
    coalition_id: strawberry.ID
    action_type: str
    parameters: JSON


# Query Root
@strawberry.type
class Query:
    @strawberry.field
    async def agent(self, info: Info, id: strawberry.ID) -> Optional[Agent]:
        """Get a specific agent by ID"""
        # Would fetch from database
        return None

    @strawberry.field
    async def agents(
        self,
        info: Info,
        limit: int = 100,
        offset: int = 0,
        status: Optional[GraphQLAgentStatus] = None,
        agent_class: Optional[GraphQLAgentClass] = None,
    ) -> List[Agent]:
        """List agents with optional filtering"""
        # Would query database with filters
        return []

    @strawberry.field
    async def coalition(self, info: Info, id: strawberry.ID) -> Optional[Coalition]:
        """Get a specific coalition by ID"""
        # Would fetch from database
        return None

    @strawberry.field
    async def coalitions(
        self,
        info: Info,
        limit: int = 100,
        offset: int = 0,
        status: Optional[GraphQLCoalitionStatus] = None,
    ) -> List[Coalition]:
        """List coalitions with optional filtering"""
        # Would query database with filters
        return []

    @strawberry.field
    async def world_state(self, info: Info) -> WorldState:
        """Get current world state"""
        return WorldState(
            current_time=datetime.utcnow(),
            total_agents=0,
            total_coalitions=0,
            active_agents=0,
            resource_totals={},
            weather_conditions={},
        )

    @strawberry.field
    async def simulation_metrics(self, info: Info) -> SimulationMetrics:
        """Get current simulation metrics"""
        return SimulationMetrics(
            timestamp=datetime.utcnow(),
            fps=60.0,
            agent_count=0,
            coalition_count=0,
            total_interactions=0,
            resource_generation_rate=0.0,
            resource_consumption_rate=0.0,
            average_agent_energy=0.0,
            average_coalition_size=0.0,
        )

    @strawberry.field
    async def search_agents(self, info: Info, query: str, limit: int = 50) -> List[Agent]:
        """Search agents by name or attributes"""
        # Would perform search
        return []

    @strawberry.field
    async def nearby_agents(
        self, info: Info, position: Position, radius: float = 10.0
    ) -> List[Agent]:
        """Find agents near a position"""
        # Would query spatial index
        return []


# Mutation Root
@strawberry.type
class Mutation:
    @strawberry.mutation
    async def create_agent(self, info: Info, input: CreateAgentInput) -> Agent:
        """Create a new agent"""
        # Would create in database
        return Agent(
            id=strawberry.ID("1"),
            name=input.name,
            agent_class=input.agent_class,
            status=GraphQLAgentStatus.IDLE,
            position=input.starting_position or Position(x=0, y=0, z=0),
            energy=input.initial_energy,
            health=100.0,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            personality=input.personality_profile
            or PersonalityProfile(
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

    @strawberry.mutation
    async def update_agent(self, info: Info, input: UpdateAgentInput) -> Agent:
        """Update an existing agent"""
        # Would update in database
        raise NotImplementedError("Agent update not implemented")

    @strawberry.mutation
    async def delete_agent(self, info: Info, id: strawberry.ID) -> bool:
        """Delete an agent"""
        # Would delete from database
        return True

    @strawberry.mutation
    async def perform_agent_action(self, info: Info, input: AgentActionInput) -> AgentAction:
        """Have an agent perform an action"""
        # Would process action
        return AgentAction(
            id=strawberry.ID("1"),
            agent_id=input.agent_id,
            action_type=input.action_type,
            target_location=input.target_location,
            target_agent_id=input.target_agent_id,
            success=True,
            energy_cost=5.0,
            timestamp=datetime.utcnow(),
            data={},
        )

    @strawberry.mutation
    async def create_coalition(self, info: Info, input: CreateCoalitionInput) -> Coalition:
        """Create a new coalition"""
        # Would create in database
        return Coalition(
            id=strawberry.ID("1"),
            name=input.name,
            description=input.description,
            status=GraphQLCoalitionStatus.FORMING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            member_count=len(input.founding_members),
            total_resources={},
            shared_goals=[],
        )

    @strawberry.mutation
    async def join_coalition(
        self, info: Info, agent_id: strawberry.ID, coalition_id: strawberry.ID
    ) -> Coalition:
        """Have an agent join a coalition"""
        # Would update membership
        raise NotImplementedError("Coalition join not implemented")

    @strawberry.mutation
    async def leave_coalition(
        self, info: Info, agent_id: strawberry.ID, coalition_id: strawberry.ID
    ) -> bool:
        """Have an agent leave a coalition"""
        # Would update membership
        return True

    @strawberry.mutation
    async def dissolve_coalition(self, info: Info, coalition_id: strawberry.ID) -> bool:
        """Dissolve a coalition"""
        # Would update status and notify members
        return True


# Subscription Root
@strawberry.type
class Subscription:
    @strawberry.subscription
    async def agent_updates(self, info: Info, agent_id: Optional[strawberry.ID] = None) -> Agent:
        """Subscribe to agent updates"""
        # Would stream agent updates
        import asyncio

        while True:
            await asyncio.sleep(1)
            yield Agent(
                id=agent_id or strawberry.ID("1"),
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

    @strawberry.subscription
    async def coalition_updates(
        self, info: Info, coalition_id: Optional[strawberry.ID] = None
    ) -> Coalition:
        """Subscribe to coalition updates"""
        # Would stream coalition updates
        import asyncio

        while True:
            await asyncio.sleep(1)
            yield Coalition(
                id=coalition_id or strawberry.ID("1"),
                name="Test Coalition",
                description="A test coalition",
                status=GraphQLCoalitionStatus.ACTIVE,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                member_count=5,
                total_resources={},
                shared_goals=[],
            )

    @strawberry.subscription
    async def world_events(
        self, info: Info, hex_id: Optional[str] = None, event_types: Optional[List[str]] = None
    ) -> JSON:
        """Subscribe to world events"""
        # Would stream filtered events
        import asyncio

        while True:
            await asyncio.sleep(1)
            yield {
                "event_type": "agent_moved",
                "hex_id": hex_id or "8928308280fffff",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {},
            }

    @strawberry.subscription
    async def simulation_metrics_stream(self, info: Info) -> SimulationMetrics:
        """Stream simulation metrics"""
        # Would stream metrics
        import asyncio

        while True:
            await asyncio.sleep(1)
            yield SimulationMetrics(
                timestamp=datetime.utcnow(),
                fps=60.0,
                agent_count=100,
                coalition_count=10,
                total_interactions=1000,
                resource_generation_rate=10.0,
                resource_consumption_rate=8.0,
                average_agent_energy=85.0,
                average_coalition_size=10.0,
            )


# Create the schema
schema = strawberry.Schema(query=Query, mutation=Mutation, subscription=Subscription)


# Export for FastAPI/other frameworks
__all__ = [
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
