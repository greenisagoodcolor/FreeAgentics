"""GraphQL schema definition for FreeAgentics API."""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional, Type

try:
    import strawberry
    from strawberry.fastapi import GraphQLRouter

    STRAWBERRY_AVAILABLE = True
except ImportError:
    STRAWBERRY_AVAILABLE = False
    strawberry = None

logger = logging.getLogger(__name__)

# Define GraphQL types at module level when strawberry is available
if STRAWBERRY_AVAILABLE:

    @strawberry.type
    class Agent:
        """GraphQL representation of an agent."""

        id: str
        name: str
        status: str
        created_at: datetime
        last_active: Optional[datetime]
        total_steps: int
        capabilities: List[str]
        performance_score: float

    @strawberry.type
    class Coalition:
        """GraphQL representation of a coalition."""

        id: str
        name: str
        status: str
        member_count: int
        leader_id: Optional[str]
        objectives_count: int
        completed_objectives: int
        performance_score: float
        coordination_efficiency: float
        created_at: datetime

    @strawberry.type
    class Objective:
        """GraphQL representation of an objective."""

        id: str
        description: str
        priority: float
        progress: float
        completed: bool
        required_capabilities: List[str]
        deadline: Optional[datetime]

    @strawberry.type
    class WorldState:
        """GraphQL representation of world state."""

        size: int
        step_count: int
        agent_count: int
        active_agents: int

    @strawberry.type
    class SystemMetrics:
        """GraphQL representation of system metrics."""

        total_agents: int
        active_agents: int
        total_coalitions: int
        active_coalitions: int
        pending_objectives: int
        inference_rate: float
        avg_response_time: float

    @strawberry.type
    class InferenceResult:
        """GraphQL representation of inference result."""

        agent_id: str
        input_text: str
        output_text: str
        confidence: float
        processing_time: float
        timestamp: datetime

    # Input types
    @strawberry.input
    class AgentInput:
        """Input type for creating/updating agents."""

        name: str
        template: Optional[str] = None
        parameters: Optional[str] = None  # JSON string

    @strawberry.input
    class CoalitionInput:
        """Input type for creating/updating coalitions."""

        name: str
        max_size: Optional[int] = None
        objective_ids: List[str]

    @strawberry.input
    class ObjectiveInput:
        """Input type for creating objectives."""

        description: str
        priority: float
        required_capabilities: List[str]
        deadline: Optional[datetime] = None

    @strawberry.input
    class InferenceInput:
        """Input type for inference requests."""

        agent_id: Optional[str] = None
        model: Optional[str] = None
        input_text: str
        temperature: Optional[float] = 0.7
        max_tokens: Optional[int] = 512

else:
    # Dummy classes when strawberry is not available
    class Agent:  # type: ignore[no-redef]
        pass

    class Coalition:  # type: ignore[no-redef]
        pass

    class Objective:  # type: ignore[no-redef]
        pass

    class WorldState:  # type: ignore[no-redef]
        pass

    class SystemMetrics:  # type: ignore[no-redef]
        pass

    class InferenceResult:  # type: ignore[no-redef]
        pass

    class AgentInput:  # type: ignore[no-redef]
        pass

    class CoalitionInput:  # type: ignore[no-redef]
        pass

    class ObjectiveInput:  # type: ignore[no-redef]
        pass

    class InferenceInput:  # type: ignore[no-redef]
        pass


def _create_graphql_types():
    """Return GraphQL type definitions."""
    return (
        Agent,
        Coalition,
        Objective,
        WorldState,
        SystemMetrics,
        InferenceResult,
    )


def _create_graphql_input_types():
    """Return GraphQL input type definitions."""
    return AgentInput, CoalitionInput, ObjectiveInput, InferenceInput


def _create_query_resolvers(
    Agent: Type[Any],
    Coalition: Type[Any],
    Objective: Type[Any],
    WorldState: Type[Any],
    SystemMetrics: Type[Any],
):
    """Create GraphQL query resolvers."""

    @strawberry.type
    class Query:
        """GraphQL query root."""

        @strawberry.field
        def agents(
            self, status: Optional[str] = None, limit: int = 100
        ) -> List[Agent]:
            """Get list of agents."""
            # Mock implementation - replace with actual agent manager calls
            mock_agents = [
                Agent(
                    id="agent_1",
                    name="Explorer Agent 1",
                    status="active",
                    created_at=datetime.now(),
                    last_active=datetime.now(),
                    total_steps=150,
                    capabilities=["exploration", "navigation"],
                    performance_score=0.85,
                ),
                Agent(
                    id="agent_2",
                    name="Coordinator Agent",
                    status="active",
                    created_at=datetime.now(),
                    last_active=datetime.now(),
                    total_steps=200,
                    capabilities=["coordination", "planning"],
                    performance_score=0.92,
                ),
            ]

            if status:
                mock_agents = [a for a in mock_agents if a.status == status]

            return mock_agents[:limit]

        @strawberry.field
        def agent(self, id: str) -> Optional[Agent]:
            """Get a specific agent by ID."""
            # Mock implementation
            if id == "agent_1":
                return Agent(
                    id="agent_1",
                    name="Explorer Agent 1",
                    status="active",
                    created_at=datetime.now(),
                    last_active=datetime.now(),
                    total_steps=150,
                    capabilities=["exploration", "navigation"],
                    performance_score=0.85,
                )
            return None

        @strawberry.field
        def coalitions(
            self, status: Optional[str] = None, limit: int = 100
        ) -> List[Coalition]:
            """Get list of coalitions."""
            # Mock implementation
            mock_coalitions = [
                Coalition(
                    id="coalition_1",
                    name="Exploration Coalition",
                    status="active",
                    member_count=3,
                    leader_id="agent_1",
                    objectives_count=2,
                    completed_objectives=1,
                    performance_score=0.78,
                    coordination_efficiency=0.85,
                    created_at=datetime.now(),
                ),
            ]

            if status:
                mock_coalitions = [
                    c for c in mock_coalitions if c.status == status
                ]

            return mock_coalitions[:limit]

        @strawberry.field
        def coalition(self, id: str) -> Optional[Coalition]:
            """Get a specific coalition by ID."""
            # Mock implementation
            if id == "coalition_1":
                return Coalition(
                    id="coalition_1",
                    name="Exploration Coalition",
                    status="active",
                    member_count=3,
                    leader_id="agent_1",
                    objectives_count=2,
                    completed_objectives=1,
                    performance_score=0.78,
                    coordination_efficiency=0.85,
                    created_at=datetime.now(),
                )
            return None

        @strawberry.field
        def objectives(
            self, completed: Optional[bool] = None, limit: int = 100
        ) -> List[Objective]:
            """Get list of objectives."""
            # Mock implementation
            mock_objectives = [
                Objective(
                    id="obj_1",
                    description="Explore sector A",
                    priority=0.8,
                    progress=1.0,
                    completed=True,
                    required_capabilities=["exploration", "navigation"],
                    deadline=None,
                ),
                Objective(
                    id="obj_2",
                    description="Map resource distribution",
                    priority=0.9,
                    progress=0.6,
                    completed=False,
                    required_capabilities=["mapping", "analysis"],
                    deadline=datetime.now(),
                ),
            ]

            if completed is not None:
                mock_objectives = [
                    o for o in mock_objectives if o.completed == completed
                ]

            return mock_objectives[:limit]

        @strawberry.field
        def world_state(self) -> Optional[WorldState]:
            """Get current world state."""
            # Mock implementation
            return WorldState(
                size=20,
                step_count=1547,
                agent_count=5,
                active_agents=3,
            )

        @strawberry.field
        def system_metrics(self) -> SystemMetrics:
            """Get system performance metrics."""
            # Mock implementation
            return SystemMetrics(
                total_agents=5,
                active_agents=3,
                total_coalitions=2,
                active_coalitions=1,
                pending_objectives=3,
                inference_rate=12.5,
                avg_response_time=245.6,
            )

        @strawberry.field
        def search_agents(
            self, query: str, capabilities: Optional[List[str]] = None
        ) -> List[Agent]:
            """Search agents by query and capabilities."""
            # Mock implementation - in reality would search actual agents
            all_agents = self.agents()

            # Simple text search in name
            filtered_agents = [
                a for a in all_agents if query.lower() in a.name.lower()
            ]

            # Filter by capabilities if provided
            if capabilities:
                filtered_agents = [
                    a
                    for a in filtered_agents
                    if any(cap in a.capabilities for cap in capabilities)
                ]

            return filtered_agents

    return Query


def _create_mutation_resolvers(
    Agent: Type[Any],
    Coalition: Type[Any],
    Objective: Type[Any],
    InferenceResult: Type[Any],
    AgentInput: Type[Any],
    CoalitionInput: Type[Any],
    ObjectiveInput: Type[Any],
    InferenceInput: Type[Any],
):
    """Create GraphQL mutation resolvers."""

    @strawberry.type
    class Mutation:
        """GraphQL mutation root."""

        @strawberry.mutation
        def create_agent(self, input: AgentInput) -> Agent:
            """Create a new agent."""
            # Mock implementation - replace with actual agent creation
            return Agent(
                id=f"agent_{datetime.now().timestamp()}",
                name=input.name,
                status="pending",
                created_at=datetime.now(),
                last_active=None,
                total_steps=0,
                capabilities=["default"],
                performance_score=0.0,
            )

        @strawberry.mutation
        def update_agent_status(
            self, agent_id: str, status: str
        ) -> Optional[Agent]:
            """Update agent status."""
            # Mock implementation
            if agent_id == "agent_1":
                return Agent(
                    id="agent_1",
                    name="Explorer Agent 1",
                    status=status,
                    created_at=datetime.now(),
                    last_active=datetime.now(),
                    total_steps=150,
                    capabilities=["exploration", "navigation"],
                    performance_score=0.85,
                )
            return None

        @strawberry.mutation
        def create_coalition(self, input: CoalitionInput) -> Coalition:
            """Create a new coalition."""
            # Mock implementation
            return Coalition(
                id=f"coalition_{datetime.now().timestamp()}",
                name=input.name,
                status="forming",
                member_count=0,
                leader_id=None,
                objectives_count=len(input.objective_ids),
                completed_objectives=0,
                performance_score=0.0,
                coordination_efficiency=0.0,
                created_at=datetime.now(),
            )

        @strawberry.mutation
        def add_coalition_member(
            self, coalition_id: str, agent_id: str
        ) -> Optional[Coalition]:
            """Add an agent to a coalition."""
            # Mock implementation
            if coalition_id == "coalition_1":
                return Coalition(
                    id="coalition_1",
                    name="Exploration Coalition",
                    status="active",
                    member_count=4,  # Incremented
                    leader_id="agent_1",
                    objectives_count=2,
                    completed_objectives=1,
                    performance_score=0.78,
                    coordination_efficiency=0.85,
                    created_at=datetime.now(),
                )
            return None

        @strawberry.mutation
        def create_objective(self, input: ObjectiveInput) -> Objective:
            """Create a new objective."""
            # Mock implementation
            return Objective(
                id=f"obj_{datetime.now().timestamp()}",
                description=input.description,
                priority=input.priority,
                progress=0.0,
                completed=False,
                required_capabilities=input.required_capabilities,
                deadline=input.deadline,
            )

        @strawberry.mutation
        def update_objective_progress(
            self, objective_id: str, progress: float
        ) -> Optional[Objective]:
            """Update objective progress."""
            # Mock implementation
            if objective_id == "obj_2":
                return Objective(
                    id="obj_2",
                    description="Map resource distribution",
                    priority=0.9,
                    progress=progress,
                    completed=progress >= 1.0,
                    required_capabilities=["mapping", "analysis"],
                    deadline=datetime.now(),
                )
            return None

        @strawberry.mutation
        def perform_inference(self, input: InferenceInput) -> InferenceResult:
            """Perform inference using an agent or model."""
            # Mock implementation
            return InferenceResult(
                agent_id=input.agent_id or "default_agent",
                input_text=input.input_text,
                output_text=f"Processed: {input.input_text[:50]}...",
                confidence=0.85,
                processing_time=0.25,
                timestamp=datetime.now(),
            )

    return Mutation


def _create_subscription_resolvers(
    Agent: Type[Any], Coalition: Type[Any], SystemMetrics: Type[Any]
):
    """Create GraphQL subscription resolvers."""

    @strawberry.type
    class Subscription:
        """GraphQL subscription root."""

        @strawberry.subscription
        async def agent_updates(self, agent_id: Optional[str] = None):
            """Subscribe to agent status updates."""
            # Mock implementation - in practice would yield real-time updates
            import asyncio

            while True:
                await asyncio.sleep(5)  # Update every 5 seconds
                yield Agent(
                    id=agent_id or "agent_1",
                    name="Explorer Agent 1",
                    status="active",
                    created_at=datetime.now(),
                    last_active=datetime.now(),
                    total_steps=150,
                    capabilities=["exploration", "navigation"],
                    performance_score=0.85,
                )

        @strawberry.subscription
        async def coalition_updates(self, coalition_id: Optional[str] = None):
            """Subscribe to coalition status updates."""
            import asyncio

            while True:
                await asyncio.sleep(10)  # Update every 10 seconds
                yield Coalition(
                    id=coalition_id or "coalition_1",
                    name="Exploration Coalition",
                    status="active",
                    member_count=3,
                    leader_id="agent_1",
                    objectives_count=2,
                    completed_objectives=1,
                    performance_score=0.78,
                    coordination_efficiency=0.85,
                    created_at=datetime.now(),
                )

        @strawberry.subscription
        async def system_metrics_stream(self):
            """Subscribe to system metrics updates."""
            import asyncio

            while True:
                await asyncio.sleep(3)  # Update every 3 seconds
                yield SystemMetrics(
                    total_agents=5,
                    active_agents=3,
                    total_coalitions=2,
                    active_coalitions=1,
                    pending_objectives=3,
                    inference_rate=12.5,
                    avg_response_time=245.6,
                )

    return Subscription


def _create_graphql_schema_and_router():
    """Create the GraphQL schema and router."""
    if not STRAWBERRY_AVAILABLE:
        return None

    # Create types
    (
        Agent,
        Coalition,
        Objective,
        WorldState,
        SystemMetrics,
        InferenceResult,
    ) = _create_graphql_types()
    (
        AgentInput,
        CoalitionInput,
        ObjectiveInput,
        InferenceInput,
    ) = _create_graphql_input_types()

    # Create resolvers
    Query = _create_query_resolvers(
        Agent, Coalition, Objective, WorldState, SystemMetrics
    )
    Mutation = _create_mutation_resolvers(
        Agent,
        Coalition,
        Objective,
        InferenceResult,
        AgentInput,
        CoalitionInput,
        ObjectiveInput,
        InferenceInput,
    )
    Subscription = _create_subscription_resolvers(
        Agent, Coalition, SystemMetrics
    )

    # Create schema
    schema = strawberry.Schema(
        query=Query,
        mutation=Mutation,
        subscription=Subscription,
    )

    # Create GraphQL router
    return GraphQLRouter(schema)


if STRAWBERRY_AVAILABLE:
    graphql_app = _create_graphql_schema_and_router()

else:
    # Fallback when strawberry-graphql is not available
    logger.warning(
        "strawberry-graphql not available - GraphQL endpoint disabled"
    )

    from fastapi import APIRouter

    graphql_app = APIRouter()

    @graphql_app.get("/graphql")
    async def graphql_unavailable():
        """Fallback endpoint when GraphQL is not available."""
        return {
            "error": "GraphQL not available",
            "message": "Install strawberry-graphql to enable GraphQL support",
            "install_command": "pip install strawberry-graphql[fastapi]",
        }
