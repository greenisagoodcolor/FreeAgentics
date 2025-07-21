"""Builder pattern implementations for test data construction.

Provides flexible, type-safe builders for creating complex test objects
with method chaining and validation.
"""

import random
import string
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, Generic, List, Optional, TypeVar

from .schemas import (
    AgentMetricsSchema,
    AgentParametersSchema,
    AgentSchema,
    AgentStatus,
    BeliefSchema,
    CoalitionObjectiveSchema,
    CoalitionSchema,
    CoalitionStatus,
    KnowledgeEdgeSchema,
    KnowledgeNodeSchema,
    PreferenceSchema,
    PyMDPConfigSchema,
)

T = TypeVar("T")


class BaseBuilder(Generic[T]):
    """Base builder with common functionality."""

    def __init__(self, schema_class: type[T]):
        self.schema_class = schema_class
        self._data: Dict[str, Any] = {}
        self._built = False

    def _check_not_built(self):
        """Ensure builder hasn't been used already."""
        if self._built:
            raise RuntimeError("Builder has already been used. Create a new builder.")

    def reset(self) -> "BaseBuilder[T]":
        """Reset the builder to initial state."""
        self._data = {}
        self._built = False
        return self

    def with_data(self, **kwargs) -> "BaseBuilder[T]":
        """Set multiple fields at once."""
        self._check_not_built()
        self._data.update(kwargs)
        return self

    def build(self) -> T:
        """Build and validate the object."""
        self._check_not_built()
        self._built = True
        return self.schema_class(**self._data)

    @staticmethod
    def random_string(length: int = 10, prefix: str = "") -> str:
        """Generate random string with optional prefix."""
        suffix = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=length)
        )
        return f"{prefix}{suffix}" if prefix else suffix

    @staticmethod
    def random_uuid() -> uuid.UUID:
        """Generate random UUID."""
        return uuid.uuid4()

    @staticmethod
    def random_timestamp(days_range: int = 30) -> datetime:
        """Generate random timestamp within days range."""
        delta = timedelta(days=random.randint(0, days_range))
        return datetime.utcnow() - delta


class AgentBuilder(BaseBuilder[AgentSchema]):
    """Builder for creating Agent test data."""

    def __init__(self):
        super().__init__(AgentSchema)
        # Set sensible defaults
        self._data = {
            "name": self.random_string(6, "Agent_"),
            "template": "grid_world",
            "status": AgentStatus.PENDING,
            "pymdp_config": {},
            "beliefs": {},
            "preferences": {},
            "metrics": {},
            "parameters": {},
        }

    def with_id(self, agent_id: Optional[uuid.UUID] = None) -> "AgentBuilder":
        """Set agent ID."""
        self._check_not_built()
        self._data["id"] = agent_id or self.random_uuid()
        return self

    def with_name(self, name: str) -> "AgentBuilder":
        """Set agent name."""
        self._check_not_built()
        self._data["name"] = name
        return self

    def with_template(self, template: str) -> "AgentBuilder":
        """Set agent template type."""
        self._check_not_built()
        self._data["template"] = template
        return self

    def with_status(self, status: AgentStatus) -> "AgentBuilder":
        """Set agent status."""
        self._check_not_built()
        self._data["status"] = status
        return self

    def active(self) -> "AgentBuilder":
        """Set agent as active."""
        return self.with_status(AgentStatus.ACTIVE)

    def with_gmn_spec(self, gmn_spec: str) -> "AgentBuilder":
        """Set GMN specification."""
        self._check_not_built()
        self._data["gmn_spec"] = gmn_spec
        return self

    def with_pymdp_config(self, **config) -> "AgentBuilder":
        """Set PyMDP configuration."""
        self._check_not_built()
        self._data["pymdp_config"] = PyMDPConfigSchema(**config).dict()
        return self

    def with_grid_world_config(
        self, grid_size: int = 5, num_actions: int = 4
    ) -> "AgentBuilder":
        """Configure for grid world environment."""
        self._check_not_built()
        return self.with_pymdp_config(
            num_states=[grid_size * grid_size],
            num_observations=[grid_size * grid_size],
            num_controls=[num_actions],
            planning_horizon=3,
        ).with_template("grid_world")

    def with_beliefs(self, **beliefs) -> "AgentBuilder":
        """Set agent beliefs."""
        self._check_not_built()
        self._data["beliefs"] = BeliefSchema(**beliefs).dict()
        return self

    def with_uniform_beliefs(self, num_states: int = 5) -> "AgentBuilder":
        """Set uniform belief distribution."""
        self._check_not_built()
        uniform_prob = 1.0 / num_states
        state_beliefs = {f"state_{i}": uniform_prob for i in range(num_states)}
        return self.with_beliefs(state_beliefs=state_beliefs, confidence=0.5)

    def with_preferences(self, **preferences) -> "AgentBuilder":
        """Set agent preferences."""
        self._check_not_built()
        self._data["preferences"] = PreferenceSchema(**preferences).dict()
        return self

    def with_position(
        self, x: float, y: float, z: Optional[float] = None
    ) -> "AgentBuilder":
        """Set agent position."""
        self._check_not_built()
        self._data["position"] = [x, y] if z is None else [x, y, z]
        return self

    def with_random_position(self, bounds: Dict[str, List[float]]) -> "AgentBuilder":
        """Set random position within bounds."""
        self._check_not_built()
        min_bounds = bounds.get("min", [0, 0])
        max_bounds = bounds.get("max", [10, 10])
        position = [
            random.uniform(min_bounds[i], max_bounds[i]) for i in range(len(min_bounds))
        ]
        self._data["position"] = position
        return self

    def with_metrics(self, **metrics) -> "AgentBuilder":
        """Set agent metrics."""
        self._check_not_built()
        self._data["metrics"] = AgentMetricsSchema(**metrics).dict()
        return self

    def with_random_metrics(self) -> "AgentBuilder":
        """Set random realistic metrics."""
        return self.with_metrics(
            total_steps=random.randint(0, 1000),
            successful_actions=random.randint(0, 500),
            failed_actions=random.randint(0, 100),
            average_free_energy=random.uniform(-10, 10),
            average_expected_utility=random.uniform(0, 10),
            inference_time_ms=random.uniform(1, 100),
            memory_usage_mb=random.uniform(10, 100),
        )

    def with_parameters(self, **parameters) -> "AgentBuilder":
        """Set agent parameters."""
        self._check_not_built()
        self._data["parameters"] = AgentParametersSchema(**parameters).dict()
        return self

    def with_exploration_parameters(
        self, exploration_rate: float = 0.2, learning_rate: float = 0.1
    ) -> "AgentBuilder":
        """Set exploration-focused parameters."""
        return self.with_parameters(
            exploration_rate=exploration_rate,
            learning_rate=learning_rate,
            discount_factor=0.95,
        )

    def with_timestamps(
        self,
        created_at: Optional[datetime] = None,
        last_active: Optional[datetime] = None,
    ) -> "AgentBuilder":
        """Set timestamps."""
        self._check_not_built()
        if created_at:
            self._data["created_at"] = created_at
        if last_active:
            self._data["last_active"] = last_active
        self._data["updated_at"] = datetime.utcnow()
        return self

    def with_inference_history(
        self, inference_count: int, total_steps: int
    ) -> "AgentBuilder":
        """Set inference statistics."""
        self._check_not_built()
        self._data["inference_count"] = inference_count
        self._data["total_steps"] = total_steps
        return self

    def as_resource_collector(self) -> "AgentBuilder":
        """Configure as resource collector agent."""
        return (
            self.with_template("resource_collector")
            .with_parameters(
                collection_radius=5.0,
                collection_rate=1.0,
                max_capacity=100,
                movement_speed=2.0,
            )
            .with_random_position({"min": [0, 0], "max": [50, 50]})
        )

    def as_explorer(self) -> "AgentBuilder":
        """Configure as explorer agent."""
        return (
            self.with_template("explorer")
            .with_parameters(
                exploration_rate=0.3,
                vision_range=10.0,
                movement_speed=3.0,
                memory_capacity=2000,
            )
            .with_exploration_parameters()
        )

    def as_coordinator(self) -> "AgentBuilder":
        """Configure as coordinator agent."""
        return self.with_template("coordinator").with_parameters(
            communication_range=20.0,
            max_coalition_size=10,
            coordination_efficiency=0.8,
        )


class CoalitionBuilder(BaseBuilder[CoalitionSchema]):
    """Builder for creating Coalition test data."""

    def __init__(self):
        super().__init__(CoalitionSchema)
        self._data = {
            "name": self.random_string(6, "Coalition_"),
            "status": CoalitionStatus.FORMING,
            "objectives": {},
            "required_capabilities": [],
            "achieved_objectives": [],
        }

    def with_id(self, coalition_id: Optional[uuid.UUID] = None) -> "CoalitionBuilder":
        """Set coalition ID."""
        self._check_not_built()
        self._data["id"] = coalition_id or self.random_uuid()
        return self

    def with_name(self, name: str) -> "CoalitionBuilder":
        """Set coalition name."""
        self._check_not_built()
        self._data["name"] = name
        return self

    def with_description(self, description: str) -> "CoalitionBuilder":
        """Set coalition description."""
        self._check_not_built()
        self._data["description"] = description
        return self

    def with_status(self, status: CoalitionStatus) -> "CoalitionBuilder":
        """Set coalition status."""
        self._check_not_built()
        self._data["status"] = status
        return self

    def active(self) -> "CoalitionBuilder":
        """Set coalition as active."""
        return self.with_status(CoalitionStatus.ACTIVE)

    def with_objective(
        self,
        objective_id: str,
        description: str,
        priority: str = "medium",
        status: str = "pending",
    ) -> "CoalitionBuilder":
        """Add an objective to the coalition."""
        self._check_not_built()
        if "objectives" not in self._data:
            self._data["objectives"] = {}

        objective = CoalitionObjectiveSchema(
            id=objective_id,
            description=description,
            priority=priority,
            status=status,
        )
        # Convert datetime to string for JSON serialization
        obj_dict = objective.dict()
        for key, value in obj_dict.items():
            if isinstance(value, datetime):
                obj_dict[key] = value.isoformat()
        self._data["objectives"][objective_id] = obj_dict
        return self

    def with_resource_optimization_objective(self) -> "CoalitionBuilder":
        """Add resource optimization objective."""
        return self.with_objective(
            "resource_opt",
            "Optimize resource collection and distribution",
            priority="high",
        )

    def with_exploration_objective(self) -> "CoalitionBuilder":
        """Add exploration objective."""
        return self.with_objective(
            "exploration",
            "Explore and map unknown territories",
            priority="medium",
        )

    def with_required_capabilities(self, *capabilities: str) -> "CoalitionBuilder":
        """Set required capabilities."""
        self._check_not_built()
        self._data["required_capabilities"] = list(capabilities)
        return self

    def with_achieved_objectives(self, *objective_ids: str) -> "CoalitionBuilder":
        """Mark objectives as achieved."""
        self._check_not_built()
        self._data["achieved_objectives"] = list(objective_ids)
        # Update objective statuses
        if "objectives" in self._data:
            for obj_id in objective_ids:
                if obj_id in self._data["objectives"]:
                    self._data["objectives"][obj_id]["status"] = "completed"
                    self._data["objectives"][obj_id]["progress"] = 1.0
        return self

    def with_scores(
        self, performance_score: float = 0.7, cohesion_score: float = 0.8
    ) -> "CoalitionBuilder":
        """Set coalition scores."""
        self._check_not_built()
        self._data["performance_score"] = performance_score
        self._data["cohesion_score"] = cohesion_score
        return self

    def with_random_scores(self) -> "CoalitionBuilder":
        """Set random realistic scores."""
        return self.with_scores(
            performance_score=random.uniform(0.5, 0.95),
            cohesion_score=random.uniform(0.6, 1.0),
        )

    def with_timestamps(
        self,
        created_at: Optional[datetime] = None,
        dissolved_at: Optional[datetime] = None,
    ) -> "CoalitionBuilder":
        """Set timestamps."""
        self._check_not_built()
        if created_at:
            self._data["created_at"] = created_at
        if dissolved_at:
            self._data["dissolved_at"] = dissolved_at
            self._data["status"] = CoalitionStatus.DISSOLVED
        self._data["updated_at"] = datetime.utcnow()
        return self

    def as_resource_coalition(self) -> "CoalitionBuilder":
        """Configure as resource-focused coalition."""
        return (
            self.with_description(
                "Coalition focused on efficient resource collection and distribution"
            )
            .with_resource_optimization_objective()
            .with_required_capabilities(
                "resource_collection", "coordination", "planning"
            )
            .with_random_scores()
        )

    def as_exploration_coalition(self) -> "CoalitionBuilder":
        """Configure as exploration-focused coalition."""
        return (
            self.with_description(
                "Coalition dedicated to exploring and mapping new territories"
            )
            .with_exploration_objective()
            .with_required_capabilities("exploration", "communication", "analysis")
            .with_random_scores()
        )


class KnowledgeNodeBuilder(BaseBuilder[KnowledgeNodeSchema]):
    """Builder for creating KnowledgeNode test data."""

    def __init__(self):
        super().__init__(KnowledgeNodeSchema)
        self._data = {
            "type": "concept",
            "label": self.random_string(10, "Node_"),
            "properties": {},
        }

    def with_id(self, node_id: Optional[uuid.UUID] = None) -> "KnowledgeNodeBuilder":
        """Set node ID."""
        self._check_not_built()
        self._data["id"] = node_id or self.random_uuid()
        return self

    def with_type(self, node_type: str) -> "KnowledgeNodeBuilder":
        """Set node type."""
        self._check_not_built()
        self._data["type"] = node_type
        return self

    def with_label(self, label: str) -> "KnowledgeNodeBuilder":
        """Set node label."""
        self._check_not_built()
        self._data["label"] = label
        return self

    def with_properties(self, **properties) -> "KnowledgeNodeBuilder":
        """Set node properties."""
        self._check_not_built()
        self._data["properties"] = properties
        return self

    def with_version(
        self, version: int, is_current: bool = True
    ) -> "KnowledgeNodeBuilder":
        """Set version information."""
        self._check_not_built()
        self._data["version"] = version
        self._data["is_current"] = is_current
        return self

    def with_confidence(self, confidence: float) -> "KnowledgeNodeBuilder":
        """Set confidence level."""
        self._check_not_built()
        self._data["confidence"] = confidence
        return self

    def with_source(self, source: str) -> "KnowledgeNodeBuilder":
        """Set information source."""
        self._check_not_built()
        self._data["source"] = source
        return self

    def with_creator_agent(self, agent_id: uuid.UUID) -> "KnowledgeNodeBuilder":
        """Set creator agent."""
        self._check_not_built()
        self._data["creator_agent_id"] = agent_id
        return self

    def with_embedding(
        self, embedding: Optional[List[float]] = None, dim: int = 128
    ) -> "KnowledgeNodeBuilder":
        """Set embedding vector."""
        self._check_not_built()
        if embedding is None:
            # Generate random normalized embedding
            embedding = [random.gauss(0, 1) for _ in range(dim)]
            # Normalize
            norm = sum(x**2 for x in embedding) ** 0.5
            embedding = [x / norm for x in embedding]
        self._data["embedding"] = embedding
        return self

    def with_timestamps(
        self,
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
    ) -> "KnowledgeNodeBuilder":
        """Set timestamps."""
        self._check_not_built()
        if created_at:
            self._data["created_at"] = created_at
        if updated_at:
            self._data["updated_at"] = updated_at
        else:
            self._data["updated_at"] = datetime.utcnow()
        return self

    def as_concept(self, concept: str) -> "KnowledgeNodeBuilder":
        """Configure as concept node."""
        return (
            self.with_type("concept")
            .with_label(f"Concept: {concept}")
            .with_properties(category="abstract", domain="general", importance="medium")
        )

    def as_entity(self, entity_name: str, entity_type: str) -> "KnowledgeNodeBuilder":
        """Configure as entity node."""
        return (
            self.with_type("entity")
            .with_label(entity_name)
            .with_properties(entity_type=entity_type, attributes={}, relationships=[])
        )

    def as_observation(
        self,
        observation: str,
        observer_id: Optional[uuid.UUID] = None,
        timestamp: Optional[datetime] = None,
    ) -> "KnowledgeNodeBuilder":
        """Configure as observation node."""
        builder = (
            self.with_type("observation")
            .with_label(f"Observation: {observation}")
            .with_properties(
                observation_type="direct",
                timestamp=(timestamp or datetime.utcnow()).isoformat(),
            )
        )

        if observer_id:
            builder = builder.with_creator_agent(observer_id)

        return builder


class KnowledgeEdgeBuilder(BaseBuilder[KnowledgeEdgeSchema]):
    """Builder for creating KnowledgeEdge test data."""

    def __init__(self):
        super().__init__(KnowledgeEdgeSchema)
        self._data = {"type": "relates_to", "properties": {}}

    def with_id(self, edge_id: Optional[uuid.UUID] = None) -> "KnowledgeEdgeBuilder":
        """Set edge ID."""
        self._check_not_built()
        self._data["id"] = edge_id or self.random_uuid()
        return self

    def with_nodes(
        self, source_id: uuid.UUID, target_id: uuid.UUID
    ) -> "KnowledgeEdgeBuilder":
        """Set source and target nodes."""
        self._check_not_built()
        self._data["source_id"] = source_id
        self._data["target_id"] = target_id
        return self

    def with_type(self, edge_type: str) -> "KnowledgeEdgeBuilder":
        """Set edge type."""
        self._check_not_built()
        self._data["type"] = edge_type
        return self

    def with_properties(self, **properties) -> "KnowledgeEdgeBuilder":
        """Set edge properties."""
        self._check_not_built()
        self._data["properties"] = properties
        return self

    def with_confidence(self, confidence: float) -> "KnowledgeEdgeBuilder":
        """Set confidence level."""
        self._check_not_built()
        self._data["confidence"] = confidence
        return self

    def with_timestamp(
        self, created_at: Optional[datetime] = None
    ) -> "KnowledgeEdgeBuilder":
        """Set creation timestamp."""
        self._check_not_built()
        self._data["created_at"] = created_at or datetime.utcnow()
        return self

    def as_causal(self, strength: float = 0.8) -> "KnowledgeEdgeBuilder":
        """Configure as causal relationship."""
        return self.with_type("causes").with_properties(
            causal_strength=strength, mechanism="direct"
        )

    def as_support(self, evidence_count: int = 1) -> "KnowledgeEdgeBuilder":
        """Configure as supporting relationship."""
        return self.with_type("supports").with_properties(
            evidence_count=evidence_count, support_type="empirical"
        )

    def as_contradiction(self) -> "KnowledgeEdgeBuilder":
        """Configure as contradicting relationship."""
        return self.with_type("contradicts").with_properties(
            conflict_type="logical", resolution_needed=True
        )
