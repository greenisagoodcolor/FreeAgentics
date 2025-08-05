"""Knowledge Graph Schema and Ontology for Agent Conversations (Task 34.1).

This module defines the comprehensive schema and ontology for agent conversation
knowledge graphs, including entity types, relation types, temporal metadata,
provenance tracking, and conflict resolution strategies.

Follows SOLID principles with proper abstractions and dependency injection.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Entity types for agent conversation knowledge graphs."""

    AGENT = "agent"
    CONCEPT = "concept"
    GOAL = "goal"
    CONSTRAINT = "constraint"
    BELIEF = "belief"
    TASK = "task"
    CONTEXT = "context"
    OBSERVATION = "observation"
    DECISION = "decision"
    OUTCOME = "outcome"

    # PyMDP-specific entity types for inference integration
    BELIEF_STATE = "belief_state"
    POLICY_SEQUENCE = "policy_sequence"
    INFERENCE_STEP = "inference_step"
    STATE_TRANSITION = "state_transition"
    FREE_ENERGY_LANDSCAPE = "free_energy_landscape"


class RelationType(Enum):
    """Relation types for agent conversation knowledge graphs."""

    REQUIRES = "requires"
    CONFLICTS_WITH = "conflicts_with"
    ENABLES = "enables"
    RELATES_TO = "relates_to"
    DEPENDS_ON = "depends_on"
    INFLUENCES = "influences"
    MENTIONS = "mentions"
    LEADS_TO = "leads_to"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"

    # PyMDP-specific relation types for inference integration
    BELIEF_UPDATE = "belief_update"
    POLICY_SELECTION = "policy_selection"
    STATE_PREDICTION = "state_prediction"
    ACTION_SAMPLING = "action_sampling"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    CONFIDENCE_WEIGHTED = "confidence_weighted"


@dataclass
class PropertySchema:
    """Schema definition for entity or relation properties."""

    required_properties: Set[str] = field(default_factory=set)
    optional_properties: Set[str] = field(default_factory=set)
    property_types: Dict[str, type] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalMetadata:
    """Temporal metadata for knowledge graph elements."""

    created_at: datetime
    last_updated: datetime
    conversation_id: str
    message_id: Optional[str] = None
    sequence_number: Optional[int] = None

    def is_valid(self) -> bool:
        """Validate temporal metadata consistency."""
        return (
            self.created_at <= self.last_updated
            and bool(self.conversation_id)
            and isinstance(self.created_at, datetime)
            and isinstance(self.last_updated, datetime)
        )


@dataclass
class Provenance:
    """Provenance information for knowledge graph elements."""

    source_type: str
    source_id: str
    extraction_method: str
    confidence_score: float
    agent_id: Optional[str] = None
    human_verified: bool = False
    extraction_timestamp: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Initialize post-creation fields."""
        if self.extraction_timestamp is None:
            self.extraction_timestamp = datetime.now()

    def is_valid(self) -> bool:
        """Validate provenance information."""
        return (
            0.0 <= self.confidence_score <= 1.0
            and bool(self.source_type)
            and bool(self.source_id)
            and bool(self.extraction_method)
        )


@dataclass
class ConversationEntity:
    """Entity in a conversation knowledge graph."""

    entity_type: EntityType
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    entity_id: str = field(default_factory=lambda: str(uuid4()))
    temporal_metadata: Optional[TemporalMetadata] = None
    provenance: Optional[Provenance] = None

    def is_valid(self) -> bool:
        """Validate entity structure."""
        return (
            bool(self.label)
            and isinstance(self.entity_type, EntityType)
            and (self.temporal_metadata is None or self.temporal_metadata.is_valid())
            and (self.provenance is None or self.provenance.is_valid())
        )


@dataclass
class ConversationRelation:
    """Relation in a conversation knowledge graph."""

    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    relation_id: str = field(default_factory=lambda: str(uuid4()))
    temporal_metadata: Optional[TemporalMetadata] = None
    provenance: Optional[Provenance] = None

    def is_valid(self) -> bool:
        """Validate relation structure."""
        return (
            bool(self.source_entity_id)
            and bool(self.target_entity_id)
            and isinstance(self.relation_type, RelationType)
            and (self.temporal_metadata is None or self.temporal_metadata.is_valid())
            and (self.provenance is None or self.provenance.is_valid())
        )


@dataclass
class ValidationResult:
    """Result of schema validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationOntology:
    """Ontology definition for agent conversation knowledge graphs."""

    def __init__(self) -> None:
        """Initialize the conversation ontology."""
        self.entity_types = set(EntityType)
        self.relation_types = set(RelationType)
        self._entity_schemas = self._initialize_entity_schemas()
        self._relation_schemas = self._initialize_relation_schemas()

        logger.info(
            "Initialized ConversationOntology with %d entity types and %d relation types",
            len(self.entity_types),
            len(self.relation_types),
        )

    def _initialize_entity_schemas(self) -> Dict[EntityType, PropertySchema]:
        """Initialize property schemas for entity types."""
        schemas = {}

        # Agent entity schema
        schemas[EntityType.AGENT] = PropertySchema(
            required_properties={"agent_id", "name"},
            optional_properties={"capabilities", "conversation_history", "status", "role"},
            property_types={
                "agent_id": str,
                "name": str,
                "capabilities": list,
                "conversation_history": list,
                "status": str,
                "role": str,
            },
        )

        # Goal entity schema
        schemas[EntityType.GOAL] = PropertySchema(
            required_properties={"goal_description"},
            optional_properties={"priority", "completion_status", "deadline", "success_criteria"},
            property_types={
                "goal_description": str,
                "priority": str,
                "completion_status": str,
                "deadline": str,
                "success_criteria": list,
            },
        )

        # Belief entity schema
        schemas[EntityType.BELIEF] = PropertySchema(
            required_properties={"belief_content"},
            optional_properties={"certainty", "evidence", "belief_type", "revision_history"},
            property_types={
                "belief_content": str,
                "certainty": float,
                "evidence": list,
                "belief_type": str,
                "revision_history": list,
            },
        )

        # Task entity schema
        schemas[EntityType.TASK] = PropertySchema(
            required_properties={"task_description"},
            optional_properties={"status", "assigned_agent", "dependencies", "estimated_effort"},
            property_types={
                "task_description": str,
                "status": str,
                "assigned_agent": str,
                "dependencies": list,
                "estimated_effort": str,
            },
        )

        # Context entity schema
        schemas[EntityType.CONTEXT] = PropertySchema(
            required_properties={"context_type"},
            optional_properties={"context_data", "scope", "validity_period", "relevance_score"},
            property_types={
                "context_type": str,
                "context_data": dict,
                "scope": str,
                "validity_period": str,
                "relevance_score": float,
            },
        )

        # Add minimal schemas for other entity types
        for entity_type in EntityType:
            if entity_type not in schemas:
                schemas[entity_type] = PropertySchema(
                    required_properties={f"{entity_type.value}_description"},
                    optional_properties={"metadata", "properties"},
                    property_types={
                        f"{entity_type.value}_description": str,
                        "metadata": dict,
                        "properties": dict,
                    },
                )

        return schemas

    def _initialize_relation_schemas(self) -> Dict[RelationType, PropertySchema]:
        """Initialize property schemas for relation types."""
        schemas = {}

        # Requires relation schema
        schemas[RelationType.REQUIRES] = PropertySchema(
            optional_properties={"strength", "evidence", "necessity_level"},
            property_types={"strength": float, "evidence": str, "necessity_level": str},
        )

        # Conflicts with relation schema
        schemas[RelationType.CONFLICTS_WITH] = PropertySchema(
            optional_properties={"conflict_type", "resolution_strategy", "severity"},
            property_types={"conflict_type": str, "resolution_strategy": str, "severity": str},
        )

        # Add minimal schemas for other relation types
        for relation_type in RelationType:
            if relation_type not in schemas:
                schemas[relation_type] = PropertySchema(
                    optional_properties={"strength", "evidence", "metadata"},
                    property_types={"strength": float, "evidence": str, "metadata": dict},
                )

        return schemas

    def get_entity_schema(self, entity_type: EntityType) -> PropertySchema:
        """Get property schema for an entity type."""
        return self._entity_schemas.get(entity_type, PropertySchema())

    def get_relation_schema(self, relation_type: RelationType) -> PropertySchema:
        """Get property schema for a relation type."""
        return self._relation_schemas.get(relation_type, PropertySchema())

    def validate_entity(self, entity: ConversationEntity) -> bool:
        """Validate entity against its schema."""
        if not entity.is_valid():
            return False

        schema = self.get_entity_schema(entity.entity_type)

        # Check required properties
        for required_prop in schema.required_properties:
            if required_prop not in entity.properties:
                logger.warning(
                    "Entity %s missing required property: %s", entity.entity_id, required_prop
                )
                return False

        # Validate property types
        for prop_name, prop_value in entity.properties.items():
            if prop_name in schema.property_types:
                expected_type = schema.property_types[prop_name]
                if not isinstance(prop_value, expected_type):
                    logger.warning(
                        "Entity %s property %s has wrong type: expected %s, got %s",
                        entity.entity_id,
                        prop_name,
                        expected_type,
                        type(prop_value),
                    )
                    return False

        return True

    def validate_relation(self, relation: ConversationRelation) -> bool:
        """Validate relation against its schema."""
        if not relation.is_valid():
            return False

        schema = self.get_relation_schema(relation.relation_type)

        # Check required properties (most relations have no required properties)
        for required_prop in schema.required_properties:
            if required_prop not in relation.properties:
                logger.warning(
                    "Relation %s missing required property: %s", relation.relation_id, required_prop
                )
                return False

        # Validate property types
        for prop_name, prop_value in relation.properties.items():
            if prop_name in schema.property_types:
                expected_type = schema.property_types[prop_name]
                if not isinstance(prop_value, expected_type):
                    logger.warning(
                        "Relation %s property %s has wrong type: expected %s, got %s",
                        relation.relation_id,
                        prop_name,
                        expected_type,
                        type(prop_value),
                    )
                    return False

        return True


class SchemaValidator:
    """Comprehensive schema validation system."""

    def __init__(self, ontology: ConversationOntology) -> None:
        """Initialize schema validator with ontology."""
        self.ontology = ontology
        self.validation_rules = self._initialize_validation_rules()
        logger.info("Initialized SchemaValidator")

    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize validation rules."""
        return {
            "require_temporal_metadata": False,
            "require_provenance": False,
            "min_confidence": 0.0,
            "max_property_nesting_depth": 5,
            "validate_property_types": True,
        }

    def validate_entity(self, entity: ConversationEntity) -> ValidationResult:
        """Perform comprehensive entity validation."""
        errors = []
        warnings = []

        # Basic structure validation
        if not entity.is_valid():
            errors.append("Entity has invalid basic structure")

        # Detailed ontology validation with specific error messages
        schema = self.ontology.get_entity_schema(entity.entity_type)

        # Check required properties
        for required_prop in schema.required_properties:
            if required_prop not in entity.properties:
                errors.append(f"Missing required property: {required_prop}")

        # Validate property types
        for prop_name, prop_value in entity.properties.items():
            if prop_name in schema.property_types:
                expected_type = schema.property_types[prop_name]
                if not isinstance(prop_value, expected_type):
                    errors.append(
                        f"Property {prop_name} has wrong type: expected {expected_type.__name__}, got {type(prop_value).__name__}"
                    )

        # Check temporal metadata if required
        if (
            self.validation_rules.get("require_temporal_metadata", False)
            and entity.temporal_metadata is None
        ):
            errors.append("Temporal metadata is required but missing")

        # Check provenance if required
        if self.validation_rules.get("require_provenance", False) and entity.provenance is None:
            errors.append("Provenance is required but missing")

        # Validate confidence if provenance exists
        if entity.provenance and entity.provenance.confidence_score < self.validation_rules.get(
            "min_confidence", 0.0
        ):
            warnings.append(
                f"Confidence score {entity.provenance.confidence_score} below minimum threshold"
            )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={"entity_type": entity.entity_type.value, "entity_id": entity.entity_id},
        )

    def validate_relation(self, relation: ConversationRelation) -> ValidationResult:
        """Perform comprehensive relation validation."""
        errors: List[str] = []
        warnings: List[str] = []

        # Basic structure validation
        if not relation.is_valid():
            errors.append("Relation has invalid basic structure")

        # Ontology validation
        if not self.ontology.validate_relation(relation):
            errors.append("Relation fails ontology validation")

        # Check temporal metadata if required
        if (
            self.validation_rules.get("require_temporal_metadata", False)
            and relation.temporal_metadata is None
        ):
            errors.append("Temporal metadata is required but missing")

        # Check provenance if required
        if self.validation_rules.get("require_provenance", False) and relation.provenance is None:
            errors.append("Provenance is required but missing")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={
                "relation_type": relation.relation_type.value,
                "relation_id": relation.relation_id,
            },
        )


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving conflicts in knowledge graph updates."""

    LATEST_TIMESTAMP = "latest_timestamp"
    HIGHEST_CONFIDENCE = "highest_confidence"
    MOST_RELIABLE_SOURCE = "most_reliable_source"
    HUMAN_VERIFICATION = "human_verification"
    KEEP_BOTH = "keep_both"

    def resolve(self, conflicting_entities: List[ConversationEntity]) -> ConversationEntity:
        """Apply the conflict resolution strategy."""
        if not conflicting_entities:
            raise ValueError("No entities to resolve")

        if len(conflicting_entities) == 1:
            return conflicting_entities[0]

        if self == ConflictResolutionStrategy.LATEST_TIMESTAMP:
            return max(
                conflicting_entities,
                key=lambda e: e.temporal_metadata.last_updated
                if e.temporal_metadata
                else datetime.min,
            )

        elif self == ConflictResolutionStrategy.HIGHEST_CONFIDENCE:
            return max(
                conflicting_entities,
                key=lambda e: e.provenance.confidence_score if e.provenance else 0.0,
            )

        elif self == ConflictResolutionStrategy.MOST_RELIABLE_SOURCE:
            # Priority order for source reliability
            source_priority = {
                "human_input": 1.0,
                "verified_sensor": 0.9,
                "sensor": 0.8,
                "nlp_pipeline": 0.7,
                "inference": 0.6,
                "default": 0.5,
            }

            def get_source_priority(entity: ConversationEntity) -> float:
                if not entity.provenance:
                    return 0.0
                return source_priority.get(entity.provenance.extraction_method, 0.5)

            return max(conflicting_entities, key=get_source_priority)

        else:
            # Default to highest confidence
            return max(
                conflicting_entities,
                key=lambda e: e.provenance.confidence_score if e.provenance else 0.0,
            )


@dataclass
class KnowledgeGraphSchema:
    """Complete knowledge graph schema definition."""

    version: str
    ontology: ConversationOntology
    metadata_schema: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    conflict_resolution_strategy: ConflictResolutionStrategy = (
        ConflictResolutionStrategy.HIGHEST_CONFIDENCE
    )

    def __post_init__(self) -> None:
        """Initialize schema components."""
        self.validator = SchemaValidator(self.ontology)
        logger.info("Initialized KnowledgeGraphSchema version %s", self.version)

    def is_compatible_with(self, other: "KnowledgeGraphSchema") -> bool:
        """Check compatibility with another schema version."""
        # Simple compatibility check based on version numbers
        try:
            current_version = tuple(map(int, self.version.split(".")))
            other_version = tuple(map(int, other.version.split(".")))

            # Compatible if major version is the same
            return current_version[0] == other_version[0]
        except (ValueError, IndexError):
            # If version parsing fails, assume incompatible
            return False

    def validate_graph_structure(
        self, entities: List[ConversationEntity], relations: List[ConversationRelation]
    ) -> ValidationResult:
        """Validate overall graph structure."""
        errors = []
        warnings = []

        # Validate all entities
        entity_ids = set()
        for entity in entities:
            result = self.validator.validate_entity(entity)
            if not result.is_valid:
                errors.extend([f"Entity {entity.entity_id}: {error}" for error in result.errors])
            warnings.extend(
                [f"Entity {entity.entity_id}: {warning}" for warning in result.warnings]
            )
            entity_ids.add(entity.entity_id)

        # Validate all relations
        for relation in relations:
            result = self.validator.validate_relation(relation)
            if not result.is_valid:
                errors.extend(
                    [f"Relation {relation.relation_id}: {error}" for error in result.errors]
                )
            warnings.extend(
                [f"Relation {relation.relation_id}: {warning}" for warning in result.warnings]
            )

            # Check that referenced entities exist
            if relation.source_entity_id not in entity_ids:
                errors.append(
                    f"Relation {relation.relation_id}: source entity {relation.source_entity_id} not found"
                )
            if relation.target_entity_id not in entity_ids:
                errors.append(
                    f"Relation {relation.relation_id}: target entity {relation.target_entity_id} not found"
                )

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={
                "entity_count": len(entities),
                "relation_count": len(relations),
                "schema_version": self.version,
            },
        )


class SchemaEvolutionManager:
    """Manager for schema evolution and migration."""

    def __init__(self, current_version: str = "1.0") -> None:
        """Initialize schema evolution manager."""
        self.current_version = current_version
        self.migration_strategies: Dict[Tuple[str, str], Any] = {}
        logger.info("Initialized SchemaEvolutionManager with version %s", current_version)

    def register_migration(self, from_version: str, to_version: str, migration_func: Any) -> None:
        """Register a migration strategy between versions."""
        migration_key = (from_version, to_version)
        self.migration_strategies[migration_key] = migration_func
        logger.info("Registered migration from %s to %s", from_version, to_version)

    def check_compatibility(
        self, schema1: KnowledgeGraphSchema, schema2: KnowledgeGraphSchema
    ) -> bool:
        """Check compatibility between two schema versions."""
        return schema1.is_compatible_with(schema2)

    def validate_evolution(
        self, old_schema: KnowledgeGraphSchema, new_schema: KnowledgeGraphSchema
    ) -> ValidationResult:
        """Validate that schema evolution maintains data integrity."""
        errors = []
        warnings = []

        # Check version progression
        if not self.check_compatibility(old_schema, new_schema):
            warnings.append("Schema versions may not be compatible")

        # Check that core entity types are preserved
        old_entity_types = old_schema.ontology.entity_types
        new_entity_types = new_schema.ontology.entity_types

        removed_types = old_entity_types - new_entity_types
        if removed_types:
            errors.append(f"Entity types removed in evolution: {removed_types}")

        # Check that core relation types are preserved
        old_relation_types = old_schema.ontology.relation_types
        new_relation_types = new_schema.ontology.relation_types

        removed_relations = old_relation_types - new_relation_types
        if removed_relations:
            errors.append(f"Relation types removed in evolution: {removed_relations}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata={
                "old_version": old_schema.version,
                "new_version": new_schema.version,
                "migration_available": (old_schema.version, new_schema.version)
                in self.migration_strategies,
            },
        )
