"""
Tests for Knowledge Graph Schema and Ontology Design (Task 34.1).

This test suite follows strict TDD principles and validates the schema design
for agent conversation knowledge graphs with proper ontology definitions.
"""

from datetime import datetime

# These imports will fail initially - that's the point of TDD
from knowledge_graph.schema import (
    ConflictResolutionStrategy,
    ConversationEntity,
    ConversationOntology,
    ConversationRelation,
    EntityType,
    KnowledgeGraphSchema,
    Provenance,
    RelationType,
    SchemaEvolutionManager,
    SchemaValidator,
    TemporalMetadata,
)


class TestConversationOntology:
    """Test the conversation-specific ontology definitions."""

    def test_entity_types_definition(self) -> None:
        """Test that all required entity types are defined for agent conversations."""
        ontology = ConversationOntology()

        # Core conversation entities
        assert EntityType.AGENT in ontology.entity_types
        assert EntityType.CONCEPT in ontology.entity_types
        assert EntityType.GOAL in ontology.entity_types
        assert EntityType.CONSTRAINT in ontology.entity_types
        assert EntityType.BELIEF in ontology.entity_types
        assert EntityType.TASK in ontology.entity_types
        assert EntityType.CONTEXT in ontology.entity_types
        assert EntityType.OBSERVATION in ontology.entity_types
        assert EntityType.DECISION in ontology.entity_types
        assert EntityType.OUTCOME in ontology.entity_types

    def test_relation_types_definition(self) -> None:
        """Test that all required relation types are defined."""
        ontology = ConversationOntology()

        # Core conversation relations
        assert RelationType.REQUIRES in ontology.relation_types
        assert RelationType.CONFLICTS_WITH in ontology.relation_types
        assert RelationType.ENABLES in ontology.relation_types
        assert RelationType.RELATES_TO in ontology.relation_types
        assert RelationType.DEPENDS_ON in ontology.relation_types
        assert RelationType.INFLUENCES in ontology.relation_types
        assert RelationType.MENTIONS in ontology.relation_types
        assert RelationType.LEADS_TO in ontology.relation_types
        assert RelationType.CONTRADICTS in ontology.relation_types
        assert RelationType.SUPPORTS in ontology.relation_types

    def test_entity_property_schemas(self) -> None:
        """Test that entity types have proper property schemas."""
        ontology = ConversationOntology()

        agent_schema = ontology.get_entity_schema(EntityType.AGENT)
        assert "agent_id" in agent_schema.required_properties
        assert "name" in agent_schema.required_properties
        assert "capabilities" in agent_schema.optional_properties
        assert "conversation_history" in agent_schema.optional_properties

        goal_schema = ontology.get_entity_schema(EntityType.GOAL)
        assert "goal_description" in goal_schema.required_properties
        assert "priority" in goal_schema.optional_properties
        assert "completion_status" in goal_schema.optional_properties

    def test_relation_property_schemas(self) -> None:
        """Test that relation types have proper property schemas."""
        ontology = ConversationOntology()

        requires_schema = ontology.get_relation_schema(RelationType.REQUIRES)
        assert "strength" in requires_schema.optional_properties
        assert "evidence" in requires_schema.optional_properties

        conflicts_schema = ontology.get_relation_schema(RelationType.CONFLICTS_WITH)
        assert "conflict_type" in conflicts_schema.optional_properties
        assert "resolution_strategy" in conflicts_schema.optional_properties


class TestTemporalMetadata:
    """Test temporal aspects of the knowledge graph schema."""

    def test_temporal_metadata_creation(self) -> None:
        """Test creating temporal metadata for knowledge graph elements."""
        metadata = TemporalMetadata(
            created_at=datetime.now(),
            last_updated=datetime.now(),
            conversation_id="conv_123",
            message_id="msg_456",
        )

        assert metadata.created_at is not None
        assert metadata.last_updated is not None
        assert metadata.conversation_id == "conv_123"
        assert metadata.message_id == "msg_456"

    def test_temporal_metadata_validation(self) -> None:
        """Test validation of temporal metadata."""
        # Valid metadata
        valid_metadata = TemporalMetadata(
            created_at=datetime.now(), last_updated=datetime.now(), conversation_id="conv_123"
        )
        assert valid_metadata.is_valid() is True

        # Invalid metadata (created_at after last_updated)
        future_time = datetime.now()
        past_time = datetime(2020, 1, 1)

        invalid_metadata = TemporalMetadata(
            created_at=future_time, last_updated=past_time, conversation_id="conv_123"
        )
        assert invalid_metadata.is_valid() is False


class TestProvenance:
    """Test provenance tracking for knowledge graph elements."""

    def test_provenance_creation(self) -> None:
        """Test creating provenance information."""
        provenance = Provenance(
            source_type="conversation_message",
            source_id="msg_789",
            extraction_method="nlp_pipeline",
            confidence_score=0.85,
            agent_id="agent_1",
        )

        assert provenance.source_type == "conversation_message"
        assert provenance.source_id == "msg_789"
        assert provenance.extraction_method == "nlp_pipeline"
        assert provenance.confidence_score == 0.85
        assert provenance.agent_id == "agent_1"

    def test_provenance_validation(self) -> None:
        """Test provenance validation."""
        # Valid provenance
        valid_provenance = Provenance(
            source_type="conversation_message",
            source_id="msg_123",
            extraction_method="manual",
            confidence_score=0.9,
        )
        assert valid_provenance.is_valid() is True

        # Invalid confidence score
        invalid_provenance = Provenance(
            source_type="conversation_message",
            source_id="msg_123",
            extraction_method="manual",
            confidence_score=1.5,  # > 1.0
        )
        assert invalid_provenance.is_valid() is False


class TestConversationEntity:
    """Test conversation-specific entity implementation."""

    def test_conversation_entity_creation(self) -> None:
        """Test creating conversation entities with full metadata."""
        entity = ConversationEntity(
            entity_type=EntityType.GOAL,
            label="Complete task X",
            properties={"priority": "high", "deadline": "2024-01-01"},
            temporal_metadata=TemporalMetadata(
                created_at=datetime.now(), last_updated=datetime.now(), conversation_id="conv_123"
            ),
            provenance=Provenance(
                source_type="conversation_message",
                source_id="msg_456",
                extraction_method="nlp_pipeline",
                confidence_score=0.9,
            ),
        )

        assert entity.entity_type == EntityType.GOAL
        assert entity.label == "Complete task X"
        assert entity.properties["priority"] == "high"
        assert entity.temporal_metadata is not None
        assert entity.temporal_metadata.conversation_id == "conv_123"
        assert entity.provenance is not None
        assert entity.provenance.confidence_score == 0.9

    def test_conversation_entity_validation(self) -> None:
        """Test entity validation against ontology schema."""
        ontology = ConversationOntology()

        # Valid entity
        valid_entity = ConversationEntity(
            entity_type=EntityType.AGENT,
            label="AI Assistant",
            properties={
                "agent_id": "agent_1",
                "name": "Claude",
                "capabilities": ["reasoning", "coding"],
            },
        )

        assert ontology.validate_entity(valid_entity) is True

        # Invalid entity (missing required properties)
        invalid_entity = ConversationEntity(
            entity_type=EntityType.AGENT,
            label="AI Assistant",
            properties={"name": "Claude"},  # Missing agent_id
        )

        assert ontology.validate_entity(invalid_entity) is False


class TestConversationRelation:
    """Test conversation-specific relation implementation."""

    def test_conversation_relation_creation(self) -> None:
        """Test creating conversation relations with metadata."""
        relation = ConversationRelation(
            source_entity_id="entity_1",
            target_entity_id="entity_2",
            relation_type=RelationType.REQUIRES,
            properties={"strength": 0.8, "evidence": "mentioned in message"},
            temporal_metadata=TemporalMetadata(
                created_at=datetime.now(), last_updated=datetime.now(), conversation_id="conv_123"
            ),
            provenance=Provenance(
                source_type="conversation_message",
                source_id="msg_789",
                extraction_method="relation_extractor",
                confidence_score=0.75,
            ),
        )

        assert relation.source_entity_id == "entity_1"
        assert relation.target_entity_id == "entity_2"
        assert relation.relation_type == RelationType.REQUIRES
        assert relation.properties["strength"] == 0.8

    def test_conversation_relation_validation(self) -> None:
        """Test relation validation against ontology schema."""
        ontology = ConversationOntology()

        # Valid relation
        valid_relation = ConversationRelation(
            source_entity_id="entity_1",
            target_entity_id="entity_2",
            relation_type=RelationType.CONFLICTS_WITH,
            properties={"conflict_type": "resource_competition"},
        )

        assert ontology.validate_relation(valid_relation) is True


class TestSchemaValidator:
    """Test the schema validation system."""

    def test_schema_validator_creation(self) -> None:
        """Test creating a schema validator."""
        ontology = ConversationOntology()
        validator = SchemaValidator(ontology)

        assert validator.ontology == ontology
        assert validator.validation_rules is not None

    def test_entity_validation(self) -> None:
        """Test comprehensive entity validation."""
        ontology = ConversationOntology()
        validator = SchemaValidator(ontology)

        # Create test entity
        entity = ConversationEntity(
            entity_type=EntityType.GOAL,
            label="Test Goal",
            properties={"goal_description": "Complete project", "priority": "medium"},
        )

        validation_result = validator.validate_entity(entity)
        assert validation_result.is_valid is True
        assert len(validation_result.errors) == 0

    def test_relation_validation(self) -> None:
        """Test comprehensive relation validation."""
        ontology = ConversationOntology()
        validator = SchemaValidator(ontology)

        # Create test relation
        relation = ConversationRelation(
            source_entity_id="goal_1",
            target_entity_id="task_1",
            relation_type=RelationType.REQUIRES,
            properties={"strength": 0.9},
        )

        validation_result = validator.validate_relation(relation)
        assert validation_result.is_valid is True
        assert len(validation_result.errors) == 0

    def test_validation_error_reporting(self) -> None:
        """Test that validation errors are properly reported."""
        ontology = ConversationOntology()
        validator = SchemaValidator(ontology)

        # Create invalid entity (missing required properties)
        invalid_entity = ConversationEntity(
            entity_type=EntityType.AGENT,
            label="Test Agent",
            properties={"name": "Test"},  # Missing agent_id
        )

        validation_result = validator.validate_entity(invalid_entity)
        assert validation_result.is_valid is False
        assert len(validation_result.errors) > 0
        assert any("agent_id" in error for error in validation_result.errors)


class TestKnowledgeGraphSchema:
    """Test the complete knowledge graph schema system."""

    def test_schema_creation(self) -> None:
        """Test creating a complete knowledge graph schema."""
        schema = KnowledgeGraphSchema(
            version="1.0",
            ontology=ConversationOntology(),
            metadata_schema={
                "temporal_required": True,
                "provenance_required": True,
                "confidence_tracking": True,
            },
        )

        assert schema.version == "1.0"
        assert schema.ontology is not None
        assert schema.metadata_schema["temporal_required"] is True

    def test_schema_evolution_compatibility(self) -> None:
        """Test schema evolution and backward compatibility."""
        schema_v1 = KnowledgeGraphSchema(version="1.0", ontology=ConversationOntology())
        schema_v2 = KnowledgeGraphSchema(version="2.0", ontology=ConversationOntology())

        # Should be able to check compatibility
        compatibility = schema_v1.is_compatible_with(schema_v2)
        assert isinstance(compatibility, bool)

    def test_schema_validation_rules(self) -> None:
        """Test that schema enforces validation rules."""
        schema = KnowledgeGraphSchema(version="1.0", ontology=ConversationOntology())

        # Schema should have validation rules
        assert hasattr(schema, "validation_rules")
        assert callable(schema.validate_graph_structure)


class TestConflictResolutionStrategy:
    """Test conflict resolution strategies for schema conflicts."""

    def test_conflict_resolution_strategies(self) -> None:
        """Test different conflict resolution strategies."""
        # Timestamp-based resolution
        timestamp_strategy = ConflictResolutionStrategy.LATEST_TIMESTAMP
        assert timestamp_strategy is not None

        # Confidence-based resolution
        confidence_strategy = ConflictResolutionStrategy.HIGHEST_CONFIDENCE
        assert confidence_strategy is not None

        # Provenance-based resolution
        provenance_strategy = ConflictResolutionStrategy.MOST_RELIABLE_SOURCE
        assert provenance_strategy is not None

    def test_conflict_resolution_application(self) -> None:
        """Test applying conflict resolution strategies."""
        strategy = ConflictResolutionStrategy.HIGHEST_CONFIDENCE

        # Create conflicting entities
        entity1 = ConversationEntity(
            entity_type=EntityType.BELIEF,
            label="Weather state",
            properties={"weather": "sunny"},
            provenance=Provenance(
                source_type="observation",
                source_id="obs_1",
                extraction_method="sensor",
                confidence_score=0.7,
            ),
        )

        entity2 = ConversationEntity(
            entity_type=EntityType.BELIEF,
            label="Weather state",
            properties={"weather": "rainy"},
            provenance=Provenance(
                source_type="observation",
                source_id="obs_2",
                extraction_method="sensor",
                confidence_score=0.9,
            ),
        )

        # Strategy should resolve in favor of higher confidence
        resolved = strategy.resolve([entity1, entity2])
        assert resolved.provenance is not None
        assert resolved.provenance.confidence_score == 0.9
        assert resolved.properties["weather"] == "rainy"


class TestSchemaEvolutionManager:
    """Test schema evolution and migration capabilities."""

    def test_schema_evolution_manager_creation(self) -> None:
        """Test creating a schema evolution manager."""
        manager = SchemaEvolutionManager()

        assert manager.current_version is not None
        assert hasattr(manager, "migration_strategies")

    def test_schema_migration(self) -> None:
        """Test migrating schema from one version to another."""
        manager = SchemaEvolutionManager()

        # Should be able to register migration strategies
        assert hasattr(manager, "register_migration")
        assert callable(manager.register_migration)

    def test_backward_compatibility_check(self) -> None:
        """Test checking backward compatibility between schema versions."""
        manager = SchemaEvolutionManager()

        schema_v1 = KnowledgeGraphSchema(version="1.0", ontology=ConversationOntology())
        schema_v2 = KnowledgeGraphSchema(version="1.1", ontology=ConversationOntology())

        # Should be able to check compatibility
        is_compatible = manager.check_compatibility(schema_v1, schema_v2)
        assert isinstance(is_compatible, bool)

    def test_schema_validation_evolution(self) -> None:
        """Test that evolved schemas maintain validation integrity."""
        manager = SchemaEvolutionManager()

        # Should validate that evolved schemas maintain data integrity
        assert hasattr(manager, "validate_evolution")
        assert callable(manager.validate_evolution)
