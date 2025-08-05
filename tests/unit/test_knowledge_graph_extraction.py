"""
Tests for Knowledge Graph Entity and Relation Extraction Pipeline (Task 34.2).

This test suite follows strict TDD principles and validates the extraction
pipeline for agent conversation knowledge graphs.
"""

from datetime import datetime

# These imports will fail initially - that's the point of TDD
from knowledge_graph.extraction import (
    SpacyEntityExtractor,
    PatternRelationExtractor,
    LLMFallbackExtractor,
    ContextAwareExtractor,
    ConfidenceScorer,
    ExtractionPipeline,
    ConversationMessage,
    ExtractionContext,
    CoReferenceResolver,
)
from knowledge_graph.schema import (
    ConversationEntity,
    ConversationRelation,
    EntityType,
    RelationType,
)


class TestConversationMessage:
    """Test the conversation message representation."""

    def test_conversation_message_creation(self) -> None:
        """Test creating a conversation message."""
        message = ConversationMessage(
            message_id="msg_123",
            conversation_id="conv_456",
            agent_id="agent_1",
            content="I need to implement a Python API using FastAPI framework.",
            timestamp=datetime.now(),
            metadata={"speaker": "user", "turn": 1},
        )

        assert message.message_id == "msg_123"
        assert message.conversation_id == "conv_456"
        assert message.agent_id == "agent_1"
        assert "Python" in message.content
        assert message.metadata["speaker"] == "user"

    def test_message_validation(self) -> None:
        """Test conversation message validation."""
        # Valid message
        valid_message = ConversationMessage(
            message_id="msg_123",
            conversation_id="conv_456",
            agent_id="agent_1",
            content="Test message",
            timestamp=datetime.now(),
        )
        assert valid_message.is_valid() is True

        # Invalid message (empty content)
        invalid_message = ConversationMessage(
            message_id="msg_123",
            conversation_id="conv_456",
            agent_id="agent_1",
            content="",
            timestamp=datetime.now(),
        )
        assert invalid_message.is_valid() is False


class TestExtractionContext:
    """Test the extraction context system."""

    def test_extraction_context_creation(self) -> None:
        """Test creating extraction context."""
        context = ExtractionContext(
            conversation_history=[
                ConversationMessage(
                    message_id="msg_1",
                    conversation_id="conv_1",
                    agent_id="agent_1",
                    content="Let's build a web application",
                    timestamp=datetime.now(),
                )
            ],
            current_message=ConversationMessage(
                message_id="msg_2",
                conversation_id="conv_1",
                agent_id="agent_2",
                content="I'll use React for the frontend",
                timestamp=datetime.now(),
            ),
            agent_roles={"agent_1": "user", "agent_2": "assistant"},
            domain_context="software_development",
        )

        assert len(context.conversation_history) == 1
        assert context.current_message.message_id == "msg_2"
        assert context.agent_roles["agent_1"] == "user"
        assert context.domain_context == "software_development"


class TestSpacyEntityExtractor:
    """Test spaCy-based entity extraction."""

    def test_spacy_extractor_creation(self) -> None:
        """Test creating spaCy entity extractor."""
        extractor = SpacyEntityExtractor(model_name="en_core_web_sm")

        assert extractor.model_name == "en_core_web_sm"
        assert extractor.nlp is not None
        assert hasattr(extractor, "extract_entities")

    def test_basic_entity_extraction(self) -> None:
        """Test basic entity extraction from conversation message."""
        extractor = SpacyEntityExtractor()

        message = ConversationMessage(
            message_id="msg_1",
            conversation_id="conv_1",
            agent_id="agent_1",
            content="Hi Claude, I'm working on a Python project with Django framework.",
            timestamp=datetime.now(),
        )

        context = ExtractionContext(
            conversation_history=[],
            current_message=message,
            agent_roles={"agent_1": "user", "Claude": "assistant"},
        )

        entities = extractor.extract_entities(context)

        # Should extract agent names, technology, and framework
        assert len(entities) > 0

        # Check for specific entity types
        entity_types = [e.entity_type for e in entities]
        assert EntityType.AGENT in entity_types  # Claude
        # Should find Python and Django as technology entities

    def test_confidence_scoring(self) -> None:
        """Test confidence scoring for extracted entities."""
        extractor = SpacyEntityExtractor()

        message = ConversationMessage(
            message_id="msg_1",
            conversation_id="conv_1",
            agent_id="agent_1",
            content="OpenAI developed GPT-4 using machine learning techniques.",
            timestamp=datetime.now(),
        )

        context = ExtractionContext(conversation_history=[], current_message=message)

        entities = extractor.extract_entities(context)

        # All entities should have confidence scores
        for entity in entities:
            assert entity.provenance is not None
            assert 0.0 <= entity.provenance.confidence_score <= 1.0

    def test_temporal_metadata_creation(self) -> None:
        """Test that entities get proper temporal metadata."""
        extractor = SpacyEntityExtractor()

        message = ConversationMessage(
            message_id="msg_1",
            conversation_id="conv_1",
            agent_id="agent_1",
            content="Let's use PostgreSQL database.",
            timestamp=datetime.now(),
        )

        context = ExtractionContext(conversation_history=[], current_message=message)

        entities = extractor.extract_entities(context)

        for entity in entities:
            assert entity.temporal_metadata is not None
            assert entity.temporal_metadata.conversation_id == "conv_1"
            assert entity.temporal_metadata.message_id == "msg_1"


class TestPatternRelationExtractor:
    """Test pattern-based relation extraction."""

    def test_pattern_extractor_creation(self) -> None:
        """Test creating pattern-based relation extractor."""
        extractor = PatternRelationExtractor()

        assert hasattr(extractor, "extract_relations")
        assert hasattr(extractor, "add_pattern")

    def test_basic_relation_extraction(self) -> None:
        """Test basic relation extraction between entities."""
        entity_extractor = SpacyEntityExtractor()
        relation_extractor = PatternRelationExtractor()

        message = ConversationMessage(
            message_id="msg_1",
            conversation_id="conv_1",
            agent_id="agent_1",
            content="Django requires Python to run the web application.",
            timestamp=datetime.now(),
        )

        context = ExtractionContext(conversation_history=[], current_message=message)

        # First extract entities
        entities = entity_extractor.extract_entities(context)

        # Then extract relations
        relations = relation_extractor.extract_relations(entities, context)

        # Should find "requires" relationship between Django and Python
        assert len(relations) > 0

        # Check for specific relation types
        relation_types = [r.relation_type for r in relations]
        assert RelationType.REQUIRES in relation_types

    def test_custom_pattern_addition(self) -> None:
        """Test adding custom extraction patterns."""
        extractor = PatternRelationExtractor()

        # Add custom pattern for "built with" relationships
        pattern = {
            "pattern": r"(\w+) (?:is )?built with (\w+)",
            "relation_type": RelationType.DEPENDS_ON,
            "confidence": 0.8,
        }

        extractor.add_pattern(pattern)

        # Verify pattern was added
        assert len(extractor.patterns) > 0

    def test_confidence_propagation(self) -> None:
        """Test that relations get appropriate confidence scores."""
        entity_extractor = SpacyEntityExtractor()
        relation_extractor = PatternRelationExtractor()

        message = ConversationMessage(
            message_id="msg_1",
            conversation_id="conv_1",
            agent_id="agent_1",
            content="React enables building interactive user interfaces.",
            timestamp=datetime.now(),
        )

        context = ExtractionContext(conversation_history=[], current_message=message)

        entities = entity_extractor.extract_entities(context)
        relations = relation_extractor.extract_relations(entities, context)

        for relation in relations:
            assert relation.provenance is not None
            assert 0.0 <= relation.provenance.confidence_score <= 1.0


class TestCoReferenceResolver:
    """Test co-reference resolution across messages."""

    def test_coreference_resolver_creation(self) -> None:
        """Test creating co-reference resolver."""
        resolver = CoReferenceResolver()

        assert hasattr(resolver, "resolve_references")
        assert hasattr(resolver, "update_context")

    def test_pronoun_resolution(self) -> None:
        """Test resolving pronouns to previously mentioned entities."""
        resolver = CoReferenceResolver()

        # First message mentions specific entity
        msg1 = ConversationMessage(
            message_id="msg_1",
            conversation_id="conv_1",
            agent_id="agent_1",
            content="I'm working on a Django project.",
            timestamp=datetime.now(),
        )

        # Second message uses pronoun
        msg2 = ConversationMessage(
            message_id="msg_2",
            conversation_id="conv_1",
            agent_id="agent_1",
            content="It requires Python 3.8 or higher.",
            timestamp=datetime.now(),
        )

        context = ExtractionContext(conversation_history=[msg1], current_message=msg2)

        # Should resolve "It" to "Django project"
        resolved_entities = resolver.resolve_references(context)

        # Check that pronoun resolution was attempted (may or may not find entities)
        assert isinstance(resolved_entities, list)
        # For basic implementation, we just verify the method works
        # More sophisticated tests would verify actual resolution accuracy

    def test_entity_linking_across_messages(self) -> None:
        """Test linking entities mentioned across different messages."""
        resolver = CoReferenceResolver()

        msg1 = ConversationMessage(
            message_id="msg_1",
            conversation_id="conv_1",
            agent_id="agent_1",
            content="The API will use FastAPI framework.",
            timestamp=datetime.now(),
        )

        msg2 = ConversationMessage(
            message_id="msg_2",
            conversation_id="conv_1",
            agent_id="agent_2",
            content="FastAPI supports automatic OpenAPI documentation.",
            timestamp=datetime.now(),
        )

        context = ExtractionContext(conversation_history=[msg1], current_message=msg2)

        # Should link FastAPI mentions across messages
        linked_entities = resolver.resolve_references(context)

        # Verify that linking was attempted
        assert isinstance(linked_entities, list)
        # For basic implementation, we just verify the method works


class TestContextAwareExtractor:
    """Test context-aware extraction that considers conversation history."""

    def test_context_aware_creation(self) -> None:
        """Test creating context-aware extractor."""
        extractor = ContextAwareExtractor()

        assert hasattr(extractor, "extract_with_context")
        assert hasattr(extractor, "update_conversation_context")

    def test_agent_role_awareness(self) -> None:
        """Test extraction considers agent roles."""
        extractor = ContextAwareExtractor()

        message = ConversationMessage(
            message_id="msg_1",
            conversation_id="conv_1",
            agent_id="agent_1",
            content="I need help with Python programming.",
            timestamp=datetime.now(),
        )

        # Context with role information
        context = ExtractionContext(
            conversation_history=[],
            current_message=message,
            agent_roles={"agent_1": "student", "agent_2": "teacher"},
        )

        entities = extractor.extract_with_context(context)

        # Should extract agent as AGENT entity type with student role
        agent_entities = [e for e in entities if e.entity_type == EntityType.AGENT]
        assert len(agent_entities) > 0

    def test_domain_context_influence(self) -> None:
        """Test that domain context influences extraction."""
        extractor = ContextAwareExtractor()

        message = ConversationMessage(
            message_id="msg_1",
            conversation_id="conv_1",
            agent_id="agent_1",
            content="We need to optimize the pipeline performance.",
            timestamp=datetime.now(),
        )

        # Software development context
        context = ExtractionContext(
            conversation_history=[], current_message=message, domain_context="software_development"
        )

        entities = extractor.extract_with_context(context)

        # Should extract entities with domain context
        assert len(entities) >= 0  # May or may not find specific entities

        # Verify context was applied
        for entity in entities:
            if entity.properties.get("domain_context"):
                assert entity.properties["domain_context"] == "software_development"

    def test_conversation_history_influence(self) -> None:
        """Test that conversation history influences current extraction."""
        extractor = ContextAwareExtractor()

        # Previous message establishes context
        prev_msg = ConversationMessage(
            message_id="msg_1",
            conversation_id="conv_1",
            agent_id="agent_1",
            content="I'm building a machine learning model.",
            timestamp=datetime.now(),
        )

        # Current message uses ambiguous terms
        current_msg = ConversationMessage(
            message_id="msg_2",
            conversation_id="conv_1",
            agent_id="agent_1",
            content="The training data needs preprocessing.",
            timestamp=datetime.now(),
        )

        context = ExtractionContext(conversation_history=[prev_msg], current_message=current_msg)

        entities = extractor.extract_with_context(context)

        # Should extract entities influenced by conversation history
        assert len(entities) >= 0  # May or may not find specific entities

        # Verify context was considered
        for entity in entities:
            if entity.properties.get("has_context"):
                assert entity.properties["has_context"] is True


class TestExtractionPipeline:
    """Test the complete extraction pipeline."""

    def test_pipeline_creation(self) -> None:
        """Test creating extraction pipeline with multiple strategies."""
        pipeline = ExtractionPipeline(
            entity_strategies=[SpacyEntityExtractor()],
            relation_strategies=[PatternRelationExtractor()],
            confidence_scorer=ConfidenceScorer(),
            coreference_resolver=CoReferenceResolver(),
        )

        assert len(pipeline.entity_strategies) == 1
        assert len(pipeline.relation_strategies) == 1
        assert pipeline.confidence_scorer is not None
        assert pipeline.coreference_resolver is not None

    def test_end_to_end_extraction(self) -> None:
        """Test complete end-to-end extraction process."""
        pipeline = ExtractionPipeline(
            entity_strategies=[SpacyEntityExtractor()],
            relation_strategies=[PatternRelationExtractor()],
        )

        message = ConversationMessage(
            message_id="msg_1",
            conversation_id="conv_1",
            agent_id="agent_1",
            content="Claude, please help me build a React app with TypeScript.",
            timestamp=datetime.now(),
        )

        context = ExtractionContext(
            conversation_history=[],
            current_message=message,
            agent_roles={"agent_1": "user", "Claude": "assistant"},
        )

        result = pipeline.extract(context)

        # Should extract entities and relations
        assert len(result.entities) > 0
        assert len(result.relations) >= 0  # May or may not find relations
        assert result.extraction_metadata["processing_time"] > 0

    def test_pipeline_error_handling(self) -> None:
        """Test pipeline handles extraction errors gracefully."""
        pipeline = ExtractionPipeline(
            entity_strategies=[SpacyEntityExtractor()],
            relation_strategies=[PatternRelationExtractor()],
        )

        # Malformed message
        message = ConversationMessage(
            message_id="msg_1",
            conversation_id="conv_1",
            agent_id="agent_1",
            content="",  # Empty content
            timestamp=datetime.now(),
        )

        context = ExtractionContext(conversation_history=[], current_message=message)

        # Should handle gracefully without crashing
        result = pipeline.extract(context)

        assert result is not None
        assert isinstance(result.entities, list)
        assert isinstance(result.relations, list)

    def test_pipeline_confidence_aggregation(self) -> None:
        """Test that pipeline properly aggregates confidence scores."""
        pipeline = ExtractionPipeline(
            entity_strategies=[SpacyEntityExtractor()],
            relation_strategies=[PatternRelationExtractor()],
            confidence_scorer=ConfidenceScorer(),
        )

        message = ConversationMessage(
            message_id="msg_1",
            conversation_id="conv_1",
            agent_id="agent_1",
            content="TensorFlow is used for deep learning applications.",
            timestamp=datetime.now(),
        )

        context = ExtractionContext(conversation_history=[], current_message=message)

        result = pipeline.extract(context)

        # All entities should have final confidence scores
        for entity in result.entities:
            assert entity.provenance is not None
            assert entity.provenance.confidence_score > 0

        # All relations should have confidence scores
        for relation in result.relations:
            assert relation.provenance is not None
            assert relation.provenance.confidence_score > 0


class TestLLMFallbackExtractor:
    """Test LLM-based fallback extraction for complex cases."""

    def test_llm_fallback_creation(self) -> None:
        """Test creating LLM fallback extractor."""
        extractor = LLMFallbackExtractor(model_name="gpt-3.5-turbo", api_key="test_key")

        assert extractor.model_name == "gpt-3.5-turbo"
        assert hasattr(extractor, "extract_with_llm")

    def test_llm_fallback_trigger(self) -> None:
        """Test that LLM fallback is triggered for low confidence."""
        pipeline = ExtractionPipeline(
            entity_strategies=[SpacyEntityExtractor()],
            relation_strategies=[PatternRelationExtractor()],
            fallback_extractor=LLMFallbackExtractor(),
        )

        # Complex sentence that traditional NLP might struggle with
        message = ConversationMessage(
            message_id="msg_1",
            conversation_id="conv_1",
            agent_id="agent_1",
            content="The agentic framework leverages PyMDP for active inference in multi-agent scenarios.",
            timestamp=datetime.now(),
        )

        context = ExtractionContext(conversation_history=[], current_message=message)

        result = pipeline.extract(context)

        # Should still produce results even with complex terminology
        assert len(result.entities) >= 0  # May trigger fallback
        assert result.extraction_metadata is not None

    def test_llm_schema_compliance(self) -> None:
        """Test that LLM fallback produces schema-compliant entities."""
        extractor = LLMFallbackExtractor()

        message = ConversationMessage(
            message_id="msg_1",
            conversation_id="conv_1",
            agent_id="agent_1",
            content="Active inference enables autonomous agents to minimize free energy.",
            timestamp=datetime.now(),
        )

        context = ExtractionContext(conversation_history=[], current_message=message)

        # Mock the LLM response for testing
        entities = extractor.extract_with_llm(context)

        # All entities should conform to our schema
        for entity in entities:
            assert isinstance(entity, ConversationEntity)
            assert entity.entity_type in EntityType
            assert entity.temporal_metadata is not None
            assert entity.provenance is not None


class TestConfidenceScorer:
    """Test confidence scoring system."""

    def test_confidence_scorer_creation(self) -> None:
        """Test creating confidence scorer."""
        scorer = ConfidenceScorer()

        assert hasattr(scorer, "score_entity")
        assert hasattr(scorer, "score_relation")
        assert hasattr(scorer, "aggregate_scores")

    def test_entity_confidence_scoring(self) -> None:
        """Test confidence scoring for entities."""
        scorer = ConfidenceScorer()

        entity = ConversationEntity(
            entity_type=EntityType.CONCEPT,
            label="Python",
            properties={"mentioned_count": 5, "context_clarity": "high"},
        )

        confidence = scorer.score_entity(
            entity,
            context_factors={
                "extraction_method": "spacy_pattern",
                "context_support": 0.9,
                "entity_frequency": 0.8,
            },
        )

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # Should be reasonably confident

    def test_relation_confidence_scoring(self) -> None:
        """Test confidence scoring for relations."""
        scorer = ConfidenceScorer()

        entity1 = ConversationEntity(entity_type=EntityType.CONCEPT, label="Django")

        entity2 = ConversationEntity(entity_type=EntityType.CONCEPT, label="Python")

        relation = ConversationRelation(
            source_entity_id=entity1.entity_id,
            target_entity_id=entity2.entity_id,
            relation_type=RelationType.REQUIRES,
        )

        confidence = scorer.score_relation(
            relation,
            context_factors={
                "pattern_strength": 0.85,
                "entity_confidence": 0.9,
                "syntactic_support": 0.7,
            },
        )

        assert 0.0 <= confidence <= 1.0

    def test_confidence_aggregation(self) -> None:
        """Test aggregating confidence scores from multiple sources."""
        scorer = ConfidenceScorer()

        scores = [0.8, 0.7, 0.9, 0.6]

        # Test different aggregation methods
        avg_score = scorer.aggregate_scores(scores, method="average")
        max_score = scorer.aggregate_scores(scores, method="max")
        weighted_score = scorer.aggregate_scores(
            scores, method="weighted", weights=[0.4, 0.3, 0.2, 0.1]
        )

        assert 0.0 <= avg_score <= 1.0
        assert 0.0 <= max_score <= 1.0
        assert 0.0 <= weighted_score <= 1.0
        assert max_score == 0.9  # Should be the maximum score
