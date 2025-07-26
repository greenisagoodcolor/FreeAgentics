"""
Test NLP Entity Extraction with real libraries (spaCy/NLTK)
Following TDD principles - write tests first, then implementation
"""

try:
    import spacy
except ImportError:
    spacy = None

import pytest

from knowledge_graph.nlp_entity_extractor import (
    Entity,
    EntityType,
    ExtractionResult,
    NLPEntityExtractor,
    Relationship,
)


@pytest.mark.skipif(spacy is None, reason="spaCy not available")
class TestNLPEntityExtraction:
    """Test suite for NLP entity extraction"""

    def test_entity_data_structure(self):
        """Test Entity data structure"""
        entity = Entity(
            text="Python",
            type=EntityType.TECHNOLOGY,
            start_pos=10,
            end_pos=16,
            confidence=0.95,
        )

        assert entity.text == "Python"
        assert entity.type == EntityType.TECHNOLOGY
        assert entity.start_pos == 10
        assert entity.end_pos == 16
        assert entity.confidence == 0.95

    def test_relationship_data_structure(self):
        """Test Relationship data structure"""
        entity1 = Entity("Python", EntityType.TECHNOLOGY, 0, 6, 0.9)
        entity2 = Entity("programming", EntityType.CONCEPT, 10, 21, 0.85)

        relationship = Relationship(source=entity1, target=entity2, type="used_for", confidence=0.8)

        assert relationship.source == entity1
        assert relationship.target == entity2
        assert relationship.type == "used_for"
        assert relationship.confidence == 0.8

    def test_extractor_initialization(self):
        """Test NLPEntityExtractor initialization with spaCy"""
        extractor = NLPEntityExtractor(model_name="en_core_web_sm")

        assert extractor.model_name == "en_core_web_sm"
        assert extractor.nlp is not None
        assert isinstance(extractor.nlp, spacy.language.Language)

    def test_extract_entities_from_simple_text(self):
        """Test extracting entities from simple text"""
        extractor = NLPEntityExtractor()

        text = "Python is a programming language created by Guido van Rossum."
        result = extractor.extract_entities(text)

        assert isinstance(result, ExtractionResult)
        assert len(result.entities) > 0

        # Check for expected entities
        entity_texts = [e.text for e in result.entities]
        assert "Python" in entity_texts
        assert "Guido van Rossum" in entity_texts

    def test_extract_technology_entities(self):
        """Test extracting technology-related entities"""
        extractor = NLPEntityExtractor()

        text = "I'm learning React, TypeScript, and Node.js for web development."
        result = extractor.extract_entities(text)

        tech_entities = [e for e in result.entities if e.type == EntityType.TECHNOLOGY]
        tech_names = [e.text for e in tech_entities]

        assert "React" in tech_names
        assert "TypeScript" in tech_names
        assert "Node.js" in tech_names

    def test_extract_person_entities(self):
        """Test extracting person entities"""
        extractor = NLPEntityExtractor()

        text = "Linus Torvalds created Linux, while Bill Gates founded Microsoft."
        result = extractor.extract_entities(text)

        person_entities = [e for e in result.entities if e.type == EntityType.PERSON]
        person_names = [e.text for e in person_entities]

        assert "Linus Torvalds" in person_names
        assert "Bill Gates" in person_names

    def test_extract_organization_entities(self):
        """Test extracting organization entities"""
        extractor = NLPEntityExtractor()

        text = "Google, Microsoft, and Apple are major tech companies."
        result = extractor.extract_entities(text)

        org_entities = [e for e in result.entities if e.type == EntityType.ORGANIZATION]
        org_names = [e.text for e in org_entities]

        assert "Google" in org_names
        assert "Microsoft" in org_names
        assert "Apple" in org_names

    def test_extract_concept_entities(self):
        """Test extracting concept entities"""
        extractor = NLPEntityExtractor()

        text = "Machine learning and artificial intelligence are transforming data science."
        result = extractor.extract_entities(text)

        concept_entities = [e for e in result.entities if e.type == EntityType.CONCEPT]
        concept_texts = [e.text for e in concept_entities]

        assert any("machine learning" in text.lower() for text in concept_texts)
        assert any("artificial intelligence" in text.lower() for text in concept_texts)
        assert any("data science" in text.lower() for text in concept_texts)

    def test_extract_relationships(self):
        """Test extracting relationships between entities"""
        extractor = NLPEntityExtractor()

        text = "Python is used for machine learning and data analysis."
        result = extractor.extract_entities(text)

        assert len(result.relationships) > 0

        # Find relationship between Python and machine learning
        python_ml_rel = None
        for rel in result.relationships:
            if rel.source.text == "Python" and "machine learning" in rel.target.text:
                python_ml_rel = rel
                break

        assert python_ml_rel is not None
        assert python_ml_rel.type == "used_for"

    def test_extract_from_conversation_context(self):
        """Test extraction with conversation context"""
        extractor = NLPEntityExtractor()

        messages = [
            "I'm working on a Django project.",
            "Django is a Python web framework.",
            "It uses the MVT pattern.",
        ]

        # Extract from each message with context
        all_entities = []
        for i, message in enumerate(messages):
            context = messages[:i] if i > 0 else []
            result = extractor.extract_entities(message, context=context)
            all_entities.extend(result.entities)

        entity_texts = [e.text for e in all_entities]
        assert "Django" in entity_texts
        assert "Python" in entity_texts
        assert "MVT pattern" in entity_texts or "MVT" in entity_texts

    def test_confidence_scores(self):
        """Test that confidence scores are reasonable"""
        extractor = NLPEntityExtractor()

        text = "JavaScript is definitely a programming language."
        result = extractor.extract_entities(text)

        # All entities should have confidence scores
        for entity in result.entities:
            assert 0.0 <= entity.confidence <= 1.0

        # "JavaScript" should have high confidence
        js_entity = next(e for e in result.entities if e.text == "JavaScript")
        assert js_entity.confidence > 0.7

    def test_custom_entity_patterns(self):
        """Test adding custom entity patterns"""
        extractor = NLPEntityExtractor()

        # Add custom patterns for technologies
        custom_patterns = [
            {"label": "TECHNOLOGY", "pattern": "React Native"},
            {"label": "TECHNOLOGY", "pattern": "Vue.js"},
            {"label": "CONCEPT", "pattern": "RESTful API"},
        ]

        extractor.add_custom_patterns(custom_patterns)

        text = "We're building a React Native app with RESTful API integration."
        result = extractor.extract_entities(text)

        entity_texts = [e.text for e in result.entities]
        assert "React Native" in entity_texts
        assert "RESTful API" in entity_texts

    def test_batch_extraction(self):
        """Test batch extraction from multiple texts"""
        extractor = NLPEntityExtractor()

        texts = [
            "Python is great for data science.",
            "Java is used for Android development.",
            "JavaScript powers web applications.",
        ]

        results = extractor.extract_entities_batch(texts)

        assert len(results) == 3

        # Check each result
        all_entities = []
        for result in results:
            all_entities.extend(result.entities)

        entity_texts = [e.text for e in all_entities]
        assert "Python" in entity_texts
        assert "Java" in entity_texts
        assert "JavaScript" in entity_texts

    def test_entity_deduplication(self):
        """Test that duplicate entities are handled properly"""
        extractor = NLPEntityExtractor()

        text = "Python is awesome. I love Python. Python rocks!"
        result = extractor.extract_entities(text)

        # Should merge duplicate Python entities
        python_entities = [e for e in result.entities if e.text == "Python"]
        assert len(python_entities) == 1

        # The merged entity should have the highest confidence
        assert python_entities[0].confidence >= 0.8

    def test_extract_with_metadata(self):
        """Test extraction with additional metadata"""
        extractor = NLPEntityExtractor()

        text = "TensorFlow is a machine learning framework by Google."
        metadata = {"domain": "AI/ML", "source": "technical_documentation"}

        result = extractor.extract_entities(text, metadata=metadata)

        assert result.metadata == metadata

        # Metadata might influence entity types or confidence
        tf_entity = next(e for e in result.entities if e.text == "TensorFlow")
        assert tf_entity.type == EntityType.TECHNOLOGY

    def test_entity_type_classification(self):
        """Test correct classification of entity types"""
        extractor = NLPEntityExtractor()

        test_cases = [
            ("Python", EntityType.TECHNOLOGY),
            ("Elon Musk", EntityType.PERSON),
            ("SpaceX", EntityType.ORGANIZATION),
            ("machine learning", EntityType.CONCEPT),
            ("New York", EntityType.LOCATION),
            ("2023", EntityType.DATE),
        ]

        for text, expected_type in test_cases:
            result = extractor.extract_entities(f"Let's talk about {text}.")
            entity = next((e for e in result.entities if text in e.text), None)
            assert entity is not None, f"Entity '{text}' not found"
            assert (
                entity.type == expected_type
            ), f"Expected {expected_type} for '{text}', got {entity.type}"

    def test_empty_text_handling(self):
        """Test handling of empty or null text"""
        extractor = NLPEntityExtractor()

        # Empty string
        result = extractor.extract_entities("")
        assert len(result.entities) == 0
        assert len(result.relationships) == 0

        # Whitespace only
        result = extractor.extract_entities("   \n\t  ")
        assert len(result.entities) == 0

    def test_special_characters_handling(self):
        """Test handling of special characters and formatting"""
        extractor = NLPEntityExtractor()

        text = "C++ and C# are different from C. Also, .NET is a framework."
        result = extractor.extract_entities(text)

        entity_texts = [e.text for e in result.entities]
        assert "C++" in entity_texts
        assert "C#" in entity_texts
        assert ".NET" in entity_texts

    def test_extraction_performance(self):
        """Test extraction performance on longer text"""
        extractor = NLPEntityExtractor()

        # Long technical text
        text = """
        In modern software development, Python has become increasingly popular for
        machine learning applications. Libraries like TensorFlow, PyTorch, and
        scikit-learn have made it easier for developers to build sophisticated
        AI models. Companies like Google, Facebook, and Microsoft are investing
        heavily in AI research. The field of natural language processing has seen
        significant advances with models like BERT and GPT-3.
        """

        import time

        start_time = time.time()
        result = extractor.extract_entities(text)
        extraction_time = time.time() - start_time

        # Should complete reasonably quickly
        assert extraction_time < 1.0  # Less than 1 second

        # Should find multiple entities
        assert len(result.entities) >= 10

        # Check for some expected entities
        entity_texts = [e.text for e in result.entities]
        assert "Python" in entity_texts
        assert "TensorFlow" in entity_texts
        assert "Google" in entity_texts
