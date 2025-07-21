"""
NLP Entity Extractor using real libraries (spaCy/NLTK).
Extracts entities and relationships from text using state-of-the-art NLP models
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span

logger = logging.getLogger(__name__)


class EntityType(Enum):
    """Types of entities that can be extracted."""

    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    TECHNOLOGY = "TECHNOLOGY"
    CONCEPT = "CONCEPT"
    LOCATION = "LOCATION"
    DATE = "DATE"
    EVENT = "EVENT"
    PRODUCT = "PRODUCT"


@dataclass
class Entity:
    """Represents an extracted entity."""

    text: str
    type: EntityType
    start_pos: int
    end_pos: int
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Relationship:
    """Represents a relationship between entities."""

    source: Entity
    target: Entity
    type: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Result of entity extraction."""

    entities: List[Entity]
    relationships: List[Relationship]
    text: str
    metadata: Optional[Dict[str, Any]] = None
    extraction_time: float = 0.0


class NLPEntityExtractor:
    """Real NLP entity extractor using spaCy and custom patterns."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the NLP entity extractor."""
        self.model_name = model_name
        self.nlp = spacy.load(model_name)
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_custom_patterns()
        self._entity_cache: Dict[str, List[Entity]] = {}

        logger.info(f"NLP Entity Extractor initialized with model: {model_name}")

    def _setup_custom_patterns(self):
        """Setup custom patterns for technology and concept detection."""
        # Technology patterns
        tech_patterns = [
            # Programming languages
            [
                {
                    "LOWER": {
                        "IN": [
                            "python",
                            "javascript",
                            "java",
                            "c++",
                            "c#",
                            "go",
                            "rust",
                            "swift",
                            "kotlin",
                        ]
                    }
                }
            ],
            [{"TEXT": {"IN": ["TypeScript", "C++", "C#", ".NET"]}}],
            # Frameworks and libraries
            [
                {
                    "LOWER": {
                        "IN": [
                            "react",
                            "angular",
                            "vue",
                            "django",
                            "flask",
                            "spring",
                            "tensorflow",
                            "pytorch",
                        ]
                    }
                }
            ],
            [
                {
                    "TEXT": {
                        "IN": [
                            "React",
                            "Angular",
                            "Vue.js",
                            "Django",
                            "Flask",
                            "TensorFlow",
                            "PyTorch",
                            "Node.js",
                        ]
                    }
                }
            ],
            [{"TEXT": "React"}, {"TEXT": "Native"}],
            # Databases
            [
                {
                    "LOWER": {
                        "IN": [
                            "mysql",
                            "postgresql",
                            "mongodb",
                            "redis",
                            "sqlite",
                            "oracle",
                        ]
                    }
                }
            ],
            [
                {
                    "TEXT": {
                        "IN": [
                            "MySQL",
                            "PostgreSQL",
                            "MongoDB",
                            "Redis",
                            "SQLite",
                        ]
                    }
                }
            ],
            # Cloud and services
            [{"LOWER": {"IN": ["aws", "azure", "gcp", "docker", "kubernetes"]}}],
            [{"TEXT": {"IN": ["AWS", "Azure", "GCP", "Docker", "Kubernetes"]}}],
        ]

        # Concept patterns
        concept_patterns = [
            # AI/ML concepts
            [{"LOWER": "machine"}, {"LOWER": "learning"}],
            [{"LOWER": "artificial"}, {"LOWER": "intelligence"}],
            [{"LOWER": "deep"}, {"LOWER": "learning"}],
            [{"LOWER": "neural"}, {"LOWER": {"IN": ["network", "networks"]}}],
            [
                {"LOWER": "natural"},
                {"LOWER": "language"},
                {"LOWER": "processing"},
            ],
            [{"LOWER": {"IN": ["nlp", "ai", "ml", "cv"]}}],
            [{"TEXT": {"IN": ["AI", "ML", "NLP", "CV", "LLM"]}}],
            # Software development concepts
            [
                {
                    "LOWER": {
                        "IN": [
                            "api",
                            "rest",
                            "graphql",
                            "microservices",
                            "devops",
                        ]
                    }
                }
            ],
            [{"TEXT": {"IN": ["API", "REST", "GraphQL", "DevOps"]}}],
            [{"LOWER": "test"}, {"LOWER": "driven"}, {"LOWER": "development"}],
            [{"TEXT": "RESTful"}, {"TEXT": "API"}],
            [{"TEXT": "MVT"}, {"TEXT": "pattern"}],
        ]

        # Add patterns to matcher
        self.matcher.add("TECHNOLOGY", tech_patterns)
        self.matcher.add("CONCEPT", concept_patterns)

    def extract_entities(
        self,
        text: str,
        context: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ExtractionResult:
        """Extract entities from text."""
        start_time = time.time()

        if not text or not text.strip():
            return ExtractionResult(
                entities=[],
                relationships=[],
                text=text,
                metadata=metadata,
                extraction_time=time.time() - start_time,
            )

        # Process text with spaCy
        doc = self.nlp(text)

        # Extract entities
        entities = []

        # Get custom pattern matches first (higher priority)
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            span = doc[start:end]

            entity_type = (
                EntityType.TECHNOLOGY if label == "TECHNOLOGY" else EntityType.CONCEPT
            )
            entity = Entity(
                text=span.text,
                type=entity_type,
                start_pos=span.start_char,
                end_pos=span.end_char,
                confidence=0.85,  # High confidence for pattern matches
                metadata={"pattern_label": label},
            )
            entities.append(entity)

        # Get spaCy named entities (lower priority)
        for ent in doc.ents:
            entity_type = self._map_spacy_label_to_entity_type(ent.label_)
            if entity_type:
                # Avoid duplicates with custom patterns
                if not self._is_duplicate_entity(
                    entities, ent.text, ent.start_char, ent.end_char
                ):
                    entity = Entity(
                        text=ent.text,
                        type=entity_type,
                        start_pos=ent.start_char,
                        end_pos=ent.end_char,
                        confidence=self._calculate_confidence(ent),
                        metadata={"spacy_label": ent.label_},
                    )
                    entities.append(entity)

        # Deduplicate entities
        entities = self._deduplicate_entities(entities)

        # Extract relationships
        relationships = self._extract_relationships(entities, doc)

        extraction_time = time.time() - start_time

        return ExtractionResult(
            entities=entities,
            relationships=relationships,
            text=text,
            metadata=metadata,
            extraction_time=extraction_time,
        )

    def extract_entities_batch(self, texts: List[str]) -> List[ExtractionResult]:
        """Extract entities from multiple texts."""
        return [self.extract_entities(text) for text in texts]

    def add_custom_patterns(self, patterns: List[Dict[str, Any]]):
        """Add custom entity patterns."""
        for pattern in patterns:
            label = pattern["label"]
            pattern_rules = pattern["pattern"]

            # Convert string patterns to spaCy pattern format
            if isinstance(pattern_rules, str):
                pattern_rules = [{"LOWER": pattern_rules.lower()}]
            elif isinstance(pattern_rules, list) and isinstance(pattern_rules[0], str):
                pattern_rules = [{"LOWER": token.lower()} for token in pattern_rules]

            # Add to appropriate category
            if label == "TECHNOLOGY":
                self.matcher.add("TECHNOLOGY", [pattern_rules])
            elif label == "CONCEPT":
                self.matcher.add("CONCEPT", [pattern_rules])

        logger.info(f"Added {len(patterns)} custom patterns")

    def _map_spacy_label_to_entity_type(self, spacy_label: str) -> Optional[EntityType]:
        """Map spaCy entity labels to our entity types."""
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,  # Geopolitical entity
            "LOC": EntityType.LOCATION,
            "DATE": EntityType.DATE,
            "EVENT": EntityType.EVENT,
            "PRODUCT": EntityType.PRODUCT,
            "WORK_OF_ART": EntityType.PRODUCT,
            "LAW": EntityType.CONCEPT,
            "LANGUAGE": EntityType.TECHNOLOGY,
        }
        return mapping.get(spacy_label)

    def _calculate_confidence(self, ent: Span) -> float:
        """Calculate confidence score for spaCy entity."""
        # Base confidence on entity type and length
        base_confidence = 0.7

        # Higher confidence for longer entities (more specific)
        length_bonus = min(0.2, len(ent.text.split()) * 0.05)

        # Higher confidence for capitalized entities
        capitalization_bonus = 0.1 if ent.text[0].isupper() else 0.0

        # Known entities get higher confidence
        if ent.label_ in ["PERSON", "ORG", "GPE"]:
            base_confidence = 0.8

        return min(0.95, base_confidence + length_bonus + capitalization_bonus)

    def _is_duplicate_entity(
        self, entities: List[Entity], text: str, start_pos: int, end_pos: int
    ) -> bool:
        """Check if entity is duplicate."""
        for entity in entities:
            # Exact match
            if (
                entity.text == text
                and entity.start_pos == start_pos
                and entity.end_pos == end_pos
            ):
                return True

            # Overlapping spans
            if entity.start_pos < end_pos and entity.end_pos > start_pos:
                return True

        return False

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities, keeping highest confidence."""
        if not entities:
            return entities

        # Group by text (case-insensitive)
        grouped: Dict[str, List[Entity]] = {}
        for entity in entities:
            key = entity.text.lower()
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(entity)

        # Keep highest confidence entity for each group
        deduplicated = []
        for group in grouped.values():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Keep the one with highest confidence
                best_entity = max(group, key=lambda e: e.confidence)
                deduplicated.append(best_entity)

        return deduplicated

    def _extract_relationships(
        self, entities: List[Entity], doc: Doc
    ) -> List[Relationship]:
        """Extract relationships between entities."""
        relationships: List[Relationship] = []

        if len(entities) < 2:
            return relationships

        # Simple rule-based relationship extraction
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i + 1 :]:
                relationship_type = self._infer_relationship_type(entity1, entity2, doc)
                if relationship_type:
                    relationship = Relationship(
                        source=entity1,
                        target=entity2,
                        type=relationship_type,
                        confidence=0.7,
                    )
                    relationships.append(relationship)

        return relationships

    def _infer_relationship_type(
        self, entity1: Entity, entity2: Entity, doc: Doc
    ) -> Optional[str]:
        """Infer relationship type between two entities."""
        # Technology + Concept relationships
        if entity1.type == EntityType.TECHNOLOGY and entity2.type == EntityType.CONCEPT:
            return "used_for"
        elif (
            entity1.type == EntityType.CONCEPT and entity2.type == EntityType.TECHNOLOGY
        ):
            return "implemented_by"

        # Person + Organization relationships
        if (
            entity1.type == EntityType.PERSON
            and entity2.type == EntityType.ORGANIZATION
        ):
            return "works_at"
        elif (
            entity1.type == EntityType.ORGANIZATION
            and entity2.type == EntityType.PERSON
        ):
            return "employs"

        # Technology + Technology relationships
        if (
            entity1.type == EntityType.TECHNOLOGY
            and entity2.type == EntityType.TECHNOLOGY
        ):
            # Look for relationship words in context
            text_between = self._get_text_between_entities(entity1, entity2, doc.text)
            if any(
                word in text_between.lower()
                for word in ["built with", "uses", "based on"]
            ):
                return "uses"
            elif any(word in text_between.lower() for word in ["and", ","]):
                return "related_to"

        return None

    def _get_text_between_entities(
        self, entity1: Entity, entity2: Entity, text: str
    ) -> str:
        """Get text between two entities."""
        start = min(entity1.end_pos, entity2.end_pos)
        end = max(entity1.start_pos, entity2.start_pos)
        if start < end:
            return text[start:end]
        return ""
