"""Knowledge Graph Entity and Relation Extraction Pipeline (Task 34.2).

This module implements a comprehensive extraction pipeline for agent conversation
knowledge graphs, including entity extraction, relation extraction, co-reference
resolution, and confidence scoring.

Follows SOLID principles with strategy pattern for pluggable extractors.
"""

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

# Optional imports for enhanced functionality
try:
    import spacy
    from spacy.matcher import Matcher
    from spacy.tokens import Doc, Span, Token
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None

from knowledge_graph.schema import (
    ConversationEntity,
    ConversationRelation,
    ConversationOntology,
    EntityType,
    RelationType,
    Provenance,
    TemporalMetadata,
)

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """Represents a message in an agent conversation."""
    
    message_id: str
    conversation_id: str
    agent_id: str
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Validate message structure."""
        return (
            bool(self.message_id) and
            bool(self.conversation_id) and
            bool(self.agent_id) and
            bool(self.content.strip()) and
            isinstance(self.timestamp, datetime)
        )


@dataclass
class ExtractionContext:
    """Context for entity and relation extraction."""
    
    conversation_history: List[ConversationMessage]
    current_message: ConversationMessage
    agent_roles: Dict[str, str] = field(default_factory=dict)
    domain_context: Optional[str] = None
    extraction_settings: Dict[str, Any] = field(default_factory=dict)
    
    def get_context_window(self, window_size: int = 5) -> List[ConversationMessage]:
        """Get recent conversation context."""
        return self.conversation_history[-window_size:] if self.conversation_history else []


@dataclass
class ExtractionResult:
    """Result of extraction pipeline."""
    
    entities: List[ConversationEntity]
    relations: List[ConversationRelation]
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0


class EntityExtractionStrategy(ABC):
    """Abstract base class for entity extraction strategies."""
    
    @abstractmethod
    def extract_entities(self, context: ExtractionContext) -> List[ConversationEntity]:
        """Extract entities from conversation context."""
        pass
    
    def get_strategy_name(self) -> str:
        """Get strategy identifier."""
        return self.__class__.__name__


class RelationExtractionStrategy(ABC):
    """Abstract base class for relation extraction strategies."""
    
    @abstractmethod
    def extract_relations(
        self, 
        entities: List[ConversationEntity], 
        context: ExtractionContext
    ) -> List[ConversationRelation]:
        """Extract relations between entities."""
        pass
    
    def get_strategy_name(self) -> str:
        """Get strategy identifier."""
        return self.__class__.__name__


class SpacyEntityExtractor(EntityExtractionStrategy):
    """spaCy-based entity extraction strategy."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize spaCy entity extractor."""
        if not SPACY_AVAILABLE:
            raise ImportError("spaCy is required for SpacyEntityExtractor")
        
        self.model_name = model_name
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            logger.warning(f"Could not load spaCy model {model_name}, using blank model")
            self.nlp = spacy.blank("en")
        
        self.matcher = Matcher(self.nlp.vocab)
        self._setup_conversation_patterns()
        
        logger.info(f"SpacyEntityExtractor initialized with model: {model_name}")
    
    def _setup_conversation_patterns(self) -> None:
        """Setup patterns specific to agent conversations."""
        # Agent name patterns
        agent_patterns = [
            [{"LOWER": {"IN": ["claude", "gpt", "assistant", "chatgpt", "ai"]}}],
            [{"TEXT": {"IN": ["Claude", "GPT", "ChatGPT", "AI", "Assistant"]}}],
        ]
        
        # Technology patterns
        tech_patterns = [
            [{"LOWER": {"IN": ["python", "javascript", "java", "rust", "go", "typescript"]}}],
            [{"TEXT": {"IN": ["Python", "JavaScript", "Java", "TypeScript", "C++", "C#"]}}],
            [{"LOWER": {"IN": ["react", "vue", "angular", "django", "flask", "fastapi"]}}],
            [{"TEXT": {"IN": ["React", "Vue", "Angular", "Django", "Flask", "FastAPI"]}}],
            [{"LOWER": {"IN": ["postgresql", "mysql", "mongodb", "redis", "sqlite"]}}],
            [{"TEXT": {"IN": ["PostgreSQL", "MySQL", "MongoDB", "Redis", "SQLite"]}}],
        ]
        
        # Concept patterns
        concept_patterns = [
            [{"LOWER": "machine"}, {"LOWER": "learning"}],
            [{"LOWER": "artificial"}, {"LOWER": "intelligence"}],
            [{"LOWER": "deep"}, {"LOWER": "learning"}],
            [{"LOWER": "neural"}, {"LOWER": {"IN": ["network", "networks"]}}],
            [{"LOWER": "active"}, {"LOWER": "inference"}],
            [{"TEXT": {"IN": ["API", "REST", "GraphQL", "WebSocket"]}}],
        ]
        
        # Task patterns
        task_patterns = [
            [{"LEMMA": {"IN": ["build", "create", "develop", "implement", "design"]}}],
            [{"LEMMA": {"IN": ["fix", "debug", "solve", "resolve", "troubleshoot"]}}],
            [{"LEMMA": {"IN": ["optimize", "improve", "refactor", "enhance"]}}],
        ]
        
        # Goal patterns
        goal_patterns = [
            [{"LOWER": {"IN": ["goal", "objective", "target", "aim", "purpose"]}}],
            [{"TEXT": {"REGEX": r"(?i)need to|want to|should|must"}}],
        ]
        
        # Add patterns to matcher
        self.matcher.add("AGENT", agent_patterns)
        self.matcher.add("TECHNOLOGY", tech_patterns)
        self.matcher.add("CONCEPT", concept_patterns) 
        self.matcher.add("TASK", task_patterns)
        self.matcher.add("GOAL", goal_patterns)
    
    def extract_entities(self, context: ExtractionContext) -> List[ConversationEntity]:
        """Extract entities from conversation context."""
        start_time = time.time()
        entities = []
        
        if not context.current_message.content.strip():
            return entities
        
        try:
            # Process message with spaCy
            doc = self.nlp(context.current_message.content)
            
            # Extract custom pattern matches
            entities.extend(self._extract_pattern_entities(doc, context))
            
            # Extract spaCy named entities
            entities.extend(self._extract_spacy_entities(doc, context))
            
            # Extract agent entities from context
            entities.extend(self._extract_agent_entities(context))
            
            # Deduplicate entities
            entities = self._deduplicate_entities(entities)
            
            processing_time = time.time() - start_time
            logger.debug(f"Extracted {len(entities)} entities in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
            # Return empty list on error
            entities = []
        
        return entities
    
    def _extract_pattern_entities(
        self, 
        doc: Doc, 
        context: ExtractionContext
    ) -> List[ConversationEntity]:
        """Extract entities using custom patterns."""
        entities = []
        matches = self.matcher(doc)
        
        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            
            # Map pattern label to entity type
            entity_type = self._map_pattern_to_entity_type(label, span.text)
            if not entity_type:
                continue
            
            entity = ConversationEntity(
                entity_type=entity_type,
                label=span.text,
                properties={
                    "text": span.text,
                    "pattern_label": label,
                    "start_pos": span.start_char,
                    "end_pos": span.end_char,
                },
                temporal_metadata=TemporalMetadata(
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                    conversation_id=context.current_message.conversation_id,
                    message_id=context.current_message.message_id,
                ),
                provenance=Provenance(
                    source_type="conversation_message",
                    source_id=context.current_message.message_id,
                    extraction_method="spacy_pattern",
                    confidence_score=0.85,  # High confidence for pattern matches
                    agent_id=context.current_message.agent_id,
                ),
            )
            entities.append(entity)
        
        return entities
    
    def _extract_spacy_entities(
        self, 
        doc: Doc, 
        context: ExtractionContext
    ) -> List[ConversationEntity]:
        """Extract entities using spaCy NER."""
        entities = []
        
        for ent in doc.ents:
            entity_type = self._map_spacy_label_to_entity_type(ent.label_)
            if not entity_type:
                continue
            
            confidence = self._calculate_spacy_confidence(ent)
            
            entity = ConversationEntity(
                entity_type=entity_type,
                label=ent.text,
                properties={
                    "text": ent.text,
                    "spacy_label": ent.label_,
                    "start_pos": ent.start_char,
                    "end_pos": ent.end_char,
                },
                temporal_metadata=TemporalMetadata(
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                    conversation_id=context.current_message.conversation_id,
                    message_id=context.current_message.message_id,
                ),
                provenance=Provenance(
                    source_type="conversation_message",
                    source_id=context.current_message.message_id,
                    extraction_method="spacy_ner",
                    confidence_score=confidence,
                    agent_id=context.current_message.agent_id,
                ),
            )
            entities.append(entity)
        
        return entities
    
    def _extract_agent_entities(self, context: ExtractionContext) -> List[ConversationEntity]:
        """Extract agent entities from conversation context."""
        entities = []
        
        # Extract current message agent
        if context.current_message.agent_id:
            agent_name = context.current_message.agent_id
            role = context.agent_roles.get(agent_name, "unknown")
            
            entity = ConversationEntity(
                entity_type=EntityType.AGENT,
                label=agent_name,
                properties={
                    "agent_id": agent_name,
                    "name": agent_name,
                    "role": role,
                },
                temporal_metadata=TemporalMetadata(
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                    conversation_id=context.current_message.conversation_id,
                    message_id=context.current_message.message_id,
                ),
                provenance=Provenance(
                    source_type="conversation_context",
                    source_id=context.current_message.message_id,
                    extraction_method="context_analysis",
                    confidence_score=1.0,  # High confidence for explicit agents
                    agent_id=context.current_message.agent_id,
                ),
            )
            entities.append(entity)
        
        # Extract mentioned agents from content
        content_lower = context.current_message.content.lower()
        for agent_id, role in context.agent_roles.items():
            if agent_id.lower() in content_lower and agent_id != context.current_message.agent_id:
                entity = ConversationEntity(
                    entity_type=EntityType.AGENT,
                    label=agent_id,
                    properties={
                        "agent_id": agent_id,
                        "name": agent_id,
                        "role": role,
                        "mentioned": True,
                    },
                    temporal_metadata=TemporalMetadata(
                        created_at=datetime.now(),
                        last_updated=datetime.now(),
                        conversation_id=context.current_message.conversation_id,
                        message_id=context.current_message.message_id,
                    ),
                    provenance=Provenance(
                        source_type="conversation_message",
                        source_id=context.current_message.message_id,
                        extraction_method="mention_detection",
                        confidence_score=0.9,
                        agent_id=context.current_message.agent_id,
                    ),
                )
                entities.append(entity)
        
        return entities
    
    def _map_pattern_to_entity_type(self, pattern_label: str, text: str) -> Optional[EntityType]:
        """Map pattern label to entity type."""
        mapping = {
            "AGENT": EntityType.AGENT,
            "TECHNOLOGY": EntityType.CONCEPT,  # Map to CONCEPT for now
            "CONCEPT": EntityType.CONCEPT,
            "TASK": EntityType.TASK,
            "GOAL": EntityType.GOAL,
        }
        return mapping.get(pattern_label)
    
    def _map_spacy_label_to_entity_type(self, spacy_label: str) -> Optional[EntityType]:
        """Map spaCy entity label to our entity type."""
        mapping = {
            "PERSON": EntityType.AGENT,
            "ORG": EntityType.AGENT,  # Organizations can be agents
            "GPE": EntityType.CONTEXT,  # Geopolitical entities as context
            "LOC": EntityType.CONTEXT,
            "DATE": EntityType.CONTEXT,
            "TIME": EntityType.CONTEXT,
            "EVENT": EntityType.TASK,  # Events as tasks
            "PRODUCT": EntityType.CONCEPT,
            "WORK_OF_ART": EntityType.CONCEPT,
            "LAW": EntityType.CONSTRAINT,
            "LANGUAGE": EntityType.CONCEPT,
        }
        return mapping.get(spacy_label)
    
    def _calculate_spacy_confidence(self, ent: Span) -> float:
        """Calculate confidence score for spaCy entity."""
        base_confidence = 0.7
        
        # Boost confidence for longer entities
        length_bonus = min(0.2, len(ent.text.split()) * 0.05)
        
        # Boost confidence for capitalized entities
        cap_bonus = 0.1 if ent.text[0].isupper() else 0.0
        
        # Boost confidence for known high-confidence labels
        if ent.label_ in ["PERSON", "ORG", "GPE"]:
            base_confidence = 0.8
        
        return min(0.95, base_confidence + length_bonus + cap_bonus)
    
    def _deduplicate_entities(self, entities: List[ConversationEntity]) -> List[ConversationEntity]:
        """Remove duplicate entities, keeping highest confidence."""
        if not entities:
            return entities
        
        # Group by normalized text
        groups: Dict[str, List[ConversationEntity]] = {}
        for entity in entities:
            key = entity.label.lower().strip()
            if key not in groups:
                groups[key] = []
            groups[key].append(entity)
        
        # Keep highest confidence entity per group
        deduplicated = []
        for group in groups.values():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                best = max(group, key=lambda e: e.provenance.confidence_score if e.provenance else 0.0)
                deduplicated.append(best)
        
        return deduplicated


class PatternRelationExtractor(RelationExtractionStrategy):
    """Pattern-based relation extraction strategy."""
    
    def __init__(self):
        """Initialize pattern-based relation extractor."""
        self.patterns: List[Dict[str, Any]] = []
        self._setup_default_patterns()
        logger.info("PatternRelationExtractor initialized")
    
    def _setup_default_patterns(self) -> None:
        """Setup default relation patterns."""
        default_patterns = [
            {
                "pattern": r"(\w+)\s+requires?\s+(\w+)",
                "relation_type": RelationType.REQUIRES,
                "confidence": 0.8,
            },
            {
                "pattern": r"(\w+)\s+enables?\s+(\w+)",
                "relation_type": RelationType.ENABLES,
                "confidence": 0.8,
            },
            {
                "pattern": r"(\w+)\s+conflicts?\s+with\s+(\w+)",
                "relation_type": RelationType.CONFLICTS_WITH,
                "confidence": 0.8,
            },
            {
                "pattern": r"(\w+)\s+depends?\s+on\s+(\w+)",
                "relation_type": RelationType.DEPENDS_ON,
                "confidence": 0.8,
            },
            {
                "pattern": r"(\w+)\s+influences?\s+(\w+)",
                "relation_type": RelationType.INFLUENCES,
                "confidence": 0.7,
            },
            {
                "pattern": r"(\w+)\s+(?:is\s+)?built\s+with\s+(\w+)",
                "relation_type": RelationType.DEPENDS_ON,
                "confidence": 0.8,
            },
            {
                "pattern": r"(\w+)\s+uses?\s+(\w+)",
                "relation_type": RelationType.DEPENDS_ON,
                "confidence": 0.7,
            },
            {
                "pattern": r"(\w+)\s+supports?\s+(\w+)",
                "relation_type": RelationType.SUPPORTS,
                "confidence": 0.7,
            },
        ]
        
        self.patterns.extend(default_patterns)
    
    def add_pattern(self, pattern: Dict[str, Any]) -> None:
        """Add custom relation pattern."""
        required_keys = ["pattern", "relation_type", "confidence"]
        if all(key in pattern for key in required_keys):
            self.patterns.append(pattern)
            logger.debug(f"Added pattern: {pattern['pattern']}")
        else:
            logger.warning(f"Invalid pattern format: {pattern}")
    
    def extract_relations(
        self, 
        entities: List[ConversationEntity], 
        context: ExtractionContext
    ) -> List[ConversationRelation]:
        """Extract relations between entities using patterns."""
        if len(entities) < 2:
            return []
        
        relations = []
        content = context.current_message.content
        
        # Try each pattern
        for pattern_def in self.patterns:
            pattern = pattern_def["pattern"]
            relation_type = pattern_def["relation_type"]
            base_confidence = pattern_def["confidence"]
            
            matches = re.finditer(pattern, content, re.IGNORECASE)
            
            for match in matches:
                groups = match.groups()
                if len(groups) >= 2:
                    source_text = groups[0].strip()
                    target_text = groups[1].strip()
                    
                    # Find matching entities
                    source_entity = self._find_entity_by_text(entities, source_text)
                    target_entity = self._find_entity_by_text(entities, target_text)
                    
                    if source_entity and target_entity and source_entity != target_entity:
                        relation = ConversationRelation(
                            source_entity_id=source_entity.entity_id,
                            target_entity_id=target_entity.entity_id,
                            relation_type=relation_type,
                            properties={
                                "pattern": pattern,
                                "matched_text": match.group(0),
                                "source_text": source_text,
                                "target_text": target_text,
                            },
                            temporal_metadata=TemporalMetadata(
                                created_at=datetime.now(),
                                last_updated=datetime.now(),
                                conversation_id=context.current_message.conversation_id,
                                message_id=context.current_message.message_id,
                            ),
                            provenance=Provenance(
                                source_type="conversation_message",
                                source_id=context.current_message.message_id,
                                extraction_method="pattern_matching",
                                confidence_score=base_confidence,
                                agent_id=context.current_message.agent_id,
                            ),
                        )
                        relations.append(relation)
        
        # Also try simple adjacency-based relations
        relations.extend(self._extract_adjacency_relations(entities, context))
        
        return relations
    
    def _find_entity_by_text(
        self, 
        entities: List[ConversationEntity], 
        text: str
    ) -> Optional[ConversationEntity]:
        """Find entity by text match."""
        text_lower = text.lower()
        
        # Exact match first
        for entity in entities:
            if entity.label.lower() == text_lower:
                return entity
        
        # Partial match
        for entity in entities:
            if text_lower in entity.label.lower() or entity.label.lower() in text_lower:
                return entity
        
        return None
    
    def _extract_adjacency_relations(
        self, 
        entities: List[ConversationEntity], 
        context: ExtractionContext
    ) -> List[ConversationRelation]:
        """Extract relations based on entity adjacency in text."""
        relations = []
        
        if len(entities) < 2:
            return relations
        
        # Sort entities by text position
        positioned_entities = [
            e for e in entities 
            if e.properties.get("start_pos") is not None
        ]
        positioned_entities.sort(key=lambda e: e.properties["start_pos"])
        
        # Find adjacent entities
        for i in range(len(positioned_entities) - 1):
            entity1 = positioned_entities[i]
            entity2 = positioned_entities[i + 1]
            
            # Check if entities are close enough to be related
            start1 = entity1.properties.get("start_pos", 0)
            end1 = entity1.properties.get("end_pos", 0)
            start2 = entity2.properties.get("start_pos", 0)
            
            if start2 - end1 < 50:  # Within 50 characters
                # Infer relation type based on entity types
                relation_type = self._infer_relation_type(entity1, entity2)
                
                if relation_type:
                    relation = ConversationRelation(
                        source_entity_id=entity1.entity_id,
                        target_entity_id=entity2.entity_id,
                        relation_type=relation_type,
                        properties={
                            "extraction_method": "adjacency",
                            "distance": start2 - end1,
                        },
                        temporal_metadata=TemporalMetadata(
                            created_at=datetime.now(),
                            last_updated=datetime.now(),
                            conversation_id=context.current_message.conversation_id,
                            message_id=context.current_message.message_id,
                        ),
                        provenance=Provenance(
                            source_type="conversation_message",
                            source_id=context.current_message.message_id,
                            extraction_method="adjacency_analysis",
                            confidence_score=0.6,  # Lower confidence for adjacency
                            agent_id=context.current_message.agent_id,
                        ),
                    )
                    relations.append(relation)
        
        return relations
    
    def _infer_relation_type(
        self, 
        entity1: ConversationEntity, 
        entity2: ConversationEntity
    ) -> Optional[RelationType]:
        """Infer relation type based on entity types."""
        type1 = entity1.entity_type
        type2 = entity2.entity_type
        
        # Agent relations
        if type1 == EntityType.AGENT and type2 == EntityType.TASK:
            return RelationType.MENTIONS
        elif type1 == EntityType.AGENT and type2 == EntityType.GOAL:
            return RelationType.MENTIONS
        
        # Task/Goal relations
        elif type1 == EntityType.TASK and type2 == EntityType.CONCEPT:
            return RelationType.RELATES_TO
        elif type1 == EntityType.GOAL and type2 == EntityType.CONCEPT:
            return RelationType.RELATES_TO
        
        # Concept relations
        elif type1 == EntityType.CONCEPT and type2 == EntityType.CONCEPT:
            return RelationType.RELATES_TO
        
        return None


class CoReferenceResolver:
    """Resolves co-references across conversation messages."""
    
    def __init__(self):
        """Initialize co-reference resolver."""
        self.entity_memory: Dict[str, List[ConversationEntity]] = {}
        self.pronoun_patterns = {
            "it", "this", "that", "they", "them", "these", "those"
        }
        logger.info("CoReferenceResolver initialized")
    
    def resolve_references(self, context: ExtractionContext) -> List[ConversationEntity]:
        """Resolve co-references in current message."""
        resolved_entities = []
        content = context.current_message.content.lower()
        
        # Update entity memory with conversation history
        self.update_context(context)
        
        # Look for pronouns and resolve them
        words = content.split()
        for i, word in enumerate(words):
            word_clean = word.strip(".,!?").lower()
            
            if word_clean in self.pronoun_patterns:
                # Try to resolve pronoun to recent entity
                resolved_entity = self._resolve_pronoun(
                    word_clean, 
                    context, 
                    position=i
                )
                if resolved_entity:
                    resolved_entities.append(resolved_entity)
        
        return resolved_entities
    
    def update_context(self, context: ExtractionContext) -> None:
        """Update entity memory with conversation context."""
        conv_id = context.current_message.conversation_id
        
        if conv_id not in self.entity_memory:
            self.entity_memory[conv_id] = []
        
        # Extract entities from all messages in context
        all_messages = context.conversation_history + [context.current_message]
        
        for message in all_messages[-5:]:  # Keep last 5 messages
            # Simple entity extraction for memory
            content = message.content
            entities = self._extract_simple_entities(message, context)
            
            # Add to memory
            for entity in entities:
                if not any(e.label.lower() == entity.label.lower() 
                          for e in self.entity_memory[conv_id]):
                    self.entity_memory[conv_id].append(entity)
    
    def _extract_simple_entities(
        self, 
        message: ConversationMessage, 
        context: ExtractionContext
    ) -> List[ConversationEntity]:
        """Simple entity extraction for memory building."""
        entities = []
        content = message.content
        
        # Look for capitalized words (likely entities)
        words = re.findall(r'\b[A-Z][a-zA-Z]+\b', content)
        
        for word in set(words):  # Remove duplicates
            if len(word) > 2:  # Skip short words
                entity = ConversationEntity(
                    entity_type=EntityType.CONCEPT,  # Default type
                    label=word,
                    temporal_metadata=TemporalMetadata(
                        created_at=message.timestamp,
                        last_updated=message.timestamp,
                        conversation_id=message.conversation_id,
                        message_id=message.message_id,
                    ),
                    provenance=Provenance(
                        source_type="conversation_message",
                        source_id=message.message_id,
                        extraction_method="coreference_memory",
                        confidence_score=0.5,
                        agent_id=message.agent_id,
                    ),
                )
                entities.append(entity)
        
        return entities
    
    def _resolve_pronoun(
        self, 
        pronoun: str, 
        context: ExtractionContext, 
        position: int
    ) -> Optional[ConversationEntity]:
        """Resolve pronoun to most likely entity."""
        conv_id = context.current_message.conversation_id
        
        if conv_id not in self.entity_memory or not self.entity_memory[conv_id]:
            return None
        
        # Get most recent entities (simple recency heuristic)
        recent_entities = self.entity_memory[conv_id][-5:]
        
        if not recent_entities:
            return None
        
        # For now, return most recent entity
        # In a more sophisticated system, would consider:
        # - Gender agreement
        # - Number agreement  
        # - Semantic similarity
        # - Syntactic context
        
        most_recent = recent_entities[-1]
        
        # Create resolved entity with updated metadata
        resolved = ConversationEntity(
            entity_type=most_recent.entity_type,
            label=most_recent.label,
            properties={
                **most_recent.properties,
                "resolved_from": pronoun,
                "resolution_position": position,
            },
            temporal_metadata=TemporalMetadata(
                created_at=datetime.now(),
                last_updated=datetime.now(),
                conversation_id=context.current_message.conversation_id,
                message_id=context.current_message.message_id,
            ),
            provenance=Provenance(
                source_type="coreference_resolution",
                source_id=context.current_message.message_id,
                extraction_method="pronoun_resolution",
                confidence_score=0.6,  # Medium confidence for resolution
                agent_id=context.current_message.agent_id,
            ),
        )
        
        return resolved


class ContextAwareExtractor:
    """Context-aware extraction that considers conversation history."""
    
    def __init__(self):
        """Initialize context-aware extractor."""
        self.domain_patterns: Dict[str, List[str]] = {
            "software_development": [
                "code", "programming", "development", "software", "application",
                "framework", "library", "api", "database", "server", "client"
            ],
            "machine_learning": [
                "model", "training", "data", "algorithm", "neural", "network",
                "prediction", "classification", "regression", "clustering"
            ],
            "project_management": [
                "project", "task", "deadline", "milestone", "requirement",
                "stakeholder", "resource", "timeline", "deliverable"
            ],
        }
        logger.info("ContextAwareExtractor initialized")
    
    def extract_with_context(self, context: ExtractionContext) -> List[ConversationEntity]:
        """Extract entities considering conversation context."""
        base_extractor = SpacyEntityExtractor()
        base_entities = base_extractor.extract_entities(context)
        
        # Enhance entities with context awareness
        enhanced_entities = self._enhance_with_context(base_entities, context)
        
        # Add context-specific entities
        context_entities = self._extract_context_entities(context)
        
        # Combine and deduplicate
        all_entities = enhanced_entities + context_entities
        return self._deduplicate_entities(all_entities)
    
    def update_conversation_context(self, context: ExtractionContext) -> None:
        """Update internal conversation context state."""
        # Could maintain more sophisticated context state here
        pass
    
    def _enhance_with_context(
        self, 
        entities: List[ConversationEntity], 
        context: ExtractionContext
    ) -> List[ConversationEntity]:
        """Enhance entities with context information."""
        enhanced = []
        
        for entity in entities:
            # Add context-based properties
            enhanced_properties = {**entity.properties}
            
            # Add domain context
            if context.domain_context:
                enhanced_properties["domain_context"] = context.domain_context
                
                # Adjust confidence based on domain relevance
                if entity.provenance and self._is_domain_relevant(entity, context.domain_context):
                    # Boost confidence for domain-relevant entities
                    entity.provenance.confidence_score = min(
                        1.0, 
                        entity.provenance.confidence_score + 0.1
                    )
            
            # Add agent role context
            agent_id = context.current_message.agent_id
            if agent_id in context.agent_roles:
                enhanced_properties["speaker_role"] = context.agent_roles[agent_id]
            
            # Add conversation context
            if context.conversation_history:
                enhanced_properties["conversation_length"] = len(context.conversation_history)
                enhanced_properties["has_context"] = True
            
            # Create enhanced entity
            enhanced_entity = ConversationEntity(
                entity_id=entity.entity_id,
                entity_type=entity.entity_type,
                label=entity.label,
                properties=enhanced_properties,
                temporal_metadata=entity.temporal_metadata,
                provenance=entity.provenance,
            )
            enhanced.append(enhanced_entity)
        
        return enhanced
    
    def _extract_context_entities(self, context: ExtractionContext) -> List[ConversationEntity]:
        """Extract additional entities based on context."""
        entities = []
        
        # Extract domain-specific entities
        if context.domain_context:
            domain_entities = self._extract_domain_entities(context)
            entities.extend(domain_entities)
        
        # Extract conversation-specific entities
        conv_entities = self._extract_conversation_entities(context)
        entities.extend(conv_entities)
        
        return entities
    
    def _extract_domain_entities(self, context: ExtractionContext) -> List[ConversationEntity]:
        """Extract entities specific to domain context."""
        entities = []
        domain = context.domain_context
        
        if domain not in self.domain_patterns:
            return entities
        
        content = context.current_message.content.lower()
        domain_terms = self.domain_patterns[domain]
        
        for term in domain_terms:
            if term in content:
                entity = ConversationEntity(
                    entity_type=EntityType.CONCEPT,
                    label=term,
                    properties={
                        "domain": domain,
                        "context_extracted": True,
                    },
                    temporal_metadata=TemporalMetadata(
                        created_at=datetime.now(),
                        last_updated=datetime.now(),
                        conversation_id=context.current_message.conversation_id,
                        message_id=context.current_message.message_id,
                    ),
                    provenance=Provenance(
                        source_type="domain_context",
                        source_id=context.current_message.message_id,
                        extraction_method="domain_patterns",
                        confidence_score=0.7,
                        agent_id=context.current_message.agent_id,
                    ),
                )
                entities.append(entity)
        
        return entities
    
    def _extract_conversation_entities(self, context: ExtractionContext) -> List[ConversationEntity]:
        """Extract entities based on conversation patterns."""
        entities = []
        
        # Look for conversation-specific patterns
        content = context.current_message.content
        
        # Questions often indicate goals or tasks
        if "?" in content:
            question_entity = ConversationEntity(
                entity_type=EntityType.GOAL,
                label="information_seeking",
                properties={
                    "type": "question",
                    "content": content,
                },
                temporal_metadata=TemporalMetadata(
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                    conversation_id=context.current_message.conversation_id,
                    message_id=context.current_message.message_id,
                ),
                provenance=Provenance(
                    source_type="conversation_pattern",
                    source_id=context.current_message.message_id,
                    extraction_method="question_detection",
                    confidence_score=0.8,
                    agent_id=context.current_message.agent_id,
                ),
            )
            entities.append(question_entity)
        
        return entities
    
    def _is_domain_relevant(self, entity: ConversationEntity, domain: str) -> bool:
        """Check if entity is relevant to domain."""
        if domain not in self.domain_patterns:
            return False
        
        domain_terms = self.domain_patterns[domain]
        entity_text = entity.label.lower()
        
        return any(term in entity_text for term in domain_terms)
    
    def _deduplicate_entities(self, entities: List[ConversationEntity]) -> List[ConversationEntity]:
        """Remove duplicate entities."""
        seen = set()
        deduplicated = []
        
        for entity in entities:
            key = (entity.entity_type, entity.label.lower())
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)
        
        return deduplicated


class ConfidenceScorer:
    """Scores confidence for extracted entities and relations."""
    
    def __init__(self):
        """Initialize confidence scorer."""
        self.entity_type_weights = {
            EntityType.AGENT: 1.0,
            EntityType.CONCEPT: 0.8,
            EntityType.TASK: 0.9,
            EntityType.GOAL: 0.9,
            EntityType.CONSTRAINT: 0.7,
            EntityType.BELIEF: 0.6,
            EntityType.CONTEXT: 0.5,
            EntityType.OBSERVATION: 0.7,
            EntityType.DECISION: 0.8,
            EntityType.OUTCOME: 0.8,
        }
        logger.info("ConfidenceScorer initialized")
    
    def score_entity(
        self, 
        entity: ConversationEntity, 
        context_factors: Dict[str, Any]
    ) -> float:
        """Score confidence for an entity."""
        base_score = entity.provenance.confidence_score if entity.provenance else 0.5
        
        # Apply entity type weighting
        type_weight = self.entity_type_weights.get(entity.entity_type, 0.5)
        
        # Apply context factors
        context_boost = 0.0
        if "extraction_method" in context_factors:
            method = context_factors["extraction_method"]
            if method == "spacy_pattern":
                context_boost += 0.1
            elif method == "spacy_ner":
                context_boost += 0.05
        
        if "context_support" in context_factors:
            context_boost += context_factors["context_support"] * 0.1
        
        if "entity_frequency" in context_factors:
            freq = context_factors["entity_frequency"]
            context_boost += min(0.1, freq * 0.05)
        
        # Calculate final score
        final_score = (base_score * type_weight) + context_boost
        return min(1.0, max(0.0, final_score))
    
    def score_relation(
        self, 
        relation: ConversationRelation, 
        context_factors: Dict[str, Any]
    ) -> float:
        """Score confidence for a relation."""
        base_score = relation.provenance.confidence_score if relation.provenance else 0.5
        
        # Apply context factors
        context_boost = 0.0
        
        if "pattern_strength" in context_factors:
            context_boost += context_factors["pattern_strength"] * 0.1
        
        if "entity_confidence" in context_factors:
            entity_conf = context_factors["entity_confidence"]
            context_boost += entity_conf * 0.15
        
        if "syntactic_support" in context_factors:
            syntax_support = context_factors["syntactic_support"]
            context_boost += syntax_support * 0.1
        
        # Calculate final score
        final_score = base_score + context_boost
        return min(1.0, max(0.0, final_score))
    
    def aggregate_scores(
        self, 
        scores: List[float], 
        method: str = "average", 
        weights: Optional[List[float]] = None
    ) -> float:
        """Aggregate multiple confidence scores."""
        if not scores:
            return 0.0
        
        if method == "average":
            return sum(scores) / len(scores)
        elif method == "max":
            return max(scores)
        elif method == "min":
            return min(scores)
        elif method == "weighted" and weights:
            if len(scores) != len(weights):
                raise ValueError("Scores and weights must have same length")
            weighted_sum = sum(s * w for s, w in zip(scores, weights))
            weight_sum = sum(weights)
            return weighted_sum / weight_sum if weight_sum > 0 else 0.0
        else:
            # Default to average
            return sum(scores) / len(scores)


class LLMFallbackExtractor:
    """LLM-based fallback extraction for complex cases."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        """Initialize LLM fallback extractor."""
        self.model_name = model_name
        self.api_key = api_key
        # Note: Actual LLM integration would require OpenAI or similar client
        logger.info(f"LLMFallbackExtractor initialized with model: {model_name}")
    
    def extract_with_llm(self, context: ExtractionContext) -> List[ConversationEntity]:
        """Extract entities using LLM (mock implementation)."""
        # This is a mock implementation for testing
        # In real implementation, would call LLM API
        entities = []
        
        content = context.current_message.content
        
        # Mock extraction of complex entities
        if "active inference" in content.lower():
            entity = ConversationEntity(
                entity_type=EntityType.CONCEPT,
                label="active inference",
                properties={
                    "complexity": "high",
                    "domain": "machine_learning",
                },
                temporal_metadata=TemporalMetadata(
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                    conversation_id=context.current_message.conversation_id,
                    message_id=context.current_message.message_id,
                ),
                provenance=Provenance(
                    source_type="llm_extraction",
                    source_id=context.current_message.message_id,
                    extraction_method="llm_fallback",
                    confidence_score=0.75,
                    agent_id=context.current_message.agent_id,
                ),
            )
            entities.append(entity)
        
        if "autonomous agents" in content.lower():
            entity = ConversationEntity(
                entity_type=EntityType.AGENT,
                label="autonomous agents",
                properties={
                    "category": "artificial_agents",
                    "complexity": "high",
                },
                temporal_metadata=TemporalMetadata(
                    created_at=datetime.now(),
                    last_updated=datetime.now(),
                    conversation_id=context.current_message.conversation_id,
                    message_id=context.current_message.message_id,
                ),
                provenance=Provenance(
                    source_type="llm_extraction",
                    source_id=context.current_message.message_id,
                    extraction_method="llm_fallback",
                    confidence_score=0.8,
                    agent_id=context.current_message.agent_id,
                ),
            )
            entities.append(entity)
        
        return entities


class ExtractionPipeline:
    """Complete extraction pipeline orchestrating multiple strategies."""
    
    def __init__(
        self,
        entity_strategies: Optional[List[EntityExtractionStrategy]] = None,
        relation_strategies: Optional[List[RelationExtractionStrategy]] = None,
        confidence_scorer: Optional[ConfidenceScorer] = None,
        coreference_resolver: Optional[CoReferenceResolver] = None,
        fallback_extractor: Optional[LLMFallbackExtractor] = None,
    ):
        """Initialize extraction pipeline."""
        self.entity_strategies = entity_strategies or [SpacyEntityExtractor()]
        self.relation_strategies = relation_strategies or [PatternRelationExtractor()]
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()
        self.coreference_resolver = coreference_resolver
        self.fallback_extractor = fallback_extractor
        
        logger.info(f"ExtractionPipeline initialized with {len(self.entity_strategies)} "
                   f"entity strategies and {len(self.relation_strategies)} relation strategies")
    
    def extract(self, context: ExtractionContext) -> ExtractionResult:
        """Run complete extraction pipeline."""
        start_time = time.time()
        
        try:
            # 1. Extract entities using all strategies
            all_entities = []
            for strategy in self.entity_strategies:
                try:
                    entities = strategy.extract_entities(context)
                    all_entities.extend(entities)
                except Exception as e:
                    logger.error(f"Error in {strategy.get_strategy_name()}: {e}")
                    continue
            
            # 2. Resolve co-references if resolver available
            if self.coreference_resolver:
                try:
                    resolved_entities = self.coreference_resolver.resolve_references(context)
                    all_entities.extend(resolved_entities)
                except Exception as e:
                    logger.error(f"Error in coreference resolution: {e}")
            
            # 3. Deduplicate entities
            all_entities = self._deduplicate_entities(all_entities)
            
            # 4. Check if fallback needed
            if self._should_use_fallback(all_entities) and self.fallback_extractor:
                try:
                    fallback_entities = self.fallback_extractor.extract_with_llm(context)
                    all_entities.extend(fallback_entities)
                    all_entities = self._deduplicate_entities(all_entities)
                except Exception as e:
                    logger.error(f"Error in LLM fallback: {e}")
            
            # 5. Extract relations using all strategies
            all_relations = []
            for strategy in self.relation_strategies:
                try:
                    relations = strategy.extract_relations(all_entities, context)
                    all_relations.extend(relations)
                except Exception as e:
                    logger.error(f"Error in {strategy.get_strategy_name()}: {e}")
                    continue
            
            # 6. Score final confidence
            self._update_final_confidence(all_entities, all_relations, context)
            
            processing_time = time.time() - start_time
            
            # 7. Create result
            result = ExtractionResult(
                entities=all_entities,
                relations=all_relations,
                extraction_metadata={
                    "processing_time": processing_time,
                    "entity_strategies": [s.get_strategy_name() for s in self.entity_strategies],
                    "relation_strategies": [s.get_strategy_name() for s in self.relation_strategies],
                    "total_entities": len(all_entities),
                    "total_relations": len(all_relations),
                    "message_length": len(context.current_message.content),
                },
                processing_time=processing_time,
            )
            
            logger.debug(f"Extraction completed: {len(all_entities)} entities, "
                        f"{len(all_relations)} relations in {processing_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in extraction pipeline: {e}")
            # Return empty result on error
            return ExtractionResult(
                entities=[],
                relations=[],
                extraction_metadata={
                    "error": str(e),
                    "processing_time": time.time() - start_time,
                },
                processing_time=time.time() - start_time,
            )
    
    def _deduplicate_entities(self, entities: List[ConversationEntity]) -> List[ConversationEntity]:
        """Deduplicate entities across strategies."""
        if not entities:
            return entities
        
        # Group by label and type
        groups: Dict[Tuple[EntityType, str], List[ConversationEntity]] = {}
        
        for entity in entities:
            key = (entity.entity_type, entity.label.lower().strip())
            if key not in groups:
                groups[key] = []
            groups[key].append(entity)
        
        # Keep highest confidence entity per group
        deduplicated = []
        for group in groups.values():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                best = max(group, key=lambda e: e.provenance.confidence_score if e.provenance else 0.0)
                deduplicated.append(best)
        
        return deduplicated
    
    def _should_use_fallback(self, entities: List[ConversationEntity]) -> bool:
        """Determine if LLM fallback should be used."""
        if not entities:
            return True  # No entities found, try fallback
        
        # Check average confidence
        if entities:
            confidences = [
                e.provenance.confidence_score if e.provenance else 0.0 
                for e in entities
            ]
            avg_confidence = sum(confidences) / len(confidences)
            return avg_confidence < 0.6  # Low confidence threshold
        
        return False
    
    def _update_final_confidence(
        self, 
        entities: List[ConversationEntity], 
        relations: List[ConversationRelation],
        context: ExtractionContext
    ) -> None:
        """Update final confidence scores for all extracted elements."""
        if not self.confidence_scorer:
            return
        
        # Update entity confidence scores
        for entity in entities:
            if entity.provenance:
                context_factors = {
                    "extraction_method": entity.provenance.extraction_method,
                    "entity_frequency": 1.0,  # Could calculate actual frequency
                    "context_support": 0.8,   # Could calculate actual context support
                }
                
                new_confidence = self.confidence_scorer.score_entity(entity, context_factors)
                entity.provenance.confidence_score = new_confidence
        
        # Update relation confidence scores
        for relation in relations:
            if relation.provenance:
                # Calculate entity confidence for this relation
                entity_confidences = []
                for entity in entities:
                    if (entity.entity_id == relation.source_entity_id or 
                        entity.entity_id == relation.target_entity_id):
                        if entity.provenance:
                            entity_confidences.append(entity.provenance.confidence_score)
                
                avg_entity_confidence = (
                    sum(entity_confidences) / len(entity_confidences) 
                    if entity_confidences else 0.5
                )
                
                context_factors = {
                    "pattern_strength": 0.8,
                    "entity_confidence": avg_entity_confidence,
                    "syntactic_support": 0.7,
                }
                
                new_confidence = self.confidence_scorer.score_relation(relation, context_factors)
                relation.provenance.confidence_score = new_confidence


# Alias for backward compatibility
ConversationExtractor = ExtractionPipeline