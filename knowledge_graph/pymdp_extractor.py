"""PyMDP Knowledge Graph Extractor for Agent Inference Integration (Task 59.3).

This module implements specialized extractors for converting PyMDP agent inference results
into knowledge graph entities and relations. It handles belief states, policy sequences,
and confidence-weighted relationships following the established extraction pipeline patterns.

Designed according to Nemesis Committee guidance for clean separation of concerns,
comprehensive observability, and production-ready error handling.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from agents.inference_engine import InferenceResult
from knowledge_graph.extraction import (
    EntityExtractionStrategy,
    RelationExtractionStrategy,
    ExtractionContext,
    ConversationMessage,
)
from knowledge_graph.schema import (
    ConversationEntity,
    ConversationRelation,
    EntityType,
    RelationType,
    TemporalMetadata,
    Provenance,
)

logger = logging.getLogger(__name__)


@dataclass
class PyMDPExtractionContext:
    """Context for PyMDP-specific knowledge extraction."""
    
    inference_result: InferenceResult
    agent_id: str
    conversation_id: str
    message_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_extraction_context(self) -> ExtractionContext:
        """Convert to standard ExtractionContext for compatibility."""
        # Create a synthetic conversation message from inference result
        content = f"Agent {self.agent_id} performed inference with action {self.inference_result.action}"
        
        current_message = ConversationMessage(
            message_id=self.message_id,
            conversation_id=self.conversation_id,
            agent_id=self.agent_id,
            content=content,
            timestamp=self.timestamp,
            metadata=self.metadata
        )
        
        return ExtractionContext(
            conversation_history=[],
            current_message=current_message,
            agent_roles={self.agent_id: "pymdp_agent"},
            domain_context="active_inference",
            extraction_settings={"pymdp_enabled": True}
        )


class PyMDPEntityExtractor(EntityExtractionStrategy):
    """Extracts entities from PyMDP inference results."""
    
    def __init__(self, confidence_threshold: float = 0.1):
        """Initialize PyMDP entity extractor.
        
        Args:
            confidence_threshold: Minimum confidence for entity extraction
        """
        self.confidence_threshold = confidence_threshold
        logger.info(f"PyMDPEntityExtractor initialized with confidence threshold {confidence_threshold}")
    
    def extract_entities(self, context: ExtractionContext) -> List[ConversationEntity]:
        """Extract entities from PyMDP inference context."""
        if not hasattr(context, 'pymdp_context'):
            # Not a PyMDP context, return empty
            return []
        
        pymdp_context = context.pymdp_context
        entities = []
        
        try:
            # Extract belief state entities
            belief_entities = self._extract_belief_entities(pymdp_context)
            entities.extend(belief_entities)
            
            # Extract inference step entity
            inference_entity = self._extract_inference_entity(pymdp_context)
            if inference_entity:
                entities.append(inference_entity)
            
            # Extract policy sequence entity if available
            policy_entity = self._extract_policy_entity(pymdp_context)
            if policy_entity:
                entities.append(policy_entity)
            
            # Extract free energy landscape entity
            energy_entity = self._extract_energy_entity(pymdp_context)
            if energy_entity:
                entities.append(energy_entity)
            
            logger.debug(f"Extracted {len(entities)} PyMDP entities")
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting PyMDP entities: {e}")
            return []
    
    def _extract_belief_entities(self, context: PyMDPExtractionContext) -> List[ConversationEntity]:
        """Extract belief state entities from inference result."""
        entities = []
        beliefs = context.inference_result.beliefs
        
        if not beliefs or not beliefs.get('states'):
            return entities
        
        try:
            states = beliefs['states']
            
            # Handle different belief state formats
            if isinstance(states, list):
                # Multiple state factors
                for i, factor_beliefs in enumerate(states):
                    entity = self._create_belief_state_entity(
                        factor_beliefs, 
                        context, 
                        factor_index=i
                    )
                    if entity:
                        entities.append(entity)
            else:
                # Single state factor
                entity = self._create_belief_state_entity(states, context)
                if entity:
                    entities.append(entity)
                    
        except Exception as e:
            logger.warning(f"Failed to extract belief entities: {e}")
        
        return entities
    
    def _create_belief_state_entity(
        self, 
        belief_distribution: Union[List, NDArray], 
        context: PyMDPExtractionContext,
        factor_index: Optional[int] = None
    ) -> Optional[ConversationEntity]:
        """Create a belief state entity from distribution."""
        try:
            if isinstance(belief_distribution, np.ndarray):
                belief_distribution = belief_distribution.tolist()
            
            # Validate belief distribution
            if not belief_distribution or len(belief_distribution) == 0:
                return None
            
            # Calculate entropy for confidence
            belief_array = np.array(belief_distribution)
            if belief_array.size == 0:
                return None
                
            entropy = -np.sum(belief_array * np.log(belief_array + 1e-10))
            max_entropy = np.log(belief_array.size)
            confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.5
            
            # Skip low-confidence beliefs
            if confidence < self.confidence_threshold:
                return None
            
            # Find most likely state
            most_likely_state = int(np.argmax(belief_array))
            
            # Create label
            label = f"belief_state_factor_{factor_index}" if factor_index is not None else "belief_state"
            
            return ConversationEntity(
                entity_type=EntityType.BELIEF_STATE,
                label=label,
                properties={
                    "belief_state_description": f"Agent belief distribution with confidence {confidence:.3f}",
                    "distribution": belief_distribution,
                    "most_likely_state": most_likely_state,
                    "entropy": float(entropy),
                    "confidence": float(confidence),
                    "factor_index": factor_index,
                    "agent_id": context.agent_id,
                },
                temporal_metadata=TemporalMetadata(
                    created_at=context.timestamp,
                    last_updated=context.timestamp,
                    conversation_id=context.conversation_id,
                    message_id=context.message_id,
                ),
                provenance=Provenance(
                    source_type="pymdp_inference",
                    source_id=context.message_id,
                    extraction_method="belief_state_analysis",
                    confidence_score=confidence,
                    agent_id=context.agent_id,
                ),
            )
            
        except Exception as e:
            logger.warning(f"Failed to create belief state entity: {e}")
            return None
    
    def _extract_inference_entity(self, context: PyMDPExtractionContext) -> Optional[ConversationEntity]:
        """Extract inference step entity."""
        try:
            result = context.inference_result
            
            return ConversationEntity(
                entity_type=EntityType.INFERENCE_STEP,
                label=f"inference_{context.timestamp.isoformat()}",
                properties={
                    "inference_step_description": f"PyMDP inference step with action {result.action} and free energy {result.free_energy:.3f}",
                    "action": result.action,
                    "free_energy": result.free_energy,
                    "confidence": result.confidence,
                    "agent_id": context.agent_id,
                    "metadata": result.metadata,
                },
                temporal_metadata=TemporalMetadata(
                    created_at=context.timestamp,
                    last_updated=context.timestamp,
                    conversation_id=context.conversation_id,
                    message_id=context.message_id,
                ),
                provenance=Provenance(
                    source_type="pymdp_inference",
                    source_id=context.message_id,
                    extraction_method="inference_step_analysis",
                    confidence_score=result.confidence,
                    agent_id=context.agent_id,
                ),
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract inference entity: {e}")
            return None
    
    def _extract_policy_entity(self, context: PyMDPExtractionContext) -> Optional[ConversationEntity]:
        """Extract policy sequence entity if available."""
        try:
            metadata = context.inference_result.metadata
            policy_sequence = metadata.get('policy_sequence')
            
            if not policy_sequence:
                return None
            
            return ConversationEntity(
                entity_type=EntityType.POLICY_SEQUENCE,
                label=f"policy_{context.timestamp.isoformat()}",
                properties={
                    "policy_sequence_description": f"Policy sequence of length {len(policy_sequence) if isinstance(policy_sequence, list) else 1}",
                    "sequence": policy_sequence,
                    "length": len(policy_sequence) if isinstance(policy_sequence, list) else 1,
                    "agent_id": context.agent_id,
                    "planning_horizon": metadata.get('planning_horizon'),
                },
                temporal_metadata=TemporalMetadata(
                    created_at=context.timestamp,
                    last_updated=context.timestamp,
                    conversation_id=context.conversation_id,
                    message_id=context.message_id,
                ),
                provenance=Provenance(
                    source_type="pymdp_inference",
                    source_id=context.message_id,
                    extraction_method="policy_analysis",
                    confidence_score=context.inference_result.confidence,
                    agent_id=context.agent_id,
                ),
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract policy entity: {e}")
            return None
    
    def _extract_energy_entity(self, context: PyMDPExtractionContext) -> Optional[ConversationEntity]:
        """Extract free energy landscape entity."""
        try:
            free_energy = context.inference_result.free_energy
            
            if free_energy is None or np.isinf(free_energy):
                return None
            
            return ConversationEntity(
                entity_type=EntityType.FREE_ENERGY_LANDSCAPE,
                label=f"energy_{context.timestamp.isoformat()}",
                properties={
                    "free_energy_landscape_description": f"Free energy landscape with value {float(free_energy):.3f}",
                    "free_energy": float(free_energy),
                    "agent_id": context.agent_id,
                    "inference_time": context.inference_result.metadata.get('inference_time_ms', 0),
                },
                temporal_metadata=TemporalMetadata(
                    created_at=context.timestamp,
                    last_updated=context.timestamp,
                    conversation_id=context.conversation_id,
                    message_id=context.message_id,
                ),
                provenance=Provenance(
                    source_type="pymdp_inference",
                    source_id=context.message_id,
                    extraction_method="free_energy_analysis",
                    confidence_score=context.inference_result.confidence,
                    agent_id=context.agent_id,
                ),
            )
            
        except Exception as e:
            logger.warning(f"Failed to extract energy entity: {e}")
            return None


class PyMDPRelationExtractor(RelationExtractionStrategy):
    """Extracts relations from PyMDP inference results."""
    
    def __init__(self, temporal_window_ms: int = 1000):
        """Initialize PyMDP relation extractor.
        
        Args:
            temporal_window_ms: Time window for temporal relations (milliseconds)
        """
        self.temporal_window_ms = temporal_window_ms
        logger.info(f"PyMDPRelationExtractor initialized with {temporal_window_ms}ms temporal window")
    
    def extract_relations(
        self, 
        entities: List[ConversationEntity], 
        context: ExtractionContext
    ) -> List[ConversationRelation]:
        """Extract relations between PyMDP entities."""
        if not hasattr(context, 'pymdp_context'):
            return []
        
        relations = []
        
        try:
            # Filter PyMDP entities
            pymdp_entities = [
                e for e in entities 
                if e.entity_type in {
                    EntityType.BELIEF_STATE,
                    EntityType.INFERENCE_STEP,
                    EntityType.POLICY_SEQUENCE,
                    EntityType.FREE_ENERGY_LANDSCAPE
                }
            ]
            
            if len(pymdp_entities) < 2:
                return relations
            
            # Extract belief update relations
            belief_relations = self._extract_belief_relations(pymdp_entities, context)
            relations.extend(belief_relations)
            
            # Extract policy selection relations
            policy_relations = self._extract_policy_relations(pymdp_entities, context)
            relations.extend(policy_relations)
            
            # Extract temporal sequence relations
            temporal_relations = self._extract_temporal_relations(pymdp_entities, context)
            relations.extend(temporal_relations)
            
            # Extract confidence-weighted relations
            confidence_relations = self._extract_confidence_relations(pymdp_entities, context)
            relations.extend(confidence_relations)
            
            logger.debug(f"Extracted {len(relations)} PyMDP relations")
            return relations
            
        except Exception as e:
            logger.error(f"Error extracting PyMDP relations: {e}")
            return []
    
    def _extract_belief_relations(
        self, 
        entities: List[ConversationEntity], 
        context: ExtractionContext
    ) -> List[ConversationRelation]:
        """Extract belief update relations."""
        relations = []
        
        belief_entities = [e for e in entities if e.entity_type == EntityType.BELIEF_STATE]
        inference_entities = [e for e in entities if e.entity_type == EntityType.INFERENCE_STEP]
        
        # Connect belief states to inference steps
        for belief_entity in belief_entities:
            for inference_entity in inference_entities:
                relation = ConversationRelation(
                    source_entity_id=belief_entity.entity_id,
                    target_entity_id=inference_entity.entity_id,
                    relation_type=RelationType.BELIEF_UPDATE,
                    properties={
                        "belief_confidence": belief_entity.properties.get('confidence', 0.0),
                        "inference_confidence": inference_entity.properties.get('confidence', 0.0),
                    },
                    temporal_metadata=TemporalMetadata(
                        created_at=datetime.now(timezone.utc),
                        last_updated=datetime.now(timezone.utc),
                        conversation_id=context.current_message.conversation_id,
                        message_id=context.current_message.message_id,
                    ),
                    provenance=Provenance(
                        source_type="pymdp_inference",
                        source_id=context.current_message.message_id,
                        extraction_method="belief_update_analysis",
                        confidence_score=min(
                            belief_entity.properties.get('confidence', 0.0),
                            inference_entity.properties.get('confidence', 0.0)
                        ),
                        agent_id=context.current_message.agent_id,
                    ),
                )
                relations.append(relation)
        
        return relations
    
    def _extract_policy_relations(
        self, 
        entities: List[ConversationEntity], 
        context: ExtractionContext
    ) -> List[ConversationRelation]:
        """Extract policy selection relations."""
        relations = []
        
        policy_entities = [e for e in entities if e.entity_type == EntityType.POLICY_SEQUENCE]
        inference_entities = [e for e in entities if e.entity_type == EntityType.INFERENCE_STEP]
        
        # Connect policies to inference steps
        for policy_entity in policy_entities:
            for inference_entity in inference_entities:
                relation = ConversationRelation(
                    source_entity_id=policy_entity.entity_id,
                    target_entity_id=inference_entity.entity_id,
                    relation_type=RelationType.POLICY_SELECTION,
                    properties={
                        "policy_length": policy_entity.properties.get('length', 0),
                        "planning_horizon": policy_entity.properties.get('planning_horizon'),
                    },
                    temporal_metadata=TemporalMetadata(
                        created_at=datetime.now(timezone.utc),
                        last_updated=datetime.now(timezone.utc),
                        conversation_id=context.current_message.conversation_id,
                        message_id=context.current_message.message_id,
                    ),
                    provenance=Provenance(
                        source_type="pymdp_inference",
                        source_id=context.current_message.message_id,
                        extraction_method="policy_selection_analysis",
                        confidence_score=inference_entity.properties.get('confidence', 0.0),
                        agent_id=context.current_message.agent_id,
                    ),
                )
                relations.append(relation)
        
        return relations
    
    def _extract_temporal_relations(
        self, 
        entities: List[ConversationEntity], 
        context: ExtractionContext
    ) -> List[ConversationRelation]:
        """Extract temporal sequence relations between entities."""
        relations = []
        
        # Sort entities by timestamp
        temporal_entities = sorted(
            entities,
            key=lambda e: e.temporal_metadata.created_at if e.temporal_metadata else datetime.min
        )
        
        # Create temporal sequence relations
        for i in range(len(temporal_entities) - 1):
            current_entity = temporal_entities[i]
            next_entity = temporal_entities[i + 1]
            
            # Check if within temporal window
            if (current_entity.temporal_metadata and next_entity.temporal_metadata):
                time_diff = (next_entity.temporal_metadata.created_at - 
                           current_entity.temporal_metadata.created_at).total_seconds() * 1000
                
                if time_diff <= self.temporal_window_ms:
                    relation = ConversationRelation(
                        source_entity_id=current_entity.entity_id,
                        target_entity_id=next_entity.entity_id,
                        relation_type=RelationType.TEMPORAL_SEQUENCE,
                        properties={
                            "time_difference_ms": float(time_diff),
                            "sequence_index": i,
                        },
                        temporal_metadata=TemporalMetadata(
                            created_at=datetime.now(timezone.utc),
                            last_updated=datetime.now(timezone.utc),
                            conversation_id=context.current_message.conversation_id,
                            message_id=context.current_message.message_id,
                        ),
                        provenance=Provenance(
                            source_type="pymdp_inference",
                            source_id=context.current_message.message_id,
                            extraction_method="temporal_sequence_analysis",
                            confidence_score=0.8,  # High confidence for temporal relations
                            agent_id=context.current_message.agent_id,
                        ),
                    )
                    relations.append(relation)
        
        return relations
    
    def _extract_confidence_relations(
        self, 
        entities: List[ConversationEntity], 
        context: ExtractionContext
    ) -> List[ConversationRelation]:
        """Extract confidence-weighted relations between entities."""
        relations = []
        
        # Find high-confidence entities
        high_confidence_entities = [
            e for e in entities 
            if e.provenance and e.provenance.confidence_score > 0.7
        ]
        
        if len(high_confidence_entities) < 2:
            return relations
        
        # Create confidence-weighted relations
        for i, entity1 in enumerate(high_confidence_entities):
            for entity2 in high_confidence_entities[i+1:]:
                # Calculate combined confidence
                conf1 = entity1.provenance.confidence_score if entity1.provenance else 0.0
                conf2 = entity2.provenance.confidence_score if entity2.provenance else 0.0
                combined_confidence = (conf1 * conf2) ** 0.5  # Geometric mean
                
                relation = ConversationRelation(
                    source_entity_id=entity1.entity_id,
                    target_entity_id=entity2.entity_id,
                    relation_type=RelationType.CONFIDENCE_WEIGHTED,
                    properties={
                        "source_confidence": conf1,
                        "target_confidence": conf2,
                        "combined_confidence": combined_confidence,
                    },
                    temporal_metadata=TemporalMetadata(
                        created_at=datetime.now(timezone.utc),
                        last_updated=datetime.now(timezone.utc),
                        conversation_id=context.current_message.conversation_id,
                        message_id=context.current_message.message_id,
                    ),
                    provenance=Provenance(
                        source_type="pymdp_inference",
                        source_id=context.current_message.message_id,
                        extraction_method="confidence_weighting",
                        confidence_score=combined_confidence,
                        agent_id=context.current_message.agent_id,
                    ),
                )
                relations.append(relation)
        
        return relations


class PyMDPKnowledgeExtractor:
    """Main orchestrator for PyMDP knowledge extraction."""
    
    def __init__(
        self,
        entity_extractor: Optional[PyMDPEntityExtractor] = None,
        relation_extractor: Optional[PyMDPRelationExtractor] = None,
    ):
        """Initialize PyMDP knowledge extractor.
        
        Args:
            entity_extractor: Custom entity extractor (uses default if None)
            relation_extractor: Custom relation extractor (uses default if None)
        """
        self.entity_extractor = entity_extractor or PyMDPEntityExtractor()
        self.relation_extractor = relation_extractor or PyMDPRelationExtractor()
        
        logger.info("PyMDPKnowledgeExtractor initialized")
    
    def extract_knowledge(
        self,
        inference_result: InferenceResult,
        agent_id: str,
        conversation_id: str,
        message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract knowledge from PyMDP inference result.
        
        Args:
            inference_result: Result from PyMDP inference
            agent_id: ID of the agent
            conversation_id: Conversation context ID
            message_id: Message ID (generated if None)
            
        Returns:
            Dictionary with extracted entities and relations
        """
        start_time = time.time()
        message_id = message_id or str(uuid4())
        
        try:
            # Create PyMDP extraction context
            pymdp_context = PyMDPExtractionContext(
                inference_result=inference_result,
                agent_id=agent_id,
                conversation_id=conversation_id,
                message_id=message_id,
            )
            
            # Convert to standard extraction context
            extraction_context = pymdp_context.to_extraction_context()
            
            # Add PyMDP context as attribute for extractors
            extraction_context.pymdp_context = pymdp_context
            
            # Extract entities
            entities = self.entity_extractor.extract_entities(extraction_context)
            
            # Extract relations
            relations = self.relation_extractor.extract_relations(entities, extraction_context)
            
            processing_time = time.time() - start_time
            
            result = {
                "entities": entities,
                "relations": relations,
                "metadata": {
                    "agent_id": agent_id,
                    "conversation_id": conversation_id,
                    "message_id": message_id,
                    "processing_time_seconds": processing_time,
                    "entity_count": len(entities),
                    "relation_count": len(relations),
                    "extraction_timestamp": datetime.now(timezone.utc).isoformat(),
                }
            }
            
            logger.info(
                f"Extracted PyMDP knowledge in {processing_time:.3f}s: "
                f"{len(entities)} entities, {len(relations)} relations"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to extract PyMDP knowledge: {e}", exc_info=True)
            return {
                "entities": [],
                "relations": [],
                "metadata": {
                    "error": str(e),
                    "processing_time_seconds": time.time() - start_time,
                }
            }