"""
Knowledge Graph Auto-Updater.
Integrates conversation monitoring, NLP entity extraction, and entity-to-node mapping
to provide real-time bidirectional sync between conversations and knowledge graph
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from database.conversation_models import Conversation, Message
from knowledge_graph.conversation_monitoring import (
    ConversationEvent,
    ConversationEventType,
    ConversationMonitor,
)
from knowledge_graph.entity_node_mapper import EntityNodeMapper, GraphEngine
from knowledge_graph.nlp_entity_extractor import (
    ExtractionResult,
    NLPEntityExtractor,
)

logger = logging.getLogger(__name__)


class ConversationAutoUpdater:
    """
    Automatically updates knowledge graph from conversations using real NLP.
    Provides bidirectional sync between conversations and knowledge graph
    """

    def __init__(
        self,
        graph_engine: Optional[GraphEngine] = None,
        nlp_model: str = "en_core_web_sm",
    ):
        """Initialize the auto-updater with real NLP components."""
        self.graph_engine = graph_engine or GraphEngine()
        self.nlp_extractor = NLPEntityExtractor(model_name=nlp_model)
        self.entity_mapper = EntityNodeMapper(self.graph_engine)
        self.conversation_monitor = ConversationMonitor(self.graph_engine)

        # Subscribe to conversation events
        self.conversation_monitor.subscribe(self._handle_conversation_event)

        # Statistics
        self.stats = {
            "messages_processed": 0,
            "entities_extracted": 0,
            "nodes_created": 0,
            "relationships_created": 0,
            "errors": 0,
        }

        logger.info("Conversation Auto-Updater initialized with real NLP")

    async def start(self):
        """Start the auto-updater."""
        await self.conversation_monitor.start()
        logger.info("Conversation Auto-Updater started")

    async def stop(self):
        """Stop the auto-updater."""
        await self.conversation_monitor.stop()
        logger.info("Conversation Auto-Updater stopped")

    async def process_conversation(self, conversation: Conversation) -> Dict[str, Any]:
        """
        Process an entire conversation and update the knowledge graph.
        Returns statistics about the processing
        """
        start_time = datetime.utcnow()
        conversation_stats = {
            "conversation_id": str(conversation.id),
            "messages_processed": 0,
            "entities_extracted": 0,
            "nodes_created": 0,
            "relationships_created": 0,
            "processing_time": 0.0,
        }

        try:
            # Process all messages in the conversation
            for message in conversation.messages:
                await self._process_message(message, conversation_stats)

            # Calculate processing time
            end_time = datetime.utcnow()
            conversation_stats["processing_time"] = (
                end_time - start_time
            ).total_seconds()

            logger.info(
                f"Processed conversation {conversation.id}: {conversation_stats}"
            )
            return conversation_stats

        except Exception as e:
            logger.error(f"Error processing conversation {conversation.id}: {e}")
            conversation_stats["error"] = str(e)
            self.stats["errors"] += 1
            return conversation_stats

    async def process_message(self, message: Message) -> ExtractionResult:
        """
        Process a single message and update the knowledge graph.
        Returns the extraction result
        """
        try:
            # Extract entities from message
            extraction_result = self.nlp_extractor.extract_entities(
                text=message.content,
                metadata={
                    "message_id": str(message.id),
                    "conversation_id": str(message.conversation_id),
                    "role": message.role,
                    "timestamp": message.created_at.isoformat(),
                },
            )

            # Map entities to nodes
            if extraction_result.entities:
                entity_mappings = await self.entity_mapper.map_entities(
                    extraction_result.entities
                )

                # Map relationships to edges
                for relationship in extraction_result.relationships:
                    await self.entity_mapper.map_relationship(
                        relationship, entity_mappings
                    )
                    self.stats["relationships_created"] += 1

                self.stats["nodes_created"] += len(entity_mappings)

            # Update statistics
            self.stats["messages_processed"] += 1
            self.stats["entities_extracted"] += len(extraction_result.entities)

            logger.debug(
                f"Processed message {message.id}: found {len(extraction_result.entities)} entities"
            )
            return extraction_result

        except Exception as e:
            logger.error(f"Error processing message {message.id}: {e}")
            self.stats["errors"] += 1
            raise

    async def _handle_conversation_event(self, event: ConversationEvent):
        """Handle conversation events from the monitor."""
        try:
            if event.type == ConversationEventType.MESSAGE_ADDED:
                # Create a temporary Message object from the event
                # In a real implementation, you'd fetch the full message from the database
                message = Message(
                    id=event.message_id,
                    conversation_id=event.conversation_id,
                    content=event.content,
                    role="user",  # Default, should be determined from event metadata
                    created_at=event.timestamp,
                )
                await self.process_message(message)

        except Exception as e:
            logger.error(f"Error handling conversation event: {e}")
            self.stats["errors"] += 1

    async def _process_message(self, message: Message, stats: Dict[str, Any]):
        """Internal method to process a message and update stats."""
        try:
            extraction_result = await self.process_message(message)

            stats["messages_processed"] += 1
            stats["entities_extracted"] += len(extraction_result.entities)

        except Exception as e:
            logger.error(f"Error processing message {message.id}: {e}")
            stats["error"] = str(e)
            raise

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return dict(self.stats)

    def reset_statistics(self):
        """Reset processing statistics."""
        self.stats = {
            "messages_processed": 0,
            "entities_extracted": 0,
            "nodes_created": 0,
            "relationships_created": 0,
            "errors": 0,
        }

    async def query_knowledge_for_conversation(
        self, conversation_id: str
    ) -> List[Dict[str, Any]]:
        """
        Query the knowledge graph for entities related to a conversation.
        Provides the bidirectional sync - getting knowledge back to inform conversations
        """
        try:
            # Find all nodes that were created from this conversation
            related_nodes = []

            for node_id, node in self.graph_engine.graph.nodes.items():
                # Check if node was created from this conversation
                if (
                    node.properties.get("conversation_id") == conversation_id
                    or node.source == conversation_id
                ):
                    related_nodes.append(
                        {
                            "id": node_id,
                            "type": node.type.value,
                            "label": node.label,
                            "properties": node.properties,
                            "confidence": node.confidence,
                        }
                    )

            # Also find connected nodes (1-hop neighborhood)
            connected_nodes = []
            for node in related_nodes:
                for edge_id, edge in self.graph_engine.graph.edges.items():
                    if edge.source_id == node["id"]:
                        target_node = self.graph_engine.graph.nodes.get(edge.target_id)
                        if target_node:
                            connected_nodes.append(
                                {
                                    "id": target_node.id,
                                    "type": target_node.type.value,
                                    "label": target_node.label,
                                    "properties": target_node.properties,
                                    "relationship": edge.type.value,
                                    "confidence": target_node.confidence,
                                }
                            )

            return {
                "conversation_id": conversation_id,
                "direct_entities": related_nodes,
                "connected_entities": connected_nodes,
                "total_entities": len(related_nodes) + len(connected_nodes),
            }

        except Exception as e:
            logger.error(
                f"Error querying knowledge for conversation {conversation_id}: {e}"
            )
            return {"error": str(e)}

    async def suggest_conversation_context(
        self, conversation_id: str, current_message: str
    ) -> List[Dict[str, Any]]:
        """
        Suggest contextual information for a conversation based on the knowledge graph.
        This enables the bidirectional sync by using graph knowledge to enhance conversations
        """
        try:
            # Extract entities from current message
            extraction_result = self.nlp_extractor.extract_entities(current_message)

            # Find related knowledge
            suggestions = []

            for entity in extraction_result.entities:
                # Find existing nodes for this entity
                existing_nodes = await self.graph_engine.find_nodes_by_name(entity.text)

                for node in existing_nodes:
                    # Get connected information
                    connected_info = []
                    for edge_id, edge in self.graph_engine.graph.edges.items():
                        if edge.source_id == node.id:
                            target_node = self.graph_engine.graph.nodes.get(
                                edge.target_id
                            )
                            if target_node:
                                connected_info.append(
                                    {
                                        "entity": target_node.label,
                                        "relationship": edge.type.value,
                                        "confidence": target_node.confidence,
                                    }
                                )

                    if connected_info:
                        suggestions.append(
                            {
                                "entity": entity.text,
                                "type": entity.type.value,
                                "related_information": connected_info,
                                "suggestion_reason": "Found related entities in knowledge graph",
                            }
                        )

            return suggestions

        except Exception as e:
            logger.error(
                f"Error generating suggestions for conversation {conversation_id}: {e}"
            )
            return []

    async def export_conversation_knowledge(
        self, conversation_id: str
    ) -> Dict[str, Any]:
        """
        Export all knowledge derived from a specific conversation.
        Useful for debugging and understanding what was learned
        """
        try:
            knowledge = await self.query_knowledge_for_conversation(conversation_id)

            # Add metadata about the extraction process
            knowledge["extraction_metadata"] = {
                "nlp_model": self.nlp_extractor.model_name,
                "entity_types_detected": list(
                    set(
                        node["properties"].get("entity_type")
                        for node in knowledge.get("direct_entities", [])
                        if "entity_type" in node.get("properties", {})
                    )
                ),
                "extraction_timestamp": datetime.utcnow().isoformat(),
                "auto_updater_stats": self.get_statistics(),
            }

            return knowledge

        except Exception as e:
            logger.error(
                f"Error exporting knowledge for conversation {conversation_id}: {e}"
            )
            return {"error": str(e)}
