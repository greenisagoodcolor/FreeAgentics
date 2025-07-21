"""Knowledge Graph integration for agents."""

import logging
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

from knowledge_graph.graph_engine import (
    EdgeType,
    KnowledgeEdge,
    KnowledgeGraph,
    KnowledgeNode,
    NodeType,
)
from knowledge_graph.storage import FileStorageBackend, StorageManager

logger = logging.getLogger(__name__)


class AgentKnowledgeGraphIntegration:
    """Integrates agent actions and observations with the knowledge graph."""
    
    def __init__(self, graph_id: Optional[str] = None):
        """Initialize KG integration."""
        self.graph_id = graph_id or str(uuid4())
        self.storage_manager = StorageManager(FileStorageBackend("./knowledge_graphs"))
        
        # Load or create graph
        try:
            self.graph = self.storage_manager.load(self.graph_id)
            logger.info(f"Loaded existing knowledge graph: {self.graph_id}")
        except:
            self.graph = KnowledgeGraph(graph_id=self.graph_id)
            logger.info(f"Created new knowledge graph: {self.graph_id}")
    
    def update_from_agent_step(
        self, 
        agent_id: str, 
        observation: Any, 
        action: str,
        beliefs: Optional[Dict[str, Any]] = None,
        free_energy: Optional[float] = None
    ) -> None:
        """Update knowledge graph from agent step.
        
        Args:
            agent_id: ID of the agent
            observation: Current observation
            action: Selected action
            beliefs: Current belief state
            free_energy: Free energy value
        """
        try:
            timestamp = datetime.now()
            
            # Create observation node
            obs_node = KnowledgeNode(
                node_type=NodeType.OBSERVATION,
                label=f"obs_{timestamp.isoformat()}",
                properties={
                    "agent_id": agent_id,
                    "observation": observation,
                    "timestamp": timestamp.isoformat()
                }
            )
            self.graph.add_node(obs_node)
            
            # Create action event node
            action_node = KnowledgeNode(
                node_type=NodeType.EVENT,
                label=f"action_{action}_{timestamp.isoformat()}",
                properties={
                    "agent_id": agent_id,
                    "action": action,
                    "timestamp": timestamp.isoformat(),
                    "free_energy": free_energy
                }
            )
            self.graph.add_node(action_node)
            
            # Link observation to action
            obs_action_edge = KnowledgeEdge(
                source_id=obs_node.node_id,
                target_id=action_node.node_id,
                edge_type=EdgeType.CAUSES,
                properties={"agent_id": agent_id}
            )
            self.graph.add_edge(obs_action_edge)
            
            # Add belief state if available
            if beliefs:
                belief_node = KnowledgeNode(
                    node_type=NodeType.BELIEF,
                    label=f"belief_{timestamp.isoformat()}",
                    properties={
                        "agent_id": agent_id,
                        "beliefs": beliefs,
                        "timestamp": timestamp.isoformat()
                    }
                )
                self.graph.add_node(belief_node)
                
                # Link belief to action
                belief_action_edge = KnowledgeEdge(
                    source_id=belief_node.node_id,
                    target_id=action_node.node_id,
                    edge_type=EdgeType.CAUSES,
                    properties={"agent_id": agent_id}
                )
                self.graph.add_edge(belief_action_edge)
            
            # Save graph periodically (every 10 updates)
            if len(self.graph.nodes) % 10 == 0:
                self.storage_manager.save(self.graph)
                logger.debug(f"Saved knowledge graph with {len(self.graph.nodes)} nodes")
                
        except Exception as e:
            logger.error(f"Failed to update knowledge graph: {e}")
    
    def get_agent_history(self, agent_id: str, limit: int = 100) -> Dict[str, Any]:
        """Get agent's action history from knowledge graph.
        
        Args:
            agent_id: ID of the agent
            limit: Maximum number of events to return
            
        Returns:
            Dictionary with agent history
        """
        try:
            # Find all nodes related to this agent
            agent_nodes = [
                node for node in self.graph.nodes.values()
                if node.properties.get("agent_id") == agent_id
            ]
            
            # Sort by timestamp
            agent_nodes.sort(
                key=lambda n: n.properties.get("timestamp", ""),
                reverse=True
            )
            
            # Build history
            history = {
                "agent_id": agent_id,
                "total_events": len(agent_nodes),
                "events": []
            }
            
            for node in agent_nodes[:limit]:
                event = {
                    "type": node.node_type.value,
                    "label": node.label,
                    "timestamp": node.properties.get("timestamp"),
                    "properties": node.properties
                }
                history["events"].append(event)
            
            return history
            
        except Exception as e:
            logger.error(f"Failed to get agent history: {e}")
            return {"agent_id": agent_id, "error": str(e)}
    
    def save(self) -> None:
        """Save the knowledge graph."""
        try:
            self.storage_manager.save(self.graph)
            logger.info(f"Saved knowledge graph {self.graph_id}")
        except Exception as e:
            logger.error(f"Failed to save knowledge graph: {e}")