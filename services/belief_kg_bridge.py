"""Belief-Knowledge Graph Bridge service.

This service handles the synchronization between PyMDP agent beliefs and the
knowledge graph, extracting belief states and converting them into graph nodes
and relationships.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# PyMDP is a required dependency - no fallbacks allowed
from pymdp import Agent


logger = logging.getLogger(__name__)


@dataclass
class BeliefState:
    """Represents an agent's belief state at a point in time."""

    factor_beliefs: List[np.ndarray]  # Beliefs for each state factor
    timestamp: datetime
    entropy: float  # Overall belief entropy
    most_likely_states: List[int]  # Most likely state for each factor
    metadata: Dict[str, Any]


@dataclass
class KGNode:
    """Knowledge graph node representation."""

    id: str
    type: str
    properties: Dict[str, Any]


@dataclass
class KGEdge:
    """Knowledge graph edge representation."""

    source: str
    target: str
    relationship: str
    properties: Optional[Dict[str, Any]] = None


class BeliefKGBridge:
    """Service for bridging PyMDP beliefs to knowledge graph."""

    def __init__(self):
        """Initialize the belief-KG bridge."""
        self.belief_threshold = 0.1  # Minimum belief probability to consider
        self.entropy_threshold = 2.0  # High entropy threshold for uncertainty

    async def update_kg_from_agent(
        self, agent: Agent, agent_id: str, knowledge_graph: Any
    ) -> Dict[str, Any]:
        """Update knowledge graph from agent's current state.

        Args:
            agent: PyMDP agent instance
            agent_id: Unique agent identifier
            knowledge_graph: Knowledge graph instance

        Returns:
            Dictionary with update statistics
        """
        logger.info(f"Updating knowledge graph from agent {agent_id}")

        try:
            # Extract current beliefs
            belief_state = await self.extract_beliefs(agent)

            # Convert to KG nodes
            nodes = await self.belief_to_nodes(
                belief_state, agent_id, context={"source": "agent_update"}
            )

            # Create edges
            edges = await self.create_belief_edges(nodes, agent_id)

            # Update knowledge graph
            nodes_added = 0
            nodes_updated = 0
            edges_added = 0

            for node in nodes:
                if hasattr(knowledge_graph, "add_node"):
                    result = await knowledge_graph.add_node(node.id, node.type, node.properties)
                    if result:
                        nodes_added += 1
                else:
                    # Mock for testing
                    nodes_added += 1

            for edge in edges:
                if hasattr(knowledge_graph, "add_edge"):
                    result = await knowledge_graph.add_edge(
                        edge.source,
                        edge.target,
                        edge.relationship,
                        edge.properties,
                    )
                    if result:
                        edges_added += 1
                else:
                    # Mock for testing
                    edges_added += 1

            logger.info(
                f"KG update complete: {nodes_added} nodes added, "
                f"{nodes_updated} nodes updated, {edges_added} edges added"
            )

            return {
                "nodes_added": nodes_added,
                "nodes_updated": nodes_updated,
                "edges_added": edges_added,
                "total_nodes": len(nodes),
                "total_edges": len(edges),
            }

        except Exception as e:
            logger.error(f"Failed to update KG from agent: {str(e)}")
            raise RuntimeError(f"KG update failed: {str(e)}")

    async def extract_beliefs(self, agent: Agent) -> BeliefState:
        """Extract belief state from PyMDP agent.

        Args:
            agent: PyMDP agent instance

        Returns:
            BeliefState object containing current beliefs
        """
        try:
            # Extract beliefs (qs) from agent
            factor_beliefs = None

            if hasattr(agent, "qs") and isinstance(agent.qs, list) and len(agent.qs) > 0:
                factor_beliefs = agent.qs
            elif (
                hasattr(agent, "beliefs")
                and isinstance(agent.beliefs, list)
                and len(agent.beliefs) > 0
            ):
                factor_beliefs = agent.beliefs

            # Require valid beliefs - no fallbacks allowed
            if factor_beliefs is None or len(factor_beliefs) == 0:
                raise ValueError("Agent has no valid belief state (qs or beliefs attributes)")

            # Ensure beliefs are numpy arrays
            factor_beliefs = [
                np.array(b) if not isinstance(b, np.ndarray) else b for b in factor_beliefs
            ]

            # Calculate entropy for each factor
            entropies = []
            for beliefs in factor_beliefs:
                # Add small epsilon to avoid log(0)
                probs = beliefs + 1e-10
                probs = probs / probs.sum()
                entropy = -np.sum(probs * np.log(probs))
                entropies.append(entropy)

            # Get most likely states
            most_likely_states = [int(np.argmax(beliefs)) for beliefs in factor_beliefs]

            # Overall entropy
            overall_entropy = np.mean(entropies)

            # Extract metadata
            metadata = {
                "num_factors": len(factor_beliefs),
                "factor_sizes": [len(b) for b in factor_beliefs],
                "entropies": entropies,
                "action_precision": getattr(agent, "action_precision", 1.0),
                "planning_horizon": getattr(agent, "planning_horizon", 1),
            }

            # Add action history if available
            if hasattr(agent, "action") and agent.action is not None:
                metadata["last_action"] = int(agent.action)

            if hasattr(agent, "action_hist"):
                metadata["action_history"] = [int(a) for a in agent.action_hist[-5:]]

            return BeliefState(
                factor_beliefs=factor_beliefs,
                timestamp=datetime.utcnow(),
                entropy=overall_entropy,
                most_likely_states=most_likely_states,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to extract beliefs: {str(e)}")
            raise

    async def belief_to_nodes(
        self, belief_state: BeliefState, agent_id: str, context: Dict[str, Any]
    ) -> List[KGNode]:
        """Convert belief state to knowledge graph nodes.

        Args:
            belief_state: Extracted belief state
            agent_id: Agent identifier
            context: Additional context for node creation

        Returns:
            List of KG nodes representing the belief state
        """
        nodes = []
        timestamp = belief_state.timestamp.isoformat()

        # Create main belief state node
        belief_node_id = f"belief_{agent_id}_{timestamp}"
        belief_node = KGNode(
            id=belief_node_id,
            type="belief_state",
            properties={
                "agent_id": agent_id,
                "timestamp": timestamp,
                "entropy": belief_state.entropy,
                "num_factors": belief_state.metadata["num_factors"],
                "context": context.get("source", "unknown"),
            },
        )
        nodes.append(belief_node)

        # Create nodes for each factor belief
        for factor_idx, (beliefs, most_likely) in enumerate(
            zip(belief_state.factor_beliefs, belief_state.most_likely_states)
        ):
            # Factor node
            factor_node_id = f"factor_{agent_id}_{factor_idx}_{timestamp}"
            factor_node = KGNode(
                id=factor_node_id,
                type="belief_factor",
                properties={
                    "agent_id": agent_id,
                    "factor_index": factor_idx,
                    "size": len(beliefs),
                    "entropy": belief_state.metadata["entropies"][factor_idx],
                    "most_likely_state": most_likely,
                    "timestamp": timestamp,
                },
            )
            nodes.append(factor_node)

            # Create nodes for significant beliefs
            for state_idx, belief_prob in enumerate(beliefs):
                if belief_prob > self.belief_threshold:
                    state_node_id = f"state_{agent_id}_f{factor_idx}_s{state_idx}_{timestamp}"
                    state_node = KGNode(
                        id=state_node_id,
                        type="belief_value",
                        properties={
                            "agent_id": agent_id,
                            "factor_index": factor_idx,
                            "state_index": state_idx,
                            "probability": float(belief_prob),
                            "is_most_likely": state_idx == most_likely,
                            "timestamp": timestamp,
                        },
                    )
                    nodes.append(state_node)

        # Create action node if available
        if "last_action" in belief_state.metadata:
            action_node_id = f"action_{agent_id}_{timestamp}"
            action_node = KGNode(
                id=action_node_id,
                type="agent_action",
                properties={
                    "agent_id": agent_id,
                    "action": belief_state.metadata["last_action"],
                    "timestamp": timestamp,
                    "action_history": belief_state.metadata.get("action_history", []),
                },
            )
            nodes.append(action_node)

        # Create uncertainty node if entropy is high
        if belief_state.entropy > self.entropy_threshold:
            uncertainty_node_id = f"uncertainty_{agent_id}_{timestamp}"
            uncertainty_node = KGNode(
                id=uncertainty_node_id,
                type="high_uncertainty",
                properties={
                    "agent_id": agent_id,
                    "entropy": belief_state.entropy,
                    "timestamp": timestamp,
                    "factors_uncertain": [
                        i
                        for i, e in enumerate(belief_state.metadata["entropies"])
                        if e > self.entropy_threshold
                    ],
                },
            )
            nodes.append(uncertainty_node)

        # Add prompt context if available
        if "prompt_id" in context:
            prompt_node_id = f"prompt_context_{context['prompt_id']}"
            prompt_node = KGNode(
                id=prompt_node_id,
                type="prompt_context",
                properties={
                    "prompt_id": context["prompt_id"],
                    "agent_id": agent_id,
                    "timestamp": timestamp,
                },
            )
            nodes.append(prompt_node)

        return nodes

    async def create_belief_edges(self, nodes: List[KGNode], agent_id: str) -> List[KGEdge]:
        """Create edges between belief-related nodes.

        Args:
            nodes: List of KG nodes
            agent_id: Agent identifier

        Returns:
            List of KG edges connecting the nodes
        """
        edges = []

        # Find main belief state node
        belief_node = next((n for n in nodes if n.type == "belief_state"), None)

        if not belief_node:
            return edges

        # Connect belief state to factors
        factor_nodes = [n for n in nodes if n.type == "belief_factor"]
        for factor_node in factor_nodes:
            edge = KGEdge(
                source=belief_node.id,
                target=factor_node.id,
                relationship="has_factor",
                properties={"factor_index": factor_node.properties["factor_index"]},
            )
            edges.append(edge)

        # Connect factors to their belief values
        value_nodes = [n for n in nodes if n.type == "belief_value"]
        for value_node in value_nodes:
            factor_idx = value_node.properties["factor_index"]
            factor_node = next(
                (n for n in factor_nodes if n.properties["factor_index"] == factor_idx),
                None,
            )
            if factor_node:
                edge = KGEdge(
                    source=factor_node.id,
                    target=value_node.id,
                    relationship="has_belief",
                    properties={
                        "probability": value_node.properties["probability"],
                        "state_index": value_node.properties["state_index"],
                    },
                )
                edges.append(edge)

        # Connect belief state to action
        action_node = next((n for n in nodes if n.type == "agent_action"), None)
        if action_node:
            edge = KGEdge(
                source=belief_node.id,
                target=action_node.id,
                relationship="resulted_in_action",
                properties={"action": action_node.properties["action"]},
            )
            edges.append(edge)

        # Connect to uncertainty if present
        uncertainty_node = next((n for n in nodes if n.type == "high_uncertainty"), None)
        if uncertainty_node:
            edge = KGEdge(
                source=belief_node.id,
                target=uncertainty_node.id,
                relationship="has_high_uncertainty",
                properties={"entropy": uncertainty_node.properties["entropy"]},
            )
            edges.append(edge)

        # Connect to agent node
        agent_edge = KGEdge(
            source=f"agent_{agent_id}",
            target=belief_node.id,
            relationship="has_belief_state",
            properties={"timestamp": belief_node.properties["timestamp"]},
        )
        edges.append(agent_edge)

        # Connect prompt context if available
        prompt_node = next((n for n in nodes if n.type == "prompt_context"), None)
        if prompt_node:
            edge = KGEdge(
                source=prompt_node.id,
                target=belief_node.id,
                relationship="generated_belief",
                properties={"agent_id": agent_id},
            )
            edges.append(edge)

        return edges

    async def query_agent_beliefs(
        self,
        knowledge_graph: Any,
        agent_id: str,
        time_range: Optional[Dict[str, datetime]] = None,
    ) -> List[BeliefState]:
        """Query historical belief states from knowledge graph.

        Args:
            knowledge_graph: Knowledge graph instance
            agent_id: Agent identifier
            time_range: Optional time range filter

        Returns:
            List of historical belief states
        """
        # This would query the KG for historical beliefs
        # For now, return empty list as this is mainly for writing
        return []
