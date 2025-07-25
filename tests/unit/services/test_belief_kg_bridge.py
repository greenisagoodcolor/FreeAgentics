"""Test suite for Belief-KG Bridge service."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from services.belief_kg_bridge import (
    BeliefKGBridge,
    BeliefState,
)


class MockAgent:
    """Mock PyMDP agent for testing."""

    def __init__(self):
        # First belief: highly concentrated (low entropy)
        # Second belief: more spread out (high entropy)
        self.qs = [np.array([0.9, 0.05, 0.03, 0.02]), np.array([0.4, 0.3, 0.3])]
        self.action = 2
        self.action_hist = [0, 1, 2, 1, 0]
        self.action_precision = 1.5
        self.planning_horizon = 3


class TestBeliefKGBridge:
    """Test the belief-KG bridge service."""

    @pytest.fixture
    def bridge(self):
        """Create bridge instance."""
        return BeliefKGBridge()

    @pytest.fixture
    def mock_agent(self):
        """Create mock agent."""
        return MockAgent()

    @pytest.fixture
    def mock_knowledge_graph(self):
        """Create mock knowledge graph."""
        kg = AsyncMock()
        kg.add_node = AsyncMock(return_value=True)
        kg.add_edge = AsyncMock(return_value=True)
        return kg

    @pytest.mark.asyncio
    async def test_extract_beliefs(self, bridge, mock_agent):
        """Test belief extraction from agent."""
        belief_state = await bridge.extract_beliefs(mock_agent)

        assert isinstance(belief_state, BeliefState)
        assert len(belief_state.factor_beliefs) == 2
        assert belief_state.most_likely_states == [0, 0]  # Argmax of beliefs
        assert belief_state.metadata["num_factors"] == 2
        assert belief_state.metadata["last_action"] == 2
        assert belief_state.metadata["action_history"] == [0, 1, 2, 1, 0]
        assert isinstance(belief_state.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_extract_beliefs_entropy_calculation(self, bridge, mock_agent):
        """Test entropy calculation in belief extraction."""
        belief_state = await bridge.extract_beliefs(mock_agent)

        # Check that entropy is calculated
        assert belief_state.entropy > 0
        assert len(belief_state.metadata["entropies"]) == 2

        # First factor has lower entropy (more certain)
        assert belief_state.metadata["entropies"][0] < belief_state.metadata["entropies"][1]

    @pytest.mark.asyncio
    async def test_extract_beliefs_fallback(self, bridge):
        """Test belief extraction with minimal agent."""
        minimal_agent = MagicMock()
        # No qs or beliefs attribute

        belief_state = await bridge.extract_beliefs(minimal_agent)

        # Should use fallback beliefs
        assert len(belief_state.factor_beliefs) == 1
        assert np.allclose(belief_state.factor_beliefs[0], 0.25)

    @pytest.mark.asyncio
    async def test_belief_to_nodes(self, bridge, mock_agent):
        """Test conversion of beliefs to KG nodes."""
        belief_state = await bridge.extract_beliefs(mock_agent)
        agent_id = "test_agent_123"

        nodes = await bridge.belief_to_nodes(
            belief_state,
            agent_id,
            context={"source": "test", "prompt_id": "prompt_456"},
        )

        # Check node types
        node_types = [n.type for n in nodes]
        assert "belief_state" in node_types
        assert "belief_factor" in node_types
        assert "belief_value" in node_types
        assert "agent_action" in node_types
        assert "prompt_context" in node_types

        # Check belief state node
        belief_node = next(n for n in nodes if n.type == "belief_state")
        assert belief_node.properties["agent_id"] == agent_id
        assert belief_node.properties["entropy"] == belief_state.entropy

        # Check factor nodes
        factor_nodes = [n for n in nodes if n.type == "belief_factor"]
        assert len(factor_nodes) == 2  # Two factors

        # Check belief value nodes (only those above threshold)
        value_nodes = [n for n in nodes if n.type == "belief_value"]
        assert len(value_nodes) > 0
        for node in value_nodes:
            assert node.properties["probability"] > bridge.belief_threshold

    @pytest.mark.asyncio
    async def test_belief_to_nodes_high_uncertainty(self, bridge):
        """Test that high uncertainty creates special node."""
        # Create high entropy belief state
        belief_state = BeliefState(
            factor_beliefs=[np.ones(4) / 4],  # Uniform = high entropy
            timestamp=datetime.utcnow(),
            entropy=2.5,  # Above threshold
            most_likely_states=[0],
            metadata={
                "num_factors": 1,
                "entropies": [2.5],
                "factor_sizes": [4],
            },
        )

        nodes = await bridge.belief_to_nodes(belief_state, "agent_123", context={})

        # Should have uncertainty node
        uncertainty_nodes = [n for n in nodes if n.type == "high_uncertainty"]
        assert len(uncertainty_nodes) == 1
        assert uncertainty_nodes[0].properties["entropy"] == 2.5

    @pytest.mark.asyncio
    async def test_create_belief_edges(self, bridge, mock_agent):
        """Test edge creation between nodes."""
        belief_state = await bridge.extract_beliefs(mock_agent)
        nodes = await bridge.belief_to_nodes(belief_state, "test_agent", context={})

        edges = await bridge.create_belief_edges(nodes, "test_agent")

        # Check edge types
        edge_types = [e.relationship for e in edges]
        assert "has_factor" in edge_types
        assert "has_belief" in edge_types
        assert "resulted_in_action" in edge_types
        assert "has_belief_state" in edge_types

        # Verify connections
        belief_node = next(n for n in nodes if n.type == "belief_state")
        factor_edges = [e for e in edges if e.relationship == "has_factor"]
        assert all(e.source == belief_node.id for e in factor_edges)

    @pytest.mark.asyncio
    async def test_update_kg_from_agent(self, bridge, mock_agent, mock_knowledge_graph):
        """Test full KG update from agent."""
        agent_id = "test_agent_123"

        result = await bridge.update_kg_from_agent(mock_agent, agent_id, mock_knowledge_graph)

        assert result["nodes_added"] > 0
        assert result["edges_added"] > 0
        assert result["total_nodes"] > 0
        assert result["total_edges"] > 0

        # Verify KG methods were called
        assert mock_knowledge_graph.add_node.called
        assert mock_knowledge_graph.add_edge.called

    @pytest.mark.asyncio
    async def test_update_kg_error_handling(self, bridge, mock_agent):
        """Test error handling in KG update."""
        mock_kg = AsyncMock()
        mock_kg.add_node = AsyncMock(side_effect=Exception("KG error"))

        with pytest.raises(RuntimeError, match="KG update failed"):
            await bridge.update_kg_from_agent(mock_agent, "test_agent", mock_kg)

    @pytest.mark.asyncio
    async def test_belief_threshold_filtering(self, bridge):
        """Test that low probability beliefs are filtered."""
        belief_state = BeliefState(
            factor_beliefs=[np.array([0.95, 0.03, 0.01, 0.01])],
            timestamp=datetime.utcnow(),
            entropy=0.5,
            most_likely_states=[0],
            metadata={
                "num_factors": 1,
                "entropies": [0.5],
                "factor_sizes": [4],
            },
        )

        nodes = await bridge.belief_to_nodes(belief_state, "agent_123", context={})

        # Only beliefs above threshold should create nodes
        value_nodes = [n for n in nodes if n.type == "belief_value"]
        assert len(value_nodes) == 1  # Only 0.95 is above 0.1 threshold
        assert value_nodes[0].properties["state_index"] == 0

    @pytest.mark.asyncio
    async def test_edge_properties(self, bridge, mock_agent):
        """Test that edges have proper properties."""
        belief_state = await bridge.extract_beliefs(mock_agent)
        nodes = await bridge.belief_to_nodes(belief_state, "test_agent", context={})

        edges = await bridge.create_belief_edges(nodes, "test_agent")

        # Check has_belief edges have probability
        belief_edges = [e for e in edges if e.relationship == "has_belief"]
        for edge in belief_edges:
            assert "probability" in edge.properties
            assert "state_index" in edge.properties

        # Check action edge has action value
        action_edges = [e for e in edges if e.relationship == "resulted_in_action"]
        if action_edges:
            assert "action" in action_edges[0].properties

    @pytest.mark.asyncio
    async def test_prompt_context_integration(self, bridge, mock_agent):
        """Test integration with prompt context."""
        belief_state = await bridge.extract_beliefs(mock_agent)

        nodes = await bridge.belief_to_nodes(
            belief_state, "agent_123", context={"prompt_id": "prompt_789"}
        )

        # Should have prompt context node
        prompt_nodes = [n for n in nodes if n.type == "prompt_context"]
        assert len(prompt_nodes) == 1
        assert prompt_nodes[0].properties["prompt_id"] == "prompt_789"

        # Check edges include prompt connection
        edges = await bridge.create_belief_edges(nodes, "agent_123")
        prompt_edges = [e for e in edges if e.relationship == "generated_belief"]
        assert len(prompt_edges) == 1

    @pytest.mark.asyncio
    async def test_query_agent_beliefs_placeholder(self, bridge, mock_knowledge_graph):
        """Test query method (currently returns empty)."""
        result = await bridge.query_agent_beliefs(mock_knowledge_graph, "agent_123")

        assert result == []  # Placeholder implementation
