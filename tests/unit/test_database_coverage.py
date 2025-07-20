"""Test database module to verify coverage setup works."""

from unittest.mock import Mock, patch

import pytest

# Import database modules to test coverage
from database import Agent, Base, Coalition, KnowledgeEdge, KnowledgeNode
from database.models import AgentStatus


class TestDatabaseModels:
    """Test database models for coverage."""

    def test_agent_model_creation(self):
        """Test Agent model creation."""
        # Create an agent instance (without database)
        agent = Agent(
            name="TestAgent",
            template="basic-explorer",
            status=AgentStatus.PENDING,
        )

        assert agent.name == "TestAgent"
        assert agent.template == "basic-explorer"
        assert agent.status == AgentStatus.PENDING

    def test_coalition_model_creation(self):
        """Test Coalition model creation."""
        # Create a coalition instance
        coalition = Coalition(
            name="TestCoalition", description="A test coalition for coverage"
        )

        assert coalition.name == "TestCoalition"
        assert coalition.description == "A test coalition for coverage"

    def test_knowledge_node_creation(self):
        """Test KnowledgeNode model creation."""
        # Create a knowledge node
        node = KnowledgeNode(content="Test knowledge content", agent_id=1)

        assert node.content == "Test knowledge content"
        assert node.agent_id == 1

    def test_knowledge_edge_creation(self):
        """Test KnowledgeEdge model creation."""
        # Create a knowledge edge
        edge = KnowledgeEdge(source_id=1, target_id=2, weight=0.8)

        assert edge.source_id == 1
        assert edge.target_id == 2
        assert edge.weight == 0.8

    def test_base_class_exists(self):
        """Test that Base class is properly imported."""
        assert Base is not None
        assert hasattr(Base, 'metadata')
