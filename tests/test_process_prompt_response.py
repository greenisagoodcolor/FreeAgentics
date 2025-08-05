# tests/test_process_prompt_response.py
import pytest
from pydantic import ValidationError
from api.models.responses import ProcessPromptResponse
from knowledge_graph.models import KnowledgeGraphResponse


def test_process_prompt_response_accepts_knowledge_graph_response():
    """Test that ProcessPromptResponse model can serialize KnowledgeGraphResponse object"""
    # Arrange
    kg_response = KnowledgeGraphResponse(
        nodes=[
            {"id": "node1", "label": "Entity", "properties": {"name": "Test Entity"}},
            {"id": "node2", "label": "Concept", "properties": {"name": "Test Concept"}},
        ],
        edges=[{"source": "node1", "target": "node2", "relationship": "relates_to"}],
        metadata={"created_at": "2025-07-29T12:00:00Z", "confidence": 0.95},
    )

    # Act & Assert - Should not raise ValidationError
    response = ProcessPromptResponse(
        status="success",
        knowledge_graph=kg_response,
        message="Knowledge graph updated successfully",
    )

    # Verify response can be serialized to dict
    response_dict = response.model_dump()
    assert response_dict["status"] == "success"
    assert "knowledge_graph" in response_dict
    assert "nodes" in response_dict["knowledge_graph"]
    assert "edges" in response_dict["knowledge_graph"]


def test_process_prompt_response_handles_dict_knowledge_graph():
    """Test that ProcessPromptResponse also accepts dict representation"""
    # Arrange
    kg_dict = {
        "nodes": [{"id": "node1", "label": "Entity", "properties": {"name": "Test Entity"}}],
        "edges": [],
        "metadata": {"created_at": "2025-07-29T12:00:00Z"},
    }

    # Act & Assert
    response = ProcessPromptResponse(
        status="success", knowledge_graph=kg_dict, message="Knowledge graph from dict"
    )

    response_dict = response.model_dump()
    assert response_dict["knowledge_graph"] == kg_dict


def test_process_prompt_response_validation_error_on_invalid_data():
    """Test that invalid knowledge graph data raises ValidationError"""
    # Act & Assert
    with pytest.raises(ValidationError):
        ProcessPromptResponse(
            status="error",
            knowledge_graph="invalid_string_data",  # Should be dict or KnowledgeGraphResponse
            message="This should fail",
        )
