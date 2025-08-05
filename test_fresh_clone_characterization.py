#!/usr/bin/env python3
"""
Characterization tests that capture the current fresh-clone bugs.

These tests document the broken behavior we need to fix:
1. KnowledgeNode.__init__() got an unexpected keyword argument 'node_type'
2. Failed to get agent history: 'NoneType' object has no attribute 'nodes'
3. WebSocket connection errors
"""

import pytest


def test_knowledge_node_parameter_bug():
    """Test that demonstrates the KnowledgeNode parameter bug."""
    from knowledge_graph.graph_engine import KnowledgeNode, NodeType

    # This should fail with current code - unexpected keyword argument 'node_type'
    with pytest.raises(TypeError, match="unexpected keyword argument 'node_type'"):
        node = KnowledgeNode(
            node_type=NodeType.OBSERVATION,  # Bug: uses node_type instead of type
            label="test_observation",
            properties={"test": "data"},
        )


def test_kg_integration_none_graph_bug():
    """Test that demonstrates the NoneType graph attribute bug."""
    from agents.kg_integration import AgentKnowledgeGraphIntegration

    # Create integration that might have None graph
    integration = AgentKnowledgeGraphIntegration()

    # Manually set graph to None to simulate the bug
    integration.graph = None

    # This should fail with 'NoneType' object has no attribute 'nodes'
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute"):
        result = integration.get_agent_history("test_agent_123")


def test_websocket_url_construction_issue():
    """Test that demonstrates WebSocket URL construction issues."""
    from web.lib.websocket_client import WebSocketClient

    # Test with localhost URL that should work but might not due to path issues
    client = WebSocketClient("ws://localhost:8000/api/v1/ws/dev")

    # This should connect but might fail due to incorrect URL handling
    # We can't actually test connection here, but we can test URL construction
    assert client.url == "ws://localhost:8000/api/v1/ws/dev"


def test_agent_integration_cascading_failures():
    """Test that demonstrates how bugs cascade through the system."""
    from agents.kg_integration import AgentKnowledgeGraphIntegration

    # Create integration
    integration = AgentKnowledgeGraphIntegration()

    # This will fail due to the node_type parameter bug
    with pytest.raises(TypeError):
        integration.update_from_agent_step(
            agent_id="test_agent",
            observation={"sensor": "camera", "value": "object_detected"},
            action="move_forward",
            beliefs={"location": [0, 0]},
            free_energy=0.5,
        )


if __name__ == "__main__":
    print("Running fresh-clone characterization tests...")
    print("These tests capture the current broken behavior that needs to be fixed.")

    # Run each test to show the failures
    tests = [
        test_knowledge_node_parameter_bug,
        test_kg_integration_none_graph_bug,
        test_websocket_url_construction_issue,
        test_agent_integration_cascading_failures,
    ]

    for test_func in tests:
        try:
            print(f"\n--- Running {test_func.__name__} ---")
            test_func()
            print("PASS: Test completed without expected error")
        except Exception as e:
            print(f"EXPECTED FAILURE: {type(e).__name__}: {e}")
