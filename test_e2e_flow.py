#!/usr/bin/env python3
"""Test the end-to-end flow: prompt → LLM → GMN → PyMDP → agent → KG → WS."""

import asyncio
import os
from datetime import datetime

# Set up test environment
os.environ["OPENAI_API_KEY"] = "test-key"  # Would use real key in production
os.environ["DATABASE_URL"] = "sqlite:///test_freeagentics.db"

from agents.agent_manager import AgentManager
from agents.gmn_pymdp_adapter import adapt_gmn_to_pymdp
from inference.active.gmn_parser import parse_gmn_spec


async def test_e2e_flow():
    """Test the complete flow."""
    print("🚀 Testing FreeAgentics End-to-End Flow")
    print("=" * 50)

    # Step 1: Create a GMN spec (simulating LLM output)
    print("\n1️⃣ Creating GMN specification...")
    gmn_spec = {
        "name": "test_explorer",
        "description": "An agent that explores a grid world",
        "states": ["exploring", "found_target", "avoiding_obstacle"],
        "observations": ["empty", "target", "obstacle", "boundary"],
        "actions": ["move_up", "move_down", "move_left", "move_right", "stay"],
        "parameters": {
            "A": [
                [0.8, 0.1, 0.1, 0.0],  # P(obs|exploring)
                [0.1, 0.8, 0.0, 0.1],  # P(obs|found_target)
                [0.1, 0.0, 0.8, 0.1],  # P(obs|avoiding_obstacle)
            ],
            "B": [
                # Transitions for each action
                [[[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]]],  # move_up
                [[[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]]],  # move_down
                [[[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]]],  # move_left
                [[[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.7]]],  # move_right
                [[[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]],  # stay
            ],
            "C": [[0.1, 0.8, 0.05, 0.05]],  # Prefer finding target
            "D": [[0.8, 0.1, 0.1]],  # Start in exploring state
        },
    }
    print(
        f"✅ Created GMN with {len(gmn_spec['states'])} states, {len(gmn_spec['actions'])} actions"
    )

    # Step 2: Parse and validate GMN
    print("\n2️⃣ Parsing GMN specification...")
    try:
        validated_gmn = parse_gmn_spec(gmn_spec)
        print("✅ GMN validation passed")
    except Exception as e:
        print(f"❌ GMN validation failed: {e}")
        return

    # Step 3: Convert to PyMDP
    print("\n3️⃣ Converting to PyMDP format...")
    try:
        adapt_gmn_to_pymdp(validated_gmn)
        print("✅ PyMDP conversion successful")
    except Exception as e:
        print(f"❌ PyMDP conversion failed: {e}")
        return

    # Step 4: Create agent
    print("\n4️⃣ Creating agent...")
    agent_manager = AgentManager()
    agent_id = f"test_agent_{datetime.now().strftime('%H%M%S')}"

    try:
        agent = agent_manager.create_agent(
            agent_id=agent_id, name="Test Explorer", gmn_config=validated_gmn
        )
        print(f"✅ Agent created: {agent_id}")
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        return

    # Step 5: Initialize knowledge graph
    print("\n5️⃣ Initializing knowledge graph...")
    from agents.kg_integration import AgentKnowledgeGraphIntegration

    kg_integration = AgentKnowledgeGraphIntegration()
    agent.kg_integration = kg_integration
    print("✅ Knowledge graph initialized")

    # Step 6: Start agent
    print("\n6️⃣ Starting agent...")
    agent_manager.start_agent(agent_id)
    print("✅ Agent started")

    # Step 7: Run some steps
    print("\n7️⃣ Running agent steps...")
    observations = ["empty", "empty", "obstacle", "empty", "target"]

    for i, obs in enumerate(observations):
        print(f"\n  Step {i + 1}: Observation = '{obs}'")

        # Map observation to index
        obs_idx = gmn_spec["observations"].index(obs)

        # Agent takes a step
        action = agent.step(obs_idx)

        # Map action index back to name
        if isinstance(action, int) and 0 <= action < len(gmn_spec["actions"]):
            action_name = gmn_spec["actions"][action]
        else:
            action_name = str(action)

        print(f"  → Action: '{action_name}'")
        print(f"  → Free energy: {agent.metrics.get('current_free_energy', 'N/A')}")

    # Step 8: Check knowledge graph
    print("\n8️⃣ Checking knowledge graph...")
    history = kg_integration.get_agent_history(agent_id, limit=10)
    print(f"✅ Knowledge graph has {history['total_events']} events")

    if history["events"]:
        print("\n  Recent events:")
        for event in history["events"][:3]:
            print(f"  - {event['type']}: {event['label'][:50]}...")

    # Step 9: Save knowledge graph
    print("\n9️⃣ Saving knowledge graph...")
    kg_integration.save()
    print("✅ Knowledge graph saved")

    # Step 10: Stop agent
    print("\n🏁 Stopping agent...")
    agent_manager.stop_agent(agent_id)
    print("✅ Agent stopped")

    print("\n" + "=" * 50)
    print("✨ End-to-end flow test completed successfully!")
    print(f"   - Agent ID: {agent_id}")
    print(f"   - Total steps: {agent.total_steps}")
    print(f"   - KG nodes: {len(kg_integration.graph.nodes)}")
    print(f"   - KG edges: {len(kg_integration.graph.edges)}")


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_e2e_flow())
