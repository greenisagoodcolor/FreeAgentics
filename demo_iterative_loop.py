#!/usr/bin/env python3
"""Demonstration script for the iterative loop functionality.

This script shows how the iterative controller enhances the conversation
by building on previous iterations and generating intelligent suggestions.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.pymdp_adapter import PyMDPCompatibilityAdapter
from knowledge_graph.graph_engine import KnowledgeGraph
from services.agent_factory import AgentFactory
from services.belief_kg_bridge import BeliefKGBridge
from services.gmn_generator import GMNGenerator
from services.iterative_controller import (
    ConversationContext,
    IterativeController,
)


async def demonstrate_iterative_loop():
    """Demonstrate the iterative loop with a series of prompts."""

    print("=== FreeAgentics Iterative Loop Demonstration ===\n")

    # Initialize components
    knowledge_graph = KnowledgeGraph()
    belief_kg_bridge = BeliefKGBridge()
    pymdp_adapter = PyMDPCompatibilityAdapter()

    # Create iterative controller
    controller = IterativeController(
        knowledge_graph=knowledge_graph,
        belief_kg_bridge=belief_kg_bridge,
        pymdp_adapter=pymdp_adapter,
    )

    # Create conversation context
    conversation_id = "demo-conversation-001"
    context = ConversationContext(conversation_id)

    # Demonstration prompts that build on each other
    prompts = [
        "Create an explorer agent for a 5x5 grid world",
        "Add goal states to make the explorer purposeful",
        "Make the agent more curious about unexplored areas",
        "Add another agent that can coordinate with the explorer",
        "Enable communication between the agents",
    ]

    # Simulate iterative processing
    for i, prompt in enumerate(prompts, 1):
        print(f"\n--- Iteration {i} ---")
        print(f"Prompt: {prompt}")

        # Prepare iteration context
        iteration_context = await controller.prepare_iteration_context(
            context, prompt
        )

        print(f"\nIteration Context:")
        print(f"  - Iteration Number: {iteration_context['iteration_number']}")
        print(
            f"  - Previous Suggestions: {iteration_context.get('previous_suggestions', [])}"
        )
        print(
            f"  - Prompt Analysis: {iteration_context['prompt_analysis']['evolution']}"
        )

        # Simulate agent creation and belief extraction
        agent_id = f"agent-{i}"
        mock_agent = type(
            'MockAgent',
            (),
            {
                'beliefs': {'state': [0.3 + i * 0.1, 0.7 - i * 0.1]},
                'action_hist': list(range(i)),
            },
        )()

        # Simulate KG updates
        kg_updates = [
            type(
                'MockUpdate', (), {'node_id': f'node-{i}a', 'applied': True}
            )(),
            type(
                'MockUpdate', (), {'node_id': f'node-{i}b', 'applied': True}
            )(),
        ]

        # Generate intelligent suggestions
        suggestions = await controller.generate_intelligent_suggestions(
            agent_id,
            mock_agent,
            context,
            {'state': [0.3 + i * 0.1, 0.7 - i * 0.1]},
            None,  # Mock DB
        )

        print(f"\nGenerated Suggestions:")
        for j, suggestion in enumerate(suggestions, 1):
            print(f"  {j}. {suggestion}")

        # Update context for next iteration
        await controller.update_conversation_context(
            context,
            prompt,
            agent_id,
            f"GMN spec for iteration {i}",
            mock_agent.beliefs,
            kg_updates,
            suggestions,
        )

        # Show conversation summary
        summary = context.get_context_summary()
        print(f"\nConversation Summary:")
        print(f"  - Total Agents: {summary['total_agents']}")
        print(f"  - KG Nodes: {summary['kg_nodes']}")
        print(f"  - Belief Evolution: {summary['belief_evolution']['trend']}")
        print(f"  - Stability: {summary['belief_evolution']['stability']:.2%}")

        if i < len(prompts):
            print("\nPress Enter to continue to next iteration...")
            input()

    print("\n=== Demonstration Complete ===")
    print("\nKey Features Demonstrated:")
    print("1. Context accumulation across iterations")
    print("2. Intelligent suggestion generation based on conversation state")
    print("3. Belief evolution tracking")
    print("4. Knowledge graph growth monitoring")
    print("5. Adaptive constraints based on iteration progress")


def main():
    """Run the demonstration."""
    try:
        asyncio.run(demonstrate_iterative_loop())
    except KeyboardInterrupt:
        print("\n\nDemonstration interrupted.")
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
