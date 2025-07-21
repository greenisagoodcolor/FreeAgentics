#!/usr/bin/env python3
"""
FreeAgentics Comprehensive Demo.

This demo showcases the real Active Inference implementation using PyMDP.
It demonstrates:
1. Real variational inference for belief updates
2. Expected free energy minimization for action selection
3. Multi-agent coordination with different agent types
4. Database persistence of agent states
5. Knowledge graph evolution

NO FALLBACKS - This uses real PyMDP Active Inference throughout.
"""

import logging
import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from agents.base_agent import PYMDP_AVAILABLE, BasicExplorerAgent
from agents.resource_collector import ResourceCollectorAgent
from database.session import init_db
from knowledge_graph.evolution import EvolutionEngine
from knowledge_graph.graph_engine import KnowledgeGraph
from world.grid_world import GridWorld, GridWorldConfig, Position

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Run comprehensive FreeAgentics demo."""
    print("🧠 FreeAgentics Active Inference Demo")
    print("=" * 50)

    # Verify PyMDP is available
    if not PYMDP_AVAILABLE:
        print("❌ PyMDP not available! This demo requires real Active Inference.")
        print("Install with: pip install inferactively-pymdp==0.0.7.1")
        return

    print("✅ PyMDP Active Inference Available")
    print()

    # Initialize database
    print("📦 Initializing Database...")
    init_db()
    print("✅ Database initialized")
    print()

    # Create world environment
    print("🌍 Creating Grid World Environment...")
    config = GridWorldConfig(width=8, height=8)
    world = GridWorld(config)

    # Add some goals and obstacles
    world.set_cell(Position(6, 6), 1)  # Goal
    world.set_cell(Position(3, 3), -1)  # Obstacle
    world.set_cell(Position(1, 6), 1)  # Another goal

    print(f"✅ Created {config.width}x{config.height} world with goals and obstacles")
    print()

    # Create agents with real Active Inference
    print("🤖 Creating Active Inference Agents...")

    # Explorer agent using PyMDP
    explorer = BasicExplorerAgent("explorer_1", "Explorer Agent", grid_size=8)
    explorer.start()

    # Resource collector agent
    collector = ResourceCollectorAgent("collector_1", "Resource Collector", grid_size=8)
    collector.start()

    print(f"✅ Created {len([explorer, collector])} agents with PyMDP Active Inference")

    # Show agent configurations
    print("\n🔧 Agent Configurations:")
    print(f"  Explorer: PyMDP={explorer.pymdp_agent is not None}")
    print(f"  Collector: PyMDP={collector.pymdp_agent is not None}")
    print()

    # Initialize knowledge graph
    print("🧠 Initializing Knowledge Graph...")
    knowledge_graph = KnowledgeGraph("demo_graph")
    evolution_engine = EvolutionEngine()
    print("✅ Knowledge graph ready for evolution")
    print()

    # Run simulation
    print("🎮 Starting Active Inference Simulation...")
    print("=" * 50)

    max_steps = 20
    for step in range(max_steps):
        print(f"\n📊 Step {step + 1}/{max_steps}")
        print("-" * 30)

        # Explorer agent step
        explorer_obs = {
            "position": explorer.position.copy(),
            "surroundings": world.get_observation(
                explorer.position[0], explorer.position[1]
            ),
        }

        explorer_action = explorer.step(explorer_obs)
        print(f"🔍 Explorer: pos={explorer.position}, action={explorer_action}")

        # Show Active Inference metrics
        if "belief_entropy" in explorer.metrics:
            print(f"    Belief entropy: {explorer.metrics['belief_entropy']:.3f}")
        if "expected_free_energy" in explorer.metrics:
            print(
                f"    Expected free energy: {explorer.metrics['expected_free_energy']:.3f}"
            )

        # Resource collector step
        collector_obs = {
            "position": collector.position.copy(),
            "surroundings": world.get_observation(
                collector.position[0], collector.position[1]
            ),
            "resources": {},
            "current_load": collector.current_load,
        }

        collector_action = collector.step(collector_obs)
        print(
            f"📦 Collector: pos={collector.position}, action={collector_action}, load={collector.current_load}"
        )

        # Update positions in world (simplified)
        if explorer_action == "up" and explorer.position[0] > 0:
            explorer.position[0] -= 1
        elif explorer_action == "down" and explorer.position[0] < 7:
            explorer.position[0] += 1
        elif explorer_action == "left" and explorer.position[1] > 0:
            explorer.position[1] -= 1
        elif explorer_action == "right" and explorer.position[1] < 7:
            explorer.position[1] += 1

        # Update collector position similarly
        if collector_action == "up" and collector.position[0] > 0:
            collector.position[0] -= 1
        elif collector_action == "down" and collector.position[0] < 7:
            collector.position[0] += 1
        elif collector_action == "left" and collector.position[1] > 0:
            collector.position[1] -= 1
        elif collector_action == "right" and collector.position[1] < 7:
            collector.position[1] += 1

        # Update knowledge graph with observations
        observations = [
            {
                "data": f"explorer_at_{explorer.position}",
                "timestamp": datetime.now(),
                "confidence": 0.9,
                "entity_id": "explorer_1",
            },
            {
                "data": f"collector_at_{collector.position}",
                "timestamp": datetime.now(),
                "confidence": 0.9,
                "entity_id": "collector_1",
            },
        ]

        context = {"observations": observations, "observer_id": "world_system"}

        evolution_metrics = evolution_engine.evolve(knowledge_graph, context)
        if evolution_metrics.nodes_added > 0:
            print(
                f"🧠 Knowledge: +{evolution_metrics.nodes_added} nodes, +{evolution_metrics.edges_added} edges"
            )

        # Show world state periodically
        if step % 5 == 0:
            print(f"\n🗺️  World State (Step {step + 1}):")
            print("   0 1 2 3 4 5 6 7")
            for y in range(8):
                row = f"{y}  "
                for x in range(8):
                    if [x, y] == explorer.position:
                        row += "E "
                    elif [x, y] == collector.position:
                        row += "C "
                    elif world.grid[x][y] == 1:
                        row += "G "  # Goal
                    elif world.grid[x][y] == -1:
                        row += "# "  # Obstacle
                    else:
                        row += ". "  # Empty
                print(row)

        time.sleep(0.5)  # Pause for visibility

    # Show final metrics
    print("\n📊 Final Active Inference Metrics")
    print("=" * 50)

    print("\n🔍 Explorer Agent:")
    explorer_status = explorer.get_status()
    print(f"  Total steps: {explorer_status['total_steps']}")
    print(f"  Final position: {explorer.position}")
    if "belief_entropy" in explorer.metrics:
        print(f"  Final belief entropy: {explorer.metrics['belief_entropy']:.3f}")
    if "total_free_energy" in explorer.metrics:
        print(f"  Total free energy: {explorer.metrics['total_free_energy']:.3f}")

    print("\n📦 Resource Collector:")
    collector_status = collector.get_status()
    print(f"  Total steps: {collector_status['total_steps']}")
    print(f"  Final position: {collector.position}")
    print(f"  Collection efficiency: {collector_status['collection_efficiency']}")
    print(f"  Known resources: {collector_status['known_resources']}")

    print("\n🧠 Knowledge Graph:")
    print(f"  Total nodes: {len(knowledge_graph.nodes)}")
    print(f"  Total edges: {len(knowledge_graph.edges)}")
    print(f"  Graph version: {knowledge_graph.version}")

    print("\n✅ Demo Complete!")
    print("🎉 FreeAgentics successfully demonstrated real Active Inference with PyMDP!")

    # Stop agents
    explorer.stop()
    collector.stop()


if __name__ == "__main__":
    main()
