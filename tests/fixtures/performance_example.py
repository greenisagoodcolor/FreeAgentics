"""Performance test data generation examples.

This script demonstrates how to generate large-scale test data
for performance testing and benchmarking.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.base import Base
from tests.fixtures import PerformanceDataFactory
from tests.fixtures.generators import (
    AgentGenerator,
    CoalitionGenerator,
    KnowledgeGraphGenerator,
)
from tests.fixtures.schemas import PerformanceTestConfigSchema


def generate_memory_efficient_dataset(
    config: PerformanceTestConfigSchema,
) -> Dict[str, Any]:
    """Generate large dataset with memory-efficient streaming."""
    print(f"Generating dataset with {config.num_agents} agents...")

    start_time = time.time()
    results = {"config": config.dict(), "generation_stats": {}, "timing": {}}

    # Generate agents in streaming batches
    agent_gen = AgentGenerator(seed=config.seed)
    agent_count = 0
    agent_start = time.time()

    # Process in batches to avoid memory issues
    for batch in agent_gen.generate_stream(count=config.num_agents, batch_size=config.batch_size):
        agent_count += len(batch)
        if agent_count % 1000 == 0:
            print(f"  Generated {agent_count} agents...")

    results["timing"]["agents"] = time.time() - agent_start
    results["generation_stats"]["agents"] = agent_count

    # Generate coalitions
    print(f"Generating {config.num_coalitions} coalitions...")
    coalition_start = time.time()

    coalition_gen = CoalitionGenerator(seed=config.seed)
    coalitions = coalition_gen.generate_batch(config.num_coalitions)

    results["timing"]["coalitions"] = time.time() - coalition_start
    results["generation_stats"]["coalitions"] = len(coalitions)

    # Generate knowledge graph
    if config.num_knowledge_nodes > 0:
        print(f"Generating knowledge graph with {config.num_knowledge_nodes} nodes...")
        kg_start = time.time()

        kg_gen = KnowledgeGraphGenerator(seed=config.seed)

        # Use scale-free graph for more realistic structure
        if config.num_knowledge_nodes > 1000:
            graph = kg_gen.generate_scale_free_graph(
                num_nodes=config.num_knowledge_nodes,
                initial_nodes=10,
                edges_per_new_node=3,
            )
        else:
            graph = kg_gen.generate_connected_graph(
                num_nodes=config.num_knowledge_nodes,
                connectivity=config.knowledge_graph_connectivity,
            )

        results["timing"]["knowledge_graph"] = time.time() - kg_start
        results["generation_stats"]["knowledge_nodes"] = len(graph["nodes"])
        results["generation_stats"]["knowledge_edges"] = len(graph["edges"])

    results["timing"]["total"] = time.time() - start_time

    return results


def run_scaling_test():
    """Test data generation at different scales."""
    scales = [
        ("Small", 100, 10, 500),
        ("Medium", 1000, 50, 5000),
        ("Large", 10000, 500, 50000),
        ("XLarge", 100000, 5000, 500000),
    ]

    results = []

    for name, agents, coalitions, nodes in scales:
        print(f"\n{'=' * 50}")
        print(f"Running {name} scale test")
        print(f"{'=' * 50}")

        config = PerformanceTestConfigSchema(
            num_agents=agents,
            num_coalitions=coalitions,
            num_knowledge_nodes=nodes,
            knowledge_graph_connectivity=0.01,  # Lower for large graphs
            batch_size=min(1000, agents // 10),
            seed=42,
        )

        result = generate_memory_efficient_dataset(config)
        result["scale"] = name
        results.append(result)

        print(f"\nResults for {name}:")
        print(
            f"  Agents: {result['generation_stats']['agents']} in {result['timing']['agents']:.2f}s"
        )
        print(
            f"  Coalitions: {result['generation_stats']['coalitions']} in {result['timing']['coalitions']:.2f}s"
        )
        if "knowledge_nodes" in result["generation_stats"]:
            print(
                f"  Knowledge Nodes: {result['generation_stats']['knowledge_nodes']} in {result['timing']['knowledge_graph']:.2f}s"
            )
        print(f"  Total Time: {result['timing']['total']:.2f}s")

    return results


def generate_spatial_distribution_test():
    """Generate agents with different spatial distributions."""
    print("\nGenerating spatial distribution test data...")

    agent_gen = AgentGenerator(position_bounds={"min": [0, 0], "max": [1000, 1000]})

    distributions = {}

    # Uniform distribution
    print("  Generating uniform distribution...")
    uniform_agents = agent_gen.generate_batch(1000)
    distributions["uniform"] = {
        "agents": len(uniform_agents),
        "description": "Uniformly distributed agents",
    }

    # Clustered distribution
    print("  Generating clustered distribution...")
    clustered_agents = agent_gen.generate_spatial_clusters(
        total_count=1000, num_clusters=5, cluster_std=50.0
    )
    distributions["clustered"] = {
        "agents": len(clustered_agents),
        "description": "5 spatial clusters with std=50.0",
    }

    # Mixed distribution (diverse templates)
    print("  Generating diverse population...")
    diverse_agents = agent_gen.generate_diverse_population(
        total_count=1000,
        distribution={
            "resource_collector": 0.4,
            "explorer": 0.3,
            "coordinator": 0.2,
            "grid_world": 0.1,
        },
    )
    distributions["diverse"] = {
        "agents": len(diverse_agents),
        "description": "Mixed agent types with custom distribution",
    }

    return distributions


def generate_database_stress_test(db_url: str = "sqlite:///:memory:"):
    """Generate data directly to database for stress testing."""
    print("\nRunning database stress test...")

    # Create engine and tables
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)

    session = SessionLocal()

    try:
        config = PerformanceTestConfigSchema(
            num_agents=5000,
            num_coalitions=100,
            num_knowledge_nodes=10000,
            batch_size=500,
        )

        factory = PerformanceDataFactory()
        results = factory.create_performance_scenario(session, config)

        print("\nDatabase Stress Test Results:")
        print(f"  Agents created: {len(results['agents'])}")
        print(f"  Coalitions created: {len(results['coalitions'])}")
        print(f"  Total creation time: {results['statistics']['total_creation_time']:.2f}s")

        # Test query performance
        print("\nTesting query performance...")

        # Count queries
        start = time.time()
        agent_count = session.query("SELECT COUNT(*) FROM agents").scalar()
        coalition_count = session.query("SELECT COUNT(*) FROM coalitions").scalar()
        node_count = session.query("SELECT COUNT(*) FROM db_knowledge_nodes").scalar()
        query_time = time.time() - start

        print(f"  Count queries completed in {query_time:.3f}s")
        print(f"    Agents: {agent_count}")
        print(f"    Coalitions: {coalition_count}")
        print(f"    Knowledge Nodes: {node_count}")

    finally:
        session.close()
        engine.dispose()


def export_test_data_samples():
    """Export sample test data in different formats."""
    print("\nExporting test data samples...")

    output_dir = Path("test_data_samples")
    output_dir.mkdir(exist_ok=True)

    # Generate sample dataset
    config = PerformanceTestConfigSchema(
        num_agents=100, num_coalitions=10, num_knowledge_nodes=200, seed=42
    )

    factory = PerformanceDataFactory()
    dataset = factory.generate_dataset(config)

    # Export to JSON
    json_file = output_dir / "sample_data.json"
    factory.export_to_file(dataset, json_file, format="json")
    print(f"  Exported JSON to {json_file}")

    # Export to CSV
    csv_file = output_dir / "sample_data"
    factory.export_to_file(dataset, csv_file, format="csv")
    print(f"  Exported CSV files to {csv_file}.*.csv")

    # Create summary report
    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "config": config.dict(),
        "statistics": dataset["statistics"],
        "timing": dataset["timing"],
    }

    summary_file = output_dir / "generation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Exported summary to {summary_file}")


def main():
    """Main entry point for performance data generation."""
    parser = argparse.ArgumentParser(description="Generate performance test data")
    parser.add_argument(
        "--test",
        choices=["scaling", "spatial", "database", "export", "all"],
        default="all",
        help="Type of test to run",
    )
    parser.add_argument(
        "--db-url",
        default="sqlite:///:memory:",
        help="Database URL for database stress test",
    )

    args = parser.parse_args()

    print("FreeAgentics Performance Test Data Generator")
    print("=" * 50)

    if args.test in ["scaling", "all"]:
        scaling_results = run_scaling_test()

        # Save results
        with open("scaling_test_results.json", "w") as f:
            json.dump(scaling_results, f, indent=2)
        print("\nScaling test results saved to scaling_test_results.json")

    if args.test in ["spatial", "all"]:
        spatial_results = generate_spatial_distribution_test()
        print("\nSpatial distribution test completed")
        for dist_type, info in spatial_results.items():
            print(f"  {dist_type}: {info['description']}")

    if args.test in ["database", "all"]:
        generate_database_stress_test(args.db_url)

    if args.test in ["export", "all"]:
        export_test_data_samples()

    print("\n✅ Performance data generation completed!")


if __name__ == "__main__":
    main()
