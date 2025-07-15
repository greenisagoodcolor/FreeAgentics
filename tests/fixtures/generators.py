"""Data generators for creating large-scale test datasets.

Provides generators for creating realistic test data at scale,
with configurable parameters and performance optimizations.
"""

import csv
import json
import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from sqlalchemy.orm import Session

from database.models import (
    Agent,
    AgentRole,
    AgentStatus,
    Coalition,
    CoalitionStatus,
)

from .builders import AgentBuilder, CoalitionBuilder, KnowledgeEdgeBuilder, KnowledgeNodeBuilder
from .schemas import (
    AgentSchema,
    CoalitionSchema,
    KnowledgeEdgeSchema,
    KnowledgeNodeSchema,
    PerformanceTestConfigSchema,
)


class DataGenerator:
    """Base class for data generators."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional random seed."""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.generated_count = 0
        self.start_time = datetime.utcnow()

    def _update_progress(self, count: int = 1):
        """Update generation progress."""
        self.generated_count += count

    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics."""
        elapsed = (datetime.utcnow() - self.start_time).total_seconds()
        return {
            "generated_count": self.generated_count,
            "elapsed_seconds": elapsed,
            "rate_per_second": self.generated_count / elapsed if elapsed > 0 else 0,
        }


class AgentGenerator(DataGenerator):
    """Generator for creating large numbers of agents."""

    def __init__(
        self,
        templates: Optional[List[str]] = None,
        position_bounds: Optional[Dict[str, List[float]]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(seed)

        self.templates = templates or [
            "grid_world",
            "resource_collector",
            "explorer",
            "coordinator",
        ]
        self.position_bounds = position_bounds or {"min": [0, 0], "max": [100, 100]}

        # Pre-generate some common data for efficiency
        self._name_prefixes = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta"]
        self._name_suffixes = [
            "Prime",
            "Secundus",
            "Tertius",
            "Major",
            "Minor",
            "Rex",
            "Nova",
            "Omega",
        ]

    def generate_single(self, **overrides) -> AgentSchema:
        """Generate a single agent."""
        template = overrides.get("template", random.choice(self.templates))

        # Build agent
        builder = AgentBuilder()

        # Apply template-specific configuration
        if template == "resource_collector":
            builder = builder.as_resource_collector()
        elif template == "explorer":
            builder = builder.as_explorer()
        elif template == "coordinator":
            builder = builder.as_coordinator()
        else:
            builder = builder.with_template(template)

        # Generate unique name
        prefix = random.choice(self._name_prefixes)
        suffix = random.choice(self._name_suffixes)
        number = random.randint(1000, 9999)
        name = f"{prefix}_{suffix}_{number}"
        builder = builder.with_name(name)

        # Set random position within bounds
        builder = builder.with_random_position(self.position_bounds)

        # Add realistic metrics
        builder = builder.with_random_metrics()

        # Apply status
        status = overrides.get("status", AgentStatus.ACTIVE)
        builder = builder.with_status(status)

        # Apply remaining overrides
        builder = builder.with_data(**overrides)

        self._update_progress()
        return builder.build()

    def generate_batch(self, count: int, **common_overrides) -> List[AgentSchema]:
        """Generate a batch of agents."""
        agents = []
        for i in range(count):
            agent = self.generate_single(**common_overrides)
            agents.append(agent)
        return agents

    def generate_stream(
        self, count: int, batch_size: int = 100, **common_overrides
    ) -> Iterator[List[AgentSchema]]:
        """Generate agents in streaming batches."""
        generated = 0
        while generated < count:
            current_batch_size = min(batch_size, count - generated)
            batch = self.generate_batch(current_batch_size, **common_overrides)
            yield batch
            generated += current_batch_size

    def generate_diverse_population(
        self, total_count: int, distribution: Optional[Dict[str, float]] = None
    ) -> List[AgentSchema]:
        """Generate a diverse population with template distribution."""
        if distribution is None:
            # Default even distribution
            distribution = {template: 1.0 / len(self.templates) for template in self.templates}

        # Normalize distribution
        total_weight = sum(distribution.values())
        distribution = {k: v / total_weight for k, v in distribution.items()}

        # Generate agents according to distribution
        agents = []
        for template, ratio in distribution.items():
            count = int(total_count * ratio)
            template_agents = self.generate_batch(count, template=template)
            agents.extend(template_agents)

        # Add remaining agents to reach exact count
        remaining = total_count - len(agents)
        if remaining > 0:
            agents.extend(self.generate_batch(remaining))

        # Shuffle for randomness
        random.shuffle(agents)
        return agents

    def generate_spatial_clusters(
        self, total_count: int, num_clusters: int = 5, cluster_std: float = 10.0
    ) -> List[AgentSchema]:
        """Generate agents in spatial clusters."""
        agents = []
        agents_per_cluster = total_count // num_clusters

        # Generate cluster centers
        min_bounds = self.position_bounds["min"]
        max_bounds = self.position_bounds["max"]
        cluster_centers = []

        for _ in range(num_clusters):
            center = [
                random.uniform(min_bounds[i] + cluster_std, max_bounds[i] - cluster_std)
                for i in range(len(min_bounds))
            ]
            cluster_centers.append(center)

        # Generate agents around cluster centers
        for i, center in enumerate(cluster_centers):
            cluster_count = agents_per_cluster
            if i == num_clusters - 1:  # Last cluster gets remaining agents
                cluster_count = total_count - len(agents)

            for _ in range(cluster_count):
                # Generate position near cluster center
                position = [
                    np.clip(np.random.normal(center[j], cluster_std), min_bounds[j], max_bounds[j])
                    for j in range(len(center))
                ]

                agent = self.generate_single()
                # Update position
                agent_dict = agent.dict()
                agent_dict["position"] = position
                agent = AgentSchema(**agent_dict)
                agents.append(agent)

        return agents


class CoalitionGenerator(DataGenerator):
    """Generator for creating coalitions with relationships."""

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

        self._coalition_prefixes = ["United", "Allied", "Strategic", "Tactical", "Elite"]
        self._coalition_types = ["Guardians", "Seekers", "Builders", "Defenders", "Innovators"]

    def generate_single(self, **overrides) -> CoalitionSchema:
        """Generate a single coalition."""
        # Generate name
        prefix = random.choice(self._coalition_prefixes)
        type_name = random.choice(self._coalition_types)
        number = random.randint(100, 999)
        name = f"{prefix} {type_name} {number}"

        # Build coalition
        builder = CoalitionBuilder().with_name(name)

        # Set type-based configuration
        if "resource" in type_name.lower() or "builder" in type_name.lower():
            builder = builder.as_resource_coalition()
        elif "seeker" in type_name.lower() or "explor" in type_name.lower():
            builder = builder.as_exploration_coalition()
        else:
            # Generic coalition with mixed objectives
            builder = (
                builder.with_resource_optimization_objective()
                .with_exploration_objective()
                .with_required_capabilities("coordination", "communication")
            )

        # Set random scores
        builder = builder.with_random_scores()

        # Apply status
        status = overrides.get("status", CoalitionStatus.ACTIVE)
        builder = builder.with_status(status)

        # Apply overrides
        builder = builder.with_data(**overrides)

        self._update_progress()
        return builder.build()

    def generate_batch(self, count: int, **common_overrides) -> List[CoalitionSchema]:
        """Generate a batch of coalitions."""
        return [self.generate_single(**common_overrides) for _ in range(count)]

    def generate_hierarchical_structure(
        self, num_top_level: int = 3, num_sub_coalitions: int = 3, depth: int = 2
    ) -> Dict[str, Any]:
        """Generate a hierarchical coalition structure."""
        structure = {"top_level": [], "hierarchy": {}, "all_coalitions": []}

        def create_level(parent_name: Optional[str], level: int, index: int) -> CoalitionSchema:
            """Recursively create coalition levels."""
            if parent_name:
                name = f"{parent_name} Division {index}"
            else:
                name = f"Top Coalition {index}"

            coalition = self.generate_single(name=name)
            structure["all_coalitions"].append(coalition)

            if level < depth:
                # Create sub-coalitions
                sub_coalitions = []
                for i in range(num_sub_coalitions):
                    sub = create_level(coalition.name, level + 1, i + 1)
                    sub_coalitions.append(sub)

                structure["hierarchy"][coalition.id] = [sc.id for sc in sub_coalitions]

            return coalition

        # Create top-level coalitions
        for i in range(num_top_level):
            top_coalition = create_level(None, 0, i + 1)
            structure["top_level"].append(top_coalition)

        return structure


class KnowledgeGraphGenerator(DataGenerator):
    """Generator for creating knowledge graphs."""

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed)

        self._concept_domains = ["physics", "behavior", "environment", "strategy", "resource"]
        self._entity_types = ["location", "object", "agent", "event", "state"]
        self._relation_types = ["relates_to", "causes", "supports", "contradicts", "derived_from"]

    def generate_node(self, node_type: Optional[str] = None, **overrides) -> KnowledgeNodeSchema:
        """Generate a single knowledge node."""
        if node_type is None:
            node_type = random.choice(["concept", "entity", "fact", "observation", "inference"])

        builder = KnowledgeNodeBuilder().with_type(node_type)

        # Type-specific generation
        if node_type == "concept":
            domain = random.choice(self._concept_domains)
            concept = f"{domain.capitalize()}Concept_{random.randint(1000, 9999)}"
            builder = builder.as_concept(concept)

        elif node_type == "entity":
            entity_type = random.choice(self._entity_types)
            entity_name = f"{entity_type.capitalize()}_{random.randint(1000, 9999)}"
            builder = builder.as_entity(entity_name, entity_type)

        elif node_type == "observation":
            observation = f"Observed phenomenon {random.randint(1000, 9999)}"
            builder = builder.as_observation(observation)

        else:
            # Generic node
            builder = builder.with_label(f"{node_type.capitalize()}_{random.randint(1000, 9999)}")

        # Add confidence
        builder = builder.with_confidence(random.uniform(0.5, 1.0))

        # Add embedding
        builder = builder.with_embedding(dim=128)

        # Apply overrides
        builder = builder.with_data(**overrides)

        self._update_progress()
        return builder.build()

    def generate_edge(
        self,
        source_id: uuid.UUID,
        target_id: uuid.UUID,
        edge_type: Optional[str] = None,
        **overrides,
    ) -> KnowledgeEdgeSchema:
        """Generate a knowledge edge."""
        if edge_type is None:
            edge_type = random.choice(self._relation_types)

        builder = (
            KnowledgeEdgeBuilder()
            .with_nodes(source_id, target_id)
            .with_type(edge_type)
            .with_confidence(random.uniform(0.3, 1.0))
        )

        # Type-specific properties
        if edge_type == "causes":
            builder = builder.as_causal(strength=random.uniform(0.5, 0.9))
        elif edge_type == "supports":
            builder = builder.as_support(evidence_count=random.randint(1, 10))
        elif edge_type == "contradicts":
            builder = builder.as_contradiction()

        # Apply overrides
        builder = builder.with_data(**overrides)

        self._update_progress()
        return builder.build()

    def generate_connected_graph(
        self, num_nodes: int, connectivity: float = 0.1, ensure_connected: bool = True
    ) -> Dict[str, Any]:
        """Generate a connected knowledge graph."""
        nodes = []
        edges = []

        # Generate nodes
        for _ in range(num_nodes):
            node = self.generate_node()
            nodes.append(node)

        # Ensure connectivity if requested
        if ensure_connected and num_nodes > 1:
            # Create a spanning tree to ensure connectivity
            shuffled_nodes = nodes.copy()
            random.shuffle(shuffled_nodes)

            for i in range(1, len(shuffled_nodes)):
                source = shuffled_nodes[i - 1]
                target = shuffled_nodes[i]
                edge = self.generate_edge(source.id, target.id)
                edges.append(edge)

        # Add random edges based on connectivity
        num_possible_edges = num_nodes * (num_nodes - 1) // 2
        num_target_edges = int(num_possible_edges * connectivity)

        # Generate additional edges
        edge_count = len(edges)
        attempts = 0
        max_attempts = num_target_edges * 10

        existing_edges = {(e.source_id, e.target_id) for e in edges}
        existing_edges.update({(e.target_id, e.source_id) for e in edges})

        while edge_count < num_target_edges and attempts < max_attempts:
            source = random.choice(nodes)
            target = random.choice(nodes)

            if source.id != target.id and (source.id, target.id) not in existing_edges:
                edge = self.generate_edge(source.id, target.id)
                edges.append(edge)
                existing_edges.add((source.id, target.id))
                existing_edges.add((target.id, source.id))
                edge_count += 1

            attempts += 1

        return {
            "nodes": nodes,
            "edges": edges,
            "properties": {
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "actual_connectivity": (
                    len(edges) / num_possible_edges if num_possible_edges > 0 else 0
                ),
                "is_connected": ensure_connected,
            },
        }

    def generate_scale_free_graph(
        self, num_nodes: int, initial_nodes: int = 3, edges_per_new_node: int = 2
    ) -> Dict[str, Any]:
        """Generate a scale-free graph using preferential attachment."""
        nodes = []
        edges = []
        node_degrees = {}

        # Create initial complete graph
        for i in range(initial_nodes):
            node = self.generate_node()
            nodes.append(node)
            node_degrees[node.id] = 0

        # Connect initial nodes
        for i in range(initial_nodes):
            for j in range(i + 1, initial_nodes):
                edge = self.generate_edge(nodes[i].id, nodes[j].id)
                edges.append(edge)
                node_degrees[nodes[i].id] += 1
                node_degrees[nodes[j].id] += 1

        # Add remaining nodes with preferential attachment
        for _ in range(initial_nodes, num_nodes):
            new_node = self.generate_node()
            nodes.append(new_node)
            node_degrees[new_node.id] = 0

            # Select nodes to connect based on degree
            degree_sum = sum(node_degrees.values())
            if degree_sum == 0:
                # Fallback to random selection
                targets = random.sample(nodes[:-1], min(edges_per_new_node, len(nodes) - 1))
            else:
                # Preferential attachment
                probabilities = [node_degrees[n.id] / degree_sum for n in nodes[:-1]]
                targets = np.random.choice(
                    nodes[:-1],
                    size=min(edges_per_new_node, len(nodes) - 1),
                    replace=False,
                    p=probabilities,
                )

            # Create edges to selected nodes
            for target in targets:
                edge = self.generate_edge(new_node.id, target.id)
                edges.append(edge)
                node_degrees[new_node.id] += 1
                node_degrees[target.id] += 1

        return {
            "nodes": nodes,
            "edges": edges,
            "properties": {
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "degree_distribution": dict(node_degrees),
                "max_degree": max(node_degrees.values()) if node_degrees else 0,
                "avg_degree": sum(node_degrees.values()) / len(node_degrees) if node_degrees else 0,
            },
        }


class PerformanceDataGenerator:
    """Generator for creating performance test datasets."""

    def __init__(self, session: Optional[Session] = None, seed: Optional[int] = None):
        self.session = session
        self.agent_gen = AgentGenerator(seed=seed)
        self.coalition_gen = CoalitionGenerator(seed=seed)
        self.graph_gen = KnowledgeGraphGenerator(seed=seed)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_dataset(self, config: PerformanceTestConfigSchema) -> Dict[str, Any]:
        """Generate a complete performance test dataset."""
        results = {
            "config": config.dict(),
            "agents": [],
            "coalitions": [],
            "knowledge_graph": None,
            "timing": {},
            "memory": {},
        }

        # Generate agents
        start_time = datetime.utcnow()
        agent_distribution = {
            "resource_collector": 0.3,
            "explorer": 0.3,
            "coordinator": 0.2,
            "grid_world": 0.2,
        }

        results["agents"] = self.agent_gen.generate_diverse_population(
            config.num_agents, distribution=agent_distribution
        )
        results["timing"]["agent_generation"] = (datetime.utcnow() - start_time).total_seconds()

        # Generate coalitions
        start_time = datetime.utcnow()
        results["coalitions"] = self.coalition_gen.generate_batch(config.num_coalitions)
        results["timing"]["coalition_generation"] = (datetime.utcnow() - start_time).total_seconds()

        # Generate knowledge graph
        start_time = datetime.utcnow()
        if config.num_knowledge_nodes > 0:
            # Use scale-free graph for more realistic structure
            results["knowledge_graph"] = self.graph_gen.generate_scale_free_graph(
                config.num_knowledge_nodes, initial_nodes=5, edges_per_new_node=3
            )
        results["timing"]["knowledge_generation"] = (datetime.utcnow() - start_time).total_seconds()

        # Calculate dataset statistics
        results["statistics"] = self._calculate_statistics(results)

        return results

    def generate_to_database(
        self, session: Session, config: PerformanceTestConfigSchema
    ) -> Dict[str, Any]:
        """Generate data directly to database."""
        from .factories import KnowledgeGraphFactory

        results = {"config": config.dict(), "counts": {}, "timing": {}, "errors": []}

        try:
            # Generate agents in batches
            start_time = datetime.utcnow()
            agent_count = 0

            for batch in self.agent_gen.generate_stream(
                config.num_agents, batch_size=config.batch_size
            ):
                # Convert schemas to models and persist
                for agent_schema in batch:
                    try:
                        agent_dict = agent_schema.dict()
                        agent = Agent(**agent_dict)
                        session.add(agent)
                        agent_count += 1
                    except Exception as e:
                        results["errors"].append(f"Agent creation error: {e}")

                # Commit batch
                session.commit()

            results["counts"]["agents"] = agent_count
            results["timing"]["agent_creation"] = (datetime.utcnow() - start_time).total_seconds()

            # Generate coalitions
            start_time = datetime.utcnow()
            coalition_count = 0

            for coalition_schema in self.coalition_gen.generate_batch(config.num_coalitions):
                try:
                    coalition_dict = coalition_schema.dict()
                    coalition = Coalition(**coalition_dict)
                    session.add(coalition)
                    coalition_count += 1
                except Exception as e:
                    results["errors"].append(f"Coalition creation error: {e}")

            session.commit()
            results["counts"]["coalitions"] = coalition_count
            results["timing"]["coalition_creation"] = (
                datetime.utcnow() - start_time
            ).total_seconds()

            # Generate knowledge graph
            if config.num_knowledge_nodes > 0:
                start_time = datetime.utcnow()
                kg_factory = KnowledgeGraphFactory()
                kg_result = kg_factory.create_knowledge_graph(
                    session,
                    num_nodes=config.num_knowledge_nodes,
                    connectivity=config.knowledge_graph_connectivity,
                )
                results["counts"]["knowledge_nodes"] = len(kg_result["nodes"])
                results["counts"]["knowledge_edges"] = len(kg_result["edges"])
                results["timing"]["knowledge_creation"] = (
                    datetime.utcnow() - start_time
                ).total_seconds()

        except Exception as e:
            results["errors"].append(f"Generation error: {e}")
            session.rollback()
            raise

        return results

    def export_to_file(self, dataset: Dict[str, Any], filepath: Path, format: str = "json") -> None:
        """Export generated dataset to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            # Convert objects to dictionaries
            export_data = {
                "config": dataset.get("config", {}),
                "agents": [
                    a.dict() if hasattr(a, "dict") else a for a in dataset.get("agents", [])
                ],
                "coalitions": [
                    c.dict() if hasattr(c, "dict") else c for c in dataset.get("coalitions", [])
                ],
                "knowledge_graph": (
                    {
                        "nodes": [
                            n.dict() if hasattr(n, "dict") else n
                            for n in dataset.get("knowledge_graph", {}).get("nodes", [])
                        ],
                        "edges": [
                            e.dict() if hasattr(e, "dict") else e
                            for e in dataset.get("knowledge_graph", {}).get("edges", [])
                        ],
                    }
                    if dataset.get("knowledge_graph")
                    else None
                ),
                "statistics": dataset.get("statistics", {}),
                "timing": dataset.get("timing", {}),
            }

            # Custom JSON encoder for UUIDs and datetimes
            def json_encoder(obj):
                if isinstance(obj, uuid.UUID):
                    return str(obj)
                elif isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, (AgentStatus, CoalitionStatus, AgentRole)):
                    return obj.value
                else:
                    return str(obj)

            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2, default=json_encoder)

        elif format == "csv":
            # Export agents to CSV
            if dataset.get("agents"):
                agents_file = filepath.with_suffix(".agents.csv")
                with open(agents_file, "w", newline="") as f:
                    if hasattr(dataset["agents"][0], "dict"):
                        fieldnames = list(dataset["agents"][0].dict().keys())
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for agent in dataset["agents"]:
                            writer.writerow(agent.dict())

            # Export coalitions to CSV
            if dataset.get("coalitions"):
                coalitions_file = filepath.with_suffix(".coalitions.csv")
                with open(coalitions_file, "w", newline="") as f:
                    if hasattr(dataset["coalitions"][0], "dict"):
                        fieldnames = list(dataset["coalitions"][0].dict().keys())
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        for coalition in dataset["coalitions"]:
                            writer.writerow(coalition.dict())

        else:
            raise ValueError(f"Unsupported format: {format}")

    def _calculate_statistics(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate dataset statistics."""
        stats = {
            "agent_count": len(dataset.get("agents", [])),
            "coalition_count": len(dataset.get("coalitions", [])),
            "total_generation_time": sum(dataset.get("timing", {}).values()),
        }

        # Agent statistics
        if dataset.get("agents"):
            templates = {}
            statuses = {}
            for agent in dataset["agents"]:
                if hasattr(agent, "template"):
                    templates[agent.template] = templates.get(agent.template, 0) + 1
                if hasattr(agent, "status"):
                    status_val = (
                        agent.status.value if hasattr(agent.status, "value") else str(agent.status)
                    )
                    statuses[status_val] = statuses.get(status_val, 0) + 1

            stats["agent_templates"] = templates
            stats["agent_statuses"] = statuses

        # Knowledge graph statistics
        if dataset.get("knowledge_graph"):
            kg = dataset["knowledge_graph"]
            if "properties" in kg:
                stats["knowledge_graph"] = kg["properties"]
            else:
                stats["knowledge_graph"] = {
                    "node_count": len(kg.get("nodes", [])),
                    "edge_count": len(kg.get("edges", [])),
                }

        return stats


# Convenience functions
def generate_agent_batch(count: int = 10, **kwargs) -> List[AgentSchema]:
    """Generate a batch of agents."""
    generator = AgentGenerator()
    return generator.generate_batch(count, **kwargs)


def generate_coalition_scenario(num_coalitions: int = 5, **kwargs) -> List[CoalitionSchema]:
    """Generate a coalition scenario."""
    generator = CoalitionGenerator()
    return generator.generate_batch(num_coalitions, **kwargs)


def generate_knowledge_graph(
    num_nodes: int = 50, connectivity: float = 0.1, graph_type: str = "random"
) -> Dict[str, Any]:
    """Generate a knowledge graph."""
    generator = KnowledgeGraphGenerator()

    if graph_type == "scale_free":
        return generator.generate_scale_free_graph(num_nodes)
    else:
        return generator.generate_connected_graph(num_nodes, connectivity)


def generate_performance_dataset(
    config: Optional[PerformanceTestConfigSchema] = None,
) -> Dict[str, Any]:
    """Generate a complete performance test dataset."""
    if config is None:
        config = PerformanceTestConfigSchema()

    generator = PerformanceDataGenerator()
    return generator.generate_dataset(config)
