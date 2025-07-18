"""Factory classes for creating test data using the builder pattern.

Provides high-level factory methods for creating complex test scenarios
with realistic data relationships and constraints.
"""

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from database.models import Agent
from database.models import AgentRole as DBAgentRole
from database.models import AgentStatus as DBAgentStatus
from database.models import Coalition
from database.models import CoalitionStatus as DBCoalitionStatus
from database.models import (
    KnowledgeEdge,
    KnowledgeNode,
    agent_coalition_association,
)

from .builders import (
    AgentBuilder,
    CoalitionBuilder,
    KnowledgeEdgeBuilder,
    KnowledgeNodeBuilder,
)
from .schemas import BatchAgentSchema, PerformanceTestConfigSchema


class AgentFactory:
    """Factory for creating Agent test data with database persistence."""

    @staticmethod
    def create(session: Optional[Session] = None, **overrides) -> Agent:
        """Create a single agent with optional overrides.

        Args:
            session: Optional database session for persistence
            **overrides: Field overrides for the agent

        Returns:
            Agent instance (persisted if session provided)
        """
        # Build agent schema with defaults
        builder = AgentBuilder()

        # Apply common patterns based on template
        template = overrides.get("template", "grid_world")
        if template == "resource_collector":
            builder = builder.as_resource_collector()
        elif template == "explorer":
            builder = builder.as_explorer()
        elif template == "coordinator":
            builder = builder.as_coordinator()

        # Apply overrides
        if "name" in overrides:
            builder = builder.with_name(overrides.pop("name"))
        if "status" in overrides:
            builder = builder.with_status(overrides.pop("status"))
        if "position" in overrides:
            pos = overrides.pop("position")
            builder = builder.with_position(*pos)

        # Apply remaining overrides
        builder = builder.with_data(**overrides)

        # Build schema and convert to model
        schema = builder.build()
        agent_data = schema.dict()

        # Convert schema fields to model fields
        agent_data["beliefs"] = agent_data.get("beliefs", {})
        agent_data["preferences"] = agent_data.get("preferences", {})
        agent_data["pymdp_config"] = agent_data.get("pymdp_config", {})

        # Create model instance
        agent = Agent(**agent_data)

        # Persist if session provided
        if session:
            session.add(agent)
            session.commit()
            session.refresh(agent)

        return agent

    @staticmethod
    def create_batch(
        session: Optional[Session] = None, count: int = 10, **common_overrides
    ) -> List[Agent]:
        """Create multiple agents with common overrides.

        Args:
            session: Optional database session
            count: Number of agents to create
            **common_overrides: Common field overrides for all agents

        Returns:
            List of Agent instances
        """
        agents = []

        # Parse batch configuration
        batch_config = BatchAgentSchema(
            count=count,
            template=common_overrides.get("template", "grid_world"),
            name_prefix=common_overrides.get("name_prefix", "Agent"),
            distribute_positions=common_overrides.get(
                "distribute_positions", True
            ),
            position_bounds=common_overrides.get("position_bounds"),
        )

        for i in range(batch_config.count):
            # Create unique overrides for this agent
            overrides = common_overrides.copy()
            overrides["name"] = f"{batch_config.name_prefix}_{i:04d}"

            # Distribute positions if requested
            if (
                batch_config.distribute_positions
                and batch_config.position_bounds
            ):
                builder = AgentBuilder()
                builder = builder.with_random_position(
                    batch_config.position_bounds
                )
                position_data = builder._data.get("position")
                if position_data:
                    overrides["position"] = position_data

            # Create agent
            agent = AgentFactory.create(None, **overrides)
            agents.append(agent)

        # Batch persist if session provided
        if session:
            session.add_all(agents)
            session.commit()
            for agent in agents:
                session.refresh(agent)

        return agents

    @staticmethod
    def create_with_history(
        session: Optional[Session] = None,
        days_active: int = 7,
        inference_rate: int = 10,
        **overrides,
    ) -> Agent:
        """Create an agent with simulated activity history.

        Args:
            session: Optional database session
            days_active: Number of days the agent has been active
            inference_rate: Average inferences per day
            **overrides: Field overrides

        Returns:
            Agent with realistic history
        """
        # Calculate timestamps and statistics
        created_at = datetime.utcnow() - timedelta(days=days_active)
        last_active = datetime.utcnow() - timedelta(
            hours=random.randint(0, 24)
        )
        total_inferences = days_active * inference_rate + random.randint(
            -10, 10
        )
        total_steps = total_inferences * random.randint(5, 15)

        # Build agent with history
        builder = (
            AgentBuilder()
            .with_timestamps(created_at=created_at, last_active=last_active)
            .with_inference_history(total_inferences, total_steps)
            .with_random_metrics()
            .active()
        )  # Historical agents are typically active

        # Apply overrides
        builder = builder.with_data(**overrides)

        # Create and persist
        schema = builder.build()
        agent = Agent(**schema.dict())

        if session:
            session.add(agent)
            session.commit()
            session.refresh(agent)

        return agent


class CoalitionFactory:
    """Factory for creating Coalition test data with member relationships."""

    @staticmethod
    def create(
        session: Optional[Session] = None,
        agents: Optional[List[Agent]] = None,
        **overrides,
    ) -> Coalition:
        """Create a coalition with optional member agents.

        Args:
            session: Optional database session
            agents: List of agents to add as members
            **overrides: Field overrides

        Returns:
            Coalition instance with members
        """
        # Build coalition
        builder = CoalitionBuilder()

        # Apply common patterns
        if "objectives" not in overrides:
            # Add default objectives based on name/type
            if "resource" in overrides.get("name", "").lower():
                builder = builder.as_resource_coalition()
            elif "explor" in overrides.get("name", "").lower():
                builder = builder.as_exploration_coalition()
            else:
                # Generic objectives
                builder = (
                    builder.with_resource_optimization_objective().with_exploration_objective()
                )

        # Apply overrides
        if "name" in overrides:
            builder = builder.with_name(overrides.pop("name"))
        if "status" in overrides:
            builder = builder.with_status(overrides.pop("status"))

        builder = builder.with_data(**overrides)

        # Create coalition
        schema = builder.build()
        coalition = Coalition(**schema.dict())

        # Handle persistence and agent relationships
        if session:
            session.add(coalition)
            session.commit()

            # Add agents as members
            if agents:
                for i, agent in enumerate(agents):
                    # Assign roles - first agent is leader, next 20% coordinators
                    if i == 0:
                        role = DBAgentRole.LEADER
                    elif i < len(agents) * 0.2:
                        role = DBAgentRole.COORDINATOR
                    else:
                        role = DBAgentRole.MEMBER

                    # Create association
                    session.execute(
                        agent_coalition_association.insert().values(
                            agent_id=agent.id,
                            coalition_id=coalition.id,
                            role=role,
                            joined_at=datetime.utcnow(),
                            contribution_score=random.uniform(0.5, 1.0),
                            trust_score=random.uniform(0.7, 1.0),
                        )
                    )

            session.commit()
            session.refresh(coalition)

        return coalition

    @staticmethod
    def create_with_agents(
        session: Session,
        num_agents: int = 5,
        agent_template: str = "grid_world",
        **coalition_overrides,
    ) -> Tuple[Coalition, List[Agent]]:
        """Create a coalition with new agents.

        Args:
            session: Database session (required)
            num_agents: Number of agents to create
            agent_template: Template for created agents
            **coalition_overrides: Coalition field overrides

        Returns:
            Tuple of (Coalition, List[Agent])
        """
        # Create agents first
        agents = AgentFactory.create_batch(
            session,
            count=num_agents,
            template=agent_template,
            status=DBAgentStatus.ACTIVE,
        )

        # Create coalition with agents
        coalition = CoalitionFactory.create(
            session, agents=agents, **coalition_overrides
        )

        return coalition, agents

    @staticmethod
    def create_coalition_network(
        session: Session,
        num_coalitions: int = 3,
        agents_per_coalition: int = 5,
        overlap_probability: float = 0.1,
    ) -> Dict[str, Any]:
        """Create a network of coalitions with possible agent overlap.

        Args:
            session: Database session
            num_coalitions: Number of coalitions to create
            agents_per_coalition: Base number of agents per coalition
            overlap_probability: Probability an agent joins multiple coalitions

        Returns:
            Dictionary with created coalitions and agents
        """
        all_agents = []
        coalitions = []

        # Create coalitions with their base agents
        for i in range(num_coalitions):
            coalition_name = f"NetworkCoalition_{i:02d}"
            coalition_type = random.choice(
                ["resource", "exploration", "defense"]
            )

            coalition, agents = CoalitionFactory.create_with_agents(
                session,
                num_agents=agents_per_coalition,
                agent_template="grid_world",
                name=coalition_name,
                description=f"Coalition focused on {coalition_type}",
            )

            coalitions.append(coalition)
            all_agents.extend(agents)

        # Add some agents to multiple coalitions
        for agent in all_agents:
            for coalition in coalitions:
                # Skip if already a member
                if any(a.id == agent.id for a in coalition.agents):
                    continue

                # Randomly add to other coalitions
                if random.random() < overlap_probability:
                    session.execute(
                        agent_coalition_association.insert().values(
                            agent_id=agent.id,
                            coalition_id=coalition.id,
                            role=DBAgentRole.MEMBER,
                            joined_at=datetime.utcnow(),
                            contribution_score=random.uniform(0.3, 0.7),
                            trust_score=random.uniform(0.5, 0.9),
                        )
                    )

        session.commit()

        # Refresh coalitions to get updated member lists
        for coalition in coalitions:
            session.refresh(coalition)

        return {
            "coalitions": coalitions,
            "total_agents": len(set(all_agents)),
            "network_stats": {
                "num_coalitions": num_coalitions,
                "avg_coalition_size": sum(len(c.agents) for c in coalitions)
                / num_coalitions,
                "agents_in_multiple": sum(
                    1
                    for a in all_agents
                    if sum(
                        1
                        for c in coalitions
                        if any(m.id == a.id for m in c.agents)
                    )
                    > 1
                ),
            },
        }


class KnowledgeGraphFactory:
    """Factory for creating knowledge graph test data."""

    @staticmethod
    def create_node(
        session: Optional[Session] = None, **overrides
    ) -> KnowledgeNode:
        """Create a single knowledge node.

        Args:
            session: Optional database session
            **overrides: Field overrides

        Returns:
            KnowledgeNode instance
        """
        builder = KnowledgeNodeBuilder()

        # Apply type-specific patterns
        node_type = overrides.get("type", "concept")
        if node_type == "concept" and "label" in overrides:
            builder = builder.as_concept(overrides["label"])
        elif node_type == "entity" and "label" in overrides:
            entity_type = overrides.get("properties", {}).get(
                "entity_type", "generic"
            )
            builder = builder.as_entity(overrides["label"], entity_type)
        elif node_type == "observation":
            builder = builder.as_observation(
                overrides.get("label", "Test observation"),
                observer_id=overrides.get("creator_agent_id"),
            )

        # Apply overrides
        builder = builder.with_data(**overrides)

        # Add embedding if not provided
        if "embedding" not in overrides:
            builder = builder.with_embedding(dim=128)

        # Create node
        schema = builder.build()
        node = KnowledgeNode(**schema.dict())

        if session:
            session.add(node)
            session.commit()
            session.refresh(node)

        return node

    @staticmethod
    def create_edge(
        session: Optional[Session] = None,
        source: Optional[KnowledgeNode] = None,
        target: Optional[KnowledgeNode] = None,
        **overrides,
    ) -> KnowledgeEdge:
        """Create a knowledge edge between nodes.

        Args:
            session: Optional database session
            source: Source node (created if not provided)
            target: Target node (created if not provided)
            **overrides: Field overrides

        Returns:
            KnowledgeEdge instance
        """
        # Create nodes if not provided
        if source is None:
            source = KnowledgeGraphFactory.create_node(session)
        if target is None:
            target = KnowledgeGraphFactory.create_node(session)

        # Build edge
        builder = KnowledgeEdgeBuilder().with_nodes(source.id, target.id)

        # Apply type-specific patterns
        edge_type = overrides.get("type", "relates_to")
        if edge_type == "causes":
            builder = builder.as_causal()
        elif edge_type == "supports":
            builder = builder.as_support()
        elif edge_type == "contradicts":
            builder = builder.as_contradiction()

        # Apply overrides
        builder = builder.with_data(**overrides)

        # Create edge
        schema = builder.build()
        edge = KnowledgeEdge(**schema.dict())

        if session:
            session.add(edge)
            session.commit()
            session.refresh(edge)

        return edge

    @staticmethod
    def create_knowledge_graph(
        session: Session,
        num_nodes: int = 20,
        connectivity: float = 0.1,
        node_types: Optional[List[str]] = None,
        edge_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Create a connected knowledge graph.

        Args:
            session: Database session
            num_nodes: Number of nodes to create
            connectivity: Edge creation probability (0-1)
            node_types: List of node types to use
            edge_types: List of edge types to use

        Returns:
            Dictionary with nodes, edges, and statistics
        """
        if node_types is None:
            node_types = ["concept", "entity", "fact", "observation"]
        if edge_types is None:
            edge_types = ["relates_to", "causes", "supports", "derived_from"]

        nodes = []
        edges = []

        # Create nodes
        for i in range(num_nodes):
            node_type = random.choice(node_types)
            node = KnowledgeGraphFactory.create_node(
                session,
                type=node_type,
                label=f"{node_type.capitalize()}_{i:03d}",
                confidence=random.uniform(0.5, 1.0),
            )
            nodes.append(node)

        # Create edges based on connectivity
        for i, source in enumerate(nodes):
            for j, target in enumerate(nodes):
                if i >= j:  # Skip self-loops and duplicate edges
                    continue

                if random.random() < connectivity:
                    edge_type = random.choice(edge_types)
                    edge = KnowledgeGraphFactory.create_edge(
                        session,
                        source=source,
                        target=target,
                        type=edge_type,
                        confidence=random.uniform(0.3, 1.0),
                    )
                    edges.append(edge)

        # Calculate statistics
        node_degrees = {node.id: 0 for node in nodes}
        for edge in edges:
            node_degrees[edge.source_id] += 1
            node_degrees[edge.target_id] += 1

        avg_degree = sum(node_degrees.values()) / len(nodes) if nodes else 0

        return {
            "nodes": nodes,
            "edges": edges,
            "statistics": {
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "avg_degree": avg_degree,
                "connectivity": (
                    len(edges) / (num_nodes * (num_nodes - 1) / 2)
                    if num_nodes > 1
                    else 0
                ),
                "node_type_distribution": {
                    node_type: sum(1 for n in nodes if n.type == node_type)
                    for node_type in node_types
                },
                "edge_type_distribution": {
                    edge_type: sum(1 for e in edges if e.type == edge_type)
                    for edge_type in edge_types
                },
            },
        }

    @staticmethod
    def create_agent_knowledge_scenario(
        session: Session,
        agent: Agent,
        num_observations: int = 10,
        num_inferences: int = 5,
    ) -> Dict[str, Any]:
        """Create a knowledge scenario for a specific agent.

        Args:
            session: Database session
            agent: Agent creating the knowledge
            num_observations: Number of observations to create
            num_inferences: Number of inferences to derive

        Returns:
            Dictionary with created knowledge elements
        """
        observations = []
        inferences = []
        edges = []

        # Create observations
        for i in range(num_observations):
            obs = KnowledgeGraphFactory.create_node(
                session,
                type="observation",
                label=f"Agent {agent.name} observation {i}",
                creator_agent_id=agent.id,
                confidence=random.uniform(0.7, 1.0),
                source=f"agent_{agent.id}",
            )
            observations.append(obs)

        # Create inferences based on observations
        for i in range(num_inferences):
            # Select random observations to base inference on
            num_sources = random.randint(1, min(3, len(observations)))
            source_obs = random.sample(observations, num_sources)

            # Create inference node
            inference = KnowledgeGraphFactory.create_node(
                session,
                type="inference",
                label=f"Inference from {len(source_obs)} observations",
                creator_agent_id=agent.id,
                confidence=min(o.confidence for o in source_obs) * 0.9,
                properties={
                    "inference_type": random.choice(
                        ["deductive", "inductive", "abductive"]
                    ),
                    "certainty": random.uniform(0.5, 0.9),
                },
            )
            inferences.append(inference)

            # Link observations to inference
            for obs in source_obs:
                edge = KnowledgeGraphFactory.create_edge(
                    session,
                    source=obs,
                    target=inference,
                    type="derived_from",
                    confidence=obs.confidence,
                )
                edges.append(edge)

        return {
            "agent": agent,
            "observations": observations,
            "inferences": inferences,
            "edges": edges,
            "total_knowledge_created": len(observations) + len(inferences),
        }


class PerformanceDataFactory:
    """Factory for creating large-scale performance test data."""

    @staticmethod
    def create_performance_scenario(
        session: Session, config: Optional[PerformanceTestConfigSchema] = None
    ) -> Dict[str, Any]:
        """Create a complete performance test scenario.

        Args:
            session: Database session
            config: Performance test configuration

        Returns:
            Dictionary with all created entities and statistics
        """
        if config is None:
            config = PerformanceTestConfigSchema()

        # Set random seed for reproducibility
        if config.seed:
            random.seed(config.seed)

        start_time = datetime.utcnow()
        results = {
            "config": config.dict(),
            "agents": [],
            "coalitions": [],
            "knowledge_graph": None,
            "statistics": {},
            "timing": {},
        }

        # Create agents in batches
        agent_start = datetime.utcnow()
        for batch_num in range(0, config.num_agents, config.batch_size):
            batch_size = min(config.batch_size, config.num_agents - batch_num)
            agents = AgentFactory.create_batch(
                session,
                count=batch_size,
                name_prefix=f"PerfAgent_B{batch_num//config.batch_size}",
                template=random.choice(
                    ["grid_world", "resource_collector", "explorer"]
                ),
                status=DBAgentStatus.ACTIVE,
                distribute_positions=True,
                position_bounds={"min": [0, 0], "max": [100, 100]},
            )
            results["agents"].extend(agents)

        results["timing"]["agent_creation"] = (
            datetime.utcnow() - agent_start
        ).total_seconds()

        # Create coalitions with agent distribution
        coalition_start = datetime.utcnow()
        if config.num_coalitions > 0:
            agents_per_coalition = max(
                3, config.num_agents // config.num_coalitions
            )

            for i in range(config.num_coalitions):
                # Select agents for this coalition
                start_idx = i * agents_per_coalition
                end_idx = min(
                    start_idx + agents_per_coalition, len(results["agents"])
                )
                coalition_agents = results["agents"][start_idx:end_idx]

                # Add some random agents from other coalitions (overlap)
                if i > 0 and random.random() < 0.3:
                    additional = random.sample(
                        results["agents"][:start_idx], min(5, start_idx)
                    )
                    coalition_agents.extend(additional)

                coalition = CoalitionFactory.create(
                    session,
                    agents=coalition_agents,
                    name=f"PerfCoalition_{i:03d}",
                    status=DBCoalitionStatus.ACTIVE,
                )
                results["coalitions"].append(coalition)

        results["timing"]["coalition_creation"] = (
            datetime.utcnow() - coalition_start
        ).total_seconds()

        # Create knowledge graph
        knowledge_start = datetime.utcnow()
        if config.num_knowledge_nodes > 0:
            kg = KnowledgeGraphFactory.create_knowledge_graph(
                session,
                num_nodes=config.num_knowledge_nodes,
                connectivity=config.knowledge_graph_connectivity,
            )
            results["knowledge_graph"] = kg

            # Link some nodes to agents
            if results["agents"] and kg["nodes"]:
                for _ in range(min(100, len(kg["nodes"]) // 2)):
                    node = random.choice(kg["nodes"])
                    agent = random.choice(results["agents"])
                    node.creator_agent_id = agent.id
                session.commit()

        results["timing"]["knowledge_creation"] = (
            datetime.utcnow() - knowledge_start
        ).total_seconds()

        # Calculate statistics
        results["statistics"] = {
            "total_agents": len(results["agents"]),
            "total_coalitions": len(results["coalitions"]),
            "avg_coalition_size": (
                sum(len(c.agents) for c in results["coalitions"])
                / len(results["coalitions"])
                if results["coalitions"]
                else 0
            ),
            "total_knowledge_nodes": (
                len(results["knowledge_graph"]["nodes"])
                if results["knowledge_graph"]
                else 0
            ),
            "total_knowledge_edges": (
                len(results["knowledge_graph"]["edges"])
                if results["knowledge_graph"]
                else 0
            ),
            "total_creation_time": (
                datetime.utcnow() - start_time
            ).total_seconds(),
        }

        return results

    @staticmethod
    def create_stress_test_data(
        session: Session, scale_factor: int = 10
    ) -> Dict[str, Any]:
        """Create stress test data with configurable scale.

        Args:
            session: Database session
            scale_factor: Multiplier for data volume (10 = 10x normal)

        Returns:
            Dictionary with created data and performance metrics
        """
        config = PerformanceTestConfigSchema(
            num_agents=100 * scale_factor,
            num_coalitions=10 * scale_factor,
            num_knowledge_nodes=1000 * scale_factor,
            knowledge_graph_connectivity=0.05,  # Lower connectivity for large graphs
            batch_size=min(1000, 100 * scale_factor),
            enable_metrics=True,
        )

        return PerformanceDataFactory.create_performance_scenario(
            session, config
        )
