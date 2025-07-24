"""Test data factories for creating realistic test objects.

Provides factory classes for creating test data with sensible defaults,
supporting both unit tests and performance benchmarks.
"""

import random
import string
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from database.models import Agent, AgentStatus, Coalition, KnowledgeEdge, KnowledgeNode


class BaseFactory:
    """Base factory with common utility methods."""

    @staticmethod
    def random_string(length: int = 10) -> str:
        """Generate a random string."""
        return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))

    @staticmethod
    def random_uuid() -> str:
        """Generate a random UUID string."""
        return str(uuid.uuid4())

    @staticmethod
    def random_timestamp(days_ago: int = 30) -> datetime:
        """Generate a random timestamp within the last N days."""
        now = datetime.utcnow()
        delta = timedelta(days=random.randint(0, days_ago))
        return now - delta


class AgentFactory(BaseFactory):
    """Factory for creating test agents."""

    # Agent type options
    AGENT_TYPES = [
        "resource_collector",
        "explorer",
        "defender",
        "coordinator",
        "scout",
    ]

    # Capability options
    CAPABILITIES = [
        "resource_collection",
        "exploration",
        "defense",
        "coordination",
        "communication",
        "analysis",
        "planning",
        "execution",
    ]

    @classmethod
    def create(cls, session: Session, **kwargs) -> Agent:
        """Create a test agent with sensible defaults.

        Args:
            session: Database session
            **kwargs: Override default values

        Returns:
            Created agent instance
        """
        defaults = {
            "agent_id": f"agent_{cls.random_uuid()}",
            "name": f"Agent_{cls.random_string(6)}",
            "agent_type": random.choice(cls.AGENT_TYPES),
            "status": AgentStatus.ACTIVE,
            "belief_state": {
                "initialized": True,
                "last_update": datetime.utcnow().isoformat(),
                "confidence": random.uniform(0.5, 1.0),
                "observations": random.randint(0, 100),
            },
            "position": {
                "lat": 37.7749 + random.uniform(-0.1, 0.1),
                "lon": -122.4194 + random.uniform(-0.1, 0.1),
                "elevation": random.uniform(0, 100),
            },
            "capabilities": random.sample(cls.CAPABILITIES, k=random.randint(2, 5)),
            "parameters": {
                "speed": random.uniform(0.5, 2.0),
                "range": random.uniform(10, 100),
                "energy": random.uniform(50, 100),
            },
            "metrics": {
                "total_actions": random.randint(0, 1000),
                "success_rate": random.uniform(0.7, 0.95),
                "avg_response_time": random.uniform(10, 100),
            },
            "created_at": cls.random_timestamp(days_ago=7),
            "updated_at": datetime.utcnow(),
        }
        defaults.update(kwargs)

        agent = Agent(**defaults)
        session.add(agent)
        session.commit()
        session.refresh(agent)
        return agent

    @classmethod
    def create_batch(cls, session: Session, count: int = 10, **kwargs) -> List[Agent]:
        """Create multiple agents in a batch.

        Args:
            session: Database session
            count: Number of agents to create
            **kwargs: Override default values for all agents

        Returns:
            List of created agents
        """
        agents = []
        for i in range(count):
            agent_kwargs = kwargs.copy()
            agent_kwargs["agent_id"] = f"batch_agent_{i}_{cls.random_uuid()}"
            agent_kwargs["name"] = f"BatchAgent_{i}"
            agents.append(Agent(**cls._get_defaults(**agent_kwargs)))

        session.add_all(agents)
        session.commit()

        for agent in agents:
            session.refresh(agent)

        return agents

    @classmethod
    def _get_defaults(cls, **kwargs) -> Dict[str, Any]:
        """Get default values with overrides."""
        defaults = {
            "agent_id": f"agent_{cls.random_uuid()}",
            "name": f"Agent_{cls.random_string(6)}",
            "agent_type": random.choice(cls.AGENT_TYPES),
            "status": AgentStatus.ACTIVE,
            "belief_state": {"initialized": True},
            "position": {"lat": 37.7749, "lon": -122.4194},
            "capabilities": random.sample(cls.CAPABILITIES, k=3),
            "created_at": datetime.utcnow(),
        }
        defaults.update(kwargs)
        return defaults


class CoalitionFactory(BaseFactory):
    """Factory for creating test coalitions."""

    COALITION_GOALS = [
        "resource_optimization",
        "area_exploration",
        "threat_response",
        "knowledge_sharing",
        "task_distribution",
    ]

    @classmethod
    def create(cls, session: Session, agents: Optional[List[Agent]] = None, **kwargs) -> Coalition:
        """Create a test coalition with agents.

        Args:
            session: Database session
            agents: List of agents to include (creates new ones if not provided)
            **kwargs: Override default values

        Returns:
            Created coalition instance
        """
        if agents is None:
            # Create some agents for the coalition
            agents = AgentFactory.create_batch(session, count=random.randint(3, 7))

        defaults = {
            "coalition_id": f"coalition_{cls.random_uuid()}",
            "name": f"Coalition_{cls.random_string(6)}",
            "goal": random.choice(cls.COALITION_GOALS),
            "status": "active",
            "formation_strategy": "capability_based",
            "metadata": {
                "formation_time": datetime.utcnow().isoformat(),
                "performance_score": random.uniform(0.6, 0.9),
                "cohesion": random.uniform(0.5, 1.0),
            },
            "created_at": cls.random_timestamp(days_ago=3),
        }
        defaults.update(kwargs)

        # Remove agents from defaults if present (use parameter instead)
        defaults.pop("agents", [])

        coalition = Coalition(**defaults)
        session.add(coalition)
        session.commit()

        # Add agents to coalition
        for agent in agents:
            coalition.agents.append(agent)

        session.commit()
        session.refresh(coalition)
        return coalition


class KnowledgeGraphFactory(BaseFactory):
    """Factory for creating test knowledge graphs."""

    NODE_TYPES = ["concept", "entity", "fact", "belie", "observation"]
    EDGE_TYPES = [
        "relates_to",
        "causes",
        "contradicts",
        "supports",
        "derived_from",
    ]

    @classmethod
    def create_node(cls, session: Session, **kwargs) -> KnowledgeNode:
        """Create a single knowledge node.

        Args:
            session: Database session
            **kwargs: Override default values

        Returns:
            Created knowledge node
        """
        defaults = {
            "node_id": f"node_{cls.random_uuid()}",
            "node_type": random.choice(cls.NODE_TYPES),
            "content": f"Knowledge content {cls.random_string(20)}",
            "metadata": {
                "source": f"agent_{cls.random_string(6)}",
                "confidence": random.uniform(0.5, 1.0),
                "timestamp": datetime.utcnow().isoformat(),
                "tags": random.sample(["physics", "behavior", "environment", "agent"], k=2),
            },
            "embedding": [random.random() for _ in range(128)],  # Mock embedding
            "created_at": cls.random_timestamp(days_ago=14),
            "updated_at": datetime.utcnow(),
        }
        defaults.update(kwargs)

        node = KnowledgeNode(**defaults)
        session.add(node)
        session.commit()
        session.refresh(node)
        return node

    @classmethod
    def create_edge(
        cls,
        session: Session,
        source_node: KnowledgeNode,
        target_node: KnowledgeNode,
        **kwargs,
    ) -> KnowledgeEdge:
        """Create a knowledge edge between two nodes.

        Args:
            session: Database session
            source_node: Source node
            target_node: Target node
            **kwargs: Override default values

        Returns:
            Created knowledge edge
        """
        defaults = {
            "edge_id": f"edge_{cls.random_uuid()}",
            "source_node_id": source_node.node_id,
            "target_node_id": target_node.node_id,
            "edge_type": random.choice(cls.EDGE_TYPES),
            "weight": random.uniform(0.1, 1.0),
            "metadata": {
                "created_by": f"agent_{cls.random_string(6)}",
                "confidence": random.uniform(0.5, 1.0),
                "evidence_count": random.randint(1, 10),
            },
            "created_at": cls.random_timestamp(days_ago=7),
        }
        defaults.update(kwargs)

        edge = KnowledgeEdge(**defaults)
        session.add(edge)
        session.commit()
        session.refresh(edge)
        return edge

    @classmethod
    def create_connected_graph(
        cls, session: Session, num_nodes: int = 10, connectivity: float = 0.3
    ) -> Dict[str, Any]:
        """Create a connected knowledge graph.

        Args:
            session: Database session
            num_nodes: Number of nodes to create
            connectivity: Probability of edge between any two nodes (0-1)

        Returns:
            Dictionary with created nodes and edges
        """
        # Create nodes
        nodes = []
        for i in range(num_nodes):
            node = cls.create_node(
                session,
                node_id=f"graph_node_{i}_{cls.random_uuid()}",
                content=f"Graph concept {i}: {cls.random_string(10)}",
            )
            nodes.append(node)

        # Create edges based on connectivity
        edges = []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if random.random() < connectivity:
                    edge = cls.create_edge(
                        session,
                        source_node=nodes[i],
                        target_node=nodes[j],
                        edge_id=f"graph_edge_{i}_{j}_{cls.random_uuid()}",
                    )
                    edges.append(edge)

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "avg_degree": (2 * len(edges)) / len(nodes) if nodes else 0,
            },
        }


class TestDataGenerator:
    """Generate complex test scenarios."""

    @staticmethod
    def create_multi_agent_scenario(
        session: Session, num_agents: int = 20, num_coalitions: int = 4
    ) -> Dict[str, Any]:
        """Create a multi-agent scenario with coalitions.

        Args:
            session: Database session
            num_agents: Total number of agents
            num_coalitions: Number of coalitions to form

        Returns:
            Dictionary with created entities
        """
        # Create agents
        agents = AgentFactory.create_batch(session, count=num_agents)

        # Distribute agents among coalitions
        agents_per_coalition = num_agents // num_coalitions
        coalitions = []

        for i in range(num_coalitions):
            start_idx = i * agents_per_coalition
            end_idx = start_idx + agents_per_coalition
            if i == num_coalitions - 1:  # Last coalition gets remaining agents
                end_idx = num_agents

            coalition_agents = agents[start_idx:end_idx]
            coalition = CoalitionFactory.create(
                session,
                agents=coalition_agents,
                name=f"TestCoalition_{i}",
                goal=f"test_goal_{i}",
            )
            coalitions.append(coalition)

        return {
            "agents": agents,
            "coalitions": coalitions,
            "unassigned_agents": [],  # All agents assigned in this scenario
            "stats": {
                "total_agents": num_agents,
                "total_coalitions": num_coalitions,
                "avg_coalition_size": num_agents / num_coalitions,
            },
        }

    @staticmethod
    def create_knowledge_evolution_scenario(
        session: Session, initial_nodes: int = 50, evolution_steps: int = 10
    ) -> Dict[str, Any]:
        """Create a scenario for testing knowledge graph evolution.

        Args:
            session: Database session
            initial_nodes: Number of initial knowledge nodes
            evolution_steps: Number of evolution iterations

        Returns:
            Dictionary with evolution history
        """
        history = []

        # Create initial graph
        initial_graph = KnowledgeGraphFactory.create_connected_graph(
            session, num_nodes=initial_nodes, connectivity=0.2
        )
        history.append(
            {
                "step": 0,
                "action": "initial",
                "nodes_added": initial_nodes,
                "edges_added": len(initial_graph["edges"]),
            }
        )

        # Simulate evolution
        all_nodes = initial_graph["nodes"].copy()

        for step in range(1, evolution_steps + 1):
            action = random.choice(["add_nodes", "add_edges", "update_nodes"])

            if action == "add_nodes":
                # Add new nodes
                new_nodes = []
                num_new = random.randint(1, 5)
                for _ in range(num_new):
                    node = KnowledgeGraphFactory.create_node(
                        session,
                        content=f"Evolution step {step}: {BaseFactory.random_string(10)}",
                    )
                    new_nodes.append(node)
                    all_nodes.append(node)

                history.append(
                    {
                        "step": step,
                        "action": "add_nodes",
                        "nodes_added": num_new,
                        "edges_added": 0,
                    }
                )

            elif action == "add_edges" and len(all_nodes) > 1:
                # Add new edges between existing nodes
                num_edges = random.randint(1, min(5, len(all_nodes) // 2))
                edges_added = 0

                for _ in range(num_edges):
                    source = random.choice(all_nodes)
                    target = random.choice(all_nodes)
                    if source.node_id != target.node_id:
                        KnowledgeGraphFactory.create_edge(session, source, target)
                        edges_added += 1

                history.append(
                    {
                        "step": step,
                        "action": "add_edges",
                        "nodes_added": 0,
                        "edges_added": edges_added,
                    }
                )

            else:  # update_nodes
                # Update metadata of existing nodes
                num_updates = random.randint(1, min(10, len(all_nodes)))
                nodes_to_update = random.sample(all_nodes, k=num_updates)

                for node in nodes_to_update:
                    node.metadata["evolution_step"] = step
                    node.metadata["updated"] = True
                    node.updated_at = datetime.utcnow()

                session.commit()

                history.append(
                    {
                        "step": step,
                        "action": "update_nodes",
                        "nodes_updated": num_updates,
                        "edges_added": 0,
                    }
                )

        return {
            "initial_graph": initial_graph,
            "final_node_count": len(all_nodes),
            "evolution_history": history,
            "total_steps": evolution_steps,
        }


if __name__ == "__main__":
    # Example usage
    from test_config import create_test_engine, setup_test_database, teardown_test_database

    print("Testing data factories...")

    # Create test engine and setup database
    engine = create_test_engine(use_sqlite=True)
    setup_test_database(engine)

    from sqlalchemy.orm import sessionmaker

    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # Test agent creation
        print("\nCreating test agent...")
        agent = AgentFactory.create(session, name="TestAgent001")
        print(f"✅ Created agent: {agent.agent_id} - {agent.name}")

        # Test batch creation
        print("\nCreating batch of agents...")
        agents = AgentFactory.create_batch(session, count=5)
        print(f"✅ Created {len(agents)} agents")

        # Test coalition creation
        print("\nCreating coalition...")
        coalition = CoalitionFactory.create(session, agents=agents[:3])
        print(f"✅ Created coalition: {coalition.coalition_id} with {len(coalition.agents)} agents")

        # Test knowledge graph
        print("\nCreating knowledge graph...")
        graph = KnowledgeGraphFactory.create_connected_graph(
            session, num_nodes=10, connectivity=0.3
        )
        print(
            f"✅ Created graph with {graph['stats']['node_count']} nodes and {graph['stats']['edge_count']} edges"
        )

        # Test complex scenario
        print("\nCreating multi-agent scenario...")
        scenario = TestDataGenerator.create_multi_agent_scenario(
            session, num_agents=20, num_coalitions=4
        )
        print(
            f"✅ Created scenario with {scenario['stats']['total_agents']} agents in {scenario['stats']['total_coalitions']} coalitions"
        )

    finally:
        session.close()
        teardown_test_database(engine)
