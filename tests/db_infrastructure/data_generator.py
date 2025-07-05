"""Test data generation utilities for load testing."""

import json
import random
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from faker import Faker

fake = Faker()


class TestDataGenerator:
    """Generate realistic test data for FreeAgentics database."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional random seed for reproducibility."""
        if seed:
            random.seed(seed)
            np.random.seed(seed)
            Faker.seed(seed)

        self.fake = Faker()

        # Agent templates
        self.agent_templates = [
            "explorer",
            "analyzer",
            "coordinator",
            "executor",
            "monitor",
            "planner",
            "optimizer",
            "validator",
        ]

        # Knowledge node types
        self.node_types = [
            "concept",
            "entity",
            "relation",
            "process",
            "state",
            "constraint",
            "goal",
            "belie",
        ]

        # Edge types
        self.edge_types = [
            "causes",
            "requires",
            "enables",
            "conflicts_with",
            "supports",
            "implements",
            "belongs_to",
            "derives_from",
        ]

    def generate_agent(
        self, template: Optional[str] = None, status: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a single agent with realistic data."""
        if not template:
            template = random.choice(self.agent_templates)

        if not status:
            status = random.choice(["PENDING", "ACTIVE", "PAUSED"])

        # Generate GMN specification
        gmn_spec = self._generate_gmn_spec(template)

        # Generate PyMDP configuration
        pymdp_config = self._generate_pymdp_config(template)

        # Generate beliefs and preferences
        beliefs = self._generate_beliefs()
        preferences = self._generate_preferences()

        # Generate position data
        position = {
            "x": random.uniform(-100, 100),
            "y": random.uniform(-100, 100),
            "z": random.uniform(0, 10),
            "grid_x": random.randint(0, 50),
            "grid_y": random.randint(0, 50),
        }

        # Generate metrics
        metrics = {
            "total_inferences": random.randint(0, 1000),
            "successful_actions": random.randint(0, 500),
            "failed_actions": random.randint(0, 100),
            "avg_inference_time": random.uniform(0.1, 2.0),
            "memory_usage_mb": random.uniform(50, 500),
        }

        # Generate parameters
        parameters = {
            "learning_rate": random.uniform(0.001, 0.1),
            "exploration_rate": random.uniform(0.1, 0.5),
            "discount_factor": random.uniform(0.9, 0.99),
            "planning_horizon": random.randint(5, 20),
            "batch_size": random.choice([16, 32, 64, 128]),
        }

        return {
            "id": str(uuid.uuid4()),
            "name": f"{fake.first_name()}_{template}_{fake.random_number(3)}",
            "template": template,
            "status": status,
            "gmn_spec": json.dumps(gmn_spec),
            "pymdp_config": json.dumps(pymdp_config),
            "beliefs": json.dumps(beliefs),
            "preferences": json.dumps(preferences),
            "position": json.dumps(position),
            "metrics": json.dumps(metrics),
            "parameters": json.dumps(parameters),
            "inference_count": random.randint(0, 1000),
            "total_steps": random.randint(0, 10000),
            "created_at": fake.date_time_between(start_date="-30d", end_date="now"),
            "last_active": fake.date_time_between(start_date="-1d", end_date="now"),
        }

    def generate_agents(self, count: int) -> List[Dict[str, Any]]:
        """Generate multiple agents."""
        return [self.generate_agent() for _ in range(count)]

    def generate_coalition(
        self, agent_ids: Optional[List[str]] = None, status: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a coalition with objectives and capabilities."""
        if not status:
            status = random.choice(["FORMING", "ACTIVE", "DISBANDING"])

        objectives = self._generate_objectives()
        required_capabilities = self._generate_capabilities()

        coalition = {
            "id": str(uuid.uuid4()),
            "name": f"Coalition_{fake.word().capitalize()}_{fake.random_number(3)}",
            "description": fake.text(max_nb_chars=200),
            "status": status,
            "objectives": json.dumps(objectives),
            "required_capabilities": json.dumps(required_capabilities),
            "achieved_objectives": json.dumps(
                random.sample(objectives, k=random.randint(0, len(objectives)))
            ),
            "performance_score": random.uniform(0.0, 1.0),
            "cohesion_score": random.uniform(0.0, 1.0),
            "created_at": fake.date_time_between(start_date="-7d", end_date="now"),
        }

        if status == "DISSOLVED":
            coalition["dissolved_at"] = fake.date_time_between(
                start_date=coalition["created_at"], end_date="now"
            )

        return coalition

    def generate_agent_coalition_membership(
        self, agent_id: str, coalition_id: str
    ) -> Dict[str, Any]:
        """Generate agent-coalition membership data."""
        return {
            "agent_id": agent_id,
            "coalition_id": coalition_id,
            "role": random.choice(["LEADER", "COORDINATOR", "MEMBER", "OBSERVER"]),
            "joined_at": fake.date_time_between(start_date="-7d", end_date="now"),
            "contribution_score": random.uniform(0.0, 1.0),
        }

    def generate_knowledge_node(self, creator_agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a knowledge node."""
        node_type = random.choice(self.node_types)

        properties = self._generate_node_properties(node_type)

        return {
            "id": str(uuid.uuid4()),
            "type": node_type,
            "label": f"{node_type}_{fake.word()}_{fake.random_number(3)}",
            "properties": json.dumps(properties),
            "version": random.randint(1, 5),
            "is_current": random.choice([True, True, True, False]),  # 75% current
            "confidence": random.uniform(0.5, 1.0),
            "source": random.choice(["observation", "inference", "communication", "prior"]),
            "creator_agent_id": creator_agent_id,
            "created_at": fake.date_time_between(start_date="-30d", end_date="now"),
        }

    def generate_knowledge_edge(self, source_id: str, target_id: str) -> Dict[str, Any]:
        """Generate a knowledge edge between nodes."""
        edge_type = random.choice(self.edge_types)

        properties = self._generate_edge_properties(edge_type)

        return {
            "id": str(uuid.uuid4()),
            "source_id": source_id,
            "target_id": target_id,
            "type": edge_type,
            "properties": json.dumps(properties),
            "confidence": random.uniform(0.3, 1.0),
            "created_at": fake.date_time_between(start_date="-30d", end_date="now"),
        }

    def generate_complete_dataset(
        self,
        num_agents: int = 100,
        num_coalitions: int = 20,
        num_knowledge_nodes: int = 1000,
        num_edges: int = 2000,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate a complete interconnected dataset."""
        # Generate agents
        agents = self.generate_agents(num_agents)
        agent_ids = [agent["id"] for agent in agents]

        # Generate coalitions
        coalitions = [self.generate_coalition() for _ in range(num_coalitions)]
        coalition_ids = [coalition["id"] for coalition in coalitions]

        # Generate memberships (each agent in 0-3 coalitions)
        memberships = []
        for agent_id in agent_ids:
            num_memberships = random.randint(0, 3)
            selected_coalitions = random.sample(
                coalition_ids, min(num_memberships, len(coalition_ids))
            )

            for coalition_id in selected_coalitions:
                memberships.append(self.generate_agent_coalition_membership(agent_id, coalition_id))

        # Generate knowledge nodes
        knowledge_nodes = []
        for _ in range(num_knowledge_nodes):
            creator_id = random.choice(agent_ids) if random.random() > 0.2 else None
            knowledge_nodes.append(self.generate_knowledge_node(creator_id))

        node_ids = [node["id"] for node in knowledge_nodes]

        # Generate edges
        edges = []
        for _ in range(num_edges):
            source_id = random.choice(node_ids)
            target_id = random.choice([nid for nid in node_ids if nid != source_id])
            edges.append(self.generate_knowledge_edge(source_id, target_id))

        return {
            "agents": agents,
            "coalitions": coalitions,
            "memberships": memberships,
            "knowledge_nodes": knowledge_nodes,
            "knowledge_edges": edges,
        }

    def _generate_gmn_spec(self, template: str) -> Dict[str, Any]:
        """Generate a GMN specification based on template."""
        base_spec = {
            "modalities": ["vision", "proprioception", "communication"],
            "hidden_state_dimensions": {
                "belie": random.randint(32, 128),
                "intention": random.randint(16, 64),
            },
            "action_dimensions": random.randint(4, 16),
        }

        # Template-specific configurations
        if template == "explorer":
            base_spec["exploration_bonus"] = random.uniform(0.1, 0.5)
        elif template == "analyzer":
            base_spec["analysis_depth"] = random.randint(3, 10)
        elif template == "coordinator":
            base_spec["coordination_range"] = random.uniform(10, 50)

        return base_spec

    def _generate_pymdp_config(self, template: str) -> Dict[str, Any]:
        """Generate PyMDP configuration."""
        return {
            "num_states": [random.randint(10, 50) for _ in range(3)],
            "num_observations": [random.randint(5, 20) for _ in range(3)],
            "num_controls": [random.randint(2, 8) for _ in range(2)],
            "planning_horizon": random.randint(3, 10),
            "inference_algo": random.choice(["marginal", "mmp", "vmp"]),
            "use_cuda": False,
            "seed": random.randint(0, 10000),
        }

    def _generate_beliefs(self) -> Dict[str, Any]:
        """Generate agent beliefs."""
        return {
            "world_state": {
                "explored_ratio": random.uniform(0, 1),
                "known_entities": random.randint(0, 100),
                "uncertainty_level": random.uniform(0, 1),
            },
            "self_model": {
                "capabilities": random.sample(
                    ["navigate", "analyze", "communicate", "plan", "execute"],
                    k=random.randint(2, 5),
                ),
                "limitations": random.sample(
                    ["energy", "knowledge", "perception", "computation"], k=random.randint(1, 3)
                ),
            },
        }

    def _generate_preferences(self) -> Dict[str, Any]:
        """Generate agent preferences."""
        return {
            "exploration_vs_exploitation": random.uniform(0, 1),
            "risk_tolerance": random.uniform(0, 1),
            "cooperation_tendency": random.uniform(0, 1),
            "goal_priorities": {
                "survival": random.uniform(0.5, 1),
                "knowledge": random.uniform(0, 1),
                "social": random.uniform(0, 1),
                "achievement": random.uniform(0, 1),
            },
        }

    def _generate_objectives(self) -> List[Dict[str, Any]]:
        """Generate coalition objectives."""
        objective_types = [
            "explore_region",
            "analyze_data",
            "coordinate_agents",
            "optimize_resource",
            "achieve_consensus",
            "defend_position",
        ]

        num_objectives = random.randint(1, 4)
        objectives = []

        for _ in range(num_objectives):
            objectives.append(
                {
                    "type": random.choice(objective_types),
                    "priority": random.choice(["high", "medium", "low"]),
                    "deadline": fake.date_time_between(
                        start_date="now", end_date="+30d"
                    ).isoformat(),
                    "success_criteria": {
                        "metric": fake.word(),
                        "target_value": random.uniform(0.5, 1.0),
                    },
                }
            )

        return objectives

    def _generate_capabilities(self) -> List[str]:
        """Generate required capabilities for coalition."""
        all_capabilities = [
            "navigation",
            "analysis",
            "communication",
            "planning",
            "execution",
            "monitoring",
            "optimization",
            "validation",
            "negotiation",
            "learning",
        ]

        num_required = random.randint(2, 6)
        return random.sample(all_capabilities, num_required)

    def _generate_node_properties(self, node_type: str) -> Dict[str, Any]:
        """Generate properties for a knowledge node."""
        base_props = {
            "timestamp": fake.date_time_between(start_date="-30d", end_date="now").isoformat(),
            "relevance": random.uniform(0, 1),
            "certainty": random.uniform(0.5, 1),
        }

        # Type-specific properties
        if node_type == "entity":
            base_props.update(
                {
                    "entity_type": random.choice(["agent", "object", "location", "resource"]),
                    "attributes": {fake.word(): fake.word() for _ in range(random.randint(1, 5))},
                }
            )
        elif node_type == "process":
            base_props.update(
                {
                    "duration": random.uniform(0.1, 100),
                    "inputs": [fake.word() for _ in range(random.randint(0, 3))],
                    "outputs": [fake.word() for _ in range(random.randint(1, 3))],
                }
            )

        return base_props

    def _generate_edge_properties(self, edge_type: str) -> Dict[str, Any]:
        """Generate properties for a knowledge edge."""
        base_props = {
            "strength": random.uniform(0.1, 1.0),
            "bidirectional": random.choice([True, False]),
        }

        # Type-specific properties
        if edge_type == "causes":
            base_props["causality_strength"] = random.uniform(0.3, 1.0)
            base_props["delay"] = random.uniform(0, 10)
        elif edge_type == "requires":
            base_props["necessity"] = random.choice(["mandatory", "optional", "conditional"])

        return base_props
