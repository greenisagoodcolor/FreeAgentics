"""
Comprehensive Test Data Factory for FreeAgentics.

This module provides a centralized factory system for creating test data
with realistic defaults, relationships, and behaviors. It ensures consistency
across all tests and reduces boilerplate code.

Following the Test Data Factory pattern for systematic test coverage.
"""

import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Import domain models
try:
    from agents.base.data_model import Agent, Position

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

    # Define minimal dataclasses for testing when imports fail
    @dataclass
    class Position:
        x: float
        y: float
        z: float = 0.0

    @dataclass
    class Agent:
        agent_id: str
        name: str
        agent_type: str
        position: Position


class DataFactory:
    """Central factory for creating test data with realistic defaults."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize factory with optional random seed for reproducibility."""
        self.seed = seed
        self._random_state = random.Random(seed) if seed is not None else random.Random()
        self._np_random_state = (
            np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        )
        self._id_counter = 0
        self._agent_names = [
            "Explorer Alpha",
            "Guardian Beta",
            "Merchant Gamma",
            "Scholar Delta",
            "Pioneer Epsilon",
            "Defender Zeta",
            "Trader Eta",
            "Researcher Theta",
        ]
        self._coalition_names = [
            "United Explorers",
            "Trade Federation",
            "Knowledge Seekers",
            "Defense Alliance",
            "Resource Collective",
            "Innovation Hub",
            "Prosperity Guild",
            "Discovery Network",
        ]

    def _next_id(self, prefix: str = "id") -> str:
        """Generate next unique ID."""
        self._id_counter += 1
        return f"{prefix}_{self._id_counter:04d}"

    def create_position(self, x: float = None, y: float = None, z: float = None) -> Position:
        """Create a position with optional coordinates."""
        if x is None:
            x = self._random_state.uniform(-100, 100)
        if y is None:
            y = self._random_state.uniform(-100, 100)
        if z is None:
            z = self._random_state.uniform(0, 10)

        if MODELS_AVAILABLE:
            return Position(x=x, y=y, z=z)
        else:
            return {"x": x, "y": y, "z": z}

    def create_agent(
        self,
        agent_id: str = None,
        name: str = None,
        agent_type: str = None,
        position: Position = None,
        personality: Dict[str, float] = None,
        capabilities: Dict[str, float] = None,
        resources: Dict[str, float] = None,
        constraints: Dict[str, Any] = None,
        **kwargs,
    ) -> Union[Agent, Dict[str, Any]]:
        """
        Create an agent with realistic defaults.

        Args:
            agent_id: Unique identifier (auto-generated if None)
            name: Agent name (randomly selected if None)
            agent_type: Type of agent (explorer, guardian, merchant, scholar)
            position: Starting position (random if None)
            personality: Personality traits
            capabilities: Agent capabilities/skills
            resources: Available resources
            constraints: Agent constraints
            **kwargs: Additional attributes

        Returns:
            Agent instance or dict if models not available
        """
        if agent_id is None:
            agent_id = self._next_id("agent")

        if name is None:
            name = random.choice(self._agent_names) + f" {agent_id[-4:]}"

        if agent_type is None:
            agent_type = random.choice(["explorer", "guardian", "merchant", "scholar"])

        if position is None:
            position = self.create_position()

        if personality is None:
            personality = {
                "curiosity": random.uniform(0.3, 0.9),
                "caution": random.uniform(0.2, 0.8),
                "cooperation": random.uniform(0.4, 0.9),
                "aggression": random.uniform(0.1, 0.6),
            }

        if capabilities is None:
            capabilities = self._generate_capabilities_for_type(agent_type)

        if resources is None:
            resources = {
                "energy": random.uniform(50, 150),
                "materials": random.randint(10, 100),
                "information": random.randint(5, 50),
                "currency": random.uniform(100, 1000),
            }

        if constraints is None:
            constraints = {
                "max_energy": 200,
                "max_inventory": 50,
                "communication_range": 50,
                "vision_range": 30,
            }

        agent_data = {
            "agent_id": agent_id,
            "name": name,
            "agent_type": agent_type,
            "position": position,
            "personality": personality,
            "capabilities": capabilities,
            "resources": resources,
            "constraints": constraints,
            "status": kwargs.get("status", "active"),
            "health": kwargs.get("health", 100.0),
            "experience": kwargs.get("experience", 0),
            "level": kwargs.get("level", 1),
            "created_at": kwargs.get("created_at", datetime.now()),
            "updated_at": kwargs.get("updated_at", datetime.now()),
        }

        # Add any additional kwargs
        agent_data.update(kwargs)

        if MODELS_AVAILABLE:
            return Agent(**agent_data)
        else:
            return agent_data

    def _generate_capabilities_for_type(self, agent_type: str) -> Dict[str, float]:
        """Generate type-specific capabilities."""
        base_capabilities = {
            "exploration": 0.5,
            "combat": 0.5,
            "trading": 0.5,
            "research": 0.5,
            "communication": 0.6,
            "resource_gathering": 0.5,
        }

        # Type-specific bonuses
        if agent_type == "explorer":
            base_capabilities["exploration"] = random.uniform(0.7, 0.9)
            base_capabilities["resource_gathering"] = random.uniform(0.6, 0.8)
        elif agent_type == "guardian":
            base_capabilities["combat"] = random.uniform(0.7, 0.9)
            base_capabilities["exploration"] = random.uniform(0.3, 0.5)
        elif agent_type == "merchant":
            base_capabilities["trading"] = random.uniform(0.7, 0.9)
            base_capabilities["communication"] = random.uniform(0.7, 0.9)
        elif agent_type == "scholar":
            base_capabilities["research"] = random.uniform(0.7, 0.9)
            base_capabilities["communication"] = random.uniform(0.6, 0.8)

        return base_capabilities

    def create_agent_batch(
        self, count: int = 5, agent_types: List[str] = None, **common_kwargs
    ) -> List[Union[Agent, Dict[str, Any]]]:
        """
        Create a batch of agents with varied characteristics.

        Args:
            count: Number of agents to create
            agent_types: List of agent types to use (cycles through if fewer than count)
            **common_kwargs: Common attributes for all agents

        Returns:
            List of agents
        """
        if agent_types is None:
            agent_types = ["explorer", "guardian", "merchant", "scholar"]

        agents = []
        for i in range(count):
            agent_type = agent_types[i % len(agent_types)]
            agent = self.create_agent(agent_type=agent_type, **common_kwargs)
            agents.append(agent)

        return agents

    def create_coalition(
        self,
        coalition_id: str = None,
        name: str = None,
        members: List[Union[Agent, str]] = None,
        business_type: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a coalition with realistic defaults.

        Args:
            coalition_id: Unique identifier
            name: Coalition name
            members: List of agents or agent IDs
            business_type: Type of coalition business
            **kwargs: Additional attributes

        Returns:
            Coalition data dict
        """
        if coalition_id is None:
            coalition_id = self._next_id("coalition")

        if name is None:
            name = random.choice(self._coalition_names) + f" {coalition_id[-4:]}"

        if members is None:
            members = []

        if business_type is None:
            business_type = random.choice(
                ["ResourceOptimization", "KnowledgeSharing", "DefenseAlliance", "TradePartnership"]
            )

        # Convert agents to IDs if needed
        member_ids = []
        for member in members:
            if isinstance(member, str):
                member_ids.append(member)
            elif hasattr(member, "agent_id"):
                member_ids.append(member.agent_id)
            else:
                member_ids.append(member.get("agent_id", str(member)))

        coalition_data = {
            "coalition_id": coalition_id,
            "name": name,
            "members": member_ids,
            "business_type": business_type,
            "status": kwargs.get("status", "active"),
            "formation_time": kwargs.get("formation_time", datetime.now()),
            "purpose": kwargs.get("purpose", f"Collaborative {business_type}"),
            "shared_resources": kwargs.get(
                "shared_resources", {"energy": 0.0, "materials": 0, "information": 0}
            ),
            "rules": kwargs.get(
                "rules",
                {"min_contribution": 10, "profit_sharing": "equal", "decision_making": "majority"},
            ),
            "performance": kwargs.get(
                "performance",
                {"success_rate": 0.0, "total_profit": 0.0, "member_satisfaction": 1.0},
            ),
            "synergy_score": kwargs.get("synergy_score", random.uniform(0.5, 0.9)),
        }

        coalition_data.update(kwargs)
        return coalition_data

    def create_world_cell(
        self, cell_id: str = None, position: Position = None, terrain_type: str = None, **kwargs
    ) -> Dict[str, Any]:
        """Create a world cell with realistic properties."""
        if cell_id is None:
            cell_id = self._next_id("cell")

        if position is None:
            position = self.create_position()

        if terrain_type is None:
            terrain_type = random.choice(
                ["plains", "forest", "mountain", "water", "desert", "urban"]
            )

        cell_data = {
            "cell_id": cell_id,
            "position": position,
            "terrain_type": terrain_type,
            "resources": self._generate_resources_for_terrain(terrain_type),
            "difficulty": self._get_terrain_difficulty(terrain_type),
            "visibility": random.uniform(0.5, 1.0),
            "temperature": random.uniform(-20, 40),
            "explored": False,
            "occupied_by": None,
            "structures": [],
        }

        cell_data.update(kwargs)
        return cell_data

    def _generate_resources_for_terrain(self, terrain_type: str) -> Dict[str, int]:
        """Generate terrain-specific resources."""
        base_resources = {"energy": 0, "materials": 0, "water": 0, "food": 0, "rare_minerals": 0}

        if terrain_type == "plains":
            base_resources["food"] = random.randint(20, 50)
            base_resources["water"] = random.randint(10, 30)
        elif terrain_type == "forest":
            base_resources["materials"] = random.randint(30, 60)
            base_resources["food"] = random.randint(15, 35)
        elif terrain_type == "mountain":
            base_resources["materials"] = random.randint(40, 80)
            base_resources["rare_minerals"] = random.randint(5, 15)
        elif terrain_type == "water":
            base_resources["water"] = random.randint(50, 100)
            base_resources["food"] = random.randint(10, 25)
        elif terrain_type == "desert":
            base_resources["energy"] = random.randint(30, 60)  # Solar
            base_resources["rare_minerals"] = random.randint(0, 10)
        elif terrain_type == "urban":
            base_resources["energy"] = random.randint(20, 40)
            base_resources["materials"] = random.randint(10, 30)

        return base_resources

    def _get_terrain_difficulty(self, terrain_type: str) -> float:
        """Get movement difficulty for terrain type."""
        difficulties = {
            "plains": 1.0,
            "forest": 1.5,
            "mountain": 2.5,
            "water": 3.0,
            "desert": 2.0,
            "urban": 1.2,
        }
        return difficulties.get(terrain_type, 1.0)

    def create_active_inference_state(
        self, num_states: int = 4, num_observations: int = 4, **kwargs
    ) -> Dict[str, np.ndarray]:
        """Create active inference state with proper probability distributions."""
        state = {
            # Beliefs (probability distribution over states)
            "beliefs": self._normalize(np.random.rand(num_states)),
            # Observations (one-hot or probability distribution)
            "observations": self._create_one_hot(
                random.randint(0, num_observations - 1), num_observations
            ),
            # Preferences (C vector in active inference)
            "preferences": self._normalize(np.random.rand(num_observations)),
            # Expected free energy for each action
            "expected_free_energy": np.random.rand(num_states) * 5,
            # Action probabilities
            "action_probabilities": self._normalize(np.random.rand(num_states)),
            # Precision (inverse temperature)
            "precision": kwargs.get("precision", random.uniform(0.5, 2.0)),
            # Free energy
            "free_energy": kwargs.get("free_energy", random.uniform(0, 5)),
        }

        state.update(kwargs)
        return state

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to sum to 1."""
        return arr / arr.sum()

    def _create_one_hot(self, index: int, size: int) -> np.ndarray:
        """Create one-hot encoded vector."""
        arr = np.zeros(size)
        arr[index] = 1.0
        return arr

    def create_gnn_graph_data(
        self,
        num_nodes: int = 10,
        num_edges: int = None,
        node_feature_dim: int = 32,
        edge_feature_dim: int = 16,
        directed: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create graph data for GNN testing.

        Args:
            num_nodes: Number of nodes
            num_edges: Number of edges (default: 1.5 * num_nodes)
            node_feature_dim: Node feature dimension
            edge_feature_dim: Edge feature dimension
            directed: Whether graph is directed
            **kwargs: Additional attributes

        Returns:
            Graph data dictionary
        """
        if num_edges is None:
            num_edges = int(1.5 * num_nodes)

        # Generate connected graph
        edge_index = self._generate_connected_edges(num_nodes, num_edges, directed)

        graph_data = {
            "node_features": np.random.randn(num_nodes, node_feature_dim).astype(np.float32),
            "edge_features": np.random.randn(num_edges, edge_feature_dim).astype(np.float32),
            "edge_index": edge_index,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "node_types": np.random.randint(0, 4, size=num_nodes),
            "edge_types": np.random.randint(0, 3, size=num_edges),
            "global_features": np.random.randn(64).astype(np.float32),
            "node_masks": np.ones(num_nodes, dtype=bool),
            "edge_masks": np.ones(num_edges, dtype=bool),
        }

        graph_data.update(kwargs)
        return graph_data

    def _generate_connected_edges(
        self, num_nodes: int, num_edges: int, directed: bool
    ) -> np.ndarray:
        """Generate edges ensuring graph connectivity."""
        edges = []

        # Ensure connectivity with a spanning tree
        for i in range(1, num_nodes):
            parent = random.randint(0, i - 1)
            edges.append([parent, i])
            if not directed:
                edges.append([i, parent])

        # Add remaining random edges
        while len(edges) < num_edges:
            src = random.randint(0, num_nodes - 1)
            dst = random.randint(0, num_nodes - 1)
            if src != dst and [src, dst] not in edges:
                edges.append([src, dst])
                if not directed and [dst, src] not in edges:
                    edges.append([dst, src])

        # Trim to exact number of edges
        edges = edges[:num_edges]

        return np.array(edges).T

    def create_api_request(
        self,
        method: str = "GET",
        endpoint: str = "/api/test",
        data: Dict[str, Any] = None,
        headers: Dict[str, str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Create API request data."""
        if headers is None:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer test_token_{self._next_id('token')}",
            }

        request_data = {
            "method": method,
            "endpoint": endpoint,
            "data": data or {},
            "headers": headers,
            "params": kwargs.get("params", {}),
            "timestamp": datetime.now().isoformat(),
        }

        request_data.update(kwargs)
        return request_data

    def create_websocket_message(
        self, message_type: str = None, data: Dict[str, Any] = None, **kwargs
    ) -> Dict[str, Any]:
        """Create WebSocket message."""
        if message_type is None:
            message_type = random.choice(["subscribe", "unsubscribe", "update", "command", "query"])

        if data is None:
            data = {"value": random.random()}

        message = {
            "id": self._next_id("msg"),
            "type": message_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
            "sequence": self._id_counter,
        }

        message.update(kwargs)
        return message

    def create_test_scenario(
        self,
        scenario_type: str = "exploration",
        num_agents: int = 3,
        world_size: int = 10,
        duration: int = 100,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a complete test scenario with agents, world, and objectives.

        Args:
            scenario_type: Type of scenario (exploration, combat, trade, research)
            num_agents: Number of agents
            world_size: Size of world grid
            duration: Scenario duration in timesteps
            **kwargs: Additional scenario parameters

        Returns:
            Complete scenario configuration
        """
        # Create agents appropriate for scenario
        agent_types = self._get_agent_types_for_scenario(scenario_type)
        agents = self.create_agent_batch(num_agents, agent_types)

        # Create world
        world_cells = []
        for i in range(world_size):
            for j in range(world_size):
                cell = self.create_world_cell(position={"x": i * 10, "y": j * 10, "z": 0})
                world_cells.append(cell)

        # Create objectives
        objectives = self._create_scenario_objectives(scenario_type)

        # Create initial coalitions if needed
        coalitions = []
        if scenario_type in ["trade", "research"]:
            coalition = self.create_coalition(
                members=agents[: num_agents // 2],
                business_type=self._get_business_type_for_scenario(scenario_type),
            )
            coalitions.append(coalition)

        scenario = {
            "id": self._next_id("scenario"),
            "type": scenario_type,
            "name": f"{scenario_type.capitalize()} Scenario {self._id_counter}",
            "agents": agents,
            "world": {
                "cells": world_cells,
                "size": world_size,
                "boundaries": {
                    "min_x": 0,
                    "max_x": world_size * 10,
                    "min_y": 0,
                    "max_y": world_size * 10,
                },
            },
            "coalitions": coalitions,
            "objectives": objectives,
            "duration": duration,
            "current_timestep": 0,
            "success_criteria": self._get_success_criteria_for_scenario(scenario_type),
            "config": {
                "allow_coalitions": True,
                "enable_combat": scenario_type == "combat",
                "resource_regeneration": True,
                "communication_enabled": True,
            },
        }

        scenario.update(kwargs)
        return scenario

    def _get_agent_types_for_scenario(self, scenario_type: str) -> List[str]:
        """Get appropriate agent types for scenario."""
        type_map = {
            "exploration": ["explorer", "explorer", "scout"],
            "combat": ["guardian", "warrior", "defender"],
            "trade": ["merchant", "trader", "broker"],
            "research": ["scholar", "scientist", "analyst"],
        }
        return type_map.get(scenario_type, ["explorer", "guardian", "merchant", "scholar"])

    def _get_business_type_for_scenario(self, scenario_type: str) -> str:
        """Get appropriate business type for scenario."""
        type_map = {
            "exploration": "ResourceOptimization",
            "combat": "DefenseAlliance",
            "trade": "TradePartnership",
            "research": "KnowledgeSharing",
        }
        return type_map.get(scenario_type, "ResourceOptimization")

    def _create_scenario_objectives(self, scenario_type: str) -> List[Dict[str, Any]]:
        """Create objectives based on scenario type."""
        objectives = []

        if scenario_type == "exploration":
            objectives.extend(
                [
                    {"type": "explore_cells", "target": 50, "reward": 100},
                    {"type": "find_resources", "target": 200, "reward": 150},
                    {"type": "map_boundaries", "target": 4, "reward": 200},
                ]
            )
        elif scenario_type == "combat":
            objectives.extend(
                [
                    {"type": "defend_position", "target": 10, "reward": 200},
                    {"type": "eliminate_threats", "target": 5, "reward": 150},
                    {"type": "secure_resources", "target": 100, "reward": 100},
                ]
            )
        elif scenario_type == "trade":
            objectives.extend(
                [
                    {"type": "establish_routes", "target": 3, "reward": 150},
                    {"type": "profit_target", "target": 1000, "reward": 200},
                    {"type": "partner_count", "target": 5, "reward": 100},
                ]
            )
        elif scenario_type == "research":
            objectives.extend(
                [
                    {"type": "discover_patterns", "target": 10, "reward": 200},
                    {"type": "collect_data", "target": 500, "reward": 100},
                    {"type": "publish_findings", "target": 3, "reward": 150},
                ]
            )

        return objectives

    def _get_success_criteria_for_scenario(self, scenario_type: str) -> Dict[str, Any]:
        """Get success criteria for scenario."""
        return {
            "min_objectives_completed": 2,
            "min_survival_rate": 0.8,
            "max_timesteps": 1000,
            "min_resource_efficiency": 0.6,
        }


# Global factory instance for convenience
factory = DataFactory()

# Convenience functions
create_agent = factory.create_agent
create_agent_batch = factory.create_agent_batch
create_coalition = factory.create_coalition
create_world_cell = factory.create_world_cell
create_active_inference_state = factory.create_active_inference_state
create_gnn_graph_data = factory.create_gnn_graph_data
create_test_scenario = factory.create_test_scenario
