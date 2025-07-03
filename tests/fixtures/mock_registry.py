"""
Mock registry for centralized mock management.

This module provides a registry pattern for managing and reusing mocks across tests,
ensuring consistency and reducing duplication.
"""

import logging
from typing import Any, Callable, Dict, Optional
from unittest.mock import Mock

from .mock_factory import MockFactory

logger = logging.getLogger(__name__)


class MockRegistry:
    """Registry for managing mock objects and factories."""

    _factories: Dict[str, Callable] = {}
    _instances: Dict[str, Any] = {}
    _default_factories = {
        "agent": MockFactory.create_agent,
        "coalition": MockFactory.create_coalition,
        "gnn_data": MockFactory.create_gnn_data,
        "websocket": MockFactory.create_websocket_connection,
        "database": MockFactory.create_database_session,
        "active_inference": MockFactory.create_active_inference_data,
        "api_response": MockFactory.create_api_response,
    }

    @classmethod
    def initialize(cls) -> None:
        """Initialize the registry with default factories."""
        cls._factories = cls._default_factories.copy()
        cls._instances.clear()
        logger.debug("Mock registry initialized with default factories")

    @classmethod
    def register_factory(cls, name: str, factory: Callable) -> None:
        """Register a mock factory.

        Args:
            name: Name to register the factory under
            factory: Callable that creates mock objects
        """
        cls._factories[name] = factory
        logger.debug(f"Registered mock factory: {name}")

    @classmethod
    def create(cls, name: str, **kwargs) -> Any:
        """Create a mock using a registered factory.

        Args:
            name: Name of the factory to use
            **kwargs: Arguments to pass to the factory

        Returns:
            Mock object created by the factory

        Raises:
            KeyError: If factory name is not registered
        """
        if name not in cls._factories:
            raise KeyError(f"No factory registered for '{name}'")

        return cls._factories[name](**kwargs)

    @classmethod
    def get_or_create(cls, key: str, factory_name: str, **kwargs) -> Any:
        """Get an existing mock instance or create a new one.

        This is useful for tests that need to share mock instances.

        Args:
            key: Unique key for the instance
            factory_name: Name of the factory to use if creating
            **kwargs: Arguments for the factory if creating

        Returns:
            Mock instance (existing or newly created)
        """
        if key not in cls._instances:
            cls._instances[key] = cls.create(factory_name, **kwargs)
            logger.debug(f"Created new instance with key: {key}")
        else:
            logger.debug(f"Retrieved existing instance with key: {key}")

        return cls._instances[key]

    @classmethod
    def clear_instances(cls) -> None:
        """Clear all stored instances."""
        cls._instances.clear()
        logger.debug("Cleared all mock instances")

    @classmethod
    def reset(cls) -> None:
        """Reset the registry to initial state."""
        cls.initialize()

    @classmethod
    def create_agent_with_coalition(
        cls, agent_id: Optional[str] = None, coalition_id: Optional[str] = None, **kwargs
    ) -> tuple[Mock, Mock]:
        """Create an agent and add it to a coalition.

        Args:
            agent_id: Agent ID (auto-generated if not provided)
            coalition_id: Coalition ID (auto-generated if not provided)
            **kwargs: Additional agent attributes

        Returns:
            Tuple of (agent_mock, coalition_mock)
        """
        agent = cls.create("agent", agent_id=agent_id, **kwargs)

        coalition = cls.create("coalition", coalition_id=coalition_id, members=[agent.agent_id])

        # Set up bidirectional relationship
        agent.coalition_id = coalition.coalition_id
        agent.coalition = coalition

        return agent, coalition

    @classmethod
    def create_agent_network(
        cls, num_agents: int = 5, num_coalitions: int = 2, agent_type: str = "explorer"
    ) -> Dict[str, Any]:
        """Create a network of agents and coalitions.

        Args:
            num_agents: Number of agents to create
            num_coalitions: Number of coalitions
            agent_type: Type of agents to create

        Returns:
            Dictionary with agents and coalitions
        """
        agents = []
        coalitions = []

        # Create agents
        for i in range(num_agents):
            agent = cls.create(
                "agent",
                agent_id=f"agent_{
                    i:03d}",
                agent_type=agent_type,
            )
            agents.append(agent)

        # Create coalitions and distribute agents
        agents_per_coalition = num_agents // num_coalitions
        agent_idx = 0

        for i in range(num_coalitions):
            coalition_members = []
            for j in range(agents_per_coalition):
                if agent_idx < num_agents:
                    agent = agents[agent_idx]
                    coalition_members.append(agent.agent_id)
                    agent_idx += 1

            coalition = cls.create(
                "coalition",
                coalition_id=f"coalition_{
                    i:03d}",
                members=coalition_members,
            )
            coalitions.append(coalition)

            # Update agent references
            for member_id in coalition_members:
                agent = next(a for a in agents if a.agent_id == member_id)
                agent.coalition_id = coalition.coalition_id
                agent.coalition = coalition

        return {
            "agents": agents,
            "coalitions": coalitions,
            "agent_map": {a.agent_id: a for a in agents},
            "coalition_map": {c.coalition_id: c for c in coalitions},
        }

    @classmethod
    def create_test_scenario(cls, scenario: str = "basic") -> Dict[str, Any]:
        """Create a complete test scenario with pre-configured mocks.

        Args:
            scenario: Type of scenario to create

        Returns:
            Dictionary with all necessary mocks for the scenario
        """
        scenarios = {
            "basic": cls._create_basic_scenario,
            "exploration": cls._create_exploration_scenario,
            "trading": cls._create_trading_scenario,
            "formation": cls._create_formation_scenario,
        }

        if scenario not in scenarios:
            raise ValueError(f"Unknown scenario: {scenario}")

        return scenarios[scenario]()

    @classmethod
    def _create_basic_scenario(cls) -> Dict[str, Any]:
        """Create a basic test scenario."""
        agent = cls.create("agent")
        database = cls.create("database")
        api_response = cls.create("api_response")

        return {"agent": agent, "database": database, "api_response": api_response}

    @classmethod
    def _create_exploration_scenario(cls) -> Dict[str, Any]:
        """Create an exploration scenario."""
        network = cls.create_agent_network(num_agents=5, num_coalitions=1, agent_type="explorer")

        # Add exploration-specific data
        for agent in network["agents"]:
            agent.exploration_target = {"x": 10, "y": 10}
            agent.path_history = []

        return network

    @classmethod
    def _create_trading_scenario(cls) -> Dict[str, Any]:
        """Create a trading scenario."""
        # Create merchants
        merchants = [
            cls.create("agent", agent_type="merchant", agent_id=f"merchant_{i}") for i in range(3)
        ]

        # Set up different resource profiles
        merchants[0].resources = {"energy": 100, "materials": 20, "information": 50}
        merchants[1].resources = {"energy": 20, "materials": 100, "information": 30}
        merchants[2].resources = {"energy": 50, "materials": 50, "information": 100}

        return {
            "merchants": merchants,
            "trade_history": [],
            "market_prices": {"energy": 1.0, "materials": 2.0, "information": 1.5},
        }

    @classmethod
    def _create_formation_scenario(cls) -> Dict[str, Any]:
        """Create a coalition formation scenario."""
        # Create diverse agents
        agents = []
        for agent_type in ["explorer", "guardian", "merchant", "scholar"]:
            for i in range(2):
                agent = cls.create("agent", agent_type=agent_type, agent_id=f"{agent_type}_{i}")
                agents.append(agent)

        # Create formation proposals
        proposals = [
            {
                "id": "proposal_1",
                "initiator": agents[0].agent_id,
                "members": [agents[0].agent_id, agents[2].agent_id],
                "purpose": "resource_gathering",
            },
            {
                "id": "proposal_2",
                "initiator": agents[4].agent_id,
                "members": [agents[4].agent_id, agents[5].agent_id, agents[6].agent_id],
                "purpose": "knowledge_sharing",
            },
        ]

        return {"agents": agents, "proposals": proposals, "coalitions": []}


# Initialize the registry on import
MockRegistry.initialize()
