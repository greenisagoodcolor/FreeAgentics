"""Message generators for creating various types of WebSocket messages."""

import random
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from faker import Faker

fake = Faker()


class MessageGenerator(ABC):
    """Abstract base class for message generators."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize message generator."""
        if seed is not None:
            random.seed(seed)
            Faker.seed(seed)
        self.message_count = 0

    @abstractmethod
    def generate(self) -> Dict[str, Any]:
        """Generate a single message."""
        pass

    def generate_batch(self, count: int) -> List[Dict[str, Any]]:
        """Generate multiple messages."""
        return [self.generate() for _ in range(count)]

    def _add_metadata(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Add standard metadata to a message."""
        self.message_count += 1
        message.update(
            {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "sequence": self.message_count,
            }
        )
        return message


class EventMessageGenerator(MessageGenerator):
    """Generates event subscription and notification messages."""

    def __init__(
        self,
        event_types: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize event message generator."""
        super().__init__(seed)

        self.event_types = event_types or [
            "agent:created",
            "agent:started",
            "agent:stopped",
            "agent:action",
            "agent:error",
            "world:updated",
            "world:agent_moved",
            "coalition:formed",
            "coalition:dissolved",
            "inference:started",
            "inference:completed",
        ]

    def generate(self) -> Dict[str, Any]:
        """Generate a random event message."""
        message_type = random.choice(
            [
                "subscribe",
                "unsubscribe",
                "ping",
            ]
        )

        if message_type == "subscribe":
            return self._generate_subscribe()
        elif message_type == "unsubscribe":
            return self._generate_unsubscribe()
        else:  # ping
            return self._generate_ping()

    def _generate_subscribe(self) -> Dict[str, Any]:
        """Generate a subscription message."""
        # Subscribe to 1-5 random event types
        num_events = random.randint(1, min(5, len(self.event_types)))
        selected_events = random.sample(self.event_types, num_events)

        message = {
            "type": "subscribe",
            "event_types": selected_events,
        }

        return self._add_metadata(message)

    def _generate_unsubscribe(self) -> Dict[str, Any]:
        """Generate an unsubscription message."""
        # Unsubscribe from 1-3 random event types
        num_events = random.randint(1, min(3, len(self.event_types)))
        selected_events = random.sample(self.event_types, num_events)

        message = {
            "type": "unsubscribe",
            "event_types": selected_events,
        }

        return self._add_metadata(message)

    def _generate_ping(self) -> Dict[str, Any]:
        """Generate a ping message."""
        message = {
            "type": "ping",
        }

        return self._add_metadata(message)


class CommandMessageGenerator(MessageGenerator):
    """Generates agent command messages."""

    def __init__(
        self,
        agent_ids: Optional[List[str]] = None,
        commands: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize command message generator."""
        super().__init__(seed)

        self.agent_ids = agent_ids or [f"agent_{i}" for i in range(10)]
        self.commands = commands or [
            "start",
            "stop",
            "pause",
            "resume",
            "move",
            "observe",
            "act",
            "plan",
            "join_coalition",
            "leave_coalition",
        ]

    def generate(self) -> Dict[str, Any]:
        """Generate a random agent command."""
        command = random.choice(self.commands)
        agent_id = random.choice(self.agent_ids)

        message = {
            "type": "agent_command",
            "data": {
                "agent_id": agent_id,
                "command": command,
                "params": self._generate_params(command),
            },
        }

        return self._add_metadata(message)

    def _generate_params(self, command: str) -> Dict[str, Any]:
        """Generate parameters for a specific command."""
        if command == "move":
            return {
                "direction": random.choice(["up", "down", "left", "right"]),
                "distance": random.randint(1, 5),
            }
        elif command == "observe":
            return {
                "radius": random.randint(1, 10),
                "filters": random.sample(
                    ["agents", "objects", "terrain"], random.randint(1, 3)
                ),
            }
        elif command == "act":
            return {
                "action_type": random.choice(
                    ["explore", "gather", "interact", "communicate"]
                ),
                "target": random.choice(
                    [
                        None,
                        f"object_{random.randint(1, 50)}",
                        f"agent_{random.randint(1, 10)}",
                    ]
                ),
            }
        elif command == "plan":
            return {
                "goal": random.choice(
                    [
                        "explore_area",
                        "find_resources",
                        "form_coalition",
                        "complete_task",
                    ]
                ),
                "horizon": random.randint(5, 20),
            }
        elif command in ["join_coalition", "leave_coalition"]:
            return {
                "coalition_id": f"coalition_{random.randint(1, 5)}",
            }
        else:
            return {}


class QueryMessageGenerator(MessageGenerator):
    """Generates query messages for retrieving system state."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize query message generator."""
        super().__init__(seed)

        self.query_types = [
            "agent_status",
            "world_state",
            "coalition_status",
            "system_metrics",
            "agent_history",
        ]

    def generate(self) -> Dict[str, Any]:
        """Generate a random query message."""
        query_type = random.choice(self.query_types)

        message = {
            "type": "query",
            "data": {
                "query_type": query_type,
                **self._generate_query_params(query_type),
            },
        }

        return self._add_metadata(message)

    def _generate_query_params(self, query_type: str) -> Dict[str, Any]:
        """Generate parameters for a specific query type."""
        if query_type == "agent_status":
            return {
                "agent_ids": random.sample(
                    [f"agent_{i}" for i in range(10)], random.randint(1, 5)
                ),
                "include_history": random.choice([True, False]),
            }
        elif query_type == "world_state":
            return {
                "region": {
                    "x": random.randint(0, 100),
                    "y": random.randint(0, 100),
                    "width": random.randint(10, 50),
                    "height": random.randint(10, 50),
                },
                "include_agents": True,
                "include_objects": True,
            }
        elif query_type == "coalition_status":
            return {
                "coalition_ids": [
                    f"coalition_{random.randint(1, 5)}"
                    for _ in range(random.randint(1, 3))
                ],
            }
        elif query_type == "system_metrics":
            return {
                "metrics": random.sample(
                    [
                        "cpu_usage",
                        "memory_usage",
                        "agent_count",
                        "inference_rate",
                        "message_throughput",
                    ],
                    random.randint(2, 5),
                ),
                "duration": random.choice([60, 300, 600, 3600]),  # seconds
            }
        elif query_type == "agent_history":
            return {
                "agent_id": f"agent_{random.randint(1, 10)}",
                "limit": random.randint(10, 100),
                "offset": 0,
            }
        else:
            return {}


class MonitoringMessageGenerator(MessageGenerator):
    """Generates monitoring configuration messages."""

    def __init__(self, seed: Optional[int] = None):
        """Initialize monitoring message generator."""
        super().__init__(seed)

        self.available_metrics = [
            "cpu_usage",
            "memory_usage",
            "gpu_usage",
            "inference_rate",
            "agent_count",
            "active_coalitions",
            "message_throughput",
            "error_rate",
            "world_update_rate",
            "knowledge_graph_size",
        ]

    def generate(self) -> Dict[str, Any]:
        """Generate a monitoring configuration message."""
        action = random.choice(
            ["start_monitoring", "stop_monitoring", "update_monitoring"]
        )

        if action == "start_monitoring":
            return self._generate_start_monitoring()
        elif action == "stop_monitoring":
            return self._generate_stop_monitoring()
        else:  # update_monitoring
            return self._generate_update_monitoring()

    def _generate_start_monitoring(self) -> Dict[str, Any]:
        """Generate a start monitoring message."""
        message = {
            "type": "start_monitoring",
            "config": {
                "metrics": random.sample(self.available_metrics, random.randint(3, 7)),
                "agents": [f"agent_{i}" for i in range(random.randint(0, 5))],
                "sample_rate": random.choice([0.5, 1.0, 2.0, 5.0, 10.0]),
                "buffer_size": random.choice([100, 500, 1000, 5000]),
            },
        }

        return self._add_metadata(message)

    def _generate_stop_monitoring(self) -> Dict[str, Any]:
        """Generate a stop monitoring message."""
        message = {
            "type": "stop_monitoring",
            "session_id": str(uuid.uuid4()),
        }

        return self._add_metadata(message)

    def _generate_update_monitoring(self) -> Dict[str, Any]:
        """Generate an update monitoring message."""
        message = {
            "type": "update_monitoring",
            "session_id": str(uuid.uuid4()),
            "updates": {
                "sample_rate": random.choice([0.5, 1.0, 2.0, 5.0, 10.0]),
                "add_metrics": random.sample(
                    self.available_metrics, random.randint(1, 3)
                ),
                "remove_metrics": random.sample(
                    self.available_metrics, random.randint(0, 2)
                ),
            },
        }

        return self._add_metadata(message)


class MixedMessageGenerator(MessageGenerator):
    """Generates a mix of different message types."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ):
        """Initialize mixed message generator."""
        super().__init__(seed)

        # Default weights for different message types
        self.weights = weights or {
            "event": 0.3,
            "command": 0.3,
            "query": 0.2,
            "monitoring": 0.1,
            "ping": 0.1,
        }

        # Initialize sub-generators
        self.generators = {
            "event": EventMessageGenerator(seed=seed),
            "command": CommandMessageGenerator(seed=seed),
            "query": QueryMessageGenerator(seed=seed),
            "monitoring": MonitoringMessageGenerator(seed=seed),
        }

    def generate(self) -> Dict[str, Any]:
        """Generate a random message based on weights."""
        # Choose message type based on weights
        message_type = random.choices(
            list(self.weights.keys()),
            weights=list(self.weights.values()),
            k=1,
        )[0]

        if message_type == "ping":
            return self._add_metadata({"type": "ping"})
        else:
            generator = self.generators.get(message_type)
            if generator:
                return generator.generate()
            else:
                raise ValueError(f"Unknown message type: {message_type}")


class RealisticScenarioGenerator(MessageGenerator):
    """Generates messages following realistic usage patterns."""

    def __init__(self, scenario: str = "default", seed: Optional[int] = None):
        """Initialize realistic scenario generator."""
        super().__init__(seed)

        self.scenario = scenario
        self.state = {
            "subscribed_events": set(),
            "active_agents": set(),
            "monitoring_sessions": set(),
        }

        # Initialize sub-generators
        self.event_gen = EventMessageGenerator(seed=seed)
        self.command_gen = CommandMessageGenerator(seed=seed)
        self.query_gen = QueryMessageGenerator(seed=seed)
        self.monitoring_gen = MonitoringMessageGenerator(seed=seed)

    def generate(self) -> Dict[str, Any]:
        """Generate a message based on realistic patterns."""
        if self.scenario == "startup":
            return self._generate_startup_sequence()
        elif self.scenario == "steady_state":
            return self._generate_steady_state()
        elif self.scenario == "shutdown":
            return self._generate_shutdown_sequence()
        else:  # default
            return self._generate_default()

    def _generate_startup_sequence(self) -> Dict[str, Any]:
        """Generate messages for system startup."""
        if self.message_count == 0:
            # First, subscribe to important events
            return self.event_gen._generate_subscribe()
        elif self.message_count < 5:
            # Start monitoring
            return self.monitoring_gen._generate_start_monitoring()
        elif self.message_count < 10:
            # Query initial state
            return self.query_gen.generate()
        else:
            # Switch to steady state
            self.scenario = "steady_state"
            return self._generate_steady_state()

    def _generate_steady_state(self) -> Dict[str, Any]:
        """Generate messages for normal operation."""
        # Weighted distribution for steady state
        rand = random.random()

        if rand < 0.4:
            # Agent commands
            return self.command_gen.generate()
        elif rand < 0.7:
            # Queries
            return self.query_gen.generate()
        elif rand < 0.9:
            # Occasional ping
            return self._add_metadata({"type": "ping"})
        else:
            # Rare monitoring updates
            return self.monitoring_gen._generate_update_monitoring()

    def _generate_shutdown_sequence(self) -> Dict[str, Any]:
        """Generate messages for system shutdown."""
        if len(self.state["monitoring_sessions"]) > 0:
            # Stop monitoring sessions
            return self.monitoring_gen._generate_stop_monitoring()
        elif len(self.state["subscribed_events"]) > 0:
            # Unsubscribe from events
            return self.event_gen._generate_unsubscribe()
        else:
            # Final ping
            return self._add_metadata({"type": "ping"})

    def _generate_default(self) -> Dict[str, Any]:
        """Generate messages with default patterns."""
        # Similar to steady state but with more variety
        generators = [
            (self.event_gen, 0.2),
            (self.command_gen, 0.3),
            (self.query_gen, 0.3),
            (self.monitoring_gen, 0.1),
        ]

        rand = random.random()
        cumulative = 0.0

        for generator, weight in generators:
            cumulative += weight
            if rand < cumulative:
                return generator.generate()

        # Default to ping
        return self._add_metadata({"type": "ping"})
