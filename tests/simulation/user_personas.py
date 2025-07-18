"""User personas for realistic simulation scenarios.

This module defines different user personas with distinct behavior patterns,
preferences, and interaction styles for comprehensive system testing.
"""

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from faker import Faker

fake = Faker()


class PersonaType(Enum):
    """Types of user personas."""

    RESEARCHER = "researcher"
    COORDINATOR = "coordinator"
    OBSERVER = "observer"
    ADMIN = "admin"
    DEVELOPER = "developer"
    ANALYST = "analyst"


class ActivityLevel(Enum):
    """User activity levels."""

    HYPERACTIVE = "hyperactive"
    ACTIVE = "active"
    MODERATE = "moderate"
    PASSIVE = "passive"
    SPORADIC = "sporadic"


class InteractionPattern(Enum):
    """User interaction patterns."""

    CONTINUOUS = "continuous"  # Always connected, steady activity
    BURSTY = "bursty"  # Periods of high activity
    SCHEDULED = "scheduled"  # Activity at specific times
    RANDOM = "random"  # Unpredictable activity
    REACTIVE = "reactive"  # Responds to events


@dataclass
class PersonaProfile:
    """Profile defining a user persona's characteristics."""

    persona_type: PersonaType
    activity_level: ActivityLevel
    interaction_pattern: InteractionPattern

    # Behavioral characteristics
    attention_span: Tuple[float, float] = (
        30.0,
        300.0,
    )  # Min/max seconds on task
    response_time: Tuple[float, float] = (0.1, 2.0)  # Reaction time range
    error_rate: float = 0.02  # Probability of mistakes
    multitasking_level: int = 1  # Number of concurrent tasks

    # Preferences
    preferred_agents: List[str] = field(default_factory=list)
    preferred_coalitions: List[str] = field(default_factory=list)
    favorite_queries: List[str] = field(default_factory=list)
    monitoring_interests: List[str] = field(default_factory=list)

    # Schedule (for scheduled pattern)
    active_hours: List[int] = field(default_factory=lambda: list(range(9, 18)))
    peak_hours: List[int] = field(default_factory=lambda: [10, 11, 14, 15])

    # Connection behavior
    connection_stability: float = 0.95  # Probability of maintaining connection
    reconnect_probability: float = (
        0.8  # Probability of reconnecting after disconnect
    )
    max_reconnect_attempts: int = 3

    # Message patterns
    message_weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize message weights based on persona type."""
        if not self.message_weights:
            self.message_weights = self._get_default_message_weights()

    def _get_default_message_weights(self) -> Dict[str, float]:
        """Get default message weights for persona type."""
        weights = {
            PersonaType.RESEARCHER: {
                "query": 0.4,
                "command": 0.3,
                "event": 0.2,
                "monitoring": 0.08,
                "ping": 0.02,
            },
            PersonaType.COORDINATOR: {
                "command": 0.5,
                "event": 0.25,
                "query": 0.15,
                "monitoring": 0.08,
                "ping": 0.02,
            },
            PersonaType.OBSERVER: {
                "event": 0.6,
                "query": 0.25,
                "monitoring": 0.1,
                "command": 0.03,
                "ping": 0.02,
            },
            PersonaType.ADMIN: {
                "monitoring": 0.4,
                "query": 0.3,
                "command": 0.2,
                "event": 0.08,
                "ping": 0.02,
            },
            PersonaType.DEVELOPER: {
                "command": 0.35,
                "query": 0.35,
                "event": 0.15,
                "monitoring": 0.13,
                "ping": 0.02,
            },
            PersonaType.ANALYST: {
                "query": 0.5,
                "monitoring": 0.3,
                "event": 0.15,
                "command": 0.03,
                "ping": 0.02,
            },
        }
        return weights.get(
            self.persona_type,
            {
                "query": 0.25,
                "command": 0.25,
                "event": 0.25,
                "monitoring": 0.2,
                "ping": 0.05,
            },
        )


class UserBehavior(ABC):
    """Abstract base class for user behavior simulation."""

    def __init__(self, user_id: str, profile: PersonaProfile):
        """Initialize user behavior."""
        self.user_id = user_id
        self.profile = profile
        self.state = {
            "connected": False,
            "busy": False,
            "current_task": None,
            "task_start_time": None,
            "subscribed_events": set(),
            "active_agents": set(),
            "monitoring_sessions": set(),
            "error_count": 0,
            "message_count": 0,
            "last_activity": None,
        }
        self.history = []
        self.rng = np.random.default_rng()

    @abstractmethod
    async def decide_next_action(self) -> Optional[Dict[str, Any]]:
        """Decide the next action based on persona behavior."""
        pass

    def should_act(self) -> bool:
        """Determine if the user should take an action now."""
        if not self.state["connected"]:
            return False

        # Check if currently busy with a task
        if self.state["busy"] and self.state["task_start_time"]:
            elapsed = (
                datetime.now() - self.state["task_start_time"]
            ).total_seconds()
            min_span, max_span = self.profile.attention_span
            task_duration = self.rng.uniform(min_span, max_span)

            if elapsed < task_duration:
                return False
            else:
                self.state["busy"] = False
                self.state["current_task"] = None

        # Check activity pattern
        if self.profile.interaction_pattern == InteractionPattern.SCHEDULED:
            current_hour = datetime.now().hour
            if current_hour not in self.profile.active_hours:
                return (
                    self.rng.random() < 0.1
                )  # 10% chance of off-hours activity

        # Activity level determines action frequency
        activity_thresholds = {
            ActivityLevel.HYPERACTIVE: 0.9,
            ActivityLevel.ACTIVE: 0.7,
            ActivityLevel.MODERATE: 0.5,
            ActivityLevel.PASSIVE: 0.3,
            ActivityLevel.SPORADIC: 0.1,
        }

        threshold = activity_thresholds.get(self.profile.activity_level, 0.5)
        return self.rng.random() < threshold

    def add_error(self) -> bool:
        """Determine if an error should be introduced."""
        if self.rng.random() < self.profile.error_rate:
            self.state["error_count"] += 1
            return True
        return False

    def get_response_delay(self) -> float:
        """Get response delay based on profile."""
        min_delay, max_delay = self.profile.response_time
        return self.rng.uniform(min_delay, max_delay)

    def record_action(self, action: Dict[str, Any]):
        """Record an action in history."""
        action["timestamp"] = datetime.now()
        action["user_id"] = self.user_id
        action["persona_type"] = self.profile.persona_type.value
        self.history.append(action)
        self.state["message_count"] += 1
        self.state["last_activity"] = datetime.now()


class ResearcherBehavior(UserBehavior):
    """Behavior simulation for researcher persona."""

    async def decide_next_action(self) -> Optional[Dict[str, Any]]:
        """Researcher focuses on querying and analyzing agent behavior."""
        if not self.should_act():
            return None

        # Choose action based on weights
        action_type = self.rng.choice(
            list(self.profile.message_weights.keys()),
            p=list(self.profile.message_weights.values()),
        )

        if action_type == "query":
            return self._generate_research_query()
        elif action_type == "command":
            return self._generate_experiment_command()
        elif action_type == "event":
            return self._generate_event_subscription()
        elif action_type == "monitoring":
            return self._generate_monitoring_config()
        else:
            return {"type": "ping"}

    def _generate_research_query(self) -> Dict[str, Any]:
        """Generate a research-oriented query."""
        query_types = [
            "agent_history",
            "coalition_performance",
            "belief_evolution",
            "action_statistics",
            "world_state_analysis",
        ]

        query_type = self.rng.choice(query_types)

        if query_type == "agent_history":
            agent_id = (
                self.rng.choice(list(self.state["active_agents"]))
                if self.state["active_agents"]
                else f"agent_{self.rng.integers(1, 20)}"
            )
            return {
                "type": "query",
                "query_type": "agent_history",
                "agent_id": agent_id,
                "time_range": self.rng.choice([3600, 7200, 14400, 86400]),
                "metrics": ["beliefs", "actions", "rewards", "performance"],
            }

        elif query_type == "coalition_performance":
            return {
                "type": "query",
                "query_type": "coalition_analysis",
                "metrics": [
                    "formation_time",
                    "goal_achievement",
                    "member_contribution",
                    "stability",
                ],
                "compare_coalitions": self.rng.choice([True, False]),
            }

        elif query_type == "belief_evolution":
            return {
                "type": "query",
                "query_type": "belief_dynamics",
                "agent_ids": (
                    list(
                        self.rng.choice(
                            list(self.state["active_agents"]),
                            size=min(3, len(self.state["active_agents"])),
                            replace=False,
                        )
                    )
                    if self.state["active_agents"]
                    else []
                ),
                "belief_types": [
                    "state_beliefs",
                    "policy_beliefs",
                    "preference_beliefs",
                ],
                "time_window": self.rng.choice([300, 600, 1800, 3600]),
            }

        return {"type": "query", "query_type": query_type}

    def _generate_experiment_command(self) -> Dict[str, Any]:
        """Generate commands for running experiments."""
        commands = [
            "create_controlled_agent",
            "modify_agent_parameters",
            "trigger_scenario",
            "reset_agent_state",
            "pause_agent",
            "resume_agent",
        ]

        command = self.rng.choice(commands)

        if command == "create_controlled_agent":
            return {
                "type": "command",
                "command": "create_agent",
                "params": {
                    "name": f"Research_Agent_{self.rng.integers(1000, 9999)}",
                    "template": self.rng.choice(
                        [
                            "grid_navigator",
                            "coalition_former",
                            "resource_collector",
                        ]
                    ),
                    "controlled": True,
                    "parameters": {
                        "learning_rate": self.rng.uniform(0.01, 0.1),
                        "exploration_rate": self.rng.uniform(0.1, 0.3),
                        "planning_horizon": self.rng.integers(3, 10),
                    },
                },
            }

        elif command == "modify_agent_parameters":
            agent_id = (
                self.rng.choice(list(self.state["active_agents"]))
                if self.state["active_agents"]
                else None
            )
            if agent_id:
                return {
                    "type": "command",
                    "command": "update_agent",
                    "agent_id": agent_id,
                    "params": {
                        "parameters": {
                            self.rng.choice(
                                [
                                    "learning_rate",
                                    "exploration_rate",
                                    "precision",
                                ]
                            ): self.rng.uniform(0.01, 0.5)
                        }
                    },
                }

        return {"type": "command", "command": command}

    def _generate_event_subscription(self) -> Dict[str, Any]:
        """Subscribe to research-relevant events."""
        research_events = [
            "agent:belief_updated",
            "agent:learning_complete",
            "agent:goal_achieved",
            "coalition:performance_measured",
            "experiment:checkpoint",
            "anomaly:detected",
        ]

        # Subscribe to multiple related events
        num_events = self.rng.integers(2, 5)
        selected_events = self.rng.choice(
            research_events, size=num_events, replace=False
        )

        return {
            "type": "subscribe",
            "event_types": selected_events.tolist(),
        }

    def _generate_monitoring_config(self) -> Dict[str, Any]:
        """Configure monitoring for research metrics."""
        return {
            "type": "start_monitoring",
            "config": {
                "metrics": [
                    "free_energy",
                    "expected_free_energy",
                    "epistemic_value",
                    "pragmatic_value",
                    "belief_entropy",
                    "policy_entropy",
                ],
                "agents": list(self.state["active_agents"])
                if self.state["active_agents"]
                else [],
                "sample_rate": self.rng.choice([0.5, 1.0, 2.0]),
                "aggregation": self.rng.choice(
                    ["mean", "std", "min_max", "percentiles"]
                ),
            },
        }


class CoordinatorBehavior(UserBehavior):
    """Behavior simulation for coordinator persona."""

    def __init__(self, user_id: str, profile: PersonaProfile):
        """Initialize coordinator with coalition management focus."""
        super().__init__(user_id, profile)
        self.state["managed_coalitions"] = set()
        self.state["coordination_tasks"] = []

    async def decide_next_action(self) -> Optional[Dict[str, Any]]:
        """Coordinator focuses on agent and coalition management."""
        if not self.should_act():
            return None

        # Check for pending coordination tasks
        if self.state["coordination_tasks"]:
            task = self.state["coordination_tasks"].pop(0)
            return self._handle_coordination_task(task)

        # Regular action selection
        action_type = self.rng.choice(
            list(self.profile.message_weights.keys()),
            p=list(self.profile.message_weights.values()),
        )

        if action_type == "command":
            return self._generate_coordination_command()
        elif action_type == "event":
            return self._generate_coordination_events()
        elif action_type == "query":
            return self._generate_status_query()
        elif action_type == "monitoring":
            return self._generate_team_monitoring()
        else:
            return {"type": "ping"}

    def _generate_coordination_command(self) -> Dict[str, Any]:
        """Generate coalition and agent coordination commands."""
        commands = [
            "form_coalition",
            "assign_agent_role",
            "delegate_task",
            "reorganize_coalition",
            "coordinate_action",
            "resolve_conflict",
        ]

        command = self.rng.choice(commands)

        if command == "form_coalition":
            num_agents = self.rng.integers(3, 8)
            return {
                "type": "command",
                "command": "form_coalition",
                "params": {
                    "name": f"Coalition_{fake.word()}_{self.rng.integers(100, 999)}",
                    "goal": self.rng.choice(
                        [
                            "explore_region",
                            "collect_resources",
                            "defend_position",
                            "research_task",
                        ]
                    ),
                    "required_agents": num_agents,
                    "formation_strategy": self.rng.choice(
                        ["capability_based", "proximity_based", "trust_based"]
                    ),
                    "coordinator_id": self.user_id,
                },
            }

        elif command == "assign_agent_role":
            if (
                self.state["managed_coalitions"]
                and self.state["active_agents"]
            ):
                return {
                    "type": "command",
                    "command": "assign_role",
                    "coalition_id": self.rng.choice(
                        list(self.state["managed_coalitions"])
                    ),
                    "agent_id": self.rng.choice(
                        list(self.state["active_agents"])
                    ),
                    "role": self.rng.choice(
                        [
                            "scout",
                            "collector",
                            "defender",
                            "coordinator",
                            "analyst",
                        ]
                    ),
                }

        elif command == "coordinate_action":
            if self.state["managed_coalitions"]:
                return {
                    "type": "command",
                    "command": "coordinate_action",
                    "coalition_id": self.rng.choice(
                        list(self.state["managed_coalitions"])
                    ),
                    "action_type": self.rng.choice(
                        [
                            "move_formation",
                            "execute_plan",
                            "synchronize_beliefs",
                            "share_observations",
                        ]
                    ),
                    "parameters": {
                        "synchronization_level": self.rng.uniform(0.5, 1.0),
                        "coordination_timeout": self.rng.integers(10, 60),
                    },
                }

        return {"type": "command", "command": command}

    def _generate_coordination_events(self) -> Dict[str, Any]:
        """Subscribe to coordination-relevant events."""
        coord_events = [
            "coalition:formation_request",
            "coalition:conflict_detected",
            "agent:role_change_request",
            "agent:assistance_needed",
            "coalition:goal_progress",
            "coalition:member_left",
        ]

        return {
            "type": "subscribe",
            "event_types": self.rng.choice(
                coord_events, size=self.rng.integers(3, 6), replace=False
            ).tolist(),
        }

    def _generate_status_query(self) -> Dict[str, Any]:
        """Query coalition and agent status."""
        query_types = [
            "coalition_status",
            "agent_availability",
            "task_progress",
            "resource_allocation",
        ]

        query_type = self.rng.choice(query_types)

        if (
            query_type == "coalition_status"
            and self.state["managed_coalitions"]
        ):
            return {
                "type": "query",
                "query_type": "coalition_status",
                "coalition_ids": list(self.state["managed_coalitions"]),
                "include_metrics": True,
                "include_member_status": True,
            }

        elif query_type == "agent_availability":
            return {
                "type": "query",
                "query_type": "available_agents",
                "capabilities": self.rng.choice(
                    [
                        ["navigation", "planning"],
                        ["combat", "defense"],
                        ["research", "analysis"],
                    ],
                    p=[0.5, 0.3, 0.2],
                ),
                "min_trust_score": self.rng.uniform(0.5, 0.8),
            }

        return {"type": "query", "query_type": query_type}

    def _generate_team_monitoring(self) -> Dict[str, Any]:
        """Monitor team performance metrics."""
        return {
            "type": "start_monitoring",
            "config": {
                "metrics": [
                    "coalition_cohesion",
                    "task_completion_rate",
                    "member_contribution",
                    "communication_frequency",
                    "conflict_rate",
                ],
                "coalitions": (
                    list(self.state["managed_coalitions"])
                    if self.state["managed_coalitions"]
                    else []
                ),
                "sample_rate": 1.0,
                "alert_thresholds": {
                    "coalition_cohesion": 0.5,
                    "conflict_rate": 0.2,
                },
            },
        }

    def _handle_coordination_task(
        self, task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle specific coordination tasks."""
        task_type = task.get("type")

        if task_type == "resolve_conflict":
            return {
                "type": "command",
                "command": "mediate_conflict",
                "params": {
                    "coalition_id": task["coalition_id"],
                    "conflicting_agents": task["agents"],
                    "resolution_strategy": self.rng.choice(
                        ["negotiate", "reassign_roles", "split_resources"]
                    ),
                },
            }

        return {"type": "ping"}


class ObserverBehavior(UserBehavior):
    """Behavior simulation for observer persona."""

    async def decide_next_action(self) -> Optional[Dict[str, Any]]:
        """Observer focuses on monitoring and event subscriptions."""
        if not self.should_act():
            return None

        action_type = self.rng.choice(
            list(self.profile.message_weights.keys()),
            p=list(self.profile.message_weights.values()),
        )

        if action_type == "event":
            return self._generate_observation_subscription()
        elif action_type == "query":
            return self._generate_observation_query()
        elif action_type == "monitoring":
            return self._generate_passive_monitoring()
        elif action_type == "command":
            return self._generate_minimal_command()
        else:
            return {"type": "ping"}

    def _generate_observation_subscription(self) -> Dict[str, Any]:
        """Subscribe to all interesting events for observation."""
        all_events = [
            "agent:created",
            "agent:started",
            "agent:stopped",
            "agent:action",
            "agent:belief_updated",
            "agent:error",
            "coalition:formed",
            "coalition:dissolved",
            "coalition:action",
            "world:updated",
            "world:event_triggered",
            "system:performance",
            "system:alert",
        ]

        # Observers tend to subscribe to many events
        num_events = self.rng.integers(5, 10)
        selected_events = self.rng.choice(
            all_events, size=num_events, replace=False
        )

        return {
            "type": "subscribe",
            "event_types": selected_events.tolist(),
            "passive": True,  # Indicates passive observation
        }

    def _generate_observation_query(self) -> Dict[str, Any]:
        """Query for observational data."""
        queries = [
            {
                "type": "query",
                "query_type": "system_overview",
                "include_stats": True,
                "time_window": self.rng.choice([300, 600, 1800, 3600]),
            },
            {
                "type": "query",
                "query_type": "event_stream",
                "limit": self.rng.integers(50, 200),
                "event_types": self.rng.choice(
                    [None, ["agent:*"], ["coalition:*"], ["world:*"]]
                ),
            },
            {
                "type": "query",
                "query_type": "activity_heatmap",
                "resolution": self.rng.choice(["1m", "5m", "15m", "1h"]),
                "metrics": ["agent_density", "action_frequency", "event_rate"],
            },
        ]

        return self.rng.choice(queries)

    def _generate_passive_monitoring(self) -> Dict[str, Any]:
        """Configure passive monitoring."""
        return {
            "type": "start_monitoring",
            "config": {
                "metrics": [
                    "active_agents",
                    "active_coalitions",
                    "event_rate",
                    "system_load",
                    "error_rate",
                ],
                "sample_rate": self.rng.choice([5.0, 10.0, 30.0]),
                "passive": True,
                "visualization": self.rng.choice(
                    ["dashboard", "timeline", "graph"]
                ),
            },
        }

    def _generate_minimal_command(self) -> Dict[str, Any]:
        """Generate minimal, non-intrusive commands."""
        # Observers rarely issue commands, usually just viewing
        commands = [
            {
                "type": "command",
                "command": "snapshot",
                "params": {"include_all": True},
            },
            {
                "type": "command",
                "command": "export_data",
                "params": {
                    "format": self.rng.choice(["json", "csv", "parquet"]),
                    "time_range": self.rng.choice([3600, 7200, 86400]),
                },
            },
        ]

        return self.rng.choice(commands)


class AdminBehavior(UserBehavior):
    """Behavior simulation for admin persona."""

    async def decide_next_action(self) -> Optional[Dict[str, Any]]:
        """Admin focuses on system management and monitoring."""
        if not self.should_act():
            return None

        # Admins have scheduled maintenance tasks
        current_hour = datetime.now().hour
        if current_hour in [0, 6, 12, 18] and self.rng.random() < 0.3:
            return self._generate_maintenance_task()

        action_type = self.rng.choice(
            list(self.profile.message_weights.keys()),
            p=list(self.profile.message_weights.values()),
        )

        if action_type == "monitoring":
            return self._generate_system_monitoring()
        elif action_type == "query":
            return self._generate_admin_query()
        elif action_type == "command":
            return self._generate_admin_command()
        elif action_type == "event":
            return self._generate_system_events()
        else:
            return {"type": "ping"}

    def _generate_system_monitoring(self) -> Dict[str, Any]:
        """Configure comprehensive system monitoring."""
        return {
            "type": "start_monitoring",
            "config": {
                "metrics": [
                    "cpu_usage",
                    "memory_usage",
                    "disk_usage",
                    "network_throughput",
                    "database_connections",
                    "websocket_connections",
                    "error_rate",
                    "response_time",
                    "queue_depth",
                ],
                "sample_rate": 1.0,
                "alert_rules": [
                    {"metric": "cpu_usage", "threshold": 80, "duration": 300},
                    {
                        "metric": "memory_usage",
                        "threshold": 85,
                        "duration": 300,
                    },
                    {
                        "metric": "error_rate",
                        "threshold": 0.05,
                        "duration": 60,
                    },
                    {
                        "metric": "response_time",
                        "threshold": 1000,
                        "duration": 120,
                    },
                ],
                "export_prometheus": True,
            },
        }

    def _generate_admin_query(self) -> Dict[str, Any]:
        """Query system health and performance."""
        queries = [
            {
                "type": "query",
                "query_type": "system_health",
                "components": [
                    "api",
                    "database",
                    "websocket",
                    "inference_engine",
                ],
                "include_diagnostics": True,
            },
            {
                "type": "query",
                "query_type": "performance_report",
                "time_range": self.rng.choice([3600, 86400, 604800]),
                "metrics": [
                    "throughput",
                    "latency",
                    "error_rate",
                    "resource_usage",
                ],
            },
            {
                "type": "query",
                "query_type": "user_activity",
                "group_by": self.rng.choice(
                    ["persona_type", "connection_pattern", "hour"]
                ),
                "include_anomalies": True,
            },
            {
                "type": "query",
                "query_type": "database_stats",
                "tables": [
                    "agents",
                    "coalitions",
                    "websocket_connections",
                    "events",
                ],
                "metrics": [
                    "row_count",
                    "size",
                    "index_usage",
                    "slow_queries",
                ],
            },
        ]

        return self.rng.choice(queries)

    def _generate_admin_command(self) -> Dict[str, Any]:
        """Generate administrative commands."""
        commands = [
            "restart_service",
            "clear_cache",
            "optimize_database",
            "backup_data",
            "update_configuration",
            "scale_resources",
            "enable_maintenance_mode",
        ]

        command = self.rng.choice(commands)

        if command == "restart_service":
            return {
                "type": "command",
                "command": "restart_service",
                "params": {
                    "service": self.rng.choice(
                        ["inference_engine", "websocket_server", "monitoring"]
                    ),
                    "graceful": True,
                    "wait_for_drain": True,
                },
            }

        elif command == "scale_resources":
            return {
                "type": "command",
                "command": "scale_resources",
                "params": {
                    "component": self.rng.choice(
                        [
                            "worker_pool",
                            "database_connections",
                            "websocket_handlers",
                        ]
                    ),
                    "action": self.rng.choice(["scale_up", "scale_down"]),
                    "amount": self.rng.integers(1, 5),
                },
            }

        elif command == "update_configuration":
            return {
                "type": "command",
                "command": "update_config",
                "params": {
                    "section": self.rng.choice(
                        ["performance", "security", "logging", "limits"]
                    ),
                    "changes": {
                        "max_connections": self.rng.integers(100, 1000),
                        "timeout": self.rng.integers(30, 300),
                        "log_level": self.rng.choice(
                            ["debug", "info", "warning", "error"]
                        ),
                    },
                },
            }

        return {"type": "command", "command": command}

    def _generate_system_events(self) -> Dict[str, Any]:
        """Subscribe to system-level events."""
        return {
            "type": "subscribe",
            "event_types": [
                "system:alert",
                "system:error",
                "system:performance_degradation",
                "system:resource_limit",
                "system:security_event",
                "system:maintenance_required",
            ],
            "priority": "high",
        }

    def _generate_maintenance_task(self) -> Dict[str, Any]:
        """Generate scheduled maintenance tasks."""
        tasks = [
            {
                "type": "command",
                "command": "cleanup_old_data",
                "params": {
                    "older_than_days": self.rng.choice([7, 14, 30]),
                    "tables": [
                        "websocket_events",
                        "agent_history",
                        "monitoring_data",
                    ],
                },
            },
            {
                "type": "command",
                "command": "analyze_tables",
                "params": {
                    "tables": ["agents", "coalitions", "events"],
                    "update_statistics": True,
                },
            },
            {
                "type": "command",
                "command": "rotate_logs",
                "params": {
                    "compress": True,
                    "archive_location": "/backup/logs",
                },
            },
        ]

        return self.rng.choice(tasks)


# Persona factory
def create_persona(persona_type: PersonaType, **kwargs) -> PersonaProfile:
    """Create a persona profile with default or custom settings."""
    profiles = {
        PersonaType.RESEARCHER: PersonaProfile(
            persona_type=PersonaType.RESEARCHER,
            activity_level=ActivityLevel.ACTIVE,
            interaction_pattern=InteractionPattern.BURSTY,
            attention_span=(60.0, 600.0),
            response_time=(0.5, 3.0),
            multitasking_level=3,
            preferred_agents=["experimental_agent_*", "research_agent_*"],
            favorite_queries=[
                "belief_evolution",
                "performance_metrics",
                "learning_curves",
            ],
            monitoring_interests=[
                "free_energy",
                "epistemic_value",
                "learning_rate",
            ],
        ),
        PersonaType.COORDINATOR: PersonaProfile(
            persona_type=PersonaType.COORDINATOR,
            activity_level=ActivityLevel.HYPERACTIVE,
            interaction_pattern=InteractionPattern.CONTINUOUS,
            attention_span=(30.0, 180.0),
            response_time=(0.1, 1.0),
            error_rate=0.01,
            multitasking_level=5,
            preferred_coalitions=["task_force_*", "coordination_group_*"],
            favorite_queries=[
                "coalition_status",
                "agent_availability",
                "task_progress",
            ],
        ),
        PersonaType.OBSERVER: PersonaProfile(
            persona_type=PersonaType.OBSERVER,
            activity_level=ActivityLevel.PASSIVE,
            interaction_pattern=InteractionPattern.CONTINUOUS,
            attention_span=(300.0, 1800.0),
            response_time=(1.0, 5.0),
            multitasking_level=1,
            connection_stability=0.99,
            favorite_queries=[
                "event_stream",
                "system_overview",
                "activity_patterns",
            ],
        ),
        PersonaType.ADMIN: PersonaProfile(
            persona_type=PersonaType.ADMIN,
            activity_level=ActivityLevel.MODERATE,
            interaction_pattern=InteractionPattern.SCHEDULED,
            attention_span=(60.0, 300.0),
            response_time=(0.2, 1.5),
            error_rate=0.005,
            multitasking_level=4,
            active_hours=list(range(24)),  # 24/7 availability
            peak_hours=[9, 10, 14, 15, 20, 21],
            monitoring_interests=[
                "system_health",
                "resource_usage",
                "error_rates",
            ],
        ),
        PersonaType.DEVELOPER: PersonaProfile(
            persona_type=PersonaType.DEVELOPER,
            activity_level=ActivityLevel.ACTIVE,
            interaction_pattern=InteractionPattern.BURSTY,
            attention_span=(120.0, 900.0),
            response_time=(0.3, 2.0),
            error_rate=0.03,  # Higher error rate due to experimentation
            multitasking_level=4,
            active_hours=list(range(8, 20)) + [22, 23],  # Late night coding
            favorite_queries=[
                "debug_info",
                "stack_traces",
                "performance_profiling",
            ],
        ),
        PersonaType.ANALYST: PersonaProfile(
            persona_type=PersonaType.ANALYST,
            activity_level=ActivityLevel.MODERATE,
            interaction_pattern=InteractionPattern.SCHEDULED,
            attention_span=(180.0, 1200.0),
            response_time=(1.0, 4.0),
            multitasking_level=2,
            active_hours=list(range(9, 18)),
            favorite_queries=[
                "aggregate_stats",
                "trend_analysis",
                "correlation_matrix",
            ],
            monitoring_interests=[
                "kpi_metrics",
                "anomaly_detection",
                "predictive_indicators",
            ],
        ),
    }

    profile = profiles.get(persona_type)
    if not profile:
        raise ValueError(f"Unknown persona type: {persona_type}")

    # Apply custom settings
    for key, value in kwargs.items():
        if hasattr(profile, key):
            setattr(profile, key, value)

    return profile


# Behavior factory
def create_behavior(
    user_id: str, persona_type: PersonaType, **kwargs
) -> UserBehavior:
    """Create a user behavior instance."""
    profile = create_persona(persona_type, **kwargs)

    behavior_classes = {
        PersonaType.RESEARCHER: ResearcherBehavior,
        PersonaType.COORDINATOR: CoordinatorBehavior,
        PersonaType.OBSERVER: ObserverBehavior,
        PersonaType.ADMIN: AdminBehavior,
        PersonaType.DEVELOPER: ResearcherBehavior,  # Similar to researcher
        PersonaType.ANALYST: ObserverBehavior,  # Similar to observer
    }

    behavior_class = behavior_classes.get(persona_type, UserBehavior)
    return behavior_class(user_id, profile)
