"""Concurrent user simulation framework.

This module manages realistic concurrent user simulations that integrate
database operations with WebSocket interactions.
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from database.models import Agent, AgentStatus, Coalition, CoalitionStatus
from tests.db_infrastructure.factories import AgentFactory, CoalitionFactory
from tests.db_infrastructure.performance_monitor import PerformanceMonitor
from tests.simulation.user_personas import (
    ActivityLevel,
    InteractionPattern,
    PersonaType,
    UserBehavior,
    create_behavior,
)
from tests.websocket_load.client_manager import WebSocketClient, WebSocketClientManager
from tests.websocket_load.message_generators import MessageGenerator
from tests.websocket_load.metrics_collector import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for simulation scenarios."""

    name: str
    description: str
    duration_seconds: float

    # User distribution
    user_distribution: Dict[PersonaType, int] = field(default_factory=dict)
    total_users: Optional[int] = None  # If None, sum of distribution

    # Timing configuration
    user_spawn_rate: float = 1.0  # Users per second
    warmup_period: float = 30.0  # Warmup time before main simulation
    cooldown_period: float = 30.0  # Cooldown after main simulation

    # Database configuration
    db_url: str = "postgresql://localhost/freeagentics_test"
    db_pool_size: int = 20
    db_max_overflow: int = 10

    # WebSocket configuration
    ws_base_url: str = "ws://localhost:8000"
    ws_path: str = "/ws"
    ws_reconnect_attempts: int = 3
    ws_reconnect_delay: float = 1.0

    # Simulation parameters
    enable_errors: bool = True
    error_injection_rate: float = 0.01
    network_latency_range: Tuple[float, float] = (0.01, 0.1)

    # Monitoring
    enable_monitoring: bool = True
    metrics_interval: float = 5.0
    export_results: bool = True
    results_path: Path = field(default_factory=lambda: Path("simulation_results"))

    def __post_init__(self):
        """Calculate total users if not specified."""
        if self.total_users is None:
            self.total_users = sum(self.user_distribution.values())


@dataclass
class SimulationMetrics:
    """Metrics collected during simulation."""

    start_time: float = 0.0
    end_time: float = 0.0

    # User metrics
    users_created: int = 0
    users_connected: int = 0
    users_active: int = 0
    users_errored: int = 0

    # Message metrics
    messages_sent: int = 0
    messages_received: int = 0
    messages_failed: int = 0

    # Database metrics
    db_operations: int = 0
    db_errors: int = 0
    db_latency_ms: List[float] = field(default_factory=list)

    # WebSocket metrics
    ws_connections: int = 0
    ws_disconnections: int = 0
    ws_errors: int = 0
    ws_latency_ms: List[float] = field(default_factory=list)

    # System metrics
    cpu_usage: List[float] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)

    # Per-persona metrics
    persona_metrics: Dict[str, Dict[str, Any]] = field(default_factory=lambda: defaultdict(dict))

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "duration_seconds": self.end_time - self.start_time,
            "users": {
                "created": self.users_created,
                "connected": self.users_connected,
                "active": self.users_active,
                "errored": self.users_errored,
            },
            "messages": {
                "sent": self.messages_sent,
                "received": self.messages_received,
                "failed": self.messages_failed,
                "success_rate": self.messages_received / max(self.messages_sent, 1),
            },
            "database": {
                "operations": self.db_operations,
                "errors": self.db_errors,
                "avg_latency_ms": np.mean(self.db_latency_ms) if self.db_latency_ms else 0,
                "p95_latency_ms": (
                    np.percentile(self.db_latency_ms, 95) if self.db_latency_ms else 0
                ),
            },
            "websocket": {
                "connections": self.ws_connections,
                "disconnections": self.ws_disconnections,
                "errors": self.ws_errors,
                "avg_latency_ms": np.mean(self.ws_latency_ms) if self.ws_latency_ms else 0,
                "p95_latency_ms": (
                    np.percentile(self.ws_latency_ms, 95) if self.ws_latency_ms else 0
                ),
            },
            "system": {
                "avg_cpu_usage": np.mean(self.cpu_usage) if self.cpu_usage else 0,
                "max_cpu_usage": np.max(self.cpu_usage) if self.cpu_usage else 0,
                "avg_memory_mb": np.mean(self.memory_usage) if self.memory_usage else 0,
                "max_memory_mb": np.max(self.memory_usage) if self.memory_usage else 0,
            },
            "personas": dict(self.persona_metrics),
        }


class SimulatedUser:
    """A simulated user with database and WebSocket operations."""

    def __init__(
        self,
        user_id: str,
        behavior: UserBehavior,
        db_session: AsyncSession,
        ws_client: WebSocketClient,
        config: SimulationConfig,
    ):
        """Initialize simulated user."""
        self.user_id = user_id
        self.behavior = behavior
        self.db_session = db_session
        self.ws_client = ws_client
        self.config = config

        # State tracking
        self.is_active = False
        self.connected = False
        self.message_count = 0
        self.error_count = 0
        self.last_action_time = None

        # Database entities
        self.owned_agents: Set[str] = set()
        self.managed_coalitions: Set[str] = set()

        # Performance tracking
        self.action_latencies: List[float] = []
        self.db_latencies: List[float] = []

    async def connect(self) -> bool:
        """Connect to WebSocket server."""
        try:
            await self.ws_client.connect()
            self.connected = True
            self.behavior.state["connected"] = True

            # Initialize subscriptions based on persona
            initial_events = self._get_initial_subscriptions()
            if initial_events:
                await self.ws_client.send_message(
                    {
                        "type": "subscribe",
                        "event_types": initial_events,
                    }
                )

            return True
        except Exception as e:
            logger.error(f"User {self.user_id} connection failed: {e}")
            self.error_count += 1
            return False

    async def disconnect(self):
        """Disconnect from WebSocket server."""
        try:
            await self.ws_client.disconnect()
            self.connected = False
            self.behavior.state["connected"] = False
        except Exception as e:
            logger.error(f"User {self.user_id} disconnect error: {e}")

    async def simulate_action(self) -> Optional[Dict[str, Any]]:
        """Simulate a single user action."""
        if not self.connected:
            return None

        start_time = time.time()

        try:
            # Get next action from behavior model
            action = await self.behavior.decide_next_action()
            if not action:
                return None

            # Add network latency simulation
            if self.config.network_latency_range:
                latency = np.random.uniform(*self.config.network_latency_range)
                await asyncio.sleep(latency)

            # Process action based on type
            result = await self._process_action(action)

            # Record metrics
            action_time = time.time() - start_time
            self.action_latencies.append(action_time * 1000)  # Convert to ms
            self.message_count += 1
            self.last_action_time = time.time()

            # Record action in behavior history
            self.behavior.record_action(
                {
                    **action,
                    "result": result,
                    "latency_ms": action_time * 1000,
                }
            )

            return result

        except Exception as e:
            logger.error(f"User {self.user_id} action error: {e}")
            self.error_count += 1

            # Inject errors if configured
            if self.config.enable_errors and np.random.random() < self.config.error_injection_rate:
                raise e

            return None

    async def _process_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Process different types of actions."""
        action_type = action.get("type")

        if action_type == "query":
            return await self._process_query(action)
        elif action_type == "command":
            return await self._process_command(action)
        elif action_type == "subscribe":
            return await self._process_subscription(action)
        elif action_type == "start_monitoring":
            return await self._process_monitoring(action)
        elif action_type == "ping":
            return await self._process_ping()
        else:
            # Send as generic WebSocket message
            await self.ws_client.send_message(action)
            return {"status": "sent"}

    async def _process_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Process query with database integration."""
        query_type = query.get("query_type")

        db_start = time.time()

        try:
            if query_type == "agent_status":
                # Query database for agent status
                agent_ids = query.get("agent_ids", [])
                if not agent_ids and self.owned_agents:
                    agent_ids = list(self.owned_agents)[:5]  # Limit to 5

                if agent_ids:
                    result = await self.db_session.execute(
                        select(Agent).where(Agent.id.in_(agent_ids))
                    )
                    agents = result.scalars().all()

                    # Send WebSocket query
                    await self.ws_client.send_message(
                        {
                            **query,
                            "agent_ids": [str(a.id) for a in agents],
                        }
                    )

                    self.db_latencies.append((time.time() - db_start) * 1000)
                    return {"agents": len(agents)}

            elif query_type == "coalition_status":
                # Query coalitions from database
                coalition_ids = query.get("coalition_ids", [])
                if not coalition_ids and self.managed_coalitions:
                    coalition_ids = list(self.managed_coalitions)

                if coalition_ids:
                    result = await self.db_session.execute(
                        select(Coalition).where(Coalition.id.in_(coalition_ids))
                    )
                    coalitions = result.scalars().all()

                    await self.ws_client.send_message(
                        {
                            **query,
                            "coalition_ids": [str(c.id) for c in coalitions],
                        }
                    )

                    self.db_latencies.append((time.time() - db_start) * 1000)
                    return {"coalitions": len(coalitions)}

            # Default: send query via WebSocket
            await self.ws_client.send_message(query)
            return {"status": "queried"}

        except Exception as e:
            logger.error(f"Query processing error: {e}")
            self.error_count += 1
            return {"error": str(e)}

    async def _process_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Process command with database operations."""
        command_type = command.get("command")

        db_start = time.time()

        try:
            if command_type == "create_agent":
                # Create agent in database
                agent_data = command.get("params", {})
                agent = Agent(
                    name=agent_data.get("name", f"Agent_{uuid.uuid4().hex[:8]}"),
                    template=agent_data.get("template", "default"),
                    status=AgentStatus.PENDING,
                    parameters=agent_data.get("parameters", {}),
                    gmn_spec=agent_data.get("gmn_spec"),
                )
                self.db_session.add(agent)
                await self.db_session.commit()

                self.owned_agents.add(str(agent.id))
                self.behavior.state["active_agents"].add(str(agent.id))

                # Send WebSocket command
                await self.ws_client.send_message(
                    {
                        **command,
                        "params": {
                            **agent_data,
                            "agent_id": str(agent.id),
                        },
                    }
                )

                self.db_latencies.append((time.time() - db_start) * 1000)
                return {"agent_id": str(agent.id)}

            elif command_type == "form_coalition":
                # Create coalition in database
                coalition_data = command.get("params", {})
                coalition = Coalition(
                    name=coalition_data.get("name", f"Coalition_{uuid.uuid4().hex[:8]}"),
                    goal=coalition_data.get("goal", "default_goal"),
                    status=CoalitionStatus.FORMING,
                    metadata=coalition_data,
                )
                self.db_session.add(coalition)
                await self.db_session.commit()

                self.managed_coalitions.add(str(coalition.id))
                self.behavior.state["managed_coalitions"].add(str(coalition.id))

                # Send WebSocket command
                await self.ws_client.send_message(
                    {
                        **command,
                        "params": {
                            **coalition_data,
                            "coalition_id": str(coalition.id),
                        },
                    }
                )

                self.db_latencies.append((time.time() - db_start) * 1000)
                return {"coalition_id": str(coalition.id)}

            # Default: send command via WebSocket
            await self.ws_client.send_message(command)
            return {"status": "commanded"}

        except Exception as e:
            logger.error(f"Command processing error: {e}")
            self.error_count += 1
            await self.db_session.rollback()
            return {"error": str(e)}

    async def _process_subscription(self, subscription: Dict[str, Any]) -> Dict[str, Any]:
        """Process event subscription."""
        event_types = subscription.get("event_types", [])

        # Update behavior state
        for event_type in event_types:
            self.behavior.state["subscribed_events"].add(event_type)

        # Send subscription via WebSocket
        await self.ws_client.send_message(subscription)

        return {"subscribed": len(event_types)}

    async def _process_monitoring(self, monitoring: Dict[str, Any]) -> Dict[str, Any]:
        """Process monitoring configuration."""
        config = monitoring.get("config", {})

        # Create monitoring session ID
        session_id = str(uuid.uuid4())
        self.behavior.state["monitoring_sessions"].add(session_id)

        # Send monitoring config via WebSocket
        await self.ws_client.send_message(
            {
                **monitoring,
                "session_id": session_id,
            }
        )

        return {"session_id": session_id}

    async def _process_ping(self) -> Dict[str, Any]:
        """Process ping message."""
        await self.ws_client.send_message({"type": "ping"})
        return {"status": "pinged"}

    def _get_initial_subscriptions(self) -> List[str]:
        """Get initial event subscriptions based on persona."""
        persona_subscriptions = {
            PersonaType.RESEARCHER: [
                "agent:belief_updated",
                "agent:learning_complete",
                "experiment:checkpoint",
            ],
            PersonaType.COORDINATOR: [
                "coalition:formation_request",
                "agent:assistance_needed",
                "coalition:member_left",
            ],
            PersonaType.OBSERVER: [
                "system:performance",
                "agent:action",
                "world:updated",
            ],
            PersonaType.ADMIN: [
                "system:alert",
                "system:error",
                "system:resource_limit",
            ],
        }

        return persona_subscriptions.get(
            self.behavior.profile.persona_type, ["agent:created", "agent:action"]
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get user metrics."""
        return {
            "user_id": self.user_id,
            "persona": self.behavior.profile.persona_type.value,
            "connected": self.connected,
            "message_count": self.message_count,
            "error_count": self.error_count,
            "owned_agents": len(self.owned_agents),
            "managed_coalitions": len(self.managed_coalitions),
            "avg_action_latency_ms": np.mean(self.action_latencies) if self.action_latencies else 0,
            "avg_db_latency_ms": np.mean(self.db_latencies) if self.db_latencies else 0,
        }


class ConcurrentSimulator:
    """Manages concurrent user simulations."""

    def __init__(self, config: SimulationConfig):
        """Initialize simulator."""
        self.config = config
        self.metrics = SimulationMetrics()

        # WebSocket manager
        self.ws_manager = WebSocketClientManager(base_url=f"{config.ws_base_url}{config.ws_path}")

        # Performance monitor
        self.perf_monitor = PerformanceMonitor()

        # Active users
        self.users: Dict[str, SimulatedUser] = {}
        self.user_tasks: Dict[str, asyncio.Task] = {}

        # Simulation state
        self.is_running = False
        self.start_time = None
        self.db_engine = None
        self.db_session_maker = None

    async def setup(self):
        """Setup simulation resources."""
        logger.info(f"Setting up simulation: {self.config.name}")

        # Setup database
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        self.db_engine = create_async_engine(
            self.config.db_url,
            pool_size=self.config.db_pool_size,
            max_overflow=self.config.db_max_overflow,
        )
        self.db_session_maker = async_sessionmaker(self.db_engine)

        # Create results directory
        if self.config.export_results:
            self.config.results_path.mkdir(parents=True, exist_ok=True)

        # Start performance monitoring
        if self.config.enable_monitoring:
            self.perf_monitor.start_monitoring()

    async def teardown(self):
        """Cleanup simulation resources."""
        logger.info("Tearing down simulation")

        # Stop all user tasks
        for task in self.user_tasks.values():
            task.cancel()

        # Wait for tasks to complete
        if self.user_tasks:
            await asyncio.gather(*self.user_tasks.values(), return_exceptions=True)

        # Disconnect all users
        for user in self.users.values():
            await user.disconnect()

        # Close database connections
        if self.db_engine:
            await self.db_engine.dispose()

        # Stop performance monitoring
        if self.config.enable_monitoring:
            self.perf_monitor.stop_monitoring()

        # Export results
        if self.config.export_results:
            await self._export_results()

    async def run(self):
        """Run the simulation."""
        try:
            await self.setup()

            self.is_running = True
            self.start_time = time.time()
            self.metrics.start_time = self.start_time

            # Run simulation phases
            await self._warmup_phase()
            await self._main_phase()
            await self._cooldown_phase()

        finally:
            self.is_running = False
            self.metrics.end_time = time.time()
            await self.teardown()

    async def _warmup_phase(self):
        """Warmup phase with gradual user spawning."""
        logger.info(f"Starting warmup phase ({self.config.warmup_period}s)")

        warmup_start = time.time()
        users_to_spawn = int(self.config.total_users * 0.3)  # 30% during warmup

        spawn_task = asyncio.create_task(self._spawn_users(users_to_spawn))
        monitor_task = asyncio.create_task(self._monitor_metrics())

        await asyncio.sleep(self.config.warmup_period)

        spawn_task.cancel()
        monitor_task.cancel()

        try:
            await spawn_task
        except asyncio.CancelledError:
            pass

        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

        logger.info(
            f"Warmup complete. Active users: {len([u for u in self.users.values() if u.connected])}"
        )

    async def _main_phase(self):
        """Main simulation phase."""
        logger.info(f"Starting main phase ({self.config.duration_seconds}s)")

        # Spawn remaining users
        remaining_users = self.config.total_users - len(self.users)
        spawn_task = asyncio.create_task(self._spawn_users(remaining_users))

        # Run monitoring
        monitor_task = asyncio.create_task(self._monitor_metrics())

        # Run for specified duration
        await asyncio.sleep(self.config.duration_seconds)

        spawn_task.cancel()
        monitor_task.cancel()

        try:
            await spawn_task
        except asyncio.CancelledError:
            pass

        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

        logger.info(f"Main phase complete. Total messages: {self.metrics.messages_sent}")

    async def _cooldown_phase(self):
        """Cooldown phase with gradual disconnection."""
        logger.info(f"Starting cooldown phase ({self.config.cooldown_period}s)")

        cooldown_start = time.time()
        users_to_disconnect = list(self.users.keys())
        np.random.shuffle(users_to_disconnect)

        # Gradually disconnect users
        disconnect_rate = len(users_to_disconnect) / self.config.cooldown_period

        for i, user_id in enumerate(users_to_disconnect):
            if not self.is_running:
                break

            # Calculate when to disconnect this user
            disconnect_time = i / disconnect_rate
            await asyncio.sleep(max(0, disconnect_time - (time.time() - cooldown_start)))

            # Disconnect user
            user = self.users.get(user_id)
            if user and user.connected:
                await user.disconnect()
                self.metrics.ws_disconnections += 1

        logger.info("Cooldown complete")

    async def _spawn_users(self, count: int):
        """Spawn users according to distribution."""
        users_spawned = 0

        # Calculate users per persona
        total_weight = sum(self.config.user_distribution.values())
        persona_counts = {
            persona: int(count * weight / total_weight)
            for persona, weight in self.config.user_distribution.items()
        }

        # Adjust for rounding
        diff = count - sum(persona_counts.values())
        if diff > 0:
            # Add remaining to most common persona
            most_common = max(persona_counts.keys(), key=lambda p: self.config.user_distribution[p])
            persona_counts[most_common] += diff

        # Spawn users by persona
        for persona_type, persona_count in persona_counts.items():
            for i in range(persona_count):
                if not self.is_running or users_spawned >= count:
                    return

                # Create user
                user_id = f"{persona_type.value}_{uuid.uuid4().hex[:8]}"
                user = await self._create_user(user_id, persona_type)

                if user:
                    self.users[user_id] = user
                    self.metrics.users_created += 1

                    # Start user simulation
                    task = asyncio.create_task(self._run_user_simulation(user))
                    self.user_tasks[user_id] = task

                    users_spawned += 1

                # Respect spawn rate
                await asyncio.sleep(1.0 / self.config.user_spawn_rate)

    async def _create_user(
        self, user_id: str, persona_type: PersonaType
    ) -> Optional[SimulatedUser]:
        """Create a simulated user."""
        try:
            # Create behavior model
            behavior = create_behavior(user_id, persona_type)

            # Create database session
            db_session = self.db_session_maker()

            # Create WebSocket client
            ws_client = await self.ws_manager.create_client(user_id)

            # Create simulated user
            user = SimulatedUser(
                user_id=user_id,
                behavior=behavior,
                db_session=db_session,
                ws_client=ws_client,
                config=self.config,
            )

            # Connect user
            if await user.connect():
                self.metrics.users_connected += 1
                self.metrics.ws_connections += 1
                return user
            else:
                self.metrics.users_errored += 1
                await db_session.close()
                return None

        except Exception as e:
            logger.error(f"Failed to create user {user_id}: {e}")
            self.metrics.users_errored += 1
            return None

    async def _run_user_simulation(self, user: SimulatedUser):
        """Run simulation for a single user."""
        try:
            while self.is_running and user.connected:
                # Simulate user action
                result = await user.simulate_action()

                if result:
                    self.metrics.messages_sent += 1

                    # Update persona metrics
                    persona_name = user.behavior.profile.persona_type.value
                    if persona_name not in self.metrics.persona_metrics:
                        self.metrics.persona_metrics[persona_name] = {
                            "users": 0,
                            "messages": 0,
                            "errors": 0,
                        }

                    self.metrics.persona_metrics[persona_name]["messages"] += 1

                # Response delay
                delay = user.behavior.get_response_delay()
                await asyncio.sleep(delay)

        except asyncio.CancelledError:
            # Normal cancellation
            pass
        except Exception as e:
            logger.error(f"User {user.user_id} simulation error: {e}")
            user.error_count += 1
            self.metrics.users_errored += 1
        finally:
            # Cleanup
            if user.connected:
                await user.disconnect()
            await user.db_session.close()

    async def _monitor_metrics(self):
        """Monitor and collect metrics during simulation."""
        while self.is_running:
            try:
                # Collect system metrics
                if self.config.enable_monitoring:
                    stats = self.perf_monitor.get_current_stats()
                    self.metrics.cpu_usage.append(stats["cpu"]["percent"])
                    self.metrics.memory_usage.append(stats["memory"]["used_mb"])

                # Collect user metrics
                active_users = sum(1 for u in self.users.values() if u.connected)
                self.metrics.users_active = active_users

                # Collect database metrics
                total_db_ops = sum(len(u.db_latencies) for u in self.users.values())
                self.metrics.db_operations = total_db_ops

                # Log progress
                elapsed = time.time() - self.start_time
                logger.info(
                    "Simulation progress - "
                    f"Time: {elapsed:.1f}s, "
                    f"Active users: {active_users}, "
                    f"Messages: {self.metrics.messages_sent}, "
                    f"DB ops: {total_db_ops}, "
                    f"Errors: {self.metrics.users_errored}"
                )

                await asyncio.sleep(self.config.metrics_interval)

            except Exception as e:
                logger.error(f"Metrics monitoring error: {e}")

    async def _export_results(self):
        """Export simulation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export metrics
        metrics_file = self.config.results_path / f"metrics_{self.config.name}_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump(self.metrics.to_dict(), f, indent=2)

        # Export user behaviors
        behaviors_file = self.config.results_path / f"behaviors_{self.config.name}_{timestamp}.json"
        behaviors = {
            user_id: {
                "metrics": user.get_metrics(),
                "history": user.behavior.history[-100:],  # Last 100 actions
            }
            for user_id, user in self.users.items()
        }

        with open(behaviors_file, "w") as f:
            json.dump(behaviors, f, indent=2)

        # Export performance data
        if self.config.enable_monitoring:
            perf_file = (
                self.config.results_path / f"performance_{self.config.name}_{timestamp}.json"
            )
            with open(perf_file, "w") as f:
                json.dump(self.perf_monitor.get_summary(), f, indent=2)

        logger.info(f"Results exported to {self.config.results_path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get simulation summary."""
        return {
            "config": {
                "name": self.config.name,
                "duration": self.config.duration_seconds,
                "total_users": self.config.total_users,
                "distribution": {p.value: c for p, c in self.config.user_distribution.items()},
            },
            "metrics": self.metrics.to_dict(),
            "users": {
                user_id: user.get_metrics()
                for user_id, user in list(self.users.items())[:10]  # Sample
            },
        }
