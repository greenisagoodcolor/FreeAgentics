"""
Security Orchestration, Automation, and Response (SOAR) Playbook Engine.
Executes automated security response workflows based on predefined playbooks.
"""

import asyncio
import logging
import os
import re
import threading
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


class ActionStatus(Enum):
    """Status of playbook actions."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"


class PlaybookTrigger(Enum):
    """Types of playbook triggers."""

    MANUAL = "manual"
    ALERT = "alert"
    SCHEDULED = "scheduled"
    THRESHOLD = "threshold"
    PATTERN = "pattern"


@dataclass
class PlaybookContext:
    """Context for playbook execution."""

    playbook_id: str
    execution_id: str
    trigger: PlaybookTrigger
    trigger_data: Dict[str, Any]
    variables: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    status: ActionStatus = ActionStatus.PENDING
    error: Optional[str] = None


@dataclass
class ActionResult:
    """Result of a playbook action."""

    action_id: str
    status: ActionStatus
    output: Any
    error: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration_ms: Optional[int] = None


class PlaybookAction(ABC):
    """Base class for playbook actions."""

    def __init__(self, action_id: str, config: Dict[str, Any]):
        self.action_id = action_id
        self.config = config
        self.timeout = config.get("timeout", 300)  # 5 minutes default

    @abstractmethod
    async def execute(self, context: PlaybookContext) -> ActionResult:
        """Execute the action."""
        pass

    def validate_config(self) -> bool:
        """Validate action configuration."""
        return True

    def _resolve_variables(self, value: Any, context: PlaybookContext) -> Any:
        """Resolve variables in configuration values."""
        if isinstance(value, str):
            # Replace {{variable}} with actual values
            pattern = r"\{\{(\w+)\}\}"
            matches = re.findall(pattern, value)
            for match in matches:
                if match in context.variables:
                    value = value.replace(f"{{{{{match}}}}}", str(context.variables[match]))
            return value
        elif isinstance(value, dict):
            return {k: self._resolve_variables(v, context) for k, v in value.items()}
        elif isinstance(value, list):
            return [self._resolve_variables(item, context) for item in value]
        return value


class IPBlockAction(PlaybookAction):
    """Action to block IP addresses."""

    async def execute(self, context: PlaybookContext) -> ActionResult:
        """Block IP addresses."""
        start_time = datetime.utcnow()
        try:
            # Resolve IP addresses
            ips = self._resolve_variables(self.config.get("ip_addresses", []), context)
            duration = self.config.get("duration_hours", 24)

            # Simulate IP blocking (in production, integrate with firewall/WAF)
            blocked_ips = []
            for ip in ips:
                logger.info(f"Blocking IP {ip} for {duration} hours")
                blocked_ips.append(ip)
                # Here you would call actual firewall API

            # Store result
            context.artifacts["blocked_ips"] = blocked_ips

            end_time = datetime.utcnow()
            return ActionResult(
                action_id=self.action_id,
                status=ActionStatus.SUCCESS,
                output={
                    "blocked_ips": blocked_ips,
                    "duration_hours": duration,
                },
                end_time=end_time,
                duration_ms=int((end_time - start_time).total_seconds() * 1000),
            )
        except Exception as e:
            logger.error(f"Failed to block IPs: {e}")
            return ActionResult(
                action_id=self.action_id,
                status=ActionStatus.FAILED,
                output=None,
                error=str(e),
            )


class UserDisableAction(PlaybookAction):
    """Action to disable user accounts."""

    async def execute(self, context: PlaybookContext) -> ActionResult:
        """Disable user accounts."""
        start_time = datetime.utcnow()
        try:
            user_ids = self._resolve_variables(self.config.get("user_ids", []), context)
            reason = self._resolve_variables(
                self.config.get("reason", "Security incident"), context
            )

            disabled_users = []
            for user_id in user_ids:
                logger.info(f"Disabling user {user_id}: {reason}")
                # In production, integrate with identity management system
                disabled_users.append(user_id)

            context.artifacts["disabled_users"] = disabled_users

            end_time = datetime.utcnow()
            return ActionResult(
                action_id=self.action_id,
                status=ActionStatus.SUCCESS,
                output={"disabled_users": disabled_users, "reason": reason},
                end_time=end_time,
                duration_ms=int((end_time - start_time).total_seconds() * 1000),
            )
        except Exception as e:
            logger.error(f"Failed to disable users: {e}")
            return ActionResult(
                action_id=self.action_id,
                status=ActionStatus.FAILED,
                output=None,
                error=str(e),
            )


class NotificationAction(PlaybookAction):
    """Action to send notifications."""

    async def execute(self, context: PlaybookContext) -> ActionResult:
        """Send notifications."""
        start_time = datetime.utcnow()
        try:
            recipients = self._resolve_variables(self.config.get("recipients", []), context)
            message = self._resolve_variables(self.config.get("message", ""), context)
            channels = self.config.get("channels", ["email"])

            notifications_sent = []
            for channel in channels:
                for recipient in recipients:
                    logger.info(f"Sending {channel} notification to {recipient}")
                    # In production, integrate with notification services
                    notifications_sent.append(
                        {
                            "channel": channel,
                            "recipient": recipient,
                            "message": message,
                        }
                    )

            end_time = datetime.utcnow()
            return ActionResult(
                action_id=self.action_id,
                status=ActionStatus.SUCCESS,
                output={"notifications": notifications_sent},
                end_time=end_time,
                duration_ms=int((end_time - start_time).total_seconds() * 1000),
            )
        except Exception as e:
            logger.error(f"Failed to send notifications: {e}")
            return ActionResult(
                action_id=self.action_id,
                status=ActionStatus.FAILED,
                output=None,
                error=str(e),
            )


class ForensicsCollectionAction(PlaybookAction):
    """Action to collect forensic data."""

    async def execute(self, context: PlaybookContext) -> ActionResult:
        """Collect forensic data."""
        start_time = datetime.utcnow()
        try:
            targets = self._resolve_variables(self.config.get("targets", []), context)
            data_types = self.config.get("data_types", ["logs", "network", "processes"])

            forensic_data = {}
            for target in targets:
                forensic_data[target] = {}
                for data_type in data_types:
                    logger.info(f"Collecting {data_type} from {target}")
                    # In production, collect actual forensic data
                    forensic_data[target][data_type] = {
                        "collected_at": datetime.utcnow().isoformat(),
                        "size_bytes": 1024 * 1024,  # Simulated
                        "hash": "sha256:" + os.urandom(32).hex(),
                    }

            # Store forensic data reference
            context.artifacts["forensic_data"] = forensic_data

            end_time = datetime.utcnow()
            return ActionResult(
                action_id=self.action_id,
                status=ActionStatus.SUCCESS,
                output={"forensic_data": forensic_data},
                end_time=end_time,
                duration_ms=int((end_time - start_time).total_seconds() * 1000),
            )
        except Exception as e:
            logger.error(f"Failed to collect forensics: {e}")
            return ActionResult(
                action_id=self.action_id,
                status=ActionStatus.FAILED,
                output=None,
                error=str(e),
            )


class ConditionalAction(PlaybookAction):
    """Action that executes based on conditions."""

    async def execute(self, context: PlaybookContext) -> ActionResult:
        """Evaluate condition and execute."""
        start_time = datetime.utcnow()
        try:
            condition = self._resolve_variables(self.config.get("condition", ""), context)

            # Simple condition evaluation (in production, use safe expression evaluator)
            result = self._evaluate_condition(condition, context)

            end_time = datetime.utcnow()
            return ActionResult(
                action_id=self.action_id,
                status=ActionStatus.SUCCESS if result else ActionStatus.SKIPPED,
                output={"condition_met": result},
                end_time=end_time,
                duration_ms=int((end_time - start_time).total_seconds() * 1000),
            )
        except Exception as e:
            logger.error(f"Failed to evaluate condition: {e}")
            return ActionResult(
                action_id=self.action_id,
                status=ActionStatus.FAILED,
                output=None,
                error=str(e),
            )

    def _evaluate_condition(self, condition: str, context: PlaybookContext) -> bool:
        """Safely evaluate condition."""
        # Simple comparison for demonstration
        # In production, use a proper expression evaluator
        if "==" in condition:
            left, right = condition.split("==")
            return left.strip() == right.strip()
        elif ">" in condition:
            left, right = condition.split(">")
            return float(left.strip()) > float(right.strip())
        elif "<" in condition:
            left, right = condition.split("<")
            return float(left.strip()) < float(right.strip())
        return False


class PlaybookEngine:
    """
    Main SOAR playbook execution engine.
    Manages playbook lifecycle and orchestrates actions.
    """

    def __init__(
        self,
        playbook_dir: str = "./playbooks",
        max_concurrent_executions: int = 10,
        default_timeout: int = 3600,  # 1 hour
    ):
        self.playbook_dir = playbook_dir
        self.max_concurrent_executions = max_concurrent_executions
        self.default_timeout = default_timeout

        # Action registry
        self.action_registry = {
            "block_ip": IPBlockAction,
            "disable_user": UserDisableAction,
            "send_notification": NotificationAction,
            "collect_forensics": ForensicsCollectionAction,
            "conditional": ConditionalAction,
        }

        # Execution tracking
        self.active_executions = {}
        self.execution_history = []
        self.execution_lock = threading.Lock()

        # Thread pool for concurrent executions
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_executions)

        # Load playbooks
        self.playbooks = {}
        self._load_playbooks()

    def _load_playbooks(self):
        """Load playbooks from directory."""
        if not os.path.exists(self.playbook_dir):
            os.makedirs(self.playbook_dir)
            return

        for filename in os.listdir(self.playbook_dir):
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                filepath = os.path.join(self.playbook_dir, filename)
                try:
                    with open(filepath, "r") as f:
                        playbook = yaml.safe_load(f)
                        playbook_id = playbook.get("id", filename)
                        self.playbooks[playbook_id] = playbook
                        logger.info(f"Loaded playbook: {playbook_id}")
                except Exception as e:
                    logger.error(f"Failed to load playbook {filename}: {e}")

    def register_action(self, action_type: str, action_class: type):
        """Register custom action type."""
        self.action_registry[action_type] = action_class

    async def execute_playbook(
        self,
        playbook_id: str,
        trigger: PlaybookTrigger,
        trigger_data: Dict[str, Any],
        variables: Optional[Dict[str, Any]] = None,
    ) -> PlaybookContext:
        """
        Execute a playbook asynchronously.
        """
        if playbook_id not in self.playbooks:
            raise ValueError(f"Playbook {playbook_id} not found")

        # Create execution context
        execution_id = str(uuid.uuid4())
        context = PlaybookContext(
            playbook_id=playbook_id,
            execution_id=execution_id,
            trigger=trigger,
            trigger_data=trigger_data,
            variables=variables or {},
        )

        # Check concurrent execution limit
        with self.execution_lock:
            if len(self.active_executions) >= self.max_concurrent_executions:
                context.status = ActionStatus.FAILED
                context.error = "Maximum concurrent executions reached"
                return context

            self.active_executions[execution_id] = context

        try:
            # Execute playbook
            playbook = self.playbooks[playbook_id]
            context.status = ActionStatus.RUNNING

            # Execute actions in sequence
            for action_config in playbook.get("actions", []):
                if context.status != ActionStatus.RUNNING:
                    break

                action_result = await self._execute_action(action_config, context)

                # Handle action result
                if action_result.status == ActionStatus.FAILED:
                    if action_config.get("continue_on_error", False):
                        logger.warning(f"Action {action_result.action_id} failed but continuing")
                    else:
                        context.status = ActionStatus.FAILED
                        context.error = (
                            f"Action {action_result.action_id} failed: {action_result.error}"
                        )
                        break

            if context.status == ActionStatus.RUNNING:
                context.status = ActionStatus.SUCCESS

        except Exception as e:
            logger.error(f"Playbook execution failed: {e}")
            context.status = ActionStatus.FAILED
            context.error = str(e)
        finally:
            # Cleanup
            context.end_time = datetime.utcnow()
            with self.execution_lock:
                del self.active_executions[execution_id]
                self.execution_history.append(context)
                # Keep only last 1000 executions
                if len(self.execution_history) > 1000:
                    self.execution_history = self.execution_history[-1000:]

        return context

    async def _execute_action(
        self, action_config: Dict[str, Any], context: PlaybookContext
    ) -> ActionResult:
        """Execute a single action."""
        action_type = action_config.get("type")
        action_id = action_config.get("id", f"{action_type}_{uuid.uuid4().hex[:8]}")

        if action_type not in self.action_registry:
            return ActionResult(
                action_id=action_id,
                status=ActionStatus.FAILED,
                output=None,
                error=f"Unknown action type: {action_type}",
            )

        try:
            # Create action instance
            action_class = self.action_registry[action_type]
            action = action_class(action_id, action_config)

            # Validate configuration
            if not action.validate_config():
                return ActionResult(
                    action_id=action_id,
                    status=ActionStatus.FAILED,
                    output=None,
                    error="Invalid action configuration",
                )

            # Execute with timeout
            result = await asyncio.wait_for(action.execute(context), timeout=action.timeout)

            return result

        except asyncio.TimeoutError:
            return ActionResult(
                action_id=action_id,
                status=ActionStatus.TIMEOUT,
                output=None,
                error=f"Action timed out after {action.timeout} seconds",
            )
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return ActionResult(
                action_id=action_id,
                status=ActionStatus.FAILED,
                output=None,
                error=str(e),
            )

    def get_execution_status(self, execution_id: str) -> Optional[PlaybookContext]:
        """Get status of a playbook execution."""
        with self.execution_lock:
            # Check active executions
            if execution_id in self.active_executions:
                return self.active_executions[execution_id]

            # Check history
            for context in reversed(self.execution_history):
                if context.execution_id == execution_id:
                    return context

        return None

    def list_playbooks(self) -> List[Dict[str, Any]]:
        """List available playbooks."""
        return [
            {
                "id": playbook_id,
                "name": playbook.get("name", playbook_id),
                "description": playbook.get("description", ""),
                "triggers": playbook.get("triggers", []),
                "actions": len(playbook.get("actions", [])),
            }
            for playbook_id, playbook in self.playbooks.items()
        ]

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get execution metrics."""
        with self.execution_lock:
            total_executions = len(self.execution_history)
            successful = sum(1 for c in self.execution_history if c.status == ActionStatus.SUCCESS)
            failed = sum(1 for c in self.execution_history if c.status == ActionStatus.FAILED)

            avg_duration = 0
            if self.execution_history:
                durations = [
                    (c.end_time - c.start_time).total_seconds()
                    for c in self.execution_history
                    if c.end_time
                ]
                if durations:
                    avg_duration = sum(durations) / len(durations)

            return {
                "total_executions": total_executions,
                "successful": successful,
                "failed": failed,
                "success_rate": successful / total_executions if total_executions > 0 else 0,
                "active_executions": len(self.active_executions),
                "average_duration_seconds": avg_duration,
            }


# Example playbook
EXAMPLE_PLAYBOOK = """
id: brute_force_response
name: Brute Force Attack Response
description: Automated response to detected brute force attacks
triggers:
  - type: alert
    conditions:
      - alert_type: brute_force
      - severity: high
actions:
  - id: block_attacker_ip
    type: block_ip
    ip_addresses: ["{{attacker_ip}}"]
    duration_hours: 24

  - id: disable_target_account
    type: disable_user
    user_ids: ["{{target_user}}"]
    reason: "Potential account compromise from brute force attack"

  - id: collect_evidence
    type: collect_forensics
    targets: ["{{affected_server}}"]
    data_types: ["logs", "auth_logs", "network_captures"]

  - id: notify_security_team
    type: send_notification
    recipients: ["security@example.com", "soc@example.com"]
    channels: ["email", "slack"]
    message: |
      Brute force attack detected and mitigated:
      - Attacker IP: {{attacker_ip}}
      - Target User: {{target_user}}
      - Blocked IPs: {{blocked_ips}}
      - Forensic data collected
"""


def save_example_playbook(playbook_dir: str = "./playbooks"):
    """Save example playbook to file."""
    os.makedirs(playbook_dir, exist_ok=True)
    filepath = os.path.join(playbook_dir, "brute_force_response.yaml")
    with open(filepath, "w") as f:
        f.write(EXAMPLE_PLAYBOOK)
    logger.info(f"Saved example playbook to {filepath}")
