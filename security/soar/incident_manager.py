"""
Security Incident Manager with case management system.
Tracks, manages, and coordinates response to security incidents.
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from .playbook_engine import PlaybookEngine, PlaybookTrigger

logger = logging.getLogger(__name__)


class IncidentSeverity(Enum):
    """Incident severity levels."""

    CRITICAL = "critical"  # Immediate action required
    HIGH = "high"  # Significant impact
    MEDIUM = "medium"  # Moderate impact
    LOW = "low"  # Minor impact
    INFO = "info"  # Informational only


class IncidentStatus(Enum):
    """Incident lifecycle status."""

    NEW = "new"
    TRIAGED = "triaged"
    IN_PROGRESS = "in_progress"
    CONTAINED = "contained"
    ERADICATED = "eradicated"
    RECOVERED = "recovered"
    CLOSED = "closed"


class IncidentType(Enum):
    """Types of security incidents."""

    MALWARE = "malware"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    DENIAL_OF_SERVICE = "denial_of_service"
    PHISHING = "phishing"
    INSIDER_THREAT = "insider_threat"
    SUPPLY_CHAIN = "supply_chain"
    ZERO_DAY = "zero_day"
    MISCONFIGURATION = "misconfiguration"
    OTHER = "other"


@dataclass
class IncidentIndicator:
    """Indicator of Compromise (IoC)."""

    type: str  # ip, domain, hash, email, etc.
    value: str
    confidence: float  # 0.0 to 1.0
    source: str
    first_seen: datetime
    last_seen: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IncidentTimeline:
    """Timeline entry for incident."""

    timestamp: datetime
    event_type: str
    description: str
    actor: str  # User or system that performed action
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IncidentCase:
    """Security incident case."""

    case_id: str
    title: str
    description: str
    type: IncidentType
    severity: IncidentSeverity
    status: IncidentStatus

    # Timeline and tracking
    created_at: datetime
    updated_at: datetime
    detected_at: Optional[datetime] = None
    contained_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None

    # Assignment and ownership
    assigned_to: Optional[str] = None
    team: Optional[str] = None
    escalation_level: int = 0

    # Indicators and evidence
    indicators: List[IncidentIndicator] = field(default_factory=list)
    affected_assets: List[str] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    # Response tracking
    playbooks_executed: List[str] = field(default_factory=list)
    timeline: List[IncidentTimeline] = field(default_factory=list)
    notes: List[Dict[str, Any]] = field(default_factory=list)

    # Metrics
    mean_time_to_detect: Optional[int] = None  # seconds
    mean_time_to_respond: Optional[int] = None  # seconds
    mean_time_to_contain: Optional[int] = None  # seconds

    # Related incidents
    related_cases: List[str] = field(default_factory=list)
    parent_case: Optional[str] = None

    # Custom fields
    custom_fields: Dict[str, Any] = field(default_factory=dict)


class IncidentManager:
    """
    Comprehensive incident management system.
    Handles case creation, tracking, and automated response coordination.
    """

    def __init__(
        self,
        playbook_engine: Optional[PlaybookEngine] = None,
        data_dir: str = "./incident_data",
        auto_triage: bool = True,
        auto_playbook_execution: bool = True,
    ):
        self.playbook_engine = playbook_engine
        self.data_dir = data_dir
        self.auto_triage = auto_triage
        self.auto_playbook_execution = auto_playbook_execution

        # Case storage
        self.cases: Dict[str, IncidentCase] = {}
        self.case_lock = threading.Lock()

        # Indicator tracking
        self.global_indicators: Dict[str, Set[str]] = defaultdict(set)
        self.indicator_lock = threading.Lock()

        # Playbook mappings
        self.incident_playbook_mapping = {
            IncidentType.MALWARE: ["malware_response", "isolate_and_analyze"],
            IncidentType.UNAUTHORIZED_ACCESS: [
                "access_violation_response",
                "credential_reset",
            ],
            IncidentType.DATA_BREACH: [
                "data_breach_response",
                "breach_notification",
            ],
            IncidentType.DENIAL_OF_SERVICE: [
                "ddos_mitigation",
                "traffic_analysis",
            ],
            IncidentType.PHISHING: ["phishing_response", "user_awareness"],
            IncidentType.INSIDER_THREAT: [
                "insider_threat_response",
                "access_review",
            ],
            IncidentType.ZERO_DAY: ["zero_day_response", "emergency_patching"],
        }

        # Severity thresholds
        self.severity_thresholds = {
            "critical": {
                "response_time": 15,
                "escalation_time": 30,
            },  # minutes
            "high": {"response_time": 60, "escalation_time": 120},
            "medium": {"response_time": 240, "escalation_time": 480},
            "low": {"response_time": 1440, "escalation_time": 2880},
        }

        # Load existing cases
        self._load_cases()

        # Start background tasks
        self._start_background_tasks()

    def _load_cases(self):
        """Load cases from disk."""
        os.makedirs(self.data_dir, exist_ok=True)
        cases_file = os.path.join(self.data_dir, "cases.json")

        if os.path.exists(cases_file):
            try:
                with open(cases_file, "r") as f:
                    cases_data = json.load(f)
                    for case_data in cases_data:
                        # Convert string dates back to datetime
                        for date_field in [
                            "created_at",
                            "updated_at",
                            "detected_at",
                            "contained_at",
                            "resolved_at",
                        ]:
                            if case_data.get(date_field):
                                case_data[date_field] = datetime.fromisoformat(
                                    case_data[date_field]
                                )

                        # Reconstruct case
                        case = IncidentCase(**case_data)
                        self.cases[case.case_id] = case

                logger.info(f"Loaded {len(self.cases)} incident cases")
            except Exception as e:
                logger.error(f"Failed to load cases: {e}")

    def _save_cases(self):
        """Save cases to disk."""
        cases_file = os.path.join(self.data_dir, "cases.json")

        try:
            cases_data = []
            for case in self.cases.values():
                case_dict = case.__dict__.copy()
                # Convert datetime to string
                for date_field in [
                    "created_at",
                    "updated_at",
                    "detected_at",
                    "contained_at",
                    "resolved_at",
                ]:
                    if case_dict.get(date_field):
                        case_dict[date_field] = case_dict[date_field].isoformat()

                cases_data.append(case_dict)

            with open(cases_file, "w") as f:
                json.dump(cases_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save cases: {e}")

    def _start_background_tasks(self):
        """Start background monitoring tasks."""
        # Start escalation monitor
        threading.Thread(target=self._escalation_monitor, daemon=True).start()

        # Start metrics calculator
        threading.Thread(target=self._metrics_calculator, daemon=True).start()

    def create_incident(
        self,
        title: str,
        description: str,
        type: IncidentType,
        severity: IncidentSeverity,
        indicators: Optional[List[Dict[str, Any]]] = None,
        affected_assets: Optional[List[str]] = None,
        detected_at: Optional[datetime] = None,
        custom_fields: Optional[Dict[str, Any]] = None,
    ) -> IncidentCase:
        """
        Create a new incident case.
        """
        case_id = f"INC-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"

        # Create case
        case = IncidentCase(
            case_id=case_id,
            title=title,
            description=description,
            type=type,
            severity=severity,
            status=IncidentStatus.NEW,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            detected_at=detected_at or datetime.utcnow(),
            affected_assets=affected_assets or [],
            custom_fields=custom_fields or {},
        )

        # Add indicators
        if indicators:
            for indicator_data in indicators:
                indicator = IncidentIndicator(
                    type=indicator_data["type"],
                    value=indicator_data["value"],
                    confidence=indicator_data.get("confidence", 0.8),
                    source=indicator_data.get("source", "manual"),
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    metadata=indicator_data.get("metadata", {}),
                )
                case.indicators.append(indicator)

                # Track globally
                with self.indicator_lock:
                    self.global_indicators[indicator.type].add(indicator.value)

        # Add initial timeline entry
        case.timeline.append(
            IncidentTimeline(
                timestamp=datetime.utcnow(),
                event_type="case_created",
                description=f"Incident case {case_id} created",
                actor="system",
                data={"severity": severity.value, "type": type.value},
            )
        )

        # Store case
        with self.case_lock:
            self.cases[case_id] = case
            self._save_cases()

        # Auto-triage if enabled
        if self.auto_triage:
            asyncio.create_task(self._auto_triage(case))

        # Execute playbooks if enabled
        if self.auto_playbook_execution and self.playbook_engine:
            asyncio.create_task(self._execute_incident_playbooks(case))

        logger.info(f"Created incident case: {case_id}")
        return case

    async def _auto_triage(self, case: IncidentCase):
        """Automatically triage incident based on indicators and patterns."""
        try:
            # Check for related incidents
            related_cases = self._find_related_incidents(case)
            if related_cases:
                case.related_cases = [c.case_id for c in related_cases]

                # If many related cases, might indicate campaign
                if len(related_cases) >= 3:
                    case.severity = IncidentSeverity.CRITICAL
                    self._add_timeline_entry(
                        case,
                        "auto_triage",
                        f"Severity escalated to CRITICAL due to {len(related_cases)} related incidents",
                        "system",
                    )

            # Assign based on type and severity
            case.team = self._determine_team(case.type, case.severity)
            case.assigned_to = self._get_on_call_analyst(case.team)

            # Update status
            case.status = IncidentStatus.TRIAGED
            case.updated_at = datetime.utcnow()

            self._add_timeline_entry(
                case,
                "auto_triage",
                f"Case triaged and assigned to {case.assigned_to} ({case.team})",
                "system",
            )

            # Save changes
            with self.case_lock:
                self._save_cases()

        except Exception as e:
            logger.error(f"Auto-triage failed for {case.case_id}: {e}")

    async def _execute_incident_playbooks(self, case: IncidentCase):
        """Execute relevant playbooks for incident."""
        if not self.playbook_engine:
            return

        try:
            # Get playbooks for incident type
            playbook_ids = self.incident_playbook_mapping.get(case.type, [])

            for playbook_id in playbook_ids:
                # Prepare variables for playbook
                variables = {
                    "case_id": case.case_id,
                    "incident_type": case.type.value,
                    "severity": case.severity.value,
                    "affected_assets": case.affected_assets,
                    "indicators": [{"type": i.type, "value": i.value} for i in case.indicators],
                }

                # Add specific variables based on indicators
                for indicator in case.indicators:
                    if indicator.type == "ip":
                        variables["attacker_ip"] = indicator.value
                    elif indicator.type == "user":
                        variables["target_user"] = indicator.value
                    elif indicator.type == "host":
                        variables["affected_server"] = indicator.value

                # Execute playbook
                logger.info(f"Executing playbook {playbook_id} for case {case.case_id}")
                context = await self.playbook_engine.execute_playbook(
                    playbook_id=playbook_id,
                    trigger=PlaybookTrigger.ALERT,
                    trigger_data={"case_id": case.case_id},
                    variables=variables,
                )

                # Track execution
                case.playbooks_executed.append(playbook_id)
                self._add_timeline_entry(
                    case,
                    "playbook_executed",
                    f"Executed playbook {playbook_id} (status: {context.status.value})",
                    "system",
                    {"execution_id": context.execution_id},
                )

        except Exception as e:
            logger.error(f"Failed to execute playbooks for {case.case_id}: {e}")

    def update_incident_status(
        self,
        case_id: str,
        status: IncidentStatus,
        notes: Optional[str] = None,
        actor: str = "analyst",
    ) -> bool:
        """Update incident status."""
        with self.case_lock:
            if case_id not in self.cases:
                return False

            case = self.cases[case_id]
            old_status = case.status
            case.status = status
            case.updated_at = datetime.utcnow()

            # Update timestamps
            if status == IncidentStatus.CONTAINED and not case.contained_at:
                case.contained_at = datetime.utcnow()
            elif status == IncidentStatus.CLOSED and not case.resolved_at:
                case.resolved_at = datetime.utcnow()

            # Add timeline entry
            self._add_timeline_entry(
                case,
                "status_change",
                f"Status changed from {old_status.value} to {status.value}",
                actor,
                {"old_status": old_status.value, "new_status": status.value},
            )

            # Add notes if provided
            if notes:
                self.add_incident_notes(case_id, notes, actor)

            self._save_cases()
            return True

    def add_incident_notes(self, case_id: str, notes: str, actor: str = "analyst") -> bool:
        """Add notes to incident."""
        with self.case_lock:
            if case_id not in self.cases:
                return False

            case = self.cases[case_id]
            case.notes.append(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "actor": actor,
                    "content": notes,
                }
            )
            case.updated_at = datetime.utcnow()

            self._add_timeline_entry(case, "notes_added", "Added investigation notes", actor)

            self._save_cases()
            return True

    def add_indicators(self, case_id: str, indicators: List[Dict[str, Any]]) -> bool:
        """Add indicators to incident."""
        with self.case_lock:
            if case_id not in self.cases:
                return False

            case = self.cases[case_id]

            for indicator_data in indicators:
                indicator = IncidentIndicator(
                    type=indicator_data["type"],
                    value=indicator_data["value"],
                    confidence=indicator_data.get("confidence", 0.8),
                    source=indicator_data.get("source", "manual"),
                    first_seen=datetime.utcnow(),
                    last_seen=datetime.utcnow(),
                    metadata=indicator_data.get("metadata", {}),
                )
                case.indicators.append(indicator)

                # Track globally
                with self.indicator_lock:
                    self.global_indicators[indicator.type].add(indicator.value)

            case.updated_at = datetime.utcnow()

            self._add_timeline_entry(
                case,
                "indicators_added",
                f"Added {len(indicators)} new indicators",
                "system",
            )

            self._save_cases()
            return True

    def _add_timeline_entry(
        self,
        case: IncidentCase,
        event_type: str,
        description: str,
        actor: str,
        data: Optional[Dict[str, Any]] = None,
    ):
        """Add entry to case timeline."""
        case.timeline.append(
            IncidentTimeline(
                timestamp=datetime.utcnow(),
                event_type=event_type,
                description=description,
                actor=actor,
                data=data or {},
            )
        )

    def _find_related_incidents(
        self, case: IncidentCase, time_window_hours: int = 24
    ) -> List[IncidentCase]:
        """Find incidents related by indicators or patterns."""
        related = []
        cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)

        with self.case_lock:
            for other_case in self.cases.values():
                if other_case.case_id == case.case_id:
                    continue

                if other_case.created_at < cutoff_time:
                    continue

                # Check for matching indicators
                case_indicators = {(i.type, i.value) for i in case.indicators}
                other_indicators = {(i.type, i.value) for i in other_case.indicators}

                if case_indicators & other_indicators:
                    related.append(other_case)
                    continue

                # Check for same affected assets
                if set(case.affected_assets) & set(other_case.affected_assets):
                    related.append(other_case)

        return related

    def _determine_team(self, incident_type: IncidentType, severity: IncidentSeverity) -> str:
        """Determine which team should handle incident."""
        # Simple mapping - extend based on organization
        team_mapping = {
            IncidentType.MALWARE: "malware-analysis",
            IncidentType.UNAUTHORIZED_ACCESS: "access-control",
            IncidentType.DATA_BREACH: "data-protection",
            IncidentType.DENIAL_OF_SERVICE: "network-security",
            IncidentType.PHISHING: "email-security",
            IncidentType.INSIDER_THREAT: "insider-risk",
            IncidentType.ZERO_DAY: "vulnerability-management",
        }

        base_team = team_mapping.get(incident_type, "general-security")

        # Escalate to senior team for critical
        if severity == IncidentSeverity.CRITICAL:
            return f"senior-{base_team}"

        return base_team

    def _get_on_call_analyst(self, team: str) -> str:
        """Get on-call analyst for team."""
        # Simplified - in production, integrate with on-call system
        return f"analyst-{team}-oncall"

    def _escalation_monitor(self):
        """Monitor incidents for escalation needs."""
        while True:
            try:
                with self.case_lock:
                    for case in self.cases.values():
                        if case.status == IncidentStatus.CLOSED:
                            continue

                        # Check response time SLA
                        threshold = self.severity_thresholds.get(
                            case.severity.value, {"response_time": 1440}
                        )

                        time_since_creation = (
                            datetime.utcnow() - case.created_at
                        ).total_seconds() / 60

                        # Escalate if no response
                        if (
                            case.status == IncidentStatus.NEW
                            and time_since_creation > threshold["response_time"]
                            and case.escalation_level == 0
                        ):
                            case.escalation_level = 1
                            self._add_timeline_entry(
                                case,
                                "escalation",
                                f'Escalated due to no response after {threshold["response_time"]} minutes',
                                "system",
                            )
                            logger.warning(f"Escalated case {case.case_id} due to SLA breach")

                # Sleep for a minute
                time.sleep(60)

            except Exception as e:
                logger.error(f"Escalation monitor error: {e}")
                time.sleep(60)

    def _metrics_calculator(self):
        """Calculate incident metrics."""
        while True:
            try:
                with self.case_lock:
                    for case in self.cases.values():
                        # Mean time to detect
                        if case.detected_at and case.created_at:
                            case.mean_time_to_detect = int(
                                (case.detected_at - case.created_at).total_seconds()
                            )

                        # Mean time to respond
                        if case.status != IncidentStatus.NEW and case.created_at:
                            first_response = None
                            for entry in case.timeline:
                                if entry.event_type in [
                                    "status_change",
                                    "notes_added",
                                ]:
                                    first_response = entry.timestamp
                                    break

                            if first_response:
                                case.mean_time_to_respond = int(
                                    (first_response - case.created_at).total_seconds()
                                )

                        # Mean time to contain
                        if case.contained_at and case.created_at:
                            case.mean_time_to_contain = int(
                                (case.contained_at - case.created_at).total_seconds()
                            )

                # Sleep for 5 minutes
                time.sleep(300)

            except Exception as e:
                logger.error(f"Metrics calculator error: {e}")
                time.sleep(300)

    def get_incident_summary(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive incident summary."""
        with self.case_lock:
            if case_id not in self.cases:
                return None

            case = self.cases[case_id]

            return {
                "case_id": case.case_id,
                "title": case.title,
                "type": case.type.value,
                "severity": case.severity.value,
                "status": case.status.value,
                "assigned_to": case.assigned_to,
                "team": case.team,
                "created_at": case.created_at.isoformat(),
                "updated_at": case.updated_at.isoformat(),
                "indicators_count": len(case.indicators),
                "affected_assets_count": len(case.affected_assets),
                "playbooks_executed": case.playbooks_executed,
                "timeline_entries": len(case.timeline),
                "notes_count": len(case.notes),
                "metrics": {
                    "mttr": case.mean_time_to_respond,
                    "mttc": case.mean_time_to_contain,
                    "mttd": case.mean_time_to_detect,
                },
                "related_cases": case.related_cases,
            }

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get metrics for security dashboard."""
        with self.case_lock:
            total_cases = len(self.cases)

            # Status breakdown
            status_counts = defaultdict(int)
            for case in self.cases.values():
                status_counts[case.status.value] += 1

            # Severity breakdown
            severity_counts = defaultdict(int)
            for case in self.cases.values():
                severity_counts[case.severity.value] += 1

            # Type breakdown
            type_counts = defaultdict(int)
            for case in self.cases.values():
                type_counts[case.type.value] += 1

            # Calculate averages
            response_times = [
                case.mean_time_to_respond
                for case in self.cases.values()
                if case.mean_time_to_respond
            ]

            contain_times = [
                case.mean_time_to_contain
                for case in self.cases.values()
                if case.mean_time_to_contain
            ]

            return {
                "total_incidents": total_cases,
                "open_incidents": sum(
                    1 for case in self.cases.values() if case.status != IncidentStatus.CLOSED
                ),
                "status_breakdown": dict(status_counts),
                "severity_breakdown": dict(severity_counts),
                "type_breakdown": dict(type_counts),
                "average_response_time": (
                    sum(response_times) / len(response_times) if response_times else 0
                ),
                "average_containment_time": (
                    sum(contain_times) / len(contain_times) if contain_times else 0
                ),
                "indicators_tracked": sum(len(v) for v in self.global_indicators.values()),
            }
