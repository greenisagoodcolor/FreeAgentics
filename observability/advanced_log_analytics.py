"""Advanced Log Analytics and Anomaly Detection for FreeAgentics.

This module provides:
- Intelligent log parsing and classification
- ML-based anomaly detection
- Real-time log streaming and analysis
- Correlation analysis across different log sources
- Predictive alerting based on log patterns
- Log-based performance insights
- Security event detection from logs
"""

import asyncio
import json
import logging
import re
import sqlite3
import statistics
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log level enumeration."""

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogCategory(Enum):
    """Log category enumeration."""

    APPLICATION = "application"
    SYSTEM = "system"
    SECURITY = "security"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    AUDIT = "audit"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class LogEntry:
    """Structured log entry representation."""

    timestamp: float
    level: LogLevel
    category: LogCategory
    component: str
    message: str
    raw_message: str
    tags: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None
    duration_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "level": self.level.value,
            "category": self.category.value,
            "component": self.component,
            "message": self.message,
            "raw_message": self.raw_message,
            "tags": self.tags,
            "metrics": self.metrics,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "error_type": self.error_type,
            "stack_trace": self.stack_trace,
            "duration_ms": self.duration_ms,
        }


@dataclass
class LogPattern:
    """Log pattern for classification."""

    name: str
    pattern: Pattern[str]
    category: LogCategory
    level: LogLevel
    extractor: Optional[Callable[[str], Dict[str, Any]]] = None
    priority: int = 0

    def matches(self, message: str) -> bool:
        """Check if pattern matches message."""
        return bool(self.pattern.search(message))

    def extract(self, message: str) -> Dict[str, Any]:
        """Extract data from message using pattern."""
        match = self.pattern.search(message)
        if not match:
            return {}

        data = match.groupdict()

        # Apply custom extractor if available
        if self.extractor:
            try:
                custom_data = self.extractor(message)
                data.update(custom_data)
            except Exception as e:
                logger.warning(f"Custom extractor failed for pattern {self.name}: {e}")

        return data


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection analysis."""

    timestamp: float
    anomaly_type: str
    severity: str
    component: str
    description: str
    affected_logs: List[LogEntry]
    anomaly_score: float
    baseline_value: Optional[float] = None
    actual_value: Optional[float] = None
    threshold: Optional[float] = None
    recommendation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "component": self.component,
            "description": self.description,
            "anomaly_score": self.anomaly_score,
            "baseline_value": self.baseline_value,
            "actual_value": self.actual_value,
            "threshold": self.threshold,
            "recommendation": self.recommendation,
            "affected_log_count": len(self.affected_logs),
        }


class LogPatternMatcher:
    """Advanced log pattern matching and classification."""

    def __init__(self):
        """Initialize pattern matcher with built-in patterns."""
        self.patterns: List[LogPattern] = []
        self._load_builtin_patterns()

    def _load_builtin_patterns(self):
        """Load built-in log patterns."""

        # Agent coordination patterns
        self.add_pattern(
            "agent_coordination_start",
            r"Starting coordination between agents? (?P<agents>[\w\-,\s]+)",
            LogCategory.APPLICATION,
            LogLevel.INFO,
        )

        self.add_pattern(
            "agent_coordination_complete",
            r"Coordination completed.*duration[:\s]+(?P<duration>\d+(?:\.\d+)?)\s*ms",
            LogCategory.PERFORMANCE,
            LogLevel.INFO,
            extractor=lambda msg: {
                "duration_ms": float(re.search(r"(\d+(?:\.\d+)?)\s*ms", msg).group(1))
            },
        )

        self.add_pattern(
            "agent_memory_high",
            r"Agent (?P<agent_id>\w+) memory usage: (?P<memory>\d+(?:\.\d+)?)\s*(?P<unit>MB|GB)",
            LogCategory.PERFORMANCE,
            LogLevel.WARNING,
            extractor=lambda msg: {"memory_mb": self._parse_memory(msg)},
        )

        # Error patterns
        self.add_pattern(
            "python_exception",
            r"(?P<error_type>\w+Error): (?P<error_message>.*)",
            LogCategory.ERROR,
            LogLevel.ERROR,
            extractor=lambda msg: {"error_type": re.search(r"(\w+Error):", msg).group(1)},
        )

        self.add_pattern(
            "database_connection_error",
            r"Database connection (?:failed|lost|timeout).*(?P<details>.*)",
            LogCategory.SYSTEM,
            LogLevel.CRITICAL,
        )

        # Performance patterns
        self.add_pattern(
            "slow_query",
            r"Slow query detected.*duration[:\s]+(?P<duration>\d+(?:\.\d+)?)\s*ms.*query[:\s]+(?P<query>.*)",
            LogCategory.PERFORMANCE,
            LogLevel.WARNING,
        )

        self.add_pattern(
            "high_cpu_usage",
            r"CPU usage: (?P<cpu_percent>\d+(?:\.\d+)?)%",
            LogCategory.PERFORMANCE,
            (
                LogLevel.WARNING
                if "90" in r"CPU usage: (?P<cpu_percent>\d+(?:\.\d+)?)%"
                else LogLevel.INFO
            ),
        )

        # Security patterns
        self.add_pattern(
            "failed_authentication",
            r"Authentication failed.*user[:\s]+(?P<user>\w+).*(?:ip|address)[:\s]+(?P<ip>\d+\.\d+\.\d+\.\d+)",
            LogCategory.SECURITY,
            LogLevel.WARNING,
        )

        self.add_pattern(
            "suspicious_activity",
            r"Suspicious activity detected.*(?P<details>.*)",
            LogCategory.SECURITY,
            LogLevel.CRITICAL,
        )

        # Business patterns
        self.add_pattern(
            "user_interaction",
            r"User (?P<user_id>\w+) performed (?P<action>\w+).*(?P<details>.*)",
            LogCategory.BUSINESS,
            LogLevel.INFO,
        )

        # HTTP request patterns
        self.add_pattern(
            "http_request",
            r"(?P<method>GET|POST|PUT|DELETE|PATCH)\s+(?P<path>/[^\s]*)\s+(?P<status>\d{3})\s+(?P<duration>\d+(?:\.\d+)?)\s*ms",
            LogCategory.APPLICATION,
            LogLevel.INFO,
        )

    def add_pattern(
        self,
        name: str,
        pattern: str,
        category: LogCategory,
        level: LogLevel,
        extractor: Optional[Callable[[str], Dict[str, Any]]] = None,
        priority: int = 0,
    ):
        """Add a new log pattern."""
        compiled_pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        log_pattern = LogPattern(
            name=name,
            pattern=compiled_pattern,
            category=category,
            level=level,
            extractor=extractor,
            priority=priority,
        )
        self.patterns.append(log_pattern)

        # Sort by priority (higher priority first)
        self.patterns.sort(key=lambda p: p.priority, reverse=True)

    def classify_log(
        self, raw_message: str, default_level: LogLevel = LogLevel.INFO
    ) -> Tuple[LogCategory, LogLevel, Dict[str, Any]]:
        """Classify log message and extract structured data."""
        for pattern in self.patterns:
            if pattern.matches(raw_message):
                extracted_data = pattern.extract(raw_message)
                return pattern.category, pattern.level, extracted_data

        # Default classification
        return LogCategory.UNKNOWN, default_level, {}

    def _parse_memory(self, message: str) -> float:
        """Parse memory value and convert to MB."""
        match = re.search(r"(\d+(?:\.\d+)?)\s*(MB|GB)", message, re.IGNORECASE)
        if not match:
            return 0.0

        value, unit = match.groups()
        value = float(value)

        if unit.upper() == "GB":
            value *= 1024

        return value


class AnomalyDetector:
    """ML-based anomaly detection for log data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize anomaly detector."""
        self.config = config or {}
        self.window_size = self.config.get("window_size", 100)
        self.z_score_threshold = self.config.get("z_score_threshold", 3.0)
        self.iqr_factor = self.config.get("iqr_factor", 1.5)
        self.min_samples = self.config.get("min_samples", 10)

        # Historical data for baseline calculation
        self.error_rate_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self.response_time_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self.log_volume_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )
        self.component_health_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    def detect_anomalies(
        self, logs: List[LogEntry], time_window_minutes: int = 5
    ) -> List[AnomalyDetectionResult]:
        """Detect anomalies in log data."""
        if not logs:
            return []

        anomalies = []
        current_time = max(log.timestamp for log in logs)
        window_start = current_time - (time_window_minutes * 60)

        # Filter logs to time window
        window_logs = [log for log in logs if log.timestamp >= window_start]

        # Group logs by component
        logs_by_component = defaultdict(list)
        for log in window_logs:
            logs_by_component[log.component].append(log)

        # Detect anomalies for each component
        for component, component_logs in logs_by_component.items():
            anomalies.extend(self._detect_error_rate_anomalies(component, component_logs))
            anomalies.extend(self._detect_response_time_anomalies(component, component_logs))
            anomalies.extend(self._detect_log_volume_anomalies(component, component_logs))
            anomalies.extend(self._detect_pattern_anomalies(component, component_logs))

        # Detect cross-component anomalies
        anomalies.extend(self._detect_correlation_anomalies(logs_by_component))

        return sorted(anomalies, key=lambda a: a.anomaly_score, reverse=True)

    def _detect_error_rate_anomalies(
        self, component: str, logs: List[LogEntry]
    ) -> List[AnomalyDetectionResult]:
        """Detect error rate anomalies."""
        if len(logs) < self.min_samples:
            return []

        error_logs = [log for log in logs if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]]
        current_error_rate = (len(error_logs) / len(logs)) * 100

        # Update history
        self.error_rate_history[component].append(current_error_rate)

        if len(self.error_rate_history[component]) < self.min_samples:
            return []

        # Calculate baseline
        historical_rates = list(self.error_rate_history[component])[:-1]  # Exclude current
        if not historical_rates:
            return []

        baseline_mean = statistics.mean(historical_rates)
        baseline_std = statistics.stdev(historical_rates) if len(historical_rates) > 1 else 0

        # Z-score anomaly detection
        if baseline_std > 0:
            z_score = abs(current_error_rate - baseline_mean) / baseline_std

            if z_score > self.z_score_threshold:
                return [
                    AnomalyDetectionResult(
                        timestamp=max(log.timestamp for log in logs),
                        anomaly_type="error_rate_spike",
                        severity="high" if z_score > 4 else "medium",
                        component=component,
                        description=f"Error rate spike detected: {current_error_rate:.1f}% vs baseline {baseline_mean:.1f}%",
                        affected_logs=error_logs,
                        anomaly_score=z_score,
                        baseline_value=baseline_mean,
                        actual_value=current_error_rate,
                        threshold=baseline_mean + (self.z_score_threshold * baseline_std),
                        recommendation="Investigate recent changes, check error logs, verify external dependencies",
                    )
                ]

        return []

    def _detect_response_time_anomalies(
        self, component: str, logs: List[LogEntry]
    ) -> List[AnomalyDetectionResult]:
        """Detect response time anomalies."""
        duration_logs = [log for log in logs if log.duration_ms is not None and log.duration_ms > 0]

        if len(duration_logs) < self.min_samples:
            return []

        current_durations = [log.duration_ms for log in duration_logs]
        current_p95 = np.percentile(current_durations, 95)

        # Update history
        self.response_time_history[component].append(current_p95)

        if len(self.response_time_history[component]) < self.min_samples:
            return []

        # Calculate baseline
        historical_p95s = list(self.response_time_history[component])[:-1]
        if not historical_p95s:
            return []

        baseline_mean = statistics.mean(historical_p95s)
        statistics.stdev(historical_p95s) if len(historical_p95s) > 1 else 0

        # IQR-based anomaly detection for response times
        q1 = np.percentile(historical_p95s, 25)
        q3 = np.percentile(historical_p95s, 75)
        iqr = q3 - q1
        upper_bound = q3 + (self.iqr_factor * iqr)

        if current_p95 > upper_bound and iqr > 0:
            anomaly_score = (current_p95 - upper_bound) / iqr

            return [
                AnomalyDetectionResult(
                    timestamp=max(log.timestamp for log in duration_logs),
                    anomaly_type="response_time_degradation",
                    severity="high" if anomaly_score > 2 else "medium",
                    component=component,
                    description=f"Response time degradation: P95 {current_p95:.1f}ms vs baseline {baseline_mean:.1f}ms",
                    affected_logs=duration_logs,
                    anomaly_score=anomaly_score,
                    baseline_value=baseline_mean,
                    actual_value=current_p95,
                    threshold=upper_bound,
                    recommendation="Check for performance bottlenecks, review recent deployments, monitor resource usage",
                )
            ]

        return []

    def _detect_log_volume_anomalies(
        self, component: str, logs: List[LogEntry]
    ) -> List[AnomalyDetectionResult]:
        """Detect log volume anomalies."""
        current_volume = len(logs)

        # Update history
        self.log_volume_history[component].append(current_volume)

        if len(self.log_volume_history[component]) < self.min_samples:
            return []

        # Calculate baseline
        historical_volumes = list(self.log_volume_history[component])[:-1]
        if not historical_volumes:
            return []

        baseline_mean = statistics.mean(historical_volumes)
        baseline_std = statistics.stdev(historical_volumes) if len(historical_volumes) > 1 else 0

        if baseline_std > 0:
            z_score = abs(current_volume - baseline_mean) / baseline_std

            if z_score > self.z_score_threshold:
                anomaly_type = (
                    "log_volume_spike" if current_volume > baseline_mean else "log_volume_drop"
                )
                severity = "high" if z_score > 4 else "medium"

                return [
                    AnomalyDetectionResult(
                        timestamp=max(log.timestamp for log in logs) if logs else 0,
                        anomaly_type=anomaly_type,
                        severity=severity,
                        component=component,
                        description=f"Log volume anomaly: {current_volume} logs vs baseline {baseline_mean:.1f}",
                        affected_logs=logs,
                        anomaly_score=z_score,
                        baseline_value=baseline_mean,
                        actual_value=current_volume,
                        threshold=baseline_mean + (self.z_score_threshold * baseline_std),
                        recommendation="Check for application issues or configuration changes",
                    )
                ]

        return []

    def _detect_pattern_anomalies(
        self, component: str, logs: List[LogEntry]
    ) -> List[AnomalyDetectionResult]:
        """Detect pattern-based anomalies."""
        anomalies = []

        # Look for unusual error patterns
        error_logs = [log for log in logs if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]]
        if error_logs:
            error_types = [log.error_type for log in error_logs if log.error_type]
            error_counter = Counter(error_types)

            # Check for new error types
            for error_type, count in error_counter.items():
                if error_type and count >= 3:  # At least 3 occurrences
                    anomalies.append(
                        AnomalyDetectionResult(
                            timestamp=max(log.timestamp for log in error_logs),
                            anomaly_type="new_error_pattern",
                            severity="medium",
                            component=component,
                            description=f"Recurring error pattern: {error_type} ({count} occurrences)",
                            affected_logs=[
                                log for log in error_logs if log.error_type == error_type
                            ],
                            anomaly_score=count,
                            actual_value=count,
                            recommendation="Investigate root cause of recurring errors",
                        )
                    )

        return anomalies

    def _detect_correlation_anomalies(
        self, logs_by_component: Dict[str, List[LogEntry]]
    ) -> List[AnomalyDetectionResult]:
        """Detect cross-component correlation anomalies."""
        if len(logs_by_component) < 2:
            return []

        anomalies = []

        # Look for components with simultaneous error spikes
        components_with_errors = {}
        for component, logs in logs_by_component.items():
            error_count = len(
                [log for log in logs if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]]
            )
            if error_count > 0:
                error_rate = (error_count / len(logs)) * 100
                components_with_errors[component] = error_rate

        # If multiple components have high error rates, it might indicate a systemic issue
        high_error_components = [
            comp
            for comp, rate in components_with_errors.items()
            if rate > 10  # More than 10% error rate
        ]

        if len(high_error_components) >= 2:
            total_affected_logs = []
            for comp in high_error_components:
                total_affected_logs.extend(
                    [
                        log
                        for log in logs_by_component[comp]
                        if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]
                    ]
                )

            anomalies.append(
                AnomalyDetectionResult(
                    timestamp=max(log.timestamp for log in total_affected_logs),
                    anomaly_type="systemic_failure",
                    severity="critical",
                    component="system",
                    description=f"Systemic failure detected across {len(high_error_components)} components",
                    affected_logs=total_affected_logs,
                    anomaly_score=len(high_error_components),
                    actual_value=len(high_error_components),
                    recommendation="Check shared dependencies, infrastructure issues, or recent system-wide changes",
                )
            )

        return anomalies


class AdvancedLogAnalytics:
    """Advanced log analytics with ML-based anomaly detection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize advanced log analytics."""
        self.config = config or {}
        self.db_path = self.config.get("db_path", "/tmp/advanced_log_analytics.db")
        self.buffer_size = self.config.get("buffer_size", 10000)
        self.analysis_interval = self.config.get("analysis_interval", 60)  # seconds

        # Components
        self.pattern_matcher = LogPatternMatcher()
        self.anomaly_detector = AnomalyDetector(self.config.get("anomaly_detection", {}))

        # In-memory log buffer for real-time analysis
        self.log_buffer: deque = deque(maxlen=self.buffer_size)
        self.processed_count = 0

        # Analysis tasks
        self.analysis_task: Optional[asyncio.Task] = None
        self.running = False

        # Initialize database
        self._init_database()

        logger.info("Advanced log analytics system initialized")

    def _init_database(self):
        """Initialize SQLite database for log storage."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create logs table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    level TEXT NOT NULL,
                    category TEXT NOT NULL,
                    component TEXT NOT NULL,
                    message TEXT NOT NULL,
                    raw_message TEXT NOT NULL,
                    tags TEXT,
                    metrics TEXT,
                    trace_id TEXT,
                    span_id TEXT,
                    agent_id TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    request_id TEXT,
                    error_type TEXT,
                    stack_trace TEXT,
                    duration_ms REAL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create anomalies table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    anomaly_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    component TEXT NOT NULL,
                    description TEXT NOT NULL,
                    anomaly_score REAL NOT NULL,
                    baseline_value REAL,
                    actual_value REAL,
                    threshold REAL,
                    recommendation TEXT,
                    affected_log_count INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_component ON logs(component)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_category ON logs(category)")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_anomalies_timestamp ON anomalies(timestamp)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_anomalies_component ON anomalies(component)"
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    async def start(self):
        """Start log analytics system."""
        if self.running:
            return

        self.running = True

        # Start analysis task
        self.analysis_task = asyncio.create_task(self._run_periodic_analysis())

        logger.info("Advanced log analytics system started")

    async def stop(self):
        """Stop log analytics system."""
        if not self.running:
            return

        self.running = False

        if self.analysis_task:
            self.analysis_task.cancel()
            try:
                await self.analysis_task
            except asyncio.CancelledError:
                pass

        logger.info("Advanced log analytics system stopped")

    def ingest_log(
        self,
        raw_message: str,
        level: str = "INFO",
        component: str = "unknown",
        timestamp: Optional[float] = None,
        **kwargs,
    ) -> LogEntry:
        """Ingest and process a raw log message."""
        timestamp = timestamp or time.time()

        # Parse log level
        try:
            log_level = LogLevel(level.upper())
        except ValueError:
            log_level = LogLevel.INFO

        # Classify log using pattern matcher
        category, detected_level, extracted_data = self.pattern_matcher.classify_log(
            raw_message, log_level
        )

        # Use detected level if it's more severe
        if detected_level.value in ["ERROR", "CRITICAL"] and log_level.value in [
            "DEBUG",
            "INFO",
            "WARNING",
        ]:
            log_level = detected_level

        # Create log entry
        log_entry = LogEntry(
            timestamp=timestamp,
            level=log_level,
            category=category,
            component=component,
            message=extracted_data.get("message", raw_message),
            raw_message=raw_message,
            tags=extracted_data.get("tags", {}),
            metrics=extracted_data.get("metrics", {}),
            trace_id=extracted_data.get("trace_id") or kwargs.get("trace_id"),
            span_id=extracted_data.get("span_id") or kwargs.get("span_id"),
            agent_id=extracted_data.get("agent_id") or kwargs.get("agent_id"),
            user_id=extracted_data.get("user_id") or kwargs.get("user_id"),
            session_id=extracted_data.get("session_id") or kwargs.get("session_id"),
            request_id=extracted_data.get("request_id") or kwargs.get("request_id"),
            error_type=extracted_data.get("error_type"),
            stack_trace=extracted_data.get("stack_trace"),
            duration_ms=extracted_data.get("duration_ms"),
        )

        # Add to buffer for real-time analysis
        self.log_buffer.append(log_entry)
        self.processed_count += 1

        # Store in database (async in background)
        asyncio.create_task(self._store_log_entry(log_entry))

        return log_entry

    async def analyze_logs(
        self, time_window_minutes: int = 15, components: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze recent logs for anomalies and insights."""
        current_time = time.time()
        window_start = current_time - (time_window_minutes * 60)

        # Get logs from buffer and database
        buffer_logs = [log for log in self.log_buffer if log.timestamp >= window_start]

        db_logs = await self._get_logs_from_db(window_start, components)
        all_logs = buffer_logs + db_logs

        # Remove duplicates (buffer might have logs also in DB)
        seen = set()
        unique_logs = []
        for log in all_logs:
            log_key = (log.timestamp, log.component, log.raw_message)
            if log_key not in seen:
                seen.add(log_key)
                unique_logs.append(log)

        # Detect anomalies
        anomalies = self.anomaly_detector.detect_anomalies(unique_logs, time_window_minutes)

        # Store anomalies
        for anomaly in anomalies:
            await self._store_anomaly(anomaly)

        # Generate analysis report
        analysis = {
            "timestamp": current_time,
            "time_window_minutes": time_window_minutes,
            "total_logs": len(unique_logs),
            "components": list(set(log.component for log in unique_logs)),
            "log_levels": dict(Counter(log.level.value for log in unique_logs)),
            "categories": dict(Counter(log.category.value for log in unique_logs)),
            "anomalies": [anomaly.to_dict() for anomaly in anomalies],
            "summary": self._generate_analysis_summary(unique_logs, anomalies),
        }

        return analysis

    def get_log_insights(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive log insights for the specified time period."""
        end_time = time.time()
        start_time = end_time - (hours * 3600)

        # This would typically query the database for historical analysis
        # For now, we'll provide insights based on recent buffer data
        recent_logs = [log for log in self.log_buffer if log.timestamp >= start_time]

        if not recent_logs:
            return {"error": "No logs available for the specified time period"}

        insights = {
            "time_period_hours": hours,
            "total_logs": len(recent_logs),
            "log_volume_trend": self._calculate_log_volume_trend(recent_logs),
            "error_analysis": self._analyze_error_patterns(recent_logs),
            "performance_insights": self._analyze_performance_patterns(recent_logs),
            "component_health": self._analyze_component_health(recent_logs),
            "top_issues": self._identify_top_issues(recent_logs),
        }

        return insights

    async def _run_periodic_analysis(self):
        """Run periodic log analysis."""
        while self.running:
            try:
                # Run analysis on recent logs
                analysis = await self.analyze_logs(15)  # 15-minute windows

                # Log summary if anomalies found
                if analysis["anomalies"]:
                    logger.warning(f"Log analysis found {len(analysis['anomalies'])} anomalies")
                    for anomaly in analysis["anomalies"][:3]:  # Log top 3
                        logger.warning(
                            f"Anomaly: {anomaly['description']} (score: {anomaly['anomaly_score']:.2f})"
                        )

            except Exception as e:
                logger.error(f"Error in periodic log analysis: {e}")

            await asyncio.sleep(self.analysis_interval)

    async def _store_log_entry(self, log_entry: LogEntry):
        """Store log entry in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO logs (
                    timestamp, level, category, component, message, raw_message,
                    tags, metrics, trace_id, span_id, agent_id, user_id,
                    session_id, request_id, error_type, stack_trace, duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    log_entry.timestamp,
                    log_entry.level.value,
                    log_entry.category.value,
                    log_entry.component,
                    log_entry.message,
                    log_entry.raw_message,
                    json.dumps(log_entry.tags) if log_entry.tags else None,
                    json.dumps(log_entry.metrics) if log_entry.metrics else None,
                    log_entry.trace_id,
                    log_entry.span_id,
                    log_entry.agent_id,
                    log_entry.user_id,
                    log_entry.session_id,
                    log_entry.request_id,
                    log_entry.error_type,
                    log_entry.stack_trace,
                    log_entry.duration_ms,
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to store log entry: {e}")

    async def _store_anomaly(self, anomaly: AnomalyDetectionResult):
        """Store anomaly in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO anomalies (
                    timestamp, anomaly_type, severity, component, description,
                    anomaly_score, baseline_value, actual_value, threshold,
                    recommendation, affected_log_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    anomaly.timestamp,
                    anomaly.anomaly_type,
                    anomaly.severity,
                    anomaly.component,
                    anomaly.description,
                    anomaly.anomaly_score,
                    anomaly.baseline_value,
                    anomaly.actual_value,
                    anomaly.threshold,
                    anomaly.recommendation,
                    len(anomaly.affected_logs),
                ),
            )

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Failed to store anomaly: {e}")

    async def _get_logs_from_db(
        self, start_time: float, components: Optional[List[str]] = None
    ) -> List[LogEntry]:
        """Get logs from database within time range."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            query = "SELECT * FROM logs WHERE timestamp >= ?"
            params = [start_time]

            if components:
                placeholders = ",".join("?" * len(components))
                query += f" AND component IN ({placeholders})"
                params.extend(components)

            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Convert rows to LogEntry objects
            logs = []
            for row in rows:
                logs.append(
                    LogEntry(
                        timestamp=row[1],
                        level=LogLevel(row[2]),
                        category=LogCategory(row[3]),
                        component=row[4],
                        message=row[5],
                        raw_message=row[6],
                        tags=json.loads(row[7]) if row[7] else {},
                        metrics=json.loads(row[8]) if row[8] else {},
                        trace_id=row[9],
                        span_id=row[10],
                        agent_id=row[11],
                        user_id=row[12],
                        session_id=row[13],
                        request_id=row[14],
                        error_type=row[15],
                        stack_trace=row[16],
                        duration_ms=row[17],
                    )
                )

            conn.close()
            return logs

        except Exception as e:
            logger.error(f"Failed to get logs from database: {e}")
            return []

    def _generate_analysis_summary(
        self, logs: List[LogEntry], anomalies: List[AnomalyDetectionResult]
    ) -> Dict[str, Any]:
        """Generate analysis summary."""
        if not logs:
            return {"status": "no_data"}

        error_logs = [log for log in logs if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]]
        error_rate = (len(error_logs) / len(logs)) * 100

        # Calculate health score
        health_score = 100
        if error_rate > 5:
            health_score -= (error_rate - 5) * 5
        if anomalies:
            health_score -= len(anomalies) * 10
        health_score = max(0, min(100, health_score))

        return {
            "status": (
                "healthy" if health_score > 80 else "degraded" if health_score > 50 else "critical"
            ),
            "health_score": health_score,
            "error_rate": error_rate,
            "anomaly_count": len(anomalies),
            "critical_anomalies": len([a for a in anomalies if a.severity == "critical"]),
            "most_active_component": (
                max(Counter(log.component for log in logs).items(), key=lambda x: x[1])[0]
                if logs
                else "unknown"
            ),
            "recommendations": self._generate_recommendations(logs, anomalies),
        }

    def _generate_recommendations(
        self, logs: List[LogEntry], anomalies: List[AnomalyDetectionResult]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []

        if anomalies:
            critical_anomalies = [a for a in anomalies if a.severity == "critical"]
            if critical_anomalies:
                recommendations.append("Immediate attention required: Critical anomalies detected")

            error_rate_anomalies = [a for a in anomalies if a.anomaly_type == "error_rate_spike"]
            if error_rate_anomalies:
                recommendations.append("Investigate recent deployments or configuration changes")

        error_logs = [log for log in logs if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]]
        if error_logs:
            error_rate = (len(error_logs) / len(logs)) * 100
            if error_rate > 10:
                recommendations.append("High error rate detected - review application health")

        if not recommendations:
            recommendations.append("System appears healthy - continue monitoring")

        return recommendations

    def _calculate_log_volume_trend(self, logs: List[LogEntry]) -> str:
        """Calculate log volume trend."""
        if len(logs) < 10:
            return "insufficient_data"

        # Sort logs by timestamp
        sorted_logs = sorted(logs, key=lambda x: x.timestamp)

        # Split into two halves
        mid_point = len(sorted_logs) // 2
        first_half = sorted_logs[:mid_point]
        second_half = sorted_logs[mid_point:]

        first_half_rate = (
            len(first_half) / ((first_half[-1].timestamp - first_half[0].timestamp) / 60)
            if len(first_half) > 1
            else 0
        )
        second_half_rate = (
            len(second_half) / ((second_half[-1].timestamp - second_half[0].timestamp) / 60)
            if len(second_half) > 1
            else 0
        )

        if second_half_rate > first_half_rate * 1.2:
            return "increasing"
        elif second_half_rate < first_half_rate * 0.8:
            return "decreasing"
        else:
            return "stable"

    def _analyze_error_patterns(self, logs: List[LogEntry]) -> Dict[str, Any]:
        """Analyze error patterns in logs."""
        error_logs = [log for log in logs if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]]

        if not error_logs:
            return {"total_errors": 0}

        error_types = Counter(log.error_type for log in error_logs if log.error_type)
        error_components = Counter(log.component for log in error_logs)

        return {
            "total_errors": len(error_logs),
            "error_rate": (len(error_logs) / len(logs)) * 100,
            "top_error_types": dict(error_types.most_common(5)),
            "top_error_components": dict(error_components.most_common(5)),
        }

    def _analyze_performance_patterns(self, logs: List[LogEntry]) -> Dict[str, Any]:
        """Analyze performance patterns in logs."""
        performance_logs = [log for log in logs if log.duration_ms is not None]

        if not performance_logs:
            return {"message": "No performance data available"}

        durations = [log.duration_ms for log in performance_logs]

        return {
            "total_requests": len(performance_logs),
            "avg_response_time": statistics.mean(durations),
            "p50_response_time": statistics.median(durations),
            "p95_response_time": np.percentile(durations, 95),
            "p99_response_time": np.percentile(durations, 99),
            "slowest_components": dict(
                Counter(
                    [
                        log.component
                        for log in performance_logs
                        if log.duration_ms > np.percentile(durations, 90)
                    ]
                ).most_common(5)
            ),
        }

    def _analyze_component_health(self, logs: List[LogEntry]) -> Dict[str, Any]:
        """Analyze health of different components."""
        components = {}

        for component in set(log.component for log in logs):
            component_logs = [log for log in logs if log.component == component]
            error_logs = [
                log for log in component_logs if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]
            ]

            error_rate = (len(error_logs) / len(component_logs)) * 100 if component_logs else 0

            # Calculate health score
            health_score = 100 - (error_rate * 2)
            if health_score < 0:
                health_score = 0

            components[component] = {
                "total_logs": len(component_logs),
                "error_rate": error_rate,
                "health_score": health_score,
                "status": (
                    "healthy"
                    if health_score > 80
                    else "degraded" if health_score > 50 else "critical"
                ),
            }

        return components

    def _identify_top_issues(self, logs: List[LogEntry]) -> List[Dict[str, Any]]:
        """Identify top issues from logs."""
        issues = []

        # Error-based issues
        error_logs = [log for log in logs if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]]
        if error_logs:
            error_types = Counter(log.error_type for log in error_logs if log.error_type)
            for error_type, count in error_types.most_common(3):
                issues.append(
                    {
                        "type": "error",
                        "title": f"Recurring {error_type}",
                        "count": count,
                        "severity": "high" if count > 10 else "medium",
                    }
                )

        # Performance-based issues
        slow_logs = [
            log for log in logs if log.duration_ms and log.duration_ms > 1000
        ]  # > 1 second
        if slow_logs:
            slow_components = Counter(log.component for log in slow_logs)
            for component, count in slow_components.most_common(2):
                issues.append(
                    {
                        "type": "performance",
                        "title": f"Slow responses in {component}",
                        "count": count,
                        "severity": "medium",
                    }
                )

        return issues


# Global instance
advanced_log_analytics = AdvancedLogAnalytics()


# Convenience functions
async def start_advanced_log_analytics():
    """Start advanced log analytics system."""
    await advanced_log_analytics.start()


async def stop_advanced_log_analytics():
    """Stop advanced log analytics system."""
    await advanced_log_analytics.stop()


def ingest_log(raw_message: str, **kwargs) -> LogEntry:
    """Ingest a log message."""
    return advanced_log_analytics.ingest_log(raw_message, **kwargs)


async def analyze_recent_logs(minutes: int = 15):
    """Analyze recent logs."""
    return await advanced_log_analytics.analyze_logs(minutes)


def get_log_insights(hours: int = 24):
    """Get log insights."""
    return advanced_log_analytics.get_log_insights(hours)
