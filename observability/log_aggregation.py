"""
Log Aggregation and Structured Analysis Pipeline for FreeAgentics.

This module provides comprehensive log collection, parsing, analysis, and streaming
capabilities for multi-agent systems with real-time monitoring and alerting.
"""

import asyncio
import json
import logging
import re
import sqlite3
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import aiofiles

# Configure logging for this module
logger = logging.getLogger(__name__)

# ============================================================================
# LOG LEVEL AND SEVERITY DEFINITIONS
# ============================================================================


class LogLevel(Enum):
    """Log level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogSource(Enum):
    """Log source enumeration."""

    AGENT = "agent"
    API = "api"
    AUTH = "auth"
    COALITION = "coalition"
    INFERENCE = "inference"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    OBSERVABILITY = "observability"
    SECURITY = "security"
    SYSTEM = "system"
    WEBSOCKET = "websocket"


class AlertSeverity(Enum):
    """Alert severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class LogEntry:
    """Structured log entry."""

    timestamp: datetime
    level: LogLevel
    source: LogSource
    message: str
    module: str
    function: Optional[str] = None
    agent_id: Optional[str] = None
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    extra_fields: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra_fields is None:
            self.extra_fields = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["level"] = self.level.value
        data["source"] = self.source.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LogEntry":
        """Create LogEntry from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            level=LogLevel(data["level"]),
            source=LogSource(data["source"]),
            message=data["message"],
            module=data["module"],
            function=data.get("function"),
            agent_id=data.get("agent_id"),
            correlation_id=data.get("correlation_id"),
            trace_id=data.get("trace_id"),
            session_id=data.get("session_id"),
            user_id=data.get("user_id"),
            extra_fields=data.get("extra_fields", {}),
        )


@dataclass
class LogPattern:
    """Log pattern for parsing."""

    name: str
    regex: str
    source: LogSource
    level_map: Dict[str, LogLevel]
    field_extractors: Dict[str, str]


@dataclass
class LogAlert:
    """Log-based alert."""

    id: str
    severity: AlertSeverity
    pattern: str
    message: str
    count: int
    first_seen: datetime
    last_seen: datetime
    conditions: Dict[str, Any]


@dataclass
class LogAggregationStats:
    """Log aggregation statistics."""

    total_logs: int
    logs_by_level: Dict[LogLevel, int]
    logs_by_source: Dict[LogSource, int]
    alerts_triggered: int
    parsing_errors: int
    processing_rate: float
    buffer_size: int


# ============================================================================
# LOG PARSING ENGINE
# ============================================================================


class LogParser:
    """Advanced log parsing engine with pattern matching."""

    def __init__(self):
        self.patterns = self._initialize_patterns()
        self.custom_extractors = {}

    def _initialize_patterns(self) -> List[LogPattern]:
        """Initialize built-in log patterns."""
        return [
            # Python logging format
            LogPattern(
                name="python_logging",
                regex=r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (\w+) - (.+)",
                source=LogSource.SYSTEM,
                level_map={
                    "DEBUG": LogLevel.DEBUG,
                    "INFO": LogLevel.INFO,
                    "WARNING": LogLevel.WARNING,
                    "ERROR": LogLevel.ERROR,
                    "CRITICAL": LogLevel.CRITICAL,
                },
                field_extractors={
                    "timestamp": r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})",
                    "level": r"(\w+)",
                    "module": r"(\w+)",
                    "message": r"(.+)",
                },
            ),
            # Agent-specific logs
            LogPattern(
                name="agent_logs",
                regex=r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+) \[(\w+)\] Agent:(\w+) - (.+)",
                source=LogSource.AGENT,
                level_map={
                    "DEBUG": LogLevel.DEBUG,
                    "INFO": LogLevel.INFO,
                    "WARN": LogLevel.WARNING,
                    "ERROR": LogLevel.ERROR,
                    "FATAL": LogLevel.CRITICAL,
                },
                field_extractors={
                    "timestamp": r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)",
                    "level": r"\[(\w+)\]",
                    "agent_id": r"Agent:(\w+)",
                    "message": r"(.+)",
                },
            ),
            # API access logs
            LogPattern(
                name="api_access",
                regex=r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+) - (\w+) - (\w+) - (.+)",
                source=LogSource.API,
                level_map={
                    "DEBUG": LogLevel.DEBUG,
                    "INFO": LogLevel.INFO,
                    "WARNING": LogLevel.WARNING,
                    "ERROR": LogLevel.ERROR,
                    "CRITICAL": LogLevel.CRITICAL,
                },
                field_extractors={
                    "timestamp": r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)",
                    "level": r"(\w+)",
                    "module": r"(\w+)",
                    "message": r"(.+)",
                },
            ),
            # Security logs
            LogPattern(
                name="security_logs",
                regex=r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+) SECURITY \[(\w+)\] (.+)",
                source=LogSource.SECURITY,
                level_map={
                    "INFO": LogLevel.INFO,
                    "WARN": LogLevel.WARNING,
                    "ERROR": LogLevel.ERROR,
                    "CRITICAL": LogLevel.CRITICAL,
                },
                field_extractors={
                    "timestamp": r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)",
                    "level": r"\[(\w+)\]",
                    "message": r"(.+)",
                },
            ),
            # JSON structured logs
            LogPattern(
                name="json_logs",
                regex=r"^\{.*\}$",
                source=LogSource.SYSTEM,
                level_map={
                    "debug": LogLevel.DEBUG,
                    "info": LogLevel.INFO,
                    "warning": LogLevel.WARNING,
                    "error": LogLevel.ERROR,
                    "critical": LogLevel.CRITICAL,
                },
                field_extractors={},
            ),
        ]

    def parse_log_line(
        self, line: str, source_hint: Optional[LogSource] = None
    ) -> Optional[LogEntry]:
        """Parse a single log line into structured format."""
        line = line.strip()
        if not line:
            return None

        # Try JSON parsing first
        if line.startswith("{") and line.endswith("}"):
            try:
                return self._parse_json_log(line, source_hint)
            except Exception:
                pass

        # Try pattern matching
        for pattern in self.patterns:
            if source_hint and pattern.source != source_hint:
                continue

            match = re.match(pattern.regex, line)
            if match:
                return self._extract_log_entry(match, pattern, line)

        # Fallback to basic parsing
        return self._parse_fallback(line, source_hint)

    def _parse_json_log(
        self, line: str, source_hint: Optional[LogSource]
    ) -> LogEntry:
        """Parse JSON-formatted log entry."""
        data = json.loads(line)

        # Extract standard fields
        timestamp = datetime.fromisoformat(
            data.get("timestamp", datetime.now().isoformat())
        )
        level = LogLevel(data.get("level", "INFO").upper())
        source = LogSource(
            data.get("source", source_hint.value if source_hint else "system")
        )
        message = data.get("message", "")
        module = data.get("module", "unknown")

        # Extract optional fields
        function = data.get("function")
        agent_id = data.get("agent_id")
        correlation_id = data.get("correlation_id")
        trace_id = data.get("trace_id")
        session_id = data.get("session_id")
        user_id = data.get("user_id")

        # Extract extra fields
        reserved_fields = {
            "timestamp",
            "level",
            "source",
            "message",
            "module",
            "function",
            "agent_id",
            "correlation_id",
            "trace_id",
            "session_id",
            "user_id",
        }
        extra_fields = {
            k: v for k, v in data.items() if k not in reserved_fields
        }

        return LogEntry(
            timestamp=timestamp,
            level=level,
            source=source,
            message=message,
            module=module,
            function=function,
            agent_id=agent_id,
            correlation_id=correlation_id,
            trace_id=trace_id,
            session_id=session_id,
            user_id=user_id,
            extra_fields=extra_fields,
        )

    def _extract_log_entry(
        self, match: re.Match, pattern: LogPattern, line: str
    ) -> LogEntry:
        """Extract log entry from regex match."""
        groups = match.groups()

        # Extract timestamp
        timestamp_str = groups[0] if groups else None
        if timestamp_str:
            try:
                # Handle different timestamp formats
                if "," in timestamp_str:
                    timestamp = datetime.strptime(
                        timestamp_str, "%Y-%m-%d %H:%M:%S,%f"
                    )
                else:
                    timestamp = datetime.fromisoformat(timestamp_str)
            except ValueError:
                timestamp = datetime.now()
        else:
            timestamp = datetime.now()

        # Extract level
        level_str = groups[1] if len(groups) > 1 else "INFO"
        level = pattern.level_map.get(level_str.upper(), LogLevel.INFO)

        # Extract message (usually the last group)
        message = groups[-1] if groups else line

        # Extract module (usually second group)
        module = groups[2] if len(groups) > 2 else "unknown"

        # Extract agent_id if present
        agent_id = None
        if pattern.name == "agent_logs" and len(groups) > 2:
            agent_id = groups[2]

        return LogEntry(
            timestamp=timestamp,
            level=level,
            source=pattern.source,
            message=message,
            module=module,
            agent_id=agent_id,
        )

    def _parse_fallback(
        self, line: str, source_hint: Optional[LogSource]
    ) -> LogEntry:
        """Fallback parser for unmatched log lines."""
        return LogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            source=source_hint or LogSource.SYSTEM,
            message=line,
            module="unknown",
        )

    def add_custom_pattern(self, pattern: LogPattern):
        """Add a custom log pattern."""
        self.patterns.append(pattern)

    def add_custom_extractor(
        self, name: str, extractor: Callable[[str], Dict[str, Any]]
    ):
        """Add a custom field extractor."""
        self.custom_extractors[name] = extractor


# ============================================================================
# LOG AGGREGATION ENGINE
# ============================================================================


class LogAggregator:
    """Main log aggregation engine."""

    def __init__(
        self,
        db_path: str = "logs/aggregation.db",
        max_buffer_size: int = 10000,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_buffer_size = max_buffer_size

        # Initialize components
        self.parser = LogParser()
        self.buffer = deque(maxlen=max_buffer_size)
        self.stats = LogAggregationStats(
            total_logs=0,
            logs_by_level=defaultdict(int),
            logs_by_source=defaultdict(int),
            alerts_triggered=0,
            parsing_errors=0,
            processing_rate=0.0,
            buffer_size=0,
        )

        # Alert system
        self.alert_patterns = []
        self.active_alerts = {}
        self.alert_callbacks = []

        # Processing control
        self.running = False
        self.processing_task = None
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Initialize database
        self._init_database()

        logger.info("🔄 Log aggregation engine initialized")

    def _init_database(self):
        """Initialize SQLite database for log storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create logs table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                source TEXT NOT NULL,
                message TEXT NOT NULL,
                module TEXT NOT NULL,
                function TEXT,
                agent_id TEXT,
                correlation_id TEXT,
                trace_id TEXT,
                session_id TEXT,
                user_id TEXT,
                extra_fields TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Create indexes
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_logs_source ON logs(source)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_logs_agent_id ON logs(agent_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_logs_correlation_id ON logs(correlation_id)"
        )

        # Create alerts table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                severity TEXT NOT NULL,
                pattern TEXT NOT NULL,
                message TEXT NOT NULL,
                count INTEGER NOT NULL,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                conditions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        conn.commit()
        conn.close()

    async def start(self):
        """Start log aggregation processing."""
        if self.running:
            logger.warning("Log aggregation already running")
            return

        self.running = True
        self.processing_task = asyncio.create_task(self._processing_loop())
        logger.info("🚀 Log aggregation started")

    async def stop(self):
        """Stop log aggregation processing."""
        self.running = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

        self.executor.shutdown(wait=True)
        logger.info("🛑 Log aggregation stopped")

    async def _processing_loop(self):
        """Main processing loop."""
        while self.running:
            try:
                await self._process_buffer()
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                await asyncio.sleep(1)

    async def _process_buffer(self):
        """Process buffered log entries."""
        if not self.buffer:
            return

        # Process in batches
        batch_size = min(100, len(self.buffer))
        batch = []

        for _ in range(batch_size):
            if self.buffer:
                batch.append(self.buffer.popleft())

        if batch:
            await self._process_batch(batch)

    async def _process_batch(self, batch: List[LogEntry]):
        """Process a batch of log entries."""
        start_time = time.time()

        # Update stats
        self.stats.total_logs += len(batch)
        for entry in batch:
            self.stats.logs_by_level[entry.level] += 1
            self.stats.logs_by_source[entry.source] += 1

        # Store in database
        await self._store_batch(batch)

        # Check for alerts
        await self._check_alerts(batch)

        # Update processing rate
        processing_time = time.time() - start_time
        if processing_time > 0:
            self.stats.processing_rate = len(batch) / processing_time

        self.stats.buffer_size = len(self.buffer)

    async def _store_batch(self, batch: List[LogEntry]):
        """Store batch in database."""

        def store_sync():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            for entry in batch:
                cursor.execute(
                    """
                    INSERT INTO logs (
                        timestamp, level, source, message, module, function,
                        agent_id, correlation_id, trace_id, session_id, user_id, extra_fields
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        entry.timestamp.isoformat(),
                        entry.level.value,
                        entry.source.value,
                        entry.message,
                        entry.module,
                        entry.function,
                        entry.agent_id,
                        entry.correlation_id,
                        entry.trace_id,
                        entry.session_id,
                        entry.user_id,
                        json.dumps(entry.extra_fields)
                        if entry.extra_fields
                        else None,
                    ),
                )

            conn.commit()
            conn.close()

        # Run database operations in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, store_sync)

    async def _check_alerts(self, batch: List[LogEntry]):
        """Check log entries against alert patterns."""
        for entry in batch:
            for pattern in self.alert_patterns:
                if self._matches_alert_pattern(entry, pattern):
                    await self._trigger_alert(entry, pattern)

    def _matches_alert_pattern(
        self, entry: LogEntry, pattern: Dict[str, Any]
    ) -> bool:
        """Check if log entry matches alert pattern."""
        # Basic pattern matching
        if pattern.get("level") and entry.level != LogLevel(pattern["level"]):
            return False

        if pattern.get("source") and entry.source != LogSource(
            pattern["source"]
        ):
            return False

        if pattern.get("message_regex"):
            if not re.search(pattern["message_regex"], entry.message):
                return False

        if pattern.get("module") and entry.module != pattern["module"]:
            return False

        return True

    async def _trigger_alert(self, entry: LogEntry, pattern: Dict[str, Any]):
        """Trigger an alert for a log entry."""
        alert_id = pattern["id"]

        if alert_id in self.active_alerts:
            # Update existing alert
            alert = self.active_alerts[alert_id]
            alert.count += 1
            alert.last_seen = entry.timestamp
        else:
            # Create new alert
            alert = LogAlert(
                id=alert_id,
                severity=AlertSeverity(pattern["severity"]),
                pattern=pattern["name"],
                message=pattern["message"],
                count=1,
                first_seen=entry.timestamp,
                last_seen=entry.timestamp,
                conditions=pattern.get("conditions", {}),
            )
            self.active_alerts[alert_id] = alert

        # Store alert in database
        await self._store_alert(alert)

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert, entry)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        self.stats.alerts_triggered += 1

    async def _store_alert(self, alert: LogAlert):
        """Store alert in database."""

        def store_sync():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO alerts (
                    id, severity, pattern, message, count, first_seen, last_seen, conditions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    alert.id,
                    alert.severity.value,
                    alert.pattern,
                    alert.message,
                    alert.count,
                    alert.first_seen.isoformat(),
                    alert.last_seen.isoformat(),
                    json.dumps(alert.conditions),
                ),
            )

            conn.commit()
            conn.close()

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.executor, store_sync)

    def ingest_log_line(self, line: str, source: Optional[LogSource] = None):
        """Ingest a single log line."""
        try:
            entry = self.parser.parse_log_line(line, source)
            if entry:
                self.buffer.append(entry)
            else:
                self.stats.parsing_errors += 1
        except Exception as e:
            logger.error(f"Error parsing log line: {e}")
            self.stats.parsing_errors += 1

    def ingest_log_entry(self, entry: LogEntry):
        """Ingest a structured log entry."""
        self.buffer.append(entry)

    async def ingest_log_file(
        self, file_path: str, source: Optional[LogSource] = None
    ):
        """Ingest logs from a file."""
        try:
            async with aiofiles.open(file_path, "r") as f:
                async for line in f:
                    self.ingest_log_line(line.strip(), source)
        except Exception as e:
            logger.error(f"Error ingesting log file {file_path}: {e}")

    async def query_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        level: Optional[LogLevel] = None,
        source: Optional[LogSource] = None,
        agent_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        message_contains: Optional[str] = None,
        limit: int = 1000,
    ) -> List[LogEntry]:
        """Query logs with filters."""

        def query_sync():
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Build query
            query = "SELECT * FROM logs WHERE 1=1"
            params = []

            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time.isoformat())

            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time.isoformat())

            if level:
                query += " AND level = ?"
                params.append(level.value)

            if source:
                query += " AND source = ?"
                params.append(source.value)

            if agent_id:
                query += " AND agent_id = ?"
                params.append(agent_id)

            if correlation_id:
                query += " AND correlation_id = ?"
                params.append(correlation_id)

            if message_contains:
                query += " AND message LIKE ?"
                params.append(f"%{message_contains}%")

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)
            results = cursor.fetchall()
            conn.close()

            # Convert to LogEntry objects
            entries = []
            for row in results:
                try:
                    entry = LogEntry(
                        timestamp=datetime.fromisoformat(row[1]),
                        level=LogLevel(row[2]),
                        source=LogSource(row[3]),
                        message=row[4],
                        module=row[5],
                        function=row[6],
                        agent_id=row[7],
                        correlation_id=row[8],
                        trace_id=row[9],
                        session_id=row[10],
                        user_id=row[11],
                        extra_fields=json.loads(row[12]) if row[12] else {},
                    )
                    entries.append(entry)
                except Exception as e:
                    logger.error(f"Error parsing log entry: {e}")
                    continue

            return entries

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, query_sync)

    def add_alert_pattern(self, pattern: Dict[str, Any]):
        """Add an alert pattern."""
        self.alert_patterns.append(pattern)

    def add_alert_callback(
        self, callback: Callable[[LogAlert, LogEntry], None]
    ):
        """Add alert callback."""
        self.alert_callbacks.append(callback)

    def get_stats(self) -> LogAggregationStats:
        """Get aggregation statistics."""
        return self.stats

    def get_active_alerts(self) -> Dict[str, LogAlert]:
        """Get active alerts."""
        return self.active_alerts.copy()


# ============================================================================
# LOG STREAMING SERVER
# ============================================================================


class LogStreamingServer:
    """Real-time log streaming server."""

    def __init__(self, aggregator: LogAggregator, port: int = 9999):
        self.aggregator = aggregator
        self.port = port
        self.clients = set()
        self.running = False
        self.server_task = None

    async def start(self):
        """Start the streaming server."""
        if self.running:
            return

        self.running = True
        self.server_task = asyncio.create_task(self._run_server())
        logger.info(f"📡 Log streaming server started on port {self.port}")

    async def stop(self):
        """Stop the streaming server."""
        self.running = False
        if self.server_task:
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass

        # Close all client connections
        for client in self.clients.copy():
            try:
                await client.close()
            except Exception:
                pass

        self.clients.clear()
        logger.info("📡 Log streaming server stopped")

    async def _run_server(self):
        """Run the streaming server."""
        try:
            from aiohttp import WSMsgType, web

            async def websocket_handler(request):
                ws = web.WebSocketResponse()
                await ws.prepare(request)

                self.clients.add(ws)
                logger.info(
                    f"📡 New log streaming client connected. Total: {len(self.clients)}"
                )

                try:
                    async for msg in ws:
                        if msg.type == WSMsgType.TEXT:
                            # Handle client messages (filters, etc.)
                            try:
                                data = json.loads(msg.data)
                                await self._handle_client_message(ws, data)
                            except Exception as e:
                                logger.error(
                                    f"Error handling client message: {e}"
                                )
                        elif msg.type == WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            break
                finally:
                    self.clients.discard(ws)
                    logger.info(
                        f"📡 Log streaming client disconnected. Total: {len(self.clients)}"
                    )

                return ws

            app = web.Application()
            app.router.add_get("/logs/stream", websocket_handler)

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "localhost", self.port)
            await site.start()

            # Keep the server running
            while self.running:
                await asyncio.sleep(1)

            await runner.cleanup()

        except Exception as e:
            logger.error(f"Log streaming server error: {e}")

    async def _handle_client_message(self, ws, data: Dict[str, Any]):
        """Handle client message."""
        message_type = data.get("type")

        if message_type == "subscribe":
            # Send recent logs
            data.get("filters", {})
            recent_logs = await self.aggregator.query_logs(
                start_time=datetime.now() - timedelta(minutes=5), limit=100
            )

            await ws.send_text(
                json.dumps(
                    {
                        "type": "logs",
                        "data": [log.to_dict() for log in recent_logs],
                    }
                )
            )

        elif message_type == "filter":
            # Update client filters (implementation depends on requirements)
            pass

    async def broadcast_log(self, entry: LogEntry):
        """Broadcast log entry to all connected clients."""
        if not self.clients:
            return

        message = json.dumps({"type": "log", "data": entry.to_dict()})

        # Send to all clients
        disconnected_clients = []
        for client in self.clients:
            try:
                await client.send_text(message)
            except Exception:
                disconnected_clients.append(client)

        # Remove disconnected clients
        for client in disconnected_clients:
            self.clients.discard(client)


# ============================================================================
# GLOBAL AGGREGATOR INSTANCE
# ============================================================================

log_aggregator = LogAggregator()
log_streaming_server = LogStreamingServer(log_aggregator)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


async def start_log_aggregation():
    """Start log aggregation system."""
    await log_aggregator.start()
    await log_streaming_server.start()


async def stop_log_aggregation():
    """Stop log aggregation system."""
    await log_aggregator.stop()
    await log_streaming_server.stop()


def create_structured_log_entry(
    level: LogLevel,
    source: LogSource,
    message: str,
    module: str,
    agent_id: Optional[str] = None,
    correlation_id: Optional[str] = None,
    **kwargs,
) -> LogEntry:
    """Create a structured log entry."""
    return LogEntry(
        timestamp=datetime.now(),
        level=level,
        source=source,
        message=message,
        module=module,
        agent_id=agent_id,
        correlation_id=correlation_id,
        extra_fields=kwargs,
    )


def log_agent_action(
    agent_id: str, action: str, details: Dict[str, Any] = None
):
    """Log agent action."""
    entry = create_structured_log_entry(
        level=LogLevel.INFO,
        source=LogSource.AGENT,
        message=f"Agent {agent_id} performed action: {action}",
        module="agent_actions",
        agent_id=agent_id,
        action=action,
        details=details or {},
    )
    log_aggregator.ingest_log_entry(entry)


def log_api_request(
    method: str, path: str, status_code: int, user_id: Optional[str] = None
):
    """Log API request."""
    entry = create_structured_log_entry(
        level=LogLevel.INFO,
        source=LogSource.API,
        message=f"{method} {path} -> {status_code}",
        module="api_requests",
        user_id=user_id,
        method=method,
        path=path,
        status_code=status_code,
    )
    log_aggregator.ingest_log_entry(entry)


def log_security_event(event_type: str, details: Dict[str, Any]):
    """Log security event."""
    entry = create_structured_log_entry(
        level=LogLevel.WARNING,
        source=LogSource.SECURITY,
        message=f"Security event: {event_type}",
        module="security",
        event_type=event_type,
        details=details,
    )
    log_aggregator.ingest_log_entry(entry)


# ============================================================================
# EXAMPLE ALERT PATTERNS
# ============================================================================

# Default alert patterns
DEFAULT_ALERT_PATTERNS = [
    {
        "id": "error_rate_high",
        "name": "High Error Rate",
        "severity": "high",
        "level": "ERROR",
        "message": "High error rate detected",
        "conditions": {"threshold": 10, "window": 60},
    },
    {
        "id": "agent_failures",
        "name": "Agent Failures",
        "severity": "medium",
        "source": "agent",
        "level": "ERROR",
        "message": "Agent failure detected",
        "conditions": {},
    },
    {
        "id": "security_breach",
        "name": "Security Breach",
        "severity": "critical",
        "source": "security",
        "level": "ERROR",
        "message": "Potential security breach detected",
        "conditions": {},
    },
]

# Initialize default alert patterns
for pattern in DEFAULT_ALERT_PATTERNS:
    log_aggregator.add_alert_pattern(pattern)
