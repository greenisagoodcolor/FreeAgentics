"""
Log Analysis Dashboard for FreeAgentics.

Provides comprehensive log analysis, visualization, and insights for the
multi-agent system with real-time monitoring capabilities.
"""

import json
import logging
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from observability.log_aggregation import (
    LogAggregator,
    LogEntry,
    LogLevel,
    LogSource,
    log_aggregator,
)

logger = logging.getLogger(__name__)

# ============================================================================
# ANALYSIS DATA STRUCTURES
# ============================================================================


@dataclass
class LogAnalysisResult:
    """Result of log analysis."""

    total_logs: int
    time_range: Tuple[datetime, datetime]
    level_distribution: Dict[str, int]
    source_distribution: Dict[str, int]
    top_errors: List[Dict[str, Any]]
    agent_activity: Dict[str, int]
    timeline_data: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]
    recommendations: List[str]


@dataclass
class AgentAnalysisResult:
    """Agent-specific analysis result."""

    agent_id: str
    total_logs: int
    error_rate: float
    activity_timeline: List[Dict[str, Any]]
    common_actions: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    correlations: List[Dict[str, Any]]


@dataclass
class AnomalyDetectionResult:
    """Anomaly detection result."""

    anomaly_type: str
    severity: str
    timestamp: datetime
    description: str
    affected_components: List[str]
    confidence: float
    metadata: Dict[str, Any]


# ============================================================================
# LOG ANALYSIS ENGINE
# ============================================================================


class LogAnalysisEngine:
    """Advanced log analysis engine with pattern recognition and anomaly detection."""

    def __init__(self, aggregator: LogAggregator):
        self.aggregator = aggregator
        self.anomaly_thresholds = {
            "error_rate": 0.1,  # 10% error rate threshold
            "response_time": 5.0,  # 5 second response time threshold
            "memory_usage": 0.8,  # 80% memory usage threshold
            "agent_failure_rate": 0.05,  # 5% agent failure rate threshold
        }

        # Pattern recognition data
        self.known_patterns = {}
        self.baseline_metrics = {}

        logger.info("🔍 Log analysis engine initialized")

    async def analyze_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        sources: Optional[List[LogSource]] = None,
    ) -> LogAnalysisResult:
        """Perform comprehensive log analysis."""
        # Default time range (last 24 hours)
        if not start_time:
            start_time = datetime.now() - timedelta(hours=24)
        if not end_time:
            end_time = datetime.now()

        # Query logs
        logs = await self._query_logs_for_analysis(start_time, end_time, sources)

        if not logs:
            return LogAnalysisResult(
                total_logs=0,
                time_range=(start_time, end_time),
                level_distribution={},
                source_distribution={},
                top_errors=[],
                agent_activity={},
                timeline_data=[],
                anomalies=[],
                recommendations=[],
            )

        # Perform analysis
        level_distribution = self._analyze_level_distribution(logs)
        source_distribution = self._analyze_source_distribution(logs)
        top_errors = self._analyze_top_errors(logs)
        agent_activity = self._analyze_agent_activity(logs)
        timeline_data = self._analyze_timeline(logs)
        anomalies = await self._detect_anomalies(logs)
        recommendations = self._generate_recommendations(logs, anomalies)

        return LogAnalysisResult(
            total_logs=len(logs),
            time_range=(start_time, end_time),
            level_distribution=level_distribution,
            source_distribution=source_distribution,
            top_errors=top_errors,
            agent_activity=agent_activity,
            timeline_data=timeline_data,
            anomalies=anomalies,
            recommendations=recommendations,
        )

    async def analyze_agent_logs(
        self,
        agent_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> AgentAnalysisResult:
        """Analyze logs for a specific agent."""
        # Default time range (last 24 hours)
        if not start_time:
            start_time = datetime.now() - timedelta(hours=24)
        if not end_time:
            end_time = datetime.now()

        # Query agent-specific logs
        logs = await self.aggregator.query_logs(
            start_time=start_time,
            end_time=end_time,
            agent_id=agent_id,
            limit=10000,
        )

        if not logs:
            return AgentAnalysisResult(
                agent_id=agent_id,
                total_logs=0,
                error_rate=0.0,
                activity_timeline=[],
                common_actions=[],
                performance_metrics={},
                correlations=[],
            )

        # Analyze agent-specific metrics
        error_rate = self._calculate_error_rate(logs)
        activity_timeline = self._analyze_agent_timeline(logs)
        common_actions = self._analyze_common_actions(logs)
        performance_metrics = self._analyze_agent_performance(logs)
        correlations = await self._analyze_agent_correlations(agent_id, logs)

        return AgentAnalysisResult(
            agent_id=agent_id,
            total_logs=len(logs),
            error_rate=error_rate,
            activity_timeline=activity_timeline,
            common_actions=common_actions,
            performance_metrics=performance_metrics,
            correlations=correlations,
        )

    async def _query_logs_for_analysis(
        self,
        start_time: datetime,
        end_time: datetime,
        sources: Optional[List[LogSource]],
    ) -> List[LogEntry]:
        """Query logs for analysis with appropriate filtering."""
        logs = []

        if sources:
            # Query each source separately
            for source in sources:
                source_logs = await self.aggregator.query_logs(
                    start_time=start_time,
                    end_time=end_time,
                    source=source,
                    limit=5000,
                )
                logs.extend(source_logs)
        else:
            # Query all logs
            logs = await self.aggregator.query_logs(
                start_time=start_time, end_time=end_time, limit=10000
            )

        return logs

    def _analyze_level_distribution(self, logs: List[LogEntry]) -> Dict[str, int]:
        """Analyze distribution of log levels."""
        distribution = defaultdict(int)
        for log in logs:
            distribution[log.level.value] += 1
        return dict(distribution)

    def _analyze_source_distribution(self, logs: List[LogEntry]) -> Dict[str, int]:
        """Analyze distribution of log sources."""
        distribution = defaultdict(int)
        for log in logs:
            distribution[log.source.value] += 1
        return dict(distribution)

    def _analyze_top_errors(self, logs: List[LogEntry]) -> List[Dict[str, Any]]:
        """Analyze top error messages."""
        error_logs = [
            log for log in logs if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]
        ]

        # Count error messages
        error_counts = Counter(log.message for log in error_logs)

        # Get top errors
        top_errors = []
        for message, count in error_counts.most_common(10):
            # Find first occurrence for timestamp
            first_occurrence = next(log for log in error_logs if log.message == message)

            top_errors.append(
                {
                    "message": message,
                    "count": count,
                    "first_seen": first_occurrence.timestamp.isoformat(),
                    "level": first_occurrence.level.value,
                    "source": first_occurrence.source.value,
                }
            )

        return top_errors

    def _analyze_agent_activity(self, logs: List[LogEntry]) -> Dict[str, int]:
        """Analyze agent activity levels."""
        activity = defaultdict(int)
        for log in logs:
            if log.agent_id:
                activity[log.agent_id] += 1
        return dict(activity)

    def _analyze_timeline(self, logs: List[LogEntry]) -> List[Dict[str, Any]]:
        """Analyze log timeline data."""
        # Group logs by hour
        hourly_data = defaultdict(lambda: defaultdict(int))

        for log in logs:
            hour_key = log.timestamp.strftime("%Y-%m-%d %H:00:00")
            hourly_data[hour_key][log.level.value] += 1

        # Convert to timeline format
        timeline = []
        for hour, level_counts in sorted(hourly_data.items()):
            timeline.append(
                {
                    "timestamp": hour,
                    "total": sum(level_counts.values()),
                    "levels": dict(level_counts),
                }
            )

        return timeline

    async def _detect_anomalies(self, logs: List[LogEntry]) -> List[Dict[str, Any]]:
        """Detect anomalies in log patterns."""
        anomalies = []

        # Error rate anomaly detection
        error_rate_anomaly = self._detect_error_rate_anomaly(logs)
        if error_rate_anomaly:
            anomalies.append(error_rate_anomaly)

        # Response time anomaly detection
        response_time_anomaly = self._detect_response_time_anomaly(logs)
        if response_time_anomaly:
            anomalies.append(response_time_anomaly)

        # Agent failure anomaly detection
        agent_failure_anomaly = self._detect_agent_failure_anomaly(logs)
        if agent_failure_anomaly:
            anomalies.append(agent_failure_anomaly)

        # Volume anomaly detection
        volume_anomaly = self._detect_volume_anomaly(logs)
        if volume_anomaly:
            anomalies.append(volume_anomaly)

        return anomalies

    def _detect_error_rate_anomaly(
        self, logs: List[LogEntry]
    ) -> Optional[Dict[str, Any]]:
        """Detect error rate anomalies."""
        total_logs = len(logs)
        error_logs = len(
            [log for log in logs if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]]
        )

        if total_logs == 0:
            return None

        error_rate = error_logs / total_logs

        if error_rate > self.anomaly_thresholds["error_rate"]:
            return {
                "type": "error_rate_anomaly",
                "severity": "high" if error_rate > 0.2 else "medium",
                "timestamp": datetime.now().isoformat(),
                "description": f"Error rate is {error_rate:.2%} (threshold: {self.anomaly_thresholds['error_rate']:.2%})",
                "affected_components": ["system"],
                "confidence": 0.9,
                "metadata": {
                    "current_rate": error_rate,
                    "threshold": self.anomaly_thresholds["error_rate"],
                    "total_logs": total_logs,
                    "error_logs": error_logs,
                },
            }

        return None

    def _detect_response_time_anomaly(
        self, logs: List[LogEntry]
    ) -> Optional[Dict[str, Any]]:
        """Detect response time anomalies."""
        # Look for response time patterns in log messages
        response_times = []
        for log in logs:
            if "response_time" in log.extra_fields:
                response_times.append(log.extra_fields["response_time"])
            # Parse response time from message (simplified)
            elif "ms" in log.message:
                import re

                match = re.search(r"(\d+(\.\d+)?)\s*ms", log.message)
                if match:
                    response_times.append(float(match.group(1)) / 1000.0)

        if not response_times:
            return None

        avg_response_time = statistics.mean(response_times)

        if avg_response_time > self.anomaly_thresholds["response_time"]:
            return {
                "type": "response_time_anomaly",
                "severity": "high" if avg_response_time > 10.0 else "medium",
                "timestamp": datetime.now().isoformat(),
                "description": f"Average response time is {avg_response_time:.2f}s (threshold: {self.anomaly_thresholds['response_time']:.2f}s)",
                "affected_components": ["api"],
                "confidence": 0.8,
                "metadata": {
                    "current_avg": avg_response_time,
                    "threshold": self.anomaly_thresholds["response_time"],
                    "sample_count": len(response_times),
                    "max_response_time": max(response_times),
                },
            }

        return None

    def _detect_agent_failure_anomaly(
        self, logs: List[LogEntry]
    ) -> Optional[Dict[str, Any]]:
        """Detect agent failure anomalies."""
        agent_logs = [log for log in logs if log.agent_id]
        agent_errors = [
            log
            for log in agent_logs
            if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]
        ]

        if not agent_logs:
            return None

        failure_rate = len(agent_errors) / len(agent_logs)

        if failure_rate > self.anomaly_thresholds["agent_failure_rate"]:
            # Find affected agents
            affected_agents = set(log.agent_id for log in agent_errors)

            return {
                "type": "agent_failure_anomaly",
                "severity": "high" if failure_rate > 0.1 else "medium",
                "timestamp": datetime.now().isoformat(),
                "description": f"Agent failure rate is {failure_rate:.2%} (threshold: {self.anomaly_thresholds['agent_failure_rate']:.2%})",
                "affected_components": list(affected_agents),
                "confidence": 0.9,
                "metadata": {
                    "current_rate": failure_rate,
                    "threshold": self.anomaly_thresholds["agent_failure_rate"],
                    "total_agent_logs": len(agent_logs),
                    "agent_errors": len(agent_errors),
                    "affected_agents": list(affected_agents),
                },
            }

        return None

    def _detect_volume_anomaly(self, logs: List[LogEntry]) -> Optional[Dict[str, Any]]:
        """Detect log volume anomalies."""
        if not logs:
            return None

        # Calculate logs per minute
        time_span = (logs[0].timestamp - logs[-1].timestamp).total_seconds() / 60.0
        if time_span == 0:
            return None

        logs_per_minute = len(logs) / time_span

        # Simple threshold-based detection (could be improved with historical data)
        if logs_per_minute > 1000:  # More than 1000 logs per minute
            return {
                "type": "volume_anomaly",
                "severity": "medium",
                "timestamp": datetime.now().isoformat(),
                "description": f"High log volume: {logs_per_minute:.0f} logs/minute",
                "affected_components": ["system"],
                "confidence": 0.7,
                "metadata": {
                    "logs_per_minute": logs_per_minute,
                    "total_logs": len(logs),
                    "time_span_minutes": time_span,
                },
            }

        return None

    def _generate_recommendations(
        self, logs: List[LogEntry], anomalies: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on log analysis."""
        recommendations = []

        # Error rate recommendations
        error_logs = [
            log for log in logs if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]
        ]
        if error_logs:
            error_rate = len(error_logs) / len(logs)
            if error_rate > 0.05:
                recommendations.append(
                    f"High error rate ({error_rate:.2%}) detected. Review error logs and implement fixes."
                )

        # Agent-specific recommendations
        agent_activity = self._analyze_agent_activity(logs)
        if agent_activity:
            inactive_agents = [
                agent_id for agent_id, count in agent_activity.items() if count < 10
            ]
            if inactive_agents:
                recommendations.append(
                    f"Low activity detected for agents: {', '.join(inactive_agents[:5])}. Check agent health."
                )

        # Source-specific recommendations
        source_distribution = self._analyze_source_distribution(logs)
        if source_distribution.get("security", 0) > 100:
            recommendations.append(
                "High security log volume detected. Review security events and consider additional monitoring."
            )

        # Anomaly-based recommendations
        for anomaly in anomalies:
            if anomaly["type"] == "error_rate_anomaly":
                recommendations.append(
                    "Error rate anomaly detected. Investigate root cause and implement error handling improvements."
                )
            elif anomaly["type"] == "response_time_anomaly":
                recommendations.append(
                    "Response time anomaly detected. Profile application performance and optimize slow operations."
                )
            elif anomaly["type"] == "agent_failure_anomaly":
                recommendations.append(
                    "Agent failure anomaly detected. Review agent configuration and implement failover mechanisms."
                )

        return recommendations

    def _calculate_error_rate(self, logs: List[LogEntry]) -> float:
        """Calculate error rate for logs."""
        if not logs:
            return 0.0

        error_count = len(
            [log for log in logs if log.level in [LogLevel.ERROR, LogLevel.CRITICAL]]
        )
        return error_count / len(logs)

    def _analyze_agent_timeline(self, logs: List[LogEntry]) -> List[Dict[str, Any]]:
        """Analyze agent activity timeline."""
        # Group by hour
        hourly_activity = defaultdict(int)

        for log in logs:
            hour_key = log.timestamp.strftime("%Y-%m-%d %H:00:00")
            hourly_activity[hour_key] += 1

        # Convert to timeline format
        timeline = []
        for hour, count in sorted(hourly_activity.items()):
            timeline.append({"timestamp": hour, "activity_count": count})

        return timeline

    def _analyze_common_actions(self, logs: List[LogEntry]) -> List[Dict[str, Any]]:
        """Analyze common actions for an agent."""
        # Extract actions from log messages and extra fields
        actions = []

        for log in logs:
            if "action" in log.extra_fields:
                actions.append(log.extra_fields["action"])
            # Parse action from message (simplified)
            elif "action:" in log.message.lower():
                import re

                match = re.search(r"action:\s*(\w+)", log.message.lower())
                if match:
                    actions.append(match.group(1))

        # Count actions
        action_counts = Counter(actions)

        # Convert to result format
        common_actions = []
        for action, count in action_counts.most_common(10):
            common_actions.append(
                {
                    "action": action,
                    "count": count,
                    "frequency": count / len(logs) if logs else 0,
                }
            )

        return common_actions

    def _analyze_agent_performance(self, logs: List[LogEntry]) -> Dict[str, float]:
        """Analyze agent performance metrics."""
        metrics = {
            "error_rate": self._calculate_error_rate(logs),
            "activity_rate": len(logs)
            / 24.0,  # logs per hour (assuming 24-hour window)
            "avg_response_time": 0.0,
            "success_rate": 0.0,
        }

        # Calculate response times
        response_times = []
        for log in logs:
            if "response_time" in log.extra_fields:
                response_times.append(log.extra_fields["response_time"])

        if response_times:
            metrics["avg_response_time"] = statistics.mean(response_times)

        # Calculate success rate
        success_logs = [
            log for log in logs if log.level in [LogLevel.INFO, LogLevel.DEBUG]
        ]
        if logs:
            metrics["success_rate"] = len(success_logs) / len(logs)

        return metrics

    async def _analyze_agent_correlations(
        self, agent_id: str, logs: List[LogEntry]
    ) -> List[Dict[str, Any]]:
        """Analyze correlations between this agent and other system components."""
        correlations = []

        # Find logs with same correlation_id
        correlation_ids = set(log.correlation_id for log in logs if log.correlation_id)

        for correlation_id in correlation_ids:
            # Query logs with same correlation_id
            correlated_logs = await self.aggregator.query_logs(
                correlation_id=correlation_id, limit=100
            )

            # Analyze involved components
            involved_agents = set(
                log.agent_id
                for log in correlated_logs
                if log.agent_id and log.agent_id != agent_id
            )
            involved_sources = set(
                log.source.value
                for log in correlated_logs
                if log.source.value != "agent"
            )

            if involved_agents or involved_sources:
                correlations.append(
                    {
                        "correlation_id": correlation_id,
                        "involved_agents": list(involved_agents),
                        "involved_sources": list(involved_sources),
                        "total_logs": len(correlated_logs),
                        "time_span": (
                            (
                                correlated_logs[0].timestamp
                                - correlated_logs[-1].timestamp
                            ).total_seconds()
                            if len(correlated_logs) > 1
                            else 0
                        ),
                    }
                )

        return correlations


# ============================================================================
# DASHBOARD GENERATOR
# ============================================================================


class LogDashboardGenerator:
    """Generate HTML dashboard for log analysis."""

    def __init__(self, analysis_engine: LogAnalysisEngine):
        self.analysis_engine = analysis_engine

    async def generate_dashboard(self, output_path: str = "logs/dashboard.html"):
        """Generate comprehensive log analysis dashboard."""
        # Perform analysis
        analysis_result = await self.analysis_engine.analyze_logs()

        # Generate HTML
        html_content = self._generate_html_dashboard(analysis_result)

        # Write to file
        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"📊 Log dashboard generated: {output_path}")
        return output_path

    def _generate_html_dashboard(self, analysis: LogAnalysisResult) -> str:
        """Generate HTML dashboard content."""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>FreeAgentics Log Analysis Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                .header {{ background-color: #2196f3; color: white; padding: 20px; border-radius: 5px; }}
                .stats {{ display: flex; gap: 20px; margin: 20px 0; }}
                .stat-card {{ background-color: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .chart-container {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                .anomaly {{ background-color: #ffebee; padding: 15px; margin: 10px 0; border-left: 4px solid #f44336; }}
                .recommendation {{ background-color: #e8f5e8; padding: 15px; margin: 10px 0; border-left: 4px solid #4caf50; }}
                .error-list {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 5px; }}
                table {{ width: 100%; border-collapse: collapse; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 FreeAgentics Log Analysis Dashboard</h1>
                <p>Analysis Period: {analysis.time_range[0].strftime("%Y-%m-%d %H:%M")} - {analysis.time_range[1].strftime("%Y-%m-%d %H:%M")}</p>
                <p>Total Logs Analyzed: {analysis.total_logs:,}</p>
            </div>

            <div class="stats">
                <div class="stat-card">
                    <h3>📊 Log Levels</h3>
                    <ul>
                        {self._generate_level_stats(analysis.level_distribution)}
                    </ul>
                </div>
                <div class="stat-card">
                    <h3>🔧 Sources</h3>
                    <ul>
                        {self._generate_source_stats(analysis.source_distribution)}
                    </ul>
                </div>
                <div class="stat-card">
                    <h3>🤖 Agent Activity</h3>
                    <p>Active Agents: {len(analysis.agent_activity)}</p>
                    <p>Most Active: {self._get_most_active_agent(analysis.agent_activity)}</p>
                </div>
            </div>

            <div class="chart-container">
                <h3>📈 Timeline Analysis</h3>
                <canvas id="timelineChart" width="400" height="200"></canvas>
            </div>

            <div class="chart-container">
                <h3>📊 Log Level Distribution</h3>
                <canvas id="levelChart" width="400" height="200"></canvas>
            </div>

            {self._generate_anomalies_section(analysis.anomalies)}

            {self._generate_recommendations_section(analysis.recommendations)}

            {self._generate_top_errors_section(analysis.top_errors)}

            <script>
                // Timeline Chart
                {self._generate_timeline_chart_script(analysis.timeline_data)}

                // Level Distribution Chart
                {self._generate_level_chart_script(analysis.level_distribution)}
            </script>
        </body>
        </html>
        """

    def _generate_level_stats(self, level_distribution: Dict[str, int]) -> str:
        """Generate level statistics HTML."""
        stats = []
        for level, count in level_distribution.items():
            stats.append(f"<li>{level}: {count:,}</li>")
        return "\n".join(stats)

    def _generate_source_stats(self, source_distribution: Dict[str, int]) -> str:
        """Generate source statistics HTML."""
        stats = []
        for source, count in source_distribution.items():
            stats.append(f"<li>{source}: {count:,}</li>")
        return "\n".join(stats)

    def _get_most_active_agent(self, agent_activity: Dict[str, int]) -> str:
        """Get most active agent."""
        if not agent_activity:
            return "None"

        most_active = max(agent_activity.items(), key=lambda x: x[1])
        return f"{most_active[0]} ({most_active[1]} logs)"

    def _generate_anomalies_section(self, anomalies: List[Dict[str, Any]]) -> str:
        """Generate anomalies section."""
        if not anomalies:
            return ""

        html = "<h3>🚨 Anomalies Detected</h3>"
        for anomaly in anomalies:
            html += f"""
            <div class="anomaly">
                <h4>{anomaly["type"]} - {anomaly["severity"].upper()}</h4>
                <p>{anomaly["description"]}</p>
                <p><strong>Confidence:</strong> {anomaly["confidence"]:.0%}</p>
                <p><strong>Affected Components:</strong> {", ".join(anomaly["affected_components"])}</p>
            </div>
            """

        return html

    def _generate_recommendations_section(self, recommendations: List[str]) -> str:
        """Generate recommendations section."""
        if not recommendations:
            return ""

        html = "<h3>💡 Recommendations</h3>"
        for rec in recommendations:
            html += f'<div class="recommendation">{rec}</div>'

        return html

    def _generate_top_errors_section(self, top_errors: List[Dict[str, Any]]) -> str:
        """Generate top errors section."""
        if not top_errors:
            return ""

        html = """
        <div class="error-list">
            <h3>🔥 Top Errors</h3>
            <table>
                <tr>
                    <th>Error Message</th>
                    <th>Count</th>
                    <th>Level</th>
                    <th>Source</th>
                    <th>First Seen</th>
                </tr>
        """

        for error in top_errors:
            html += f"""
                <tr>
                    <td>{error["message"][:100]}...</td>
                    <td>{error["count"]}</td>
                    <td>{error["level"]}</td>
                    <td>{error["source"]}</td>
                    <td>{error["first_seen"]}</td>
                </tr>
            """

        html += """
            </table>
        </div>
        """

        return html

    def _generate_timeline_chart_script(
        self, timeline_data: List[Dict[str, Any]]
    ) -> str:
        """Generate timeline chart JavaScript."""
        labels = [item["timestamp"] for item in timeline_data]
        data = [item["total"] for item in timeline_data]

        return f"""
        const timelineCtx = document.getElementById('timelineChart').getContext('2d');
        new Chart(timelineCtx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [{{
                    label: 'Total Logs',
                    data: {json.dumps(data)},
                    borderColor: '#2196f3',
                    backgroundColor: 'rgba(33, 150, 243, 0.1)',
                    tension: 0.4
                }}]
            }},
            options: {{
                responsive: true,
                scales: {{
                    x: {{
                        display: true,
                        title: {{
                            display: true,
                            text: 'Time'
                        }}
                    }},
                    y: {{
                        display: true,
                        title: {{
                            display: true,
                            text: 'Log Count'
                        }}
                    }}
                }}
            }}
        }});
        """

    def _generate_level_chart_script(self, level_distribution: Dict[str, int]) -> str:
        """Generate level distribution chart JavaScript."""
        labels = list(level_distribution.keys())
        data = list(level_distribution.values())

        colors = {
            "DEBUG": "#9e9e9e",
            "INFO": "#2196f3",
            "WARNING": "#ff9800",
            "ERROR": "#f44336",
            "CRITICAL": "#e91e63",
        }

        background_colors = [colors.get(label, "#9e9e9e") for label in labels]

        return f"""
        const levelCtx = document.getElementById('levelChart').getContext('2d');
        new Chart(levelCtx, {{
            type: 'doughnut',
            data: {{
                labels: {json.dumps(labels)},
                datasets: [{{
                    data: {json.dumps(data)},
                    backgroundColor: {json.dumps(background_colors)}
                }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    legend: {{
                        position: 'bottom'
                    }}
                }}
            }}
        }});
        """


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

# Initialize global instances
log_analysis_engine = LogAnalysisEngine(log_aggregator)
log_dashboard_generator = LogDashboardGenerator(log_analysis_engine)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


async def generate_log_dashboard(output_path: str = "logs/dashboard.html"):
    """Generate log analysis dashboard."""
    return await log_dashboard_generator.generate_dashboard(output_path)


async def analyze_recent_logs(hours: int = 24):
    """Analyze recent logs."""
    start_time = datetime.now() - timedelta(hours=hours)
    return await log_analysis_engine.analyze_logs(start_time=start_time)


async def analyze_agent_performance(agent_id: str, hours: int = 24):
    """Analyze specific agent performance."""
    start_time = datetime.now() - timedelta(hours=hours)
    return await log_analysis_engine.analyze_agent_logs(agent_id, start_time=start_time)
