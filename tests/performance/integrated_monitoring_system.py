"""Integrated Performance Monitoring and Analysis System.

This module provides a comprehensive system that integrates all performance monitoring,
profiling, analysis, and reporting tools into a unified interface for complete observability
of the FreeAgentics system.
"""

import asyncio
import json
import logging
import signal
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
import yaml

from tests.performance.agent_simulation_framework import (
    AgentSimulationFramework,
)
from tests.performance.monitoring_dashboard import (
    DashboardConfig,
    MetricsDashboard,
)
from tests.performance.performance_profiler import (
    ComponentProfiler,
)
from tests.performance.performance_report_generator import (
    PerformanceReportGenerator,
)
from tests.performance.regression_analyzer import (
    RegressionAnalyzer,
    RegressionSeverity,
    initialize_regression_analyzer,
)

# Import load test frameworks
from tests.performance.test_coordination_load import CoordinationLoadTester
from tests.performance.test_database_load_real import DatabaseLoadTester

# Import all monitoring components
from tests.performance.unified_metrics_collector import (
    MetricSource,
    UnifiedMetricsCollector,
)
from tests.websocket_load.websocket_load_test import WebSocketLoadTest

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for the integrated monitoring system."""

    # Metrics collection
    enable_metrics: bool = True
    metrics_buffer_size: int = 10000
    metrics_persistence: bool = True

    # Dashboard
    enable_dashboard: bool = True
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8090

    # Profiling
    enable_profiling: bool = True
    profile_cpu: bool = True
    profile_memory: bool = True

    # Regression analysis
    enable_regression_analysis: bool = True
    regression_baseline_duration: int = 300  # seconds
    regression_check_interval: int = 300  # seconds

    # Alerting
    enable_alerts: bool = True
    alert_email: Optional[str] = None
    alert_webhook: Optional[str] = None

    # Reporting
    enable_automated_reports: bool = True
    report_interval_hours: int = 24
    report_formats: List[str] = None

    # Load testing integration
    enable_load_testing: bool = False
    load_test_scenarios: List[str] = None

    def __post_init__(self):
        if self.report_formats is None:
            self.report_formats = ["html", "json", "markdown"]
        if self.load_test_scenarios is None:
            self.load_test_scenarios = []


class IntegratedMonitoringSystem:
    """Comprehensive monitoring system integrating all performance tools."""

    def __init__(self, config: MonitoringConfig = None):
        """Initialize the integrated monitoring system."""
        self.config = config or MonitoringConfig()
        self.is_running = False

        # Initialize components
        self.metrics_collector = UnifiedMetricsCollector(
            buffer_size=self.config.metrics_buffer_size,
            persistence_enabled=self.config.metrics_persistence,
        )

        self.dashboard = None
        if self.config.enable_dashboard:
            self.dashboard = MetricsDashboard(
                DashboardConfig(
                    host=self.config.dashboard_host,
                    port=self.config.dashboard_port,
                )
            )

        self.profiler = (
            ComponentProfiler(
                enable_cpu_profiling=self.config.profile_cpu,
                enable_memory_profiling=self.config.profile_memory,
            )
            if self.config.enable_profiling
            else None
        )

        self.regression_analyzer = None
        if self.config.enable_regression_analysis:
            self.regression_analyzer = initialize_regression_analyzer(
                self.metrics_collector
            )

        self.report_generator = PerformanceReportGenerator()

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []

        # Alert rules setup
        self._setup_default_alert_rules()

        logger.info("Integrated monitoring system initialized")

    def _setup_default_alert_rules(self):
        """Set up default alert rules for common issues."""
        if not self.config.enable_alerts:
            return

        # Database alerts
        self.metrics_collector.add_alert_rule(
            name="High Database Latency",
            metric_name="query_latency_ms",
            source=MetricSource.DATABASE,
            condition="p95 > 100",
            threshold=100,
            severity="warning",
            description="Database query latency exceeds 100ms",
        )

        # WebSocket alerts
        self.metrics_collector.add_alert_rule(
            name="WebSocket Error Rate",
            metric_name="error_rate",
            source=MetricSource.WEBSOCKET,
            condition="avg > 0.05",
            threshold=0.05,
            severity="critical",
            description="WebSocket error rate exceeds 5%",
        )

        # Agent performance alerts
        self.metrics_collector.add_alert_rule(
            name="Slow Inference",
            metric_name="inference_time_ms",
            source=MetricSource.AGENT,
            condition="p95 > 50",
            threshold=50,
            severity="warning",
            description="Agent inference time exceeds 50ms",
        )

        # System resource alerts
        self.metrics_collector.add_alert_rule(
            name="High Memory Usage",
            metric_name="memory_usage_percent",
            source=MetricSource.SYSTEM,
            condition="avg > 80",
            threshold=80,
            severity="critical",
            description="System memory usage exceeds 80%",
        )

        self.metrics_collector.add_alert_rule(
            name="High CPU Usage",
            metric_name="cpu_usage_percent",
            source=MetricSource.SYSTEM,
            condition="avg > 90",
            threshold=90,
            severity="critical",
            description="System CPU usage exceeds 90%",
        )

    async def start(self):
        """Start all monitoring components."""
        if self.is_running:
            logger.warning("Monitoring system already running")
            return

        self.is_running = True
        logger.info("Starting integrated monitoring system...")

        # Start metrics collection
        if self.config.enable_metrics:
            await self.metrics_collector.start()

            # Start metric collection tasks
            self._background_tasks.append(
                asyncio.create_task(self._collect_all_metrics())
            )

        # Start dashboard
        if self.config.enable_dashboard and self.dashboard:
            await self.dashboard.start()

        # Start regression monitoring
        if self.config.enable_regression_analysis:
            self._background_tasks.append(
                asyncio.create_task(self._regression_monitoring_loop())
            )

        # Start automated reporting
        if self.config.enable_automated_reports:
            self._background_tasks.append(
                asyncio.create_task(self._automated_reporting_loop())
            )

        # Start load testing if enabled
        if self.config.enable_load_testing:
            self._background_tasks.append(
                asyncio.create_task(self._load_testing_loop())
            )

        logger.info("Integrated monitoring system started successfully")
        logger.info(
            f"Dashboard available at http://{self.config.dashboard_host}:{self.config.dashboard_port}"
        )

    async def stop(self):
        """Stop all monitoring components."""
        if not self.is_running:
            return

        self.is_running = False
        logger.info("Stopping integrated monitoring system...")

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Stop components
        if self.config.enable_metrics:
            await self.metrics_collector.stop()

        if self.dashboard:
            await self.dashboard.stop()

        # Generate final report
        if self.config.enable_automated_reports:
            await self._generate_final_report()

        logger.info("Integrated monitoring system stopped")

    async def _collect_all_metrics(self):
        """Background task to collect metrics from all sources."""
        while self.is_running:
            try:
                # Collect from all sources
                await asyncio.gather(
                    self.metrics_collector.collect_database_metrics(),
                    self.metrics_collector.collect_websocket_metrics(),
                    self.metrics_collector.collect_agent_metrics(),
                    self.metrics_collector.collect_system_metrics(),
                    return_exceptions=True,
                )

                await asyncio.sleep(5)  # Collect every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(5)

    async def _regression_monitoring_loop(self):
        """Background task for continuous regression monitoring."""
        # First establish baseline if needed
        if not self.regression_analyzer._baselines:
            logger.info("Establishing performance baselines...")
            await self.regression_analyzer.establish_baseline(
                duration_seconds=self.config.regression_baseline_duration
            )

        while self.is_running:
            try:
                await asyncio.sleep(self.config.regression_check_interval)

                # Analyze for regressions
                regressions = await self.regression_analyzer.analyze_regressions()

                if regressions:
                    # Log critical regressions
                    critical_count = sum(
                        1
                        for r in regressions
                        if r.severity
                        in [
                            RegressionSeverity.CRITICAL,
                            RegressionSeverity.MAJOR,
                        ]
                    )

                    if critical_count > 0:
                        logger.error(
                            f"🚨 {critical_count} critical/major performance regressions detected!"
                        )

                        # Generate regression report
                        report = self.regression_analyzer.generate_regression_report(
                            regressions, include_recommendations=True
                        )

                        # Save report
                        report_path = (
                            Path("tests/performance/reports")
                            / f"regression_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                        )
                        report_path.parent.mkdir(exist_ok=True)

                        with open(report_path, "w") as f:
                            f.write(report)

                        logger.info(f"Regression report saved to {report_path}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in regression monitoring: {e}")
                await asyncio.sleep(60)

    async def _automated_reporting_loop(self):
        """Background task for automated report generation."""
        while self.is_running:
            try:
                # Wait for report interval
                await asyncio.sleep(self.config.report_interval_hours * 3600)

                # Generate comprehensive report
                await self.generate_comprehensive_report()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in automated reporting: {e}")
                await asyncio.sleep(3600)

    async def _load_testing_loop(self):
        """Background task for continuous load testing."""
        while self.is_running:
            try:
                for scenario in self.config.load_test_scenarios:
                    if not self.is_running:
                        break

                    logger.info(f"Running load test scenario: {scenario}")

                    if scenario == "database":
                        await self._run_database_load_test()
                    elif scenario == "websocket":
                        await self._run_websocket_load_test()
                    elif scenario == "agent":
                        await self._run_agent_load_test()
                    elif scenario == "coordination":
                        await self._run_coordination_load_test()

                    # Wait between scenarios
                    await asyncio.sleep(300)

                # Wait before next cycle
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in load testing: {e}")
                await asyncio.sleep(600)

    async def _run_database_load_test(self):
        """Run database load test with monitoring."""
        tester = DatabaseLoadTester()

        # Profile the load test
        if self.profiler:
            async with self.profiler.profile_component_async("database", "load_test"):
                await tester.run_comprehensive_test()
        else:
            await tester.run_comprehensive_test()

    async def _run_websocket_load_test(self):
        """Run WebSocket load test with monitoring."""
        test_config = {
            "num_connections": 100,
            "duration": 60,
            "message_rate": 10,
        }

        tester = WebSocketLoadTest(test_config)

        if self.profiler:
            async with self.profiler.profile_component_async("websocket", "load_test"):
                await tester.run()
        else:
            await tester.run()

    async def _run_agent_load_test(self):
        """Run agent simulation load test."""
        framework = AgentSimulationFramework()

        if self.profiler:
            async with self.profiler.profile_component_async("agent", "simulation"):
                await framework.run_scalability_test(agent_counts=[10, 20, 30, 40, 50])
        else:
            await framework.run_scalability_test(agent_counts=[10, 20, 30, 40, 50])

    async def _run_coordination_load_test(self):
        """Run coordination load test."""
        tester = CoordinationLoadTester()

        if self.profiler:
            async with self.profiler.profile_component_async(
                "coordination", "load_test"
            ):
                await tester.run_comprehensive_test()
        else:
            await tester.run_comprehensive_test()

    async def generate_comprehensive_report(self) -> Dict[str, str]:
        """Generate comprehensive performance report."""
        logger.info("Generating comprehensive performance report...")

        reports = {}
        timestamp = datetime.now()

        # Get metrics summary
        metrics_summary = await self.metrics_collector.get_metrics_summary(
            window_seconds=self.config.report_interval_hours * 3600
        )

        # Generate different report formats
        for format in self.config.report_formats:
            if format == "json":
                report_data = {
                    "timestamp": timestamp.isoformat(),
                    "metrics_summary": metrics_summary,
                    "profiling_data": self._get_profiling_summary()
                    if self.profiler
                    else {},
                    "regression_analysis": (
                        self._get_regression_summary()
                        if self.regression_analyzer
                        else {}
                    ),
                    "alerts": list(self.metrics_collector._alert_history)[
                        -100:
                    ],  # Last 100 alerts
                }

                report_path = (
                    Path("tests/performance/reports")
                    / f"comprehensive_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
                )
                report_path.parent.mkdir(exist_ok=True)

                with open(report_path, "w") as f:
                    json.dump(report_data, f, indent=2, default=str)

                reports["json"] = str(report_path)

            elif format == "html":
                html_report = self._generate_html_report(timestamp, metrics_summary)

                report_path = (
                    Path("tests/performance/reports")
                    / f"comprehensive_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.html"
                )

                with open(report_path, "w") as f:
                    f.write(html_report)

                reports["html"] = str(report_path)

            elif format == "markdown":
                md_report = self._generate_markdown_report(timestamp, metrics_summary)

                report_path = (
                    Path("tests/performance/reports")
                    / f"comprehensive_report_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
                )

                with open(report_path, "w") as f:
                    f.write(md_report)

                reports["markdown"] = str(report_path)

        logger.info(f"Generated reports: {', '.join(reports.keys())}")
        return reports

    def _get_profiling_summary(self) -> Dict[str, Any]:
        """Get summary of profiling data."""
        if not self.profiler:
            return {}

        summary = {}
        for component in ["database", "websocket", "agent", "inference"]:
            component_summary = self.profiler.get_component_summary(component)
            if "error" not in component_summary:
                summary[component] = component_summary

        return summary

    def _get_regression_summary(self) -> Dict[str, Any]:
        """Get summary of regression analysis."""
        if not self.regression_analyzer:
            return {}

        return {
            "baselines": len(self.regression_analyzer._baselines),
            "recent_regressions": len(self.regression_analyzer._regression_history),
            "last_analysis": (
                self.regression_analyzer._regression_history[-1].detected_at.isoformat()
                if self.regression_analyzer._regression_history
                else None
            ),
        }

    def _generate_html_report(
        self, timestamp: datetime, metrics_summary: Dict[str, Any]
    ) -> str:
        """Generate HTML format report."""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>FreeAgentics Performance Report - {timestamp.strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .metric-card {{ background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .alert {{ padding: 10px; margin: 5px 0; border-radius: 3px; }}
        .alert-critical {{ background-color: #e74c3c; color: white; }}
        .alert-warning {{ background-color: #f39c12; color: white; }}
        .alert-info {{ background-color: #3498db; color: white; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .chart-container {{ margin: 20px 0; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 FreeAgentics Performance Report</h1>
        <p><strong>Generated:</strong> {timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Analysis Window:</strong> {metrics_summary.get('window_seconds', 0) / 3600:.1f} hours</p>

        <h2>📊 Performance Summary</h2>
        <div class="metric-card">
            <h3>Key Metrics</h3>
            <ul>
                <li><strong>Total Metrics Collected:</strong> {metrics_summary.get('total_points', 0):,}</li>
                <li><strong>Active Sources:</strong> {len(metrics_summary.get('sources', {}))}</li>
                <li><strong>Recent Alerts:</strong> {metrics_summary.get('alert_count', 0)}</li>
            </ul>
        </div>
"""

        # Add source-specific metrics
        for source_name, source_data in metrics_summary.get("sources", {}).items():
            html += """
        <h2>📈 {source_name.title()} Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Current</th>
                <th>Average</th>
                <th>Min</th>
                <th>Max</th>
                <th>P95</th>
                <th>P99</th>
            </tr>
"""
            for metric_key, metric_data in source_data.items():
                metric_data.get("stats", {})
                html += """
            <tr>
                <td>{metric_data.get('name', metric_key)}</td>
                <td>{stats.get('latest', 0):.2f}</td>
                <td>{stats.get('avg', 0):.2f}</td>
                <td>{stats.get('min', 0):.2f}</td>
                <td>{stats.get('max', 0):.2f}</td>
                <td>{stats.get('p95', 0):.2f}</td>
                <td>{stats.get('p99', 0):.2f}</td>
            </tr>
"""
            html += "        </table>\n"

        # Add recent alerts
        recent_alerts = metrics_summary.get("recent_alerts", [])
        if recent_alerts:
            html += """
        <h2>⚠️ Recent Alerts</h2>
"""
            for alert in recent_alerts[-10:]:  # Last 10 alerts
                f"alert-{alert.get('severity', 'info')}"
                html += """
        <div class="alert {severity_class}">
            <strong>{alert.get('rule_name', 'Unknown')}</strong> -
            {alert.get('metric_name', 'Unknown')} {alert.get('condition', '')}
            (value: {alert.get('actual_value', 0):.2f}, threshold: {alert.get('threshold', 0):.2f})
            <span style="float: right;">{alert.get('timestamp', '')}</span>
        </div>
"""

        html += """
    </div>
</body>
</html>
"""
        return html

    def _generate_markdown_report(
        self, timestamp: datetime, metrics_summary: Dict[str, Any]
    ) -> str:
        """Generate Markdown format report."""
        md = """# FreeAgentics Performance Report

**Generated:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Window:** {metrics_summary.get('window_seconds', 0) / 3600:.1f} hours

## Executive Summary

This report provides a comprehensive overview of system performance across all monitored components.

### Key Statistics
- **Total Metrics Collected:** {metrics_summary.get('total_points', 0):,}
- **Active Sources:** {len(metrics_summary.get('sources', {}))}
- **Recent Alerts:** {metrics_summary.get('alert_count', 0)}

## Performance Metrics by Component

"""

        # Add metrics for each source
        for source_name, source_data in metrics_summary.get("sources", {}).items():
            md += f"### {source_name.title()} Performance\n\n"
            md += "| Metric | Current | Average | Min | Max | P95 | P99 |\n"
            md += "|--------|---------|---------|-----|-----|-----|-----|\n"

            for metric_key, metric_data in source_data.items():
                stats = metric_data.get("stats", {})
                md += f"| {metric_data.get('name', metric_key)} | "
                md += f"{stats.get('latest', 0):.2f} | "
                md += f"{stats.get('avg', 0):.2f} | "
                md += f"{stats.get('min', 0):.2f} | "
                md += f"{stats.get('max', 0):.2f} | "
                md += f"{stats.get('p95', 0):.2f} | "
                md += f"{stats.get('p99', 0):.2f} |\n"
            md += "\n"

        # Add profiling summary if available
        if self.profiler:
            profiling_summary = self._get_profiling_summary()
            if profiling_summary:
                md += "## Profiling Analysis\n\n"
                for component, data in profiling_summary.items():
                    md += f"### {component.title()}\n"
                    md += f"- **Profile Count:** {data.get('profile_count', 0)}\n"
                    md += f"- **Average Duration:** {data.get('duration_stats', {}).get('avg', 0):.3f}s\n"
                    md += f"- **CPU Time:** {data.get('cpu_time_stats', {}).get('avg', 0):.3f}s\n"
                    md += f"- **Peak Memory:** {data.get('memory_peak_stats', {}).get('max', 0):.1f}MB\n"

                    bottlenecks = data.get("bottleneck_summary", {})
                    if bottlenecks:
                        md += f"- **Bottlenecks:** {', '.join(f'{k}({v})' for k, v in bottlenecks.items())}\n"
                    md += "\n"

        # Add regression analysis if available
        if self.regression_analyzer and self.regression_analyzer._regression_history:
            md += "## Recent Performance Regressions\n\n"

            recent_regressions = self.regression_analyzer._regression_history[-10:]
            for reg in recent_regressions:
                severity_icon = {
                    "critical": "🔴",
                    "major": "🟠",
                    "moderate": "🟡",
                    "minor": "🟢",
                }.get(reg.severity.value, "⚫")

                md += f"{severity_icon} **{reg.source.value}.{reg.metric_name}** - "
                md += f"{reg.change_percent:.1f}% regression "
                md += f"({reg.baseline_value:.2f} → {reg.current_value:.2f})\n"

        # Add recent alerts
        recent_alerts = metrics_summary.get("recent_alerts", [])
        if recent_alerts:
            md += "\n## Recent Alerts\n\n"
            for alert in recent_alerts[-10:]:
                severity_icon = {
                    "critical": "🚨",
                    "warning": "⚠️",
                    "info": "ℹ️",
                }.get(alert.get("severity", "info"), "📌")

                md += f"{severity_icon} **{alert.get('rule_name', 'Unknown')}** - "
                md += f"{alert.get('metric_name', 'Unknown')} {alert.get('condition', '')} "
                md += f"(value: {alert.get('actual_value', 0):.2f})\n"

        md += "\n---\n*Generated by FreeAgentics Integrated Monitoring System*\n"

        return md

    async def _generate_final_report(self):
        """Generate final report when stopping."""
        try:
            reports = await self.generate_comprehensive_report()
            logger.info(f"Final reports generated: {reports}")
        except Exception as e:
            logger.error(f"Failed to generate final report: {e}")


# CLI Interface
@click.group()
def cli():
    """FreeAgentics Integrated Performance Monitoring CLI."""
    pass


@cli.command()
@click.option(
    "--config", type=click.Path(exists=True), help="Configuration file (YAML)"
)
@click.option("--dashboard/--no-dashboard", default=True, help="Enable dashboard")
@click.option("--port", default=8090, help="Dashboard port")
@click.option("--profiling/--no-profiling", default=True, help="Enable profiling")
@click.option(
    "--regression/--no-regression",
    default=True,
    help="Enable regression analysis",
)
def start(config, dashboard, port, profiling, regression):
    """Start the integrated monitoring system."""
    # Load configuration
    monitoring_config = MonitoringConfig()

    if config:
        with open(config, "r") as f:
            config_data = yaml.safe_load(f)
            for key, value in config_data.items():
                if hasattr(monitoring_config, key):
                    setattr(monitoring_config, key, value)

    # Override with CLI options
    monitoring_config.enable_dashboard = dashboard
    monitoring_config.dashboard_port = port
    monitoring_config.enable_profiling = profiling
    monitoring_config.enable_regression_analysis = regression

    # Create and start system
    system = IntegratedMonitoringSystem(monitoring_config)

    async def run():
        await system.start()

        # Set up signal handlers
        stop_event = asyncio.Event()

        def signal_handler():
            stop_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_event_loop().add_signal_handler(sig, signal_handler)

        # Wait for stop signal
        await stop_event.wait()

        # Stop system
        await system.stop()

    # Run the monitoring system
    asyncio.run(run())


@cli.command()
@click.option(
    "--format",
    type=click.Choice(["json", "html", "markdown", "all"]),
    default="markdown",
    help="Report format",
)
@click.option("--window", default=3600, help="Analysis window in seconds")
def report(format, window):
    """Generate a performance report."""

    async def generate():
        # Initialize minimal system for reporting
        collector = UnifiedMetricsCollector()
        RegressionAnalyzer(collector)

        # Get metrics summary
        summary = await collector.get_metrics_summary(window_seconds=window)

        # Generate report
        timestamp = datetime.now()

        if format == "json" or format == "all":
            report_data = {
                "timestamp": timestamp.isoformat(),
                "window_seconds": window,
                "metrics_summary": summary,
            }

            report_path = (
                Path("tests/performance/reports")
                / f"report_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            )
            report_path.parent.mkdir(exist_ok=True)

            with open(report_path, "w") as f:
                json.dump(report_data, f, indent=2, default=str)

            click.echo(f"JSON report saved to: {report_path}")

        if format == "markdown" or format == "all":
            # Use existing report generator
            generator = PerformanceReportGenerator()

            # Load results and generate report
            results = generator.load_benchmark_results()
            metrics = generator.extract_metrics(results)
            regressions = generator.detect_regressions(metrics)
            charts = generator.generate_performance_charts(metrics)
            report_file = generator.generate_summary_report(
                metrics, regressions, charts
            )

            click.echo(f"Markdown report saved to: {report_file}")

    asyncio.run(generate())


@cli.command()
@click.option("--duration", default=300, help="Baseline duration in seconds")
def baseline(duration):
    """Establish performance baselines."""

    async def establish():
        collector = UnifiedMetricsCollector()
        analyzer = RegressionAnalyzer(collector)

        await collector.start()

        click.echo(f"Establishing baselines for {duration} seconds...")
        baselines = await analyzer.establish_baseline(duration_seconds=duration)

        await collector.stop()

        click.echo(f"Established baselines for {len(baselines)} metrics")
        click.echo("Baselines saved successfully")

    asyncio.run(establish())


if __name__ == "__main__":
    cli()
