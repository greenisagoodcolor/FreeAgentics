"""Real-time Performance Monitoring Dashboard.

This module provides a web-based dashboard for real-time performance monitoring,
visualization, and alerting. It integrates with the unified metrics collector
to display live performance data across all system components.
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
import aiohttp_cors
import numpy as np
from aiohttp import web

from tests.performance.unified_metrics_collector import (
    MetricSource,
    MetricType,
    UnifiedMetricsCollector,
    start_unified_collection,
    unified_collector,
)

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Dashboard configuration settings."""

    host: str = "0.0.0.0"
    port: int = 8090
    update_interval: float = 1.0  # seconds
    history_window: int = 3600  # seconds (1 hour)
    max_connections: int = 100
    enable_alerts: bool = True
    enable_export: bool = True


class MetricsDashboard:
    """Real-time performance monitoring dashboard server."""

    def __init__(self, config: DashboardConfig = None):
        """Initialize the dashboard server."""
        self.config = config or DashboardConfig()
        self.app = web.Application()
        self.websocket_clients: Set[web.WebSocketResponse] = set()
        self.metrics_collector = unified_collector
        self._running = False
        self._update_task: Optional[asyncio.Task] = None

        # Setup routes
        self._setup_routes()

        # Setup CORS
        cors = aiohttp_cors.setup(
            self.app,
            defaults={
                "*": aiohttp_cors.ResourceOptions(
                    allow_credentials=True, expose_headers="*", allow_headers="*", allow_methods="*"
                )
            },
        )

        # Configure CORS on all routes
        for route in list(self.app.router.routes()):
            cors.add(route)

        logger.info(f"Dashboard server initialized on {self.config.host}:{self.config.port}")

    def _setup_routes(self):
        """Setup HTTP routes."""
        # Static files
        static_dir = Path(__file__).parent / "dashboard_static"
        if static_dir.exists():
            self.app.router.add_static("/", static_dir, name="static")

        # API routes
        self.app.router.add_get("/api/metrics/summary", self.handle_metrics_summary)
        self.app.router.add_get(
            "/api/metrics/history/{source}/{metric}", self.handle_metric_history
        )
        self.app.router.add_get("/api/metrics/sources", self.handle_list_sources)
        self.app.router.add_get("/api/alerts", self.handle_get_alerts)
        self.app.router.add_post("/api/alerts/rules", self.handle_add_alert_rule)
        self.app.router.add_get("/api/export/{format}", self.handle_export_metrics)
        self.app.router.add_get("/ws", self.websocket_handler)

        # Health check
        self.app.router.add_get("/health", self.handle_health_check)

        # Dashboard HTML (if no static files)
        self.app.router.add_get("/", self.handle_dashboard_html)

    async def start(self):
        """Start the dashboard server."""
        if self._running:
            logger.warning("Dashboard already running")
            return

        self._running = True

        # Start metrics collection if not already running
        await start_unified_collection()

        # Start update task
        self._update_task = asyncio.create_task(self._update_loop())

        # Start web server
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.config.host, self.config.port)
        await site.start()

        logger.info(f"Dashboard server started at http://{self.config.host}:{self.config.port}")

    async def stop(self):
        """Stop the dashboard server."""
        self._running = False

        # Cancel update task
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        # Close all websocket connections
        for ws in list(self.websocket_clients):
            await ws.close()

        logger.info("Dashboard server stopped")

    async def _update_loop(self):
        """Send periodic updates to websocket clients."""
        while self._running:
            try:
                # Get current metrics
                update_data = await self._prepare_update_data()

                # Send to all connected clients
                if self.websocket_clients:
                    message = json.dumps(
                        {
                            "type": "metrics_update",
                            "timestamp": datetime.now().isoformat(),
                            "data": update_data,
                        }
                    )

                    # Send to all clients, removing disconnected ones
                    disconnected = []
                    for ws in self.websocket_clients:
                        try:
                            await ws.send_str(message)
                        except ConnectionResetError:
                            disconnected.append(ws)

                    # Remove disconnected clients
                    for ws in disconnected:
                        self.websocket_clients.discard(ws)

                await asyncio.sleep(self.config.update_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Update loop error: {e}")
                await asyncio.sleep(self.config.update_interval)

    async def _prepare_update_data(self) -> Dict[str, Any]:
        """Prepare metrics update data for clients."""
        # Get recent metrics summary
        summary = await self.metrics_collector.get_metrics_summary(window_seconds=60)

        # Prepare data for different charts
        update_data = {"summary": summary, "charts": {}}

        # Database metrics
        if MetricSource.DATABASE.value in summary["sources"]:
            db_metrics = summary["sources"][MetricSource.DATABASE.value]
            update_data["charts"]["database"] = {
                "query_latency": self._extract_chart_data(db_metrics, "query_latency_ms"),
                "connection_pool": self._extract_chart_data(db_metrics, "connection_pool_size"),
                "transaction_rate": self._extract_chart_data(db_metrics, "transaction_rate"),
            }

        # WebSocket metrics
        if MetricSource.WEBSOCKET.value in summary["sources"]:
            ws_metrics = summary["sources"][MetricSource.WEBSOCKET.value]
            update_data["charts"]["websocket"] = {
                "connections_rate": self._extract_chart_data(ws_metrics, "connections_per_second"),
                "messages_rate": self._extract_chart_data(ws_metrics, "messages_per_second"),
                "latency": self._extract_chart_data(ws_metrics, "current_latency_ms"),
                "error_rate": self._extract_chart_data(ws_metrics, "error_rate"),
            }

        # Agent metrics
        if MetricSource.AGENT.value in summary["sources"]:
            agent_metrics = summary["sources"][MetricSource.AGENT.value]
            update_data["charts"]["agent"] = {
                "inference_time": self._extract_chart_data(agent_metrics, "inference_time_ms"),
                "active_agents": self._extract_chart_data(agent_metrics, "active_agents"),
                "throughput": self._extract_chart_data(agent_metrics, "agent_throughput"),
                "belief_updates": self._extract_chart_data(agent_metrics, "belief_updates_per_sec"),
            }

        # System metrics
        if MetricSource.SYSTEM.value in summary["sources"]:
            sys_metrics = summary["sources"][MetricSource.SYSTEM.value]
            update_data["charts"]["system"] = {
                "cpu_usage": self._extract_chart_data(sys_metrics, "cpu_usage_percent"),
                "memory_usage": self._extract_chart_data(sys_metrics, "memory_usage_percent"),
                "disk_io_read": self._extract_chart_data(sys_metrics, "disk_read_mb_per_sec"),
                "disk_io_write": self._extract_chart_data(sys_metrics, "disk_write_mb_per_sec"),
            }

        # Add alerts
        update_data["alerts"] = summary.get("recent_alerts", [])

        return update_data

    def _extract_chart_data(self, metrics: Dict[str, Any], metric_name: str) -> Dict[str, Any]:
        """Extract chart-ready data from metrics."""
        for key, data in metrics.items():
            if metric_name in key:
                stats = data["stats"]
                return {
                    "current": stats["latest"],
                    "avg": stats["avg"],
                    "min": stats["min"],
                    "max": stats["max"],
                    "p95": stats["p95"],
                    "p99": stats["p99"],
                }
        return {"current": 0, "avg": 0, "min": 0, "max": 0, "p95": 0, "p99": 0}

    # HTTP Handlers

    async def handle_health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response(
            {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "websocket_clients": len(self.websocket_clients),
                "metrics_collector": "running" if self._running else "stopped",
            }
        )

    async def handle_metrics_summary(self, request: web.Request) -> web.Response:
        """Get metrics summary."""
        window = int(request.query.get("window", 300))
        source = request.query.get("source")

        if source:
            try:
                source = MetricSource(source)
            except ValueError:
                return web.json_response({"error": f"Invalid source: {source}"}, status=400)

        summary = await self.metrics_collector.get_metrics_summary(
            source=source, window_seconds=window
        )

        return web.json_response(summary)

    async def handle_metric_history(self, request: web.Request) -> web.Response:
        """Get historical data for a specific metric."""
        source = request.match_info["source"]
        metric = request.match_info["metric"]
        duration = int(request.query.get("duration", 3600))

        try:
            source_enum = MetricSource(source)
        except ValueError:
            return web.json_response({"error": f"Invalid source: {source}"}, status=400)

        history = self.metrics_collector.get_metric_history(metric, source_enum, duration)

        # Convert to JSON-serializable format
        data = [{"timestamp": ts.isoformat(), "value": value} for ts, value in history]

        return web.json_response(
            {
                "source": source,
                "metric": metric,
                "duration_seconds": duration,
                "data_points": len(data),
                "data": data,
            }
        )

    async def handle_list_sources(self, request: web.Request) -> web.Response:
        """List available metric sources."""
        sources = [source.value for source in MetricSource]
        return web.json_response({"sources": sources})

    async def handle_get_alerts(self, request: web.Request) -> web.Response:
        """Get alert history and rules."""
        # Get alert history from collector
        with self.metrics_collector._lock:
            alerts = list(self.metrics_collector._alert_history)
            rules = list(self.metrics_collector._alert_rules)

        return web.json_response(
            {
                "alerts": alerts,
                "rules": rules,
                "total_alerts": len(alerts),
                "active_rules": len([r for r in rules if r.get("enabled", True)]),
            }
        )

    async def handle_add_alert_rule(self, request: web.Request) -> web.Response:
        """Add a new alert rule."""
        try:
            data = await request.json()

            # Validate required fields
            required = ["name", "metric_name", "source", "condition", "threshold"]
            missing = [f for f in required if f not in data]
            if missing:
                return web.json_response(
                    {"error": f"Missing required fields: {missing}"}, status=400
                )

            # Add the alert rule
            self.metrics_collector.add_alert_rule(
                name=data["name"],
                metric_name=data["metric_name"],
                source=MetricSource(data["source"]),
                condition=data["condition"],
                threshold=float(data["threshold"]),
                window_seconds=data.get("window_seconds", 300),
                severity=data.get("severity", "warning"),
                description=data.get("description", ""),
            )

            return web.json_response(
                {"status": "success", "message": f"Alert rule '{data['name']}' added successfully"}
            )

        except Exception as e:
            return web.json_response({"error": str(e)}, status=400)

    async def handle_export_metrics(self, request: web.Request) -> web.Response:
        """Export metrics in various formats."""
        format = request.match_info["format"]

        if format not in ["json", "prometheus"]:
            return web.json_response({"error": f"Unsupported format: {format}"}, status=400)

        try:
            export_data = await self.metrics_collector.export_metrics(format=format)

            if format == "json":
                return web.json_response(json.loads(export_data))
            else:
                return web.Response(text=export_data, content_type="text/plain")

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """Handle WebSocket connections for real-time updates."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        # Add to clients set
        self.websocket_clients.add(ws)

        try:
            # Send initial data
            initial_data = await self._prepare_update_data()
            await ws.send_str(
                json.dumps(
                    {
                        "type": "initial_data",
                        "timestamp": datetime.now().isoformat(),
                        "data": initial_data,
                    }
                )
            )

            # Keep connection alive
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    # Handle client messages if needed
                    data = json.loads(msg.data)
                    if data.get("type") == "ping":
                        await ws.send_str(
                            json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()})
                        )
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")

        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
        finally:
            # Remove from clients set
            self.websocket_clients.discard(ws)

        return ws

    async def handle_dashboard_html(self, request: web.Request) -> web.Response:
        """Serve the dashboard HTML page."""
        html_content = self._generate_dashboard_html()
        return web.Response(text=html_content, content_type="text/html")

    def _generate_dashboard_html(self) -> str:
        """Generate the dashboard HTML with embedded JavaScript."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FreeAgentics Performance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 1rem 2rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header h1 {
            margin: 0;
            font-size: 1.5rem;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-card h3 {
            margin: 0 0 1rem 0;
            color: #2c3e50;
            font-size: 1.1rem;
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            color: #3498db;
            margin: 0.5rem 0;
        }
        .metric-label {
            color: #7f8c8d;
            font-size: 0.9rem;
        }
        .chart-container {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
        }
        .chart-container h3 {
            margin: 0 0 1rem 0;
            color: #2c3e50;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }
        .status-good { background-color: #27ae60; }
        .status-warning { background-color: #f39c12; }
        .status-critical { background-color: #e74c3c; }
        .alerts-container {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-top: 2rem;
        }
        .alert-item {
            padding: 0.75rem;
            margin: 0.5rem 0;
            border-radius: 4px;
            display: flex;
            align-items: center;
        }
        .alert-warning { background-color: #fff3cd; color: #856404; }
        .alert-critical { background-color: #f8d7da; color: #721c24; }
        .connection-status {
            position: fixed;
            bottom: 1rem;
            right: 1rem;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            font-size: 0.9rem;
        }
        .connected { color: #27ae60; }
        .disconnected { color: #e74c3c; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 FreeAgentics Performance Dashboard</h1>
    </div>

    <div class="container">
        <!-- Key Metrics Grid -->
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Agent Inference Time</h3>
                <div class="metric-value" id="inference-time">--</div>
                <div class="metric-label">milliseconds (p95)</div>
            </div>
            <div class="metric-card">
                <h3>Active Agents</h3>
                <div class="metric-value" id="active-agents">--</div>
                <div class="metric-label">currently running</div>
            </div>
            <div class="metric-card">
                <h3>WebSocket Connections</h3>
                <div class="metric-value" id="ws-connections">--</div>
                <div class="metric-label">messages/sec</div>
            </div>
            <div class="metric-card">
                <h3>Database Queries</h3>
                <div class="metric-value" id="db-queries">--</div>
                <div class="metric-label">queries/sec</div>
            </div>
        </div>

        <!-- Charts -->
        <div class="chart-container">
            <h3>System Performance</h3>
            <canvas id="system-chart" height="100"></canvas>
        </div>

        <div class="chart-container">
            <h3>Agent Performance</h3>
            <canvas id="agent-chart" height="100"></canvas>
        </div>

        <!-- Alerts -->
        <div class="alerts-container">
            <h3>Recent Alerts</h3>
            <div id="alerts-list">
                <p style="color: #7f8c8d;">No alerts</p>
            </div>
        </div>
    </div>

    <div class="connection-status" id="connection-status">
        <span class="disconnected">● Disconnected</span>
    </div>

    <script>
        // WebSocket connection
        let ws = null;
        let reconnectInterval = null;

        // Chart instances
        let systemChart = null;
        let agentChart = null;

        // Data buffers
        const dataBufferSize = 60;  // 60 data points
        const systemData = {
            labels: [],
            cpu: [],
            memory: []
        };
        const agentData = {
            labels: [],
            inferenceTime: [],
            throughput: []
        };

        function connect() {
            const wsUrl = `ws://${window.location.host}/ws`;
            ws = new WebSocket(wsUrl);

            ws.onopen = () => {
                console.log('Connected to dashboard');
                updateConnectionStatus(true);
                if (reconnectInterval) {
                    clearInterval(reconnectInterval);
                    reconnectInterval = null;
                }
            };

            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                if (message.type === 'metrics_update' || message.type === 'initial_data') {
                    updateDashboard(message.data);
                }
            };

            ws.onclose = () => {
                console.log('Disconnected from dashboard');
                updateConnectionStatus(false);
                if (!reconnectInterval) {
                    reconnectInterval = setInterval(connect, 5000);
                }
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }

        function updateConnectionStatus(connected) {
            const status = document.getElementById('connection-status');
            if (connected) {
                status.innerHTML = '<span class="connected">● Connected</span>';
            } else {
                status.innerHTML = '<span class="disconnected">● Disconnected</span>';
            }
        }

        function updateDashboard(data) {
            // Update key metrics
            if (data.charts) {
                if (data.charts.agent) {
                    updateMetric('inference-time', data.charts.agent.inference_time?.p95 || 0);
                    updateMetric('active-agents', data.charts.agent.active_agents?.current || 0);
                }
                if (data.charts.websocket) {
                    updateMetric('ws-connections', data.charts.websocket.messages_rate?.current || 0);
                }
                if (data.charts.database) {
                    updateMetric('db-queries', data.charts.database.transaction_rate?.current || 0);
                }
            }

            // Update charts
            updateCharts(data);

            // Update alerts
            updateAlerts(data.alerts || []);
        }

        function updateMetric(id, value) {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = typeof value === 'number' ? value.toFixed(2) : value;
            }
        }

        function updateCharts(data) {
            const timestamp = new Date().toLocaleTimeString();

            // Update system data
            if (data.charts && data.charts.system) {
                systemData.labels.push(timestamp);
                systemData.cpu.push(data.charts.system.cpu_usage?.current || 0);
                systemData.memory.push(data.charts.system.memory_usage?.current || 0);

                // Keep buffer size
                if (systemData.labels.length > dataBufferSize) {
                    systemData.labels.shift();
                    systemData.cpu.shift();
                    systemData.memory.shift();
                }

                if (systemChart) {
                    systemChart.update();
                }
            }

            // Update agent data
            if (data.charts && data.charts.agent) {
                agentData.labels.push(timestamp);
                agentData.inferenceTime.push(data.charts.agent.inference_time?.avg || 0);
                agentData.throughput.push(data.charts.agent.throughput?.current || 0);

                // Keep buffer size
                if (agentData.labels.length > dataBufferSize) {
                    agentData.labels.shift();
                    agentData.inferenceTime.shift();
                    agentData.throughput.shift();
                }

                if (agentChart) {
                    agentChart.update();
                }
            }
        }

        function updateAlerts(alerts) {
            const alertsList = document.getElementById('alerts-list');
            if (alerts.length === 0) {
                alertsList.innerHTML = '<p style="color: #7f8c8d;">No alerts</p>';
                return;
            }

            const recentAlerts = alerts.slice(-5).reverse();  // Show last 5 alerts
            alertsList.innerHTML = recentAlerts.map(alert => {
                const severity = alert.severity || 'warning';
                const timestamp = new Date(alert.timestamp).toLocaleTimeString();
                return `
                    <div class="alert-item alert-${severity}">
                        <span class="status-indicator status-${severity}"></span>
                        <div>
                            <strong>${alert.rule_name}</strong>: ${alert.metric_name}
                            ${alert.condition} (${alert.actual_value.toFixed(2)})
                            <small style="float: right;">${timestamp}</small>
                        </div>
                    </div>
                `;
            }).join('');
        }

        function initCharts() {
            // System chart
            const systemCtx = document.getElementById('system-chart').getContext('2d');
            systemChart = new Chart(systemCtx, {
                type: 'line',
                data: {
                    labels: systemData.labels,
                    datasets: [{
                        label: 'CPU Usage (%)',
                        data: systemData.cpu,
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.4
                    }, {
                        label: 'Memory Usage (%)',
                        data: systemData.memory,
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'top',
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });

            // Agent chart
            const agentCtx = document.getElementById('agent-chart').getContext('2d');
            agentChart = new Chart(agentCtx, {
                type: 'line',
                data: {
                    labels: agentData.labels,
                    datasets: [{
                        label: 'Inference Time (ms)',
                        data: agentData.inferenceTime,
                        borderColor: '#27ae60',
                        backgroundColor: 'rgba(39, 174, 96, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y'
                    }, {
                        label: 'Throughput (ops/sec)',
                        data: agentData.throughput,
                        borderColor: '#f39c12',
                        backgroundColor: 'rgba(243, 156, 18, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: {
                        mode: 'index',
                        intersect: false,
                    },
                    plugins: {
                        legend: {
                            position: 'top',
                        }
                    },
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {
                                display: true,
                                text: 'Inference Time (ms)'
                            }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Throughput (ops/sec)'
                            },
                            grid: {
                                drawOnChartArea: false,
                            }
                        }
                    }
                }
            });
        }

        // Initialize on page load
        document.addEventListener('DOMContentLoaded', () => {
            initCharts();
            connect();
        });

        // Send periodic ping to keep connection alive
        setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    </script>
</body>
</html>"""


async def start_dashboard(config: DashboardConfig = None):
    """Start the monitoring dashboard."""
    dashboard = MetricsDashboard(config)
    await dashboard.start()
    return dashboard


if __name__ == "__main__":
    # Run dashboard standalone
    async def main():
        config = DashboardConfig(host="0.0.0.0", port=8090, update_interval=1.0)

        dashboard = await start_dashboard(config)

        try:
            # Keep running
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            await dashboard.stop()

    asyncio.run(main())
