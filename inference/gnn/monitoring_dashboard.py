"""
Module for FreeAgentics Active Inference implementation.
"""

import threading
import time
from datetime import datetime, timedelta

import werkzeug.serving
from flask import Flask, jsonify, render_template_string, request
from flask_cors import CORS

from .metrics_collector import get_metrics_collector
from .monitoring import get_logger, get_monitor

"""
Monitoring Dashboard for GNN Processing
This module provides a web-based dashboard for real-time monitoring
and visualization of GNN processing statistics.
"""
try:
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
logger = get_logger().logger


class MonitoringDashboard:
    """
    Web-based monitoring dashboard for GNN processing.
    Provides real-time visualization of:
    - Processing statistics
    - Performance metrics
    - System resource usage
    - Error tracking
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 5000, update_interval: int = 5) -> None:
        """
        Initialize monitoring dashboard.
        Args:
            host: Dashboard host address
            port: Dashboard port
            update_interval: Update interval in seconds
        """
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for the monitoring dashboard")
        self.host = host
        self.port = port
        self.update_interval = update_interval
        self.app = Flask(__name__)
        CORS(self.app)
        self.monitor = get_monitor()
        self.collector = get_metrics_collector()
        self._setup_routes()
        self._running = False
        self._update_thread = None

    def _setup_routes(self):
        """Setup Flask routes"""

        @self.app.route("/")
        def index():
            """Dashboard homepage"""
            return render_template_string(DASHBOARD_HTML)

        @self.app.route("/api/stats/realtime")
        def realtime_stats():
            """Get real-time statistics"""
            stats = {
                "performance": self.monitor.get_statistics(),
                "metrics": self.collector.get_real_time_stats(),
                "timestamp": datetime.utcnow().isoformat(),
            }
            return jsonify(stats)

        @self.app.route("/api/stats/historical")
        def historical_stats():
            """Get historical statistics"""
            hours = int(request.args.get("hours", 24))
            group_by = request.args.get("group_by", "hour")
            start_time = datetime.utcnow() - timedelta(hours=hours)
            stats = self.collector.db.get_aggregated_stats(start_time=start_time, group_by=group_by)
            return jsonify(stats)

        @self.app.route("/api/metrics/graphs")
        def graph_metrics():
            """Get recent graph processing metrics"""
            limit = int(request.args.get("limit", 100))
            metrics = self.collector.db.query_graph_metrics(limit=limit)
            return jsonify(metrics)

        @self.app.route("/api/system/resources")
        def system_resources():
            """Get system resource usage"""
            system_stats = self.monitor.get_statistics("_system")
            return jsonify(
                {
                    "cpu": {
                        "current": system_stats.get("cpu_percent", {}).get("mean", 0),
                        "max": system_stats.get("cpu_percent", {}).get("max", 0),
                    },
                    "memory": {
                        "current": system_stats.get("memory_mb", {}).get("mean", 0),
                        "max": system_stats.get("memory_mb", {}).get("max", 0),
                    },
                    "gpu": {
                        "memory": (
                            system_stats.get("gpu_memory_mb", {}).get("mean", 0)
                            if "gpu_memory_mb" in system_stats
                            else None
                        )
                    },
                }
            )

        @self.app.route("/api/alerts")
        def get_alerts():
            """Get recent alerts"""
            return jsonify([])

    def start(self):
        """Start the dashboard"""
        self._running = True

        def run_app():
            werkzeug.serving.run_simple(
                self.host, self.port, self.app, use_reloader=False, use_debugger=False
            )

        app_thread = threading.Thread(target=run_app)
        app_thread.daemon = True
        app_thread.start()
        logger.info(f"Monitoring dashboard started at http://{self.host}:{self.port}")

    def stop(self):
        """Stop the dashboard"""
        self._running = False
        logger.info("Monitoring dashboard stopped")
        return None


if __name__ == "__main__":
    dashboard = MonitoringDashboard()
    if dashboard:
        dashboard.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            dashboard.stop()
DASHBOARD_HTML = "<html><body><h1>GNN Monitor</h1></body></html>"
