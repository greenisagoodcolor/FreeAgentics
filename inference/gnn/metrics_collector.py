"""
Module for FreeAgentics Active Inference implementation.
"""

import json
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .monitoring import get_logger

"""
Metrics Collection System for GNN Processing
This module provides comprehensive metrics collection and aggregation
for GNN processing operations.
"""
logger = get_logger().logger


@dataclass
class GraphMetrics:
    """Metrics for graph processing"""

    graph_id: str
    num_nodes: int
    num_edges: int
    avg_degree: float
    density: float
    processing_time: float
    model_architecture: str
    task_type: str
    success: bool
    error_message: Optional[str] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelMetrics:
    """Metrics for model performance"""

    model_id: str
    architecture: str
    parameters: int
    inference_time: float
    memory_usage_mb: float
    accuracy: Optional[float] = None
    loss: Optional[float] = None
    throughput_graphs_per_sec: Optional[float] = None


@dataclass
class SystemMetrics:
    """System-wide metrics"""

    timestamp: datetime
    active_jobs: int
    queued_jobs: int
    completed_jobs: int
    failed_jobs: int
    avg_queue_time: float
    avg_processing_time: float
    total_graphs_processed: int
    total_nodes_processed: int
    total_edges_processed: int
    cpu_usage_percent: float
    memory_usage_mb: float
    gpu_usage_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None


class MetricsDatabase:
    """

    SQLite-based metrics storage for persistent tracking.
    Provides efficient storage and querying of metrics data.
    """

    def __init__(self, db_path: str = "gnn_metrics.db") -> None:
        """
        Initialize metrics database.
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "\n                CREATE TABLE IF NOT EXISTS graph_metrics (\n                    id INTEGER PRIMARY KEY AUTOINCREMENT, \n                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, \n                    graph_id TEXT NOT NULL, \n                    num_nodes INTEGER NOT NULL, \n                    num_edges INTEGER NOT NULL, \n                    avg_degree REAL NOT NULL, \n                    density REAL NOT NULL, \n                    processing_time REAL NOT NULL, \n                    model_architecture TEXT NOT NULL, \n                    task_type TEXT NOT NULL, \n                    success BOOLEAN NOT NULL, \n                    error_message TEXT, \n                    additional_metrics TEXT\n                )\n            "
            )
            cursor.execute(
                "\n                CREATE TABLE IF NOT EXISTS model_metrics (\n                    id INTEGER PRIMARY KEY AUTOINCREMENT, \n                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, \n                    model_id TEXT NOT NULL, \n                    architecture TEXT NOT NULL, \n                    parameters INTEGER NOT NULL, \n                    inference_time REAL NOT NULL, \n                    memory_usage_mb REAL NOT NULL, \n                    accuracy REAL, \n                    loss REAL, \n                    throughput_graphs_per_sec REAL\n                )\n            "
            )
            cursor.execute(
                "\n                CREATE TABLE IF NOT EXISTS system_metrics (\n                    id INTEGER PRIMARY KEY AUTOINCREMENT, \n                    timestamp DATETIME NOT NULL, \n                    active_jobs INTEGER NOT NULL, \n                    queued_jobs INTEGER NOT NULL, \n                    completed_jobs INTEGER NOT NULL, \n                    failed_jobs INTEGER NOT NULL, \n                    avg_queue_time REAL NOT NULL, \n                    avg_processing_time REAL NOT NULL, \n                    total_graphs_processed INTEGER NOT NULL, \n                    total_nodes_processed INTEGER NOT NULL, \n                    total_edges_processed INTEGER NOT NULL, \n                    cpu_usage_percent REAL NOT NULL, \n                    memory_usage_mb REAL NOT NULL, \n                    gpu_usage_percent REAL, \n                    gpu_memory_mb REAL\n                )\n            "
            )
            cursor.execute(
                "\n                CREATE INDEX IF NOT EXISTS idx_graph_metrics_timestamp\n                ON graph_metrics(timestamp)\n            "
            )
            cursor.execute(
                "\n                CREATE INDEX IF NOT EXISTS idx_graph_metrics_graph_id\n                ON graph_metrics(graph_id)\n            "
            )
            cursor.execute(
                "\n                CREATE INDEX IF NOT EXISTS idx_model_metrics_timestamp\n                ON model_metrics(timestamp)\n            "
            )
            cursor.execute(
                "\n                CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp\n                ON system_metrics(timestamp)\n            "
            )
            conn.commit()

    def insert_graph_metrics(self, metrics: GraphMetrics) -> None:
        """Insert graph processing metrics"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO graph_metrics (
                        graph_id,
                        num_nodes,
                        num_edges,
                        avg_degree,
                        density,
                        processing_time,
                        model_architecture,
                        task_type,
                        success,
                        error_message,
                        additional_metrics
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metrics.graph_id,
                        metrics.num_nodes,
                        metrics.num_edges,
                        metrics.avg_degree,
                        metrics.density,
                        metrics.processing_time,
                        metrics.model_architecture,
                        metrics.task_type,
                        metrics.success,
                        metrics.error_message,
                        json.dumps(metrics.additional_metrics),
                    ),
                )
                conn.commit()

    def insert_model_metrics(self, metrics: ModelMetrics) -> None:
        """Insert model performance metrics"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO model_metrics (
                        model_id,
                        architecture,
                        parameters,
                        inference_time,
                        memory_usage_mb,
                        accuracy,
                        loss,
                        throughput_graphs_per_sec
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metrics.model_id,
                        metrics.architecture,
                        metrics.parameters,
                        metrics.inference_time,
                        metrics.memory_usage_mb,
                        metrics.accuracy,
                        metrics.loss,
                        metrics.throughput_graphs_per_sec,
                    ),
                )
                conn.commit()

    def insert_system_metrics(self, metrics: SystemMetrics) -> None:
        """Insert system-wide metrics"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO system_metrics (
                        timestamp,
                        active_jobs,
                        queued_jobs,
                        completed_jobs,
                        failed_jobs,
                        avg_queue_time,
                        avg_processing_time,
                        total_graphs_processed,
                        total_nodes_processed,
                        total_edges_processed,
                        cpu_usage_percent,
                        memory_usage_mb,
                        gpu_usage_percent,
                        gpu_memory_mb
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        metrics.timestamp,
                        metrics.active_jobs,
                        metrics.queued_jobs,
                        metrics.completed_jobs,
                        metrics.failed_jobs,
                        metrics.avg_queue_time,
                        metrics.avg_processing_time,
                        metrics.total_graphs_processed,
                        metrics.total_nodes_processed,
                        metrics.total_edges_processed,
                        metrics.cpu_usage_percent,
                        metrics.memory_usage_mb,
                        metrics.gpu_usage_percent,
                        metrics.gpu_memory_mb,
                    ),
                )
                conn.commit()

    def query_graph_metrics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        graph_id: Optional[str] = None,
        model_architecture: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query graph metrics with filters"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = "SELECT * FROM graph_metrics WHERE 1=1"
            params = []
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            if graph_id:
                query += " AND graph_id = ?"
                params.append(graph_id)
            if model_architecture:
                query += " AND model_architecture = ?"
                params.append(model_architecture)
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            cursor.execute(query, params)
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                if result["additional_metrics"]:
                    result["additional_metrics"] = json.loads(
                        result["additional_metrics"])
                results.append(result)
            return results

    def get_aggregated_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        group_by: str = "hour",
    ) -> List[Dict[str, Any]]:
        """Get aggregated statistics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if group_by == "hour":
                date_format = "%Y-%m-%d %H:00:00"
            elif group_by == "day":
                date_format = "%Y-%m-%d"
            else:
                date_format = "%Y-%m-%d %H:00:00"
            query = f"""
                SELECT
                    strftime('{date_format}', timestamp) as period,
                    COUNT(*) as total_operations,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_operations,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed_operations,
                    AVG(processing_time) as avg_processing_time,
                    MIN(processing_time) as min_processing_time,
                    MAX(processing_time) as max_processing_time,
                    SUM(num_nodes) as total_nodes,
                    SUM(num_edges) as total_edges,
                    AVG(density) as avg_density
                FROM graph_metrics
                WHERE 1=1
            """
            params = []
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            query += " GROUP BY period ORDER BY period DESC"
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]


class MetricsCollector:
    """
    Centralized metrics collection service.
    Collects, aggregates, and stores metrics from all GNN processing operations.
    """

    def __init__(
        self,
        db_path: str = "gnn_metrics.db",
        buffer_size: int = 100,
        flush_interval: int = 60,
    ) -> None:
        """
        Initialize metrics collector.
        Args:
            db_path: Path to metrics database
            buffer_size: Size of metrics buffer before flush
            flush_interval: Interval in seconds to flush metrics
        """
        self.db = MetricsDatabase(db_path)
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval
        self._graph_metrics_buffer = []
        self._model_metrics_buffer = []
        self._system_metrics_buffer = []
        self._lock = threading.Lock()
        self._running = True
        self._flush_thread = threading.Thread(target=self._flush_loop)
        self._flush_thread.daemon = True
        self._flush_thread.start()
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)

    def collect_graph_metrics(self, metrics: GraphMetrics) -> None:
        """Collect graph processing metrics"""
        with self._lock:
            self._graph_metrics_buffer.append(metrics)
            self.counters["total_graphs"] += 1
            self.counters["total_nodes"] += metrics.num_nodes
            self.counters["total_edges"] += metrics.num_edges
            if metrics.success:
                self.counters["successful_operations"] += 1
            else:
                self.counters["failed_operations"] += 1
            self.timers["processing_times"].append(metrics.processing_time)
            if len(self._graph_metrics_buffer) >= self.buffer_size:
                self._flush_graph_metrics()

    def collect_model_metrics(self, metrics: ModelMetrics) -> None:
        """Collect model performance metrics"""
        with self._lock:
            self._model_metrics_buffer.append(metrics)
            self.timers["inference_times"].append(metrics.inference_time)
            if len(self._model_metrics_buffer) >= self.buffer_size:
                self._flush_model_metrics()

    def collect_system_metrics(self, metrics: SystemMetrics) -> None:
        """Collect system-wide metrics"""
        with self._lock:
            self._system_metrics_buffer.append(metrics)
            if len(self._system_metrics_buffer) >= self.buffer_size:
                self._flush_system_metrics()

    def _flush_graph_metrics(self) -> None:
        """Flush graph metrics buffer to database"""
        if not self._graph_metrics_buffer:
            return
        try:
            for metrics in self._graph_metrics_buffer:
                self.db.insert_graph_metrics(metrics)
            logger.info(
                f"Flushed {len(self._graph_metrics_buffer)} graph metrics")
            self._graph_metrics_buffer.clear()
        except Exception as e:
            logger.error(f"Error flushing graph metrics: {e}")

    def _flush_model_metrics(self) -> None:
        """Flush model metrics buffer to database"""
        if not self._model_metrics_buffer:
            return
        try:
            for metrics in self._model_metrics_buffer:
                self.db.insert_model_metrics(metrics)
            logger.info(
                f"Flushed {len(self._model_metrics_buffer)} model metrics")
            self._model_metrics_buffer.clear()
        except Exception as e:
            logger.error(f"Error flushing model metrics: {e}")

    def _flush_system_metrics(self) -> None:
        """Flush system metrics buffer to database"""
        if not self._system_metrics_buffer:
            return
        try:
            for metrics in self._system_metrics_buffer:
                self.db.insert_system_metrics(metrics)
            logger.info(
                f"Flushed {len(self._system_metrics_buffer)} system metrics")
            self._system_metrics_buffer.clear()
        except Exception as e:
            logger.error(f"Error flushing system metrics: {e}")

    def _flush_loop(self) -> None:
        """Background thread to periodically flush metrics"""
        while self._running:
            time.sleep(self.flush_interval)
            with self._lock:
                self._flush_graph_metrics()
                self._flush_model_metrics()
                self._flush_system_metrics()

    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time statistics"""
        with self._lock:
            stats = {"counters": dict(self.counters), "timers": {}}
            for name, values in self.timers.items():
                if values:
                    stats["timers"][name] = {
                        "count": len(values),
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "percentiles": {
                            "50": np.percentile(values, 50),
                            "90": np.percentile(values, 90),
                            "99": np.percentile(values, 99),
                        },
                    }
            return stats

    def generate_report(
        self,
        output_path: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> None:
        """Generate comprehensive metrics report"""
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "period": {
                "start": start_time.isoformat() if start_time else "all_time",
                "end": end_time.isoformat() if end_time else "now",
            },
        }
        daily_stats = self.db.get_aggregated_stats(start_time, end_time, "day")
        hourly_stats = self.db.get_aggregated_stats(
            start_time, end_time, "hour")
        report["daily_statistics"] = daily_stats
        report["hourly_statistics"] = hourly_stats
        report["real_time_stats"] = self.get_real_time_stats()
        failures = self.db.query_graph_metrics(
            start_time=start_time, end_time=end_time, limit=100)
        failures = [f for f in failures if not f["success"]]
        report["recent_failures"] = failures[:10]
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Metrics report saved to: {output_path}")

    def stop(self) -> None:
        """Stop the metrics collector"""
        self._running = False
        with self._lock:
            self._flush_graph_metrics()
            self._flush_model_metrics()
            self._flush_system_metrics()
        if self._flush_thread.is_alive():
            self._flush_thread.join(timeout=5)


_collector = None


def get_metrics_collector(db_path: str = "gnn_metrics.db") -> MetricsCollector:
    """Get or create metrics collector instance"""
    global _collector
    if _collector is None:
        _collector = MetricsCollector(db_path)
    return _collector


if __name__ == "__main__":
    collector = get_metrics_collector()
    graph_metrics = GraphMetrics(
        graph_id="graph_001",
        num_nodes=100,
        num_edges=250,
        avg_degree=5.0,
        density=0.05,
        processing_time=1.23,
        model_architecture="GCN",
        task_type="node_classification",
        success=True,
    )
    collector.collect_graph_metrics(graph_metrics)
    model_metrics = ModelMetrics(
        model_id="model_gcn_v1",
        architecture="GCN",
        parameters=50000,
        inference_time=0.05,
        memory_usage_mb=256.5,
        accuracy=0.92,
    )
    collector.collect_model_metrics(model_metrics)
    stats = collector.get_real_time_stats()
    print("Real-time stats:", json.dumps(stats, indent=2))
    collector.generate_report("metrics_report.json")
    collector.stop()
