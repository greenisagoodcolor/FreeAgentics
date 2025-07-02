"""
Module for FreeAgentics Active Inference implementation.
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import psutil
import torch

"""
Monitoring and Logging System for GNN Processing Core
This module provides comprehensive monitoring, logging, and performance tracking
for all GNN processing operations.
"""
# Configure module logger
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""

    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_used_mb: float
    cpu_percent: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class ProcessingStats:
    """Statistics for processing operations"""

    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    total_processing_time: float = 0.0
    total_graphs_processed: int = 0
    total_nodes_processed: int = 0
    total_edges_processed: int = 0
    avg_processing_time: float = 0.0
    peak_memory_usage_mb: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)


class GNNLogger:
    """
    Enhanced logger for GNN processing with structured logging support.
    Features:
    - Structured JSON logging
    - Context-aware logging
    - Performance tracking
    - Error aggregation
    """

    def __init__(
        self,
        name: str = "gnn_processing",
        log_dir: Optional[str] = None,
        level: int = logging.INFO,
        enable_console: bool = True,
        enable_file: bool = True,
        max_file_size_mb: int = 100,
        backup_count: int = 5,
    ) -> None:
        """
        Initialize GNN logger.
        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
            enable_console: Enable console output
            enable_file: Enable file output
            max_file_size_mb: Maximum log file size
            backup_count: Number of backup files to keep
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear existing handlers
        # Create formatters
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
        )
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        # File handler
        if enable_file and log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                log_path / f"{name}.log",
                maxBytes=max_file_size_mb * 1024 * 1024,
                backupCount=backup_count,
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            # JSON file handler for structured logs
            json_handler = RotatingFileHandler(
                log_path / f"{name}_structured.json",
                maxBytes=max_file_size_mb * 1024 * 1024,
                backupCount=backup_count,
            )
            json_handler.setFormatter(JsonFormatter())
            self.logger.addHandler(json_handler)

    def log_operation(
            self,
            operation: str,
            status: str,
            duration: Optional[float] = None,
            **kwargs: Any) -> None:
        """Log an operation with structured data"""
        log_data = {
            "operation": operation,
            "status": status,
            "timestamp": datetime.utcnow().isoformat(),
            "duration_seconds": duration,
            **kwargs,
        }
        if status == "success":
            self.logger.info(
                f"{operation} completed", extra={
                    "structured": log_data})
        elif status == "error":
            self.logger.error(
                f"{operation} failed", extra={
                    "structured": log_data})
        else:
            self.logger.warning(
                f"{operation} status: {status}", extra={
                    "structured": log_data})

    def log_performance(self, metrics: PerformanceMetrics) -> None:
        """Log performance metrics"""
        self.logger.info(
            f"Performance: {
                metrics.operation}", extra={
                "structured": metrics.to_dict()})

    def log_error(
            self,
            error: Exception,
            operation: str,
            **context: Any) -> None:
        """Log error with context"""
        error_data = {
            "operation": operation,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.utcnow().isoformat(),
            **context,
        }
        self.logger.error(
            f"Error in {operation}: {error}",
            extra={"structured": error_data},
            exc_info=True,
        )


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        # Add structured data if available
        if hasattr(record, "structured"):
            log_data["data"] = record.structured
        # Add exception info if available
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data)


class PerformanceMonitor:
    """
    Monitors and tracks performance metrics for GNN operations.
    Features:
    - Real-time performance tracking
    - Resource usage monitoring
    - Historical metrics storage
    - Alerting on thresholds
    """

    def __init__(
        self,
        window_size: int = 1000,
        alert_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initialize performance monitor.
        Args:
            window_size: Size of sliding window for metrics
            alert_thresholds: Thresholds for alerting
        """
        self.window_size = window_size
        self.metrics_history: defaultdict[str, deque[PerformanceMetrics]] = defaultdict(
            lambda: deque(maxlen=window_size))
        self.alert_thresholds = alert_thresholds or {
            "memory_mb": 8192,  # 8GB
            "processing_time": 300,  # 5 minutes
            "cpu_percent": 90,
            "gpu_memory_mb": 16384,  # 16GB
        }
        self.alert_callbacks: List[Callable[[
            str, float, PerformanceMetrics], None]] = []
        self._lock = threading.Lock()
        # Start resource monitoring thread
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_resources)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def start_operation(self, operation: str) -> "OperationContext":
        """Start monitoring an operation"""
        return OperationContext(self, operation)

    def record_metrics(self, metrics: PerformanceMetrics) -> None:
        """Record performance metrics"""
        with self._lock:
            self.metrics_history[metrics.operation].append(metrics)
            # Check thresholds
            self._check_alerts(metrics)

    def _check_alerts(self, metrics: PerformanceMetrics) -> None:
        """Check if metrics exceed thresholds"""
        alerts = []
        if metrics.memory_used_mb > self.alert_thresholds["memory_mb"]:
            alerts.append(("memory", metrics.memory_used_mb))
        if metrics.duration > self.alert_thresholds["processing_time"]:
            alerts.append(("processing_time", metrics.duration))
        if metrics.cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append(("cpu", metrics.cpu_percent))
        if metrics.gpu_memory_mb and metrics.gpu_memory_mb > self.alert_thresholds[
                "gpu_memory_mb"]:
            alerts.append(("gpu_memory", metrics.gpu_memory_mb))
        # Trigger callbacks
        for alert_type, value in alerts:
            for callback in self.alert_callbacks:
                callback(alert_type, value, metrics)

    def add_alert_callback(
        self, callback: Callable[[str, float, PerformanceMetrics], None]
    ) -> None:
        """Add callback for alerts"""
        self.alert_callbacks.append(callback)

    def get_statistics(
            self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics"""
        with self._lock:
            if operation:
                metrics = list(self.metrics_history.get(operation, []))
            else:
                metrics = []
                for op_metrics in self.metrics_history.values():
                    metrics.extend(op_metrics)
            if not metrics:
                return {}
            durations = [m.duration for m in metrics]
            memory_usage = [m.memory_used_mb for m in metrics]
            cpu_usage = [m.cpu_percent for m in metrics]
            stats = {
                "count": len(metrics),
                "duration": {
                    "mean": np.mean(durations),
                    "std": np.std(durations),
                    "min": np.min(durations),
                    "max": np.max(durations),
                    "percentiles": {
                        "50": np.percentile(
                            durations,
                            50),
                        "90": np.percentile(
                            durations,
                            90),
                        "99": np.percentile(
                            durations,
                            99),
                    },
                },
                "memory_mb": {
                    "mean": np.mean(memory_usage),
                    "max": np.max(memory_usage),
                    "min": np.min(memory_usage),
                },
                "cpu_percent": {
                    "mean": np.mean(cpu_usage),
                    "max": np.max(cpu_usage)},
            }
            # Add GPU stats if available
            gpu_memory = [m.gpu_memory_mb for m in metrics if m.gpu_memory_mb]
            if gpu_memory:
                stats["gpu_memory_mb"] = {
                    "mean": np.mean(gpu_memory),
                    "max": np.max(gpu_memory),
                }
            return stats

    def _monitor_resources(self) -> None:
        """Background thread to monitor system resources"""
        while self._monitoring:
            try:
                # Monitor CPU
                cpu_percent = psutil.cpu_percent(interval=1)
                # Monitor memory
                memory = psutil.virtual_memory()
                # Monitor GPU if available
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                    gpu_utilization = torch.cuda.utilization()
                else:
                    gpu_memory = None
                    gpu_utilization = None
                # Store system metrics
                system_metrics = PerformanceMetrics(
                    operation="_system",
                    start_time=time.time(),
                    end_time=time.time(),
                    duration=0,
                    memory_used_mb=memory.used / 1024 / 1024,
                    cpu_percent=cpu_percent,
                    gpu_memory_mb=gpu_memory,
                    gpu_utilization=gpu_utilization,
                )
                with self._lock:
                    self.metrics_history["_system"].append(system_metrics)
                time.sleep(5)  # Monitor every 5 seconds
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(10)

    def stop(self) -> None:
        """Stop monitoring"""
        self._monitoring = False
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)


class OperationContext:
    """Context manager for monitoring operations"""

    def __init__(self, monitor: PerformanceMonitor, operation: str) -> None:
        self.monitor = monitor
        self.operation = operation
        self.start_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.start_cpu_time: Optional[Any] = None

    def __enter__(self) -> "OperationContext":
        self.start_time = time.time()
        # Get memory usage
        process = psutil.Process()
        self.start_memory = process.memory_info().rss / 1024 / 1024
        # Get CPU time
        self.start_cpu_time = process.cpu_times()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        end_time = time.time()
        if self.start_time is None:
            return
        duration = end_time - self.start_time
        # Get resource usage
        process = psutil.Process()
        end_memory = process.memory_info().rss / 1024 / 1024
        if self.start_memory is None:
            return
        memory_used = end_memory - self.start_memory
        # Calculate CPU usage
        end_cpu_time = process.cpu_times()
        if self.start_cpu_time is None:
            return
        cpu_time = (end_cpu_time.user - self.start_cpu_time.user) + (
            end_cpu_time.system - self.start_cpu_time.system
        )
        cpu_percent = (cpu_time / duration) * 100 if duration > 0 else 0
        # Get GPU metrics if available
        gpu_memory = None
        gpu_utilization = None
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_utilization = torch.cuda.utilization()
        # Record metrics
        metrics = PerformanceMetrics(
            operation=self.operation,
            start_time=self.start_time,
            end_time=end_time,
            duration=duration,
            memory_used_mb=memory_used,
            cpu_percent=cpu_percent,
            gpu_memory_mb=gpu_memory,
            gpu_utilization=gpu_utilization,
        )
        self.monitor.record_metrics(metrics)


class MetricsVisualizer:
    """

    Visualizes performance metrics and statistics.
    Provides various visualization options for monitoring data.
    """

    def __init__(self, monitor: PerformanceMonitor) -> None:
        """
        Initialize visualizer.
        Args:
            monitor: Performance monitor instance
        """
        self.monitor = monitor

    def plot_operation_timeline(
            self,
            operation: str,
            metric: str = "duration",
            save_path: Optional[str] = None) -> None:
        """Plot timeline of operation metrics"""
        try:
            import matplotlib.dates as mdates
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available for visualization")
            return
        metrics = list(self.monitor.metrics_history.get(operation, []))
        if not metrics:
            logger.warning(f"No metrics found for operation: {operation}")
            return
        # Extract data
        timestamps = [m.start_time for m in metrics]
        if metric == "duration":
            values = [m.duration for m in metrics]
            ylabel = "Duration (seconds)"
        elif metric == "memory":
            values = [m.memory_used_mb for m in metrics]
            ylabel = "Memory Usage (MB)"
        elif metric == "cpu":
            values = [m.cpu_percent for m in metrics]
            ylabel = "CPU Usage (%)"
        else:
            logger.error(f"Unknown metric: {metric}")
            return
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, values, marker="o", markersize=4)
        plt.xlabel("Time")
        plt.ylabel(ylabel)
        plt.title(f"{operation} - {metric.capitalize()} Over Time")
        plt.grid(True, alpha=0.3)
        # Format x-axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.gcf().autofmt_xdate()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def plot_resource_usage(
        self,
        time_window: int = 3600,  # 1 hour
        save_path: Optional[str] = None,
    ) -> None:
        """Plot system resource usage"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Matplotlib not available for visualization")
            return
        # Get system metrics
        system_metrics = list(self.monitor.metrics_history.get("_system", []))
        if not system_metrics:
            logger.warning("No system metrics available")
            return
        # Filter by time window
        current_time = time.time()
        system_metrics = [
            m for m in system_metrics if current_time -
            m.start_time <= time_window]
        if not system_metrics:
            logger.warning("No recent system metrics")
            return
        # Extract data
        timestamps = [m.start_time for m in system_metrics]
        cpu_usage = [m.cpu_percent for m in system_metrics]
        memory_usage = [m.memory_used_mb for m in system_metrics]
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        # CPU usage
        ax1.plot(timestamps, cpu_usage, "b-", label="CPU %")
        ax1.set_ylabel("CPU Usage (%)")
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        # Memory usage
        ax2.plot(timestamps, memory_usage, "r-", label="Memory")
        ax2.set_ylabel("Memory Usage (MB)")
        ax2.set_xlabel("Time")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        # Add GPU if available
        gpu_memory = [
            m.gpu_memory_mb for m in system_metrics if m.gpu_memory_mb]
        if gpu_memory:
            ax3 = ax2.twinx()
            gpu_timestamps = [
                m.start_time for m in system_metrics if m.gpu_memory_mb]
            ax3.plot(gpu_timestamps, gpu_memory, "g-", label="GPU Memory")
            ax3.set_ylabel("GPU Memory (MB)", color="g")
            ax3.tick_params(axis="y", labelcolor="g")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def generate_report(self, output_path: str) -> None:
        """Generate comprehensive performance report"""
        report = {"timestamp": datetime.utcnow().isoformat(), "operations": {}}
        # Get statistics for each operation
        for operation in self.monitor.metrics_history:
            if operation == "_system":
                continue
            stats = self.monitor.get_statistics(operation)
            if stats:
                report["operations"][operation] = stats
        # Get system statistics
        system_stats = self.monitor.get_statistics("_system")
        if system_stats:
            report["system"] = system_stats
        # Save report
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Performance report saved to: {output_path}")


# Global instances
_logger = None
_monitor = None


def get_logger(name: str = "gnn_processing",
               log_dir: Optional[str] = None) -> GNNLogger:
    """Get or create logger instance"""
    global _logger
    if _logger is None:
        _logger = GNNLogger(name=name, log_dir=log_dir)
    return _logger


def get_monitor() -> PerformanceMonitor:
    """Get or create performance monitor instance"""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor


# Decorators for easy monitoring
def monitor_performance(
    operation: str = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to monitor function performance"""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            op_name = operation or f"{func.__module__}.{func.__name__}"
            monitor = get_monitor()
            with monitor.start_operation(op_name):
                result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator


def log_operation(
        operation: str = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to log operation execution"""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            op_name = operation or f"{func.__module__}.{func.__name__}"
            logger = get_logger()
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.log_operation(op_name, "success", duration)
                return result
            except Exception as e:
                logger.log_error(e, op_name)
                raise

        return wrapper

    return decorator


# Example usage
if __name__ == "__main__":
    # Initialize logger and monitor
    logger = get_logger(log_dir="./logs")
    monitor = get_monitor()

    # Add alert callback
    def alert_callback(
            alert_type: str,
            value: float,
            metrics: PerformanceMetrics) -> None:
        logger.logger.warning(
            f"Alert: {alert_type} exceeded threshold with value {value}")

    monitor.add_alert_callback(alert_callback)

    # Example monitored operation
    @monitor_performance("example_operation")
    @log_operation("example_operation")
    def example_operation() -> torch.Tensor:
        # Simulate some work
        time.sleep(0.1)
        data = torch.randn(1000, 1000)
        result = torch.matmul(data, data.T)
        return result

    # Run example
    result = example_operation()
    # Get statistics
    stats = monitor.get_statistics("example_operation")
    print("Statistics:", json.dumps(stats, indent=2))
    # Generate visualizations
    visualizer = MetricsVisualizer(monitor)
    visualizer.generate_report("performance_report.json")
