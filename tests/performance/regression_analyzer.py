"""Automated Performance Regression Analysis.

This module provides automated tools for detecting performance regressions,
analyzing trends, and validating architectural limitations across different
system versions and configurations.
"""

import asyncio
import json
import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from tests.performance.unified_metrics_collector import (
    AggregatedMetric,
    MetricSource,
    MetricType,
    UnifiedMetricsCollector,
)

logger = logging.getLogger(__name__)


class RegressionType(Enum):
    """Types of performance regressions."""

    LATENCY = "latency"  # Response time increase
    THROUGHPUT = "throughput"  # Throughput decrease
    MEMORY = "memory"  # Memory usage increase
    CPU = "cpu"  # CPU usage increase
    ERROR_RATE = "error_rate"  # Error rate increase
    EFFICIENCY = "efficiency"  # Efficiency decrease


class RegressionSeverity(Enum):
    """Severity levels for regressions."""

    INFO = "info"  # <5% change
    MINOR = "minor"  # 5-10% change
    MODERATE = "moderate"  # 10-25% change
    MAJOR = "major"  # 25-50% change
    CRITICAL = "critical"  # >50% change


@dataclass
class RegressionResult:
    """Result of regression analysis."""

    metric_name: str
    source: MetricSource
    regression_type: RegressionType
    severity: RegressionSeverity
    baseline_value: float
    current_value: float
    change_percent: float
    confidence_level: float
    p_value: float
    sample_size: int
    time_window: str
    detected_at: datetime
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Trend analysis results."""

    metric_name: str
    source: MetricSource
    trend_direction: str  # "improving", "degrading", "stable"
    slope: float
    r_squared: float
    forecast_1h: float
    forecast_24h: float
    change_points: List[datetime] = field(default_factory=list)
    seasonality: Optional[Dict[str, float]] = None


class RegressionAnalyzer:
    """Automated performance regression analyzer."""

    def __init__(
        self,
        metrics_collector: UnifiedMetricsCollector,
        baseline_dir: str = "tests/performance/baselines",
        significance_level: float = 0.05,
        min_sample_size: int = 30,
    ):
        """Initialize regression analyzer.

        Args:
            metrics_collector: Unified metrics collector instance
            baseline_dir: Directory for baseline data
            significance_level: Statistical significance level (p-value threshold)
            min_sample_size: Minimum sample size for analysis
        """
        self.metrics_collector = metrics_collector
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.significance_level = significance_level
        self.min_sample_size = min_sample_size

        # Baseline storage
        self._baselines: Dict[str, Dict[str, Any]] = {}
        self._load_baselines()

        # Regression history
        self._regression_history: List[RegressionResult] = []

        # Architectural limits from documentation
        self._architectural_limits = {
            "agent": {
                "max_agents": 50,
                "efficiency_at_50": 0.284,  # 28.4%
                "memory_per_agent_mb": 34.5,
                "coordination_overhead": 0.72,  # 72%
            },
            "inference": {
                "target_latency_ms": 10.0,
                "matrix_cache_speedup": 353.0,
                "selective_update_speedup": 10.0,
            },
            "websocket": {"target_latency_ms": 100.0, "connections_per_second": 1000.0},
            "database": {"query_latency_ms": 50.0, "transaction_rate": 1000.0},
        }

    def _load_baselines(self):
        """Load baseline data from disk."""
        baseline_file = self.baseline_dir / "performance_baselines.json"
        if baseline_file.exists():
            try:
                with open(baseline_file, "r") as f:
                    self._baselines = json.load(f)
                logger.info(f"Loaded {len(self._baselines)} baselines")
            except Exception as e:
                logger.error(f"Failed to load baselines: {e}")
                self._baselines = {}

    def save_baselines(self):
        """Save current baselines to disk."""
        baseline_file = self.baseline_dir / "performance_baselines.json"
        try:
            with open(baseline_file, "w") as f:
                json.dump(self._baselines, f, indent=2)
            logger.info("Baselines saved successfully")
        except Exception as e:
            logger.error(f"Failed to save baselines: {e}")

    async def establish_baseline(
        self, duration_seconds: int = 300, sources: Optional[List[MetricSource]] = None
    ) -> Dict[str, Any]:
        """Establish performance baselines from current metrics.

        Args:
            duration_seconds: Duration to collect baseline data
            sources: Specific sources to baseline (None = all)

        Returns:
            Dictionary of baseline metrics
        """
        logger.info(f"Establishing baseline for {duration_seconds} seconds")

        # Collect metrics for specified duration
        start_time = datetime.now()
        baseline_data = defaultdict(list)

        while (datetime.now() - start_time).total_seconds() < duration_seconds:
            # Get current metrics
            summary = await self.metrics_collector.get_metrics_summary(window_seconds=60)

            for source_name, metrics in summary["sources"].items():
                if sources and MetricSource(source_name) not in sources:
                    continue

                for metric_key, metric_data in metrics.items():
                    baseline_data[metric_key].append(metric_data["stats"])

            await asyncio.sleep(10)  # Collect every 10 seconds

        # Calculate baseline statistics
        baselines = {}
        for metric_key, samples in baseline_data.items():
            if len(samples) < self.min_sample_size:
                continue

            # Extract values
            values = [s["avg"] for s in samples]

            baselines[metric_key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "p50": np.percentile(values, 50),
                "p95": np.percentile(values, 95),
                "p99": np.percentile(values, 99),
                "sample_count": len(values),
                "established_at": datetime.now().isoformat(),
                "duration_seconds": duration_seconds,
            }

        # Update stored baselines
        self._baselines.update(baselines)
        self.save_baselines()

        logger.info(f"Established baselines for {len(baselines)} metrics")
        return baselines

    async def analyze_regressions(
        self, window_seconds: int = 300, sources: Optional[List[MetricSource]] = None
    ) -> List[RegressionResult]:
        """Analyze current metrics for regressions against baselines.

        Args:
            window_seconds: Time window for analysis
            sources: Specific sources to analyze (None = all)

        Returns:
            List of detected regressions
        """
        regressions = []

        # Get current metrics
        summary = await self.metrics_collector.get_metrics_summary(window_seconds=window_seconds)

        for source_name, metrics in summary["sources"].items():
            if sources and MetricSource(source_name) not in sources:
                continue

            source = MetricSource(source_name)

            for metric_key, metric_data in metrics.items():
                # Check if we have a baseline
                if metric_key not in self._baselines:
                    continue

                baseline = self._baselines[metric_key]
                current_stats = metric_data["stats"]

                # Determine regression type
                regression_type = self._determine_regression_type(metric_data["name"], source)

                # Perform statistical test
                regression = self._test_for_regression(
                    metric_name=metric_data["name"],
                    source=source,
                    regression_type=regression_type,
                    baseline=baseline,
                    current_stats=current_stats,
                    sample_count=current_stats["count"],
                )

                if regression:
                    regressions.append(regression)
                    self._regression_history.append(regression)

        # Check architectural limits
        limit_violations = await self._check_architectural_limits(summary)
        regressions.extend(limit_violations)

        return regressions

    def _determine_regression_type(self, metric_name: str, source: MetricSource) -> RegressionType:
        """Determine the type of regression for a metric."""
        # Latency metrics
        if any(keyword in metric_name.lower() for keyword in ["latency", "time", "duration"]):
            return RegressionType.LATENCY

        # Throughput metrics
        if any(keyword in metric_name.lower() for keyword in ["throughput", "rate", "per_second"]):
            return RegressionType.THROUGHPUT

        # Memory metrics
        if "memory" in metric_name.lower():
            return RegressionType.MEMORY

        # CPU metrics
        if "cpu" in metric_name.lower():
            return RegressionType.CPU

        # Error metrics
        if "error" in metric_name.lower():
            return RegressionType.ERROR_RATE

        # Efficiency metrics
        if "efficiency" in metric_name.lower():
            return RegressionType.EFFICIENCY

        # Default based on source
        if source == MetricSource.SYSTEM:
            return RegressionType.CPU
        else:
            return RegressionType.LATENCY

    def _test_for_regression(
        self,
        metric_name: str,
        source: MetricSource,
        regression_type: RegressionType,
        baseline: Dict[str, float],
        current_stats: Dict[str, float],
        sample_count: int,
    ) -> Optional[RegressionResult]:
        """Test if current metrics show regression from baseline."""
        if sample_count < self.min_sample_size:
            return None

        baseline_value = baseline["mean"]
        baseline_std = baseline["std"]
        current_value = current_stats["avg"]

        # Calculate change
        if baseline_value == 0:
            return None

        change_percent = ((current_value - baseline_value) / baseline_value) * 100

        # For throughput and efficiency, negative change is bad
        if regression_type in [RegressionType.THROUGHPUT, RegressionType.EFFICIENCY]:
            change_percent = -change_percent

        # Skip if improvement
        if change_percent < 0:
            return None

        # Perform t-test
        # Approximate t-test using summary statistics
        pooled_std = np.sqrt((baseline_std**2 + current_stats.get("std", baseline_std) ** 2) / 2)

        if pooled_std == 0:
            return None

        t_statistic = abs(current_value - baseline_value) / (pooled_std * np.sqrt(2 / sample_count))

        # Approximate p-value
        df = 2 * sample_count - 2
        p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_statistic), df))

        # Check significance
        if p_value > self.significance_level:
            return None

        # Determine severity
        severity = self._determine_severity(abs(change_percent))

        # Calculate confidence level
        confidence_level = 1 - p_value

        return RegressionResult(
            metric_name=metric_name,
            source=source,
            regression_type=regression_type,
            severity=severity,
            baseline_value=baseline_value,
            current_value=current_value,
            change_percent=abs(change_percent),
            confidence_level=confidence_level,
            p_value=p_value,
            sample_size=sample_count,
            time_window=f"{sample_count * 10}s",  # Assuming 10s intervals
            detected_at=datetime.now(),
            details={
                "baseline_std": baseline_std,
                "current_std": current_stats.get("std", 0),
                "t_statistic": t_statistic,
            },
        )

    def _determine_severity(self, change_percent: float) -> RegressionSeverity:
        """Determine regression severity based on change percentage."""
        if change_percent < 5:
            return RegressionSeverity.INFO
        elif change_percent < 10:
            return RegressionSeverity.MINOR
        elif change_percent < 25:
            return RegressionSeverity.MODERATE
        elif change_percent < 50:
            return RegressionSeverity.MAJOR
        else:
            return RegressionSeverity.CRITICAL

    async def _check_architectural_limits(self, summary: Dict[str, Any]) -> List[RegressionResult]:
        """Check metrics against documented architectural limits."""
        violations = []

        # Check agent limits
        if MetricSource.AGENT.value in summary["sources"]:
            agent_metrics = summary["sources"][MetricSource.AGENT.value]

            # Check active agents
            for key, data in agent_metrics.items():
                if "active_agents" in key:
                    active_agents = data["stats"]["latest"]
                    if active_agents > self._architectural_limits["agent"]["max_agents"]:
                        violations.append(
                            RegressionResult(
                                metric_name="active_agents",
                                source=MetricSource.AGENT,
                                regression_type=RegressionType.EFFICIENCY,
                                severity=RegressionSeverity.CRITICAL,
                                baseline_value=self._architectural_limits["agent"]["max_agents"],
                                current_value=active_agents,
                                change_percent=(active_agents - 50) / 50 * 100,
                                confidence_level=1.0,
                                p_value=0.0,
                                sample_size=data["stats"]["count"],
                                time_window="current",
                                detected_at=datetime.now(),
                                details={
                                    "limit_type": "architectural",
                                    "expected_efficiency": self._architectural_limits["agent"][
                                        "efficiency_at_50"
                                    ],
                                },
                            )
                        )

        # Check inference limits
        if MetricSource.INFERENCE.value in summary["sources"]:
            inference_metrics = summary["sources"][MetricSource.INFERENCE.value]

            for key, data in inference_metrics.items():
                if "inference_time" in key:
                    latency = data["stats"]["p95"]
                    target = self._architectural_limits["inference"]["target_latency_ms"]
                    if latency > target * 2:  # 2x target is a violation
                        violations.append(
                            RegressionResult(
                                metric_name="inference_latency",
                                source=MetricSource.INFERENCE,
                                regression_type=RegressionType.LATENCY,
                                severity=RegressionSeverity.MAJOR,
                                baseline_value=target,
                                current_value=latency,
                                change_percent=(latency - target) / target * 100,
                                confidence_level=0.95,
                                p_value=0.05,
                                sample_size=data["stats"]["count"],
                                time_window="current",
                                detected_at=datetime.now(),
                                details={"limit_type": "architectural_target"},
                            )
                        )

        return violations

    async def analyze_trends(
        self,
        metric_name: str,
        source: MetricSource,
        duration_hours: int = 24,
        forecast_hours: int = 24,
    ) -> TrendAnalysis:
        """Analyze performance trends and forecast future values.

        Args:
            metric_name: Name of the metric
            source: Source of the metric
            duration_hours: Historical data duration
            forecast_hours: Future forecast duration

        Returns:
            Trend analysis results
        """
        # Get historical data
        history = self.metrics_collector.get_metric_history(
            metric_name, source, duration_hours * 3600
        )

        if len(history) < 10:
            raise ValueError(f"Insufficient data for trend analysis: {len(history)} points")

        # Convert to arrays
        timestamps = np.array([h[0].timestamp() for h in history])
        values = np.array([h[1] for h in history])

        # Normalize timestamps
        time_range = timestamps[-1] - timestamps[0]
        x = (timestamps - timestamps[0]) / time_range

        # Fit linear regression
        slope, intercept = np.polyfit(x, values, 1)

        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((values - y_pred) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Determine trend direction
        if abs(slope) < 0.01:
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "degrading" if self._is_higher_worse(metric_name) else "improving"
        else:
            trend_direction = "improving" if self._is_higher_worse(metric_name) else "degrading"

        # Forecast future values
        current_time = timestamps[-1]
        forecast_1h = slope * ((current_time + 3600 - timestamps[0]) / time_range) + intercept
        forecast_24h = slope * ((current_time + 86400 - timestamps[0]) / time_range) + intercept

        # Detect change points (simplified)
        change_points = self._detect_change_points(timestamps, values)

        # Check for seasonality (simplified - hourly pattern)
        seasonality = self._detect_seasonality(timestamps, values)

        return TrendAnalysis(
            metric_name=metric_name,
            source=source,
            trend_direction=trend_direction,
            slope=slope * time_range,  # Slope per second
            r_squared=r_squared,
            forecast_1h=forecast_1h,
            forecast_24h=forecast_24h,
            change_points=[datetime.fromtimestamp(ts) for ts in change_points],
            seasonality=seasonality,
        )

    def _is_higher_worse(self, metric_name: str) -> bool:
        """Determine if higher values are worse for a metric."""
        positive_metrics = ["throughput", "rate", "efficiency", "speedup"]
        return not any(m in metric_name.lower() for m in positive_metrics)

    def _detect_change_points(
        self, timestamps: np.ndarray, values: np.ndarray, window_size: int = 20
    ) -> List[float]:
        """Detect significant change points in time series."""
        if len(values) < window_size * 2:
            return []

        change_points = []

        for i in range(window_size, len(values) - window_size):
            # Compare statistics before and after point
            before = values[i - window_size : i]
            after = values[i : i + window_size]

            # Perform t-test
            t_stat, p_value = scipy_stats.ttest_ind(before, after)

            if p_value < 0.01:  # Significant change
                change_points.append(timestamps[i])

        return change_points

    def _detect_seasonality(
        self, timestamps: np.ndarray, values: np.ndarray, period_hours: int = 24
    ) -> Optional[Dict[str, float]]:
        """Detect seasonal patterns in metrics."""
        if len(values) < period_hours * 2:
            return None

        # Group by hour of day
        hours = [(datetime.fromtimestamp(ts).hour) for ts in timestamps]
        hourly_values = defaultdict(list)

        for hour, value in zip(hours, values):
            hourly_values[hour].append(value)

        # Calculate hourly averages
        seasonality = {}
        for hour, vals in hourly_values.items():
            if len(vals) >= 5:  # Minimum samples
                seasonality[f"hour_{hour}"] = {
                    "avg": np.mean(vals),
                    "std": np.std(vals),
                    "samples": len(vals),
                }

        return seasonality if len(seasonality) > 12 else None

    def generate_regression_report(
        self, regressions: List[RegressionResult], include_recommendations: bool = True
    ) -> str:
        """Generate a comprehensive regression analysis report."""
        if not regressions:
            return "No performance regressions detected."

        # Group by severity
        by_severity = defaultdict(list)
        for reg in regressions:
            by_severity[reg.severity].append(reg)

        report = [
            "# Performance Regression Analysis Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Summary",
            f"- Total Regressions: {len(regressions)}",
            f"- Critical: {len(by_severity[RegressionSeverity.CRITICAL])}",
            f"- Major: {len(by_severity[RegressionSeverity.MAJOR])}",
            f"- Moderate: {len(by_severity[RegressionSeverity.MODERATE])}",
            f"- Minor: {len(by_severity[RegressionSeverity.MINOR])}",
            "",
            "## Detailed Analysis",
            "",
        ]

        # Add details for each severity level
        for severity in [
            RegressionSeverity.CRITICAL,
            RegressionSeverity.MAJOR,
            RegressionSeverity.MODERATE,
            RegressionSeverity.MINOR,
        ]:
            if severity not in by_severity:
                continue

            severity_regs = by_severity[severity]
            report.extend([f"### {severity.value.upper()} Regressions", ""])

            for reg in severity_regs:
                report.extend(
                    [
                        f"#### {reg.source.value}.{reg.metric_name}",
                        f"- **Type**: {reg.regression_type.value}",
                        f"- **Baseline**: {reg.baseline_value:.2f}",
                        f"- **Current**: {reg.current_value:.2f}",
                        f"- **Change**: {reg.change_percent:.1f}%",
                        f"- **Confidence**: {reg.confidence_level:.1%} (p={reg.p_value:.4f})",
                        f"- **Samples**: {reg.sample_size}",
                        "",
                    ]
                )

                if include_recommendations:
                    recommendations = self._generate_recommendations(reg)
                    if recommendations:
                        report.extend(
                            ["**Recommendations:**", *[f"- {rec}" for rec in recommendations], ""]
                        )

        # Add architectural limit violations
        architectural_violations = [
            r for r in regressions if r.details.get("limit_type") == "architectural"
        ]

        if architectural_violations:
            report.extend(["## Architectural Limit Violations", ""])

            for violation in architectural_violations:
                report.extend(
                    [
                        f"- **{violation.metric_name}**: {violation.current_value:.1f} "
                        f"(limit: {violation.baseline_value:.1f})",
                        "",
                    ]
                )

        return "\n".join(report)

    def _generate_recommendations(self, regression: RegressionResult) -> List[str]:
        """Generate specific recommendations for a regression."""
        recommendations = []

        # Type-specific recommendations
        if regression.regression_type == RegressionType.LATENCY:
            if regression.change_percent > 50:
                recommendations.append(
                    "Significant latency increase detected. Profile the operation "
                    "to identify bottlenecks."
                )
            recommendations.append("Consider implementing caching or optimizing algorithms.")

        elif regression.regression_type == RegressionType.MEMORY:
            recommendations.append(
                "Increased memory usage detected. Check for memory leaks "
                "or inefficient data structures."
            )
            if regression.source == MetricSource.AGENT:
                recommendations.append(
                    "Consider reducing agent belief state size or implementing "
                    "state compression."
                )

        elif regression.regression_type == RegressionType.THROUGHPUT:
            recommendations.append(
                "Throughput degradation detected. Check for blocking operations "
                "or resource contention."
            )

        elif regression.regression_type == RegressionType.CPU:
            recommendations.append(
                "High CPU usage detected. Profile CPU-intensive operations "
                "and consider optimization or parallelization."
            )

        # Source-specific recommendations
        if regression.source == MetricSource.DATABASE:
            recommendations.append(
                "Database performance regression. Check query optimization, "
                "indexes, and connection pool settings."
            )

        elif regression.source == MetricSource.WEBSOCKET:
            recommendations.append(
                "WebSocket performance issue. Check message queue sizes " "and connection handling."
            )

        # Severity-based recommendations
        if regression.severity in [RegressionSeverity.CRITICAL, RegressionSeverity.MAJOR]:
            recommendations.insert(
                0,
                "URGENT: This regression requires immediate attention "
                "as it significantly impacts system performance.",
            )

        return recommendations

    async def compare_versions(
        self, version1_data: Dict[str, Any], version2_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare performance between two versions.

        Args:
            version1_data: Performance data for version 1
            version2_data: Performance data for version 2

        Returns:
            Comparison results with improvements and regressions
        """
        comparison = {
            "version1": version1_data.get("version", "unknown"),
            "version2": version2_data.get("version", "unknown"),
            "improvements": [],
            "regressions": [],
            "unchanged": [],
            "summary": {},
        }

        # Compare each metric
        v1_metrics = version1_data.get("metrics", {})
        v2_metrics = version2_data.get("metrics", {})

        all_metrics = set(v1_metrics.keys()) | set(v2_metrics.keys())

        for metric in all_metrics:
            if metric not in v1_metrics or metric not in v2_metrics:
                continue

            v1_value = v1_metrics[metric]["avg"]
            v2_value = v2_metrics[metric]["avg"]

            if v1_value == 0:
                continue

            change_percent = ((v2_value - v1_value) / v1_value) * 100

            metric_comparison = {
                "metric": metric,
                "v1_value": v1_value,
                "v2_value": v2_value,
                "change_percent": change_percent,
            }

            # Determine if improvement or regression
            is_improvement = (
                change_percent < 0 if self._is_higher_worse(metric) else change_percent > 0
            )

            if abs(change_percent) < 2:  # <2% change is noise
                comparison["unchanged"].append(metric_comparison)
            elif is_improvement:
                comparison["improvements"].append(metric_comparison)
            else:
                comparison["regressions"].append(metric_comparison)

        # Calculate summary statistics
        comparison["summary"] = {
            "total_metrics": len(all_metrics),
            "improvements": len(comparison["improvements"]),
            "regressions": len(comparison["regressions"]),
            "unchanged": len(comparison["unchanged"]),
            "overall_verdict": (
                "improved"
                if len(comparison["improvements"]) > len(comparison["regressions"])
                else "degraded"
            ),
        }

        return comparison


# Global analyzer instance
regression_analyzer = None


def initialize_regression_analyzer(metrics_collector: UnifiedMetricsCollector):
    """Initialize the global regression analyzer."""
    global regression_analyzer
    regression_analyzer = RegressionAnalyzer(metrics_collector)
    return regression_analyzer
