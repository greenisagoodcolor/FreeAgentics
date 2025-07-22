#!/usr/bin/env python3
"""
Performance Regression Detection System
=======================================

Advanced performance regression detection with statistical analysis,
trend detection, and intelligent alerting following PERF-ENGINEER methodology.
"""

import json
import logging
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RegressionAlert:
    """Performance regression alert."""

    metric_name: str
    category: str
    current_value: float
    baseline_value: float
    regression_percent: float
    severity: str  # critical, warning, info
    confidence: float  # 0.0 to 1.0
    trend_analysis: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime


@dataclass
class PerformanceTrend:
    """Performance trend analysis."""

    metric_name: str
    data_points: List[Tuple[datetime, float]]
    trend_direction: str  # improving, degrading, stable
    trend_strength: float  # -1.0 to 1.0
    volatility: float  # standard deviation
    prediction: Optional[float]


class PerformanceRegressionDetector:
    """Advanced performance regression detection."""

    # Regression thresholds
    CRITICAL_REGRESSION_THRESHOLD = 0.20  # 20%
    WARNING_REGRESSION_THRESHOLD = 0.10  # 10%
    IMPROVEMENT_THRESHOLD = -0.05  # 5% improvement

    # Statistical significance thresholds
    MIN_SAMPLES_FOR_TREND = 5
    CONFIDENCE_THRESHOLD = 0.7

    def __init__(self, history_file: str = "performance_history.json"):
        """Initialize regression detector."""
        self.history_file = Path(history_file)
        self.performance_history = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load performance history from file."""
        if not self.history_file.exists():
            return []

        try:
            with open(self.history_file) as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            return []

    def _save_history(self):
        """Save performance history to file."""
        try:
            with open(self.history_file, "w") as f:
                json.dump(self.performance_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save history: {e}")

    def add_performance_data(self, results: Dict[str, Any]):
        """Add new performance data to history."""
        # Add timestamp if not present
        if "timestamp" not in results:
            results["timestamp"] = datetime.now().isoformat()

        self.performance_history.append(results)

        # Keep only last 90 days of data
        cutoff_date = datetime.now() - timedelta(days=90)
        self.performance_history = [
            entry
            for entry in self.performance_history
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_date
        ]

        self._save_history()

    def get_metric_history(
        self, metric_name: str, category: str = None
    ) -> List[Tuple[datetime, float]]:
        """Get historical data for a specific metric."""
        history = []

        for entry in self.performance_history:
            timestamp = datetime.fromisoformat(entry["timestamp"])

            # Look for metric in results
            metrics = entry.get("metrics", {})
            if isinstance(metrics, list):
                # Handle list format
                for metric in metrics:
                    if metric.get("name") == metric_name and (
                        category is None or metric.get("category") == category
                    ):
                        history.append((timestamp, metric.get("value", 0)))
                        break
            elif isinstance(metrics, dict):
                # Handle dict format
                key = f"{category}.{metric_name}" if category else metric_name
                if key in metrics:
                    value = (
                        metrics[key].get("value")
                        if isinstance(metrics[key], dict)
                        else metrics[key]
                    )
                    history.append((timestamp, float(value)))

        return sorted(history, key=lambda x: x[0])

    def analyze_trend(self, metric_name: str, category: str = None) -> PerformanceTrend:
        """Analyze performance trend for a metric."""
        history = self.get_metric_history(metric_name, category)

        if len(history) < self.MIN_SAMPLES_FOR_TREND:
            return PerformanceTrend(
                metric_name=metric_name,
                data_points=history,
                trend_direction="insufficient_data",
                trend_strength=0.0,
                volatility=0.0,
                prediction=None,
            )

        # Extract values and timestamps
        [t.timestamp() for t, v in history]
        values = [v for t, v in history]

        # Calculate trend using linear regression
        n = len(values)
        x = np.array(range(n))
        y = np.array(values)

        # Linear regression
        slope = np.polyfit(x, y, 1)[0]

        # Normalize slope by average value to get relative trend
        avg_value = np.mean(values)
        if avg_value != 0:
            trend_strength = slope / avg_value * n  # Scale by number of points
        else:
            trend_strength = 0.0

        # Determine trend direction
        if abs(trend_strength) < 0.05:
            trend_direction = "stable"
        elif trend_strength > 0:
            trend_direction = "degrading"
        else:
            trend_direction = "improving"

        # Calculate volatility
        volatility = np.std(values) / avg_value if avg_value != 0 else 0.0

        # Simple prediction (next value)
        prediction = values[-1] + slope if slope else None

        return PerformanceTrend(
            metric_name=metric_name,
            data_points=history,
            trend_direction=trend_direction,
            trend_strength=trend_strength,
            volatility=volatility,
            prediction=prediction,
        )

    def detect_regression(self, current_results: Dict[str, Any]) -> List[RegressionAlert]:
        """Detect performance regressions in current results."""
        alerts = []

        # Get current metrics
        current_metrics = current_results.get("metrics", {})
        if isinstance(current_metrics, list):
            # Convert list to dict format
            metrics_dict = {}
            for metric in current_metrics:
                key = f"{metric.get('category', 'unknown')}.{metric.get('name')}"
                metrics_dict[key] = metric
            current_metrics = metrics_dict

        # Analyze each metric
        for metric_key, metric_data in current_metrics.items():
            if isinstance(metric_data, dict):
                metric_name = metric_data.get("name", metric_key.split(".")[-1])
                category = metric_data.get("category", metric_key.split(".")[0])
                current_value = metric_data.get("value", 0)
            else:
                # Handle simple value format
                parts = metric_key.split(".")
                category = parts[0] if len(parts) > 1 else "unknown"
                metric_name = parts[-1]
                current_value = float(metric_data)

            # Get trend analysis
            trend = self.analyze_trend(metric_name, category)

            if trend.trend_direction == "insufficient_data":
                continue

            # Calculate baseline from recent history
            recent_values = [v for t, v in trend.data_points[-10:]]  # Last 10 values
            if not recent_values:
                continue

            baseline_value = statistics.median(recent_values)

            # Skip if no meaningful baseline
            if baseline_value == 0:
                continue

            # Calculate regression percentage
            regression_percent = (current_value - baseline_value) / baseline_value

            # Determine severity
            severity = self._determine_severity(regression_percent, trend)

            if severity != "none":
                # Calculate confidence based on trend consistency and data quality
                confidence = self._calculate_confidence(trend, regression_percent)

                # Generate recommendations
                recommendations = self._generate_recommendations(
                    metric_name, category, regression_percent, trend
                )

                alert = RegressionAlert(
                    metric_name=metric_name,
                    category=category,
                    current_value=current_value,
                    baseline_value=baseline_value,
                    regression_percent=regression_percent * 100,
                    severity=severity,
                    confidence=confidence,
                    trend_analysis={
                        "direction": trend.trend_direction,
                        "strength": trend.trend_strength,
                        "volatility": trend.volatility,
                        "data_points": len(trend.data_points),
                    },
                    recommendations=recommendations,
                    timestamp=datetime.now(),
                )

                alerts.append(alert)

        return alerts

    def _determine_severity(self, regression_percent: float, trend: PerformanceTrend) -> str:
        """Determine severity level of regression."""
        # Factor in trend direction
        if trend.trend_direction == "improving" and regression_percent < 0:
            return "none"  # Continued improvement

        # Check thresholds
        if regression_percent >= self.CRITICAL_REGRESSION_THRESHOLD:
            return "critical"
        elif regression_percent >= self.WARNING_REGRESSION_THRESHOLD:
            return "warning"
        elif regression_percent <= self.IMPROVEMENT_THRESHOLD:
            return "improvement"
        else:
            return "none"

    def _calculate_confidence(self, trend: PerformanceTrend, regression_percent: float) -> float:
        """Calculate confidence level for the regression alert."""
        confidence = 0.5  # Base confidence

        # More data points = higher confidence
        data_points = len(trend.data_points)
        if data_points >= 20:
            confidence += 0.3
        elif data_points >= 10:
            confidence += 0.2
        elif data_points >= 5:
            confidence += 0.1

        # Lower volatility = higher confidence
        if trend.volatility < 0.1:
            confidence += 0.2
        elif trend.volatility < 0.2:
            confidence += 0.1

        # Consistent trend = higher confidence
        if trend.trend_direction == "degrading" and regression_percent > 0:
            confidence += 0.2
        elif trend.trend_direction == "improving" and regression_percent < 0:
            confidence += 0.2

        # Larger regression = higher confidence (if it's actual regression)
        if abs(regression_percent) > 0.2:
            confidence += 0.1

        return min(confidence, 1.0)

    def _generate_recommendations(
        self,
        metric_name: str,
        category: str,
        regression_percent: float,
        trend: PerformanceTrend,
    ) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Category-specific recommendations
        if category == "agent_spawning":
            if "memory" in metric_name.lower():
                recommendations.append("Review agent initialization for memory leaks")
                recommendations.append("Consider object pooling for agent creation")
            else:
                recommendations.append("Profile agent creation bottlenecks")
                recommendations.append("Consider async agent initialization")

        elif category == "pymdp_inference":
            recommendations.append("Profile PyMDP computation hotspots")
            recommendations.append("Check for inefficient matrix operations")
            recommendations.append("Consider caching frequent computations")

        elif category == "memory_usage":
            recommendations.append("Run memory profiler to identify leaks")
            recommendations.append("Review garbage collection settings")
            recommendations.append("Check for circular references")

        elif category == "api_performance":
            recommendations.append("Profile API endpoint performance")
            recommendations.append("Check database query optimization")
            recommendations.append("Review caching strategy")

        elif category == "frontend_performance":
            recommendations.append("Analyze bundle size and dependencies")
            recommendations.append("Consider code splitting and lazy loading")
            recommendations.append("Optimize images and assets")

        # Trend-based recommendations
        if trend.trend_direction == "degrading":
            recommendations.append(
                "Performance degrading trend detected - investigate recent changes"
            )

        if trend.volatility > 0.3:
            recommendations.append("High performance volatility - check for environmental factors")

        # Severity-based recommendations
        if regression_percent > 0.5:  # 50%+ regression
            recommendations.append(
                "URGENT: Severe performance regression requires immediate attention"
            )

        return recommendations

    def generate_regression_report(self, alerts: List[RegressionAlert]) -> str:
        """Generate human-readable regression report."""
        if not alerts:
            return "✅ No performance regressions detected"

        report = []
        report.append("=" * 80)
        report.append("PERFORMANCE REGRESSION DETECTION REPORT")
        report.append("=" * 80)
        report.append(f"Timestamp: {datetime.now()}")
        report.append(f"Total Alerts: {len(alerts)}")
        report.append("")

        # Group by severity
        critical_alerts = [a for a in alerts if a.severity == "critical"]
        warning_alerts = [a for a in alerts if a.severity == "warning"]
        improvement_alerts = [a for a in alerts if a.severity == "improvement"]

        if critical_alerts:
            report.append("🔴 CRITICAL REGRESSIONS")
            report.append("-" * 40)
            for alert in critical_alerts:
                self._add_alert_to_report(report, alert)
            report.append("")

        if warning_alerts:
            report.append("🟡 WARNING REGRESSIONS")
            report.append("-" * 40)
            for alert in warning_alerts:
                self._add_alert_to_report(report, alert)
            report.append("")

        if improvement_alerts:
            report.append("🟢 PERFORMANCE IMPROVEMENTS")
            report.append("-" * 40)
            for alert in improvement_alerts:
                self._add_alert_to_report(report, alert)
            report.append("")

        report.append("=" * 80)
        return "\n".join(report)

    def _add_alert_to_report(self, report: List[str], alert: RegressionAlert):
        """Add alert details to report."""
        report.append(f"Metric: {alert.metric_name} ({alert.category})")
        report.append(
            f"Change: {alert.regression_percent:+.1f}% (confidence: {alert.confidence:.1%})"
        )
        report.append(f"Current: {alert.current_value:.2f}, Baseline: {alert.baseline_value:.2f}")

        trend = alert.trend_analysis
        report.append(f"Trend: {trend['direction']} (strength: {trend['strength']:.3f})")

        if alert.recommendations:
            report.append("Recommendations:")
            for rec in alert.recommendations[:3]:  # Top 3 recommendations
                report.append(f"  • {rec}")

        report.append("")

    def save_alerts(self, alerts: List[RegressionAlert], filename: str = None):
        """Save regression alerts to file."""
        if not filename:
            filename = f"regression_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        alerts_data = {
            "timestamp": datetime.now().isoformat(),
            "alert_count": len(alerts),
            "alerts": [asdict(alert) for alert in alerts],
        }

        with open(filename, "w") as f:
            json.dump(alerts_data, f, indent=2, default=str)

        logger.info(f"Regression alerts saved to {filename}")


def main():
    """CLI entry point for regression detection."""
    import argparse

    parser = argparse.ArgumentParser(description="Performance regression detection")
    parser.add_argument(
        "--results-file", type=str, required=True, help="Performance results JSON file"
    )
    parser.add_argument(
        "--history-file",
        type=str,
        default="performance_history.json",
        help="Performance history file",
    )
    parser.add_argument("--save-alerts", type=str, help="Save alerts to file")

    args = parser.parse_args()

    # Load current results
    with open(args.results_file) as f:
        results = json.load(f)

    # Initialize detector
    detector = PerformanceRegressionDetector(args.history_file)

    # Add current data to history
    detector.add_performance_data(results)

    # Detect regressions
    alerts = detector.detect_regression(results)

    # Generate report
    report = detector.generate_regression_report(alerts)
    print(report)

    # Save alerts if requested
    if args.save_alerts:
        detector.save_alerts(alerts, args.save_alerts)

    # Exit with error code if critical regressions found
    critical_count = sum(1 for a in alerts if a.severity == "critical")
    if critical_count > 0:
        print(f"\n❌ {critical_count} critical regressions detected!")
        return 1

    print("\n✅ Performance regression check passed")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
