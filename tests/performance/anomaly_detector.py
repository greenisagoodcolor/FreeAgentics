"""Real-time Anomaly Detection for Performance Metrics.

This module provides advanced anomaly detection capabilities using statistical
methods and machine learning to identify performance issues in real-time.
"""

import logging
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

from tests.performance.unified_metrics_collector import (
    MetricPoint,
    MetricSource,
)

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies detected."""

    SPIKE = "spike"  # Sudden increase
    DROP = "drop"  # Sudden decrease
    TREND = "trend"  # Gradual change
    PATTERN = "pattern"  # Pattern deviation
    MULTIVARIATE = "multivariate"  # Multi-metric anomaly
    SEASONAL = "seasonal"  # Seasonal deviation
    THRESHOLD = "threshold"  # Simple threshold breach


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""

    LOW = "low"  # 1-2 std deviations
    MEDIUM = "medium"  # 2-3 std deviations
    HIGH = "high"  # 3-4 std deviations
    CRITICAL = "critical"  # >4 std deviations


@dataclass
class Anomaly:
    """Detected anomaly information."""

    metric_name: str
    source: MetricSource
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    timestamp: datetime
    value: float
    expected_value: float
    deviation: float
    confidence: float
    context: Dict[str, Any] = field(default_factory=dict)
    related_metrics: List[str] = field(default_factory=list)


@dataclass
class AnomalyDetectorConfig:
    """Configuration for anomaly detection."""

    # Detection methods
    enable_statistical: bool = True
    enable_ml: bool = True
    enable_threshold: bool = True
    enable_pattern: bool = True

    # Statistical parameters
    zscore_threshold: float = 3.0
    iqr_multiplier: float = 1.5
    ewma_alpha: float = 0.3

    # ML parameters
    isolation_contamination: float = 0.1
    ml_training_samples: int = 1000
    ml_retrain_interval: int = 3600  # seconds

    # Pattern detection
    pattern_window_size: int = 60
    pattern_similarity_threshold: float = 0.8

    # General settings
    min_samples: int = 30
    anomaly_cooldown: int = 300  # seconds
    correlation_threshold: float = 0.7


class AnomalyDetector:
    """Advanced anomaly detection for performance metrics."""

    def __init__(self, config: AnomalyDetectorConfig = None):
        """Initialize anomaly detector."""
        self.config = config or AnomalyDetectorConfig()

        # Metric history storage
        self._metric_history: Dict[str, Deque[MetricPoint]] = defaultdict(
            lambda: deque(maxlen=self.config.ml_training_samples)
        )

        # Statistical models
        self._baselines: Dict[str, Dict[str, float]] = {}
        self._ewma_values: Dict[str, float] = {}

        # ML models
        self._isolation_forests: Dict[str, IsolationForest] = {}
        self._scalers: Dict[str, StandardScaler] = {}
        self._last_ml_training: Dict[str, datetime] = {}

        # Pattern storage
        self._normal_patterns: Dict[str, List[np.ndarray]] = defaultdict(list)

        # Anomaly tracking
        self._anomaly_history: Deque[Anomaly] = deque(maxlen=1000)
        self._last_anomaly_time: Dict[str, datetime] = {}

        # Correlation matrix
        self._correlation_matrix: Optional[np.ndarray] = None
        self._correlation_metrics: List[str] = []

        logger.info("Anomaly detector initialized")

    def add_metric_point(self, metric_point: MetricPoint):
        """Add a new metric point for analysis."""
        key = f"{metric_point.source.value}.{metric_point.name}"
        self._metric_history[key].append(metric_point)

        # Update EWMA
        if key in self._ewma_values:
            self._ewma_values[key] = (
                self.config.ewma_alpha * metric_point.value
                + (1 - self.config.ewma_alpha) * self._ewma_values[key]
            )
        else:
            self._ewma_values[key] = metric_point.value

    async def detect_anomalies(self, metric_point: MetricPoint) -> List[Anomaly]:
        """Detect anomalies for a metric point."""
        key = f"{metric_point.source.value}.{metric_point.name}"

        # Add to history
        self.add_metric_point(metric_point)

        # Check if we have enough data
        if len(self._metric_history[key]) < self.config.min_samples:
            return []

        # Check cooldown
        if key in self._last_anomaly_time:
            cooldown_end = self._last_anomaly_time[key] + timedelta(
                seconds=self.config.anomaly_cooldown
            )
            if datetime.now() < cooldown_end:
                return []

        anomalies = []

        # Run different detection methods
        if self.config.enable_statistical:
            statistical_anomalies = await self._detect_statistical_anomalies(key, metric_point)
            anomalies.extend(statistical_anomalies)

        if self.config.enable_ml:
            ml_anomalies = await self._detect_ml_anomalies(key, metric_point)
            anomalies.extend(ml_anomalies)

        if self.config.enable_threshold:
            threshold_anomalies = await self._detect_threshold_anomalies(key, metric_point)
            anomalies.extend(threshold_anomalies)

        if self.config.enable_pattern:
            pattern_anomalies = await self._detect_pattern_anomalies(key, metric_point)
            anomalies.extend(pattern_anomalies)

        # Deduplicate and prioritize
        anomalies = self._prioritize_anomalies(anomalies)

        # Update tracking
        if anomalies:
            self._last_anomaly_time[key] = datetime.now()
            self._anomaly_history.extend(anomalies)

        return anomalies

    async def _detect_statistical_anomalies(
        self, key: str, metric_point: MetricPoint
    ) -> List[Anomaly]:
        """Detect anomalies using statistical methods."""
        anomalies = []
        history = list(self._metric_history[key])
        values = [p.value for p in history]

        # Calculate statistics
        mean = np.mean(values)
        std = np.std(values)

        if std == 0:
            return anomalies

        # Z-score test
        zscore = abs((metric_point.value - mean) / std)

        if zscore > self.config.zscore_threshold:
            severity = self._calculate_severity(zscore)
            anomaly_type = AnomalyType.SPIKE if metric_point.value > mean else AnomalyType.DROP

            anomalies.append(
                Anomaly(
                    metric_name=metric_point.name,
                    source=metric_point.source,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    timestamp=metric_point.timestamp,
                    value=metric_point.value,
                    expected_value=mean,
                    deviation=zscore,
                    confidence=min(0.99, 1 - (1 / zscore)),
                    context={"method": "zscore", "mean": mean, "std": std},
                )
            )

        # IQR test
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - self.config.iqr_multiplier * iqr
        upper_bound = q3 + self.config.iqr_multiplier * iqr

        if metric_point.value < lower_bound or metric_point.value > upper_bound:
            severity = AnomalySeverity.MEDIUM
            anomaly_type = (
                AnomalyType.SPIKE if metric_point.value > upper_bound else AnomalyType.DROP
            )

            anomalies.append(
                Anomaly(
                    metric_name=metric_point.name,
                    source=metric_point.source,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    timestamp=metric_point.timestamp,
                    value=metric_point.value,
                    expected_value=(q1 + q3) / 2,
                    deviation=abs(metric_point.value - (q1 + q3) / 2),
                    confidence=0.75,
                    context={
                        "method": "iqr",
                        "lower_bound": lower_bound,
                        "upper_bound": upper_bound,
                    },
                )
            )

        # Trend detection using linear regression
        if len(values) >= 20:
            x = np.arange(len(values))
            slope, intercept, r_value, _, _ = scipy_stats.linregress(x, values)

            if abs(r_value) > 0.8:  # Strong correlation
                expected = slope * len(values) + intercept
                deviation = abs(metric_point.value - expected)

                if deviation > 2 * std:
                    anomalies.append(
                        Anomaly(
                            metric_name=metric_point.name,
                            source=metric_point.source,
                            anomaly_type=AnomalyType.TREND,
                            severity=AnomalySeverity.LOW,
                            timestamp=metric_point.timestamp,
                            value=metric_point.value,
                            expected_value=expected,
                            deviation=deviation,
                            confidence=abs(r_value),
                            context={
                                "method": "trend",
                                "slope": slope,
                                "r_squared": r_value**2,
                            },
                        )
                    )

        return anomalies

    async def _detect_ml_anomalies(self, key: str, metric_point: MetricPoint) -> List[Anomaly]:
        """Detect anomalies using machine learning."""
        if not self.config.enable_ml:
            return []

        anomalies = []

        # Check if we need to train/retrain model
        if key not in self._isolation_forests or self._should_retrain(key):
            self._train_ml_model(key)

        if key not in self._isolation_forests:
            return anomalies

        # Prepare features
        features = self._extract_features(key, metric_point)
        if features is None:
            return anomalies

        # Scale features
        features_scaled = self._scalers[key].transform([features])

        # Predict
        prediction = self._isolation_forests[key].predict(features_scaled)
        anomaly_score = self._isolation_forests[key].score_samples(features_scaled)[0]

        if prediction[0] == -1:  # Anomaly detected
            # Calculate severity based on anomaly score
            severity = self._ml_severity(anomaly_score)

            anomalies.append(
                Anomaly(
                    metric_name=metric_point.name,
                    source=metric_point.source,
                    anomaly_type=AnomalyType.MULTIVARIATE,
                    severity=severity,
                    timestamp=metric_point.timestamp,
                    value=metric_point.value,
                    expected_value=self._ewma_values.get(key, metric_point.value),
                    deviation=abs(anomaly_score),
                    confidence=min(0.95, abs(anomaly_score)),
                    context={
                        "method": "isolation_forest",
                        "anomaly_score": anomaly_score,
                        "features": features,
                    },
                )
            )

        return anomalies

    async def _detect_threshold_anomalies(
        self, key: str, metric_point: MetricPoint
    ) -> List[Anomaly]:
        """Detect simple threshold-based anomalies."""
        anomalies = []

        # Define thresholds based on metric type
        thresholds = self._get_metric_thresholds(metric_point.name, metric_point.source)

        for threshold_type, (operator, value) in thresholds.items():
            breach = False

            if operator == ">":
                breach = metric_point.value > value
            elif operator == "<":
                breach = metric_point.value < value
            elif operator == ">=":
                breach = metric_point.value >= value
            elif operator == "<=":
                breach = metric_point.value <= value

            if breach:
                anomalies.append(
                    Anomaly(
                        metric_name=metric_point.name,
                        source=metric_point.source,
                        anomaly_type=AnomalyType.THRESHOLD,
                        severity=AnomalySeverity.HIGH,
                        timestamp=metric_point.timestamp,
                        value=metric_point.value,
                        expected_value=value,
                        deviation=abs(metric_point.value - value),
                        confidence=1.0,
                        context={
                            "method": "threshold",
                            "threshold_type": threshold_type,
                            "operator": operator,
                            "threshold_value": value,
                        },
                    )
                )

        return anomalies

    async def _detect_pattern_anomalies(self, key: str, metric_point: MetricPoint) -> List[Anomaly]:
        """Detect pattern-based anomalies."""
        anomalies = []

        history = list(self._metric_history[key])
        if len(history) < self.config.pattern_window_size:
            return anomalies

        # Extract recent pattern
        recent_values = [p.value for p in history[-self.config.pattern_window_size :]]
        recent_pattern = np.array(recent_values)

        # Normalize pattern
        if recent_pattern.std() > 0:
            recent_pattern = (recent_pattern - recent_pattern.mean()) / recent_pattern.std()
        else:
            return anomalies

        # Compare with stored normal patterns
        if key in self._normal_patterns:
            similarities = []

            for normal_pattern in self._normal_patterns[key]:
                similarity = self._calculate_pattern_similarity(recent_pattern, normal_pattern)
                similarities.append(similarity)

            max_similarity = max(similarities) if similarities else 0

            if max_similarity < self.config.pattern_similarity_threshold:
                anomalies.append(
                    Anomaly(
                        metric_name=metric_point.name,
                        source=metric_point.source,
                        anomaly_type=AnomalyType.PATTERN,
                        severity=AnomalySeverity.MEDIUM,
                        timestamp=metric_point.timestamp,
                        value=metric_point.value,
                        expected_value=(
                            recent_values[-2] if len(recent_values) > 1 else metric_point.value
                        ),
                        deviation=1 - max_similarity,
                        confidence=1 - max_similarity,
                        context={
                            "method": "pattern",
                            "max_similarity": max_similarity,
                            "pattern_length": len(recent_pattern),
                        },
                    )
                )
        else:
            # Store as normal pattern if no anomalies detected by other methods
            if not anomalies:
                self._normal_patterns[key].append(recent_pattern)

                # Keep only recent patterns
                if len(self._normal_patterns[key]) > 10:
                    self._normal_patterns[key].pop(0)

        return anomalies

    def _calculate_severity(self, zscore: float) -> AnomalySeverity:
        """Calculate anomaly severity based on z-score."""
        if zscore < 2:
            return AnomalySeverity.LOW
        elif zscore < 3:
            return AnomalySeverity.MEDIUM
        elif zscore < 4:
            return AnomalySeverity.HIGH
        else:
            return AnomalySeverity.CRITICAL

    def _ml_severity(self, anomaly_score: float) -> AnomalySeverity:
        """Calculate severity from ML anomaly score."""
        # Isolation Forest scores: normal ~0, anomaly < 0
        abs_score = abs(anomaly_score)

        if abs_score < 0.2:
            return AnomalySeverity.LOW
        elif abs_score < 0.4:
            return AnomalySeverity.MEDIUM
        elif abs_score < 0.6:
            return AnomalySeverity.HIGH
        else:
            return AnomalySeverity.CRITICAL

    def _should_retrain(self, key: str) -> bool:
        """Check if ML model should be retrained."""
        if key not in self._last_ml_training:
            return True

        time_since_training = datetime.now() - self._last_ml_training[key]
        return time_since_training.total_seconds() > self.config.ml_retrain_interval

    def _train_ml_model(self, key: str):
        """Train ML model for anomaly detection."""
        history = list(self._metric_history[key])

        if len(history) < 100:  # Need sufficient data
            return

        # Extract features for training
        features = []
        for i in range(10, len(history)):
            feature_vector = self._extract_features_from_window(
                [p.value for p in history[i - 10 : i]]
            )
            if feature_vector is not None:
                features.append(feature_vector)

        if len(features) < 50:
            return

        # Train scaler
        self._scalers[key] = StandardScaler()
        features_scaled = self._scalers[key].fit_transform(features)

        # Train Isolation Forest
        self._isolation_forests[key] = IsolationForest(
            contamination=self.config.isolation_contamination,
            random_state=42,
            n_estimators=100,
        )
        self._isolation_forests[key].fit(features_scaled)

        self._last_ml_training[key] = datetime.now()
        logger.debug(f"Trained ML model for {key}")

    def _extract_features(self, key: str, metric_point: MetricPoint) -> Optional[List[float]]:
        """Extract features for ML model."""
        history = list(self._metric_history[key])

        if len(history) < 10:
            return None

        recent_values = [p.value for p in history[-10:]]
        return self._extract_features_from_window(recent_values)

    def _extract_features_from_window(self, values: List[float]) -> Optional[List[float]]:
        """Extract statistical features from a window of values."""
        if len(values) < 5:
            return None

        features = [
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.percentile(values, 25),
            np.percentile(values, 75),
            scipy_stats.skew(values),
            scipy_stats.kurtosis(values),
            values[-1] - values[0],  # Change over window
            np.mean(np.diff(values)),  # Average rate of change
        ]

        return features

    def _get_metric_thresholds(
        self, metric_name: str, source: MetricSource
    ) -> Dict[str, Tuple[str, float]]:
        """Get threshold definitions for a metric."""
        thresholds = {}

        # Source-specific thresholds
        if source == MetricSource.DATABASE:
            if "latency" in metric_name:
                thresholds["high_latency"] = (">", 100)  # ms
            elif "connection" in metric_name:
                thresholds["max_connections"] = (">", 100)

        elif source == MetricSource.WEBSOCKET:
            if "error_rate" in metric_name:
                thresholds["high_errors"] = (">", 0.05)  # 5%
            elif "latency" in metric_name:
                thresholds["high_latency"] = (">", 200)  # ms

        elif source == MetricSource.AGENT:
            if "inference_time" in metric_name:
                thresholds["slow_inference"] = (">", 100)  # ms
            elif "memory" in metric_name:
                thresholds["high_memory"] = (">", 100)  # MB per agent

        elif source == MetricSource.SYSTEM:
            if "cpu" in metric_name:
                thresholds["high_cpu"] = (">", 90)  # %
            elif "memory" in metric_name:
                thresholds["high_memory"] = (">", 90)  # %

        return thresholds

    def _calculate_pattern_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Calculate similarity between two patterns."""
        if len(pattern1) != len(pattern2):
            return 0.0

        # Use correlation coefficient as similarity measure
        if pattern1.std() == 0 or pattern2.std() == 0:
            return 1.0 if np.array_equal(pattern1, pattern2) else 0.0

        correlation = np.corrcoef(pattern1, pattern2)[0, 1]

        # Convert to 0-1 range
        return (correlation + 1) / 2

    def _prioritize_anomalies(self, anomalies: List[Anomaly]) -> List[Anomaly]:
        """Deduplicate and prioritize anomalies."""
        if not anomalies:
            return anomalies

        # Group by metric
        grouped = defaultdict(list)
        for anomaly in anomalies:
            key = f"{anomaly.source.value}.{anomaly.metric_name}"
            grouped[key].append(anomaly)

        # Select highest severity per metric
        prioritized = []
        for metric_anomalies in grouped.values():
            # Sort by severity and confidence
            sorted_anomalies = sorted(
                metric_anomalies,
                key=lambda a: (self._severity_score(a.severity), a.confidence),
                reverse=True,
            )
            prioritized.append(sorted_anomalies[0])

        return prioritized

    def _severity_score(self, severity: AnomalySeverity) -> int:
        """Convert severity to numeric score."""
        scores = {
            AnomalySeverity.LOW: 1,
            AnomalySeverity.MEDIUM: 2,
            AnomalySeverity.HIGH: 3,
            AnomalySeverity.CRITICAL: 4,
        }
        return scores.get(severity, 0)

    async def detect_correlated_anomalies(self, window_seconds: int = 300) -> List[Dict[str, Any]]:
        """Detect correlated anomalies across multiple metrics."""
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        recent_anomalies = [a for a in self._anomaly_history if a.timestamp >= cutoff_time]

        if len(recent_anomalies) < 2:
            return []

        # Group anomalies by time window (1 minute buckets)
        time_buckets = defaultdict(list)
        for anomaly in recent_anomalies:
            bucket = anomaly.timestamp.replace(second=0, microsecond=0)
            time_buckets[bucket].append(anomaly)

        # Find correlated anomalies
        correlations = []
        for bucket, bucket_anomalies in time_buckets.items():
            if len(bucket_anomalies) >= 2:
                # Check if anomalies are from different metrics
                metrics = set(f"{a.source.value}.{a.metric_name}" for a in bucket_anomalies)

                if len(metrics) >= 2:
                    correlation = {
                        "timestamp": bucket,
                        "anomaly_count": len(bucket_anomalies),
                        "metrics": list(metrics),
                        "anomalies": bucket_anomalies,
                        "severity": max(a.severity for a in bucket_anomalies),
                        "correlation_type": self._determine_correlation_type(bucket_anomalies),
                    }
                    correlations.append(correlation)

        return correlations

    def _determine_correlation_type(self, anomalies: List[Anomaly]) -> str:
        """Determine the type of correlation between anomalies."""
        # Check if all anomalies are of same type
        types = set(a.anomaly_type for a in anomalies)

        if len(types) == 1:
            if AnomalyType.SPIKE in types:
                return "synchronized_spike"
            elif AnomalyType.DROP in types:
                return "synchronized_drop"

        # Check sources
        sources = set(a.source for a in anomalies)

        if len(sources) == 1:
            return "single_component_issue"
        elif MetricSource.SYSTEM in sources:
            return "system_wide_issue"
        else:
            return "multi_component_issue"

    def get_anomaly_summary(self, window_seconds: int = 3600) -> Dict[str, Any]:
        """Get summary of recent anomalies."""
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        recent_anomalies = [a for a in self._anomaly_history if a.timestamp >= cutoff_time]

        # Group by various dimensions
        by_type = defaultdict(int)
        by_severity = defaultdict(int)
        by_source = defaultdict(int)
        by_metric = defaultdict(int)

        for anomaly in recent_anomalies:
            by_type[anomaly.anomaly_type.value] += 1
            by_severity[anomaly.severity.value] += 1
            by_source[anomaly.source.value] += 1
            by_metric[f"{anomaly.source.value}.{anomaly.metric_name}"] += 1

        # Find most anomalous metrics
        top_metrics = sorted(by_metric.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "total_anomalies": len(recent_anomalies),
            "window_seconds": window_seconds,
            "by_type": dict(by_type),
            "by_severity": dict(by_severity),
            "by_source": dict(by_source),
            "top_anomalous_metrics": top_metrics,
            "latest_anomaly": (
                recent_anomalies[-1].timestamp.isoformat() if recent_anomalies else None
            ),
        }


# Global anomaly detector instance
anomaly_detector = AnomalyDetector()


async def detect_anomalies(metric_point: MetricPoint) -> List[Anomaly]:
    """Detect anomalies for a metric point."""
    return await anomaly_detector.detect_anomalies(metric_point)


def get_anomaly_summary(window_seconds: int = 3600) -> Dict[str, Any]:
    """Get anomaly summary."""
    return anomaly_detector.get_anomaly_summary(window_seconds)
