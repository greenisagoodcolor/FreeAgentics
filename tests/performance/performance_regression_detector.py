"""
Automated Performance Regression Detection System
===============================================

This module provides automated performance regression detection including:
- Baseline performance tracking and comparison
- Statistical analysis for performance degradation detection
- Automated regression alerts and notifications
- Performance trend analysis with anomaly detection
- CI/CD integration for performance gates
- Historical performance data management
- Machine learning-based performance prediction
"""

import asyncio
import json
import logging
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class PerformanceBaseline:
    """Represents a performance baseline for a specific metric."""

    metric_name: str
    baseline_value: float
    baseline_std: float
    sample_count: int
    confidence_interval: Tuple[float, float]
    last_updated: datetime
    version: str = "unknown"
    environment: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionDetectionResult:
    """Result of a regression detection analysis."""

    metric_name: str
    current_value: float
    baseline_value: float
    deviation_percentage: float
    regression_detected: bool
    confidence_score: float
    severity: str  # 'critical', 'major', 'minor', 'negligible'
    statistical_significance: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceTestRun:
    """Represents a complete performance test run."""

    run_id: str
    timestamp: datetime
    version: str
    environment: str
    branch: str
    commit_hash: str
    metrics: Dict[str, float]
    test_duration_seconds: float
    test_metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceRegressionDetector:
    """Advanced performance regression detection system."""

    def __init__(self, data_directory: str = "performance_data"):
        self.data_directory = Path(data_directory)
        self.data_directory.mkdir(exist_ok=True)

        # Performance baselines by metric
        self.baselines: Dict[str, PerformanceBaseline] = {}

        # Historical test runs
        self.test_runs: List[PerformanceTestRun] = []

        # Regression detection configuration
        self.regression_config = {
            'confidence_threshold': 0.95,
            'min_sample_size': 10,
            'max_acceptable_deviation': 0.15,  # 15% deviation threshold
            'critical_deviation': 0.30,  # 30% deviation is critical
            'lookback_window_days': 30,
            'baseline_update_threshold': 0.05,  # 5% change triggers baseline update
            'anomaly_detection_sensitivity': 0.1,
        }

        # Machine learning models for anomaly detection
        self.anomaly_detector = IsolationForest(
            contamination=0.1, random_state=42
        )
        self.scaler = StandardScaler()
        self.ml_models_trained = False

        # Load existing data
        self._load_historical_data()

    def _load_historical_data(self):
        """Load historical performance data."""
        try:
            # Load baselines
            baselines_file = self.data_directory / "baselines.json"
            if baselines_file.exists():
                with open(baselines_file, 'r') as f:
                    data = json.load(f)
                    for metric_name, baseline_data in data.items():
                        self.baselines[metric_name] = PerformanceBaseline(
                            metric_name=baseline_data['metric_name'],
                            baseline_value=baseline_data['baseline_value'],
                            baseline_std=baseline_data['baseline_std'],
                            sample_count=baseline_data['sample_count'],
                            confidence_interval=tuple(
                                baseline_data['confidence_interval']
                            ),
                            last_updated=datetime.fromisoformat(
                                baseline_data['last_updated']
                            ),
                            version=baseline_data.get('version', 'unknown'),
                            environment=baseline_data.get(
                                'environment', 'unknown'
                            ),
                            metadata=baseline_data.get('metadata', {}),
                        )

            # Load test runs
            test_runs_file = self.data_directory / "test_runs.json"
            if test_runs_file.exists():
                with open(test_runs_file, 'r') as f:
                    data = json.load(f)
                    for run_data in data:
                        self.test_runs.append(
                            PerformanceTestRun(
                                run_id=run_data['run_id'],
                                timestamp=datetime.fromisoformat(
                                    run_data['timestamp']
                                ),
                                version=run_data['version'],
                                environment=run_data['environment'],
                                branch=run_data['branch'],
                                commit_hash=run_data['commit_hash'],
                                metrics=run_data['metrics'],
                                test_duration_seconds=run_data[
                                    'test_duration_seconds'
                                ],
                                test_metadata=run_data.get(
                                    'test_metadata', {}
                                ),
                            )
                        )

            logger.info(
                f"Loaded {len(self.baselines)} baselines and {len(self.test_runs)} test runs"
            )

        except Exception as e:
            logger.error(f"Error loading historical data: {e}")

    def save_data(self):
        """Save current data to files."""
        try:
            # Save baselines
            baselines_data = {}
            for metric_name, baseline in self.baselines.items():
                baselines_data[metric_name] = {
                    'metric_name': baseline.metric_name,
                    'baseline_value': baseline.baseline_value,
                    'baseline_std': baseline.baseline_std,
                    'sample_count': baseline.sample_count,
                    'confidence_interval': list(baseline.confidence_interval),
                    'last_updated': baseline.last_updated.isoformat(),
                    'version': baseline.version,
                    'environment': baseline.environment,
                    'metadata': baseline.metadata,
                }

            with open(self.data_directory / "baselines.json", 'w') as f:
                json.dump(baselines_data, f, indent=2)

            # Save test runs
            test_runs_data = []
            for run in self.test_runs:
                test_runs_data.append(
                    {
                        'run_id': run.run_id,
                        'timestamp': run.timestamp.isoformat(),
                        'version': run.version,
                        'environment': run.environment,
                        'branch': run.branch,
                        'commit_hash': run.commit_hash,
                        'metrics': run.metrics,
                        'test_duration_seconds': run.test_duration_seconds,
                        'test_metadata': run.test_metadata,
                    }
                )

            with open(self.data_directory / "test_runs.json", 'w') as f:
                json.dump(test_runs_data, f, indent=2)

            logger.info("Performance data saved successfully")

        except Exception as e:
            logger.error(f"Error saving data: {e}")

    def add_performance_test_run(
        self,
        version: str,
        environment: str,
        branch: str,
        commit_hash: str,
        metrics: Dict[str, float],
        test_duration_seconds: float,
        test_metadata: Dict[str, Any] = None,
    ) -> str:
        """Add a new performance test run."""
        run_id = str(uuid.uuid4())

        test_run = PerformanceTestRun(
            run_id=run_id,
            timestamp=datetime.now(),
            version=version,
            environment=environment,
            branch=branch,
            commit_hash=commit_hash,
            metrics=metrics,
            test_duration_seconds=test_duration_seconds,
            test_metadata=test_metadata or {},
        )

        self.test_runs.append(test_run)

        # Update baselines if needed
        self._update_baselines(test_run)

        # Save data
        self.save_data()

        logger.info(
            f"Added performance test run: {run_id} for version {version}"
        )
        return run_id

    def _update_baselines(self, test_run: PerformanceTestRun):
        """Update baselines with new test run data."""
        for metric_name, value in test_run.metrics.items():
            if metric_name not in self.baselines:
                # Create new baseline
                self._create_baseline(
                    metric_name, test_run.environment, test_run.version
                )
            else:
                # Check if baseline needs updating
                baseline = self.baselines[metric_name]

                # Get recent values for this metric
                recent_values = self._get_recent_metric_values(
                    metric_name, test_run.environment
                )

                if (
                    len(recent_values)
                    >= self.regression_config['min_sample_size']
                ):
                    # Calculate new baseline statistics
                    new_baseline_value = statistics.mean(recent_values)
                    new_baseline_std = (
                        statistics.stdev(recent_values)
                        if len(recent_values) > 1
                        else 0
                    )

                    # Check if baseline should be updated
                    change_percentage = (
                        abs(new_baseline_value - baseline.baseline_value)
                        / baseline.baseline_value
                    )

                    if (
                        change_percentage
                        > self.regression_config['baseline_update_threshold']
                    ):
                        # Update baseline
                        confidence_interval = (
                            self._calculate_confidence_interval(recent_values)
                        )

                        self.baselines[metric_name] = PerformanceBaseline(
                            metric_name=metric_name,
                            baseline_value=new_baseline_value,
                            baseline_std=new_baseline_std,
                            sample_count=len(recent_values),
                            confidence_interval=confidence_interval,
                            last_updated=datetime.now(),
                            version=test_run.version,
                            environment=test_run.environment,
                            metadata={'updated_from_run': test_run.run_id},
                        )

                        logger.info(
                            f"Updated baseline for {metric_name}: {new_baseline_value:.3f} "
                            f"(change: {change_percentage*100:.1f}%)"
                        )

    def _create_baseline(
        self, metric_name: str, environment: str, version: str
    ):
        """Create a new baseline for a metric."""
        recent_values = self._get_recent_metric_values(
            metric_name, environment
        )

        if len(recent_values) >= self.regression_config['min_sample_size']:
            baseline_value = statistics.mean(recent_values)
            baseline_std = (
                statistics.stdev(recent_values)
                if len(recent_values) > 1
                else 0
            )
            confidence_interval = self._calculate_confidence_interval(
                recent_values
            )

            self.baselines[metric_name] = PerformanceBaseline(
                metric_name=metric_name,
                baseline_value=baseline_value,
                baseline_std=baseline_std,
                sample_count=len(recent_values),
                confidence_interval=confidence_interval,
                last_updated=datetime.now(),
                version=version,
                environment=environment,
                metadata={'created_from_samples': len(recent_values)},
            )

            logger.info(
                f"Created new baseline for {metric_name}: {baseline_value:.3f}"
            )

    def _get_recent_metric_values(
        self, metric_name: str, environment: str
    ) -> List[float]:
        """Get recent values for a metric."""
        cutoff_date = datetime.now() - timedelta(
            days=self.regression_config['lookback_window_days']
        )

        values = []
        for run in self.test_runs:
            if (
                run.timestamp >= cutoff_date
                and run.environment == environment
                and metric_name in run.metrics
            ):
                values.append(run.metrics[metric_name])

        return values

    def _calculate_confidence_interval(
        self, values: List[float], confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for a set of values."""
        if len(values) < 2:
            return (0, 0)

        mean = statistics.mean(values)
        std_err = statistics.stdev(values) / np.sqrt(len(values))

        # Use t-distribution for small samples
        if len(values) < 30:
            t_value = stats.t.ppf((1 + confidence) / 2, len(values) - 1)
        else:
            t_value = stats.norm.ppf((1 + confidence) / 2)

        margin_of_error = t_value * std_err

        return (mean - margin_of_error, mean + margin_of_error)

    def detect_regressions(
        self, test_run_id: str, comparison_environment: str = None
    ) -> List[RegressionDetectionResult]:
        """Detect performance regressions for a test run."""
        # Find the test run
        test_run = None
        for run in self.test_runs:
            if run.run_id == test_run_id:
                test_run = run
                break

        if not test_run:
            raise ValueError(f"Test run not found: {test_run_id}")

        environment = comparison_environment or test_run.environment
        results = []

        for metric_name, current_value in test_run.metrics.items():
            if metric_name not in self.baselines:
                logger.warning(f"No baseline found for metric: {metric_name}")
                continue

            baseline = self.baselines[metric_name]

            # Skip if baseline is for different environment
            if baseline.environment != environment:
                continue

            # Calculate regression
            regression_result = self._analyze_regression(
                metric_name, current_value, baseline, test_run
            )

            results.append(regression_result)

        return results

    def _analyze_regression(
        self,
        metric_name: str,
        current_value: float,
        baseline: PerformanceBaseline,
        test_run: PerformanceTestRun,
    ) -> RegressionDetectionResult:
        """Analyze a single metric for regression."""
        # Calculate deviation
        if baseline.baseline_value == 0:
            deviation_percentage = 0
        else:
            deviation_percentage = (
                current_value - baseline.baseline_value
            ) / baseline.baseline_value

        # Determine if regression is detected
        abs_deviation = abs(deviation_percentage)
        regression_detected = (
            abs_deviation > self.regression_config['max_acceptable_deviation']
        )

        # Calculate confidence score using statistical significance
        if baseline.baseline_std > 0:
            z_score = (
                abs(current_value - baseline.baseline_value)
                / baseline.baseline_std
            )
            confidence_score = stats.norm.cdf(z_score)
        else:
            confidence_score = 0.5  # Neutral confidence if no std dev

        # Determine severity
        if abs_deviation > self.regression_config['critical_deviation']:
            severity = 'critical'
        elif (
            abs_deviation > self.regression_config['max_acceptable_deviation']
        ):
            if abs_deviation > 0.20:  # 20%
                severity = 'major'
            else:
                severity = 'minor'
        else:
            severity = 'negligible'

        # Calculate statistical significance
        if baseline.sample_count > 1:
            # Use t-test for small samples
            recent_values = self._get_recent_metric_values(
                metric_name, test_run.environment
            )
            if len(recent_values) > 1:
                t_stat, p_value = stats.ttest_1samp(
                    recent_values, baseline.baseline_value
                )
                statistical_significance = 1 - p_value
            else:
                statistical_significance = 0.5
        else:
            statistical_significance = 0.5

        return RegressionDetectionResult(
            metric_name=metric_name,
            current_value=current_value,
            baseline_value=baseline.baseline_value,
            deviation_percentage=deviation_percentage,
            regression_detected=regression_detected,
            confidence_score=confidence_score,
            severity=severity,
            statistical_significance=statistical_significance,
        )

    def detect_anomalies(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Detect anomalies in metrics using machine learning."""
        if not self.ml_models_trained:
            self._train_anomaly_detection_models()

        if not self.ml_models_trained:
            return {}  # No models available

        # Prepare data for anomaly detection
        metric_names = list(metrics.keys())
        values = [metrics[name] for name in metric_names]

        try:
            # Scale the values
            scaled_values = self.scaler.transform([values])

            # Detect anomalies
            anomaly_scores = self.anomaly_detector.decision_function(
                scaled_values
            )
            self.anomaly_detector.predict(scaled_values)

            # Return anomaly scores for each metric
            anomaly_results = {}
            for i, metric_name in enumerate(metric_names):
                anomaly_results[metric_name] = float(anomaly_scores[0])

            return anomaly_results

        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return {}

    def _train_anomaly_detection_models(self):
        """Train machine learning models for anomaly detection."""
        if len(self.test_runs) < 20:  # Need minimum data for training
            logger.warning(
                "Insufficient data for training anomaly detection models"
            )
            return

        try:
            # Prepare training data
            all_metrics = set()
            for run in self.test_runs:
                all_metrics.update(run.metrics.keys())

            training_data = []
            for run in self.test_runs:
                row = []
                for metric in sorted(all_metrics):
                    row.append(run.metrics.get(metric, 0))
                training_data.append(row)

            # Train scaler
            self.scaler.fit(training_data)

            # Train anomaly detector
            scaled_data = self.scaler.transform(training_data)
            self.anomaly_detector.fit(scaled_data)

            self.ml_models_trained = True
            logger.info("Anomaly detection models trained successfully")

        except Exception as e:
            logger.error(f"Error training anomaly detection models: {e}")

    def generate_regression_report(
        self, test_run_id: str, comparison_environment: str = None
    ) -> Dict[str, Any]:
        """Generate a comprehensive regression report."""
        # Get test run
        test_run = None
        for run in self.test_runs:
            if run.run_id == test_run_id:
                test_run = run
                break

        if not test_run:
            return {'error': f'Test run not found: {test_run_id}'}

        # Detect regressions
        regression_results = self.detect_regressions(
            test_run_id, comparison_environment
        )

        # Detect anomalies
        anomaly_results = self.detect_anomalies(test_run.metrics)

        # Count regression types
        regression_counts = {
            'critical': 0,
            'major': 0,
            'minor': 0,
            'negligible': 0,
        }

        for result in regression_results:
            if result.regression_detected:
                regression_counts[result.severity] += 1

        # Overall assessment
        overall_status = 'pass'
        if regression_counts['critical'] > 0:
            overall_status = 'fail'
        elif regression_counts['major'] > 0:
            overall_status = 'warning'
        elif regression_counts['minor'] > 0:
            overall_status = 'caution'

        # Generate report
        report = {
            'test_run_info': {
                'run_id': test_run.run_id,
                'timestamp': test_run.timestamp.isoformat(),
                'version': test_run.version,
                'environment': test_run.environment,
                'branch': test_run.branch,
                'commit_hash': test_run.commit_hash,
                'test_duration_seconds': test_run.test_duration_seconds,
            },
            'regression_analysis': {
                'overall_status': overall_status,
                'total_metrics_analyzed': len(regression_results),
                'regressions_detected': sum(
                    1 for r in regression_results if r.regression_detected
                ),
                'regression_counts': regression_counts,
                'detailed_results': [
                    {
                        'metric_name': r.metric_name,
                        'current_value': r.current_value,
                        'baseline_value': r.baseline_value,
                        'deviation_percentage': r.deviation_percentage * 100,
                        'regression_detected': r.regression_detected,
                        'severity': r.severity,
                        'confidence_score': r.confidence_score,
                        'statistical_significance': r.statistical_significance,
                    }
                    for r in regression_results
                ],
            },
            'anomaly_detection': {
                'anomalies_detected': len(
                    [
                        score
                        for score in anomaly_results.values()
                        if score < -0.5
                    ]
                ),
                'anomaly_scores': anomaly_results,
            },
            'recommendations': self._generate_regression_recommendations(
                regression_results, test_run
            ),
            'baseline_info': {
                'total_baselines': len(self.baselines),
                'baseline_ages': {
                    name: (datetime.now() - baseline.last_updated).days
                    for name, baseline in self.baselines.items()
                },
            },
        }

        return report

    def _generate_regression_recommendations(
        self,
        regression_results: List[RegressionDetectionResult],
        test_run: PerformanceTestRun,
    ) -> List[str]:
        """Generate recommendations based on regression analysis."""
        recommendations = []

        # Critical regressions
        critical_regressions = [
            r for r in regression_results if r.severity == 'critical'
        ]
        if critical_regressions:
            recommendations.append(
                f"CRITICAL: {len(critical_regressions)} critical performance regressions detected. "
                f"Immediate investigation required before deployment."
            )

        # Major regressions
        major_regressions = [
            r for r in regression_results if r.severity == 'major'
        ]
        if major_regressions:
            recommendations.append(
                f"MAJOR: {len(major_regressions)} major performance regressions detected. "
                f"Review and optimize before deployment."
            )

        # Specific metric recommendations
        response_time_regressions = [
            r
            for r in regression_results
            if 'response_time' in r.metric_name.lower()
            and r.regression_detected
        ]
        if response_time_regressions:
            recommendations.append(
                "Response time regressions detected. Check for inefficient queries, "
                "network issues, or resource contention."
            )

        memory_regressions = [
            r
            for r in regression_results
            if 'memory' in r.metric_name.lower() and r.regression_detected
        ]
        if memory_regressions:
            recommendations.append(
                "Memory usage regressions detected. Look for memory leaks, "
                "excessive object creation, or caching issues."
            )

        throughput_regressions = [
            r
            for r in regression_results
            if 'throughput' in r.metric_name.lower()
            or 'rps' in r.metric_name.lower()
            and r.regression_detected
        ]
        if throughput_regressions:
            recommendations.append(
                "Throughput regressions detected. Investigate bottlenecks in "
                "processing pipeline or database queries."
            )

        # Baseline recommendations
        old_baselines = [
            name
            for name, baseline in self.baselines.items()
            if (datetime.now() - baseline.last_updated).days > 30
        ]
        if old_baselines:
            recommendations.append(
                f"Consider updating baselines for {len(old_baselines)} metrics "
                f"that haven't been updated in over 30 days."
            )

        # General recommendations
        if not any(r.regression_detected for r in regression_results):
            recommendations.append(
                "No significant performance regressions detected. "
                "Performance is stable compared to baseline."
            )

        return recommendations

    def get_performance_trends(
        self, metric_name: str, environment: str, days: int = 30
    ) -> Dict[str, Any]:
        """Get performance trends for a specific metric."""
        cutoff_date = datetime.now() - timedelta(days=days)

        # Get data points
        data_points = []
        for run in self.test_runs:
            if (
                run.timestamp >= cutoff_date
                and run.environment == environment
                and metric_name in run.metrics
            ):
                data_points.append(
                    {
                        'timestamp': run.timestamp,
                        'value': run.metrics[metric_name],
                        'version': run.version,
                        'commit_hash': run.commit_hash,
                    }
                )

        if len(data_points) < 2:
            return {'error': 'Insufficient data for trend analysis'}

        # Sort by timestamp
        data_points.sort(key=lambda x: x['timestamp'])

        # Calculate trend
        values = [point['value'] for point in data_points]
        timestamps = [point['timestamp'].timestamp() for point in data_points]

        # Linear regression for trend
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            timestamps, values
        )

        # Determine trend direction
        if abs(slope) < 0.001:  # Very small slope
            trend_direction = 'stable'
        elif slope > 0:
            trend_direction = 'increasing'
        else:
            trend_direction = 'decreasing'

        # Calculate volatility
        volatility = statistics.stdev(values) if len(values) > 1 else 0

        return {
            'metric_name': metric_name,
            'environment': environment,
            'period_days': days,
            'data_points': len(data_points),
            'trend_direction': trend_direction,
            'slope': slope,
            'r_squared': r_value**2,
            'statistical_significance': 1 - p_value,
            'volatility': volatility,
            'current_value': values[-1],
            'average_value': statistics.mean(values),
            'min_value': min(values),
            'max_value': max(values),
            'data_series': [
                {
                    'timestamp': point['timestamp'].isoformat(),
                    'value': point['value'],
                    'version': point['version'],
                }
                for point in data_points
            ],
        }

    def export_performance_data(self, filename: str):
        """Export all performance data to a file."""
        data = {
            'export_timestamp': datetime.now().isoformat(),
            'baselines': {
                name: {
                    'metric_name': baseline.metric_name,
                    'baseline_value': baseline.baseline_value,
                    'baseline_std': baseline.baseline_std,
                    'sample_count': baseline.sample_count,
                    'confidence_interval': list(baseline.confidence_interval),
                    'last_updated': baseline.last_updated.isoformat(),
                    'version': baseline.version,
                    'environment': baseline.environment,
                    'metadata': baseline.metadata,
                }
                for name, baseline in self.baselines.items()
            },
            'test_runs': [
                {
                    'run_id': run.run_id,
                    'timestamp': run.timestamp.isoformat(),
                    'version': run.version,
                    'environment': run.environment,
                    'branch': run.branch,
                    'commit_hash': run.commit_hash,
                    'metrics': run.metrics,
                    'test_duration_seconds': run.test_duration_seconds,
                    'test_metadata': run.test_metadata,
                }
                for run in self.test_runs
            ],
            'configuration': self.regression_config,
            'statistics': {
                'total_test_runs': len(self.test_runs),
                'total_baselines': len(self.baselines),
                'environments': list(
                    set(run.environment for run in self.test_runs)
                ),
                'metrics_tracked': list(
                    set(
                        metric
                        for run in self.test_runs
                        for metric in run.metrics.keys()
                    )
                ),
            },
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Performance data exported to {filename}")


# Example usage and demonstration
async def demo_regression_detection():
    """Demonstrate regression detection capabilities."""
    print("=" * 80)
    print("AUTOMATED PERFORMANCE REGRESSION DETECTION DEMO")
    print("=" * 80)

    detector = PerformanceRegressionDetector()

    # Add some sample test runs
    print("Adding sample performance test runs...")

    # Baseline runs (good performance)
    for i in range(15):
        metrics = {
            'api_response_time_ms': 150 + np.random.normal(0, 10),
            'api_requests_per_second': 100 + np.random.normal(0, 5),
            'memory_usage_mb': 512 + np.random.normal(0, 20),
            'cpu_usage_percent': 25 + np.random.normal(0, 3),
            'db_query_time_ms': 25 + np.random.normal(0, 5),
        }

        detector.add_performance_test_run(
            version=f"v1.0.{i}",
            environment="production",
            branch="main",
            commit_hash=f"abc123{i:02d}",
            metrics=metrics,
            test_duration_seconds=300,
        )

    print(f"Added {len(detector.test_runs)} baseline test runs")

    # Add a test run with regressions
    print("\nAdding test run with performance regressions...")

    regressed_metrics = {
        'api_response_time_ms': 250,  # 67% slower
        'api_requests_per_second': 75,  # 25% slower
        'memory_usage_mb': 720,  # 40% more memory
        'cpu_usage_percent': 45,  # 80% more CPU
        'db_query_time_ms': 60,  # 140% slower
    }

    regression_run_id = detector.add_performance_test_run(
        version="v1.1.0",
        environment="production",
        branch="feature/new-feature",
        commit_hash="def456",
        metrics=regressed_metrics,
        test_duration_seconds=300,
    )

    # Detect regressions
    print(f"\nAnalyzing regressions for test run: {regression_run_id}")

    regression_results = detector.detect_regressions(regression_run_id)

    print(f"Found {len(regression_results)} metrics to analyze")

    # Generate comprehensive report
    report = detector.generate_regression_report(regression_run_id)

    # Display results
    print("\n--- REGRESSION ANALYSIS REPORT ---")
    print(
        f"Overall Status: {report['regression_analysis']['overall_status'].upper()}"
    )
    print(
        f"Total Metrics Analyzed: {report['regression_analysis']['total_metrics_analyzed']}"
    )
    print(
        f"Regressions Detected: {report['regression_analysis']['regressions_detected']}"
    )

    print("\nRegression Counts:")
    for severity, count in report['regression_analysis'][
        'regression_counts'
    ].items():
        if count > 0:
            print(f"  {severity.upper()}: {count}")

    print("\nDetailed Results:")
    for result in report['regression_analysis']['detailed_results']:
        if result['regression_detected']:
            print(
                f"  🔴 {result['metric_name']}: {result['current_value']:.1f} "
                f"(baseline: {result['baseline_value']:.1f}, "
                f"deviation: {result['deviation_percentage']:.1f}%, "
                f"severity: {result['severity']})"
            )
        else:
            print(
                f"  ✅ {result['metric_name']}: {result['current_value']:.1f} "
                f"(baseline: {result['baseline_value']:.1f}, "
                f"deviation: {result['deviation_percentage']:.1f}%)"
            )

    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")

    # Show performance trends
    print("\n--- PERFORMANCE TRENDS ---")

    trends = detector.get_performance_trends(
        'api_response_time_ms', 'production', 30
    )
    if 'error' not in trends:
        print(
            f"Response Time Trend: {trends['trend_direction']} "
            f"(R²: {trends['r_squared']:.3f}, "
            f"Current: {trends['current_value']:.1f}ms, "
            f"Average: {trends['average_value']:.1f}ms)"
        )

    # Export data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_filename = f"performance_regression_data_{timestamp}.json"
    detector.export_performance_data(export_filename)

    print(f"\nPerformance data exported to: {export_filename}")

    print("\n" + "=" * 80)
    print("REGRESSION DETECTION DEMO COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(demo_regression_detection())
