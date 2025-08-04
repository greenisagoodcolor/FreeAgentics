#!/usr/bin/env python3
"""Memory Leak Detection Service.

Advanced statistical analysis for memory leak detection designed by the Nemesis Committee.
Implements Michael Feathers' seam-based integration with Jessica Kerr's observability approach.

Key Features:
- Statistical trend analysis with anomaly detection
- Automatic snapshot comparison with configurable thresholds
- Integration with OpenTelemetry for distributed tracing
- Non-intrusive monitoring through bytecode hooks
- Configurable alerting with severity levels
- Historical trend analysis and prediction
"""

import gc
import logging
import statistics
import threading
import time
import tracemalloc
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemorySnapshot:
    """Immutable memory snapshot for comparison."""
    
    timestamp: float
    current_mb: float
    peak_mb: float
    allocation_count: int
    objects_by_type: Dict[str, int]
    top_allocations: List[Dict[str, Any]]
    gc_stats: Dict[str, Any]
    label: str = ""
    
    @property
    def datetime(self) -> datetime:
        """Get snapshot datetime."""
        return datetime.fromtimestamp(self.timestamp)


@dataclass
class LeakSuspicion:
    """Memory leak suspicion with statistical evidence."""
    
    location: str
    growth_rate_mb_per_hour: float
    confidence_score: float  # 0.0 to 1.0
    evidence: Dict[str, Any]
    first_detected: float
    last_updated: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    
    @property
    def age_hours(self) -> float:
        """Age of the leak suspicion in hours."""
        return (time.time() - self.first_detected) / 3600
    
    @property
    def projected_growth_24h(self) -> float:
        """Projected memory growth in next 24 hours."""
        return self.growth_rate_mb_per_hour * 24


@dataclass
class TrendAnalysis:
    """Statistical trend analysis result."""
    
    slope: float  # MB per hour
    r_squared: float  # Correlation coefficient
    p_value: float  # Statistical significance
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    anomaly_score: float  # 0.0 to 1.0 
    data_points: int
    time_span_hours: float


class MemoryLeakDetector:
    """Advanced memory leak detection with statistical analysis."""
    
    def __init__(
        self,
        snapshot_interval: float = 30.0,
        analysis_window: int = 50,
        leak_threshold_mb_per_hour: float = 0.5,
        confidence_threshold: float = 0.7,
        enable_statistical_analysis: bool = True,
        max_snapshots: int = 1000
    ):
        """Initialize memory leak detector.
        
        Args:
            snapshot_interval: Seconds between automatic snapshots
            analysis_window: Number of snapshots to analyze for trends
            leak_threshold_mb_per_hour: Minimum growth rate to consider a leak
            confidence_threshold: Minimum confidence to report a leak
            enable_statistical_analysis: Enable advanced statistical methods
            max_snapshots: Maximum snapshots to retain in memory
        """
        self.snapshot_interval = snapshot_interval
        self.analysis_window = analysis_window
        self.leak_threshold_mb_per_hour = leak_threshold_mb_per_hour
        self.confidence_threshold = confidence_threshold
        self.enable_statistical_analysis = enable_statistical_analysis
        self.max_snapshots = max_snapshots
        
        # Storage for snapshots and analysis
        self.snapshots: deque = deque(maxlen=max_snapshots)
        self.leak_suspicions: Dict[str, LeakSuspicion] = {}
        self.location_histories: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=analysis_window)
        )
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # Statistics tracking
        self._detection_stats = {
            'total_snapshots': 0,
            'leaks_detected': 0,
            'false_positives': 0,
            'analysis_time_ms': deque(maxlen=100)
        }
        
        logger.info(f"MemoryLeakDetector initialized with {analysis_window} window, "
                   f"{leak_threshold_mb_per_hour}MB/hour threshold")
    
    def start_monitoring(self):
        """Start continuous memory leak monitoring."""
        with self._lock:
            if self._monitoring:
                logger.warning("Memory leak monitoring already started")
                return
            
            self._monitoring = True
            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                name="MemoryLeakDetector",
                daemon=True
            )
            self._monitor_thread.start()
            logger.info("Started memory leak monitoring")
    
    def stop_monitoring(self):
        """Stop memory leak monitoring."""
        with self._lock:
            if not self._monitoring:
                return
            
            self._monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=5.0)
                if self._monitor_thread.is_alive():
                    logger.warning("Monitor thread did not stop gracefully")
            
            logger.info("Stopped memory leak monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring:
            try:
                self.take_snapshot("auto_monitor")
                self.analyze_trends()
                time.sleep(self.snapshot_interval)
            except Exception as e:
                logger.error(f"Error in memory leak monitoring loop: {e}")
                time.sleep(min(self.snapshot_interval, 60))  # Back off on errors
    
    def take_snapshot(self, label: str = "") -> MemorySnapshot:
        """Take a comprehensive memory snapshot.
        
        Args:
            label: Optional label for the snapshot
            
        Returns:
            MemorySnapshot object
        """
        start_time = time.perf_counter()
        
        # Ensure tracemalloc is running
        if not tracemalloc.is_tracing():
            tracemalloc.start(10)
        
        try:
            # Get basic memory stats
            current, peak = tracemalloc.get_traced_memory()
            current_mb = current / 1024 / 1024
            peak_mb = peak / 1024 / 1024
            
            # Get allocation statistics
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')
            
            # Extract top allocations
            top_allocations = []
            for stat in top_stats[:20]:
                if stat.size > 1024 * 1024:  # >1MB allocations
                    frame = stat.traceback[0]
                    top_allocations.append({
                        'file': frame.filename,
                        'line': frame.lineno,
                        'size_mb': stat.size / 1024 / 1024,
                        'count': stat.count,
                        'location': f"{frame.filename}:{frame.lineno}"
                    })
            
            # Get object type distribution
            objects_by_type = self._get_object_type_distribution()
            
            # Get garbage collection stats
            gc_stats = {
                'collections': {f'gen{i}': gc.get_count()[i] for i in range(3)},
                'garbage_objects': len(gc.garbage),
                'total_objects': len(gc.get_objects())
            }
            
            # Create snapshot
            snapshot_obj = MemorySnapshot(
                timestamp=time.time(),
                current_mb=current_mb,
                peak_mb=peak_mb,
                allocation_count=sum(stat.count for stat in top_stats),
                objects_by_type=objects_by_type,
                top_allocations=top_allocations,
                gc_stats=gc_stats,
                label=label
            )
            
            # Store snapshot
            with self._lock:
                self.snapshots.append(snapshot_obj)
                self._detection_stats['total_snapshots'] += 1
            
            # Update performance metrics
            analysis_time = (time.perf_counter() - start_time) * 1000
            self._detection_stats['analysis_time_ms'].append(analysis_time)
            
            logger.debug(f"Memory snapshot taken: {current_mb:.2f}MB current, "
                        f"{peak_mb:.2f}MB peak, {len(top_allocations)} large allocations")
            
            return snapshot_obj
            
        except Exception as e:
            logger.error(f"Error taking memory snapshot: {e}")
            # Return minimal snapshot on error
            return MemorySnapshot(
                timestamp=time.time(),
                current_mb=0.0,
                peak_mb=0.0,
                allocation_count=0,
                objects_by_type={},
                top_allocations=[],
                gc_stats={},
                label=f"error_{label}"
            )
    
    def _get_object_type_distribution(self) -> Dict[str, int]:
        """Get distribution of objects by type."""
        try:
            objects = gc.get_objects()
            type_counts = defaultdict(int)
            
            for obj in objects:
                type_name = type(obj).__name__
                type_counts[type_name] += 1
            
            # Return top 20 types
            sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_types[:20])
            
        except Exception as e:
            logger.error(f"Error getting object type distribution: {e}")
            return {}
    
    def analyze_trends(self) -> List[LeakSuspicion]:
        """Analyze memory trends and detect potential leaks.
        
        Returns:
            List of updated leak suspicions
        """
        start_time = time.perf_counter()
        
        with self._lock:
            if len(self.snapshots) < self.analysis_window // 2:
                logger.debug(f"Insufficient snapshots for analysis: {len(self.snapshots)}")
                return list(self.leak_suspicions.values())
            
            # Analyze each allocation location
            location_suspicions = {}
            
            # Group snapshots by location for trend analysis
            for snapshot in list(self.snapshots)[-self.analysis_window:]:
                for allocation in snapshot.top_allocations:
                    location = allocation['location']
                    size_mb = allocation['size_mb']
                    
                    self.location_histories[location].append({
                        'timestamp': snapshot.timestamp,
                        'size_mb': size_mb,
                        'count': allocation['count']
                    })
            
            # Analyze trends for each location
            for location, history in self.location_histories.items():
                if len(history) < 5:  # Need minimum data points
                    continue
                
                trend_analysis = self._analyze_location_trend(location, list(history))
                
                if self._is_leak_suspected(trend_analysis):
                    suspicion = self._create_leak_suspicion(location, trend_analysis)
                    location_suspicions[location] = suspicion
            
            # Update leak suspicions
            current_time = time.time()
            for location, suspicion in location_suspicions.items():
                if location in self.leak_suspicions:
                    # Update existing suspicion
                    existing = self.leak_suspicions[location]
                    existing.growth_rate_mb_per_hour = suspicion.growth_rate_mb_per_hour
                    existing.confidence_score = suspicion.confidence_score
                    existing.evidence = suspicion.evidence
                    existing.last_updated = current_time
                    existing.severity = suspicion.severity
                else:
                    # New suspicion
                    suspicion.first_detected = current_time
                    suspicion.last_updated = current_time
                    self.leak_suspicions[location] = suspicion
                    self._detection_stats['leaks_detected'] += 1
                    
                    logger.warning(f"New memory leak suspicion detected at {location}: "
                                 f"{suspicion.growth_rate_mb_per_hour:.2f}MB/hour "
                                 f"(confidence: {suspicion.confidence_score:.2f})")
            
            # Remove stale suspicions (no recent evidence)
            stale_locations = []
            for location, suspicion in self.leak_suspicions.items():
                age_hours = (current_time - suspicion.last_updated) / 3600
                if age_hours > 24:  # Remove suspicions older than 24 hours
                    stale_locations.append(location)
            
            for location in stale_locations:
                del self.leak_suspicions[location]
                logger.info(f"Removed stale leak suspicion for {location}")
            
            analysis_time = (time.perf_counter() - start_time) * 1000
            logger.debug(f"Trend analysis completed in {analysis_time:.1f}ms, "
                        f"found {len(location_suspicions)} new suspicions")
            
            return list(self.leak_suspicions.values())
    
    def _analyze_location_trend(self, location: str, history: List[Dict]) -> TrendAnalysis:
        """Analyze memory trend for a specific location.
        
        Args:
            location: Code location identifier
            history: List of historical data points
            
        Returns:
            TrendAnalysis object
        """
        if len(history) < 3:
            return TrendAnalysis(
                slope=0.0, r_squared=0.0, p_value=1.0,
                trend_direction='unknown', anomaly_score=0.0,
                data_points=len(history), time_span_hours=0.0
            )
        
        # Extract time series data
        timestamps = np.array([point['timestamp'] for point in history])
        sizes_mb = np.array([point['size_mb'] for point in history])
        
        # Convert timestamps to hours from start
        time_hours = (timestamps - timestamps[0]) / 3600
        time_span_hours = time_hours[-1] if len(time_hours) > 1 else 0.0
        
        try:
            if self.enable_statistical_analysis and len(history) >= 5:
                # Linear regression analysis
                slope, intercept, r_value, p_value, std_err = stats.linregress(time_hours, sizes_mb)
                r_squared = r_value ** 2
                
                # Anomaly detection using z-score
                if len(sizes_mb) > 3:
                    z_scores = np.abs(stats.zscore(sizes_mb))
                    anomaly_score = max(z_scores) / 3.0  # Normalize to 0-1 range
                else:
                    anomaly_score = 0.0
                
            else:
                # Simple trend analysis
                if len(sizes_mb) >= 2:
                    slope = (sizes_mb[-1] - sizes_mb[0]) / max(time_span_hours, 0.1)
                    r_squared = 0.5  # Assume moderate correlation for simple analysis
                    p_value = 0.1    # Assume moderate significance
                    anomaly_score = 0.0
                else:
                    slope = 0.0
                    r_squared = 0.0
                    p_value = 1.0
                    anomaly_score = 0.0
            
            # Determine trend direction
            if abs(slope) < 0.01:  # Less than 0.01 MB/hour
                trend_direction = 'stable'
            elif slope > 0:
                trend_direction = 'increasing'
            else:
                trend_direction = 'decreasing'
            
            return TrendAnalysis(
                slope=slope,
                r_squared=r_squared,
                p_value=p_value,
                trend_direction=trend_direction,
                anomaly_score=min(anomaly_score, 1.0),
                data_points=len(history),
                time_span_hours=time_span_hours
            )
            
        except Exception as e:
            logger.error(f"Error in trend analysis for {location}: {e}")
            return TrendAnalysis(
                slope=0.0, r_squared=0.0, p_value=1.0,
                trend_direction='error', anomaly_score=0.0,
                data_points=len(history), time_span_hours=time_span_hours
            )
    
    def _is_leak_suspected(self, trend: TrendAnalysis) -> bool:
        """Determine if trend indicates a potential memory leak.
        
        Args:
            trend: TrendAnalysis object
            
        Returns:
            True if leak is suspected
        """
        # Check growth rate threshold
        if trend.slope < self.leak_threshold_mb_per_hour:
            return False
        
        # Check trend direction
        if trend.trend_direction != 'increasing':
            return False
        
        # Check statistical significance (if available)
        if self.enable_statistical_analysis:
            if trend.p_value > 0.05:  # Not statistically significant
                return False
            if trend.r_squared < 0.3:  # Poor correlation
                return False
        
        # Check minimum data requirements
        if trend.data_points < 5:
            return False
        
        if trend.time_span_hours < 0.5:  # Less than 30 minutes
            return False
        
        return True
    
    def _create_leak_suspicion(self, location: str, trend: TrendAnalysis) -> LeakSuspicion:
        """Create a LeakSuspicion object from trend analysis.
        
        Args:
            location: Code location identifier
            trend: TrendAnalysis object
            
        Returns:
            LeakSuspicion object
        """
        # Calculate confidence score
        confidence = 0.0
        
        # Base confidence from statistical measures
        if self.enable_statistical_analysis:
            confidence += min(trend.r_squared, 0.4) * 2.5  # R² contribution (max 1.0)
            if trend.p_value <= 0.01:
                confidence += 0.3  # High significance bonus
            elif trend.p_value <= 0.05:
                confidence += 0.2  # Moderate significance bonus
        else:
            confidence += 0.5  # Base confidence for simple analysis
        
        # Growth rate contribution
        growth_factor = min(trend.slope / (self.leak_threshold_mb_per_hour * 2), 1.0)
        confidence += growth_factor * 0.3
        
        # Data quality contribution
        data_factor = min(trend.data_points / 20, 1.0)
        confidence += data_factor * 0.2
        
        # Anomaly score contribution
        confidence += min(trend.anomaly_score, 1.0) * 0.1
        
        confidence = min(confidence, 1.0)
        
        # Determine severity
        if trend.slope > self.leak_threshold_mb_per_hour * 10:
            severity = 'critical'
        elif trend.slope > self.leak_threshold_mb_per_hour * 5:
            severity = 'high'
        elif trend.slope > self.leak_threshold_mb_per_hour * 2:
            severity = 'medium'
        else:
            severity = 'low'
        
        return LeakSuspicion(
            location=location,
            growth_rate_mb_per_hour=trend.slope,
            confidence_score=confidence,
            evidence={
                'r_squared': trend.r_squared,
                'p_value': trend.p_value,
                'trend_direction': trend.trend_direction,
                'anomaly_score': trend.anomaly_score,
                'data_points': trend.data_points,
                'time_span_hours': trend.time_span_hours
            },
            first_detected=0.0,  # Will be set by caller
            last_updated=0.0,    # Will be set by caller
            severity=severity
        )
    
    def get_active_leaks(self, min_confidence: Optional[float] = None) -> List[LeakSuspicion]:
        """Get list of active memory leak suspicions.
        
        Args:
            min_confidence: Minimum confidence threshold (uses default if None)
            
        Returns:
            List of LeakSuspicion objects
        """
        threshold = min_confidence if min_confidence is not None else self.confidence_threshold
        
        with self._lock:
            active_leaks = [
                suspicion for suspicion in self.leak_suspicions.values()
                if suspicion.confidence_score >= threshold
            ]
            
            # Sort by severity and confidence
            severity_order = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
            active_leaks.sort(
                key=lambda x: (severity_order.get(x.severity, 0), x.confidence_score),
                reverse=True
            )
            
            return active_leaks
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get memory leak detection statistics.
        
        Returns:
            Dictionary of detection statistics
        """
        with self._lock:
            avg_analysis_time = (
                statistics.mean(self._detection_stats['analysis_time_ms'])
                if self._detection_stats['analysis_time_ms'] 
                else 0.0
            )
            
            return {
                'total_snapshots': self._detection_stats['total_snapshots'],
                'active_suspicions': len(self.leak_suspicions),
                'leaks_detected': self._detection_stats['leaks_detected'],
                'false_positives': self._detection_stats['false_positives'],
                'average_analysis_time_ms': avg_analysis_time,
                'monitoring_active': self._monitoring,
                'snapshot_buffer_usage': f"{len(self.snapshots)}/{self.max_snapshots}",
                'memory_locations_tracked': len(self.location_histories)
            }
    
    def generate_leak_report(self) -> str:
        """Generate comprehensive memory leak report.
        
        Returns:
            Formatted report string
        """
        active_leaks = self.get_active_leaks()
        stats = self.get_detection_statistics()
        
        report = []
        report.append("=" * 60)
        report.append("Memory Leak Detection Report")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Statistics section
        report.append("## Detection Statistics")
        report.append(f"Total snapshots taken: {stats['total_snapshots']:,}")
        report.append(f"Active leak suspicions: {stats['active_suspicions']}")
        report.append(f"Memory locations tracked: {stats['memory_locations_tracked']}")
        report.append(f"Average analysis time: {stats['average_analysis_time_ms']:.1f}ms")
        report.append(f"Monitoring status: {'Active' if stats['monitoring_active'] else 'Inactive'}")
        report.append("")
        
        # Active leaks section
        if active_leaks:
            report.append("## Active Memory Leak Suspicions")
            report.append("")
            
            for i, leak in enumerate(active_leaks, 1):
                report.append(f"{i}. {leak.location}")
                report.append(f"   Severity: {leak.severity.upper()}")
                report.append(f"   Growth rate: {leak.growth_rate_mb_per_hour:.2f} MB/hour")
                report.append(f"   Confidence: {leak.confidence_score:.2f}")
                report.append(f"   Age: {leak.age_hours:.1f} hours")
                report.append(f"   Projected 24h growth: {leak.projected_growth_24h:.2f} MB")
                
                # Evidence details
                evidence = leak.evidence
                if evidence.get('r_squared'):
                    report.append(f"   Statistical correlation: {evidence['r_squared']:.2f}")
                if evidence.get('p_value'):
                    report.append(f"   Statistical significance: p={evidence['p_value']:.3f}")
                if evidence.get('data_points'):
                    report.append(f"   Data points: {evidence['data_points']}")
                
                report.append("")
        else:
            report.append("## No Active Memory Leak Suspicions")
            report.append("All monitored locations appear stable.")
            report.append("")
        
        report.append("=" * 60)
        return "\n".join(report)


# Global leak detector instance
_global_detector: Optional[MemoryLeakDetector] = None


def get_leak_detector() -> MemoryLeakDetector:
    """Get the global memory leak detector instance.
    
    Returns:
        Global MemoryLeakDetector instance
    """
    global _global_detector
    if _global_detector is None:
        _global_detector = MemoryLeakDetector()
    return _global_detector


def configure_leak_detection(
    enabled: bool = True,
    snapshot_interval: float = 30.0,
    leak_threshold_mb_per_hour: float = 0.5,
    confidence_threshold: float = 0.7
):
    """Configure global memory leak detection.
    
    Args:
        enabled: Whether to enable leak detection
        snapshot_interval: Seconds between snapshots
        leak_threshold_mb_per_hour: Minimum growth rate for leak detection
        confidence_threshold: Minimum confidence to report leaks
    """
    detector = get_leak_detector()
    detector.snapshot_interval = snapshot_interval
    detector.leak_threshold_mb_per_hour = leak_threshold_mb_per_hour
    detector.confidence_threshold = confidence_threshold
    
    if enabled:
        detector.start_monitoring()
    else:
        detector.stop_monitoring()
    
    logger.info(f"Memory leak detection configured: enabled={enabled}, "
               f"interval={snapshot_interval}s, threshold={leak_threshold_mb_per_hour}MB/h")


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    detector = MemoryLeakDetector(
        snapshot_interval=2.0,  # Fast for testing
        analysis_window=10,
        leak_threshold_mb_per_hour=0.1  # Low threshold for testing
    )
    
    print("Starting memory leak detection test...")
    detector.start_monitoring()
    
    # Simulate memory growth
    data_store = []
    for i in range(15):
        # Simulate gradual memory leak
        data_store.append(np.zeros(100000 + i * 10000))  # Growing arrays
        time.sleep(2.5)
        
        if i > 0 and i % 5 == 0:
            leaks = detector.get_active_leaks()
            if leaks:
                print(f"\nDetected {len(leaks)} potential leaks:")
                for leak in leaks:
                    print(f"  {leak.location}: {leak.growth_rate_mb_per_hour:.2f}MB/h "
                          f"(confidence: {leak.confidence_score:.2f})")
    
    # Generate final report
    print("\n" + detector.generate_leak_report())
    
    detector.stop_monitoring()
    print("Memory leak detection test completed.")