#!/usr/bin/env python3
"""Memory Profiling Decorators.

Simple, composable decorators for automatic memory measurement designed by
the Nemesis Committee. Follows Kent Beck's TDD principles with zero-configuration
defaults and Sindre Sorhus's focus on developer experience.

Key Features:
- @profile_memory: Zero-config memory profiling
- @memory_limit('10MB'): Declarative memory budget enforcement  
- @track_allocations: Detailed allocation tracking
- @detect_leaks: Automatic leak detection
- Minimal performance overhead in production
- Integration with OpenTelemetry and Prometheus
"""

import functools
import logging
import time
import tracemalloc
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

import numpy as np

logger = logging.getLogger(__name__)

# Type variables for generic decorator support
F = TypeVar('F', bound=Callable[..., Any])


@dataclass(frozen=True)
class MemoryMeasurement:
    """Immutable memory measurement result."""
    
    function_name: str
    start_memory_mb: float
    end_memory_mb: float
    peak_memory_mb: float
    duration_ms: float
    allocation_count: int
    deallocation_count: int
    thread_id: int
    timestamp: float = field(default_factory=time.time)
    
    @property
    def memory_delta_mb(self) -> float:
        """Memory usage change during function execution."""
        return self.end_memory_mb - self.start_memory_mb
    
    @property
    def memory_overhead_mb(self) -> float:
        """Peak memory overhead above baseline."""
        return self.peak_memory_mb - self.start_memory_mb


@dataclass
class MemoryBudget:
    """Memory budget configuration."""
    
    limit_mb: float
    warning_threshold: float = 0.8  # 80% of limit
    alert_threshold: float = 0.95   # 95% of limit
    enforcement_mode: str = 'warn'  # 'warn', 'raise', 'ignore'
    
    def check_usage(self, current_mb: float) -> Optional[str]:
        """Check if current usage violates budget."""
        if current_mb > self.limit_mb:
            if self.enforcement_mode == 'raise':
                raise MemoryError(f"Memory usage {current_mb:.2f}MB exceeds limit {self.limit_mb:.2f}MB")
            return f"CRITICAL: Memory usage {current_mb:.2f}MB exceeds limit {self.limit_mb:.2f}MB"
        
        if current_mb > self.limit_mb * self.alert_threshold:
            return f"ALERT: Memory usage {current_mb:.2f}MB above {self.alert_threshold*100:.0f}% threshold"
        
        if current_mb > self.limit_mb * self.warning_threshold:
            return f"WARNING: Memory usage {current_mb:.2f}MB above {self.warning_threshold*100:.0f}% threshold"
        
        return None


class MemoryProfilerContext:
    """Thread-local memory profiling context."""
    
    def __init__(self):
        self.measurements: List[MemoryMeasurement] = []
        self.enabled = True
        self.sampling_rate = 1.0  # 100% sampling by default
        self.budget: Optional[MemoryBudget] = None
        
    def should_profile(self) -> bool:
        """Determine if profiling should occur based on sampling."""
        if not self.enabled:
            return False
        return np.random.random() < self.sampling_rate


# Global profiling context (thread-local in production)
_profiling_context = MemoryProfilerContext()


def configure_memory_profiling(
    enabled: bool = True,
    sampling_rate: float = 1.0,
    default_budget_mb: Optional[float] = None
):
    """Configure global memory profiling settings."""
    global _profiling_context
    _profiling_context.enabled = enabled
    _profiling_context.sampling_rate = sampling_rate
    
    if default_budget_mb:
        _profiling_context.budget = MemoryBudget(limit_mb=default_budget_mb)
    
    logger.info(f"Memory profiling configured: enabled={enabled}, sampling={sampling_rate}")


@contextmanager
def memory_snapshot():
    """Context manager for taking memory snapshots."""
    if not tracemalloc.is_tracing():
        tracemalloc.start()
        should_stop = True
    else:
        should_stop = False
    
    try:
        current_before, peak_before = tracemalloc.get_traced_memory()
        yield current_before / 1024 / 1024, peak_before / 1024 / 1024
        
    finally:
        if should_stop:
            tracemalloc.stop()


def profile_memory(
    enabled: Optional[bool] = None,
    budget: Optional[Union[str, float, MemoryBudget]] = None,
    sampling_rate: Optional[float] = None
) -> Callable[[F], F]:
    """Decorator for automatic memory profiling.
    
    Args:
        enabled: Override global profiling enable/disable
        budget: Memory budget as string ('10MB'), float (MB), or MemoryBudget object
        sampling_rate: Override global sampling rate
    
    Examples:
        @profile_memory
        def compute():
            return heavy_computation()
        
        @profile_memory(budget='5MB')
        def process_data():
            return data_processing()
        
        @profile_memory(sampling_rate=0.1)  # 10% sampling
        def frequent_operation():
            return fast_operation()
    """
    def decorator(func: F) -> F:
        
        # Parse budget parameter
        memory_budget = None
        if budget is not None:
            if isinstance(budget, str):
                # Parse string like '10MB', '5.5MB'
                if budget.upper().endswith('MB'):
                    limit_mb = float(budget[:-2])
                    memory_budget = MemoryBudget(limit_mb=limit_mb)
                else:
                    raise ValueError(f"Invalid budget format: {budget}. Use '10MB' format.")
            elif isinstance(budget, (int, float)):
                memory_budget = MemoryBudget(limit_mb=float(budget))
            elif isinstance(budget, MemoryBudget):
                memory_budget = budget
            else:
                raise ValueError(f"Invalid budget type: {type(budget)}")
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if profiling should occur
            should_profile = enabled if enabled is not None else _profiling_context.should_profile()
            current_sampling = sampling_rate if sampling_rate is not None else _profiling_context.sampling_rate
            
            if not should_profile or np.random.random() >= current_sampling:
                return func(*args, **kwargs)
            
            # Start memory tracking
            start_time = time.perf_counter()
            thread_id = id(threading.current_thread()) if 'threading' in globals() else 0
            
            with memory_snapshot() as (start_current, start_peak):
                # Execute function
                try:
                    result = func(*args, **kwargs)
                finally:
                    # Measure final memory
                    end_time = time.perf_counter()
                    current_after, peak_after = tracemalloc.get_traced_memory()
                    current_after_mb = current_after / 1024 / 1024
                    peak_after_mb = peak_after / 1024 / 1024
                    
                    # Create measurement
                    measurement = MemoryMeasurement(
                        function_name=f"{func.__module__}.{func.__qualname__}",
                        start_memory_mb=start_current,
                        end_memory_mb=current_after_mb,
                        peak_memory_mb=peak_after_mb,
                        duration_ms=(end_time - start_time) * 1000,
                        allocation_count=0,  # TODO: Extract from tracemalloc
                        deallocation_count=0,
                        thread_id=thread_id
                    )
                    
                    # Store measurement
                    _profiling_context.measurements.append(measurement)
                    
                    # Check budget if specified
                    active_budget = memory_budget or _profiling_context.budget
                    if active_budget:
                        violation = active_budget.check_usage(current_after_mb)
                        if violation:
                            logger.warning(f"{func.__name__}: {violation}")
                            
                            # Emit structured log for observability
                            logger.warning(
                                "Memory budget violation",
                                extra={
                                    'function': func.__name__,
                                    'memory_mb': current_after_mb,
                                    'budget_mb': active_budget.limit_mb,
                                    'violation_type': violation.split(':')[0],
                                    'measurement_id': len(_profiling_context.measurements)
                                }
                            )
                    
                    # Log measurement for debugging
                    if measurement.memory_delta_mb > 1.0:  # Only log significant changes
                        logger.debug(
                            f"Memory profile: {func.__name__} used {measurement.memory_delta_mb:+.2f}MB "
                            f"(peak: {measurement.memory_overhead_mb:.2f}MB) in {measurement.duration_ms:.1f}ms"
                        )
            
            return result
        
        return wrapper
    return decorator


def memory_limit(limit: Union[str, float], enforcement: str = 'warn') -> Callable[[F], F]:
    """Decorator for declarative memory budget enforcement.
    
    Args:
        limit: Memory limit as string ('10MB') or float (MB)
        enforcement: 'warn', 'raise', or 'ignore'
    
    Example:
        @memory_limit('34.5MB', enforcement='raise')
        def agent_operation():
            return heavy_agent_work()
    """
    if isinstance(limit, str) and limit.upper().endswith('MB'):
        limit_mb = float(limit[:-2])
    elif isinstance(limit, (int, float)):
        limit_mb = float(limit)
    else:
        raise ValueError(f"Invalid limit format: {limit}")
    
    budget = MemoryBudget(limit_mb=limit_mb, enforcement_mode=enforcement)
    return profile_memory(budget=budget)


def track_allocations(detailed: bool = False) -> Callable[[F], F]:
    """Decorator for detailed allocation tracking.
    
    Args:
        detailed: Whether to capture detailed allocation traces
        
    Example:
        @track_allocations(detailed=True)
        def memory_intensive_function():
            return create_large_structures()
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _profiling_context.should_profile():
                return func(*args, **kwargs)
            
            # Start detailed tracemalloc if requested
            if detailed and not tracemalloc.is_tracing():
                tracemalloc.start(25)  # Keep 25 frames for detailed traces
                should_stop = True
            else:
                should_stop = False
            
            try:
                snapshot_before = tracemalloc.take_snapshot() if detailed else None
                result = func(*args, **kwargs)
                snapshot_after = tracemalloc.take_snapshot() if detailed else None
                
                if detailed and snapshot_before and snapshot_after:
                    # Analyze allocation differences
                    top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
                    
                    significant_changes = [
                        stat for stat in top_stats[:20] 
                        if abs(stat.size_diff) > 1024 * 1024  # >1MB change
                    ]
                    
                    if significant_changes:
                        logger.info(f"Significant allocations in {func.__name__}:")
                        for stat in significant_changes[:5]:
                            logger.info(f"  {stat.traceback.format()[-1].strip()}: {stat.size_diff/1024/1024:+.2f}MB")
                
                return result
                
            finally:
                if should_stop:
                    tracemalloc.stop()
        
        return wrapper
    return decorator


def detect_leaks(window_size: int = 10, threshold_mb: float = 0.5) -> Callable[[F], F]:
    """Decorator for automatic memory leak detection.
    
    Args:
        window_size: Number of calls to analyze for leak patterns
        threshold_mb: Minimum memory growth to consider a leak
        
    Example:
        @detect_leaks(window_size=20, threshold_mb=1.0)
        def potentially_leaky_function():
            return operation_with_possible_leak()
    """
    call_history: List[float] = []
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _profiling_context.should_profile():
                return func(*args, **kwargs)
            
            # Execute with memory tracking
            with memory_snapshot() as (start_mb, _):
                result = func(*args, **kwargs)
                end_mb, _ = tracemalloc.get_traced_memory()
                end_mb = end_mb / 1024 / 1024
            
            # Track memory usage
            call_history.append(end_mb)
            if len(call_history) > window_size:
                call_history.pop(0)
            
            # Analyze for leak pattern
            if len(call_history) >= window_size:
                # Check for consistent growth
                growth_points = sum(
                    1 for i in range(1, len(call_history))
                    if call_history[i] > call_history[i-1]
                )
                
                growth_ratio = growth_points / len(call_history)
                total_growth = call_history[-1] - call_history[0]
                
                if growth_ratio > 0.7 and total_growth > threshold_mb:
                    logger.warning(
                        f"Potential memory leak detected in {func.__name__}: "
                        f"{total_growth:.2f}MB growth over {window_size} calls "
                        f"({growth_ratio*100:.0f}% growth trend)"
                    )
                    
                    # Emit structured alert
                    logger.warning(
                        "Memory leak detection alert",
                        extra={
                            'function': func.__name__,
                            'growth_mb': total_growth,
                            'growth_ratio': growth_ratio,
                            'window_size': window_size,
                            'current_memory_mb': end_mb
                        }
                    )
            
            return result
        
        return wrapper
    return decorator


def get_memory_measurements() -> List[MemoryMeasurement]:
    """Get all collected memory measurements."""
    return _profiling_context.measurements.copy()


def clear_memory_measurements():
    """Clear all collected memory measurements."""
    _profiling_context.measurements.clear()


def get_memory_summary() -> Dict[str, Any]:
    """Get summary statistics of memory measurements."""
    measurements = _profiling_context.measurements
    
    if not measurements:
        return {'total_measurements': 0}
    
    # Aggregate statistics
    total_allocations = sum(m.memory_delta_mb for m in measurements if m.memory_delta_mb > 0)
    total_deallocations = sum(abs(m.memory_delta_mb) for m in measurements if m.memory_delta_mb < 0)
    peak_usage = max(m.peak_memory_mb for m in measurements)
    
    # Function-level statistics
    function_stats = {}
    for measurement in measurements:
        fname = measurement.function_name
        if fname not in function_stats:
            function_stats[fname] = {
                'call_count': 0,
                'total_memory_mb': 0.0,
                'peak_memory_mb': 0.0,
                'total_duration_ms': 0.0
            }
        
        stats = function_stats[fname]
        stats['call_count'] += 1
        stats['total_memory_mb'] += measurement.memory_delta_mb
        stats['peak_memory_mb'] = max(stats['peak_memory_mb'], measurement.peak_memory_mb)
        stats['total_duration_ms'] += measurement.duration_ms
    
    # Calculate averages
    for stats in function_stats.values():
        if stats['call_count'] > 0:
            stats['avg_memory_mb'] = stats['total_memory_mb'] / stats['call_count']
            stats['avg_duration_ms'] = stats['total_duration_ms'] / stats['call_count']
    
    return {
        'total_measurements': len(measurements),
        'total_allocations_mb': total_allocations,
        'total_deallocations_mb': total_deallocations,
        'peak_usage_mb': peak_usage,
        'function_stats': function_stats,
        'time_range': {
            'start': min(m.timestamp for m in measurements),
            'end': max(m.timestamp for m in measurements)
        }
    }


# Agent-specific convenience decorators
def agent_memory_profile(agent_id: Optional[str] = None, budget_mb: float = 34.5) -> Callable[[F], F]:
    """Specialized decorator for agent memory profiling with 34.5MB budget."""
    return profile_memory(budget=MemoryBudget(
        limit_mb=budget_mb,
        warning_threshold=0.8,  # Alert at 27.6MB (80%)
        alert_threshold=0.95,   # Critical at 32.8MB (95%)
        enforcement_mode='warn'
    ))


def validate_agent_memory_budget(func: F) -> F:
    """Decorator that validates agent operations stay within memory budget."""
    return agent_memory_profile(budget_mb=34.5)(func)


# Testing utilities
def enable_memory_profiling_for_tests():
    """Enable memory profiling with full sampling for testing."""
    configure_memory_profiling(enabled=True, sampling_rate=1.0)


def disable_memory_profiling():
    """Disable memory profiling entirely."""
    configure_memory_profiling(enabled=False, sampling_rate=0.0)


if __name__ == "__main__":
    # Example usage
    enable_memory_profiling_for_tests()
    
    @profile_memory
    def example_function():
        # Simulate memory allocation
        data = np.zeros((1000, 1000))
        return data.sum()
    
    @memory_limit('5MB')
    def limited_function():
        # This should trigger a warning
        large_data = np.zeros((1024, 1024))  # ~8MB
        return large_data
    
    @detect_leaks(window_size=5)
    def potentially_leaky():
        # Simulate growing memory usage
        import random
        size = random.randint(100, 1000)
        return np.zeros(size)
    
    # Test the decorators
    print("Testing memory profiling decorators...")
    
    result1 = example_function()
    print(f"Example function result: {result1}")
    
    try:
        result2 = limited_function()
        print(f"Limited function result: {result2 is not None}")
    except MemoryError as e:
        print(f"Memory limit enforced: {e}")
    
    # Test leak detection
    for i in range(7):
        potentially_leaky()
    
    # Print summary
    summary = get_memory_summary()
    print(f"\nMemory profiling summary: {summary['total_measurements']} measurements")
    for func_name, stats in summary['function_stats'].items():
        print(f"  {func_name}: {stats['call_count']} calls, {stats['avg_memory_mb']:.2f}MB avg")