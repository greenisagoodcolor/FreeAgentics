"""Comprehensive Memory Profiling Infrastructure for FreeAgentics.

Implements the Nemesis Committee's vision for rigorous memory tracking
to ensure agents stay within the 34.5MB budget and enable early detection
of memory leaks or inefficient allocation patterns.

Architecture:
- MemoryTracker: Core tracking with snapshot comparison
- MemoryDecorator: Automatic profiling for functions/methods
- AgentMemoryValidator: Per-agent memory budget enforcement
- MemoryLeakDetector: Snapshot-based leak detection
- MemoryReporter: Actionable reports with optimization suggestions
"""

import functools
import gc
import json
import os
import sys
import threading
import time
import tracemalloc
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import psutil
import pytest

# Try to import PyMDP for realistic testing
try:
    from pymdp.agent import Agent as PyMDPAgent
    from pymdp import utils as pymdp_utils
    PYMDP_AVAILABLE = True
except ImportError:
    PYMDP_AVAILABLE = False


@dataclass
class MemorySnapshot:
    """Point-in-time memory state."""
    timestamp: float
    current_mb: float
    peak_mb: float
    allocated_blocks: int
    rss_mb: float  # Resident Set Size from psutil
    vms_mb: float  # Virtual Memory Size
    gc_stats: Dict[str, int]
    top_allocations: List[Tuple[str, int]]  # (traceback, size)
    
    
@dataclass
class MemoryReport:
    """Comprehensive memory analysis report."""
    function_name: str
    start_memory_mb: float
    end_memory_mb: float
    peak_memory_mb: float
    memory_delta_mb: float
    execution_time_ms: float
    gc_collections: Dict[str, int]
    memory_timeline: List[float]
    leak_detected: bool
    optimization_suggestions: List[str]
    agent_count: int = 0
    per_agent_memory_mb: float = 0.0
    budget_usage_percent: float = 0.0
    

class MemoryTracker:
    """Advanced memory tracking with leak detection."""
    
    AGENT_MEMORY_BUDGET_MB = 34.5
    ALERT_THRESHOLD_PERCENT = 80
    
    def __init__(self):
        """Initialize memory tracker."""
        self.snapshots: List[MemorySnapshot] = []
        self.baseline_snapshot: Optional[MemorySnapshot] = None
        self.is_tracking = False
        
    def start_tracking(self):
        """Begin memory tracking."""
        if self.is_tracking:
            return
            
        # Force garbage collection for clean baseline
        gc.collect()
        gc.collect()  # Second collection for cyclic references
        
        # Start tracemalloc if not already started
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            
        self.is_tracking = True
        self.baseline_snapshot = self._take_snapshot()
        
    def stop_tracking(self) -> MemorySnapshot:
        """Stop tracking and return final snapshot."""
        if not self.is_tracking:
            raise RuntimeError("Memory tracking not started")
            
        final_snapshot = self._take_snapshot()
        self.is_tracking = False
        return final_snapshot
        
    def _take_snapshot(self) -> MemorySnapshot:
        """Capture current memory state."""
        gc.collect()
        
        # Get tracemalloc stats
        current, peak = tracemalloc.get_traced_memory()
        snapshot = tracemalloc.take_snapshot()
        
        # Get top allocations
        top_stats = snapshot.statistics('traceback')[:10]
        top_allocations = [
            (str(stat.traceback), stat.size)
            for stat in top_stats
        ]
        
        # Get process memory info
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        # Get GC stats
        gc_stats = {
            f"gen{i}": gc.get_count()[i]
            for i in range(gc.get_count().__len__())
        }
        
        return MemorySnapshot(
            timestamp=time.time(),
            current_mb=current / 1024 / 1024,
            peak_mb=peak / 1024 / 1024,
            allocated_blocks=snapshot.statistics('filename').__len__(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            gc_stats=gc_stats,
            top_allocations=top_allocations
        )
        
    def check_memory_leak(self, start: MemorySnapshot, end: MemorySnapshot,
                         threshold_mb: float = 1.0) -> bool:
        """Detect potential memory leaks."""
        # Check for significant memory increase
        memory_increase = end.current_mb - start.current_mb
        
        # Check for increasing allocated blocks
        block_increase = end.allocated_blocks - start.allocated_blocks
        
        # GC should have reduced memory if no leaks
        gc_ineffective = all(
            end.gc_stats.get(f"gen{i}", 0) > start.gc_stats.get(f"gen{i}", 0)
            for i in range(3)
        ) and memory_increase > threshold_mb
        
        return memory_increase > threshold_mb or gc_ineffective
        

class MemoryDecorator:
    """Decorator for automatic memory profiling."""
    
    def __init__(self, tracker: Optional[MemoryTracker] = None):
        """Initialize decorator with optional shared tracker."""
        self.tracker = tracker or MemoryTracker()
        
    def profile_memory(self, sample_interval: float = 0.1):
        """Decorator to profile function memory usage."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Tuple[Any, MemoryReport]:
                # Start tracking
                self.tracker.start_tracking()
                start_time = time.time()
                
                # Sample memory during execution
                memory_timeline = []
                
                # Custom memory sampling
                import threading
                stop_sampling = threading.Event()
                result = None
                exception = None
                
                def sample_memory():
                    """Sample memory usage periodically."""
                    while not stop_sampling.is_set():
                        current, _ = tracemalloc.get_traced_memory()
                        memory_timeline.append(current / 1024 / 1024)
                        time.sleep(sample_interval)
                
                # Start sampling thread
                sampler = threading.Thread(target=sample_memory)
                sampler.daemon = True
                sampler.start()
                
                # Run the function
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    exception = e
                finally:
                    stop_sampling.set()
                    sampler.join(timeout=1.0)
                
                if exception:
                    raise exception
                
                # Stop tracking
                end_snapshot = self.tracker.stop_tracking()
                execution_time = (time.time() - start_time) * 1000
                
                # Generate report
                report = self._generate_report(
                    func.__name__,
                    self.tracker.baseline_snapshot,
                    end_snapshot,
                    execution_time,
                    memory_timeline
                )
                
                return result, report
                
            return wrapper
        return decorator
        
    def _generate_report(self, function_name: str,
                        start: MemorySnapshot,
                        end: MemorySnapshot,
                        execution_time: float,
                        timeline: List[float]) -> MemoryReport:
        """Generate memory profiling report."""
        memory_delta = end.current_mb - start.current_mb
        leak_detected = self.tracker.check_memory_leak(start, end)
        
        # Calculate GC collections during execution
        gc_collections = {
            gen: end.gc_stats.get(gen, 0) - start.gc_stats.get(gen, 0)
            for gen in start.gc_stats
        }
        
        # Generate optimization suggestions
        suggestions = self._generate_suggestions(
            memory_delta,
            end.peak_mb,
            leak_detected,
            gc_collections
        )
        
        return MemoryReport(
            function_name=function_name,
            start_memory_mb=start.current_mb,
            end_memory_mb=end.current_mb,
            peak_memory_mb=end.peak_mb,
            memory_delta_mb=memory_delta,
            execution_time_ms=execution_time,
            gc_collections=gc_collections,
            memory_timeline=timeline,
            leak_detected=leak_detected,
            optimization_suggestions=suggestions
        )
        
    def _generate_suggestions(self, delta_mb: float, peak_mb: float,
                            leak: bool, gc_stats: Dict[str, int]) -> List[str]:
        """Generate actionable optimization suggestions."""
        suggestions = []
        
        if leak:
            suggestions.append(
                "MEMORY LEAK DETECTED: Memory increased significantly and GC "
                "couldn't reclaim it. Check for circular references or "
                "unclosed resources."
            )
            
        if peak_mb > MemoryTracker.AGENT_MEMORY_BUDGET_MB:
            suggestions.append(
                f"BUDGET EXCEEDED: Peak memory {peak_mb:.1f}MB exceeds "
                f"{MemoryTracker.AGENT_MEMORY_BUDGET_MB}MB agent budget. "
                "Consider using sparse matrices or streaming processing."
            )
            
        if delta_mb > 10:
            suggestions.append(
                f"HIGH ALLOCATION: Function allocated {delta_mb:.1f}MB. "
                "Consider object pooling or pre-allocation strategies."
            )
            
        if gc_stats.get('gen2', 0) > 5:
            suggestions.append(
                "EXCESSIVE GC: Many generation 2 collections indicate "
                "long-lived objects. Review object lifecycle management."
            )
            
        return suggestions


class AgentMemoryValidator:
    """Validates per-agent memory usage against budget."""
    
    def __init__(self, budget_mb: float = 34.5):
        """Initialize validator with budget."""
        self.budget_mb = budget_mb
        self.alert_threshold_mb = budget_mb * 0.8  # 80% alert
        self.agent_memory: Dict[str, float] = {}
        
    def track_agent(self, agent_id: str, memory_mb: float):
        """Track memory for specific agent."""
        self.agent_memory[agent_id] = memory_mb
        
    def validate_agent(self, agent_id: str) -> Tuple[bool, str]:
        """Validate agent memory against budget."""
        if agent_id not in self.agent_memory:
            return False, f"Agent {agent_id} not tracked"
            
        memory_mb = self.agent_memory[agent_id]
        usage_percent = (memory_mb / self.budget_mb) * 100
        
        if memory_mb > self.budget_mb:
            return False, (
                f"CRITICAL: Agent {agent_id} uses {memory_mb:.1f}MB "
                f"({usage_percent:.0f}% of {self.budget_mb}MB budget)"
            )
        elif memory_mb > self.alert_threshold_mb:
            return True, (
                f"WARNING: Agent {agent_id} uses {memory_mb:.1f}MB "
                f"({usage_percent:.0f}% of budget) - approaching limit"
            )
        else:
            return True, (
                f"OK: Agent {agent_id} uses {memory_mb:.1f}MB "
                f"({usage_percent:.0f}% of budget)"
            )
            
    def validate_all(self) -> List[Tuple[str, bool, str]]:
        """Validate all tracked agents."""
        results = []
        for agent_id in self.agent_memory:
            valid, message = self.validate_agent(agent_id)
            results.append((agent_id, valid, message))
        return results


# Test the memory profiling infrastructure
class TestMemoryProfiling:
    """Test memory profiling capabilities."""
    
    def test_memory_tracker_basic(self):
        """Test basic memory tracking."""
        tracker = MemoryTracker()
        
        # Start tracking
        tracker.start_tracking()
        
        # Allocate some memory
        data = np.random.rand(1000, 1000)  # ~7.6MB
        
        # Stop tracking
        end_snapshot = tracker.stop_tracking()
        
        # Verify tracking worked
        assert end_snapshot.current_mb > tracker.baseline_snapshot.current_mb
        assert end_snapshot.peak_mb >= end_snapshot.current_mb
        assert len(end_snapshot.top_allocations) > 0
        
    def test_memory_decorator(self):
        """Test memory profiling decorator."""
        decorator = MemoryDecorator()
        
        @decorator.profile_memory(sample_interval=0.01)
        def memory_intensive_function():
            # Allocate increasing amounts of memory
            arrays = []
            for i in range(5):
                arrays.append(np.random.rand(500, 500))  # ~1.9MB each
                time.sleep(0.01)  # Allow sampling
            return sum(a.sum() for a in arrays)
            
        # Run profiled function
        result, report = memory_intensive_function()
        
        # Verify report
        assert report.function_name == "memory_intensive_function"
        assert report.memory_delta_mb > 0  # Some memory allocated
        assert len(report.memory_timeline) >= 1  # At least one sample
        assert report.execution_time_ms > 40  # At least 40ms
        
    def test_agent_memory_validation(self):
        """Test per-agent memory validation."""
        validator = AgentMemoryValidator(budget_mb=34.5)
        
        # Track different agents
        validator.track_agent("agent_efficient", 15.2)
        validator.track_agent("agent_warning", 28.5)
        validator.track_agent("agent_exceeded", 45.0)
        
        # Validate agents
        valid1, msg1 = validator.validate_agent("agent_efficient")
        assert valid1 and "OK" in msg1
        
        valid2, msg2 = validator.validate_agent("agent_warning")
        assert valid2 and "WARNING" in msg2
        
        valid3, msg3 = validator.validate_agent("agent_exceeded")
        assert not valid3 and "CRITICAL" in msg3
        
    def test_memory_leak_detection(self):
        """Test memory leak detection."""
        tracker = MemoryTracker()
        
        # Simulate a memory leak
        leaked_data = []
        
        def leaky_function():
            tracker.start_tracking()
            start_snapshot = tracker.baseline_snapshot
            
            # Keep appending without cleanup
            for i in range(100):
                leaked_data.append(np.random.rand(100, 100))
                
            end_snapshot = tracker.stop_tracking()
            return tracker.check_memory_leak(start_snapshot, end_snapshot)
            
        leak_detected = leaky_function()
        assert leak_detected  # Should detect the leak
        
        # Cleanup
        leaked_data.clear()
        
    @pytest.mark.skip(reason="PyMDP integration test - runs separately")
    def test_pymdp_agent_memory(self):
        """Test PyMDP agent memory usage."""
        decorator = MemoryDecorator()
        validator = AgentMemoryValidator()
        
        @decorator.profile_memory()
        def create_pymdp_agent():
            # Create a realistic PyMDP agent
            num_states = [10, 5, 3]  # Multi-factor state space
            num_observations = num_states
            num_actions = 4
            
            # Create generative model
            A = pymdp_utils.random_A_matrix(num_observations, num_states)
            B = pymdp_utils.random_B_matrix(num_states, [num_actions] * len(num_states))
            C = pymdp_utils.obj_array_zeros(num_observations)
            D = pymdp_utils.obj_array_uniform(num_states)
            
            # Create agent
            agent = PyMDPAgent(
                A=A, B=B, C=C, D=D,
                inference_algo="VANILLA",
                inference_horizon=3,
                policy_len=2,
                use_utility=True,
                use_states_info_gain=True,
                use_param_info_gain=False
            )
            
            # Run some inference steps
            for _ in range(10):
                obs = [np.random.randint(0, n) for n in num_observations]
                agent.infer_states(obs)
                agent.infer_policies()
                agent.sample_action()
                
            return agent
            
        # Profile agent creation
        agent, report = create_pymdp_agent()
        
        # Validate memory usage
        validator.track_agent("pymdp_test_agent", report.peak_memory_mb)
        valid, message = validator.validate_agent("pymdp_test_agent")
        
        print(f"\nPyMDP Agent Memory Report:")
        print(f"  Peak memory: {report.peak_memory_mb:.1f}MB")
        print(f"  Validation: {message}")
        print(f"  Suggestions: {report.optimization_suggestions}")
        
        # Assert agent fits in budget
        assert valid, f"PyMDP agent exceeds memory budget: {message}"


if __name__ == "__main__":
    # Run basic demonstration
    print("Memory Profiling Infrastructure Demonstration\n")
    
    # Create components
    tracker = MemoryTracker()
    decorator = MemoryDecorator(tracker)
    validator = AgentMemoryValidator()
    
    # Profile a sample function
    @decorator.profile_memory()
    def demo_function():
        """Demonstration function with memory operations."""
        data = []
        for i in range(5):
            # Allocate 5MB chunks
            chunk = np.random.rand(650, 1000)  # ~5MB
            data.append(chunk)
            time.sleep(0.05)  # Simulate processing
            
        # Simulate memory leak
        global leaked_reference
        leaked_reference = data[0]  # Keep reference
        
        return sum(d.sum() for d in data)
        
    # Run profiled function
    result, report = demo_function()
    
    # Display results
    print(f"Function: {report.function_name}")
    print(f"Memory delta: {report.memory_delta_mb:.1f}MB")
    print(f"Peak memory: {report.peak_memory_mb:.1f}MB")
    print(f"Execution time: {report.execution_time_ms:.0f}ms")
    print(f"Leak detected: {report.leak_detected}")
    print(f"\nSuggestions:")
    for suggestion in report.optimization_suggestions:
        print(f"  - {suggestion}")