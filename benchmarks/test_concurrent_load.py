#!/usr/bin/env python3
"""
Concurrent Load Performance Benchmarks
PERF-ENGINEER: Bryan Cantrill + Brendan Gregg Methodology
"""

import pytest
import time
import asyncio
import threading
import multiprocessing as mp
import psutil
import statistics
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable
import numpy as np
from dataclasses import dataclass
import queue


@dataclass
class LoadTestResult:
    """Concurrent load test result."""
    scenario: str
    concurrency_level: int
    total_operations: int
    duration_seconds: float
    throughput_ops_sec: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    errors: int
    cpu_usage_percent: float
    memory_usage_mb: float


class ConcurrentLoadBenchmarks:
    """Comprehensive concurrent load benchmarks."""
    
    def __init__(self):
        self.cpu_count = psutil.cpu_count(logical=False)
        self.results_queue = queue.Queue()
    
    def simulate_agent_work(self, agent_id: int, iterations: int = 100) -> Dict[str, Any]:
        """Simulate agent computational work."""
        start_time = time.perf_counter()
        
        # Simulate belief updates (matrix operations)
        beliefs = np.random.rand(10, 10)
        observations = np.random.rand(10)
        
        for _ in range(iterations):
            # Simulate inference
            beliefs = beliefs @ beliefs.T
            beliefs = beliefs / np.sum(beliefs)
            
            # Simulate observation update  
            beliefs += np.outer(observations, observations)
            beliefs = beliefs / np.sum(beliefs)
        
        duration = time.perf_counter() - start_time
        
        return {
            'agent_id': agent_id,
            'duration': duration,
            'iterations': iterations,
            'final_belief_sum': float(np.sum(beliefs))
        }
    
    def message_passing_work(self, sender_id: int, receiver_id: int, message_count: int = 10):
        """Simulate inter-agent message passing."""
        messages = []
        
        for i in range(message_count):
            msg = {
                'id': f"{sender_id}-{receiver_id}-{i}",
                'sender': sender_id,
                'receiver': receiver_id,
                'content': f"Message {i}",
                'timestamp': time.time()
            }
            
            # Simulate message processing
            time.sleep(0.001)  # 1ms processing time
            messages.append(msg)
        
        return messages
    
    @pytest.mark.benchmark(group="concurrent-load")
    def test_thread_pool_scaling(self, benchmark):
        """Test thread pool scaling with different worker counts."""
        
        def run_with_workers(num_workers: int):
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                start_time = time.perf_counter()
                
                # Submit tasks
                futures = []
                for i in range(100):
                    future = executor.submit(self.simulate_agent_work, i, 50)
                    futures.append(future)
                
                # Collect results
                results = []
                for future in as_completed(futures):
                    results.append(future.result())
                
                duration = time.perf_counter() - start_time
                throughput = len(results) / duration
                
                return {
                    'num_workers': num_workers,
                    'duration': duration,
                    'throughput': throughput,
                    'results': results
                }
        
        # Test different worker counts
        worker_counts = [1, 2, 4, 8, self.cpu_count, self.cpu_count * 2]
        scaling_results = []
        
        for workers in worker_counts:
            result = run_with_workers(workers)
            scaling_results.append(result)
            
            print(f"\nWorkers: {workers}")
            print(f"  Duration: {result['duration']:.2f}s")
            print(f"  Throughput: {result['throughput']:.1f} ops/s")
        
        # Analyze scaling efficiency
        baseline_throughput = scaling_results[0]['throughput']
        
        print("\nScaling Efficiency:")
        for result in scaling_results:
            efficiency = (result['throughput'] / baseline_throughput) / result['num_workers'] * 100
            print(f"  {result['num_workers']} workers: {efficiency:.1f}% efficiency")
    
    @pytest.mark.benchmark(group="concurrent-load")
    def test_async_concurrency(self, benchmark):
        """Test async/await concurrency patterns."""
        
        async def async_agent_work(agent_id: int) -> Dict[str, Any]:
            """Async version of agent work."""
            start_time = time.perf_counter()
            
            # Simulate async I/O operations
            await asyncio.sleep(0.01)  # 10ms simulated I/O
            
            # Do some computation
            result = self.simulate_agent_work(agent_id, 10)
            
            result['async_duration'] = time.perf_counter() - start_time
            return result
        
        async def run_async_load(concurrency: int):
            """Run async load test."""
            start_time = time.perf_counter()
            
            # Create tasks
            tasks = []
            for i in range(100):
                task = asyncio.create_task(async_agent_work(i))
                tasks.append(task)
                
                # Control concurrency
                if len(tasks) >= concurrency:
                    done, tasks = await asyncio.wait(
                        tasks, 
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    tasks = list(tasks)
            
            # Wait for remaining tasks
            if tasks:
                await asyncio.gather(*tasks)
            
            duration = time.perf_counter() - start_time
            return {
                'concurrency': concurrency,
                'duration': duration,
                'throughput': 100 / duration
            }
        
        # Test different concurrency levels
        async def test_all():
            concurrency_levels = [1, 10, 50, 100]
            results = []
            
            for level in concurrency_levels:
                result = await run_async_load(level)
                results.append(result)
                print(f"\nAsync Concurrency: {level}")
                print(f"  Duration: {result['duration']:.2f}s")
                print(f"  Throughput: {result['throughput']:.1f} ops/s")
            
            return results
        
        # Run async benchmark
        results = asyncio.run(test_all())
    
    @pytest.mark.benchmark(group="concurrent-load")  
    def test_load_patterns(self, benchmark):
        """Test different load patterns."""
        
        class LoadPattern:
            @staticmethod
            def constant_load(duration: float, ops_per_sec: float) -> List[float]:
                """Generate constant load pattern."""
                operations = []
                interval = 1.0 / ops_per_sec
                current_time = 0
                
                while current_time < duration:
                    operations.append(current_time)
                    current_time += interval
                
                return operations
            
            @staticmethod
            def burst_load(duration: float, burst_size: int, burst_interval: float) -> List[float]:
                """Generate burst load pattern."""
                operations = []
                current_time = 0
                
                while current_time < duration:
                    # Add burst
                    for _ in range(burst_size):
                        operations.append(current_time)
                    current_time += burst_interval
                
                return operations
            
            @staticmethod
            def ramp_load(duration: float, start_ops: float, end_ops: float) -> List[float]:
                """Generate ramping load pattern."""
                operations = []
                current_time = 0
                time_step = 0.1
                
                while current_time < duration:
                    # Calculate current rate
                    progress = current_time / duration
                    current_rate = start_ops + (end_ops - start_ops) * progress
                    interval = 1.0 / current_rate
                    
                    operations.append(current_time)
                    current_time += interval
                
                return operations
        
        def execute_load_pattern(pattern_name: str, operations: List[float]):
            """Execute operations according to pattern."""
            start_time = time.perf_counter()
            latencies = []
            errors = 0
            
            with ThreadPoolExecutor(max_workers=self.cpu_count) as executor:
                futures = []
                
                for op_time in operations:
                    # Wait until operation time
                    wait_time = op_time - (time.perf_counter() - start_time)
                    if wait_time > 0:
                        time.sleep(wait_time)
                    
                    # Submit operation
                    op_start = time.perf_counter()
                    future = executor.submit(self.simulate_agent_work, len(futures), 10)
                    futures.append((future, op_start))
                
                # Collect results
                for future, op_start in futures:
                    try:
                        result = future.result(timeout=5.0)
                        latency = (time.perf_counter() - op_start) * 1000
                        latencies.append(latency)
                    except Exception:
                        errors += 1
            
            # Calculate statistics
            if latencies:
                return {
                    'pattern': pattern_name,
                    'total_ops': len(operations),
                    'successful_ops': len(latencies),
                    'errors': errors,
                    'latency_p50': np.percentile(latencies, 50),
                    'latency_p95': np.percentile(latencies, 95),
                    'latency_p99': np.percentile(latencies, 99)
                }
            else:
                return None
        
        # Test different patterns
        patterns = [
            ('Constant Load', LoadPattern.constant_load(5.0, 20)),
            ('Burst Load', LoadPattern.burst_load(5.0, 50, 1.0)),
            ('Ramp Load', LoadPattern.ramp_load(5.0, 5, 50))
        ]
        
        print("\nLoad Pattern Testing:")
        for pattern_name, operations in patterns:
            result = execute_load_pattern(pattern_name, operations)
            if result:
                print(f"\n{pattern_name}:")
                print(f"  Operations: {result['total_ops']}")
                print(f"  Successful: {result['successful_ops']}")
                print(f"  Errors: {result['errors']}")
                print(f"  Latency P50: {result['latency_p50']:.1f}ms")
                print(f"  Latency P95: {result['latency_p95']:.1f}ms")
                print(f"  Latency P99: {result['latency_p99']:.1f}ms")
    
    @pytest.mark.benchmark(group="concurrent-load")
    def test_contention_scenarios(self, benchmark):
        """Test resource contention scenarios."""
        
        # Shared resource with lock
        shared_counter = 0
        counter_lock = threading.Lock()
        
        # Shared resource with atomic operations
        atomic_counter = mp.Value('i', 0)
        
        def contended_increment(iterations: int):
            """Increment with lock contention."""
            nonlocal shared_counter
            
            for _ in range(iterations):
                with counter_lock:
                    shared_counter += 1
        
        def atomic_increment(iterations: int):
            """Increment with atomic operations."""
            for _ in range(iterations):
                with atomic_counter.get_lock():
                    atomic_counter.value += 1
        
        def lock_free_work(iterations: int):
            """Work without shared state."""
            local_counter = 0
            for _ in range(iterations):
                local_counter += 1
            return local_counter
        
        # Test scenarios
        scenarios = [
            ("Lock Contention", contended_increment),
            ("Atomic Operations", atomic_increment),
            ("Lock-Free", lock_free_work)
        ]
        
        worker_counts = [1, 2, 4, 8]
        iterations_per_worker = 10000
        
        print("\nContention Scenario Testing:")
        
        for scenario_name, work_func in scenarios:
            print(f"\n{scenario_name}:")
            
            for workers in worker_counts:
                shared_counter = 0
                atomic_counter.value = 0
                
                start_time = time.perf_counter()
                
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [
                        executor.submit(work_func, iterations_per_worker)
                        for _ in range(workers)
                    ]
                    
                    for future in futures:
                        future.result()
                
                duration = time.perf_counter() - start_time
                throughput = (workers * iterations_per_worker) / duration
                
                print(f"  {workers} workers: {throughput:.0f} ops/s ({duration:.3f}s)")
    
    @pytest.mark.benchmark(group="concurrent-load")
    def test_queue_performance(self, benchmark):
        """Test different queue implementations under load."""
        
        def producer_consumer_test(queue_impl, num_producers: int, num_consumers: int, items_per_producer: int):
            """Test producer-consumer pattern."""
            start_time = time.perf_counter()
            processed_items = []
            
            def producer(producer_id: int):
                for i in range(items_per_producer):
                    item = {
                        'producer': producer_id,
                        'item': i,
                        'timestamp': time.time()
                    }
                    queue_impl.put(item)
            
            def consumer(consumer_id: int):
                items = []
                while True:
                    try:
                        item = queue_impl.get(timeout=0.1)
                        items.append(item)
                        queue_impl.task_done()
                    except:
                        break
                return items
            
            # Start producers and consumers
            with ThreadPoolExecutor(max_workers=num_producers + num_consumers) as executor:
                # Start producers
                producer_futures = [
                    executor.submit(producer, i)
                    for i in range(num_producers)
                ]
                
                # Start consumers  
                consumer_futures = [
                    executor.submit(consumer, i)
                    for i in range(num_consumers)
                ]
                
                # Wait for producers
                for future in producer_futures:
                    future.result()
                
                # Signal completion
                for _ in range(num_consumers):
                    queue_impl.put(None)
                
                # Collect consumer results
                for future in consumer_futures:
                    items = future.result()
                    processed_items.extend(items)
            
            duration = time.perf_counter() - start_time
            
            return {
                'duration': duration,
                'total_items': len(processed_items),
                'throughput': len(processed_items) / duration
            }
        
        # Test different queue implementations
        import queue
        from collections import deque
        
        queue_types = [
            ("Queue (FIFO)", queue.Queue()),
            ("LifoQueue (LIFO)", queue.LifoQueue()),
            ("PriorityQueue", queue.PriorityQueue())
        ]
        
        print("\nQueue Performance Under Load:")
        
        for queue_name, queue_impl in queue_types:
            result = producer_consumer_test(
                queue_impl,
                num_producers=4,
                num_consumers=4,
                items_per_producer=1000
            )
            
            print(f"\n{queue_name}:")
            print(f"  Duration: {result['duration']:.2f}s")
            print(f"  Items processed: {result['total_items']}")
            print(f"  Throughput: {result['throughput']:.0f} items/s")


def run_concurrent_benchmarks():
    """Run all concurrent load benchmarks."""
    pytest.main([
        __file__,
        "-v",
        "--benchmark-only",
        "--benchmark-columns=min,max,mean,stddev,median",
        "--benchmark-sort=mean",
        "--benchmark-group-by=group",
        "-s"  # Don't capture output
    ])


if __name__ == "__main__":
    print("="*60)
    print("PERF-ENGINEER: Concurrent Load Performance Benchmarks")
    print("Bryan Cantrill + Brendan Gregg Methodology")
    print("="*60)
    
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    run_concurrent_benchmarks()