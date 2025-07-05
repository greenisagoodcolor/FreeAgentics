# FreeAgentics Performance Tuning Guide

Based on comprehensive threading vs multiprocessing benchmarks, this guide provides practical recommendations for optimizing FreeAgentics Active Inference agent performance.

## Executive Summary

**Threading is 3-49x faster** than multiprocessing for FreeAgentics agents. Use threading-based coordination with optimized settings for best performance.

## Recommended Configuration

### 1. Agent Configuration

```python
# Optimal agent settings for production
agent_config = {
    'performance_mode': 'fast',           # Use optimized PyMDP settings
    'selective_update_interval': 2,       # Update beliefs every 2 steps
    'enable_observability': False,        # Disable for performance
    'grid_size': 10,                      # Reasonable complexity
    'use_pymdp': True,                    # Keep PyMDP enabled
    'use_llm': False,                     # Disable LLM for speed
}

agent = BasicExplorerAgent(
    agent_id="optimized_agent",
    name="Production Agent",
    **agent_config
)
```

### 2. Thread Pool Configuration

```python
from agents.optimized_threadpool_manager import OptimizedThreadPoolManager

# Optimal thread pool settings
thread_pool = OptimizedThreadPoolManager(
    initial_workers=min(num_agents, mp.cpu_count()),
    max_workers=min(num_agents * 2, mp.cpu_count() * 2),
    min_workers=max(2, mp.cpu_count() // 4),
    scaling_threshold=0.7,
    monitoring_interval=0.5
)

# Register agents
for agent in agents:
    thread_pool.register_agent(agent.agent_id, agent)
```

### 3. Batch Processing

```python
# Process multiple agents efficiently
observations = {
    agent_id: get_observation(agent_id)
    for agent_id in agent_ids
}

# Execute all agents concurrently
results = thread_pool.step_all_agents(observations, timeout=1.0)

# Process results
for agent_id, result in results.items():
    if result.success:
        handle_action(agent_id, result.result)
    else:
        handle_error(agent_id, result.error)
```

## Performance Modes

### Fast Mode (Recommended for Production)

```python
# High-performance settings
config = {
    'performance_mode': 'fast',
    'policy_length': 1,              # Single-step planning
    'param_info_gain': False,        # Disable parameter learning
    'gamma': 8.0,                    # Lower precision for speed
    'alpha': 8.0,                    # Lower action precision
    'selective_update_interval': 2,   # Skip some belief updates
    'matrix_cache': True,            # Enable matrix caching
}
```

**Performance:** ~1.4ms per operation, 680+ ops/sec per agent

### Balanced Mode

```python
# Balanced performance/accuracy
config = {
    'performance_mode': 'balanced',
    'policy_length': 2,              # Two-step planning
    'param_info_gain': True,         # Enable learning
    'gamma': 12.0,                   # Moderate precision
    'alpha': 12.0,                   # Moderate action precision
    'selective_update_interval': 1,   # Update every step
}
```

**Performance:** ~3-5ms per operation, good accuracy

### Accurate Mode

```python
# Maximum accuracy (use sparingly)
config = {
    'performance_mode': 'accurate',
    'policy_length': 3,              # Full planning horizon
    'param_info_gain': True,         # Full parameter learning
    'gamma': 16.0,                   # High precision
    'alpha': 16.0,                   # High action precision
    'selective_update_interval': 1,   # Update every step
}
```

**Performance:** ~10-20ms per operation, highest accuracy

## Scaling Guidelines

### Agent Count Recommendations

| Agents | Thread Workers | Expected Performance | Use Case             |
| ------ | -------------- | -------------------- | -------------------- |
| 1-5    | 2-5            | 500+ ops/sec         | Development, testing |
| 5-20   | 8-20           | 200+ ops/sec         | Production scenarios |
| 20-50  | 16-32          | 100+ ops/sec         | Large simulations    |
| 50+    | 32-64          | 50+ ops/sec          | Massive coordination |

### Performance Expectations

```python
# Target performance metrics
target_metrics = {
    'single_agent_latency_ms': 1.5,
    'multi_agent_throughput_ops_sec': 200,
    'memory_per_agent_mb': 2,
    'scaling_efficiency_percent': 60,
}
```

## Memory Optimization

### 1. Shared Memory Model

```python
# Leverage threading's shared memory
class SharedAgentResources:
    def __init__(self):
        self.observation_model = None  # Shared A matrix
        self.transition_model = None   # Shared B matrix
        self.preferences = None        # Shared C vectors

    def get_cached_model(self, agent_type):
        # Return shared model components
        return self.observation_model, self.transition_model
```

### 2. Memory-Efficient Agents

```python
# Minimize per-agent memory usage
agent_config = {
    'save_belief_hist': False,        # Don't save history
    'debug_mode': False,             # Disable debug info
    'matrix_cache_size': 10,         # Limit cache size
    'gc_interval': 100,              # Periodic garbage collection
}
```

## Communication Patterns

### 1. Efficient Message Passing

```python
# Use shared data structures for coordination
import queue
from threading import Lock

class AgentCoordinator:
    def __init__(self):
        self.shared_state = {}
        self.message_queue = queue.Queue()
        self.state_lock = Lock()

    def update_shared_state(self, agent_id, state):
        with self.state_lock:
            self.shared_state[agent_id] = state

    def broadcast_message(self, sender, message):
        self.message_queue.put((sender, message))
```

### 2. Batch Communication

```python
# Batch message processing for efficiency
def process_messages_batch(coordinator, agents):
    messages = []

    # Collect all messages
    while not coordinator.message_queue.empty():
        messages.append(coordinator.message_queue.get())

    # Process in batch
    for agent in agents:
        relevant_messages = [
            msg for sender, msg in messages
            if msg.get('target') == agent.agent_id
        ]
        agent.process_messages(relevant_messages)
```

## Monitoring and Profiling

### 1. Performance Monitoring

```python
import time
from collections import defaultdict

class PerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)

    def time_operation(self, operation_name):
        def decorator(func):
            def wrapper(*args, **kwargs):
                start = time.time()
                result = func(*args, **kwargs)
                duration = (time.time() - start) * 1000
                self.metrics[operation_name].append(duration)
                return result
            return wrapper
        return decorator

    def get_stats(self):
        stats = {}
        for op, times in self.metrics.items():
            stats[op] = {
                'avg_ms': sum(times) / len(times),
                'count': len(times),
                'total_ms': sum(times)
            }
        return stats
```

### 2. Health Checks

```python
def check_agent_health(thread_pool):
    """Monitor agent and thread pool health."""
    pool_status = thread_pool.get_pool_status()
    perf_stats = thread_pool.get_performance_stats()

    health = {
        'pool_load': pool_status['load_factor'],
        'active_tasks': pool_status['active_tasks'],
        'avg_success_rate': sum(
            stats['successful_tasks'] / max(stats['total_tasks'], 1)
            for stats in perf_stats.values()
        ) / len(perf_stats) if perf_stats else 0
    }

    # Alert if performance degrades
    if health['pool_load'] > 0.9:
        logger.warning("High thread pool load")
    if health['avg_success_rate'] < 0.95:
        logger.warning("Low agent success rate")

    return health
```

## Troubleshooting

### Common Performance Issues

1. **High Latency**
   - Check if observability is enabled (disable for production)
   - Verify performance_mode is set to 'fast'
   - Reduce selective_update_interval

2. **Memory Growth**
   - Disable belief history saving
   - Implement periodic garbage collection
   - Check for memory leaks in custom code

3. **Poor Scaling**
   - Optimize thread pool size
   - Reduce inter-agent communication
   - Consider agent workload distribution

### Performance Debugging

```python
import cProfile
import pstats

def profile_agent_performance(agent, num_steps=100):
    """Profile agent performance to identify bottlenecks."""
    profiler = cProfile.Profile()

    profiler.enable()
    for i in range(num_steps):
        observation = generate_test_observation(i)
        agent.step(observation)
    profiler.disable()

    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

## Production Deployment Checklist

- [ ] Set `performance_mode` to 'fast'
- [ ] Disable observability (`enable_observability: False`)
- [ ] Configure thread pool for your agent count
- [ ] Implement performance monitoring
- [ ] Set up health checks
- [ ] Test under load
- [ ] Monitor memory usage
- [ ] Profile critical paths
- [ ] Implement graceful degradation
- [ ] Set up alerting for performance issues

## Advanced Optimizations

### 1. Custom Performance Modes

```python
# Define custom performance profiles
PERFORMANCE_PROFILES = {
    'realtime': {
        'policy_length': 1,
        'selective_update_interval': 3,
        'gamma': 6.0,
        'alpha': 6.0,
        'target_latency_ms': 5
    },
    'batch': {
        'policy_length': 2,
        'selective_update_interval': 1,
        'gamma': 12.0,
        'alpha': 12.0,
        'target_latency_ms': 20
    }
}
```

### 2. Dynamic Performance Tuning

```python
class AdaptivePerformanceTuner:
    def __init__(self, target_latency_ms=10):
        self.target_latency = target_latency_ms
        self.recent_latencies = []

    def adjust_agent_config(self, agent, current_latency):
        if current_latency > self.target_latency * 1.5:
            # Too slow, reduce accuracy for speed
            agent.selective_update_interval = min(
                agent.selective_update_interval + 1, 5
            )
        elif current_latency < self.target_latency * 0.5:
            # Too fast, can increase accuracy
            agent.selective_update_interval = max(
                agent.selective_update_interval - 1, 1
            )
```

## Conclusion

Following these guidelines will provide optimal performance for FreeAgentics Active Inference agents:

- **3-49x performance improvement** over multiprocessing
- **Efficient resource utilization** through threading
- **Scalable architecture** supporting 20+ concurrent agents
- **Production-ready configuration** with monitoring and health checks

For specific use cases, benchmark your workload and adjust parameters accordingly.
