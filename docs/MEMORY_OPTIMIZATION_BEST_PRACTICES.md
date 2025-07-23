# Memory Optimization Best Practices

## Overview

This document outlines best practices for memory optimization in the FreeAgentics multi-agent system. The goal is to reduce per-agent memory usage from 34.5MB to under 10MB while maintaining or improving performance efficiency above 50% for 50+ agents.

## Key Memory Optimization Strategies

### 1. Agent Memory Structure Optimization

#### Lazy Loading
- **Belief States**: Use `LazyBeliefArray` to load belief states only when needed
- **Observations**: Store in shared memory-mapped buffers
- **Computations**: Defer expensive calculations until required

```python
from agents.memory_optimization import LazyBeliefArray

# Instead of:
agent.beliefs = np.zeros((1000, 1000))  # 8MB always loaded

# Use:
agent.beliefs = LazyBeliefArray((1000, 1000), sparsity_threshold=0.9)
```

#### Memory Sharing
- **Parameters**: Share common parameters (transition matrices, priors) across agents
- **Computation Buffers**: Use pooled temporary buffers for calculations
- **Observations**: Use shared memory-mapped observation buffer

```python
from agents.memory_optimization import get_agent_optimizer

optimizer = get_agent_optimizer()
optimized_agent = optimizer.optimize_agent(agent)
```

### 2. Garbage Collection Tuning

#### Adaptive GC Thresholds
The system automatically adjusts GC thresholds based on:
- Memory pressure (current usage vs limit)
- Agent count
- GC overhead percentage

```python
from agents.memory_optimization import optimize_gc_for_agents

# Optimize GC for specific agent configuration
optimize_gc_for_agents(agent_count=50, memory_limit_mb=1024)
```

#### GC Context Management
Use GC context managers for batch operations:

```python
from agents.memory_optimization import get_gc_tuner

gc_tuner = get_gc_tuner()
gc_context = gc_tuner.GCContextManager(gc_tuner)

# For batch operations
with gc_context.batch_operation(disable_gc=True):
    # Process many agents without GC interruption
    results = process_all_agents(agents)

# For low-latency operations
with gc_context.low_latency():
    # Defer GC during time-sensitive operations
    result = time_sensitive_operation()
```

### 3. Data Structure Optimization

#### Sparse Representations
Use sparse arrays for belief states and observation matrices:

```python
from agents.memory_optimization import SparseBeliefState

# Compress belief states with >90% zeros
compressed_beliefs = compressor.compress(dense_beliefs, threshold=0.9)
```

#### Compressed History
Use compressed circular buffers for action/observation history:

```python
from agents.memory_optimization import CompressedHistory

# Instead of:
agent.action_history = []  # Grows unbounded

# Use:
agent.action_history = CompressedHistory(max_size=1000, compression_level=6)
```

### 4. Memory Profiling and Monitoring

#### Continuous Monitoring
Enable memory profiling to track usage patterns:

```python
from agents.memory_optimization import get_memory_profiler

profiler = get_memory_profiler()
profiler.start_monitoring()

# Register agents for tracking
profiler.register_agent(agent_id, agent_object)
```

#### Memory Reports
Generate regular memory reports to identify issues:

```python
# Get comprehensive memory report
report = profiler.get_memory_report()

# Get per-agent optimization suggestions
suggestions = profiler.optimize_agent_memory(agent_id)
```

### 5. ThreadPool Manager Integration

#### Memory-Optimized Agent Registration
Use the optimized threadpool manager with memory optimization enabled:

```python
from agents.optimized_threadpool_manager import OptimizedThreadPoolManager

manager = OptimizedThreadPoolManager(
    initial_workers=8,
    max_workers=32,
    enable_memory_optimization=True,
    target_memory_per_agent_mb=10.0,
)

# Register agents with automatic optimization
manager.register_agent(agent_id, agent)
```

#### Memory Usage Monitoring
Monitor memory usage during execution:

```python
# Get memory report
memory_report = manager.get_memory_report()

# Optimize memory usage
optimization_results = manager.optimize_memory_usage()
```

## Performance Targets

### Memory Usage Targets
- **Per-Agent Memory**: < 10MB (down from 34.5MB)
- **Total System Memory**: < 512MB for 50 agents
- **Memory Efficiency**: > 50% (up from 28.4%)

### Performance Targets
- **GC Overhead**: < 5% of execution time
- **Agent Registration**: < 100ms per agent
- **Task Execution**: < 50ms per agent per step

## Implementation Checklist

### For New Agents
- [ ] Use `LazyBeliefArray` for belief states
- [ ] Implement `CompressedHistory` for action history
- [ ] Register with memory lifecycle manager
- [ ] Use shared parameters where possible
- [ ] Implement proper cleanup methods

### For Existing Agents
- [ ] Profile current memory usage
- [ ] Apply agent memory optimizer
- [ ] Update data structures to use efficient variants
- [ ] Enable memory monitoring
- [ ] Validate memory reduction

### For System Integration
- [ ] Enable memory optimization in threadpool manager
- [ ] Configure appropriate GC thresholds
- [ ] Set up memory monitoring dashboard
- [ ] Implement memory alerts
- [ ] Create memory optimization pipeline

## Common Pitfalls

### Memory Leaks
- **Circular References**: Use weak references for agent callbacks
- **Unclosed Resources**: Always close memory-mapped files
- **Growing Caches**: Implement LRU or size-limited caches

### Performance Degradation
- **Excessive GC**: Monitor GC overhead and adjust thresholds
- **Memory Fragmentation**: Use memory pools for frequent allocations
- **Lazy Loading Overhead**: Balance lazy loading with access patterns

### Debugging Memory Issues
- **Memory Profiling**: Use `tracemalloc` for allocation tracking
- **Leak Detection**: Monitor allocation patterns over time
- **Agent Analysis**: Use per-agent memory optimization reports

## Code Examples

### Basic Agent Optimization
```python
from agents.memory_optimization import get_agent_optimizer

# Optimize existing agent
optimizer = get_agent_optimizer()
optimized_agent = optimizer.optimize_agent(original_agent)

# Check memory usage
memory_usage = optimized_agent.get_memory_usage_mb()
print(f"Agent memory: {memory_usage:.1f}MB")
```

### Memory-Efficient Agent Creation
```python
from agents.memory_optimization import (
    LazyBeliefArray,
    CompressedHistory,
    get_agent_optimizer,
)

class OptimizedAgent:
    def __init__(self, agent_id: str):
        self.id = agent_id
        self.position = np.array([0.0, 0.0], dtype=np.float32)

        # Use lazy loading for beliefs
        self.beliefs = LazyBeliefArray((100, 100), sparsity_threshold=0.9)

        # Use compressed history
        self.action_history = CompressedHistory(max_size=1000)

        # Use shared parameters
        optimizer = get_agent_optimizer()
        self.shared_params = optimizer.shared_params
```

### Memory Monitoring Setup
```python
from agents.memory_optimization import get_memory_profiler

profiler = get_memory_profiler()
profiler.start_monitoring()

# Profile specific operation
with profiler.profile_operation("agent_step"):
    result = agent.step(observation)

# Get memory report
report = profiler.get_memory_report()
```

## Validation and Testing

### Unit Tests
- Test memory optimization components individually
- Verify memory reduction targets
- Test GC tuning effectiveness

### Integration Tests
- Test memory optimization with full agent system
- Validate performance with 50+ agents
- Test memory leak detection

### Performance Benchmarks
- Compare memory usage before/after optimization
- Measure GC overhead reduction
- Validate efficiency improvements

## Monitoring and Alerts

### Memory Metrics
- **Per-Agent Memory**: Track individual agent memory usage
- **Total System Memory**: Monitor aggregate memory consumption
- **Memory Growth Rate**: Detect memory leaks early
- **GC Statistics**: Monitor garbage collection performance

### Alert Conditions
- **High Memory Usage**: > 80% of target memory limit
- **Memory Leaks**: Sustained memory growth over time
- **GC Overhead**: > 10% of execution time
- **Agent Memory**: Individual agent > 15MB

## Conclusion

By following these best practices, the FreeAgentics system can achieve:
- **70% reduction** in per-agent memory usage (34.5MB → 10MB)
- **50%+ efficiency** with 50+ agents (vs 28.4% baseline)
- **Stable performance** with automatic memory management
- **Proactive monitoring** to prevent memory issues

The memory optimization system is designed to be:
- **Automatic**: Minimal configuration required
- **Adaptive**: Adjusts to changing workloads
- **Transparent**: No changes to agent logic required
- **Monitorable**: Comprehensive metrics and reporting
