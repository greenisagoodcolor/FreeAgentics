# Multi-Agent Coordination Load Tests

This directory contains specialized load tests that validate the documented architectural limitations of the FreeAgentics multi-agent system.

## Overview

The load tests are designed to demonstrate real performance characteristics, not idealized scenarios. They specifically validate:

- **Python GIL constraints**: ~50 agent practical limit
- **Efficiency at scale**: 28.4% efficiency with 50 agents
- **Coordination overhead**: ~72% efficiency loss due to coordination
- **Real-world scenarios**: Task handoffs, resource contention, consensus building

## Test Components

### 1. `test_coordination_load.py`

Core load testing framework that measures:

- Agent spawning and lifecycle management
- Message queue performance
- Coordination latency
- Failure resilience
- Actual vs theoretical efficiency

### 2. `agent_simulation_framework.py`

Provides realistic agent simulation capabilities:

- Agent spawning with configurable parameters
- Lifecycle management (spawn, monitor, terminate)
- Multiple agent types (Explorer, Collector, Coordinator)
- Simulation environment with tick-based execution

### 3. `run_coordination_load_tests.py`

Main test orchestrator that:

- Runs comprehensive test suite
- Generates performance reports
- Creates visualization plots
- Validates against documented limitations

## Running the Tests

### Quick Test (Reduced Parameters)

```bash
python tests/performance/run_coordination_load_tests.py --quick
```

### Full Test Suite

```bash
python tests/performance/run_coordination_load_tests.py \
    --max-agents 50 \
    --duration 60 \
    --generate-plots \
    --output-dir ./load_test_results
```

### Custom Configuration

```bash
# Test with 30 agents for 30 seconds
python tests/performance/run_coordination_load_tests.py \
    --max-agents 30 \
    --duration 30

# Just run the coordination overhead analysis
python tests/performance/test_coordination_load.py
```

## Test Scenarios

### 1. Coordination Overhead Analysis

Measures the actual overhead introduced by multi-agent coordination:

- Sequential baseline performance
- Concurrent agent execution
- Message passing overhead
- Synchronization costs

### 2. Task Handoff Simulation

Tests coordination when agents hand off tasks:

- Handoff request/acknowledgment protocol
- Success rate under load
- Latency measurements

### 3. Resource Contention

Simulates multiple agents competing for shared resources:

- Lock contention
- Negotiation through messaging
- Resolution strategies

### 4. Consensus Building

Tests distributed consensus mechanisms:

- Voting protocols
- Message propagation
- Time to consensus

### 5. Failure Resilience

Validates system behavior under failures:

- Agent crashes
- Recovery mechanisms
- System stability

## Expected Results

Based on architectural analysis, the tests should show:

### Efficiency Degradation Curve

```
Agents | Efficiency | Status
-------|------------|--------
1      | 100%       | ✅ Baseline
5      | ~85%       | ✅ Good
10     | ~70%       | ✅ Acceptable
20     | ~50%       | ✅ Degrading
30     | ~40%       | ⚠️ Limited
40     | ~35%       | ⚠️ Poor
50     | ~28.4%     | ❌ Practical limit
```

### Coordination Metrics at 50 Agents

- **Efficiency**: 28.4% (72% loss)
- **Message latency**: <100ms
- **Handoff success rate**: >80%
- **Consensus time**: <500ms
- **Recovery time**: <200ms

## Interpreting Results

### Green Flags ✅

- Efficiency loss matches documented ~72%
- System handles 50 agents
- Coordination latency < 100ms
- Graceful degradation

### Red Flags ❌

- Efficiency loss > 80%
- Cannot handle 50 agents
- High message queue drops
- Cascading failures

## Output Files

The test suite generates:

1. **JSON Results**: Detailed metrics and measurements
2. **Performance Plots**:
   - Efficiency degradation curve
   - Message queue performance
   - Scaling characteristics
3. **Text Report**: Summary with key findings

## Performance Tuning

If tests show poor performance:

1. **Check agent configuration**:

   ```python
   agent.config["performance_mode"] = "fast"
   agent.config["selective_update_interval"] = 2
   ```

2. **Adjust message queue size**:

   ```python
   message_queue = MessageQueue(max_size=10000)
   ```

3. **Tune thread pool size**:

   ```python
   executor = ThreadPoolExecutor(max_workers=32)
   ```

## Integration with CI/CD

Add to your CI pipeline:

```yaml
- name: Run Load Tests
  run: |
    python tests/performance/run_coordination_load_tests.py \
      --quick \
      --max-agents 20
```

## Troubleshooting

### High Memory Usage

- Reduce agent count
- Enable performance mode
- Check for memory leaks in agent implementations

### Poor Scaling

- Verify Python GIL impact
- Check for synchronization bottlenecks
- Review message queue performance

### Test Failures

- Ensure all dependencies installed
- Check system resources
- Review agent implementations

## Future Enhancements

Potential improvements to the load testing framework:

1. **Distributed testing**: Multi-process agent pools
2. **Network simulation**: Add realistic network delays
3. **Workload profiles**: Industry-specific scenarios
4. **Long-running tests**: Stability over hours/days
5. **Performance regression**: Automated detection
