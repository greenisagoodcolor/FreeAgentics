# NEMESIS Audit: Mocked Performance Tests Removal

## Files Containing Performance Theater (time.sleep/asyncio.sleep)

### Files to Fix

1. **tests/unit/test_gnn_validator.py**

   - Line: `time.sleep(0.01)  # Small delay to increase contention`
   - Action: Remove sleep, test should work without artificial delays

2. **tests/unit/test_knowledge_graph.py**

   - Line: `time.sleep(0.01)`
   - Action: Remove sleep used to ensure timestamp difference

3. **tests/performance/inference_benchmarks.py** (CRITICAL - I created this!)

   - Multiple time.sleep() calls when PyMDP unavailable:
     - `time.sleep(0.005)` in VariationalInferenceBenchmark
     - `time.sleep(0.003)` in BeliefPropagationBenchmark
     - `time.sleep(0.004)` in MessagePassingBenchmark
     - `time.sleep(0.01)` in InferenceProfilingBenchmark
   - Action: Remove ALL sleeps, raise RuntimeError when PyMDP unavailable

4. **tests/performance/pymdp_benchmarks.py** (CRITICAL - I created this!)

   - Multiple time.sleep() fallbacks:
     - `time.sleep(0.001)` in BeliefUpdateBenchmark
     - `time.sleep(0.005)` in ExpectedFreeEnergyBenchmark
     - `time.sleep(0.01)` in MatrixCachingBenchmark
     - `time.sleep(0.001 * self.num_agents)` in AgentScalingBenchmark
   - Action: Remove ALL sleeps, raise RuntimeError when PyMDP unavailable

5. **tests/integration/test_observability_simple.py**

   - `time.sleep(0.001)  # 1ms inference`
   - `time.sleep(0.001)  # 1ms actual work`
   - Action: Remove sleeps, use actual operations or remove test

6. **tests/integration/test_observability_integration.py**
   - Multiple asyncio.sleep() calls:
     - `await asyncio.sleep(0.1)` (multiple instances)
     - `await asyncio.sleep(0.01)  # 10ms inference`
     - `await asyncio.sleep(0.01)  # Small delay for async processing`
   - Action: Remove all async sleeps, test real async operations

### Files Already Disabled

- tests/performance/test_database_load_mock.py.DISABLED_MOCKS
- tests/performance/test_websocket_stress.py.DISABLED_MOCKS
- tests/performance/test_websocket_stress_quick.py.DISABLED_MOCKS

### Files to Keep (Real Performance Tests)

- tests/performance/test_database_load.py (uses real PostgreSQL)
- tests/integration/test_performance_benchmarks.py (uses real PyMDP agents)

## Action Plan

1. Fix all time.sleep() and asyncio.sleep() instances
2. Ensure all benchmark code fails when dependencies unavailable
3. Remove any mock timing or delay simulation
4. Validate no performance theater remains
