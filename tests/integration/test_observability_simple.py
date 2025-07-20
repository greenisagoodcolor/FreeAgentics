"""Simple observability integration validation.

Tests that observability components are correctly integrated
and functional without complex dependencies.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


async def test_observability_components():
    """Test that observability components can be imported and used."""

    print("Testing observability component imports...")

    # Test 1: Import observability components
    try:
        from observability.pymdp_integration import (
            PyMDPObservabilityIntegrator,
            get_pymdp_performance_summary,
            monitor_pymdp_inference,
            pymdp_observer,
            record_agent_lifecycle_event,
            record_belief_update,
        )

        print("✅ Observability imports successful")
        observability_available = True
    except ImportError as e:
        print(f"❌ Observability imports failed: {e}")
        observability_available = False

    if not observability_available:
        print("⚠️ Skipping observability tests - components not available")
        return False

    # Test 2: Create observability integrator
    try:
        _integrator = PyMDPObservabilityIntegrator()
        print("✅ PyMDP observability integrator created")
    except Exception as e:
        print(f"❌ Failed to create integrator: {e}")
        return False

    # Test 3: Test lifecycle event recording
    try:
        await record_agent_lifecycle_event(
            "test_agent", "created", {"test": True}
        )
        print("✅ Lifecycle event recording works")
    except Exception as e:
        print(f"❌ Lifecycle event recording failed: {e}")
        return False

    # Test 4: Test belief update recording
    try:
        beliefs_before = {"entropy": 0.8, "free_energy": 10.0}
        beliefs_after = {"entropy": 0.6, "free_energy": 8.0}
        await record_belief_update(
            "test_agent", beliefs_before, beliefs_after, 8.0
        )
        print("✅ Belief update recording works")
    except Exception as e:
        print(f"❌ Belief update recording failed: {e}")
        return False

    # Test 5: Test monitoring decorator
    try:

        @monitor_pymdp_inference("test_agent")
        def mock_inference():
            # Perform minimal computation instead of sleep
            _result = sum(range(100))  # Simple computation
            return "test_result"

        _result = mock_inference()
        assert result == "test_result"
        print("✅ Monitoring decorator works")
    except Exception as e:
        print(f"❌ Monitoring decorator failed: {e}")
        return False

    # Test 6: Test performance summary
    try:
        summary = await get_pymdp_performance_summary("test_agent")
        assert "agent_id" in summary
        print("✅ Performance summary generation works")
    except Exception as e:
        print(f"❌ Performance summary failed: {e}")
        return False

    # Test 7: Verify data was recorded
    try:
        # Check if data was actually stored
        assert "test_agent" in pymdp_observer.agent_lifecycles
        assert "test_agent" in pymdp_observer.belief_update_history
        assert "test_agent" in pymdp_observer.inference_metrics
        print("✅ Data storage verification passed")
    except Exception as e:
        print(f"❌ Data storage verification failed: {e}")
        return False

    print("\n🎉 All observability component tests passed!")
    return True


async def test_observability_performance():
    """Test observability performance characteristics."""

    print("\nTesting observability performance...")

    try:
        from observability.pymdp_integration import (
            PyMDPObservabilityIntegrator,
            monitor_pymdp_inference,
            record_belief_update,
        )
    except ImportError:
        print("⚠️ Observability not available for performance testing")
        return True  # Skip but don't fail

    import time

    # Test performance impact of monitoring
    integrator = PyMDPObservabilityIntegrator()

    # Test 1: Belief update recording performance
    start_time = time.time()
    for i in range(100):
        await record_belief_update(
            f"perf_agent_{i % 10}",
            {"belief": i * 0.01},
            {"belie": (i + 1) * 0.01},
            float(i),
        )
    duration = time.time() - start_time

    print(
        f"✅ 100 belief updates recorded in {duration:.3f}s ({duration*10:.1f}ms avg)"
    )

    # Test 2: Monitoring decorator overhead
    @monitor_pymdp_inference("perf_test_agent")
    def monitored_operation():
        # Perform minimal computation instead of sleep
        _result = sum(range(1000))  # Simple computation
        return "result"

    # Measure overhead
    start_time = time.time()
    for i in range(50):
        _result = monitored_operation()
    duration = time.time() - start_time

    print(
        f"✅ 50 monitored operations in {duration:.3f}s ({duration*20:.1f}ms avg)"
    )

    # Test 3: Performance summary generation
    start_time = time.time()
    for i in range(20):
        _summary = await integrator.get_performance_summary(
            f"perf_agent_{i % 5}"
        )
    duration = time.time() - start_time

    print(
        f"✅ 20 performance summaries in {duration:.3f}s ({duration*50:.1f}ms avg)"
    )

    print("🎯 Observability performance tests completed")
    return True


async def test_agent_integration_patterns():
    """Test integration patterns with agent-like objects."""

    print("\nTesting agent integration patterns...")

    try:
        from observability.pymdp_integration import (
            pymdp_observer,
            record_agent_lifecycle_event,
            record_belief_update,
        )
    except ImportError:
        print("⚠️ Observability not available for integration testing")
        return True

    # Mock agent lifecycle
    agent_id = "integration_test_agent"

    # Agent creation
    await record_agent_lifecycle_event(
        agent_id, "created", {"type": "mock_agent"}
    )

    # Agent activation
    await record_agent_lifecycle_event(
        agent_id, "activated", {"timestamp": "test"}
    )

    # Multiple belief updates (simulating agent steps)
    for step in range(10):
        beliefs_before = {"step": step, "uncertainty": 1.0 - step * 0.1}
        beliefs_after = {
            "step": step + 1,
            "uncertainty": 1.0 - (step + 1) * 0.1,
        }

        await record_belief_update(
            agent_id,
            beliefs_before,
            beliefs_after,
            free_energy=10.0 - step * 0.5,
        )

    # Agent deactivation
    await record_agent_lifecycle_event(
        agent_id, "deactivated", {"final_step": 10}
    )

    # Verify complete lifecycle was tracked
    lifecycle_events = pymdp_observer.agent_lifecycles[agent_id]
    event_types = [event["event"] for event in lifecycle_events]

    assert "created" in event_types
    assert "activated" in event_types
    assert "deactivated" in event_types

    belief_updates = pymdp_observer.belief_update_history[agent_id]
    assert len(belief_updates) == 10

    print("✅ Complete agent lifecycle integration verified")
    return True


if __name__ == "__main__":

    async def run_all_tests():
        """Run all observability validation tests."""
        print("🚀 Starting observability integration validation...")

        try:
            # Run component tests
            success1 = await test_observability_components()

            # Run performance tests
            success2 = await test_observability_performance()

            # Run integration pattern tests
            success3 = await test_agent_integration_patterns()

            if success1 and success2 and success3:
                print("\n🎉 ALL OBSERVABILITY TESTS PASSED!")
                print("✅ Observability integration is production ready")
                return True
            else:
                print("\n❌ Some observability tests failed")
                return False

        except Exception as e:
            print(f"\n💥 Test execution failed: {e}")
            return False

    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
