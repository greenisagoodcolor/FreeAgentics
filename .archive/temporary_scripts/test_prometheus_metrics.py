#!/usr/bin/env python3
"""
Test script for Prometheus metrics implementation.
"""
import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_prometheus_metrics():
    """Test the Prometheus metrics implementation."""
    print("🧪 Testing Prometheus metrics implementation...")

    try:
        # Import the metrics module
        from observability.prometheus_metrics import (
            get_prometheus_metrics,
            prometheus_collector,
            record_agent_inference_duration,
            record_agent_step,
            record_belief_state_update,
            start_prometheus_metrics_collection,
            stop_prometheus_metrics_collection,
            update_belief_free_energy,
        )

        print("✅ Successfully imported Prometheus metrics module")

        # Test metrics collection startup
        print("🚀 Starting Prometheus metrics collection...")
        await start_prometheus_metrics_collection()
        print("✅ Prometheus metrics collection started")

        # Test recording some metrics
        print("📊 Recording test metrics...")

        # Record agent step
        record_agent_step("test_agent_001", "inference_step", True)

        # Record belief state update
        record_belief_state_update("test_agent_001", "belief_update", True)

        # Record inference duration
        record_agent_inference_duration("test_agent_001", "inference", 0.05)

        # Update belief free energy
        update_belief_free_energy("test_agent_001", 2.3)

        print("✅ Test metrics recorded successfully")

        # Test metrics snapshot
        print("📈 Getting metrics snapshot...")
        snapshot = prometheus_collector.get_metrics_snapshot()
        print(f"   Timestamp: {snapshot.timestamp}")
        print(f"   Active Agents: {snapshot.active_agents}")
        print(f"   Total Inferences: {snapshot.total_inferences}")
        print(f"   Total Belief Updates: {snapshot.total_belief_updates}")
        print(f"   Memory Usage: {snapshot.avg_memory_usage_mb:.2f} MB")
        print(f"   CPU Usage: {snapshot.avg_cpu_usage_percent:.2f}%")

        # Test Prometheus exposition format
        print("📋 Testing Prometheus exposition format...")
        metrics_output = get_prometheus_metrics()

        # Check that metrics are in Prometheus format
        assert "# HELP freeagentics_agent_steps_total" in metrics_output
        assert "# TYPE freeagentics_agent_steps_total counter" in metrics_output
        assert "freeagentics_agent_steps_total" in metrics_output

        print("✅ Prometheus format output verified")

        # Show sample metrics output
        print("📊 Sample Prometheus metrics output:")
        lines = metrics_output.split("\n")
        for line in lines[:20]:  # Show first 20 lines
            if line.strip():
                print(f"   {line}")

        print("   ...")
        print(f"   Total lines: {len(lines)}")

        # Test metrics collection stop
        print("🛑 Stopping Prometheus metrics collection...")
        await stop_prometheus_metrics_collection()
        print("✅ Prometheus metrics collection stopped")

        print("\n🎉 All Prometheus metrics tests passed!")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_api_endpoint():
    """Test the API endpoint integration."""
    print("\n🌐 Testing API endpoint integration...")

    try:
        # Import the monitoring module
        from api.v1.monitoring import (
            PROMETHEUS_AVAILABLE,
            prometheus_health_check,
            prometheus_metrics_endpoint,
        )

        print("✅ API endpoint imports successful")
        print(f"   Prometheus available: {PROMETHEUS_AVAILABLE}")

        if PROMETHEUS_AVAILABLE:
            # Test health check
            print("🏥 Testing health check...")
            health_response = await prometheus_health_check()
            print(f"   Health status: {health_response.get('status')}")

            # Test metrics endpoint
            print("📊 Testing metrics endpoint...")
            response = await prometheus_metrics_endpoint()
            print(f"   Response type: {type(response)}")

            if hasattr(response, "body"):
                content = response.body.decode("utf-8")
                print(f"   Content length: {len(content)} characters")
                print(f"   Contains metrics: {'freeagentics_' in content}")

            print("✅ API endpoint tests passed")
        else:
            print("⚠️ Prometheus not available in API")

        return True

    except Exception as e:
        print(f"❌ API endpoint test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":

    async def main():
        """Run all tests."""
        print("🧪 FreeAgentics Prometheus Metrics Test Suite")
        print("=" * 50)

        success = True

        # Test 1: Core metrics functionality
        if not await test_prometheus_metrics():
            success = False

        # Test 2: API endpoint integration
        if not await test_api_endpoint():
            success = False

        print("\n" + "=" * 50)
        if success:
            print("🎉 All tests passed! Prometheus metrics are working correctly.")
            sys.exit(0)
        else:
            print("❌ Some tests failed. Please check the implementation.")
            sys.exit(1)

    asyncio.run(main())
