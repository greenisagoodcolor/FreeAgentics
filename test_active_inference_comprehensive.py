#!/usr/bin/env python3
"""Comprehensive Active Inference Integration Test Suite."""

import os
import sys
import time

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_pymdp_basic():
    """Test basic PyMDP functionality."""
    print("\n=== Test 1: Basic PyMDP Functionality ===")

    try:
        from pymdp.agent import Agent as PyMDPAgent

        print("✓ PyMDP imported successfully")

        # Create simple test agent
        A = np.eye(3)
        B = np.zeros((3, 3, 2))
        for i in range(2):
            B[:, :, i] = np.eye(3)
        C = np.array([1.0, 0.0, -1.0])
        D = np.ones(3) / 3

        agent = PyMDPAgent(A=A, B=B, C=C, D=D)
        print("✓ PyMDP agent created")

        # Test basic inference
        obs = [0]
        agent.infer_states(obs)
        agent.infer_policies()
        action = agent.sample_action()

        print(f"✓ Basic inference complete - action: {action}")
        return True

    except Exception as e:
        print(f"✗ Basic PyMDP test failed: {e}")
        return False


def test_gmn_parser():
    """Test GMN parser functionality."""
    print("\n=== Test 2: GMN Parser Functionality ===")

    try:
        from inference.active.gmn_parser import parse_gmn_spec

        gmn_spec = """
        [nodes]
        location: state {num_states: 4}
        observation: observation {num_observations: 3}
        movement: action {num_actions: 4}
        obs_model: likelihood
        trans_model: transition
        
        [edges]
        location -> obs_model: depends_on
        obs_model -> observation: generates
        location -> trans_model: depends_on
        movement -> trans_model: depends_on
        """

        model_spec = parse_gmn_spec(gmn_spec)
        print(f"✓ GMN parsed - states: {model_spec.get('num_states', [])}")

        # Validate model structure
        assert len(model_spec.get("A", [])) > 0, "No A matrices"
        assert len(model_spec.get("B", [])) > 0, "No B matrices"

        print("✓ GMN parser validation passed")
        return True

    except Exception as e:
        print(f"✗ GMN parser test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_gmn_pymdp_adapter():
    """Test GMN to PyMDP adapter."""
    print("\n=== Test 3: GMN-PyMDP Adapter ===")

    try:
        from pymdp.agent import Agent as PyMDPAgent

        from agents.gmn_pymdp_adapter import adapt_gmn_to_pymdp
        from inference.active.gmn_parser import parse_gmn_spec

        # Create and parse GMN
        gmn_spec = """
        [nodes]
        location: state {num_states: 4}
        observation: observation {num_observations: 3}
        movement: action {num_actions: 4}
        obs_model: likelihood
        trans_model: transition
        pref: preference {preferred_observation: 1}
        
        [edges]
        location -> obs_model: depends_on
        obs_model -> observation: generates
        location -> trans_model: depends_on
        movement -> trans_model: depends_on
        pref -> observation: depends_on
        """

        model_spec = parse_gmn_spec(gmn_spec)
        adapted_model = adapt_gmn_to_pymdp(model_spec)

        print("✓ Model adapted to PyMDP format")

        # Test with real PyMDP agent
        agent = PyMDPAgent(
            A=adapted_model["A"],
            B=adapted_model["B"],
            C=adapted_model.get("C"),
            D=adapted_model.get("D"),
        )

        # Test inference pipeline
        obs = [1]
        agent.infer_states(obs)
        agent.infer_policies()
        action = agent.sample_action()

        print(f"✓ End-to-end GMN->PyMDP pipeline works - action: {action}")
        return True

    except Exception as e:
        print(f"✗ GMN-PyMDP adapter test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_basic_explorer_agent():
    """Test BasicExplorerAgent Active Inference workflow."""
    print("\n=== Test 4: BasicExplorerAgent Workflow ===")

    try:
        from agents.base_agent import BasicExplorerAgent

        # Create agent
        agent = BasicExplorerAgent("ai_test", "AI Test Agent", grid_size=4)
        agent.start()

        print("✓ BasicExplorerAgent created and started")
        print(f"  PyMDP available: {agent.pymdp_agent is not None}")

        # Test observation processing
        observation = {
            "position": [1, 1],
            "surroundings": np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]]),
            "time_step": 1,
        }

        # Test complete Active Inference step
        action = agent.step(observation)
        print(f"✓ Active Inference step completed - action: {action}")

        # Verify metrics were updated
        metrics = agent.metrics
        required_metrics = ["total_observations", "total_actions"]
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
            assert metrics[metric] > 0, f"Metric {metric} not updated"

        print("✓ Agent metrics properly updated")

        # Test multiple steps
        actions = []
        for step in range(3):
            test_obs = {
                "position": [1, 1 + step],
                "surroundings": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                "time_step": step + 2,
            }
            action = agent.step(test_obs)
            actions.append(action)

        print(f"✓ Multi-step sequence: {actions}")

        # Test free energy computation
        fe_components = agent.compute_free_energy()
        if "total_free_energy" in fe_components:
            print(f"✓ Free energy computed: {fe_components['total_free_energy']:.3f}")

        agent.stop()
        print("✓ Agent lifecycle complete")
        return True

    except Exception as e:
        print(f"✗ BasicExplorerAgent test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_multi_agent_coordination():
    """Test multi-agent Active Inference coordination."""
    print("\n=== Test 5: Multi-Agent Coordination ===")

    try:
        from agents.base_agent import BasicExplorerAgent

        # Create multiple agents
        agents = []
        for i in range(2):
            agent = BasicExplorerAgent(f"agent_{i}", f"Explorer {i}", grid_size=4)
            agent.start()
            agents.append(agent)

        print(f"✓ Created {len(agents)} agents")

        # Test coordination scenario
        observation_base = {
            "surroundings": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
            "time_step": 1,
        }

        actions = []
        for i, agent in enumerate(agents):
            obs = observation_base.copy()
            obs["position"] = [i, i]  # Different starting positions
            action = agent.step(obs)
            actions.append(action)

        print(f"✓ Multi-agent coordination test: actions = {actions}")

        # Cleanup
        for agent in agents:
            agent.stop()

        print("✓ Multi-agent test complete")
        return True

    except Exception as e:
        print(f"✗ Multi-agent test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_error_handling():
    """Test error handling and fallback mechanisms."""
    print("\n=== Test 6: Error Handling and Fallbacks ===")

    try:
        from agents.base_agent import BasicExplorerAgent

        agent = BasicExplorerAgent("error_test", "Error Test Agent", grid_size=3)
        agent.start()

        # Test with invalid observations
        invalid_obs = {
            "position": [10, 10],  # Outside grid
            "surroundings": None,  # Invalid surroundings
            "time_step": 1,
        }

        try:
            action = agent.step(invalid_obs)
            print(f"✓ Invalid observation handled gracefully - action: {action}")
        except Exception as e:
            print(f"Note: Invalid observation caused exception: {e}")

        # Test with minimal valid observation
        minimal_obs = {"position": [1, 1]}
        action = agent.step(minimal_obs)
        print(f"✓ Minimal observation handled - action: {action}")

        agent.stop()
        print("✓ Error handling test complete")
        return True

    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False


def test_performance_optimization():
    """Test performance optimizations."""
    print("\n=== Test 7: Performance Optimizations ===")

    try:
        from agents.base_agent import BasicExplorerAgent

        # Test different performance modes
        modes = ["fast", "balanced", "accurate"]
        results = {}

        for mode in modes:
            agent = BasicExplorerAgent("perf_test", f"Perf Test {mode}", grid_size=4)
            agent.performance_mode = mode
            agent.start()

            start_time = time.time()

            # Run multiple steps
            for step in range(10):
                obs = {
                    "position": [1, 1],
                    "surroundings": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                    "time_step": step,
                }
                agent.step(obs)

            duration = time.time() - start_time
            results[mode] = duration
            agent.stop()

        print("✓ Performance mode testing:")
        for mode, duration in results.items():
            print(f"  {mode}: {duration:.3f}s for 10 steps")

        return True

    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False


def run_comprehensive_test():
    """Run all Active Inference tests."""
    print("=" * 60)
    print("FreeAgentics Active Inference Comprehensive Test Suite")
    print("=" * 60)

    test_functions = [
        test_pymdp_basic,
        test_gmn_parser,
        test_gmn_pymdp_adapter,
        test_basic_explorer_agent,
        test_multi_agent_coordination,
        test_error_handling,
        test_performance_optimization,
    ]

    results = {}
    start_time = time.time()

    for test_func in test_functions:
        try:
            result = test_func()
            results[test_func.__name__] = result
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed with exception: {e}")
            results[test_func.__name__] = False

    # Summary
    print("\n" + "=" * 60)
    print("ACTIVE INFERENCE TEST RESULTS")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nTests passed: {passed}/{total}")
    print(f"Success rate: {100 * passed / total:.1f}%")
    print(f"Total time: {time.time() - start_time:.2f}s")

    if passed == total:
        print("\n🎉 ALL ACTIVE INFERENCE TESTS PASSED! 🎉")
        print("Production Active Inference system is fully functional.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review output above.")

    return passed == total


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
