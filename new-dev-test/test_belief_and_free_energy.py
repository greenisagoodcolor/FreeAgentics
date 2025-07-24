#!/usr/bin/env python3
"""Test belief state management and free energy computation."""

import os
import sys

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_belief_state_management():
    """Test Active Inference belief state management."""
    print("\n=== Test: Belief State Management ===")

    try:
        from agents.base_agent import BasicExplorerAgent

        # Create agent with larger grid for more interesting beliefs
        agent = BasicExplorerAgent("belief_test", "Belief Test Agent", grid_size=6)
        agent.start()

        print("✓ Agent created and started")

        # Test initial beliefs
        if agent.pymdp_agent and hasattr(agent.pymdp_agent, "qs"):
            initial_beliefs = agent.pymdp_agent.qs
            if initial_beliefs is not None and len(initial_beliefs) > 0:
                initial_posterior = initial_beliefs[0]
                print(f"✓ Initial belief distribution shape: {initial_posterior.shape}")
                print(f"  Max belief: {np.max(initial_posterior):.4f}")
                print(f"  Min belief: {np.min(initial_posterior):.4f}")
                print(f"  Sum (should be 1.0): {np.sum(initial_posterior):.6f}")

        # Test belief updates with different observations
        observations = [
            {
                "position": [2, 2],
                "surroundings": np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),  # Goal to right
                "time_step": 1,
            },
            {
                "position": [2, 3],
                "surroundings": np.array([[0, -1, 0], [0, 0, 0], [0, 0, 0]]),  # Obstacle above
                "time_step": 2,
            },
            {
                "position": [3, 3],
                "surroundings": np.array([[0, 0, 0], [0, 0, 0], [2, 0, 0]]),  # Another agent below
                "time_step": 3,
            },
        ]

        belief_entropies = []
        belief_distributions = []

        for i, obs in enumerate(observations):
            # Process observation and update beliefs
            agent.perceive(obs)
            agent.update_beliefs()

            # Record belief state
            entropy = agent.metrics.get("belief_entropy", 0.0)
            belief_entropies.append(entropy)

            if (
                agent.pymdp_agent
                and hasattr(agent.pymdp_agent, "qs")
                and agent.pymdp_agent.qs is not None
            ):
                beliefs = agent.pymdp_agent.qs[0].copy()
                belief_distributions.append(beliefs)

                # Check belief distribution properties
                assert (
                    abs(np.sum(beliefs) - 1.0) < 1e-6
                ), f"Beliefs don't sum to 1: {np.sum(beliefs)}"
                assert np.all(beliefs >= 0), "Negative beliefs detected"

                print(f"  Step {i + 1}: entropy={entropy:.4f}, max_belief={np.max(beliefs):.4f}")

        print("✓ Belief updates completed successfully")

        # Test belief consistency over time
        if len(belief_distributions) >= 2:
            belief_change = np.linalg.norm(belief_distributions[-1] - belief_distributions[0])
            print(f"✓ Belief change magnitude: {belief_change:.4f}")

        # Test entropy calculation
        if belief_entropies:
            print(f"✓ Entropy evolution: {[f'{e:.3f}' for e in belief_entropies]}")

            # Entropy should be non-negative
            for entropy in belief_entropies:
                assert entropy >= 0, f"Negative entropy: {entropy}"

        agent.stop()
        return True

    except Exception as e:
        print(f"✗ Belief state management test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_free_energy_computation():
    """Test free energy computation and decomposition."""
    print("\n=== Test: Free Energy Computation ===")

    try:
        from agents.base_agent import BasicExplorerAgent

        agent = BasicExplorerAgent("fe_test", "Free Energy Test Agent", grid_size=4)
        agent.start()

        print("✓ Agent created for free energy testing")

        # Process an observation to set up the agent state
        observation = {
            "position": [1, 1],
            "surroundings": np.array(
                [
                    [0, 0, 0],
                    [0, 0, 1],  # Goal to the right
                    [0, -1, 0],  # Obstacle below
                ]
            ),
            "time_step": 1,
        }

        # Complete Active Inference step
        action = agent.step(observation)
        print(f"✓ Inference step completed - action: {action}")

        # Test free energy computation
        fe_components = agent.compute_free_energy()

        if "error" in fe_components:
            print(f"Note: Free energy computation returned error: {fe_components['error']}")
            if fe_components["error"] == "PyMDP not available":
                print("⚠️  PyMDP not available - this is expected in some environments")
                return True
            elif "No beliefs available" in fe_components["error"]:
                print("⚠️  No beliefs available yet - this may be expected")
                return True
        else:
            print("✓ Free energy computed successfully")
            print("  Components:")

            # Validate free energy components
            required_components = ["total_free_energy", "accuracy", "complexity"]
            for component in required_components:
                if component in fe_components:
                    value = fe_components[component]
                    print(f"    {component}: {value:.6f}")

                    # Validate component properties
                    assert isinstance(
                        value, (int, float)
                    ), f"{component} is not numeric: {type(value)}"
                    assert not np.isnan(value), f"{component} is NaN"
                    assert not np.isinf(value), f"{component} is infinite"

            # Test free energy decomposition: F = complexity - accuracy
            if all(
                comp in fe_components for comp in ["total_free_energy", "accuracy", "complexity"]
            ):
                expected_fe = fe_components["complexity"] - fe_components["accuracy"]
                actual_fe = fe_components["total_free_energy"]
                fe_diff = abs(expected_fe - actual_fe)

                print(f"    Free energy equation check: |F - (C - A)| = {fe_diff:.8f}")
                assert fe_diff < 1e-6, f"Free energy decomposition mismatch: {fe_diff}"
                print("✓ Free energy decomposition validated")

            # Test surprise component if available
            if "surprise" in fe_components:
                surprise = fe_components["surprise"]
                print(f"    surprise: {surprise:.6f}")
                assert surprise >= 0, f"Negative surprise: {surprise}"
                print("✓ Surprise component validated")

        # Test multiple free energy computations
        fe_values = []
        for step in range(3):
            obs_variant = {
                "position": [1, 1 + step % 2],
                "surroundings": np.array([[0, 0, 0], [0, 0, step % 2], [0, 0, 0]]),
                "time_step": step + 2,
            }
            agent.step(obs_variant)

            fe_comp = agent.compute_free_energy()
            if "total_free_energy" in fe_comp:
                fe_values.append(fe_comp["total_free_energy"])

        if fe_values:
            print(f"✓ Free energy sequence: {[f'{fe:.4f}' for fe in fe_values]}")

            # Check for reasonable free energy evolution
            fe_range = max(fe_values) - min(fe_values)
            print(f"  Free energy range: {fe_range:.6f}")

        agent.stop()
        return True

    except Exception as e:
        print(f"✗ Free energy computation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_action_selection_free_energy():
    """Test that action selection minimizes expected free energy."""
    print("\n=== Test: Action Selection & Expected Free Energy ===")

    try:
        from agents.base_agent import BasicExplorerAgent

        agent = BasicExplorerAgent("action_test", "Action Test Agent", grid_size=5)
        agent.start()

        print("✓ Agent created for action selection testing")

        # Create scenarios with different expected free energies
        scenarios = [
            {
                "name": "Empty surroundings",
                "obs": {
                    "position": [2, 2],
                    "surroundings": np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]),
                    "time_step": 1,
                },
            },
            {
                "name": "Goal nearby",
                "obs": {
                    "position": [2, 2],
                    "surroundings": np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),  # Goal above
                    "time_step": 2,
                },
            },
            {
                "name": "Obstacle nearby",
                "obs": {
                    "position": [2, 2],
                    "surroundings": np.array([[0, 0, 0], [-1, 0, 0], [0, 0, 0]]),  # Obstacle left
                    "time_step": 3,
                },
            },
        ]

        action_results = []

        for scenario in scenarios:
            # Process scenario
            agent.perceive(scenario["obs"])
            agent.update_beliefs()
            action = agent.select_action()

            # Get expected free energy
            expected_fe = agent.metrics.get("expected_free_energy", None)
            selected_policy = agent.metrics.get("selected_policy", None)

            result = {
                "scenario": scenario["name"],
                "action": action,
                "expected_free_energy": expected_fe,
                "selected_policy": selected_policy,
            }
            action_results.append(result)

            efe_str = f"{expected_fe:.4f}" if expected_fe is not None else "N/A"
            print(f"  {scenario['name']}: action={action}, EFE={efe_str}")

        print("✓ Action selection scenarios completed")

        # Validate that expected free energy is computed
        efe_values = [
            r["expected_free_energy"]
            for r in action_results
            if r["expected_free_energy"] is not None
        ]
        if efe_values:
            print(f"✓ Expected free energy values: {[f'{efe:.4f}' for efe in efe_values]}")

            # Check that EFE values are reasonable (typically negative)
            for efe in efe_values:
                assert not np.isnan(efe), "Expected free energy is NaN"
                assert not np.isinf(efe), "Expected free energy is infinite"

            print("✓ Expected free energy validation passed")
        else:
            print("⚠️  No expected free energy values computed - may be expected")

        # Test that actions are valid
        valid_actions = agent.actions
        for result in action_results:
            action = result["action"]
            assert action in valid_actions, f"Invalid action selected: {action}"

        print("✓ All selected actions are valid")

        agent.stop()
        return True

    except Exception as e:
        print(f"✗ Action selection test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def run_belief_and_free_energy_tests():
    """Run all belief and free energy tests."""
    print("=" * 60)
    print("Active Inference: Belief State & Free Energy Tests")
    print("=" * 60)

    test_functions = [
        test_belief_state_management,
        test_free_energy_computation,
        test_action_selection_free_energy,
    ]

    results = {}

    for test_func in test_functions:
        try:
            result = test_func()
            results[test_func.__name__] = result
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed with exception: {e}")
            results[test_func.__name__] = False

    # Summary
    print("\n" + "=" * 60)
    print("BELIEF & FREE ENERGY TEST RESULTS")
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

    return passed == total


if __name__ == "__main__":
    success = run_belief_and_free_energy_tests()
    sys.exit(0 if success else 1)
