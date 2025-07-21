#!/usr/bin/env python3
"""Debug script to understand PyMDP API behavior."""

import numpy as np
from pymdp import utils
from pymdp.agent import Agent as PyMDPAgent


def test_pymdp_api():
    """Test PyMDP API to understand actual return types."""
    print("Testing PyMDP API behavior...")

    # Create minimal agent
    num_obs = [3]
    num_states = [3]
    num_controls = [3]

    A = utils.random_A_matrix(num_obs, num_states)
    B = utils.random_B_matrix(num_states, num_controls)

    agent = PyMDPAgent(A=A, B=B)

    print(f"Agent created with A type: {type(A)}, B type: {type(B)}")

    # Test infer_states
    print("\n=== Testing infer_states ===")
    try:
        obs = [0]
        result = agent.infer_states(obs)
        print("infer_states returned:")
        print(f"  Type: {type(result)}")
        print(f"  Value: {result}")

        if hasattr(result, "__len__"):
            print(f"  Length: {len(result)}")

        if isinstance(result, list):
            for i, item in enumerate(result):
                print(f"  Item {i}: type={type(item)}, value={item}")
        elif hasattr(result, "shape"):
            print(f"  Shape: {result.shape}")
            print(f"  Dtype: {result.dtype}")

        # Test extracting from object array
        if isinstance(result, np.ndarray) and result.dtype == np.object_:
            item = result.item()
            print(f"  Extracted item: type={type(item)}, value={item}")
            if isinstance(item, np.ndarray):
                print(f"    Item shape: {item.shape}, dtype: {item.dtype}")

    except Exception as e:
        print(f"infer_states failed: {e}")

    # Test different observation formats
    print("\n=== Testing different observation formats ===")
    for obs_test in [0, [1], np.array([2])]:
        try:
            print(f"\nTesting observation: {obs_test} (type: {type(obs_test)})")
            result = agent.infer_states(
                obs_test
                if isinstance(obs_test, list)
                else [obs_test] if isinstance(obs_test, int) else obs_test.tolist()
            )
            print(f"  Success: {type(result)}")
            if isinstance(result, np.ndarray) and result.dtype == np.object_:
                item = result.item()
                print(f"  Extracted: {type(item)}")
        except Exception as e:
            print(f"  Failed: {e}")

    # Test infer_policies
    print("\n=== Testing infer_policies ===")
    try:
        result = agent.infer_policies()
        print("infer_policies returned:")
        print(f"  Type: {type(result)}")
        print(f"  Value: {result}")

        if isinstance(result, tuple):
            print(f"  Tuple length: {len(result)}")
            for i, item in enumerate(result):
                print(f"  Item {i}: type={type(item)}, shape={getattr(item, 'shape', 'N/A')}")
    except Exception as e:
        print(f"infer_policies failed: {e}")

    # Test sample_action
    print("\n=== Testing sample_action ===")
    try:
        result = agent.sample_action()
        print("sample_action returned:")
        print(f"  Type: {type(result)}")
        print(f"  Value: {result}")
        print(f"  Shape: {getattr(result, 'shape', 'N/A')}")
        print(f"  Dtype: {getattr(result, 'dtype', 'N/A')}")
    except Exception as e:
        print(f"sample_action failed: {e}")

    # Test full workflow
    print("\n=== Testing full workflow ===")
    try:
        # Step 1: infer states
        obs = [0]
        qs = agent.infer_states(obs)
        print(f"Step 1 - infer_states: {type(qs)}")

        # Step 2: infer policies
        q_pi, G = agent.infer_policies()
        print(f"Step 2 - infer_policies: q_pi={type(q_pi)}, G={type(G)}")

        # Step 3: sample action
        action = agent.sample_action()
        print(f"Step 3 - sample_action: {type(action)}")

    except Exception as e:
        print(f"Full workflow failed: {e}")


if __name__ == "__main__":
    test_pymdp_api()
