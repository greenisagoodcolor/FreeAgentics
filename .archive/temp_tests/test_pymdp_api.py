"""Test script to understand PyMDP API"""

import pymdp
from pymdp import utils

# Test 1: Check how to create matrices
print("Testing PyMDP API...")

# Simple 2-state, 2-observation model
num_obs = [2]  # Single modality with 2 observations
num_states = [2]  # Single hidden state factor with 2 states

# Create A matrix (observation model)
A = utils.obj_array_zeros([(obs, *num_states) for obs in num_obs])
A[0][0, 0] = 0.9  # P(obs=0|state=0)
A[0][1, 0] = 0.1  # P(obs=1|state=0)
A[0][0, 1] = 0.1  # P(obs=0|state=1)
A[0][1, 1] = 0.9  # P(obs=1|state=1)

print(f"A shape: {A[0].shape}")
print(f"A type: {type(A)}")

# Create B matrix (transition model)
num_controls = [2]  # 2 possible actions
B = utils.obj_array_zeros([(s, s, num_controls[i]) for i, s in enumerate(num_states)])
# Action 0: Stay in same state
B[0][0, 0, 0] = 0.9
B[0][1, 0, 0] = 0.1
B[0][0, 1, 0] = 0.1
B[0][1, 1, 0] = 0.9
# Action 1: Switch states
B[0][0, 0, 1] = 0.1
B[0][1, 0, 1] = 0.9
B[0][0, 1, 1] = 0.9
B[0][1, 1, 1] = 0.1

print(f"B shape: {B[0].shape}")

try:
    # Create agent
    agent = pymdp.agent.Agent(A=A, B=B, policy_len=1)
    print("Agent created successfully!")

    # Test inference
    obs = [0]  # Observe state 0
    qs = agent.infer_states(obs)
    print(f"Belief state: {qs[0]}")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
