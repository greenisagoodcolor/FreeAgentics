#!/usr/bin/env python3
"""Test PyMDP compatibility fix."""

import numpy as np
from pymdp.agent import Agent

print("Testing PyMDP compatibility...")

# Simple model
A = np.ones((3, 3)) / 3
B = np.eye(3).reshape(3, 3, 1).repeat(3, axis=2)
C = np.array([0.0, 0.0, 1.0])
D = np.ones(3) / 3

# Try creating agent without control_fns (should work with newer PyMDP)
try:
    agent = Agent(A=A, B=B, C=C, D=D)
    print("✅ Agent created successfully without control_fns")
except TypeError as e:
    if "control_fns" in str(e):
        print("❌ Still getting control_fns error:", e)
    else:
        print("❌ Different error:", e)

# Check for F attribute
if hasattr(agent, "F"):
    print("✅ Agent has F attribute")
else:
    print("⚠️ Agent missing F attribute - adding it")
    agent.F = 0.0

# Test inference
try:
    obs = 0
    qs = agent.infer_states(obs)
    print(f"✅ Belief inference works: shape = {qs[0].shape}")

    q_pi, G = agent.infer_policies()
    print("✅ Policy inference works")

    action = agent.sample_action()
    print(f"✅ Action selection works: action = {action}")
except Exception as e:
    print(f"❌ Inference failed: {e}")

print("\nPyMDP compatibility test complete!")
