#!/usr/bin/env python3
"""Debug PyMDP B matrix format requirements."""

import numpy as np
from pymdp import utils

# Test different B matrix formats
print("Testing PyMDP B matrix normalization requirements...")

# Format 1: (num_actions, num_states, num_states) - rows sum to 1
B1 = np.zeros((3, 3, 3))
for action in range(3):
    B1[action] = np.eye(3)

print("\nFormat 1 - Identity matrices:")
print(f"Shape: {B1.shape}")
print(f"Sum along axis 0: {B1.sum(axis=0)}")
print(f"Sum along axis 1: {B1.sum(axis=1)}")
print(f"Sum along axis 2: {B1.sum(axis=2)}")
print(f"Is normalized (axis 0): {utils.is_normalized(B1, axis=0)}")
print(f"Is normalized (axis 1): {utils.is_normalized(B1, axis=1)}")

# Format 2: Transpose to make columns sum to 1
B2 = np.zeros((3, 3, 3))
for action in range(3):
    B2[action] = np.eye(3).T

print("\nFormat 2 - Transposed identity:")
print(f"Shape: {B2.shape}")
print(f"Sum along axis 0: {B2.sum(axis=0)}")
print(f"Sum along axis 1: {B2.sum(axis=1)}")
print(f"Is normalized: {utils.is_normalized(B2)}")

# Format 3: Check what PyMDP actually expects
print("\nChecking utils.is_normalized on B matrix...")
# B[action, next_state, current_state] - so sum over next_state (axis=1) should be 1
B3 = np.zeros((3, 3, 3))
for action in range(3):
    for curr_state in range(3):
        # From current state, distribute probability to next states
        B3[action, :, curr_state] = [0.33, 0.33, 0.34]  # Sums to 1

print(f"Format 3 - Manual normalization:")
print(f"Shape: {B3.shape}")
print(f"Sum along axis 1 (over next states): {B3.sum(axis=1)}")
print(f"Is normalized: {utils.is_normalized(B3)}")

# Let's check the actual implementation
print("\nChecking is_normalized default behavior...")
test_B = np.eye(3)[np.newaxis, :, :].repeat(3, axis=0)
print(f"Test B shape: {test_B.shape}")
obj_array_B = utils.to_obj_array(test_B)
print(f"Object array shape: {[b.shape for b in obj_array_B]}")

# The correct format based on PyMDP docs
print("\nCorrect format (from PyMDP docs):")
# B[f][a, s', s] where:
# f = factor index (for factorized state spaces)
# a = action index
# s' = next state
# s = current state
# Should sum to 1 over s' (axis 1) for each a, s
B_correct = np.zeros((3, 3, 3))
for a in range(3):
    B_correct[a] = np.eye(3)  # Identity transition

print(f"B_correct sum over axis 1: {B_correct.sum(axis=1)}")
print(f"Each element should be 1.0")

# Test with PyMDP's actual normalization check
print(f"\nDirect check with axis=1: {np.all(B_correct.sum(axis=1) == 1.0)}")
