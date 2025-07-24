#!/usr/bin/env python3
"""Simple debug for PyMDP B matrix format."""

import numpy as np
from pymdp import utils

# Check PyMDP's is_normalized implementation
print("Checking how PyMDP validates B matrix normalization...")

# Create identity transition matrix
B = np.zeros((3, 3, 3))
for action in range(3):
    B[action] = np.eye(3)

print(f"\nB matrix shape: {B.shape}")
print(f"B[0] (first action):\n{B[0]}")

# Convert to object array as PyMDP does
B_obj = utils.to_obj_array(B)
print(f"\nB as object array has {len(B_obj)} elements")

# Check normalization manually
print("\nManual normalization check:")
print(f"Sum over axis 0: \n{B.sum(axis=0)}")
print(f"Sum over axis 1: \n{B.sum(axis=1)}")
print(f"Sum over axis 2: \n{B.sum(axis=2)}")

# Let's see what PyMDP expects by looking at its normalization
print("\nChecking PyMDP is_normalized on B:")
print(f"Is normalized: {utils.is_normalized(B_obj)}")

# Try transposing to see if that helps
B_transposed = B.transpose(0, 2, 1)  # Swap last two dimensions
B_transposed_obj = utils.to_obj_array(B_transposed)
print(f"\nTransposed B is normalized: {utils.is_normalized(B_transposed_obj)}")

print("\nLet me check what axis PyMDP sums over...")
# Look at the source to understand
import inspect

print(inspect.getsource(utils.is_normalized))
