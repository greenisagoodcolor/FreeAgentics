#!/usr/bin/env python3
"""Debug GMN to PyMDP conversion issue."""

import numpy as np
from inference.active.gmn_parser import GMNParser

# Use complete GMN spec with all required nodes
from inference.active.gmn_parser import EXAMPLE_GMN_SPEC
gmn_spec = EXAMPLE_GMN_SPEC

# Parse and convert
parser = GMNParser()
graph = parser.parse(gmn_spec)
model = parser.to_pymdp_model(graph)

print("Model structure:")
for key, value in model.items():
    if isinstance(value, list):
        print(f"  {key}: {len(value)} items")
        for i, item in enumerate(value):
            print(f"    [{i}] type={type(item)}, shape={getattr(item, 'shape', 'N/A')}")
    else:
        print(f"  {key}: {type(value)}")

# Check if matrices are numpy arrays
print("\nChecking A matrices:")
if "A" in model:
    for i, A in enumerate(model["A"]):
        print(f"  A[{i}]: is numpy? {isinstance(A, np.ndarray)}, type={type(A)}")

print("\nChecking B matrices:")  
if "B" in model:
    for i, B in enumerate(model["B"]):
        print(f"  B[{i}]: is numpy? {isinstance(B, np.ndarray)}, type={type(B)}")

# Try creating PyMDP agent
print("\nTrying to create PyMDP agent...")
try:
    from pymdp.agent import Agent as PyMDPAgent
    
    # PyMDP expects A, B as positional args
    # Extract matrices from model
    A = model.get("A", [])
    B = model.get("B", [])
    C = model.get("C", None) 
    D = model.get("D", None)
    
    print(f"\nCreating agent with:")
    print(f"  A type: {type(A)}, len: {len(A)}")
    print(f"  B type: {type(B)}, len: {len(B)}")
    print(f"  C type: {type(C)}, len: {len(C) if C else 'None'}")
    print(f"  D type: {type(D)}, len: {len(D) if D else 'None'}")
    
    # Debug what's in A
    print(f"\nA is a {type(A)}:")
    if isinstance(A, list):
        print(f"  A[0] type: {type(A[0])}, shape: {A[0].shape if hasattr(A[0], 'shape') else 'N/A'}")
    
    if A and B:
        # Use adapter to convert format
        from agents.gmn_pymdp_adapter import adapt_gmn_to_pymdp
        
        adapted_model = adapt_gmn_to_pymdp(model)
        print(f"\nAfter adaptation:")
        print(f"  A type: {type(adapted_model['A'])}, shape: {adapted_model['A'].shape}")
        print(f"  B type: {type(adapted_model['B'])}, shape: {adapted_model['B'].shape}")
        print(f"  A normalized? {np.allclose(adapted_model['A'].sum(axis=0), 1.0)}")
        
        agent = PyMDPAgent(
            A=adapted_model['A'], 
            B=adapted_model['B'],
            C=adapted_model.get('C'),
            D=adapted_model.get('D')
        )
        print("\nSUCCESS: PyMDP agent created!")
    else:
        print("ERROR: Missing required A or B matrices")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()