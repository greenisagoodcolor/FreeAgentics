#!/usr/bin/env python3
"""Test GMN parser and PyMDP integration."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test imports
print("Testing imports...")
try:
    from inference.active.gmn_parser import GMNParser
    print("✓ GMN parser imported")
except Exception as e:
    print(f"✗ GMN parser import failed: {e}")

try:
    import pymdp
    print("✓ PyMDP imported")
except Exception as e:
    print(f"✗ PyMDP import failed: {e}")

try:
    from pymdp.agent import Agent as PyMDPAgent
    print("✓ PyMDP Agent imported")
except Exception as e:
    print(f"✗ PyMDP Agent import failed: {e}")

# Test GMN parsing
print("\nTesting GMN parsing...")
gmn_spec = """
[nodes]
location: state {num_states: 4}
obs_location: observation {num_observations: 4}
move: action {num_actions: 5}

[edges]
location -> obs_location: depends_on
"""

try:
    parser = GMNParser()
    graph = parser.parse(gmn_spec)
    print(f"✓ GMN parsed: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    # Convert to PyMDP
    model = parser.to_pymdp_model(graph)
    print(f"✓ PyMDP model created: states={model.get('num_states')}, obs={model.get('num_obs')}, actions={model.get('num_actions')}")
except Exception as e:
    print(f"✗ GMN parsing/conversion failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")