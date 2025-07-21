"""Adapter to convert GMN model format to PyMDP v0.0.7.1 format."""

import numpy as np
from typing import Dict, Any


def adapt_gmn_to_pymdp(gmn_model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert GMN parser output to PyMDP Agent compatible format.

    PyMDP v0.0.7.1 expects:
    - Single numpy arrays (not lists) for single-factor models
    - Properly normalized matrices

    Args:
        gmn_model: Output from GMNParser.to_pymdp_model()

    Returns:
        Dictionary with PyMDP compatible parameters
    """
    adapted = {}

    # Handle A matrix (likelihood)
    A = gmn_model.get("A", [])
    if isinstance(A, list) and len(A) == 1:
        # Single factor model - extract the array
        adapted["A"] = A[0]
    elif isinstance(A, list) and len(A) > 1:
        # Multi-factor model - PyMDP v0.0.7.1 might not support this
        raise ValueError("Multi-factor models not supported in PyMDP v0.0.7.1")
    else:
        adapted["A"] = A

    # Handle B matrix (transition)
    B = gmn_model.get("B", [])
    if isinstance(B, list) and len(B) == 1:
        adapted["B"] = B[0]
    elif isinstance(B, list) and len(B) > 1:
        raise ValueError("Multi-factor models not supported in PyMDP v0.0.7.1")
    else:
        adapted["B"] = B

    # Handle C vector (preferences) - can be None
    C = gmn_model.get("C", None)
    if isinstance(C, list) and len(C) == 1:
        adapted["C"] = C[0]
    elif isinstance(C, list) and len(C) == 0:
        adapted["C"] = None
    else:
        adapted["C"] = C

    # Handle D vector (initial beliefs) - can be None
    D = gmn_model.get("D", None)
    if isinstance(D, list) and len(D) == 1:
        adapted["D"] = D[0]
    elif isinstance(D, list) and len(D) == 0:
        adapted["D"] = None
    else:
        adapted["D"] = D

    # Validate normalization
    if "A" in adapted and adapted["A"] is not None:
        A_sums = adapted["A"].sum(axis=0)
        if not np.allclose(A_sums, 1.0):
            # Renormalize if needed
            adapted["A"] = adapted["A"] / A_sums[np.newaxis, :]

    return adapted
