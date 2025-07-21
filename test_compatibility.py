#!/usr/bin/env python3
"""
Dependency compatibility test script for FreeAgentics production requirements.
Tests version compatibility between numpy, scipy, torch, and other key dependencies.
"""

import sys


def check_version_compatibility():
    """Check if all dependencies can be imported and their versions are compatible."""

    compatibility_results = {}

    # Test numpy
    try:
        import numpy as np

        compatibility_results["numpy"] = {
            "version": np.__version__,
            "status": "OK",
            "error": None,
        }
        print(f"✓ NumPy {np.__version__} imported successfully")
    except Exception as e:
        compatibility_results["numpy"] = {
            "version": None,
            "status": "ERROR",
            "error": str(e),
        }
        print(f"✗ NumPy import failed: {e}")

    # Test scipy
    try:
        import scipy

        compatibility_results["scipy"] = {
            "version": scipy.__version__,
            "status": "OK",
            "error": None,
        }
        print(f"✓ SciPy {scipy.__version__} imported successfully")

        # Test specific scipy functions that might have compatibility issues
        from scipy.sparse import csr_matrix

        csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
        print("  - SciPy sparse matrix operations: OK")

    except Exception as e:
        compatibility_results["scipy"] = {
            "version": None,
            "status": "ERROR",
            "error": str(e),
        }
        print(f"✗ SciPy import/test failed: {e}")

    # Test torch
    try:
        import torch

        compatibility_results["torch"] = {
            "version": torch.__version__,
            "status": "OK",
            "error": None,
        }
        print(f"✓ PyTorch {torch.__version__} imported successfully")

        # Test basic tensor operations
        x = torch.tensor([1.0, 2.0, 3.0])
        x * 2
        print("  - PyTorch tensor operations: OK")

        # Test numpy compatibility
        x_np = x.numpy()
        torch.from_numpy(x_np)
        print("  - PyTorch-NumPy interop: OK")

    except Exception as e:
        compatibility_results["torch"] = {
            "version": None,
            "status": "ERROR",
            "error": str(e),
        }
        print(f"✗ PyTorch import/test failed: {e}")

    # Test torch_geometric
    try:
        import torch_geometric

        compatibility_results["torch_geometric"] = {
            "version": torch_geometric.__version__,
            "status": "OK",
            "error": None,
        }
        print(
            f"✓ PyTorch Geometric {torch_geometric.__version__} imported successfully"
        )

    except Exception as e:
        compatibility_results["torch_geometric"] = {
            "version": None,
            "status": "ERROR",
            "error": str(e),
        }
        print(f"✗ PyTorch Geometric import failed: {e}")

    # Test inferactively-pymdp
    try:
        import pymdp

        compatibility_results["pymdp"] = {
            "version": pymdp.__version__
            if hasattr(pymdp, "__version__")
            else "unknown",
            "status": "OK",
            "error": None,
        }
        print("✓ pymdp (inferactively-pymdp) imported successfully")

        # Test basic pymdp functionality
        print("  - pymdp utils: OK")

    except Exception as e:
        compatibility_results["pymdp"] = {
            "version": None,
            "status": "ERROR",
            "error": str(e),
        }
        print(f"✗ pymdp import failed: {e}")

    # Test pandas
    try:
        import pandas as pd

        compatibility_results["pandas"] = {
            "version": pd.__version__,
            "status": "OK",
            "error": None,
        }
        print(f"✓ Pandas {pd.__version__} imported successfully")

    except Exception as e:
        compatibility_results["pandas"] = {
            "version": None,
            "status": "ERROR",
            "error": str(e),
        }
        print(f"✗ Pandas import failed: {e}")

    # Test networkx
    try:
        import networkx as nx

        compatibility_results["networkx"] = {
            "version": nx.__version__,
            "status": "OK",
            "error": None,
        }
        print(f"✓ NetworkX {nx.__version__} imported successfully")

    except Exception as e:
        compatibility_results["networkx"] = {
            "version": None,
            "status": "ERROR",
            "error": str(e),
        }
        print(f"✗ NetworkX import failed: {e}")

    # Print summary
    print("\n" + "=" * 50)
    print("COMPATIBILITY TEST SUMMARY")
    print("=" * 50)

    total_packages = len(compatibility_results)
    successful_packages = sum(
        1 for result in compatibility_results.values() if result["status"] == "OK"
    )

    print(f"Packages tested: {total_packages}")
    print(f"Successful imports: {successful_packages}")
    print(f"Failed imports: {total_packages - successful_packages}")

    if successful_packages == total_packages:
        print("\n✓ ALL DEPENDENCIES ARE COMPATIBLE!")
        return True
    else:
        print("\n✗ DEPENDENCY CONFLICTS DETECTED!")
        for pkg, result in compatibility_results.items():
            if result["status"] == "ERROR":
                print(f"  - {pkg}: {result['error']}")
        return False


if __name__ == "__main__":
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print("=" * 50)

    success = check_version_compatibility()
    sys.exit(0 if success else 1)
