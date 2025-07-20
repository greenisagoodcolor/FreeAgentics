#!/usr/bin/env python3
"""
Run all working tests to establish baseline coverage.
Following Michael Feathers' approach - document what exists before improving.
"""

import subprocess
import sys
import json
from pathlib import Path

def run_tests_for_module(test_pattern, module_name):
    """Run tests for a specific module and return coverage."""
    cmd = [
        sys.executable, "-m", "pytest",
        f"tests/unit/{test_pattern}",
        "-v",
        f"--cov={module_name}",
        "--cov-report=json",
        "--cov-report=term",
        "--tb=no",
        "-q"
    ]
    
    print(f"\n{'='*60}")
    print(f"Running tests for {module_name}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        # Try to read coverage data
        coverage_file = Path("coverage.json")
        if coverage_file.exists():
            with open(coverage_file) as f:
                coverage_data = json.load(f)
                total = coverage_data.get("totals", {})
                percent = total.get("percent_covered", 0)
                print(f"\nCoverage for {module_name}: {percent:.2f}%")
                return percent
    except Exception as e:
        print(f"Error running tests for {module_name}: {e}")
        return 0
    
    return 0

def main():
    """Run tests for all critical modules."""
    
    test_configs = [
        # Auth module tests (avoiding JWT import issues)
        ("test_auth_init_coverage.py", "auth"),
        ("test_password_security.py", "auth"),
        ("test_mfa_service.py", "auth.mfa_service"),
        ("test_rbac_*.py", "auth"),
        
        # Agent module tests
        ("test_base_agent.py", "agents.base_agent"),
        ("test_agent_manager_simple.py", "agents.agent_manager"),
        ("test_agent_memory_*.py", "agents"),
        ("test_error_handling.py", "agents.error_handling"),
        
        # Database module tests
        ("test_database_base.py", "database.base"),
        ("test_database_models.py", "database.models"),
        ("test_database_session*.py", "database.session"),
        
        # API module tests
        ("test_api_health*.py", "api.v1.health"),
        ("test_api_main*.py", "api.main"),
    ]
    
    results = {}
    
    for test_pattern, module in test_configs:
        coverage = run_tests_for_module(test_pattern, module)
        results[module] = coverage
    
    # Summary
    print(f"\n{'='*60}")
    print("COVERAGE SUMMARY")
    print(f"{'='*60}")
    
    for module, coverage in sorted(results.items()):
        status = "✓" if coverage >= 80 else "✗"
        print(f"{status} {module:40} {coverage:6.2f}%")
    
    # Calculate weighted average
    total_coverage = sum(results.values())
    avg_coverage = total_coverage / len(results) if results else 0
    
    print(f"\nAverage Coverage: {avg_coverage:.2f}%")
    print(f"Target: 80.00%")
    print(f"Gap: {80.0 - avg_coverage:.2f}%")

if __name__ == "__main__":
    main()