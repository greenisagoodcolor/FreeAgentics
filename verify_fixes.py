#!/usr/bin/env python3
"""
Verify the three critical fixes work in a clean environment.
This script tests that FreeAgentics can run without:
1. Database (PostgreSQL)
2. PyMDP compatibility issues
3. LLM API keys
"""

import os
import sys
import subprocess

def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Command: {cmd}")
    print('='*60)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ SUCCESS: {description}")
        if result.stdout:
            print("Output:", result.stdout[:200], "..." if len(result.stdout) > 200 else "")
    else:
        print(f"❌ FAILED: {description}")
        print("Error:", result.stderr[:500])
    
    return result.returncode == 0

def main():
    print("🔍 VERIFYING CRITICAL FIXES")
    print("="*70)
    
    # Ensure we're in the right directory
    if not os.path.exists("pyproject.toml"):
        print("❌ Not in FreeAgentics directory!")
        return 1
    
    # Test 1: Run without DATABASE_URL
    print("\n1. Testing without database...")
    os.environ.pop('DATABASE_URL', None)
    
    success1 = run_command(
        "python -c 'from database.session import get_database_url; print(get_database_url() or \"No DB - OK!\")'",
        "Database optional check"
    )
    
    # Test 2: Import and use PyMDP
    print("\n2. Testing PyMDP compatibility...")
    
    test_pymdp = """
import numpy as np
from agents.pymdp_adapter import PyMDPCompatibilityAdapter

# Create adapter
adapter = PyMDPCompatibilityAdapter()

# Create simple model
model = {
    'A': np.ones((3, 3)) / 3,
    'B': np.eye(3).reshape(3, 3, 1).repeat(3, axis=2),
    'C': np.array([0., 0., 1.]),
    'D': np.ones(3) / 3
}

# Create agent (should not fail with control_fns error)
agent = adapter.create_agent(model, agent_id='test')
print(f"Agent created: {agent is not None}")
print(f"Has F attribute: {hasattr(agent, 'F')}")
"""
    
    success2 = run_command(
        f"python -c '{test_pymdp}'",
        "PyMDP agent creation"
    )
    
    # Test 3: Use mock LLM without API key
    print("\n3. Testing mock LLM provider...")
    
    # Remove any API keys
    for key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY']:
        os.environ.pop(key, None)
    
    test_llm = """
from llm.factory import LLMProviderFactory

# Should default to mock
provider = LLMProviderFactory.create_provider()
print(f"Provider type: {type(provider).__name__}")

# Generate GMN
import asyncio
gmn = asyncio.run(provider.generate_gmn('Create an explorer agent'))
print(f"Generated GMN length: {len(gmn)}")
print("GMN preview:", gmn[:100], "...")
"""
    
    success3 = run_command(
        f"python -c '{test_llm}'",
        "Mock LLM provider"
    )
    
    # Test 4: Full demo flow
    print("\n4. Testing full demo flow...")
    
    success4 = run_command(
        "python demo_test.py",
        "Complete demo without dependencies"
    )
    
    # Summary
    print("\n" + "="*70)
    print("📊 VERIFICATION SUMMARY")
    print("="*70)
    
    all_success = all([success1, success2, success3, success4])
    
    print(f"\n{'✅' if success1 else '❌'} Database optional: {'PASS' if success1 else 'FAIL'}")
    print(f"{'✅' if success2 else '❌'} PyMDP compatibility: {'PASS' if success2 else 'FAIL'}")
    print(f"{'✅' if success3 else '❌'} Mock LLM provider: {'PASS' if success3 else 'FAIL'}")
    print(f"{'✅' if success4 else '❌'} Full demo flow: {'PASS' if success4 else 'FAIL'}")
    
    if all_success:
        print("\n🎉 ALL CRITICAL FIXES VERIFIED!")
        print("The system can now run in demo mode without any external dependencies.")
    else:
        print("\n⚠️ Some fixes still need work.")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())