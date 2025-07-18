#!/usr/bin/env python3
"""
Simple targeted fix for PyMDP test BasicExplorerAgent constructor issues.
"""

import os


def fix_test_file(filepath):
    """Fix BasicExplorerAgent constructor calls in test file."""
    print(f"Fixing: {filepath}")

    with open(filepath, "r") as f:
        content = f.read()

    original = content

    # Replace specific known patterns
    replacements = [
        # Pattern: BasicExplorerAgent("id", (x, y), grid_size=n)
        (
            'BasicExplorerAgent("nemesis_agent_1", position=(0, 0),',
            'BasicExplorerAgent("nemesis_agent_1", "nemesis_agent",',
        ),
        (
            'BasicExplorerAgent(\n            agent_id="nemesis_agent_1",\n            position=(0, 0),',
            'BasicExplorerAgent(\n            agent_id="nemesis_agent_1",\n            name="nemesis_agent",',
        ),
        (
            'BasicExplorerAgent(\n            agent_id="action_type_test",\n            position=(0, 0),',
            'BasicExplorerAgent(\n            agent_id="action_type_test",\n            name="action_type_test_agent",',
        ),
        (
            'BasicExplorerAgent(\n            agent_id="math_validation",\n            position=(1, 1),',
            'BasicExplorerAgent(\n            agent_id="math_validation",\n            name="math_validation_agent",',
        ),
        (
            'BasicExplorerAgent("perf_test", (0, 0), grid_size=3)',
            'BasicExplorerAgent("perf_test", "perf_test_agent", grid_size=3)',
        ),
        (
            'BasicExplorerAgent("edge_case_test", (0, 0))',
            'BasicExplorerAgent("edge_case_test", "edge_case_test_agent")',
        ),
        (
            'BasicExplorerAgent("consistency_test", (2, 2), grid_size=5)',
            'BasicExplorerAgent("consistency_test", "consistency_test_agent", grid_size=5)',
        ),
        (
            'BasicExplorerAgent("nemesis_1", (0, 0), grid_size=3)',
            'BasicExplorerAgent("nemesis_1", "nemesis_1_agent", grid_size=3)',
        ),
        (
            'BasicExplorerAgent("nemesis_2", (2, 2), grid_size=5)',
            'BasicExplorerAgent("nemesis_2", "nemesis_2_agent", grid_size=5)',
        ),
        (
            'BasicExplorerAgent("nemesis_3", (1, 1), grid_size=4)',
            'BasicExplorerAgent("nemesis_3", "nemesis_3_agent", grid_size=4)',
        ),
        # For test_action_sampling_issue.py
        (
            'agent = BasicExplorerAgent("test_agent", position=[5, 5])',
            'agent = BasicExplorerAgent("test_agent", "test_agent")',
        ),
        (
            'BasicExplorerAgent("numpy_test", position=[0, 0])',
            'BasicExplorerAgent("numpy_test", "numpy_test_agent")',
        ),
    ]

    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
            print(f"  ✓ Fixed: {old[:50]}...")

    # Write back if changes were made
    if content != original:
        with open(filepath, "w") as f:
            f.write(content)
        return True
    else:
        print(f"  - No changes needed")
        return False


def main():
    """Main function to fix PyMDP tests."""
    test_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "tests",
        "integration",
    )

    pymdp_test_files = [
        "test_pymdp_validation.py",
        "test_action_sampling_issue.py",
        "test_nemesis_pymdp_validation.py",
        "test_pymdp_hard_failure_integration.py",
    ]

    total_fixed = 0

    for test_file in pymdp_test_files:
        filepath = os.path.join(test_dir, test_file)
        if os.path.exists(filepath):
            if fix_test_file(filepath):
                total_fixed += 1

    print(f"\nTotal files fixed: {total_fixed}")


if __name__ == "__main__":
    main()
