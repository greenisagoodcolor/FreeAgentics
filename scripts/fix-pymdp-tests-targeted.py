#!/usr/bin/env python3
"""
Targeted fix for PyMDP test issues.
Specifically fixes BasicExplorerAgent constructor calls.
"""

import os
import re


def fix_test_file(filepath):
    """Fix BasicExplorerAgent constructor calls in test file."""
    print(f"Fixing: {filepath}")

    with open(filepath, "r") as f:
        content = f.read()

    # Count fixes
    fixes = 0

    # Pattern 1: BasicExplorerAgent with position tuple as second argument
    # BasicExplorerAgent("id", (0, 0), ...) -> BasicExplorerAgent("id", "id_agent", ...)
    def fix_constructor_with_position(match):
        nonlocal fixes
        fixes += 1
        agent_id = match.group(1)
        # Extract just the ID part for the name
        name = agent_id.strip('"').strip("'") + "_agent"
        rest = match.group(3)
        return f'BasicExplorerAgent({agent_id}, "{name}"{rest}'

    # Fix patterns like: BasicExplorerAgent("id", (x, y), ...)
    content = re.sub(
        r'BasicExplorerAgent\((["\'][^"\']+["\'])\s*,\s*\([0-9]+\s*,\s*[0-9]+\)\s*(.*?)\)',
        fix_constructor_with_position,
        content,
    )

    # Pattern 2: Also fix cases with variables as IDs
    def fix_constructor_with_var(match):
        nonlocal fixes
        fixes += 1
        agent_id = match.group(1)
        rest = match.group(3)
        return f'BasicExplorerAgent({agent_id}, f"{{{agent_id}}}_agent"{rest}'

    # Fix patterns like: BasicExplorerAgent(var, (x, y), ...)
    content = re.sub(
        r"BasicExplorerAgent\(([a-zA-Z_][a-zA-Z0-9_]*)\s*,\s*\([0-9]+\s*,\s*[0-9]+\)\s*(.*?)\)",
        fix_constructor_with_var,
        content,
    )

    # Pattern 3: Fix cases where position is passed as third argument with grid_size
    # BasicExplorerAgent("id", (x, y), grid_size=n) -> BasicExplorerAgent("id", "id_agent", grid_size=n)
    content = re.sub(
        r'BasicExplorerAgent\((["\'][^"\']+["\'])\s*,\s*\([0-9]+\s*,\s*[0-9]+\)\s*,\s*grid_size\s*=\s*([0-9]+)\)',
        lambda m: f'BasicExplorerAgent({m.group(1)}, {m.group(1).strip(chr(39)).strip(chr(34)) + "_agent"!r}, grid_size={m.group(2)})',
        content,
    )

    # Write back if changes were made
    if fixes > 0:
        with open(filepath, "w") as f:
            f.write(content)
        print(f"  ✓ Fixed {fixes} BasicExplorerAgent constructor calls")
        return True
    else:
        print(f"  - No fixes needed")
        return False


def main():
    """Main function to fix PyMDP tests."""
    test_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "integration"
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
