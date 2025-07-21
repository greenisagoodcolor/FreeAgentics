#!/usr/bin/env python3
"""
Report on performance theater and graceful degradation patterns in the codebase.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

# Patterns to find
PERFORMANCE_THEATER_PATTERNS = [
    # Sleep patterns
    (r"time\.sleep\s*\([^)]+\)", "time.sleep() call"),
    (r"await\s+asyncio\.sleep\s*\(0\.\d+\)", "asyncio.sleep() with small delay"),
    # Progress bars
    (r"from\s+tqdm\s+import", "tqdm import"),
    (r"tqdm\s*\([^)]+\)", "tqdm usage"),
    # Mock/Dummy responses
    (r"return\s+.*mock", "return mock data"),
    (r"return\s+.*dummy", "return dummy data"),
    (r"return\s+.*fake", "return fake data"),
    # Graceful degradation patterns
    (r"fallback_func\s*=", "fallback function"),
    (r"graceful.*degradation", "graceful degradation"),
    (r"safe_execute.*fallback", "safe_execute with fallback"),
]

# Try/except patterns that need conversion
TRY_EXCEPT_PATTERNS = [
    # PyMDP-specific error handling
    (r"try:.*pymdp.*except.*return", "try/except with PyMDP returning fallback"),
    (r"except.*:.*return\s+None", "except returning None"),
    (r"except.*:.*return\s+\[\]", "except returning empty list"),
    (r"except.*:.*return\s+\{\}", "except returning empty dict"),
]

# Files to exclude
EXCLUDE_FILES = {
    "convert_to_hard_failures.py",
    "report_performance_theater.py",
    "test_pymdp_hard_failure_integration.py",
    "hard_failure_handlers.py",
    "pymdp_adapter.py",
}

# Directories to search
SEARCH_DIRS = ["agents", "inference", "api", "database"]


def find_files(base_path: Path) -> List[Path]:
    """Find all Python files to check."""
    files = []
    for dir_name in SEARCH_DIRS:
        dir_path = base_path / dir_name
        if dir_path.exists():
            for file_path in dir_path.rglob("*.py"):
                if file_path.name not in EXCLUDE_FILES:
                    files.append(file_path)
    return files


def check_file(file_path: Path) -> Dict[str, List[Tuple[int, str]]]:
    """Check file for patterns."""
    issues = {"performance_theater": [], "graceful_degradation": [], "try_except_fallbacks": []}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.split("\n")

        # Check each line
        for i, line in enumerate(lines, 1):
            # Performance theater patterns
            for pattern, description in PERFORMANCE_THEATER_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    if "graceful" in description or "fallback" in description:
                        issues["graceful_degradation"].append((i, line.strip()))
                    else:
                        issues["performance_theater"].append((i, line.strip()))

            # Try/except patterns
            for pattern, description in TRY_EXCEPT_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    issues["try_except_fallbacks"].append((i, line.strip()))

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return issues


def main():
    """Generate report."""
    base_path = Path("/home/green/FreeAgentics")

    print("# Performance Theater and Graceful Degradation Report")
    print("=" * 60)
    print("\nSearching for patterns that need to be converted to hard failures...")

    # Find files
    files = find_files(base_path)
    print(f"\nChecking {len(files)} Python files...")

    # Categorize issues
    all_issues = {"performance_theater": {}, "graceful_degradation": {}, "try_except_fallbacks": {}}

    for file_path in files:
        issues = check_file(file_path)
        for category, file_issues in issues.items():
            if file_issues:
                rel_path = file_path.relative_to(base_path)
                all_issues[category][str(rel_path)] = file_issues

    # Print summary
    print("\n## Summary")
    print(
        f"- Performance Theater: {sum(len(f) for f in all_issues['performance_theater'].values())} issues in {len(all_issues['performance_theater'])} files"
    )
    print(
        f"- Graceful Degradation: {sum(len(f) for f in all_issues['graceful_degradation'].values())} issues in {len(all_issues['graceful_degradation'])} files"
    )
    print(
        f"- Try/Except Fallbacks: {sum(len(f) for f in all_issues['try_except_fallbacks'].values())} issues in {len(all_issues['try_except_fallbacks'])} files"
    )

    # Detailed report
    print("\n## Performance Theater Issues")
    print("These are fake delays, progress bars, and mock responses:")
    for file_path, issues in sorted(all_issues["performance_theater"].items()):
        print(f"\n### {file_path}")
        for line_num, code in issues[:5]:  # Show first 5
            print(
                f"- Line {line_num}: `{code[:80]}...`"
                if len(code) > 80
                else f"- Line {line_num}: `{code}`"
            )
        if len(issues) > 5:
            print(f"- ... and {len(issues) - 5} more issues")

    print("\n## Graceful Degradation Issues")
    print("These patterns hide real errors:")
    for file_path, issues in sorted(all_issues["graceful_degradation"].items()):
        print(f"\n### {file_path}")
        for line_num, code in issues[:5]:
            print(
                f"- Line {line_num}: `{code[:80]}...`"
                if len(code) > 80
                else f"- Line {line_num}: `{code}`"
            )
        if len(issues) > 5:
            print(f"- ... and {len(issues) - 5} more issues")

    print("\n## Try/Except Fallback Issues")
    print("These need to be converted to raise exceptions:")
    for file_path, issues in sorted(all_issues["try_except_fallbacks"].items()):
        print(f"\n### {file_path}")
        for line_num, code in issues[:5]:
            print(
                f"- Line {line_num}: `{code[:80]}...`"
                if len(code) > 80
                else f"- Line {line_num}: `{code}`"
            )
        if len(issues) > 5:
            print(f"- ... and {len(issues) - 5} more issues")

    print("\n## Recommendations")
    print("1. Convert all try/except blocks that return None/empty to raise exceptions")
    print("2. Remove all time.sleep() calls - replace with real computation if needed")
    print("3. Remove fallback_func parameters from safe_execute calls")
    print("4. Replace graceful degradation with hard failures")
    print("5. Remove any mock/dummy/fake return values")
    print("\n## Next Steps")
    print("1. Review agents/pymdp_error_handling.py - contains main graceful degradation logic")
    print("2. Update agents/fallback_handlers.py usage to hard_failure_handlers.py")
    print("3. Run tests: pytest tests/integration/test_pymdp_hard_failure_integration.py")
    print("4. Document all changes in AGENTLESSONS.md")


if __name__ == "__main__":
    main()
