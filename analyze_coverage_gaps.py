#!/usr/bin/env python3
"""Analyze coverage gaps and identify priority modules for improvement."""

import json
import sys
from typing import Dict, List, Tuple


def load_coverage_data(coverage_file: str = "coverage.json") -> Dict:
    """Load coverage data from JSON file."""
    try:
        with open(coverage_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Coverage file '{coverage_file}' not found.")
        print("Please run coverage tests first.")
        sys.exit(1)


def analyze_coverage(data: Dict) -> List[Tuple[str, float, int, int]]:
    """Analyze coverage data and return list of (module, coverage%, statements, missing)."""
    results = []

    for module, info in data.get("files", {}).items():
        summary = info.get("summary", {})
        coverage = summary.get("percent_covered", 0.0)
        statements = summary.get("num_statements", 0)
        missing = summary.get("missing_lines", 0)

        # Skip empty files and test files
        if statements > 0 and not module.startswith("tests/"):
            results.append((module, coverage, statements, missing))

    # Sort by coverage percentage (ascending) and then by statements (descending)
    results.sort(key=lambda x: (x[1], -x[2]))

    return results


def categorize_modules(results: List[Tuple[str, float, int, int]]) -> Dict[str, List]:
    """Categorize modules by coverage level."""
    categories = {
        "critical": [],  # 0-20% coverage
        "high": [],  # 20-50% coverage
        "medium": [],  # 50-80% coverage
        "good": [],  # 80%+ coverage
    }

    for module, coverage, statements, missing in results:
        if coverage < 20:
            categories["critical"].append((module, coverage, statements, missing))
        elif coverage < 50:
            categories["high"].append((module, coverage, statements, missing))
        elif coverage < 80:
            categories["medium"].append((module, coverage, statements, missing))
        else:
            categories["good"].append((module, coverage, statements, missing))

    return categories


def print_report(categories: Dict[str, List], totals: Dict):
    """Print coverage analysis report."""
    print("=" * 80)
    print("COVERAGE ANALYSIS REPORT - Michael Feathers Style")
    print("=" * 80)
    print()

    # Overall statistics
    total_coverage = totals.get("percent_covered", 0.0)
    total_statements = totals.get("num_statements", 0)
    total_missing = totals.get("missing_lines", 0)

    print(f"OVERALL COVERAGE: {total_coverage:.2f}%")
    print(f"Total Statements: {total_statements:,}")
    print(f"Missing Lines: {total_missing:,}")
    print(f"Target: 80.0% (Gap: {80.0 - total_coverage:.2f}%)")
    print()

    # Critical modules (need immediate attention)
    print("CRITICAL PRIORITY (0-20% coverage) - IMMEDIATE ACTION REQUIRED:")
    print("-" * 80)
    if categories["critical"]:
        for module, coverage, statements, missing in categories["critical"][:20]:
            print(f"{module:<50} {coverage:>6.2f}% ({statements:>4} stmts, {missing:>4} missing)")
    else:
        print("None - Great job!")
    print()

    # High priority modules
    print("HIGH PRIORITY (20-50% coverage):")
    print("-" * 80)
    if categories["high"]:
        for module, coverage, statements, missing in categories["high"][:10]:
            print(f"{module:<50} {coverage:>6.2f}% ({statements:>4} stmts, {missing:>4} missing)")
    else:
        print("None")
    print()

    # Summary
    print("SUMMARY BY CATEGORY:")
    print("-" * 80)
    print(f"Critical (<20%):  {len(categories['critical'])} modules")
    print(f"High (20-50%):    {len(categories['high'])} modules")
    print(f"Medium (50-80%):  {len(categories['medium'])} modules")
    print(f"Good (80%+):      {len(categories['good'])} modules")
    print()

    # Action plan
    print("ACTION PLAN (Following 'Working Effectively with Legacy Code' principles):")
    print("-" * 80)
    print("1. Focus on CRITICAL modules first - these have the highest risk")
    print("2. Write characterization tests to document existing behavior")
    print("3. Add tests for bug fixes and new features")
    print("4. NO coverage exclusions or omit directives allowed")
    print("5. Aim for 80%+ coverage on all production code")
    print()

    # Top modules to improve
    print("TOP 10 MODULES TO IMPROVE (by impact):")
    print("-" * 80)
    all_low_coverage = categories["critical"] + categories["high"]
    # Sort by potential impact (missing lines * importance)
    all_low_coverage.sort(key=lambda x: x[3], reverse=True)

    for i, (module, coverage, statements, missing) in enumerate(all_low_coverage[:10], 1):
        print(f"{i:2}. {module:<45} ({missing:>4} lines to cover)")


def main():
    """Main function."""
    # Load coverage data
    data = load_coverage_data()

    # Analyze coverage
    results = analyze_coverage(data)

    # Categorize modules
    categories = categorize_modules(results)

    # Get totals
    totals = data.get("totals", {})

    # Print report
    print_report(categories, totals)


if __name__ == "__main__":
    main()
