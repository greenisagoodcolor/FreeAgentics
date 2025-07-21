#!/usr/bin/env python3
"""
Coverage Gap Analysis Tool
Generates comprehensive reports on test coverage gaps with prioritization
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


class CoverageAnalyzer:
    """Analyzes coverage data and identifies gaps with prioritization."""

    # Critical modules that require high coverage
    CRITICAL_MODULES = {
        "agents/base_agent.py",
        "agents/pymdp_adapter.py",
        "agents/coalition_coordinator.py",
        "inference/active/gmn_parser.py",
        "inference/active/pymdp_integration.py",
        "inference/gnn/model.py",
        "api/v1/auth.py",
        "auth/security_implementation.py",
        "database/models.py",
        "coalitions/coalition_manager.py",
    }

    # Module categories for grouping
    MODULE_CATEGORIES = {
        "core": ["agents/", "inference/", "coalitions/"],
        "api": ["api/"],
        "auth": ["auth/"],
        "data": ["database/", "knowledge_graph/"],
        "infrastructure": ["config/", "observability/", "world/"],
    }

    def __init__(self, coverage_file: Path):
        self.coverage_file = coverage_file
        self.data = self._load_coverage_data()

    def _load_coverage_data(self) -> Dict:
        """Load coverage data from JSON file."""
        try:
            with open(self.coverage_file) as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading coverage file: {e}")
            sys.exit(1)

    def _categorize_module(self, module_path: str) -> str:
        """Categorize a module based on its path."""
        for category, patterns in self.MODULE_CATEGORIES.items():
            if any(module_path.startswith(pattern) for pattern in patterns):
                return category
        return "other"

    def _is_critical(self, module_path: str) -> bool:
        """Check if module is critical."""
        return any(module_path.endswith(critical) for critical in self.CRITICAL_MODULES)

    def analyze_gaps(self) -> Dict[str, Any]:
        """Analyze coverage gaps and categorize them."""
        gaps: Dict[str, Any] = {
            "summary": {},
            "critical_gaps": [],
            "zero_coverage": [],
            "low_coverage": [],
            "by_category": defaultdict(list),
            "gnn_specific": [],
            "recommendations": [],
        }

        total_coverage = self.data["totals"]["percent_covered"]
        gaps["summary"] = {
            "total_coverage": total_coverage,
            "total_lines": self.data["totals"]["num_statements"],
            "covered_lines": self.data["totals"]["covered_lines"],
            "missing_lines": self.data["totals"]["missing_lines"],
            "total_files": len(self.data["files"]),
        }

        # Analyze each file
        for file_path, file_data in self.data["files"].items():
            coverage_pct = file_data["summary"]["percent_covered"]
            missing_lines = file_data["missing_lines"]
            excluded_lines = file_data["excluded_lines"]

            module_info = {
                "path": file_path,
                "coverage": coverage_pct,
                "missing_lines": len(missing_lines),
                "missing_line_numbers": missing_lines[:20],  # First 20 lines
                "excluded_lines": len(excluded_lines),
                "category": self._categorize_module(file_path),
                "is_critical": self._is_critical(file_path),
            }

            # Categorize by coverage level
            if coverage_pct == 0:
                gaps["zero_coverage"].append(module_info)
            elif coverage_pct < 50:
                gaps["low_coverage"].append(module_info)

            # Check for critical modules
            if module_info["is_critical"] and coverage_pct < 80:
                gaps["critical_gaps"].append(module_info)

            # Check for GNN modules (mentioned in tasks)
            if "gnn" in file_path.lower() or "graph" in file_path.lower():
                gaps["gnn_specific"].append(module_info)

            # Group by category
            gaps["by_category"][module_info["category"]].append(module_info)

        # Sort lists by coverage (lowest first)
        for key in [
            "zero_coverage",
            "low_coverage",
            "critical_gaps",
            "gnn_specific",
        ]:
            gaps[key].sort(key=lambda x: x["coverage"])

        # Generate recommendations
        gaps["recommendations"] = self._generate_recommendations(gaps)

        return gaps

    def _generate_recommendations(self, gaps: Dict) -> List[str]:
        """Generate actionable recommendations based on gaps."""
        recommendations = []

        # Critical module recommendations
        if gaps["critical_gaps"]:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "Critical Modules",
                    "action": f"Focus on {len(gaps['critical_gaps'])} critical modules with <80% coverage",
                    "modules": [m["path"] for m in gaps["critical_gaps"][:5]],
                }
            )

        # Zero coverage modules
        if len(gaps["zero_coverage"]) > 10:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "Zero Coverage",
                    "action": f"Address {len(gaps['zero_coverage'])} modules with 0% coverage",
                    "modules": [m["path"] for m in gaps["zero_coverage"][:5]],
                }
            )

        # GNN modules (from task requirements)
        gnn_gaps = [m for m in gaps["gnn_specific"] if m["coverage"] < 50]
        if gnn_gaps:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "GNN Modules",
                    "action": f"Improve coverage for {len(gnn_gaps)} GNN/Graph modules",
                    "modules": [m["path"] for m in gnn_gaps[:5]],
                }
            )

        # Category-specific recommendations
        for category, modules in gaps["by_category"].items():
            low_coverage = [m for m in modules if m["coverage"] < 50]
            if len(low_coverage) > 5:
                recommendations.append(
                    {
                        "priority": "MEDIUM",
                        "category": f"{category.title()} Layer",
                        "action": f"Improve {len(low_coverage)} modules in {category} layer",
                        "average_coverage": sum(m["coverage"] for m in modules) / len(modules),
                    }
                )

        return recommendations

    def generate_report(self, output_format: str = "text") -> str:
        """Generate coverage gap report in specified format."""
        gaps = self.analyze_gaps()

        if output_format == "json":
            return json.dumps(gaps, indent=2)
        elif output_format == "markdown":
            return self._generate_markdown_report(gaps)
        else:
            return self._generate_text_report(gaps)

    def _generate_text_report(self, gaps: Dict) -> str:
        """Generate human-readable text report."""
        lines = []
        lines.append("=" * 80)
        lines.append("COVERAGE GAP ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Summary
        summary = gaps["summary"]
        lines.append("SUMMARY:")
        lines.append(f"  Total Coverage: {summary['total_coverage']:.2f}%")
        lines.append(f"  Total Lines: {summary['total_lines']:,}")
        lines.append(f"  Covered Lines: {summary['covered_lines']:,}")
        lines.append(f"  Missing Lines: {summary['missing_lines']:,}")
        lines.append(f"  Total Files: {summary['total_files']}")
        lines.append("")

        # Critical gaps
        if gaps["critical_gaps"]:
            lines.append("CRITICAL MODULES WITH LOW COVERAGE:")
            for module in gaps["critical_gaps"][:10]:
                lines.append(
                    f"  - {module['path']}: {module['coverage']:.1f}% ({module['missing_lines']} lines missing)"
                )
            lines.append("")

        # Zero coverage
        if gaps["zero_coverage"]:
            lines.append(f"MODULES WITH 0% COVERAGE ({len(gaps['zero_coverage'])} total):")
            for module in gaps["zero_coverage"][:10]:
                lines.append(f"  - {module['path']}")
            if len(gaps["zero_coverage"]) > 10:
                lines.append(f"  ... and {len(gaps['zero_coverage']) - 10} more")
            lines.append("")

        # GNN specific
        if gaps["gnn_specific"]:
            lines.append("GNN/GRAPH MODULES COVERAGE:")
            for module in gaps["gnn_specific"]:
                lines.append(f"  - {module['path']}: {module['coverage']:.1f}%")
            lines.append("")

        # Recommendations
        lines.append("RECOMMENDATIONS:")
        for i, rec in enumerate(gaps["recommendations"], 1):
            lines.append(f"{i}. [{rec['priority']}] {rec['category']}: {rec['action']}")
            if "modules" in rec:
                for module in rec["modules"][:3]:
                    lines.append(f"     - {module}")
        lines.append("")

        return "\n".join(lines)

    def _generate_markdown_report(self, gaps: Dict) -> str:
        """Generate markdown report."""
        lines = []
        lines.append("# Coverage Gap Analysis Report")
        lines.append("")

        # Summary
        summary = gaps["summary"]
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Total Coverage**: {summary['total_coverage']:.2f}%")
        lines.append(f"- **Total Lines**: {summary['total_lines']:,}")
        lines.append(f"- **Covered Lines**: {summary['covered_lines']:,}")
        lines.append(f"- **Missing Lines**: {summary['missing_lines']:,}")
        lines.append(f"- **Total Files**: {summary['total_files']}")
        lines.append("")

        # Critical gaps
        if gaps["critical_gaps"]:
            lines.append("## Critical Modules with Low Coverage")
            lines.append("")
            lines.append("| Module | Coverage | Missing Lines |")
            lines.append("|--------|----------|---------------|")
            for module in gaps["critical_gaps"][:10]:
                lines.append(
                    f"| {module['path']} | {module['coverage']:.1f}% | {module['missing_lines']} |"
                )
            lines.append("")

        # Zero coverage
        if gaps["zero_coverage"]:
            lines.append(f"## Modules with 0% Coverage ({len(gaps['zero_coverage'])} total)")
            lines.append("")
            for module in gaps["zero_coverage"][:20]:
                lines.append(f"- `{module['path']}`")
            if len(gaps["zero_coverage"]) > 20:
                lines.append(f"- ... and {len(gaps['zero_coverage']) - 20} more")
            lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        for i, rec in enumerate(gaps["recommendations"], 1):
            lines.append(f"### {i}. {rec['category']} ({rec['priority']} Priority)")
            lines.append(f"{rec['action']}")
            if "modules" in rec:
                lines.append("")
                lines.append("Key modules to focus on:")
                for module in rec["modules"][:5]:
                    lines.append(f"- `{module}`")
            lines.append("")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze test coverage gaps")
    parser.add_argument(
        "--coverage-file",
        default="coverage.json",
        help="Path to coverage.json file",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format",
    )
    parser.add_argument("--output", help="Output file (default: stdout)")

    args = parser.parse_args()

    # Check if coverage file exists
    coverage_path = Path(args.coverage_file)
    if not coverage_path.exists():
        print(f"Error: Coverage file not found: {coverage_path}")
        print("Run coverage report first with: ./scripts/coverage-dev.sh")
        sys.exit(1)

    # Analyze coverage
    analyzer = CoverageAnalyzer(coverage_path)
    report = analyzer.generate_report(args.format)

    # Output report
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to: {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
