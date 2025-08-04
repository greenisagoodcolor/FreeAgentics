#!/usr/bin/env python3
"""
Coverage checking and reporting with module-specific thresholds.

This script implements the Nemesis Committee's progressive coverage strategy:
- Module-specific thresholds based on criticality
- Coverage debt tracking for zero-coverage modules
- Actionable reporting with next steps
- Performance-aware collection with budgets

Usage:
    python scripts/coverage-check.py [--profile=dev|ci|release] [--module=path]
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import subprocess
import time
import argparse


@dataclass
class ModuleThreshold:
    """Coverage threshold configuration for a module."""
    path: str
    min_coverage: float
    target_coverage: float
    criticality: str  # critical, business, infrastructure, debt
    improvement_target: Optional[str] = None  # e.g., "40% by 2025-09"


@dataclass
class CoverageResult:
    """Coverage results for a module."""
    module: str
    covered_lines: int
    total_lines: int
    coverage_percent: float
    missing_lines: List[int]
    collection_time: float
    branch_coverage: Optional[float] = None


@dataclass
class CoverageReport:
    """Complete coverage analysis report."""
    timestamp: str
    profile: str
    total_coverage: float
    module_results: List[CoverageResult]
    threshold_violations: List[str]
    coverage_debt: Dict[str, str]
    performance_metrics: Dict[str, float]
    next_actions: List[str]


class CoverageChecker:
    """Module-specific coverage validation with actionable reporting."""
    
    # Module threshold configuration based on Nemesis Committee recommendations
    MODULE_THRESHOLDS = [
        # Critical path modules - highest coverage required
        ModuleThreshold("api/", 90.0, 95.0, "critical"),
        ModuleThreshold("auth/", 90.0, 95.0, "critical"),
        ModuleThreshold("database/", 90.0, 95.0, "critical"),
        ModuleThreshold("agents/", 85.0, 90.0, "critical"),
        
        # Business logic - high coverage required
        ModuleThreshold("coalitions/", 80.0, 85.0, "business"),
        ModuleThreshold("services/", 80.0, 85.0, "business"),
        ModuleThreshold("knowledge_graph/", 75.0, 80.0, "business"),
        
        # Infrastructure - moderate coverage required
        ModuleThreshold("inference/", 60.0, 75.0, "infrastructure"),
        ModuleThreshold("llm/", 60.0, 75.0, "infrastructure"),
        ModuleThreshold("world/", 60.0, 75.0, "infrastructure"),
        ModuleThreshold("websocket/", 65.0, 80.0, "infrastructure"),
        
        # Coverage debt modules - improvement tracking
        ModuleThreshold("inference/gnn/", 0.0, 40.0, "debt", "40% by 2025-09"),
        ModuleThreshold("observability/", 0.0, 45.0, "debt", "45% by 2025-08"),
        ModuleThreshold("security/", 0.0, 70.0, "debt", "70% by 2025-09"),
    ]
    
    PERFORMANCE_BUDGETS = {
        "collection_time_per_module": 30.0,  # seconds
        "total_collection_time": 300.0,      # seconds
        "memory_usage_mb": 512.0,            # MB
    }
    
    def __init__(self, profile: str = "default"):
        self.profile = profile
        self.root_path = Path.cwd()
        self.coverage_data = {}
        self.start_time = time.time()
        
    def run_coverage_collection(self, module_path: Optional[str] = None) -> Dict:
        """Run coverage collection with performance monitoring."""
        print(f"🔍 Running coverage collection (profile: {self.profile})")
        
        # Select coverage configuration based on profile
        config_file = self._get_config_file()
        
        cmd = [
            "python", "-m", "pytest",
            f"--cov-config={config_file}",
            "--cov=.",
            "--cov-report=json:coverage-data.json",
            "--cov-report=term-missing",
            "--tb=no",
            "-q"
        ]
        
        if module_path:
            # Filter tests to specific module for faster feedback
            test_path = f"tests/unit/test_{module_path.replace('/', '_')}"
            if Path(test_path).exists():
                cmd.append(test_path)
        
        collection_start = time.time()
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.PERFORMANCE_BUDGETS["total_collection_time"])
            collection_time = time.time() - collection_start
            
            print(f"✅ Coverage collection completed in {collection_time:.1f}s")
            
            # Load coverage data
            with open("coverage-data.json", "r") as f:
                self.coverage_data = json.load(f)
                
            return {
                "success": True,
                "collection_time": collection_time,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            print(f"⚠️  Coverage collection timed out after {self.PERFORMANCE_BUDGETS['total_collection_time']}s")
            return {"success": False, "error": "timeout"}
            
        except Exception as e:
            print(f"❌ Coverage collection failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_config_file(self) -> str:
        """Get coverage configuration file based on profile."""
        config_map = {
            "dev": ".coveragerc.dev",
            "ci": ".coveragerc.ci", 
            "release": ".coveragerc",
            "default": ".coveragerc"
        }
        return config_map.get(self.profile, ".coveragerc")
    
    def analyze_module_coverage(self) -> List[CoverageResult]:
        """Analyze coverage results per module with threshold checking."""
        results = []
        
        if not self.coverage_data or "files" not in self.coverage_data:
            print("⚠️  No coverage data available")
            return results
        
        print(f"🔍 Processing {len(self.coverage_data['files'])} files for coverage analysis")
        
        # Group files by module
        module_groups = {}
        for file_path, file_data in self.coverage_data["files"].items():
            # Normalize path and extract module
            normalized_path = file_path.replace("\\", "/")
            if normalized_path.startswith("./"):
                normalized_path = normalized_path[2:]
                
            module = self._extract_module(normalized_path)
            if module not in module_groups:
                module_groups[module] = []
            module_groups[module].append((normalized_path, file_data))
        
        # Calculate coverage per module
        for module, files in module_groups.items():
            total_lines = 0
            covered_lines = 0
            missing_lines = []
            
            for file_path, file_data in files:
                summary = file_data.get("summary", {})
                statements = summary.get("num_statements", 0)
                covered = summary.get("covered_lines", 0)
                
                # Only count files with actual statements
                if statements > 0:
                    total_lines += statements
                    covered_lines += covered
                    
                    # Collect missing lines
                    file_missing = file_data.get("missing_lines", [])
                    missing_lines.extend(file_missing)
            
            if total_lines > 0:
                coverage_percent = (covered_lines / total_lines) * 100
                
                result = CoverageResult(
                    module=module,
                    covered_lines=covered_lines,
                    total_lines=total_lines,
                    coverage_percent=coverage_percent,
                    missing_lines=missing_lines,
                    collection_time=0.0  # Per-module timing not available
                )
                results.append(result)
        
        return results
    
    def _extract_module(self, file_path: str) -> str:
        """Extract module name from file path."""
        parts = file_path.split("/")
        if len(parts) > 0:
            return parts[0]
        return "root"
    
    def check_thresholds(self, results: List[CoverageResult]) -> Tuple[List[str], Dict[str, str]]:
        """Check module coverage against thresholds."""
        violations = []
        coverage_debt = {}
        
        # Create lookup map of results
        result_map = {r.module: r for r in results}
        
        for threshold in self.MODULE_THRESHOLDS:
            module_name = threshold.path.rstrip("/")
            result = result_map.get(module_name)
            
            if not result:
                if threshold.criticality != "debt":
                    violations.append(f"Module {module_name} has no coverage data")
                continue
            
            coverage = result.coverage_percent
            
            if threshold.criticality == "debt":
                # Track improvement for debt modules
                if threshold.improvement_target:
                    coverage_debt[module_name] = f"Current: {coverage:.1f}%, Target: {threshold.improvement_target}"
            else:
                # Check threshold violations
                if coverage < threshold.min_coverage:
                    violations.append(
                        f"Module {module_name} coverage {coverage:.1f}% below minimum {threshold.min_coverage}% "
                        f"(criticality: {threshold.criticality})"
                    )
        
        return violations, coverage_debt
    
    def generate_next_actions(self, results: List[CoverageResult], violations: List[str]) -> List[str]:
        """Generate actionable next steps based on coverage analysis."""
        actions = []
        
        if not results:
            actions.append("📋 Run coverage collection first: python scripts/coverage-check.py")
            return actions
        
        # Sort results by coverage to prioritize lowest coverage
        sorted_results = sorted(results, key=lambda r: r.coverage_percent)
        
        for result in sorted_results[:5]:  # Top 5 lowest coverage modules
            if result.coverage_percent < 50:
                actions.append(
                    f"🎯 {result.module}: Add basic characterization tests "
                    f"({result.coverage_percent:.1f}% coverage, {len(result.missing_lines)} uncovered lines)"
                )
        
        # Specific actions for zero-coverage modules
        zero_coverage = [r for r in results if r.coverage_percent == 0]
        for result in zero_coverage:
            actions.append(
                f"🚨 {result.module}: Create first test - start with test_imports.py to verify module loads"
            )
        
        # Threshold violation actions
        for violation in violations[:3]:  # Limit to avoid overwhelming output
            module = violation.split()[1]
            actions.append(f"⚡ Fix critical threshold violation: {module}")
        
        if not actions:
            actions.append("✅ All modules meet minimum coverage thresholds")
            actions.append("🎯 Consider increasing coverage for modules below target thresholds")
        
        return actions
    
    def generate_report(self, results: List[CoverageResult]) -> CoverageReport:
        """Generate comprehensive coverage report."""
        violations, coverage_debt = self.check_thresholds(results)
        next_actions = self.generate_next_actions(results, violations)
        
        # Calculate total coverage
        total_covered = sum(r.covered_lines for r in results)
        total_lines = sum(r.total_lines for r in results)
        total_coverage = (total_covered / total_lines * 100) if total_lines > 0 else 0
        
        # Performance metrics
        total_time = time.time() - self.start_time
        performance_metrics = {
            "total_collection_time": total_time,
            "modules_analyzed": len(results),
            "avg_time_per_module": total_time / len(results) if results else 0
        }
        
        return CoverageReport(
            timestamp=datetime.now().isoformat(),
            profile=self.profile,
            total_coverage=total_coverage,
            module_results=results,
            threshold_violations=violations,
            coverage_debt=coverage_debt,
            performance_metrics=performance_metrics,
            next_actions=next_actions
        )
    
    def save_report(self, report: CoverageReport, output_path: str = "coverage-report.json"):
        """Save coverage report to JSON file."""
        with open(output_path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        print(f"📄 Coverage report saved to {output_path}")
    
    def print_summary(self, report: CoverageReport):
        """Print human-readable coverage summary."""
        print("\n" + "="*80)
        print(f"📊 COVERAGE REPORT - {report.profile.upper()} PROFILE")
        print("="*80)
        
        print(f"📈 Total Coverage: {report.total_coverage:.1f}%")
        print(f"🕒 Collection Time: {report.performance_metrics['total_collection_time']:.1f}s")
        print(f"📁 Modules Analyzed: {report.performance_metrics['modules_analyzed']}")
        
        if report.threshold_violations:
            print(f"\n❌ Threshold Violations ({len(report.threshold_violations)}):")
            for violation in report.threshold_violations:
                print(f"   {violation}")
        
        if report.coverage_debt:
            print(f"\n📊 Coverage Debt Tracking:")
            for module, status in report.coverage_debt.items():
                print(f"   {module}: {status}")
        
        print(f"\n🎯 Next Actions:")
        for action in report.next_actions:
            print(f"   {action}")
        
        # Module breakdown
        print(f"\n📂 Module Coverage Breakdown:")
        sorted_modules = sorted(report.module_results, key=lambda r: r.coverage_percent, reverse=True)
        for result in sorted_modules:
            status = "✅" if result.coverage_percent >= 75 else "⚠️" if result.coverage_percent >= 50 else "❌"
            print(f"   {status} {result.module:20} {result.coverage_percent:6.1f}% ({result.covered_lines}/{result.total_lines})")


def main():
    parser = argparse.ArgumentParser(description="Module-specific coverage checking")
    parser.add_argument("--profile", choices=["dev", "ci", "release", "default"], default="default",
                       help="Coverage profile to use")
    parser.add_argument("--module", help="Check specific module only")
    parser.add_argument("--no-collect", action="store_true", help="Skip coverage collection, analyze existing data")
    parser.add_argument("--output", default="coverage-report.json", help="Output file for JSON report")
    
    args = parser.parse_args()
    
    checker = CoverageChecker(profile=args.profile)
    
    # Run coverage collection unless skipped
    if not args.no_collect:
        collection_result = checker.run_coverage_collection(args.module)
        if not collection_result["success"]:
            print(f"❌ Coverage collection failed: {collection_result.get('error', 'unknown error')}")
            sys.exit(1)
    else:
        # Load existing coverage data when skipping collection
        try:
            with open("coverage-data.json", "r") as f:
                checker.coverage_data = json.load(f)
            print("📊 Loaded existing coverage data")
        except FileNotFoundError:
            print("❌ No existing coverage data found. Run without --no-collect first.")
            sys.exit(1)
    
    # Analyze results
    results = checker.analyze_module_coverage()
    report = checker.generate_report(results)
    
    # Output results
    checker.print_summary(report)
    checker.save_report(report, args.output)
    
    # Exit with appropriate code
    if report.threshold_violations:
        print(f"\n❌ Coverage check failed: {len(report.threshold_violations)} threshold violations")
        sys.exit(1)
    else:
        print(f"\n✅ Coverage check passed")
        sys.exit(0)


if __name__ == "__main__":
    main()