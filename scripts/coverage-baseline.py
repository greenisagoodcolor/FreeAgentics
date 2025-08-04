#!/usr/bin/env python3
"""
Coverage baseline establishment and tracking.

This script establishes baseline coverage metrics for all modules and tracks
improvement over time. Part of the Nemesis Committee's progressive coverage strategy.

Usage:
    python scripts/coverage-baseline.py [--establish] [--compare]
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import subprocess
import argparse


@dataclass
class BaselineEntry:
    """Baseline coverage entry for a module."""
    module: str
    baseline_coverage: float
    baseline_date: str
    target_coverage: float
    target_date: str
    current_coverage: Optional[float] = None
    improvement_velocity: Optional[float] = None  # % improvement per month


@dataclass
class BaselineReport:
    """Complete baseline tracking report."""
    established_date: str
    last_updated: str
    baselines: List[BaselineEntry]
    overall_improvement: float
    modules_improving: int
    modules_regressing: int


class CoverageBaseline:
    """Coverage baseline establishment and tracking."""
    
    BASELINE_FILE = "coverage-baseline.json"
    ZERO_COVERAGE_TARGETS = {
        # Task 54 zero-coverage modules with improvement targets
        "inference/gnn": {"target": 40.0, "date": "2025-09-01"},
        "inference/llm": {"target": 60.0, "date": "2025-08-01"},
        "observability": {"target": 45.0, "date": "2025-08-01"},
        "security": {"target": 70.0, "date": "2025-09-01"},
    }
    
    def __init__(self):
        self.baseline_path = Path(self.BASELINE_FILE)
        
    def establish_baseline(self) -> BaselineReport:
        """Establish initial coverage baseline for all modules."""
        print("📊 Establishing coverage baseline...")
        
        # Run current coverage analysis
        current_coverage = self._get_current_coverage()
        
        baselines = []
        established_date = datetime.now().isoformat()
        
        # Create baseline entries for each module
        for module, coverage in current_coverage.items():
            # Determine target based on module type
            target_coverage, target_date = self._get_target_for_module(module, coverage)
            
            baseline = BaselineEntry(
                module=module,
                baseline_coverage=coverage,
                baseline_date=established_date,
                target_coverage=target_coverage,
                target_date=target_date,
                current_coverage=coverage,
                improvement_velocity=0.0
            )
            baselines.append(baseline)
        
        report = BaselineReport(
            established_date=established_date,
            last_updated=established_date,
            baselines=baselines,
            overall_improvement=0.0,
            modules_improving=0,
            modules_regressing=0
        )
        
        # Save baseline
        self._save_baseline(report)
        
        print(f"✅ Baseline established for {len(baselines)} modules")
        return report
    
    def _get_current_coverage(self) -> Dict[str, float]:
        """Get current coverage data by running coverage collection."""
        try:
            # Run coverage collection
            cmd = [
                "python", "-m", "pytest",
                "--cov=.",
                "--cov-report=json:coverage-current.json",
                "--tb=no", "-q"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"⚠️  Coverage collection completed with warnings")
            
            # Load and parse coverage data
            with open("coverage-current.json", "r") as f:
                coverage_data = json.load(f)
            
            # Group by module and calculate coverage
            module_coverage = {}
            
            if "files" in coverage_data:
                module_groups = {}
                for file_path, file_data in coverage_data["files"].items():
                    module = self._extract_module(file_path)
                    if module not in module_groups:
                        module_groups[module] = {"covered": 0, "total": 0}
                    
                    summary = file_data.get("summary", {})
                    module_groups[module]["covered"] += summary.get("covered_lines", 0)
                    module_groups[module]["total"] += summary.get("num_statements", 0)
                
                # Calculate percentages
                for module, data in module_groups.items():
                    if data["total"] > 0:
                        coverage_percent = (data["covered"] / data["total"]) * 100
                        module_coverage[module] = round(coverage_percent, 2)
                    else:
                        module_coverage[module] = 0.0
            
            return module_coverage
            
        except Exception as e:
            print(f"❌ Failed to get current coverage: {e}")
            return {}
    
    def _extract_module(self, file_path: str) -> str:
        """Extract module name from file path."""
        # Normalize path
        normalized = file_path.replace("\\", "/")
        if normalized.startswith("./"):
            normalized = normalized[2:]
        
        # Extract first directory as module
        parts = normalized.split("/")
        if len(parts) > 0 and parts[0]:
            return parts[0]
        return "root"
    
    def _get_target_for_module(self, module: str, current_coverage: float) -> tuple:
        """Determine target coverage and date for a module."""
        # Check if it's a zero-coverage target module
        for target_module, config in self.ZERO_COVERAGE_TARGETS.items():
            if module.startswith(target_module):
                return config["target"], config["date"]
        
        # Default targets based on current coverage and module criticality
        if module in ["api", "auth", "database", "agents"]:
            # Critical modules
            return max(90.0, current_coverage + 10), "2025-08-15"
        elif module in ["coalitions", "services", "knowledge_graph"]:
            # Business logic
            return max(80.0, current_coverage + 15), "2025-09-01"
        else:
            # Infrastructure modules
            return max(60.0, current_coverage + 20), "2025-10-01"
    
    def compare_to_baseline(self) -> Optional[BaselineReport]:
        """Compare current coverage to established baseline."""
        if not self.baseline_path.exists():
            print("❌ No baseline exists. Run with --establish first.")
            return None
        
        print("📈 Comparing current coverage to baseline...")
        
        # Load existing baseline
        with open(self.baseline_path, "r") as f:
            baseline_data = json.load(f)
        
        # Get current coverage
        current_coverage = self._get_current_coverage()
        
        # Update baseline entries with current data
        baselines = []
        modules_improving = 0
        modules_regressing = 0
        total_improvement = 0.0
        
        for entry_data in baseline_data["baselines"]:
            entry = BaselineEntry(**entry_data)
            
            # Update with current coverage
            current = current_coverage.get(entry.module, 0.0)
            entry.current_coverage = current
            
            # Calculate improvement
            improvement = current - entry.baseline_coverage
            total_improvement += improvement
            
            # Calculate velocity (improvement per month since baseline)
            baseline_date = datetime.fromisoformat(entry.baseline_date)
            months_elapsed = (datetime.now() - baseline_date).days / 30.0
            if months_elapsed > 0:
                entry.improvement_velocity = improvement / months_elapsed
            
            # Track improvement/regression
            if improvement > 0.5:  # At least 0.5% improvement
                modules_improving += 1
            elif improvement < -0.5:  # More than 0.5% regression
                modules_regressing += 1
            
            baselines.append(entry)
        
        # Create updated report
        report = BaselineReport(
            established_date=baseline_data["established_date"],
            last_updated=datetime.now().isoformat(),
            baselines=baselines,
            overall_improvement=total_improvement / len(baselines) if baselines else 0,
            modules_improving=modules_improving,
            modules_regressing=modules_regressing
        )
        
        # Save updated baseline
        self._save_baseline(report)
        
        return report
    
    def _save_baseline(self, report: BaselineReport):
        """Save baseline report to file."""
        with open(self.baseline_path, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        print(f"💾 Baseline saved to {self.baseline_path}")
    
    def print_baseline_report(self, report: BaselineReport):
        """Print human-readable baseline report."""
        print("\n" + "="*80)
        print("📊 COVERAGE BASELINE REPORT")
        print("="*80)
        
        print(f"📅 Baseline Established: {report.established_date}")
        print(f"🔄 Last Updated: {report.last_updated}")
        print(f"📈 Overall Improvement: {report.overall_improvement:+.1f}%")
        print(f"📊 Modules Improving: {report.modules_improving}")
        print(f"📉 Modules Regressing: {report.modules_regressing}")
        
        print(f"\n📂 Module Progress:")
        print(f"{'Module':<20} {'Baseline':<10} {'Current':<10} {'Target':<10} {'Progress':<12} {'Velocity':<10}")
        print("-" * 80)
        
        # Sort by improvement (descending)
        sorted_baselines = sorted(report.baselines, 
                                key=lambda b: (b.current_coverage or 0) - b.baseline_coverage, 
                                reverse=True)
        
        for baseline in sorted_baselines:
            current = baseline.current_coverage or 0.0
            improvement = current - baseline.baseline_coverage
            velocity = baseline.improvement_velocity or 0.0
            
            # Progress toward target
            target_progress = "✅" if current >= baseline.target_coverage else "🎯"
            
            # Status indicators
            if improvement > 0.5:
                status = "📈"
            elif improvement < -0.5:
                status = "📉"
            else:
                status = "➡️"
            
            print(f"{baseline.module:<20} {baseline.baseline_coverage:>8.1f}% "
                  f"{current:>8.1f}% {baseline.target_coverage:>8.1f}% "
                  f"{status} {improvement:+6.1f}% {velocity:>7.1f}%/mo")
        
        # Zero-coverage module callouts
        zero_modules = [b for b in report.baselines if (b.current_coverage or 0) == 0]
        if zero_modules:
            print(f"\n🚨 Zero Coverage Modules ({len(zero_modules)}):")
            for baseline in zero_modules:
                print(f"   {baseline.module}: Target {baseline.target_coverage}% by {baseline.target_date}")
        
        # Achievement callouts
        achieved_targets = [b for b in report.baselines 
                          if (b.current_coverage or 0) >= b.target_coverage]
        if achieved_targets:
            print(f"\n🎉 Target Achievements ({len(achieved_targets)}):")
            for baseline in achieved_targets:
                print(f"   {baseline.module}: {baseline.current_coverage:.1f}% (target: {baseline.target_coverage}%)")


def main():
    parser = argparse.ArgumentParser(description="Coverage baseline establishment and tracking")
    parser.add_argument("--establish", action="store_true", 
                       help="Establish new coverage baseline")
    parser.add_argument("--compare", action="store_true",
                       help="Compare current coverage to baseline")
    
    args = parser.parse_args()
    
    if not args.establish and not args.compare:
        # Default to compare if baseline exists, establish if not
        baseline_checker = CoverageBaseline()
        if baseline_checker.baseline_path.exists():
            args.compare = True
        else:
            args.establish = True
    
    baseline_checker = CoverageBaseline()
    
    if args.establish:
        report = baseline_checker.establish_baseline()
        baseline_checker.print_baseline_report(report)
    
    if args.compare:
        report = baseline_checker.compare_to_baseline()
        if report:
            baseline_checker.print_baseline_report(report)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()