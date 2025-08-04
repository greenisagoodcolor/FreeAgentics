#!/usr/bin/env python3
"""
Simple Performance Gate for CI/CD
==================================

Focused on developer release requirements only.
No production monitoring or advanced features.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional


@dataclass
class GateResult:
    """Simple pass/fail result."""
    passed: bool
    message: str
    details: Dict[str, Any]


class PerformanceGate:
    """Simple performance regression gate for CI/CD."""
    
    # Developer release thresholds from CLAUDE.md
    AGENT_SPAWN_TARGET_MS = 50.0
    MEMORY_BUDGET_MB = 34.5
    REGRESSION_THRESHOLD = 0.10  # 10%
    
    def __init__(self, baseline_file: str = "minimal_baseline.json"):
        """Initialize with baseline file."""
        self.baseline_file = Path(baseline_file)
        self.baseline = self._load_baseline()
    
    def _load_baseline(self) -> Dict[str, Any]:
        """Load baseline metrics."""
        if not self.baseline_file.exists():
            return {}
        
        try:
            with open(self.baseline_file) as f:
                data = json.load(f)
                return data.get("metrics", {})
        except Exception as e:
            print(f"Warning: Failed to load baseline: {e}")
            return {}
    
    def check(self, current_file: str) -> GateResult:
        """Check current metrics against baseline."""
        try:
            with open(current_file) as f:
                current_data = json.load(f)
        except Exception as e:
            return GateResult(
                passed=False,
                message=f"Failed to load current metrics: {e}",
                details={}
            )
        
        current_metrics = current_data.get("metrics", {})
        
        if not self.baseline:
            return GateResult(
                passed=True,
                message="No baseline found - accepting current metrics",
                details={"current": current_metrics}
            )
        
        failures = []
        checks = {}
        
        # Check agent spawn time
        if "agent_spawning" in current_metrics:
            current_spawn = current_metrics["agent_spawning"].get("p95_ms", 0)
            baseline_spawn = self.baseline.get("agent_spawning", {}).get("p95_ms", 0)
            
            # Hard limit check
            if current_spawn > self.AGENT_SPAWN_TARGET_MS:
                failures.append(f"Agent spawn P95 ({current_spawn:.1f}ms) > target ({self.AGENT_SPAWN_TARGET_MS}ms)")
            
            # Regression check
            if baseline_spawn > 0:
                regression = (current_spawn - baseline_spawn) / baseline_spawn
                if regression > self.REGRESSION_THRESHOLD:
                    failures.append(f"Agent spawn regressed {regression*100:.1f}% (baseline: {baseline_spawn:.1f}ms → current: {current_spawn:.1f}ms)")
            
            checks["agent_spawn"] = {
                "current": current_spawn,
                "baseline": baseline_spawn,
                "target": self.AGENT_SPAWN_TARGET_MS,
                "passed": current_spawn <= self.AGENT_SPAWN_TARGET_MS
            }
        
        # Check memory usage
        if "memory_usage" in current_metrics:
            current_memory = current_metrics["memory_usage"].get("per_agent_mb", 0)
            baseline_memory = self.baseline.get("memory_usage", {}).get("per_agent_mb", 0)
            
            # Hard limit check
            if current_memory > self.MEMORY_BUDGET_MB:
                failures.append(f"Memory per agent ({current_memory:.1f}MB) > budget ({self.MEMORY_BUDGET_MB}MB)")
            
            # Regression check
            if baseline_memory > 0:
                regression = (current_memory - baseline_memory) / baseline_memory
                if regression > self.REGRESSION_THRESHOLD:
                    failures.append(f"Memory usage regressed {regression*100:.1f}% (baseline: {baseline_memory:.1f}MB → current: {current_memory:.1f}MB)")
            
            checks["memory"] = {
                "current": current_memory,
                "baseline": baseline_memory,
                "budget": self.MEMORY_BUDGET_MB,
                "passed": current_memory <= self.MEMORY_BUDGET_MB
            }
        
        # Check PyMDP performance
        if "pymdp_inference" in current_metrics:
            current_pymdp = current_metrics["pymdp_inference"].get("p95_ms", 0)
            baseline_pymdp = self.baseline.get("pymdp_inference", {}).get("p95_ms", 0)
            
            if baseline_pymdp > 0:
                regression = (current_pymdp - baseline_pymdp) / baseline_pymdp
                if regression > self.REGRESSION_THRESHOLD:
                    failures.append(f"PyMDP inference regressed {regression*100:.1f}% (baseline: {baseline_pymdp:.1f}ms → current: {current_pymdp:.1f}ms)")
            
            checks["pymdp"] = {
                "current": current_pymdp,
                "baseline": baseline_pymdp,
                "passed": baseline_pymdp == 0 or regression <= self.REGRESSION_THRESHOLD
            }
        
        passed = len(failures) == 0
        message = "✅ All performance gates passed" if passed else f"❌ {len(failures)} performance gate(s) failed"
        
        return GateResult(
            passed=passed,
            message=message,
            details={
                "failures": failures,
                "checks": checks,
                "threshold": f"{self.REGRESSION_THRESHOLD*100:.0f}%"
            }
        )


def main():
    """CLI entry point."""
    if len(sys.argv) != 2:
        print("Usage: python performance_gate.py <current_metrics.json>")
        sys.exit(1)
    
    current_file = sys.argv[1]
    gate = PerformanceGate()
    result = gate.check(current_file)
    
    print(result.message)
    
    if result.details.get("failures"):
        print("\nFailures:")
        for failure in result.details["failures"]:
            print(f"  • {failure}")
    
    if result.details.get("checks"):
        print(f"\nChecks (threshold: {result.details['threshold']}):")
        for name, check in result.details["checks"].items():
            status = "✅" if check["passed"] else "❌"
            print(f"  {status} {name}: {check['current']:.1f} (baseline: {check.get('baseline', 'N/A')})")
    
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()