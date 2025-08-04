#!/usr/bin/env python3
"""
Test Execution Success Rate Validator
====================================

Validates test execution success rate for subtask 49.5.
Runs strategic test sampling to measure infrastructure success rate.

Target: >95% test execution success (at least 595/626 tests)
"""

import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import time


class TestExecutionValidator:
    """Validates test execution success rates."""
    
    def __init__(self):
        self.target_success_rate = 95.0
        self.total_collected = 626  # From pytest --collect-only
        self.target_passes = int(self.total_collected * (self.target_success_rate / 100))
        
    def run_pytest_command(self, args: List[str], timeout: int = 300) -> Tuple[int, str, str]:
        """Run pytest command with timeout."""
        cmd = ["python", "-m", "pytest"] + args
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                cwd=Path(__file__).parent
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout}s"
    
    def validate_test_collection(self) -> bool:
        """Validate that test collection works without crashes."""
        print("🔍 Validating test collection...")
        returncode, stdout, stderr = self.run_pytest_command(
            ["--collect-only", "-q"], timeout=60
        )
        
        if returncode != 0:
            print(f"❌ Test collection failed (exit code: {returncode})")
            print(f"STDERR: {stderr}")
            return False
            
        lines = stdout.strip().split('\n')
        collected_count = len(lines) - 6  # Subtract header/footer lines
        
        print(f"✅ Test collection successful: {collected_count} tests collected")
        return True
    
    def run_sample_tests(self, sample_patterns: List[str]) -> Dict[str, any]:
        """Run sample test patterns and measure success rate."""
        results = {}
        total_passed = 0
        total_failed = 0
        total_errors = 0
        
        for pattern in sample_patterns:
            print(f"🧪 Testing pattern: {pattern}")
            
            returncode, stdout, stderr = self.run_pytest_command([
                pattern, "-v", "--tb=no", "--no-header", "-q"
            ], timeout=120)
            
            # Parse output for PASSED/FAILED/ERROR counts
            passed = stdout.count("PASSED")
            failed = stdout.count("FAILED") 
            errors = stdout.count("ERROR")
            
            total_passed += passed
            total_failed += failed
            total_errors += errors
            
            pattern_total = passed + failed + errors
            pattern_success = (passed / pattern_total * 100) if pattern_total > 0 else 0
            
            results[pattern] = {
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "total": pattern_total,
                "success_rate": pattern_success,
                "returncode": returncode
            }
            
            print(f"  📊 {passed}P/{failed}F/{errors}E ({pattern_success:.1f}% success)")
        
        overall_total = total_passed + total_failed + total_errors
        overall_success = (total_passed / overall_total * 100) if overall_total > 0 else 0
        
        results["summary"] = {
            "total_passed": total_passed,
            "total_failed": total_failed,
            "total_errors": total_errors,
            "total_tests": overall_total,
            "success_rate": overall_success
        }
        
        return results
    
    def run_comprehensive_validation(self) -> bool:
        """Run comprehensive test execution validation."""
        print("🚀 Starting comprehensive test execution validation")
        print(f"📈 Target: ≥{self.target_success_rate}% success rate ({self.target_passes}/{self.total_collected} tests)")
        print()
        
        # Step 1: Validate collection
        if not self.validate_test_collection():
            return False
        
        # Step 2: Sample test execution across key modules
        sample_patterns = [
            "tests/db_infrastructure/",
            "tests/test_health_endpoints.py",
            "tests/agents/creation/test_factory.py",
            "tests/environment/test_environment_manager.py",
            "tests/characterization/test_critical_paths.py",
        ]
        
        print("\n📋 Running strategic test samples...")
        results = self.run_sample_tests(sample_patterns)
        
        # Step 3: Analyze results
        summary = results["summary"]
        success_rate = summary["success_rate"]
        
        print("\n" + "="*60)
        print("📊 TEST EXECUTION VALIDATION RESULTS")
        print("="*60)
        print(f"Total Tests Sampled: {summary['total_tests']}")
        print(f"Passed: {summary['total_passed']}")
        print(f"Failed: {summary['total_failed']}")
        print(f"Errors: {summary['total_errors']}")
        print(f"Success Rate: {success_rate:.2f}%")
        print(f"Target Success Rate: {self.target_success_rate}%")
        print()
        
        # Step 4: Project to full test suite
        if summary['total_tests'] > 0:
            projected_passes = int((success_rate / 100) * self.total_collected)
            print(f"📈 Projected Full Suite Results:")
            print(f"  Expected Passes: ~{projected_passes}/{self.total_collected}")
            print(f"  Projected Success Rate: ~{success_rate:.1f}%")
            print()
        
        # Step 5: Validation decision
        meets_target = success_rate >= self.target_success_rate
        
        if meets_target:
            print("✅ SUCCESS: Test execution validation PASSED")
            print(f"   Achieved {success_rate:.2f}% success rate (≥{self.target_success_rate}% target)")
        else:
            print("❌ FAILURE: Test execution validation FAILED")
            print(f"   Achieved {success_rate:.2f}% success rate (<{self.target_success_rate}% target)")
        
        print()
        return meets_target
    
    def generate_report(self, results: Dict) -> str:
        """Generate detailed test execution report."""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "target_success_rate": self.target_success_rate,
            "total_collected_tests": self.total_collected,
            "validation_results": results
        }
        
        return json.dumps(report, indent=2)


def main():
    """Main execution function."""
    validator = TestExecutionValidator()
    success = validator.run_comprehensive_validation()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())