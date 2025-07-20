#!/usr/bin/env python3
"""Monitor quality gates and report when ready for release."""

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


class QualityGateMonitor:
    """Monitor quality gates for release readiness."""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.last_check = None
        self.check_interval = 300  # 5 minutes

    def quick_check(self) -> Dict[str, bool]:
        """Perform quick checks on key quality gates."""
        results = {}

        # Quick pytest check (just see if it runs)
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--collect-only", "-q"],
                capture_output=True,
                timeout=30,
                cwd=self.project_root,
            )
            results["pytest_collect"] = result.returncode == 0
        except:
            results["pytest_collect"] = False

        # Quick flake8 count
        try:
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "flake8",
                    ".",
                    "--count",
                    "--select=E9,F63,F7,F82",
                    "--show-source",
                    "--statistics",
                    "--exclude=.git,__pycache__,venv,.venv,migrations,.archive,build,dist,*.egg-info,.mypy_cache,.pytest_cache,security_audit_env,security_scan_env,node_modules,htmlcov",
                ],
                capture_output=True,
                timeout=60,
                cwd=self.project_root,
            )
            # E9,F63,F7,F82 are the "showstopper" flake8 issues
            results["flake8_critical"] = result.returncode == 0
        except:
            results["flake8_critical"] = False

        # Check if npm build still works
        try:
            web_dir = self.project_root / "web"
            result = subprocess.run(
                ["npm", "run", "build"],
                capture_output=True,
                timeout=120,
                cwd=web_dir,
            )
            results["npm_build"] = result.returncode == 0
        except:
            results["npm_build"] = False

        # Check Docker build
        try:
            result = subprocess.run(
                ["make", "docker-build"],
                capture_output=True,
                timeout=300,
                cwd=self.project_root,
            )
            results["docker_build"] = result.returncode == 0
        except:
            results["docker_build"] = False

        return results

    def check_agent_progress(self) -> List[str]:
        """Check for recent agent activity and fixes."""
        recent_files = []

        try:
            # Look for files modified in the last hour
            result = subprocess.run(
                [
                    "find",
                    ".",
                    "-type",
                    "f",
                    "-mmin",
                    "-60",
                    "-name",
                    "*.py",
                    "-o",
                    "-name",
                    "*.md",
                ],
                capture_output=True,
                text=True,
                cwd=self.project_root,
            )

            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
                recent_files = [
                    f
                    for f in files
                    if f
                    and not any(
                        skip in f
                        for skip in [
                            'venv/',
                            'node_modules/',
                            '.git/',
                            '__pycache__/',
                            'htmlcov/',
                        ]
                    )
                ]
        except:
            pass

        return recent_files

    def generate_status_summary(self) -> str:
        """Generate a summary of current status."""
        timestamp = datetime.now().isoformat()
        quick_results = self.quick_check()
        recent_activity = self.check_agent_progress()

        summary = f"""
# Quality Gate Monitoring Report
**Timestamp**: {timestamp}
**Monitor**: FINAL-RELEASE-VALIDATOR

## Quick Check Results
- Pytest Collection: {'✅ PASS' if quick_results.get('pytest_collect') else '❌ FAIL'}
- Flake8 Critical: {'✅ PASS' if quick_results.get('flake8_critical') else '❌ FAIL'}
- NPM Build: {'✅ PASS' if quick_results.get('npm_build') else '❌ FAIL'}
- Docker Build: {'✅ PASS' if quick_results.get('docker_build') else '❌ FAIL'}

## Recent Agent Activity
Files modified in last hour: {len(recent_activity)}
"""

        if recent_activity:
            summary += "\nRecent changes:\n"
            for file in recent_activity[:10]:  # Show first 10
                summary += f"- {file}\n"

        # Check if ready for full validation
        critical_passed = quick_results.get(
            'pytest_collect', False
        ) and quick_results.get('flake8_critical', False)

        if critical_passed:
            summary += "\n## Status: Ready for Full Validation\n"
            summary += (
                "Critical checks passed. Run full quality gate validation.\n"
            )
        else:
            summary += "\n## Status: Waiting for Fixes\n"
            summary += "Critical issues remain. Continue monitoring.\n"

        return summary

    def monitor_once(self) -> bool:
        """Perform one monitoring cycle. Returns True if ready for release."""
        print("\n" + "=" * 60)
        print(
            f"🔍 Quality Gate Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print("=" * 60)

        summary = self.generate_status_summary()
        print(summary)

        # Save summary
        report_path = (
            self.project_root
            / f"quality_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        with open(report_path, 'w') as f:
            f.write(summary)

        # Check if we should run full validation
        quick_results = self.quick_check()
        if quick_results.get('pytest_collect') and quick_results.get(
            'flake8_critical'
        ):
            print("\n🎯 Critical checks passed! Running full validation...")

            # Run full validation
            try:
                result = subprocess.run(
                    ["python", "scripts/validate_quality_gates.py"],
                    cwd=self.project_root,
                )
                return result.returncode == 0
            except Exception as e:
                print(f"❌ Full validation failed: {e}")
                return False

        return False

    def monitor_continuous(self):
        """Continuously monitor until ready for release."""
        print("🚀 Starting Quality Gate Monitor")
        print(f"Checking every {self.check_interval} seconds...")

        while True:
            try:
                ready = self.monitor_once()

                if ready:
                    print("\n✅ ALL QUALITY GATES PASSED!")
                    print("Ready to tag v1.0.0-alpha+")
                    break

                print(f"\n⏳ Next check in {self.check_interval} seconds...")
                time.sleep(self.check_interval)

            except KeyboardInterrupt:
                print("\n👋 Monitor stopped by user")
                break
            except Exception as e:
                print(f"\n❌ Monitor error: {e}")
                print(f"Retrying in {self.check_interval} seconds...")
                time.sleep(self.check_interval)


if __name__ == "__main__":
    monitor = QualityGateMonitor()

    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        # Single check mode
        ready = monitor.monitor_once()
        sys.exit(0 if ready else 1)
    else:
        # Continuous monitoring mode
        monitor.monitor_continuous()
