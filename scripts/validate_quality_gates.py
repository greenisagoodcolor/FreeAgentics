#!/usr/bin/env python3
"""Validate all quality gates for v1.0.0-alpha+ release."""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Tuple


class QualityGateValidator:
    """Validator for all release quality gates."""

    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "gates": {},
            "summary": {"passed": 0, "failed": 0, "total": 0},
        }
        self.project_root = Path(__file__).parent.parent

    def run_command(self, cmd: List[str], timeout: int = 300) -> Tuple[int, str, str]:
        """Run a command and return exit code, stdout, stderr."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.project_root,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return -1, "", str(e)

    def check_pre_commit(self) -> bool:
        """Check if pre-commit run --all-files passes."""
        print("🔍 Checking pre-commit hooks...")
        returncode, stdout, stderr = self.run_command(
            ["pre-commit", "run", "--all-files"], timeout=600
        )

        passed = returncode == 0
        self.results["gates"]["pre_commit"] = {
            "passed": passed,
            "output": stdout + stderr if not passed else "All hooks passed",
        }
        return passed

    def check_pytest(self) -> bool:
        """Check if pytest passes with adequate coverage."""
        print("🔍 Running pytest with coverage...")
        returncode, stdout, stderr = self.run_command(
            [
                "python",
                "-m",
                "pytest",
                "-q",
                "--tb=short",
                "--cov=.",
                "--cov-report=term-missing",
            ],
            timeout=600,
        )

        # Extract coverage percentage
        coverage: float = 0.0
        for line in stdout.split("\n"):
            if "TOTAL" in line and "%" in line:
                try:
                    parts = line.split()
                    for part in parts:
                        if part.endswith("%"):
                            coverage = float(part.rstrip("%"))
                            break
                except Exception:
                    pass

        passed = returncode == 0 and coverage >= 80
        self.results["gates"]["pytest"] = {
            "passed": passed,
            "coverage": coverage,
            "output": f"Exit code: {returncode}, Coverage: {coverage}%",
        }
        return passed

    def check_npm_build(self) -> bool:
        """Check if npm run build succeeds."""
        print("🔍 Building frontend...")
        web_dir = self.project_root / "web"
        if not web_dir.exists():
            self.results["gates"]["npm_build"] = {
                "passed": False,
                "output": "web directory not found",
            }
            return False

        returncode, stdout, stderr = self.run_command(
            ["npm", "run", "build"], timeout=300
        )

        passed = returncode == 0
        self.results["gates"]["npm_build"] = {
            "passed": passed,
            "output": stderr if not passed else "Build successful",
        }
        return passed

    def check_docker_build(self) -> bool:
        """Check if make docker-build succeeds."""
        print("🔍 Building Docker images...")
        returncode, stdout, stderr = self.run_command(
            ["make", "docker-build"], timeout=600
        )

        passed = returncode == 0
        self.results["gates"]["docker_build"] = {
            "passed": passed,
            "output": stderr if not passed else "Docker build successful",
        }
        return passed

    def check_flake8(self) -> bool:
        """Check if flake8 has 0 errors."""
        print("🔍 Running flake8...")
        returncode, stdout, stderr = self.run_command(
            [
                "python",
                "-m",
                "flake8",
                ".",
                "--count",
                "--statistics",
                "--exclude=.git,__pycache__,venv,.venv,migrations,.archive,build,dist,*.egg-info,.mypy_cache,.pytest_cache,security_audit_env,security_scan_env,node_modules,htmlcov",
            ],
            timeout=300,
        )

        # Extract error count
        error_count = 0
        for line in stdout.split("\n"):
            if line.strip():
                try:
                    parts = line.split()
                    if parts and parts[0].isdigit():
                        error_count += int(parts[0])
                except Exception:
                    pass

        passed = error_count == 0
        self.results["gates"]["flake8"] = {
            "passed": passed,
            "errors": error_count,
            "output": f"{error_count} errors found",
        }
        return passed

    def check_mypy(self) -> bool:
        """Check if mypy has 0 errors."""
        print("🔍 Running mypy...")
        returncode, stdout, stderr = self.run_command(
            [
                "python",
                "-m",
                "mypy",
                ".",
                "--exclude",
                "venv|security_audit_env|security_scan_env|node_modules|htmlcov",
            ],
            timeout=300,
        )

        # Count errors
        error_count = len(
            [line for line in (stdout + stderr).split("\n") if "error:" in line]
        )

        passed = error_count == 0
        self.results["gates"]["mypy"] = {
            "passed": passed,
            "errors": error_count,
            "output": f"{error_count} errors found",
        }
        return passed

    def validate_all(self) -> bool:
        """Run all quality gate checks."""
        print("🚀 FreeAgentics v1.0.0-alpha+ Quality Gate Validation")
        print("=" * 60)

        gates = [
            ("Pre-commit hooks", self.check_pre_commit),
            ("Pytest + Coverage", self.check_pytest),
            ("NPM Build", self.check_npm_build),
            ("Docker Build", self.check_docker_build),
            ("Flake8", self.check_flake8),
            ("Mypy", self.check_mypy),
        ]

        for name, check_func in gates:
            try:
                passed = check_func()
                self.results["summary"]["total"] += 1
                if passed:
                    self.results["summary"]["passed"] += 1
                    print(f"✅ {name}: PASSED")
                else:
                    self.results["summary"]["failed"] += 1
                    print(f"❌ {name}: FAILED")
            except Exception as e:
                self.results["summary"]["total"] += 1
                self.results["summary"]["failed"] += 1
                print(f"❌ {name}: ERROR - {str(e)}")
                self.results["gates"][name.lower().replace(" ", "_")] = {
                    "passed": False,
                    "output": str(e),
                }

        print("\n" + "=" * 60)
        print(
            f"📊 Summary: {self.results['summary']['passed']}/{self.results['summary']['total']} gates passed"
        )

        # Save results
        report_path = (
            self.project_root
            / f"quality_gate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📄 Report saved to: {report_path}")

        all_passed = self.results["summary"]["failed"] == 0
        if all_passed:
            print("\n✅ All quality gates passed! Ready for v1.0.0-alpha+ release")
        else:
            print("\n❌ Some quality gates failed. Please fix issues before release.")

        return all_passed


if __name__ == "__main__":
    validator = QualityGateValidator()
    success = validator.validate_all()
    sys.exit(0 if success else 1)
