#!/usr/bin/env python3
"""
Focused OWASP Top 10 Security Assessment for FreeAgentics Application Code
Task 14.11 - Agent 4 Implementation

This script focuses on the application code, excluding third-party dependencies.
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class FocusedOWASPAssessment:
    """Focused OWASP Top 10 assessment for application code only."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.findings: List[Dict[str, Any]] = []
        self.files_analyzed = 0

        # Exclude third-party directories
        self.exclude_dirs = {
            "node_modules",
            "venv",
            ".git",
            "__pycache__",
            ".pytest_cache",
            "htmlcov",
            "test-reports",
            "coverage",
            "build",
            "dist",
            ".eggs",
            "env",
            "ENV",
            "venv",
            "lib",
            "lib64",
            "share",
            "bin",
            "Scripts",
            "Include",
            "tcl",
            "Doc",
            "Tools",
            "libs",
            "DLLs",
            "pyvenv.cfg",
        }

        # Focus on application directories
        self.focus_dirs = {
            "api",
            "auth",
            "agents",
            "coalitions",
            "database",
            "inference",
            "knowledge_graph",
            "observability",
            "security",
            "world",
            "web",
            "examples",
            "scripts",
            "tests",
            "monitoring",
            "utils",
            "config",
        }

    def should_analyze_file(self, file_path: Path) -> bool:
        """Check if file should be analyzed."""
        # Skip if in excluded directories
        for part in file_path.parts:
            if part in self.exclude_dirs:
                return False

        # Only analyze Python and JavaScript files
        if file_path.suffix not in [".py", ".js", ".jsx", ".ts", ".tsx"]:
            return False

        # Skip test files for some checks
        if any(
            test_dir in file_path.parts
            for test_dir in ["tests", "test", "__tests__"]
        ):
            return True  # We want to analyze tests but with different criteria

        return True

    def add_finding(
        self,
        category: str,
        severity: str,
        title: str,
        description: str,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        evidence: Optional[str] = None,
        remediation: Optional[str] = None,
    ):
        """Add a security finding."""
        self.findings.append(
            {
                "category": category,
                "severity": severity,
                "title": title,
                "description": description,
                "file_path": file_path,
                "line_number": line_number,
                "evidence": evidence,
                "remediation": remediation,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def test_a01_broken_access_control(self):
        """Test for A01:2021 – Broken Access Control."""
        print("\n[*] Testing A01: Broken Access Control...")

        # Check API endpoints for authentication
        api_files = []
        for api_dir in ["api", "auth"]:
            api_files.extend(self.project_root.glob(f"{api_dir}/**/*.py"))

        unprotected_endpoints = []
        for file_path in api_files:
            if not self.should_analyze_file(file_path):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                self.files_analyzed += 1

                # Look for FastAPI endpoints
                endpoint_patterns = [
                    r"@app\.(get|post|put|delete|patch)\s*\([^)]*\)",
                    r"@router\.(get|post|put|delete|patch)\s*\([^)]*\)",
                ]

                for pattern in endpoint_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_number = content[: match.start()].count("\n") + 1

                        # Check if endpoint is protected
                        # Look for authentication in the next few lines
                        lines = content.split("\n")
                        endpoint_line = line_number - 1

                        # Check next 10 lines for authentication
                        protected = False
                        for i in range(
                            max(0, endpoint_line - 5),
                            min(len(lines), endpoint_line + 10),
                        ):
                            if any(
                                auth_pattern in lines[i]
                                for auth_pattern in [
                                    "@require_permission",
                                    "@require_role",
                                    "@login_required",
                                    "get_current_user",
                                    "require_permission",
                                    "Depends(get_current_user)",
                                ]
                            ):
                                protected = True
                                break

                        if not protected:
                            # Skip health/docs endpoints
                            if any(
                                skip in match.group(0)
                                for skip in [
                                    "/health",
                                    "/docs",
                                    "/redoc",
                                    "/openapi",
                                ]
                            ):
                                continue

                            unprotected_endpoints.append(
                                {
                                    "file": str(file_path),
                                    "line": line_number,
                                    "endpoint": match.group(0),
                                }
                            )

            except Exception as e:
                print(f"  ! Error analyzing {file_path}: {e}")

        if unprotected_endpoints:
            for endpoint in unprotected_endpoints:
                self.add_finding(
                    "A01: Broken Access Control",
                    "HIGH",
                    "Unprotected API endpoint",
                    f"API endpoint lacks authentication protection",
                    file_path=endpoint["file"],
                    line_number=endpoint["line"],
                    evidence=endpoint["endpoint"],
                    remediation="Add authentication decorator or dependency",
                )
        else:
            print("  ✓ No unprotected endpoints found")

    def test_a02_cryptographic_failures(self):
        """Test for A02:2021 – Cryptographic Failures."""
        print("\n[*] Testing A02: Cryptographic Failures...")

        # Check for hardcoded secrets
        app_files = []
        for app_dir in ["api", "auth", "agents", "database", "config"]:
            app_files.extend(self.project_root.glob(f"{app_dir}/**/*.py"))

        # Also check root-level config files
        app_files.extend(self.project_root.glob("*.py"))

        secret_patterns = [
            (r'secret\s*=\s*["\'][^"\']{16,}["\']', "Hardcoded secret"),
            (r'key\s*=\s*["\'][^"\']{16,}["\']', "Hardcoded key"),
            (r'password\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded password"),
            (r'token\s*=\s*["\'][^"\']{20,}["\']', "Hardcoded token"),
        ]

        for file_path in app_files:
            if not self.should_analyze_file(file_path):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                for pattern, description in secret_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        # Skip obvious placeholders
                        if any(
                            placeholder in match.group(0).lower()
                            for placeholder in [
                                "your_",
                                "example",
                                "placeholder",
                                "test_",
                                "dummy",
                                "change_me",
                            ]
                        ):
                            continue

                        line_number = content[: match.start()].count("\n") + 1

                        self.add_finding(
                            "A02: Cryptographic Failures",
                            "HIGH",
                            description,
                            f"Hardcoded secret detected in {file_path.name}",
                            file_path=str(file_path),
                            line_number=line_number,
                            evidence=match.group(0),
                            remediation="Use environment variables or secure secret management",
                        )

            except Exception:
                continue

        print("  ✓ Cryptographic failures analysis completed")

    def test_a03_injection(self):
        """Test for A03:2021 – Injection."""
        print("\n[*] Testing A03: Injection...")

        app_files = []
        for app_dir in [
            "api",
            "auth",
            "agents",
            "database",
            "knowledge_graph",
        ]:
            app_files.extend(self.project_root.glob(f"{app_dir}/**/*.py"))

        injection_patterns = [
            (
                r"cursor\.execute\([^)]*%[^)]*\)",
                "SQL injection via string formatting",
            ),
            (r"\.query\([^)]*%[^)]*\)", "SQL injection via query formatting"),
            (r"SELECT.*\+.*FROM", "SQL injection via string concatenation"),
            (r'f"SELECT.*\{[^}]*\}.*"', "SQL injection via f-string"),
            (r"f\'SELECT.*\{[^}]*\}.*\'", "SQL injection via f-string"),
        ]

        for file_path in app_files:
            if not self.should_analyze_file(file_path):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                for pattern, description in injection_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_number = content[: match.start()].count("\n") + 1

                        self.add_finding(
                            "A03: Injection",
                            "HIGH",
                            description,
                            f"Potential SQL injection vulnerability in {file_path.name}",
                            file_path=str(file_path),
                            line_number=line_number,
                            evidence=match.group(0),
                            remediation="Use parameterized queries or ORM methods",
                        )

            except Exception:
                continue

        print("  ✓ Injection analysis completed")

    def test_a04_insecure_design(self):
        """Test for A04:2021 – Insecure Design."""
        print("\n[*] Testing A04: Insecure Design...")

        # Check for rate limiting
        rate_limit_files = list(self.project_root.glob("**/rate_limit*.py"))
        rate_limit_files.extend(self.project_root.glob("**/*rate*limit*.py"))

        # Filter application files only
        app_rate_limit_files = [
            f for f in rate_limit_files if self.should_analyze_file(f)
        ]

        if not app_rate_limit_files:
            self.add_finding(
                "A04: Insecure Design",
                "MEDIUM",
                "Rate limiting implementation not found",
                "No rate limiting files found in application code",
                remediation="Implement rate limiting for API endpoints",
            )
        else:
            print(
                f"  ✓ Rate limiting implementation found ({len(app_rate_limit_files)} files)"
            )

        # Check for input validation
        validation_patterns = [
            r"pydantic\.BaseModel",
            r"from pydantic import",
            r"validate\w*\(",
            r"sanitize\w*\(",
        ]

        validation_files = []
        for app_dir in ["api", "auth", "agents"]:
            app_files = self.project_root.glob(f"{app_dir}/**/*.py")

            for file_path in app_files:
                if not self.should_analyze_file(file_path):
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    for pattern in validation_patterns:
                        if re.search(pattern, content):
                            validation_files.append(file_path)
                            break

                except Exception:
                    continue

        if not validation_files:
            self.add_finding(
                "A04: Insecure Design",
                "MEDIUM",
                "Input validation not clearly implemented",
                "No clear input validation patterns found in API endpoints",
                remediation="Implement comprehensive input validation with Pydantic",
            )
        else:
            print(
                f"  ✓ Input validation detected in {len(validation_files)} files"
            )

    def test_a05_security_misconfiguration(self):
        """Test for A05:2021 – Security Misconfiguration."""
        print("\n[*] Testing A05: Security Misconfiguration...")

        # Check for debug mode in production
        config_files = []
        config_files.extend(self.project_root.glob("*.py"))
        config_files.extend(self.project_root.glob("config/**/*.py"))

        for file_path in config_files:
            if not self.should_analyze_file(file_path):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Check for debug mode
                if re.search(r"DEBUG\s*=\s*True", content):
                    line_number = (
                        content[: content.find("DEBUG = True")].count("\n") + 1
                    )

                    self.add_finding(
                        "A05: Security Misconfiguration",
                        "MEDIUM",
                        "Debug mode enabled",
                        f"Debug mode enabled in {file_path.name}",
                        file_path=str(file_path),
                        line_number=line_number,
                        evidence="DEBUG = True",
                        remediation="Set DEBUG = False in production",
                    )

            except Exception:
                continue

        print("  ✓ Security misconfiguration analysis completed")

    def test_a06_vulnerable_components(self):
        """Test for A06:2021 – Vulnerable and Outdated Components."""
        print("\n[*] Testing A06: Vulnerable and Outdated Components...")

        # Check for requirements files
        req_files = [
            "requirements.txt",
            "requirements-core.txt",
            "requirements-dev.txt",
            "requirements-production.txt",
        ]

        found_requirements = []
        for req_file in req_files:
            if (self.project_root / req_file).exists():
                found_requirements.append(req_file)

        if found_requirements:
            print(
                f"  ✓ Found requirements files: {', '.join(found_requirements)}"
            )
        else:
            self.add_finding(
                "A06: Vulnerable Components",
                "MEDIUM",
                "No requirements files found",
                "Cannot verify dependency security without requirements files",
                remediation="Maintain requirements files for dependency tracking",
            )

        # Check for package.json
        if (self.project_root / "web" / "package.json").exists():
            print("  ✓ Found package.json for frontend dependencies")
        else:
            print("  ⚠ No package.json found")

    def test_a07_auth_failures(self):
        """Test for A07:2021 – Authentication Failures."""
        print("\n[*] Testing A07: Authentication Failures...")

        # Check for secure password hashing
        auth_files = []
        auth_files.extend(self.project_root.glob("auth/**/*.py"))
        auth_files.extend(self.project_root.glob("api/**/auth*.py"))

        secure_hashing_found = False

        for file_path in auth_files:
            if not self.should_analyze_file(file_path):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Check for secure hashing
                if re.search(r"bcrypt|scrypt|argon2|PBKDF2", content):
                    secure_hashing_found = True

                # Check for weak hashing
                weak_patterns = [
                    r"hashlib\.md5",
                    r"hashlib\.sha1",
                    r"hashlib\.sha256\([^)]*password[^)]*\)",
                ]

                for pattern in weak_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_number = content[: match.start()].count("\n") + 1

                        self.add_finding(
                            "A07: Authentication Failures",
                            "HIGH",
                            "Weak password hashing",
                            f"Weak hashing algorithm detected in {file_path.name}",
                            file_path=str(file_path),
                            line_number=line_number,
                            evidence=match.group(0),
                            remediation="Use bcrypt, scrypt, or argon2 for password hashing",
                        )

            except Exception:
                continue

        if secure_hashing_found:
            print("  ✓ Secure password hashing implementation found")
        else:
            self.add_finding(
                "A07: Authentication Failures",
                "MEDIUM",
                "No secure password hashing detected",
                "No evidence of secure password hashing in auth files",
                remediation="Implement secure password hashing with bcrypt",
            )

    def test_a08_software_integrity(self):
        """Test for A08:2021 – Software and Data Integrity Failures."""
        print("\n[*] Testing A08: Software Integrity...")

        # Check for unsafe deserialization
        app_files = []
        for app_dir in ["api", "auth", "agents", "database"]:
            app_files.extend(self.project_root.glob(f"{app_dir}/**/*.py"))

        unsafe_patterns = [
            (r"pickle\.load\(", "Unsafe pickle deserialization"),
            (r"pickle\.loads\(", "Unsafe pickle deserialization"),
            (r"eval\(", "Unsafe eval usage"),
            (r"exec\(", "Unsafe exec usage"),
        ]

        for file_path in app_files:
            if not self.should_analyze_file(file_path):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                for pattern, description in unsafe_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_number = content[: match.start()].count("\n") + 1

                        self.add_finding(
                            "A08: Software Integrity",
                            "HIGH",
                            description,
                            f"Unsafe deserialization pattern in {file_path.name}",
                            file_path=str(file_path),
                            line_number=line_number,
                            evidence=match.group(0),
                            remediation="Use safe deserialization methods",
                        )

            except Exception:
                continue

        print("  ✓ Software integrity analysis completed")

    def test_a09_logging_monitoring(self):
        """Test for A09:2021 – Security Logging and Monitoring Failures."""
        print("\n[*] Testing A09: Logging & Monitoring...")

        # Check for security logging
        security_log_files = []
        security_log_files.extend(
            self.project_root.glob("**/security_log*.py")
        )
        security_log_files.extend(self.project_root.glob("**/audit*.py"))
        security_log_files.extend(
            self.project_root.glob("observability/**/*.py")
        )

        app_log_files = [
            f for f in security_log_files if self.should_analyze_file(f)
        ]

        if app_log_files:
            print(f"  ✓ Security logging files found: {len(app_log_files)}")
        else:
            self.add_finding(
                "A09: Logging & Monitoring",
                "MEDIUM",
                "No security logging implementation found",
                "No dedicated security logging files detected",
                remediation="Implement security audit logging",
            )

        # Check for monitoring endpoints
        api_files = list(self.project_root.glob("api/**/*.py"))

        monitoring_endpoints = False
        for file_path in api_files:
            if not self.should_analyze_file(file_path):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                monitoring_patterns = [
                    r"/health",
                    r"/metrics",
                    r"/monitoring",
                    r"/status",
                ]

                for pattern in monitoring_patterns:
                    if re.search(pattern, content):
                        monitoring_endpoints = True
                        break

            except Exception:
                continue

        if monitoring_endpoints:
            print("  ✓ Monitoring endpoints detected")
        else:
            self.add_finding(
                "A09: Logging & Monitoring",
                "MEDIUM",
                "No monitoring endpoints found",
                "No health check or monitoring endpoints detected",
                remediation="Implement monitoring and health check endpoints",
            )

    def test_a10_ssrf(self):
        """Test for A10:2021 – Server-Side Request Forgery."""
        print("\n[*] Testing A10: SSRF...")

        # Check for URL requests in application code
        app_files = []
        for app_dir in ["api", "agents", "knowledge_graph"]:
            app_files.extend(self.project_root.glob(f"{app_dir}/**/*.py"))

        url_patterns = [
            r"requests\.get\(",
            r"requests\.post\(",
            r"httpx\.get\(",
            r"httpx\.post\(",
        ]

        potential_ssrf = 0
        for file_path in app_files:
            if not self.should_analyze_file(file_path):
                continue

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                for pattern in url_patterns:
                    matches = re.finditer(pattern, content)
                    for match in matches:
                        line_number = content[: match.start()].count("\n") + 1

                        # Get the full line
                        lines = content.split("\n")
                        if line_number <= len(lines):
                            line_content = lines[line_number - 1]

                            # Check if URL might be user-controlled
                            if any(
                                keyword in line_content.lower()
                                for keyword in [
                                    "request.",
                                    "user",
                                    "input",
                                    "param",
                                    "args",
                                    "form",
                                ]
                            ):
                                potential_ssrf += 1

                                self.add_finding(
                                    "A10: SSRF",
                                    "HIGH",
                                    "Potential SSRF vulnerability",
                                    f"User-controlled URL request in {file_path.name}",
                                    file_path=str(file_path),
                                    line_number=line_number,
                                    evidence=line_content.strip(),
                                    remediation="Validate and whitelist URLs before requests",
                                )

            except Exception:
                continue

        if potential_ssrf == 0:
            print("  ✓ No obvious SSRF vulnerabilities found")

    def generate_focused_report(self) -> Dict[str, Any]:
        """Generate focused assessment report."""
        severity_counts = {
            "CRITICAL": 0,
            "HIGH": 0,
            "MEDIUM": 0,
            "LOW": 0,
        }

        category_counts: Dict[str, int] = {}
        file_counts: Dict[str, int] = {}

        for finding in self.findings:
            severity_counts[finding["severity"]] += 1
            category = finding["category"]
            category_counts[category] = category_counts.get(category, 0) + 1

            if finding.get("file_path"):
                file_counts[finding["file_path"]] = (
                    file_counts.get(finding["file_path"], 0) + 1
                )

        return {
            "assessment_date": datetime.now().isoformat(),
            "assessment_type": "focused_application_analysis",
            "project_root": str(self.project_root),
            "files_analyzed": self.files_analyzed,
            "total_findings": len(self.findings),
            "severity_summary": severity_counts,
            "category_summary": category_counts,
            "file_summary": file_counts,
            "findings": self.findings,
        }

    def run_focused_assessment(self):
        """Run focused OWASP Top 10 assessment."""
        print("=" * 70)
        print("FOCUSED OWASP TOP 10 SECURITY ASSESSMENT")
        print("Task 14.11 - Application Code Analysis")
        print("=" * 70)
        print(f"Project Root: {self.project_root}")
        print(f"Focus: Application code only (excluding dependencies)")
        print("=" * 70)

        # Run all tests
        self.test_a01_broken_access_control()
        self.test_a02_cryptographic_failures()
        self.test_a03_injection()
        self.test_a04_insecure_design()
        self.test_a05_security_misconfiguration()
        self.test_a06_vulnerable_components()
        self.test_a07_auth_failures()
        self.test_a08_software_integrity()
        self.test_a09_logging_monitoring()
        self.test_a10_ssrf()

        # Generate report
        report = self.generate_focused_report()

        print("\n" + "=" * 70)
        print("FOCUSED ASSESSMENT SUMMARY")
        print("=" * 70)
        print(f"Application files analyzed: {report['files_analyzed']}")
        print(f"Total findings: {report['total_findings']}")
        print("\nFindings by severity:")
        for severity, count in report["severity_summary"].items():
            print(f"  {severity}: {count}")
        print("\nFindings by category:")
        for category, count in report["category_summary"].items():
            print(f"  {category}: {count}")

        # Save report
        report_path = (
            self.project_root
            / "security"
            / "owasp_focused_assessment_report.json"
        )
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nFocused report saved to: {report_path}")
        return report


def main():
    """Run focused OWASP assessment."""
    assessor = FocusedOWASPAssessment()
    report = assessor.run_focused_assessment()

    # Summary for task completion
    print(f"\n✅ Task 14.11 Assessment Complete")
    print(f"Files analyzed: {report['files_analyzed']}")
    print(f"Total findings: {report['total_findings']}")

    if report["severity_summary"]["HIGH"] > 0:
        print(
            f"⚠️  HIGH priority issues: {report['severity_summary']['HIGH']}"
        )
    if report["severity_summary"]["MEDIUM"] > 0:
        print(
            f"ℹ️  MEDIUM priority issues: {report['severity_summary']['MEDIUM']}"
        )


if __name__ == "__main__":
    main()
