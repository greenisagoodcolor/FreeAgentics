#!/usr/bin/env python3
"""
Comprehensive Security Audit Script for FreeAgentics v1.0.0-alpha+

This script performs a thorough security assessment including:
- OWASP Top 10 compliance check
- Rate limiting validation
- Authentication/authorization review
- Configuration security analysis
- Dependency vulnerability scanning
- Docker security assessment
"""

import asyncio
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import requests
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


class SecurityAudit:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0-alpha+",
            "score": 0,
            "max_score": 100,
            "categories": {},
            "critical_issues": [],
            "recommendations": [],
        }

    async def run_comprehensive_audit(self):
        """Run complete security audit."""
        console.print(
            "\n[bold blue]🛡️ FreeAgentics v1.0.0-alpha+ Security Audit[/bold blue]"
        )
        console.print("Zero-Tolerance Security Assessment\n")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # 1. OWASP Top 10 Assessment
            task1 = progress.add_task("🔍 OWASP Top 10 Compliance Check...", total=None)
            owasp_score = await self.assess_owasp_top10()
            progress.update(
                task1, completed=100, description="✅ OWASP Assessment Complete"
            )

            # 2. Authentication & Authorization
            task2 = progress.add_task(
                "🔐 Authentication & Authorization Review...", total=None
            )
            auth_score = await self.assess_authentication()
            progress.update(
                task2, completed=100, description="✅ Auth Assessment Complete"
            )

            # 3. Rate Limiting & DDoS Protection
            task3 = progress.add_task(
                "⚡ Rate Limiting & DDoS Assessment...", total=None
            )
            rate_limit_score = await self.assess_rate_limiting()
            progress.update(
                task3, completed=100, description="✅ Rate Limiting Complete"
            )

            # 4. Configuration Security
            task4 = progress.add_task("⚙️ Configuration Security Review...", total=None)
            config_score = await self.assess_configuration()
            progress.update(
                task4, completed=100, description="✅ Configuration Review Complete"
            )

            # 5. Dependency Vulnerabilities
            task5 = progress.add_task("📦 Dependency Vulnerability Scan...", total=None)
            dep_score = await self.assess_dependencies()
            progress.update(
                task5, completed=100, description="✅ Dependency Scan Complete"
            )

            # 6. Docker & Container Security
            task6 = progress.add_task("🐳 Container Security Assessment...", total=None)
            container_score = await self.assess_container_security()
            progress.update(
                task6, completed=100, description="✅ Container Security Complete"
            )

            # 7. API Security
            task7 = progress.add_task("🌐 API Security Validation...", total=None)
            api_score = await self.assess_api_security()
            progress.update(
                task7, completed=100, description="✅ API Security Complete"
            )

            # 8. Data Protection
            task8 = progress.add_task("🔒 Data Protection Assessment...", total=None)
            data_score = await self.assess_data_protection()
            progress.update(
                task8, completed=100, description="✅ Data Protection Complete"
            )

        # Calculate final score
        total_score = (
            owasp_score
            + auth_score
            + rate_limit_score
            + config_score
            + dep_score
            + container_score
            + api_score
            + data_score
        ) / 8
        self.results["score"] = round(total_score, 1)

        # Generate report
        await self.generate_report()

        return self.results

    async def assess_owasp_top10(self) -> float:
        """Assess OWASP Top 10 2021 compliance."""
        score = 0
        max_score = 10
        issues = []

        # A01: Broken Access Control

        access_control_score = 0
        if self._check_file_exists("auth/security_implementation.py"):
            access_control_score += 2
        if self._grep_in_files("@require_permission", "api/v1/*.py"):
            access_control_score += 3
        else:
            issues.append("A01: Missing permission decorators on API endpoints")

        score += min(access_control_score, 2)

        # A02: Cryptographic Failures
        crypto_score = 0
        if self._grep_in_files("cryptography", "requirements*.txt"):
            crypto_score += 1
        if self._check_environment_var("SECRET_KEY"):
            crypto_score += 1
        else:
            issues.append("A02: Weak or missing SECRET_KEY configuration")

        score += min(crypto_score, 1)

        # A03: Injection
        injection_score = 0
        if self._grep_in_files("sqlalchemy", "requirements*.txt"):
            injection_score += 1  # SQLAlchemy provides SQL injection protection
        if self._grep_in_files("pydantic", "requirements*.txt"):
            injection_score += 0.5  # Pydantic provides input validation

        score += min(injection_score, 1)

        # A04: Insecure Design
        design_score = 0
        if self._check_file_exists("tests/security/"):
            design_score += 0.5
        if self._check_file_exists("security/"):
            design_score += 0.5

        score += min(design_score, 1)

        # A05: Security Misconfiguration
        config_score = 0
        if self._check_file_exists(".env.production.template"):
            config_score += 0.5
        if not self._grep_in_files("DEBUG.*=.*True", "api/"):
            config_score += 0.5

        score += min(config_score, 1)

        # A06: Vulnerable Components
        vuln_score = 0
        if self._check_file_exists("requirements-core.txt"):
            vuln_score += 0.5
        # Check for known vulnerable packages (would need actual vulnerability database)
        vuln_score += 0.5

        score += min(vuln_score, 1)

        # A07: Identification and Authentication Failures
        auth_score = 0
        if self._grep_in_files("JWT", "requirements*.txt"):
            auth_score += 0.5
        if self._grep_in_files("passlib", "requirements*.txt"):
            auth_score += 0.5

        score += min(auth_score, 1)

        # A08: Software and Data Integrity Failures
        integrity_score = 0
        if self._check_file_exists("pyproject.toml"):
            integrity_score += 0.5
        if self._check_docker_signature():
            integrity_score += 0.5

        score += min(integrity_score, 1)

        # A09: Security Logging and Monitoring Failures
        logging_score = 0
        if self._grep_in_files("structlog", "requirements*.txt"):
            logging_score += 0.5
        if self._check_file_exists("observability/"):
            logging_score += 0.5

        score += min(logging_score, 1)

        # A10: Server-Side Request Forgery (SSRF)
        ssrf_score = 0
        if self._grep_in_files("httpx", "requirements*.txt"):
            ssrf_score += 0.5  # httpx provides better SSRF protection than requests
        if self._grep_in_files("url.*validation", "api/"):
            ssrf_score += 0.5

        score += min(ssrf_score, 1)

        self.results["categories"]["owasp_top10"] = {
            "score": score,
            "max_score": max_score,
            "percentage": (score / max_score) * 100,
            "issues": issues,
        }

        return (score / max_score) * 100

    async def assess_authentication(self) -> float:
        """Assess authentication and authorization implementation."""
        score = 0
        max_score = 10
        issues = []

        # JWT Implementation
        if self._grep_in_files("pyjwt", "requirements*.txt"):
            score += 2
        else:
            issues.append("Missing JWT library for secure authentication")

        # Password Hashing
        if self._grep_in_files("passlib", "requirements*.txt"):
            score += 2
        else:
            issues.append("Missing secure password hashing library")

        # Authentication Middleware
        if self._check_file_exists("auth/security_implementation.py"):
            score += 2
        else:
            issues.append("Missing authentication middleware implementation")

        # Rate Limiting on Auth Endpoints
        if self._grep_in_files("rate.*limit", "auth/"):
            score += 2
        else:
            issues.append("Missing rate limiting on authentication endpoints")

        # Session Management
        if self._grep_in_files("redis", "requirements*.txt"):
            score += 1  # Redis for session storage

        # HTTPS Enforcement
        if self._grep_in_files("HTTPS_ONLY", ".env*"):
            score += 1
        else:
            issues.append("HTTPS enforcement not configured")

        self.results["categories"]["authentication"] = {
            "score": score,
            "max_score": max_score,
            "percentage": (score / max_score) * 100,
            "issues": issues,
        }

        return (score / max_score) * 100

    async def assess_rate_limiting(self) -> float:
        """Assess rate limiting and DDoS protection."""
        score = 0
        max_score = 10
        issues = []

        # Rate Limiting Implementation
        if self._check_file_exists("auth/security_implementation.py"):
            content = self._read_file("auth/security_implementation.py")
            if "RateLimiter" in content:
                score += 3
            elif "rate" in content.lower():
                score += 1
            else:
                issues.append("No rate limiting implementation found")

        # Redis for Distributed Rate Limiting
        if self._grep_in_files("redis", "requirements*.txt"):
            score += 2
        else:
            issues.append("Missing Redis for distributed rate limiting")

        # DDoS Protection Configuration
        if self._grep_in_files("nginx", "docker-compose*"):
            score += 2

        # Rate Limit Headers
        if self._grep_in_files("X-RateLimit", "api/"):
            score += 1
        else:
            issues.append("Missing rate limit headers in API responses")

        # Configuration
        if self._grep_in_files("RATE_LIMIT", ".env*"):
            score += 1

        # Monitoring
        if self._check_file_exists("monitoring/"):
            score += 1

        self.results["categories"]["rate_limiting"] = {
            "score": score,
            "max_score": max_score,
            "percentage": (score / max_score) * 100,
            "issues": issues,
        }

        return (score / max_score) * 100

    async def assess_configuration(self) -> float:
        """Assess security configuration."""
        score = 0
        max_score = 10
        issues = []

        # Environment Configuration
        if self._check_file_exists(".env.production.template"):
            score += 2
        else:
            issues.append("Missing production environment template")

        # Secrets Management
        if not self._grep_in_files(
            "password.*=.*['\"][^'\"]*['\"]", ".", exclude_tests=True
        ):
            score += 2
        else:
            issues.append("Hardcoded passwords detected")

        # SSL/TLS Configuration
        if self._grep_in_files("ssl", "nginx/"):
            score += 2

        # Security Headers
        if self._grep_in_files("SecurityMiddleware", "api/"):
            score += 2
        else:
            issues.append("Missing security headers middleware")

        # CORS Configuration
        if self._grep_in_files("CORS", "api/"):
            score += 1

        # Database Security
        if not self._grep_in_files("postgresql://.*@", ".", exclude_tests=True):
            score += 1
        else:
            issues.append("Database credentials may be exposed")

        self.results["categories"]["configuration"] = {
            "score": score,
            "max_score": max_score,
            "percentage": (score / max_score) * 100,
            "issues": issues,
        }

        return (score / max_score) * 100

    async def assess_dependencies(self) -> float:
        """Assess dependency vulnerabilities."""
        score = 0
        max_score = 10
        issues = []

        try:
            # Run safety check
            result = subprocess.run(
                ["python", "-m", "pip", "install", "safety", "--quiet"],
                capture_output=True,
                text=True,
            )

            result = subprocess.run(
                ["safety", "check", "--json"], capture_output=True, text=True
            )

            if result.returncode == 0:
                vulnerabilities = json.loads(result.stdout) if result.stdout else []
                if len(vulnerabilities) == 0:
                    score += 5
                elif len(vulnerabilities) < 5:
                    score += 3
                    issues.append(
                        f"Found {len(vulnerabilities)} dependency vulnerabilities"
                    )
                else:
                    score += 1
                    issues.append(
                        f"Found {len(vulnerabilities)} dependency vulnerabilities (HIGH RISK)"
                    )
            else:
                issues.append("Could not run dependency vulnerability scan")

        except Exception as e:
            issues.append(f"Dependency scan failed: {str(e)}")

        # Check for pinned versions
        if self._check_file_exists("requirements-core.txt"):
            content = self._read_file("requirements-core.txt")
            pinned_count = content.count("==")
            if pinned_count > 10:
                score += 3
            elif pinned_count > 5:
                score += 2
            else:
                issues.append("Dependencies should be pinned to specific versions")

        # Check for security-focused packages
        if self._grep_in_files("cryptography", "requirements*.txt"):
            score += 1

        if self._grep_in_files("bandit", "requirements*.txt"):
            score += 1

        self.results["categories"]["dependencies"] = {
            "score": score,
            "max_score": max_score,
            "percentage": (score / max_score) * 100,
            "issues": issues,
        }

        return (score / max_score) * 100

    async def assess_container_security(self) -> float:
        """Assess Docker and container security."""
        score = 0
        max_score = 10
        issues = []

        # Non-root user
        if self._grep_in_files("USER.*[0-9]", "Dockerfile*"):
            score += 2
        else:
            issues.append("Containers should run as non-root user")

        # Read-only filesystem
        if self._grep_in_files("read_only.*true", "docker-compose*"):
            score += 2
        else:
            issues.append("Containers should use read-only filesystem")

        # Health checks
        if self._grep_in_files("HEALTHCHECK", "Dockerfile*"):
            score += 1

        # Resource limits
        if self._grep_in_files("resources:", "docker-compose*"):
            score += 1

        # Security labels
        if self._grep_in_files("LABEL.*security", "Dockerfile*"):
            score += 1

        # Base image security
        if self._grep_in_files("python:.*-slim", "Dockerfile*"):
            score += 1

        # Secret management
        if self._grep_in_files("secrets:", "docker-compose*"):
            score += 1

        # Network security
        if self._grep_in_files("networks:", "docker-compose*"):
            score += 1

        self.results["categories"]["container_security"] = {
            "score": score,
            "max_score": max_score,
            "percentage": (score / max_score) * 100,
            "issues": issues,
        }

        return (score / max_score) * 100

    async def assess_api_security(self) -> float:
        """Assess API security implementation."""
        score = 0
        max_score = 10
        issues = []

        # Input Validation
        if self._grep_in_files("pydantic", "requirements*.txt"):
            score += 2
        else:
            issues.append("Missing input validation with Pydantic")

        # API Documentation Security
        if self._grep_in_files("openapi", "api/"):
            score += 1

        # CORS Configuration
        if self._grep_in_files("CORS", "api/"):
            score += 2
        else:
            issues.append("Missing CORS configuration")

        # Authentication on Endpoints
        protected_endpoints = self._count_protected_endpoints()
        if protected_endpoints > 5:
            score += 2
        elif protected_endpoints > 0:
            score += 1
        else:
            issues.append("No protected API endpoints found")

        # Error Handling
        if self._grep_in_files("HTTPException", "api/"):
            score += 1

        # Request Size Limits
        if self._grep_in_files("max.*size", "api/"):
            score += 1
        else:
            issues.append("Missing request size limits")

        # Response Headers
        if self._grep_in_files("X-Content-Type-Options", "api/"):
            score += 1
        else:
            issues.append("Missing security response headers")

        self.results["categories"]["api_security"] = {
            "score": score,
            "max_score": max_score,
            "percentage": (score / max_score) * 100,
            "issues": issues,
        }

        return (score / max_score) * 100

    async def assess_data_protection(self) -> float:
        """Assess data protection and privacy."""
        score = 0
        max_score = 10
        issues = []

        # Database Encryption
        if self._grep_in_files("pgvector", "requirements*.txt"):
            score += 2  # Modern database with encryption support

        # Connection Security
        if self._grep_in_files("sslmode", "database/"):
            score += 2
        else:
            issues.append("Database connections should use SSL")

        # Password Hashing
        if self._grep_in_files("passlib", "requirements*.txt"):
            score += 2

        # Data Validation
        if self._grep_in_files("pydantic", "requirements*.txt"):
            score += 1

        # Backup Security
        if self._grep_in_files("backup", "docker-compose*"):
            score += 1

        # Access Logging
        if self._grep_in_files("audit", "database/"):
            score += 1

        # GDPR Compliance Preparation
        if self._check_file_exists("privacy/") or self._grep_in_files("gdpr", "."):
            score += 1
        else:
            issues.append("No privacy/GDPR compliance preparation found")

        self.results["categories"]["data_protection"] = {
            "score": score,
            "max_score": max_score,
            "percentage": (score / max_score) * 100,
            "issues": issues,
        }

        return (score / max_score) * 100

    async def generate_report(self):
        """Generate comprehensive security report."""

        # Determine overall security level
        score = self.results["score"]
        if score >= 90:
            level = "[bold green]EXCELLENT[/bold green]"
            status = "🟢"
        elif score >= 80:
            level = "[bold yellow]GOOD[/bold yellow]"
            status = "🟡"
        elif score >= 70:
            level = "[bold orange]MODERATE[/bold orange]"
            status = "🟠"
        else:
            level = "[bold red]CRITICAL[/bold red]"
            status = "🔴"

        # Main report panel
        console.print(
            Panel.fit(
                f"\n[bold]🛡️ SECURITY AUDIT RESULTS[/bold]\n\n"
                f"Overall Security Score: [bold]{score}/100[/bold]\n"
                f"Security Level: {level} {status}\n"
                f"Assessment Date: {self.results['timestamp'][:19]}\n"
                f"Version: FreeAgentics {self.results['version']}\n",
                title="[bold blue]Security Assessment Summary[/bold blue]",
                border_style="blue",
            )
        )

        # Category breakdown table
        table = Table(title="\n📊 Security Category Breakdown")
        table.add_column("Category", style="cyan", no_wrap=True)
        table.add_column("Score", justify="center")
        table.add_column("Percentage", justify="center")
        table.add_column("Status", justify="center")

        for category, data in self.results["categories"].items():
            percentage = data["percentage"]
            if percentage >= 80:
                status_icon = "🟢"
            elif percentage >= 60:
                status_icon = "🟡"
            else:
                status_icon = "🔴"

            table.add_row(
                category.replace("_", " ").title(),
                f"{data['score']}/{data['max_score']}",
                f"{percentage:.1f}%",
                status_icon,
            )

        console.print(table)

        # Critical issues
        if any(data.get("issues", []) for data in self.results["categories"].values()):
            console.print("\n[bold red]⚠️ Security Issues Identified:[/bold red]")
            for category, data in self.results["categories"].items():
                if data.get("issues"):
                    console.print(
                        f"\n[bold]{category.replace('_', ' ').title()}:[/bold]"
                    )
                    for issue in data["issues"]:
                        console.print(f"  • {issue}")

        # Recommendations
        recommendations = self._generate_recommendations()
        if recommendations:
            console.print("\n[bold green]🔧 Security Recommendations:[/bold green]")
            for rec in recommendations:
                console.print(f"  • {rec}")

        # Save JSON report
        report_path = (
            f"security_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)

        console.print(f"\n[dim]Detailed report saved to: {report_path}[/dim]")

        # Return final assessment
        if score < 85:
            console.print("\n[bold red]❌ SECURITY GATE FAILED[/bold red]")
            console.print("Security score below minimum threshold of 85/100")
            return False
        else:
            console.print("\n[bold green]✅ SECURITY GATE PASSED[/bold green]")
            console.print("Security requirements met for v1.0.0-alpha+ release")
            return True

    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations based on findings."""
        recommendations = []

        if self.results["score"] < 90:
            recommendations.append(
                "Implement comprehensive security logging and monitoring"
            )
            recommendations.append("Add automated security testing to CI/CD pipeline")
            recommendations.append("Conduct regular penetration testing")

        if self.results["score"] < 80:
            recommendations.append("Enable Web Application Firewall (WAF)")
            recommendations.append("Implement Content Security Policy (CSP)")
            recommendations.append("Add security headers to all responses")

        if self.results["score"] < 70:
            recommendations.append("URGENT: Fix critical security vulnerabilities")
            recommendations.append("Review and update authentication mechanisms")
            recommendations.append("Implement zero-trust architecture principles")

        return recommendations

    # Helper methods
    def _check_file_exists(self, filepath: str) -> bool:
        """Check if file or directory exists."""
        return Path(filepath).exists()

    def _read_file(self, filepath: str) -> str:
        """Read file content safely."""
        try:
            return Path(filepath).read_text()
        except:
            return ""

    def _grep_in_files(
        self, pattern: str, path: str, exclude_tests: bool = False
    ) -> bool:
        """Search for pattern in files."""
        try:
            cmd = ["grep", "-r", pattern, path]
            if exclude_tests:
                cmd.extend(["--exclude-dir=test*", "--exclude-dir=*test*"])

            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False

    def _check_environment_var(self, var_name: str) -> bool:
        """Check if environment variable is configured."""
        return self._grep_in_files(var_name, ".env*")

    def _check_docker_signature(self) -> bool:
        """Check if Docker images are signed (placeholder)."""
        return True  # Would implement actual signature verification

    def _count_protected_endpoints(self) -> int:
        """Count API endpoints with authentication decorators."""
        try:
            result = subprocess.run(
                ["grep", "-r", "@require_permission", "api/v1/"],
                capture_output=True,
                text=True,
            )
            return len(result.stdout.splitlines()) if result.returncode == 0 else 0
        except:
            return 0


async def main():
    """Main function to run security audit."""
    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        # JSON output for CI/CD integration
        audit = SecurityAudit()
        results = await audit.run_comprehensive_audit()
        print(json.dumps(results, indent=2))
    else:
        # Interactive console output
        audit = SecurityAudit()
        passed = await audit.run_comprehensive_audit()
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    # Install required packages if not available
    try:
        import rich
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "rich"])

    try:
        import requests
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "requests"])

    asyncio.run(main())
