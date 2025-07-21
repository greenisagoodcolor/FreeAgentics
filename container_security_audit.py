#!/usr/bin/env python3
"""Container Security and Secrets Management Audit.

This script performs a comprehensive security assessment of:
1. Docker container configurations
2. Secrets management
3. File permissions
4. Network security
5. Environment variable security
"""

import json
import os
import stat
import sys
from pathlib import Path
from typing import Dict, List, Any
import re


class ContainerSecurityAuditor:
    """Comprehensive container security auditor."""

    def __init__(self):
        """Initialize the container security auditor."""
        self.results = {
            "timestamp": __import__("time").time(),
            "audits": [],
            "summary": {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0},
        }

    def add_finding(
        self,
        category: str,
        severity: str,
        title: str,
        description: str,
        recommendation: str = "",
        details: Dict[str, Any] = None,
    ):
        """Add a security finding."""
        finding = {
            "category": category,
            "severity": severity,
            "title": title,
            "description": description,
            "recommendation": recommendation,
            "details": details or {},
        }

        self.results["audits"].append(finding)
        self.results["summary"][severity] += 1

        # Color coding for output
        color_map = {
            "critical": "\033[91m",  # Red
            "high": "\033[91m",  # Red
            "medium": "\033[93m",  # Yellow
            "low": "\033[94m",  # Blue
            "info": "\033[92m",  # Green
        }
        reset = "\033[0m"

        print(f"{color_map.get(severity, '')}{severity.upper()}{reset}: [{category}] {title}")
        if description:
            print(f"  Description: {description}")
        if recommendation:
            print(f"  Recommendation: {recommendation}")
        print()

    def audit_dockerfile_security(self):
        """Audit Dockerfile security configurations."""
        print("=== Dockerfile Security Audit ===")

        dockerfile_paths = [
            "Dockerfile.production",
            "web/Dockerfile.production",
            "Dockerfile",
            "web/Dockerfile",
        ]

        for dockerfile_path in dockerfile_paths:
            if os.path.exists(dockerfile_path):
                self._audit_single_dockerfile(dockerfile_path)

    def _audit_single_dockerfile(self, dockerfile_path: str):
        """Audit a single Dockerfile."""
        print(f"Auditing {dockerfile_path}...")

        try:
            with open(dockerfile_path, "r") as f:
                content = f.read()

            lines = content.split("\n")

            # Check for non-root user
            has_user_instruction = any(line.strip().startswith("USER ") for line in lines)
            if has_user_instruction:
                # Check if it's not root
                user_lines = [line for line in lines if line.strip().startswith("USER ")]
                latest_user = user_lines[-1].strip().split()[1] if user_lines else ""

                if latest_user not in ["root", "0"]:
                    self.add_finding(
                        "Container Security",
                        "info",
                        "Non-root user configured",
                        f"Container runs as user: {latest_user}",
                        "Good security practice - maintain this configuration",
                        {"dockerfile": dockerfile_path, "user": latest_user},
                    )
                else:
                    self.add_finding(
                        "Container Security",
                        "high",
                        "Container runs as root",
                        f"Container configured to run as root user in {dockerfile_path}",
                        "Configure container to run as non-root user for better security",
                        {"dockerfile": dockerfile_path},
                    )
            else:
                self.add_finding(
                    "Container Security",
                    "high",
                    "No USER instruction found",
                    f"Dockerfile {dockerfile_path} doesn't specify non-root user",
                    "Add USER instruction to run container as non-root user",
                    {"dockerfile": dockerfile_path},
                )

            # Check for package manager cache cleanup
            has_cleanup = any(
                "rm -rf /var/lib/apt/lists/*" in line
                or "apt-get clean" in line
                or "apk del" in line
                for line in lines
            )

            if has_cleanup:
                self.add_finding(
                    "Container Security",
                    "info",
                    "Package cache cleanup present",
                    "Dockerfile includes package manager cache cleanup",
                    "Good practice - reduces image size and attack surface",
                    {"dockerfile": dockerfile_path},
                )
            else:
                self.add_finding(
                    "Container Security",
                    "medium",
                    "Package cache not cleaned",
                    f"Dockerfile {dockerfile_path} doesn't clean package manager cache",
                    "Add package cache cleanup to reduce image size and attack surface",
                    {"dockerfile": dockerfile_path},
                )

            # Check for HEALTHCHECK
            has_healthcheck = any(line.strip().startswith("HEALTHCHECK") for line in lines)
            if has_healthcheck:
                self.add_finding(
                    "Container Security",
                    "info",
                    "Health check configured",
                    "Container includes health check configuration",
                    "Good practice - maintain health check configuration",
                    {"dockerfile": dockerfile_path},
                )
            else:
                self.add_finding(
                    "Container Security",
                    "low",
                    "No health check configured",
                    f"Dockerfile {dockerfile_path} doesn't include HEALTHCHECK",
                    "Consider adding HEALTHCHECK for better container monitoring",
                    {"dockerfile": dockerfile_path},
                )

            # Check for secrets in Dockerfile
            secret_patterns = [
                r"(?i)(password|pwd|secret|key|token|api[_-]?key)",
                r"[A-Za-z0-9+/]{20,}={0,2}",  # Base64-like strings
                r"[0-9a-fA-F]{32,}",  # Hex strings
                r"sk_[a-zA-Z0-9]{24,}",  # API keys
                r"pk_[a-zA-Z0-9]{24,}",  # Public keys
            ]

            for i, line in enumerate(lines, 1):
                for pattern in secret_patterns:
                    if re.search(pattern, line) and not line.strip().startswith("#"):
                        self.add_finding(
                            "Secrets Management",
                            "critical",
                            "Potential secret in Dockerfile",
                            f"Line {i} in {dockerfile_path} may contain sensitive data",
                            "Remove secrets from Dockerfile and use environment variables or secret management",
                            {
                                "dockerfile": dockerfile_path,
                                "line": i,
                                "content": line.strip()[:50] + "...",
                            },
                        )
                        break

        except Exception as e:
            self.add_finding(
                "Container Security",
                "medium",
                f"Failed to audit {dockerfile_path}",
                f"Error reading Dockerfile: {e}",
                f"Ensure {dockerfile_path} is readable and properly formatted",
            )

    def audit_docker_compose_security(self):
        """Audit Docker Compose security configurations."""
        print("=== Docker Compose Security Audit ===")

        compose_files = [
            "docker-compose.production.yml",
            "docker-compose.yml",
            "docker-compose.override.yml",
        ]

        for compose_file in compose_files:
            if os.path.exists(compose_file):
                self._audit_docker_compose_file(compose_file)

    def _audit_docker_compose_file(self, compose_file: str):
        """Audit a Docker Compose file."""
        print(f"Auditing {compose_file}...")

        try:
            with open(compose_file, "r") as f:
                content = f.read()

            # Check for privileged containers
            if "privileged: true" in content:
                self.add_finding(
                    "Container Security",
                    "critical",
                    "Privileged container detected",
                    f"Privileged containers found in {compose_file}",
                    "Remove privileged: true unless absolutely necessary",
                    {"compose_file": compose_file},
                )

            # Check for host network mode
            if "network_mode: host" in content:
                self.add_finding(
                    "Network Security",
                    "high",
                    "Host network mode detected",
                    f"Host network mode found in {compose_file}",
                    "Use bridge networks instead of host mode for better isolation",
                    {"compose_file": compose_file},
                )

            # Check for read-only root filesystem
            if "read_only: true" in content:
                self.add_finding(
                    "Container Security",
                    "info",
                    "Read-only filesystem configured",
                    "Containers configured with read-only root filesystem",
                    "Good security practice - maintain this configuration",
                    {"compose_file": compose_file},
                )

            # Check for resource limits
            if "deploy:" in content and ("limits:" in content or "reservations:" in content):
                self.add_finding(
                    "Resource Security",
                    "info",
                    "Resource limits configured",
                    "Resource limits configured for containers",
                    "Good practice - helps prevent resource exhaustion attacks",
                    {"compose_file": compose_file},
                )
            else:
                self.add_finding(
                    "Resource Security",
                    "medium",
                    "No resource limits configured",
                    f"No resource limits found in {compose_file}",
                    "Configure memory and CPU limits to prevent resource exhaustion",
                    {"compose_file": compose_file},
                )

            # Check for environment variable files
            env_file_pattern = r"env_file:\s*-?\s*([^\s\n]+)"
            env_files = re.findall(env_file_pattern, content)

            for env_file in env_files:
                if os.path.exists(env_file):
                    self._audit_env_file(env_file)
                else:
                    self.add_finding(
                        "Configuration",
                        "medium",
                        "Missing environment file",
                        f"Environment file {env_file} referenced but not found",
                        f"Create {env_file} or remove reference from compose file",
                        {"compose_file": compose_file, "env_file": env_file},
                    )

        except Exception as e:
            self.add_finding(
                "Container Security",
                "medium",
                f"Failed to audit {compose_file}",
                f"Error reading compose file: {e}",
                f"Ensure {compose_file} is readable and properly formatted",
            )

    def _audit_env_file(self, env_file: str):
        """Audit environment file for sensitive data."""
        print(f"Auditing environment file: {env_file}")

        try:
            with open(env_file, "r") as f:
                lines = f.readlines()

            # Check file permissions
            file_stat = os.stat(env_file)
            file_mode = file_stat.st_mode

            # Check if file is readable by others
            if file_mode & stat.S_IROTH:
                self.add_finding(
                    "File Permissions",
                    "high",
                    "Environment file readable by others",
                    f"{env_file} is readable by all users",
                    f"Set restrictive permissions: chmod 600 {env_file}",
                    {"file": env_file, "permissions": oct(file_mode)},
                )

            # Check for sensitive variables without proper values
            sensitive_vars = ["PASSWORD", "SECRET", "KEY", "TOKEN", "API_KEY"]

            for i, line in enumerate(lines, 1):
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        var_name, var_value = line.split("=", 1)
                        var_name = var_name.strip()
                        var_value = var_value.strip()

                        # Check for sensitive variables with default/weak values
                        if any(sensitive in var_name.upper() for sensitive in sensitive_vars):
                            if var_value in [
                                "",
                                "changeme",
                                "password",
                                "secret",
                                "123456",
                                "admin",
                            ]:
                                self.add_finding(
                                    "Secrets Management",
                                    "critical",
                                    "Weak or default credential",
                                    f"Sensitive variable {var_name} has weak/default value",
                                    "Set strong, unique values for all sensitive variables",
                                    {"file": env_file, "line": i, "variable": var_name},
                                )

        except Exception as e:
            self.add_finding(
                "File Security",
                "medium",
                f"Failed to audit {env_file}",
                f"Error reading environment file: {e}",
                f"Ensure {env_file} is readable",
            )

    def audit_secrets_management(self):
        """Audit secrets management implementation."""
        print("=== Secrets Management Audit ===")

        secrets_dir = Path("secrets")
        auth_keys_dir = Path("auth/keys")

        # Check secrets directory
        if secrets_dir.exists():
            self.add_finding(
                "Secrets Management",
                "info",
                "Secrets directory exists",
                "Dedicated secrets directory found",
                "Ensure proper access controls and encryption for secrets",
                {"directory": str(secrets_dir)},
            )

            # Check permissions on secrets directory
            dir_stat = secrets_dir.stat()
            if dir_stat.st_mode & stat.S_IROTH or dir_stat.st_mode & stat.S_IWOTH:
                self.add_finding(
                    "File Permissions",
                    "high",
                    "Secrets directory accessible by others",
                    "Secrets directory has overly permissive permissions",
                    "Set restrictive permissions: chmod 700 secrets/",
                    {"directory": str(secrets_dir), "permissions": oct(dir_stat.st_mode)},
                )

            # Audit individual secret files
            for secret_file in secrets_dir.glob("*"):
                if secret_file.is_file():
                    self._audit_secret_file(secret_file)

        # Check auth keys directory
        if auth_keys_dir.exists():
            for key_file in auth_keys_dir.glob("*.pem"):
                self._audit_key_file(key_file)

    def _audit_secret_file(self, secret_file: Path):
        """Audit individual secret file."""
        try:
            file_stat = secret_file.stat()

            # Check permissions
            if file_stat.st_mode & stat.S_IROTH or file_stat.st_mode & stat.S_IWOTH:
                self.add_finding(
                    "File Permissions",
                    "critical",
                    "Secret file accessible by others",
                    f"Secret file {secret_file.name} has overly permissive permissions",
                    f"Set restrictive permissions: chmod 600 {secret_file}",
                    {"file": str(secret_file), "permissions": oct(file_stat.st_mode)},
                )

            # Check file size (empty files might indicate missing secrets)
            if file_stat.st_size == 0:
                self.add_finding(
                    "Secrets Management",
                    "medium",
                    "Empty secret file",
                    f"Secret file {secret_file.name} is empty",
                    "Ensure secret files contain proper values",
                    {"file": str(secret_file)},
                )

        except Exception as e:
            self.add_finding(
                "File Security",
                "medium",
                f"Failed to audit secret file {secret_file.name}",
                f"Error accessing secret file: {e}",
                "Check file permissions and existence",
            )

    def _audit_key_file(self, key_file: Path):
        """Audit cryptographic key file."""
        try:
            file_stat = key_file.stat()

            # Check permissions - private keys should be very restrictive
            if "private" in key_file.name.lower():
                if file_stat.st_mode & (stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH):
                    self.add_finding(
                        "Cryptography",
                        "critical",
                        "Private key accessible by others",
                        f"Private key {key_file.name} has overly permissive permissions",
                        f"Set restrictive permissions: chmod 600 {key_file}",
                        {"file": str(key_file), "permissions": oct(file_stat.st_mode)},
                    )
                else:
                    self.add_finding(
                        "Cryptography",
                        "info",
                        "Private key properly secured",
                        f"Private key {key_file.name} has appropriate permissions",
                        "Good security practice - maintain these permissions",
                        {"file": str(key_file)},
                    )

            # Check key content format
            try:
                with open(key_file, "r") as f:
                    content = f.read()

                if "BEGIN" in content and "END" in content:
                    self.add_finding(
                        "Cryptography",
                        "info",
                        "Valid key format detected",
                        f"Key file {key_file.name} appears to be properly formatted",
                        "Ensure keys are rotated regularly",
                        {"file": str(key_file)},
                    )
                else:
                    self.add_finding(
                        "Cryptography",
                        "medium",
                        "Invalid key format",
                        f"Key file {key_file.name} doesn't appear to be properly formatted",
                        "Verify key file contains valid cryptographic material",
                        {"file": str(key_file)},
                    )
            except:
                pass  # If we can't read the file, permissions are likely correct

        except Exception as e:
            self.add_finding(
                "Cryptography",
                "medium",
                f"Failed to audit key file {key_file.name}",
                f"Error accessing key file: {e}",
                "Check file permissions and existence",
            )

    def audit_network_security(self):
        """Audit network security configuration."""
        print("=== Network Security Audit ===")

        # Check nginx configuration
        nginx_configs = [
            "nginx/nginx.conf",
            "nginx/conf.d/ssl-freeagentics.conf",
            "nginx/snippets/ssl-params.conf",
        ]

        for config_file in nginx_configs:
            if os.path.exists(config_file):
                self._audit_nginx_config(config_file)

    def _audit_nginx_config(self, config_file: str):
        """Audit nginx configuration for security."""
        print(f"Auditing nginx config: {config_file}")

        try:
            with open(config_file, "r") as f:
                content = f.read()

            # Check for SSL/TLS configuration
            if "ssl_certificate" in content:
                self.add_finding(
                    "Network Security",
                    "info",
                    "SSL/TLS configured",
                    f"SSL configuration found in {config_file}",
                    "Ensure SSL certificates are valid and regularly updated",
                    {"config_file": config_file},
                )

            # Check for security headers
            security_headers = [
                "X-Frame-Options",
                "X-Content-Type-Options",
                "X-XSS-Protection",
                "Strict-Transport-Security",
                "Content-Security-Policy",
            ]

            missing_headers = []
            for header in security_headers:
                if header not in content:
                    missing_headers.append(header)

            if missing_headers:
                self.add_finding(
                    "Network Security",
                    "medium",
                    "Missing security headers",
                    f"Security headers missing from {config_file}: {', '.join(missing_headers)}",
                    "Add missing security headers to nginx configuration",
                    {"config_file": config_file, "missing_headers": missing_headers},
                )

            # Check for server signature hiding
            if "server_tokens off" in content:
                self.add_finding(
                    "Network Security",
                    "info",
                    "Server tokens disabled",
                    "Server signature is hidden",
                    "Good security practice - maintain this configuration",
                    {"config_file": config_file},
                )

        except Exception as e:
            self.add_finding(
                "Network Security",
                "medium",
                f"Failed to audit {config_file}",
                f"Error reading nginx config: {e}",
                f"Ensure {config_file} is readable",
            )

    def generate_security_score(self) -> float:
        """Calculate overall security score."""
        summary = self.results["summary"]
        total_findings = sum(summary.values())

        if total_findings == 0:
            return 100.0

        # Weighted scoring
        weighted_score = (
            summary["critical"] * -25
            + summary["high"] * -15
            + summary["medium"] * -8
            + summary["low"] * -3
            + summary["info"] * 2  # Positive findings
        )

        base_score = 100.0
        final_score = max(0.0, base_score + weighted_score)

        return final_score

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive security audit report."""
        security_score = self.generate_security_score()

        # Determine rating
        if security_score >= 95:
            rating = "EXCELLENT"
            status = "🟢"
        elif security_score >= 85:
            rating = "GOOD"
            status = "🟡"
        elif security_score >= 70:
            rating = "ACCEPTABLE"
            status = "🟡"
        else:
            rating = "NEEDS IMPROVEMENT"
            status = "🔴"

        report = {
            "timestamp": self.results["timestamp"],
            "security_score": security_score,
            "security_rating": rating,
            "status_indicator": status,
            "summary": self.results["summary"],
            "total_findings": sum(self.results["summary"].values()),
            "findings": self.results["audits"],
            "recommendations": self._generate_recommendations(),
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate security recommendations."""
        recommendations = []

        critical_findings = [f for f in self.results["audits"] if f["severity"] == "critical"]
        high_findings = [f for f in self.results["audits"] if f["severity"] == "high"]

        if critical_findings:
            recommendations.append("🔴 CRITICAL: Address all critical security issues immediately")

        if high_findings:
            recommendations.append("🟡 HIGH: Resolve high-priority security issues")

        # Category-specific recommendations
        categories = set(f["category"] for f in self.results["audits"])

        if "Secrets Management" in categories:
            recommendations.append("🔐 Review and strengthen secrets management practices")

        if "Container Security" in categories:
            recommendations.append("📦 Enhance container security configurations")

        if "Network Security" in categories:
            recommendations.append("🌐 Improve network security and SSL/TLS settings")

        if not critical_findings and not high_findings:
            recommendations.append("✅ Security posture is strong - continue monitoring")

        return recommendations

    def run_full_audit(self):
        """Run complete container security audit."""
        print("🔒 FreeAgentics Container Security Audit")
        print("=" * 50)

        self.audit_dockerfile_security()
        self.audit_docker_compose_security()
        self.audit_secrets_management()
        self.audit_network_security()

        return self.generate_report()


def main():
    """Main execution function."""
    auditor = ContainerSecurityAuditor()
    report = auditor.run_full_audit()

    # Print summary
    print(f"\n{'='*50}")
    print(f"🔒 CONTAINER SECURITY AUDIT REPORT {report['status_indicator']}")
    print(f"{'='*50}")
    print(f"Security Score: {report['security_score']:.1f}/100")
    print(f"Security Rating: {report['security_rating']}")
    print(f"Total Findings: {report['total_findings']}")
    print(f"Critical: {report['summary']['critical']}")
    print(f"High: {report['summary']['high']}")
    print(f"Medium: {report['summary']['medium']}")
    print(f"Low: {report['summary']['low']}")
    print(f"Info: {report['summary']['info']}")

    if report["recommendations"]:
        print("\nRecommendations:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")

    # Save report
    report_file = f"container_security_audit_{int(__import__('time').time())}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nDetailed report saved to: {report_file}")

    return report["security_score"] >= 85


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
