#!/usr/bin/env python3
"""
Security Validation Script for Production Readiness
Task 21: Final Security Audit and Penetration Testing

This script performs comprehensive security validation including:
- Security configuration validation
- Penetration testing simulation
- Vulnerability scanning
- Security best practices verification
"""

import hashlib
import json
import logging
import os
import stat
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SecurityValidator:
    """Comprehensive security validation suite"""

    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "environment": "production",
            "security_audit": {},
            "vulnerabilities": [],
            "security_score": 0,
            "recommendations": [],
            "compliance_status": {},
        }
        self.critical_issues = []
        self.security_warnings = []
        self.security_passes = []

    def log_security_result(self, category: str, test_name: str, result: Dict[str, Any]):
        """Log security test result"""
        if category not in self.results["security_audit"]:
            self.results["security_audit"][category] = {}
        self.results["security_audit"][category][test_name] = result

        severity = result.get("severity", "INFO")
        message = result.get("message", "No message")

        if severity == "CRITICAL":
            self.critical_issues.append(f"{category}: {test_name}")
            logger.error(f"CRITICAL: {category} - {test_name}: {message}")
        elif severity == "HIGH":
            self.security_warnings.append(f"{category}: {test_name}")
            logger.warning(f"HIGH: {category} - {test_name}: {message}")
        elif severity == "MEDIUM":
            self.security_warnings.append(f"{category}: {test_name}")
            logger.warning(f"MEDIUM: {category} - {test_name}: {message}")
        elif severity == "LOW":
            logger.info(f"LOW: {category} - {test_name}: {message}")
        else:
            self.security_passes.append(f"{category}: {test_name}")
            logger.info(f"PASS: {category} - {test_name}: {message}")

    def validate_file_permissions(self):
        """Validate file permissions for security-sensitive files"""
        logger.info("Validating file permissions...")

        # Critical files that should have restricted permissions
        critical_files = {
            ".env.production": 0o600,
            "auth/keys/jwt_private.pem": 0o600,
            "auth/keys/jwt_public.pem": 0o644,
            "nginx/ssl/key.pem": 0o600,
            "nginx/ssl/cert.pem": 0o644,
            "secrets/postgres_password.txt": 0o600,
            "secrets/redis_password.txt": 0o600,
            "secrets/secret_key.txt": 0o600,
        }

        permission_issues = []
        for file_path, expected_perms in critical_files.items():
            if os.path.exists(file_path):
                actual_perms = stat.S_IMODE(os.stat(file_path).st_mode)
                if actual_perms != expected_perms:
                    permission_issues.append(
                        {
                            "file": file_path,
                            "expected": oct(expected_perms),
                            "actual": oct(actual_perms),
                        }
                    )

        self.log_security_result(
            "permissions",
            "file_permissions",
            {
                "severity": "CRITICAL" if permission_issues else "PASS",
                "issues": permission_issues,
                "message": f"File permission issues: {len(permission_issues)}",
            },
        )

    def validate_secret_strength(self):
        """Validate strength of secrets and passwords"""
        logger.info("Validating secret strength...")

        weak_secrets = []

        # Check .env.production for weak secrets
        if os.path.exists(".env.production"):
            with open(".env.production", "r") as f:
                content = f.read()

            # Check for weak patterns
            weak_patterns = [
                ("password123", "Weak password detected"),
                ("admin", "Default admin credentials"),
                ("secret123", "Weak secret key"),
                ("your_", "Placeholder values not replaced"),
                ("dev_", "Development secrets in production"),
                ("localhost", "Local configuration in production"),
            ]

            for pattern, description in weak_patterns:
                if pattern in content.lower():
                    weak_secrets.append(description)

        # Check JWT secret strength
        if os.path.exists("auth/keys/jwt_private.pem"):
            with open("auth/keys/jwt_private.pem", "r") as f:
                jwt_key = f.read()

            if len(jwt_key) < 2048:  # RSA key should be at least 2048 bits
                weak_secrets.append("JWT private key may be too short")

        self.log_security_result(
            "secrets",
            "secret_strength",
            {
                "severity": "HIGH" if weak_secrets else "PASS",
                "weak_secrets": weak_secrets,
                "message": f"Weak secrets found: {len(weak_secrets)}",
            },
        )

    def validate_docker_security(self):
        """Validate Docker security configuration"""
        logger.info("Validating Docker security...")

        security_issues = []

        # Check docker-compose.production.yml
        if os.path.exists("docker-compose.production.yml"):
            with open("docker-compose.production.yml", "r") as f:
                content = f.read()

            # Check for security best practices
            security_checks = {
                "privileged_containers": "privileged: true" in content,
                "root_user": 'user: "0:0"' in content or "user: root" in content,
                "host_network": "network_mode: host" in content,
                "no_read_only": "read_only: true" not in content,
                "no_resource_limits": "limits:" not in content,
            }

            for check, is_issue in security_checks.items():
                if is_issue:
                    security_issues.append(check)

        self.log_security_result(
            "docker",
            "container_security",
            {
                "severity": "HIGH" if security_issues else "PASS",
                "security_issues": security_issues,
                "message": f"Docker security issues: {len(security_issues)}",
            },
        )

    def validate_ssl_security(self):
        """Validate SSL/TLS security configuration"""
        logger.info("Validating SSL/TLS security...")

        ssl_issues = []

        # Check nginx SSL configuration
        if os.path.exists("nginx/nginx.conf"):
            with open("nginx/nginx.conf", "r") as f:
                content = f.read()

            # Check for SSL best practices
            ssl_checks = {
                "weak_protocols": any(
                    protocol in content for protocol in ["SSLv2", "SSLv3", "TLSv1", "TLSv1.1"]
                ),
                "weak_ciphers": any(cipher in content for cipher in ["RC4", "DES", "3DES", "MD5"]),
                "no_hsts": "Strict-Transport-Security" not in content,
                "no_ocsp": "ssl_stapling" not in content,
                "weak_dh": "ssl_dhparam" not in content,
            }

            for check, is_issue in ssl_checks.items():
                if is_issue:
                    ssl_issues.append(check)

        self.log_security_result(
            "ssl",
            "ssl_configuration",
            {
                "severity": "HIGH" if ssl_issues else "PASS",
                "ssl_issues": ssl_issues,
                "message": f"SSL security issues: {len(ssl_issues)}",
            },
        )

    def validate_api_security(self):
        """Validate API security configuration"""
        logger.info("Validating API security...")

        api_issues = []

        # Check for security middleware
        middleware_files = [
            "api/middleware/security_headers.py",
            "api/middleware/rate_limiter.py",
            "api/middleware/security_monitoring.py",
        ]

        missing_middleware = [f for f in middleware_files if not os.path.exists(f)]
        if missing_middleware:
            api_issues.extend(missing_middleware)

        # Check main API configuration
        if os.path.exists("api/main.py"):
            with open("api/main.py", "r") as f:
                content = f.read()

            # Check for security features
            security_features = {
                "cors_middleware": "CORSMiddleware" in content,
                "rate_limiting": "RateLimiter" in content,
                "security_headers": "SecurityHeaders" in content,
                "csrf_protection": "csrf" in content.lower(),
                "input_validation": "validate" in content.lower(),
            }

            missing_features = [
                feature for feature, present in security_features.items() if not present
            ]
            if missing_features:
                api_issues.extend(missing_features)

        self.log_security_result(
            "api",
            "api_security",
            {
                "severity": "HIGH" if api_issues else "PASS",
                "api_issues": api_issues,
                "message": f"API security issues: {len(api_issues)}",
            },
        )

    def validate_database_security(self):
        """Validate database security configuration"""
        logger.info("Validating database security...")

        db_issues = []

        # Check PostgreSQL configuration
        if os.path.exists("postgres/postgresql-production.conf"):
            with open("postgres/postgresql-production.conf", "r") as f:
                content = f.read()

            # Check for security settings
            security_settings = {
                "ssl_enabled": "ssl = on" in content,
                "log_connections": "log_connections = on" in content,
                "log_statement": "log_statement" in content,
                "password_encryption": "password_encryption" in content,
            }

            missing_settings = [
                setting for setting, present in security_settings.items() if not present
            ]
            if missing_settings:
                db_issues.extend(missing_settings)

        # Check database initialization scripts
        if os.path.exists("postgres/init/01-init-secure.sql"):
            with open("postgres/init/01-init-secure.sql", "r") as f:
                content = f.read()

            # Check for security measures
            if "CREATE ROLE" not in content:
                db_issues.append("No role-based access control setup")
            if "GRANT" not in content:
                db_issues.append("No explicit permissions granted")

        self.log_security_result(
            "database",
            "database_security",
            {
                "severity": "MEDIUM" if db_issues else "PASS",
                "db_issues": db_issues,
                "message": f"Database security issues: {len(db_issues)}",
            },
        )

    def validate_authentication_security(self):
        """Validate authentication security"""
        logger.info("Validating authentication security...")

        auth_issues = []

        # Check authentication implementation
        auth_files = [
            "auth/security_implementation.py",
            "auth/jwt_handler.py",
            "auth/rbac_security_enhancements.py",
        ]

        for auth_file in auth_files:
            if os.path.exists(auth_file):
                with open(auth_file, "r") as f:
                    content = f.read()

                # Check for security features
                security_features = {
                    "password_hashing": any(
                        term in content for term in ["bcrypt", "scrypt", "pbkdf2"]
                    ),
                    "rate_limiting": "rate_limit" in content.lower(),
                    "session_management": "session" in content.lower(),
                    "jwt_validation": "jwt" in content.lower() and "verify" in content.lower(),
                }

                missing_features = [
                    feature for feature, present in security_features.items() if not present
                ]
                if missing_features:
                    auth_issues.extend([f"{auth_file}: {feature}" for feature in missing_features])

        self.log_security_result(
            "authentication",
            "auth_security",
            {
                "severity": "HIGH" if auth_issues else "PASS",
                "auth_issues": auth_issues,
                "message": f"Authentication security issues: {len(auth_issues)}",
            },
        )

    def validate_monitoring_security(self):
        """Validate monitoring and logging security"""
        logger.info("Validating monitoring security...")

        monitoring_issues = []

        # Check security monitoring
        if os.path.exists("observability/security_monitoring.py"):
            with open("observability/security_monitoring.py", "r") as f:
                content = f.read()

            # Check for security monitoring features
            security_features = {
                "intrusion_detection": "intrusion" in content.lower(),
                "anomaly_detection": "anomaly" in content.lower(),
                "security_alerts": "alert" in content.lower(),
                "audit_logging": "audit" in content.lower(),
            }

            missing_features = [
                feature for feature, present in security_features.items() if not present
            ]
            if missing_features:
                monitoring_issues.extend(missing_features)

        # Check log security
        if os.path.exists("logs/security_audit.log"):
            # Check if log file has appropriate permissions
            log_perms = stat.S_IMODE(os.stat("logs/security_audit.log").st_mode)
            if log_perms & 0o044:  # World or group readable
                monitoring_issues.append("Security log file has overly permissive permissions")

        self.log_security_result(
            "monitoring",
            "monitoring_security",
            {
                "severity": "MEDIUM" if monitoring_issues else "PASS",
                "monitoring_issues": monitoring_issues,
                "message": f"Monitoring security issues: {len(monitoring_issues)}",
            },
        )

    def validate_security_tests(self):
        """Validate security test coverage"""
        logger.info("Validating security test coverage...")

        test_coverage = []

        # Check for security test categories
        security_test_categories = {
            "authentication": "tests/security/test_authentication_*.py",
            "authorization": "tests/security/test_authorization_*.py",
            "input_validation": "tests/security/test_*_validation*.py",
            "crypto": "tests/security/test_crypto*.py",
            "penetration": "tests/security/test_*penetration*.py",
        }

        for category, pattern in security_test_categories.items():
            test_files = list(Path(".").glob(pattern))
            if test_files:
                test_coverage.append(f"{category}: {len(test_files)} tests")
            else:
                test_coverage.append(f"{category}: 0 tests")

        total_security_tests = (
            len(list(Path("tests/security").glob("test_*.py")))
            if Path("tests/security").exists()
            else 0
        )

        self.log_security_result(
            "testing",
            "security_test_coverage",
            {
                "severity": "MEDIUM" if total_security_tests < 20 else "PASS",
                "total_tests": total_security_tests,
                "test_coverage": test_coverage,
                "message": f"Security tests: {total_security_tests} found",
            },
        )

    def simulate_penetration_test(self):
        """Simulate basic penetration testing"""
        logger.info("Simulating penetration testing...")

        vulnerabilities = []

        # Check for common vulnerabilities
        vuln_checks = {
            "default_credentials": self.check_default_credentials(),
            "exposed_debug_info": self.check_debug_exposure(),
            "insecure_configs": self.check_insecure_configs(),
            "weak_encryption": self.check_weak_encryption(),
        }

        for vuln_type, issues in vuln_checks.items():
            if issues:
                vulnerabilities.extend(issues)

        # Simulate OWASP Top 10 checks
        owasp_results = self.simulate_owasp_checks()
        vulnerabilities.extend(owasp_results)

        self.log_security_result(
            "penetration",
            "vulnerability_scan",
            {
                "severity": "CRITICAL" if vulnerabilities else "PASS",
                "vulnerabilities": vulnerabilities,
                "message": f"Vulnerabilities found: {len(vulnerabilities)}",
            },
        )

        return vulnerabilities

    def check_default_credentials(self):
        """Check for default credentials"""
        issues = []

        # Check for common default credentials in config files
        default_patterns = ["admin:admin", "root:root", "admin:password", "user:user", "test:test"]

        config_files = [".env.production", "docker-compose.production.yml"]

        for config_file in config_files:
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    content = f.read()

                for pattern in default_patterns:
                    if pattern in content:
                        issues.append(f"Default credentials found in {config_file}: {pattern}")

        return issues

    def check_debug_exposure(self):
        """Check for debug information exposure"""
        issues = []

        # Check for debug settings in production
        if os.path.exists(".env.production"):
            with open(".env.production", "r") as f:
                content = f.read()

            if "DEBUG=true" in content or "DEBUG=True" in content:
                issues.append("Debug mode enabled in production")

        # Check for debug endpoints
        if os.path.exists("api/main.py"):
            with open("api/main.py", "r") as f:
                content = f.read()

            if "/debug" in content or "debug=True" in content:
                issues.append("Debug endpoints exposed")

        return issues

    def check_insecure_configs(self):
        """Check for insecure configurations"""
        issues = []

        # Check for insecure headers
        if os.path.exists("nginx/nginx.conf"):
            with open("nginx/nginx.conf", "r") as f:
                content = f.read()

            if "server_tokens on" in content:
                issues.append("Server tokens exposed in nginx")

        # Check for insecure CORS
        if os.path.exists("api/main.py"):
            with open("api/main.py", "r") as f:
                content = f.read()

            if 'allow_origins=["*"]' in content:
                issues.append("Insecure CORS configuration (allows all origins)")

        return issues

    def check_weak_encryption(self):
        """Check for weak encryption algorithms"""
        issues = []

        # Check for weak algorithms in code (only in security-related files)
        weak_algorithms = ["md5", "sha1", "des", "rc4"]

        security_files = list(Path("auth").glob("**/*.py")) + list(Path("api").glob("**/*.py"))
        for file_path in security_files:
            if file_path.is_file():
                try:
                    with open(file_path, "r") as f:
                        content = f.read().lower()

                    for weak_alg in weak_algorithms:
                        if weak_alg in content and "import" in content:
                            issues.append(
                                f"Weak encryption algorithm {weak_alg} found in {file_path}"
                            )
                            break
                except:
                    pass

        return issues

    def simulate_owasp_checks(self):
        """Simulate OWASP Top 10 vulnerability checks"""
        vulnerabilities = []

        owasp_checks = {
            "A01_Broken_Access_Control": self.check_access_control(),
            "A02_Cryptographic_Failures": self.check_crypto_failures(),
            "A03_Injection": self.check_injection_vulnerabilities(),
            "A04_Insecure_Design": self.check_insecure_design(),
            "A05_Security_Misconfiguration": self.check_security_misconfig(),
            "A06_Vulnerable_Components": self.check_vulnerable_components(),
            "A07_Authentication_Failures": self.check_auth_failures(),
            "A08_Software_Integrity_Failures": self.check_integrity_failures(),
            "A09_Logging_Failures": self.check_logging_failures(),
            "A10_SSRF": self.check_ssrf_vulnerabilities(),
        }

        for owasp_id, issues in owasp_checks.items():
            if issues:
                vulnerabilities.extend([f"{owasp_id}: {issue}" for issue in issues])

        return vulnerabilities

    def check_access_control(self):
        """Check for broken access control"""
        issues = []

        # Check for RBAC implementation
        if not os.path.exists("auth/rbac_security_enhancements.py"):
            issues.append("No RBAC implementation found")

        # Check for authorization middleware
        if not os.path.exists("api/middleware/authorization.py"):
            issues.append("No authorization middleware found")

        return issues

    def check_crypto_failures(self):
        """Check for cryptographic failures"""
        issues = []

        # Check for proper key management
        if not os.path.exists("auth/keys/"):
            issues.append("No key management directory found")

        # Check for secure random generation
        if os.path.exists("auth/security_implementation.py"):
            with open("auth/security_implementation.py", "r") as f:
                content = f.read()

            if "secrets.SystemRandom" not in content and "os.urandom" not in content:
                issues.append("No secure random generation detected")

        return issues

    def check_injection_vulnerabilities(self):
        """Check for injection vulnerabilities"""
        issues = []

        # Check for SQL injection protection
        if not os.path.exists("database/models.py"):
            issues.append("No ORM models found for SQL injection protection")

        # Check for input validation
        if not os.path.exists("api/models/security_validators.py"):
            issues.append("No input validation models found")

        return issues

    def check_insecure_design(self):
        """Check for insecure design patterns"""
        issues = []

        # Check for proper error handling
        if not os.path.exists("api/middleware/error_handlers.py"):
            issues.append("No centralized error handling found")

        return issues

    def check_security_misconfig(self):
        """Check for security misconfigurations"""
        issues = []

        # Check for proper security headers
        if not os.path.exists("api/middleware/security_headers.py"):
            issues.append("No security headers middleware found")

        return issues

    def check_vulnerable_components(self):
        """Check for vulnerable components"""
        issues = []

        # Check for requirements.txt
        if os.path.exists("requirements.txt"):
            # This would normally run safety check or similar
            issues.append("Dependency vulnerability scan needed")

        return issues

    def check_auth_failures(self):
        """Check for authentication failures"""
        issues = []

        # Check for proper session management
        if not os.path.exists("auth/jwt_handler.py"):
            issues.append("No JWT handler found")

        return issues

    def check_integrity_failures(self):
        """Check for software integrity failures"""
        issues = []

        # Check for integrity verification
        if not os.path.exists("scripts/verify-deployment.sh"):
            issues.append("No deployment verification script found")

        return issues

    def check_logging_failures(self):
        """Check for logging and monitoring failures"""
        issues = []

        # Check for security logging
        if not os.path.exists("auth/security_logging.py"):
            issues.append("No security logging implementation found")

        return issues

    def check_ssrf_vulnerabilities(self):
        """Check for SSRF vulnerabilities"""
        issues = []

        # Check for URL validation (only check if there are unvalidated external requests)
        code_files = list(Path("api").glob("**/*.py"))
        for file_path in code_files:
            if file_path.is_file():
                try:
                    with open(file_path, "r") as f:
                        content = f.read()

                    if (
                        "requests.get(" in content
                        and "url" in content
                        and "validate" not in content.lower()
                        and "whitelist" not in content.lower()
                    ):
                        issues.append(f"Potential SSRF vulnerability in {file_path}")
                        break
                except:
                    pass

        return issues

    def calculate_security_score(self):
        """Calculate overall security score"""
        total_checks = (
            len(self.security_passes) + len(self.security_warnings) + len(self.critical_issues)
        )

        if total_checks == 0:
            return 0

        # Weight different severity levels
        pass_weight = 1.0
        warning_weight = 0.5
        critical_weight = 0.0

        weighted_score = (
            len(self.security_passes) * pass_weight
            + len(self.security_warnings) * warning_weight
            + len(self.critical_issues) * critical_weight
        )

        return int((weighted_score / total_checks) * 100)

    def generate_recommendations(self):
        """Generate security recommendations"""
        recommendations = []

        if self.critical_issues:
            recommendations.append("🚨 CRITICAL: Address all critical security issues immediately")
            for issue in self.critical_issues:
                recommendations.append(f"  - Fix: {issue}")

        if self.security_warnings:
            recommendations.append("⚠️ HIGH/MEDIUM: Review and address security warnings")
            for warning in self.security_warnings[:5]:  # Show first 5
                recommendations.append(f"  - Review: {warning}")

        # General recommendations
        general_recommendations = [
            "🔐 Implement regular security audits",
            "🛡️ Set up automated vulnerability scanning",
            "📊 Enable comprehensive security monitoring",
            "🔄 Establish incident response procedures",
            "📚 Conduct security training for development team",
        ]

        recommendations.extend(general_recommendations)

        return recommendations

    def run_security_validation(self):
        """Run comprehensive security validation"""
        logger.info("Starting comprehensive security validation...")

        # Run all security validations
        self.validate_file_permissions()
        self.validate_secret_strength()
        self.validate_docker_security()
        self.validate_ssl_security()
        self.validate_api_security()
        self.validate_database_security()
        self.validate_authentication_security()
        self.validate_monitoring_security()
        self.validate_security_tests()

        # Run penetration testing simulation
        vulnerabilities = self.simulate_penetration_test()

        # Calculate security score
        security_score = self.calculate_security_score()

        # Generate recommendations
        recommendations = self.generate_recommendations()

        # Update results
        self.results.update(
            {
                "security_score": security_score,
                "vulnerabilities": vulnerabilities,
                "recommendations": recommendations,
                "compliance_status": {
                    "critical_issues": len(self.critical_issues),
                    "security_warnings": len(self.security_warnings),
                    "security_passes": len(self.security_passes),
                    "production_ready": len(self.critical_issues) == 0,
                },
            }
        )

        # Generate report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"security_validation_report_{timestamp}.json"

        with open(report_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        # Generate markdown report
        markdown_file = f"security_validation_report_{timestamp}.md"
        self.generate_markdown_report(markdown_file)

        # Print summary
        print("\n" + "=" * 60)
        print("  SECURITY VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Security Score: {security_score}%")
        print(f"Critical Issues: {len(self.critical_issues)}")
        print(f"Security Warnings: {len(self.security_warnings)}")
        print(f"Security Passes: {len(self.security_passes)}")
        print(f"Vulnerabilities Found: {len(vulnerabilities)}")
        print(f"Production Ready: {'YES' if len(self.critical_issues) == 0 else 'NO'}")
        print("=" * 60)

        if len(self.critical_issues) == 0:
            print("🔐 SECURITY VALIDATION PASSED! 🛡️")
        else:
            print("❌ SECURITY ISSUES FOUND - Fix critical issues:")
            for issue in self.critical_issues:
                print(f"  - {issue}")

        print(f"\nReports generated: {report_file}, {markdown_file}")

        return len(self.critical_issues) == 0

    def generate_markdown_report(self, filename: str):
        """Generate markdown security report"""
        with open(filename, "w") as f:
            f.write("# FreeAgentics Security Validation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Environment:** {self.results['environment']}\n")
            f.write(f"**Security Score:** {self.results['security_score']}%\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            compliance = self.results["compliance_status"]
            f.write(f"- **Critical Issues:** {compliance['critical_issues']}\n")
            f.write(f"- **Security Warnings:** {compliance['security_warnings']}\n")
            f.write(f"- **Security Passes:** {compliance['security_passes']}\n")
            f.write(
                f"- **Production Ready:** {'✅ YES' if compliance['production_ready'] else '❌ NO'}\n\n"
            )

            # Critical Issues
            if self.critical_issues:
                f.write("## 🚨 Critical Security Issues (Fix Immediately)\n\n")
                for issue in self.critical_issues:
                    f.write(f"- {issue}\n")
                f.write("\n")

            # Vulnerabilities
            if self.results["vulnerabilities"]:
                f.write("## 🔍 Security Vulnerabilities\n\n")
                for vuln in self.results["vulnerabilities"]:
                    f.write(f"- {vuln}\n")
                f.write("\n")

            # Recommendations
            f.write("## 📋 Security Recommendations\n\n")
            for rec in self.results["recommendations"]:
                f.write(f"{rec}\n")
            f.write("\n")

            # Detailed Audit Results
            f.write("## Detailed Security Audit Results\n\n")
            for category, tests in self.results["security_audit"].items():
                f.write(f"### {category.replace('_', ' ').title()}\n\n")
                for test_name, result in tests.items():
                    severity = result.get("severity", "PASS")
                    icon = {
                        "CRITICAL": "❌",
                        "HIGH": "🔴",
                        "MEDIUM": "🟡",
                        "LOW": "🟢",
                        "PASS": "✅",
                    }.get(severity, "ℹ️")
                    f.write(f"- {icon} **{test_name}**: {result.get('message', 'No message')}\n")
                f.write("\n")


def main():
    """Main function"""
    validator = SecurityValidator()
    success = validator.run_security_validation()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
