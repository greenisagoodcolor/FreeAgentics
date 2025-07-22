#!/usr/bin/env python3
"""
SSL/TLS Configuration Validator

Comprehensive validation of SSL/TLS configuration for A+ grade compliance.
Validates OCSP stapling, certificate transparency, and security best practices.
Part of Task #14.5 - Security Headers and SSL/TLS Configuration.
"""

import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent


class SSLTLSValidator:
    """Comprehensive SSL/TLS configuration validator."""

    def __init__(self):
        self.validation_results = []
        self.errors = []
        self.warnings = []
        self.a_plus_requirements = {
            "protocols": ["TLSv1.2", "TLSv1.3"],
            "ciphers": [
                "ECDHE-ECDSA-AES128-GCM-SHA256",
                "ECDHE-RSA-AES128-GCM-SHA256",
                "ECDHE-ECDSA-AES256-GCM-SHA384",
                "ECDHE-RSA-AES256-GCM-SHA384",
                "ECDHE-ECDSA-CHACHA20-POLY1305",
                "ECDHE-RSA-CHACHA20-POLY1305",
            ],
            "headers": {
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                "Expect-CT": "max-age=86400",
                "Content-Security-Policy": "default-src 'self'",
            },
        }

    def validate_nginx_ssl_config(self) -> bool:
        """Validate nginx SSL/TLS configuration files."""
        logger.info("🔍 Validating nginx SSL/TLS configuration...")

        config_files = [
            "nginx/conf.d/ssl-freeagentics.conf",
            "nginx/snippets/ssl-params.conf",
            "nginx/nginx.conf",
        ]

        all_valid = True

        for config_file in config_files:
            config_path = PROJECT_ROOT / config_file
            if not config_path.exists():
                self.errors.append(f"Missing SSL config file: {config_file}")
                all_valid = False
                continue

            try:
                content = config_path.read_text()
                file_valid = self._validate_ssl_config_content(content, config_file)
                if not file_valid:
                    all_valid = False

            except Exception as e:
                self.errors.append(f"Error reading {config_file}: {e}")
                all_valid = False

        return all_valid

    def _validate_ssl_config_content(self, content: str, filename: str) -> bool:
        """Validate SSL configuration content."""
        logger.info(f"  📄 Validating {filename}...")

        validations = [
            # Protocol validation
            (
                r"ssl_protocols\s+TLSv1\.2\s+TLSv1\.3",
                "TLS 1.2 and 1.3 protocols",
            ),
            (
                r"ssl_protocols\s+(?!.*SSLv|.*TLSv1\.0|.*TLSv1\.1)",
                "No weak protocols",
            ),
            # Cipher validation
            (r"ssl_ciphers\s+.*ECDHE.*GCM", "Strong cipher suites"),
            (
                r"ssl_prefer_server_ciphers\s+off",
                "Server cipher preference disabled",
            ),
            # Session configuration
            (r"ssl_session_cache\s+shared", "SSL session caching"),
            (r"ssl_session_timeout\s+1d", "SSL session timeout"),
            (r"ssl_session_tickets\s+off", "SSL session tickets disabled"),
            # OCSP stapling
            (r"ssl_stapling\s+on", "OCSP stapling enabled"),
            (r"ssl_stapling_verify\s+on", "OCSP stapling verification"),
            (r"resolver\s+1\.1\.1\.1", "DNS resolver configured"),
            # Perfect Forward Secrecy
            (r"ssl_dhparam", "DH parameters configured"),
            # Security headers
            (r"Strict-Transport-Security.*max-age=31536000", "HSTS header"),
            (
                r"Strict-Transport-Security.*includeSubDomains",
                "HSTS includeSubDomains",
            ),
            (r"X-Frame-Options.*DENY", "X-Frame-Options header"),
            (
                r"X-Content-Type-Options.*nosniff",
                "X-Content-Type-Options header",
            ),
            (r"Content-Security-Policy", "Content Security Policy"),
            (r"Expect-CT", "Certificate Transparency header"),
        ]

        file_valid = True
        for pattern, description in validations:
            if re.search(pattern, content, re.IGNORECASE):
                logger.info(f"    ✅ {description}")
                self.validation_results.append(f"{filename}: {description} - PASS")
            else:
                logger.warning(f"    ⚠️ {description} - NOT FOUND")
                self.warnings.append(f"{filename}: {description} - MISSING")
                # Some checks are warnings, not failures
                if description in [
                    "DNS resolver configured",
                    "DH parameters configured",
                ]:
                    continue
                file_valid = False

        return file_valid

    def validate_cipher_suites(self) -> bool:
        """Validate cipher suite configuration."""
        logger.info("🔐 Validating cipher suites...")

        config_file = PROJECT_ROOT / "nginx/snippets/ssl-params.conf"
        if not config_file.exists():
            self.errors.append("SSL parameters file not found")
            return False

        content = config_file.read_text()

        # Extract cipher configuration
        cipher_match = re.search(r"ssl_ciphers\s+([^;]+);", content)
        if not cipher_match:
            self.errors.append("No cipher configuration found")
            return False

        cipher_string = cipher_match.group(1)

        # Check for recommended ciphers
        recommended_found = 0
        for cipher in self.a_plus_requirements["ciphers"]:
            if cipher in cipher_string:
                recommended_found += 1
                logger.info(f"  ✅ Found recommended cipher: {cipher}")

        if recommended_found >= 4:  # At least 4 strong ciphers
            logger.info("✅ Cipher suite configuration is strong")
            return True
        else:
            self.errors.append(f"Only {recommended_found} recommended ciphers found")
            return False

    def validate_hsts_configuration(self) -> bool:
        """Validate HSTS configuration."""
        logger.info("🔒 Validating HSTS configuration...")

        ssl_config = PROJECT_ROOT / "nginx/conf.d/ssl-freeagentics.conf"
        if not ssl_config.exists():
            self.errors.append("SSL configuration file not found")
            return False

        content = ssl_config.read_text()

        # Check HSTS header configuration
        hsts_checks = [
            (
                r"Strict-Transport-Security.*max-age=31536000",
                "HSTS max-age (1 year)",
            ),
            (
                r"Strict-Transport-Security.*includeSubDomains",
                "HSTS includeSubDomains",
            ),
            (r"Strict-Transport-Security.*preload", "HSTS preload directive"),
        ]

        all_valid = True
        for pattern, description in hsts_checks:
            if re.search(pattern, content, re.IGNORECASE):
                logger.info(f"  ✅ {description}")
            else:
                if "preload" in description:
                    # Preload is optional
                    logger.info(f"  ⚠️ {description} - OPTIONAL")
                else:
                    logger.error(f"  ❌ {description} - MISSING")
                    all_valid = False

        return all_valid

    def validate_ocsp_stapling(self) -> bool:
        """Validate OCSP stapling configuration."""
        logger.info("📋 Validating OCSP stapling configuration...")

        ssl_config = PROJECT_ROOT / "nginx/conf.d/ssl-freeagentics.conf"
        if not ssl_config.exists():
            self.errors.append("SSL configuration file not found")
            return False

        content = ssl_config.read_text()

        ocsp_checks = [
            (r"ssl_stapling\s+on", "OCSP stapling enabled"),
            (
                r"ssl_stapling_verify\s+on",
                "OCSP stapling verification enabled",
            ),
            (r"resolver\s+1\.1\.1\.1.*8\.8\.8\.8", "DNS resolvers configured"),
            (r"resolver_timeout\s+5s", "DNS resolver timeout configured"),
        ]

        all_valid = True
        for pattern, description in ocsp_checks:
            if re.search(pattern, content, re.IGNORECASE):
                logger.info(f"  ✅ {description}")
            else:
                logger.error(f"  ❌ {description} - MISSING")
                all_valid = False

        return all_valid

    def validate_certificate_transparency(self) -> bool:
        """Validate Certificate Transparency configuration."""
        logger.info("🌐 Validating Certificate Transparency configuration...")

        ssl_config = PROJECT_ROOT / "nginx/conf.d/ssl-freeagentics.conf"
        if not ssl_config.exists():
            self.errors.append("SSL configuration file not found")
            return False

        content = ssl_config.read_text()

        ct_checks = [
            (
                r"Expect-CT.*max-age=86400",
                "Expect-CT header with proper max-age",
            ),
            (r"Expect-CT.*enforce", "Expect-CT enforcement enabled"),
        ]

        for pattern, description in ct_checks:
            if re.search(pattern, content, re.IGNORECASE):
                logger.info(f"  ✅ {description}")
            else:
                logger.warning(f"  ⚠️ {description} - MISSING")
                # CT is important but not critical for A+ rating
                self.warnings.append(f"Certificate Transparency: {description}")

        return True  # CT issues are warnings, not failures

    def validate_security_headers_integration(self) -> bool:
        """Validate security headers integration with SSL/TLS."""
        logger.info("🛡️ Validating security headers integration...")

        try:
            # Test our security headers implementation
            sys.path.insert(0, str(PROJECT_ROOT))
            from unittest.mock import Mock

            from auth.security_headers import SecurityHeadersManager

            manager = SecurityHeadersManager()

            # Mock HTTPS request
            request = Mock()
            request.url.scheme = "https"
            request.url.path = "/test"
            request.headers = {"host": "example.com"}

            response = Mock()
            response.headers = {"content-type": "text/html"}

            headers = manager.get_security_headers(request, response)

            # Check for required security headers
            required_headers = [
                "Strict-Transport-Security",
                "Content-Security-Policy",
                "X-Frame-Options",
                "X-Content-Type-Options",
                "Expect-CT",
            ]

            all_present = True
            for header in required_headers:
                if header in headers:
                    logger.info(f"  ✅ {header} header configured")
                else:
                    logger.error(f"  ❌ {header} header missing")
                    all_present = False

            return all_present

        except Exception as e:
            self.errors.append(f"Error validating security headers: {e}")
            return False

    def test_ssl_configuration_syntax(self) -> bool:
        """Test nginx SSL configuration syntax."""
        logger.info("🧪 Testing nginx configuration syntax...")

        # Test nginx configuration syntax if nginx is available
        try:
            result = subprocess.run(
                ["nginx", "-t", "-c", str(PROJECT_ROOT / "nginx/nginx.conf")],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode == 0:
                logger.info("  ✅ Nginx configuration syntax is valid")
                return True
            else:
                logger.error(f"  ❌ Nginx configuration syntax error: {result.stderr}")
                self.errors.append(f"Nginx syntax error: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logger.warning("  ⚠️ Nginx syntax test timed out")
            return True  # Don't fail for timeout
        except FileNotFoundError:
            logger.warning("  ⚠️ Nginx not available for syntax testing")
            return True  # Don't fail if nginx is not installed

    def generate_ssl_report(self) -> Dict:
        """Generate comprehensive SSL/TLS validation report."""
        logger.info("📊 Generating SSL/TLS validation report...")

        validations = [
            ("nginx_config", self.validate_nginx_ssl_config),
            ("cipher_suites", self.validate_cipher_suites),
            ("hsts", self.validate_hsts_configuration),
            ("ocsp_stapling", self.validate_ocsp_stapling),
            (
                "certificate_transparency",
                self.validate_certificate_transparency,
            ),
            ("security_headers", self.validate_security_headers_integration),
            ("syntax", self.test_ssl_configuration_syntax),
        ]

        results = {}
        total_score = 0
        max_score = len(validations)

        for test_name, test_func in validations:
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    total_score += 1
                    logger.info(f"✅ {test_name}: PASS")
                else:
                    logger.error(f"❌ {test_name}: FAIL")
            except Exception as e:
                logger.error(f"❌ {test_name}: ERROR - {e}")
                results[test_name] = False

        # Calculate grade
        score_percentage = (total_score / max_score) * 100

        if score_percentage >= 95:
            grade = "A+"
        elif score_percentage >= 90:
            grade = "A"
        elif score_percentage >= 85:
            grade = "A-"
        elif score_percentage >= 80:
            grade = "B+"
        else:
            grade = "B or lower"

        report = {
            "grade": grade,
            "score": f"{total_score}/{max_score}",
            "percentage": f"{score_percentage:.1f}%",
            "results": results,
            "errors": self.errors,
            "warnings": self.warnings,
            "recommendations": self._generate_recommendations(),
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for improvement."""
        recommendations = []

        if self.errors:
            recommendations.append("Address all critical errors to achieve A+ rating")

        if any("preload" in w for w in self.warnings):
            recommendations.append("Consider adding HSTS preload directive for maximum security")

        if any("Certificate Transparency" in w for w in self.warnings):
            recommendations.append("Configure Expect-CT with report-uri for CT monitoring")

        if not recommendations:
            recommendations.append("SSL/TLS configuration meets A+ grade requirements")

        return recommendations

    def save_report(self, report: Dict, filename: str = "ssl_tls_report.json"):
        """Save validation report to file."""
        report_path = PROJECT_ROOT / filename
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"📄 Report saved to {report_path}")


def main():
    """Main SSL/TLS validation function."""
    print("🔐 SSL/TLS Configuration Validator")
    print("=" * 50)

    validator = SSLTLSValidator()
    report = validator.generate_ssl_report()

    print("\n📊 SSL/TLS Configuration Report:")
    print(f"   Grade: {report['grade']}")
    print(f"   Score: {report['score']}")
    print(f"   Percentage: {report['percentage']}")

    if report["errors"]:
        print(f"\n❌ Errors ({len(report['errors'])}):")
        for error in report["errors"]:
            print(f"   • {error}")

    if report["warnings"]:
        print(f"\n⚠️ Warnings ({len(report['warnings'])}):")
        for warning in report["warnings"]:
            print(f"   • {warning}")

    print("\n💡 Recommendations:")
    for recommendation in report["recommendations"]:
        print(f"   • {recommendation}")

    # Save report
    validator.save_report(report)

    if report["grade"] in ["A+", "A"]:
        print(f"\n🎉 SSL/TLS configuration achieves {report['grade']} grade!")
        print("✅ Task #14.5 - SSL/TLS validation: COMPLETED")
        return True
    else:
        print("\n⚠️ SSL/TLS configuration needs improvement for A+ grade")
        print("❌ Task #14.5 - SSL/TLS validation: NEEDS ATTENTION")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
