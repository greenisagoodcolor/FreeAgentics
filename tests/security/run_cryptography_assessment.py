#!/usr/bin/env python3
"""
Comprehensive Cryptography Assessment Runner

This script runs the complete cryptography security assessment for the FreeAgentics platform,
generating detailed reports with compliance mapping and remediation guidance.

Usage:
    python run_cryptography_assessment.py [options]

Options:
    --output-dir: Directory for assessment reports (default: ./reports)
    --format: Output format - json, html, pdf (default: json)
    --verbose: Enable verbose logging
    --compliance: Include compliance mapping in reports
    --remediation: Include remediation guidance
"""

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.security.cryptography_assessment_config import (
    ComplianceStandard,
    SecurityLevel,
    calculate_security_score,
    get_compliance_status,
)
from tests.security.test_cryptography_assessment import (
    CryptographicAlgorithmAssessment,
    CryptographicVulnerabilityTesting,
    EncryptionImplementationTesting,
    KeyManagementAssessment,
    SSLTLSSecurityAssessment,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CryptographyAssessmentRunner:
    """Main assessment runner class."""

    def __init__(self, output_dir: str = "./reports", output_format: str = "json"):
        self.output_dir = Path(output_dir)
        self.output_format = output_format
        self.assessment_results = {}
        self.findings = []
        self.start_time = datetime.now()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Cryptography Assessment Runner initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Output format: {self.output_format}")

    def run_comprehensive_assessment(self) -> Dict[str, Any]:
        """Run the complete cryptography assessment."""
        logger.info("Starting comprehensive cryptography assessment...")

        assessment_modules = [
            (
                "Cryptographic Algorithm Assessment",
                CryptographicAlgorithmAssessment,
                [
                    "assess_hash_algorithms",
                    "assess_symmetric_encryption",
                    "assess_asymmetric_encryption",
                    "assess_key_derivation_functions",
                ],
            ),
            (
                "Key Management Assessment",
                KeyManagementAssessment,
                [
                    "assess_key_generation_strength",
                    "assess_key_storage_security",
                    "assess_key_rotation_lifecycle",
                ],
            ),
            (
                "Encryption Implementation Testing",
                EncryptionImplementationTesting,
                [
                    "test_symmetric_encryption_security",
                    "test_asymmetric_encryption_security",
                    "test_digital_signatures",
                ],
            ),
            (
                "SSL/TLS Security Assessment",
                SSLTLSSecurityAssessment,
                [
                    "assess_ssl_context_configuration",
                    "assess_certificate_pinning",
                    "test_cipher_suite_strength",
                ],
            ),
            (
                "Cryptographic Vulnerability Testing",
                CryptographicVulnerabilityTesting,
                [
                    "test_timing_attack_resistance",
                    "test_weak_randomness_detection",
                    "test_padding_oracle_attacks",
                ],
            ),
        ]

        for module_name, module_class, test_methods in assessment_modules:
            logger.info(f"Running {module_name}...")

            try:
                module_instance = module_class()
                module_results = {}

                for method_name in test_methods:
                    if hasattr(module_instance, method_name):
                        logger.debug(f"  Executing {method_name}...")

                        try:
                            method = getattr(module_instance, method_name)
                            test_result = method()
                            module_results[method_name] = test_result

                            # Process results into findings
                            self._process_test_results(module_name, method_name, test_result)

                        except Exception as e:
                            logger.error(f"Error in {method_name}: {e}")
                            module_results[method_name] = {
                                "error": str(e),
                                "passed": [],
                                "failed": [f"Test execution failed: {e}"],
                                "warnings": [],
                            }
                    else:
                        logger.warning(f"Method {method_name} not found in {module_class.__name__}")

                self.assessment_results[module_name] = module_results
                logger.info(f"Completed {module_name}")

            except Exception as e:
                logger.error(f"Failed to run {module_name}: {e}")
                self.assessment_results[module_name] = {
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }

        # Generate final assessment report
        final_report = self._generate_final_report()

        logger.info("Comprehensive cryptography assessment completed")
        return final_report

    def _process_test_results(self, module_name: str, test_name: str, results: Dict[str, Any]):
        """Process test results into standardized findings."""

        # Process passed tests
        for passed_msg in results.get("passed", []):
            self.findings.append(
                {
                    "id": f"{module_name}_{test_name}_{len(self.findings)}",
                    "module": module_name,
                    "test": test_name,
                    "type": "pass",
                    "severity": SecurityLevel.INFO.value,
                    "message": passed_msg,
                    "compliance_standards": self._map_to_compliance_standards(passed_msg),
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Process failed tests
        for failed_msg in results.get("failed", []):
            severity = self._determine_severity(failed_msg, module_name)
            self.findings.append(
                {
                    "id": f"{module_name}_{test_name}_{len(self.findings)}",
                    "module": module_name,
                    "test": test_name,
                    "type": "failure",
                    "severity": severity.value,
                    "message": failed_msg,
                    "compliance_standards": self._map_to_compliance_standards(failed_msg),
                    "remediation": self._get_remediation_guidance(failed_msg, module_name),
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Process warnings
        for warning_msg in results.get("warnings", []):
            self.findings.append(
                {
                    "id": f"{module_name}_{test_name}_{len(self.findings)}",
                    "module": module_name,
                    "test": test_name,
                    "type": "warning",
                    "severity": SecurityLevel.MEDIUM.value,
                    "message": warning_msg,
                    "compliance_standards": self._map_to_compliance_standards(warning_msg),
                    "remediation": self._get_remediation_guidance(warning_msg, module_name),
                    "timestamp": datetime.now().isoformat(),
                }
            )

    def _determine_severity(self, message: str, module_name: str) -> SecurityLevel:
        """Determine severity level based on message content."""
        message_lower = message.lower()

        # Critical severity indicators
        critical_indicators = [
            "md5",
            "sha1",
            "des",
            "3des",
            "rc4",
            "development secret",
            "hardcoded",
            "no encryption",
            "plaintext",
            "broken",
            "compromised",
        ]

        if any(indicator in message_lower for indicator in critical_indicators):
            return SecurityLevel.CRITICAL

        # High severity indicators
        high_indicators = [
            "weak",
            "insufficient",
            "vulnerable",
            "key size",
            "timing attack",
            "oracle",
            "random",
            "predictable",
        ]

        if any(indicator in message_lower for indicator in high_indicators):
            return SecurityLevel.HIGH

        # Module-specific severity
        if "Algorithm Assessment" in module_name:
            return SecurityLevel.HIGH
        elif "Key Management" in module_name:
            return SecurityLevel.HIGH
        elif "Vulnerability Testing" in module_name:
            return SecurityLevel.MEDIUM

        return SecurityLevel.MEDIUM

    def _map_to_compliance_standards(self, message: str) -> List[str]:
        """Map findings to relevant compliance standards."""
        standards = []
        message_lower = message.lower()

        # NIST SP 800-57 mappings
        nist_keywords = [
            "algorithm",
            "key size",
            "rsa",
            "aes",
            "sha",
            "cryptographic",
        ]
        if any(keyword in message_lower for keyword in nist_keywords):
            standards.append(ComplianceStandard.NIST_SP_800_57.value)

        # FIPS 140-2 mappings
        fips_keywords = ["fips", "approved", "validated", "module"]
        if any(keyword in message_lower for keyword in fips_keywords):
            standards.append(ComplianceStandard.FIPS_140_2.value)

        # OWASP mappings
        owasp_keywords = ["password", "storage", "hash", "bcrypt", "salt"]
        if any(keyword in message_lower for keyword in owasp_keywords):
            standards.append(ComplianceStandard.OWASP_CRYPTOGRAPHIC_STORAGE.value)

        # TLS mappings
        tls_keywords = ["tls", "ssl", "cipher", "certificate", "pinning"]
        if any(keyword in message_lower for keyword in tls_keywords):
            standards.append(ComplianceStandard.RFC_8446_TLS_1_3.value)

        return standards

    def _get_remediation_guidance(self, message: str, module_name: str) -> str:
        """Get remediation guidance for specific findings."""
        message_lower = message.lower()

        # Algorithm-specific remediation
        if "md5" in message_lower or "sha1" in message_lower:
            return "Replace MD5/SHA-1 with SHA-256 or stronger hash algorithms"

        if "des" in message_lower or "3des" in message_lower:
            return "Replace DES/3DES with AES-256-GCM or ChaCha20-Poly1305"

        if "rsa-1024" in message_lower:
            return "Upgrade to RSA-2048 or stronger, or consider ECDSA P-256"

        if "development secret" in message_lower:
            return "Generate and configure production-specific secrets"

        if "timing" in message_lower:
            return "Implement constant-time comparison functions"

        if "random" in message_lower and "weak" in message_lower:
            return "Use cryptographically secure random number generators (CSPRNG)"

        if "certificate" in message_lower and "pin" in message_lower:
            return "Implement certificate pinning with backup pins and monitoring"

        if "password" in message_lower:
            return "Use bcrypt, scrypt, or Argon2 with appropriate work factors"

        # Module-specific general guidance
        if "Algorithm Assessment" in module_name:
            return "Review and update cryptographic algorithm choices to meet current security standards"
        elif "Key Management" in module_name:
            return "Implement proper key generation, storage, and rotation procedures"
        elif "SSL/TLS" in module_name:
            return "Update TLS configuration to use strong protocols and cipher suites"
        elif "Vulnerability" in module_name:
            return "Review implementation for common cryptographic vulnerabilities"

        return "Review finding and implement appropriate security measures"

    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate the final assessment report."""
        end_time = datetime.now()
        duration = end_time - self.start_time

        # Calculate security score
        security_score = calculate_security_score(self.findings)

        # Get compliance status
        compliance_status = get_compliance_status(self.findings)

        # Generate executive summary
        executive_summary = self._generate_executive_summary(security_score, compliance_status)

        # Create detailed findings report
        detailed_findings = self._organize_findings_by_category()

        final_report = {
            "assessment_metadata": {
                "assessment_type": "Comprehensive Cryptography Security Assessment",
                "platform": "FreeAgentics",
                "assessment_version": "1.0",
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": int(duration.total_seconds()),
                "assessor": "Automated Cryptography Assessment Framework",
            },
            "executive_summary": executive_summary,
            "security_score": security_score,
            "compliance_status": {std.value: status for std, status in compliance_status.items()},
            "detailed_findings": detailed_findings,
            "recommendations": self._generate_recommendations(),
            "assessment_results": self.assessment_results,
            "raw_findings": self.findings,
        }

        return final_report

    def _generate_executive_summary(
        self, security_score: Dict[str, Any], compliance_status: Dict
    ) -> Dict[str, Any]:
        """Generate executive summary of assessment results."""
        total_findings = len(self.findings)
        critical_findings = len(
            [f for f in self.findings if f.get("severity") == SecurityLevel.CRITICAL.value]
        )
        high_findings = len(
            [f for f in self.findings if f.get("severity") == SecurityLevel.HIGH.value]
        )

        # Determine overall security posture
        if critical_findings > 0:
            posture = "CRITICAL"
            posture_description = (
                "Critical cryptographic vulnerabilities identified requiring immediate attention"
            )
        elif high_findings > 5:
            posture = "HIGH_RISK"
            posture_description = (
                "Multiple high-severity cryptographic issues requiring prompt remediation"
            )
        elif high_findings > 0:
            posture = "MODERATE_RISK"
            posture_description = "Some cryptographic improvements needed"
        else:
            posture = "LOW_RISK"
            posture_description = (
                "Cryptographic implementation appears secure with minor improvements needed"
            )

        # Compliance summary
        compliant_standards = len([s for s in compliance_status.values() if s == "COMPLIANT"])
        total_standards = len(compliance_status)

        return {
            "overall_security_posture": posture,
            "posture_description": posture_description,
            "security_score": security_score["overall_score"],
            "total_findings": total_findings,
            "critical_findings": critical_findings,
            "high_severity_findings": high_findings,
            "compliance_summary": {
                "compliant_standards": compliant_standards,
                "total_standards": total_standards,
                "compliance_percentage": (
                    round((compliant_standards / total_standards) * 100, 1)
                    if total_standards > 0
                    else 0
                ),
            },
            "key_concerns": self._identify_key_concerns(),
            "immediate_actions_required": critical_findings > 0 or high_findings > 3,
        }

    def _identify_key_concerns(self) -> List[str]:
        """Identify key security concerns from findings."""
        concerns = []

        # Check for critical algorithm issues
        critical_algo_findings = [
            f
            for f in self.findings
            if f.get("severity") == SecurityLevel.CRITICAL.value
            and any(
                keyword in f.get("message", "").lower()
                for keyword in ["md5", "sha1", "des", "3des"]
            )
        ]
        if critical_algo_findings:
            concerns.append("Use of deprecated cryptographic algorithms")

        # Check for key management issues
        key_mgmt_findings = [
            f
            for f in self.findings
            if "Key Management" in f.get("module", "")
            and f.get("severity") in [SecurityLevel.CRITICAL.value, SecurityLevel.HIGH.value]
        ]
        if key_mgmt_findings:
            concerns.append("Key management security issues")

        # Check for implementation vulnerabilities
        vuln_findings = [
            f
            for f in self.findings
            if "Vulnerability" in f.get("module", "")
            and f.get("severity") in [SecurityLevel.CRITICAL.value, SecurityLevel.HIGH.value]
        ]
        if vuln_findings:
            concerns.append("Cryptographic implementation vulnerabilities")

        # Check for TLS/SSL issues
        tls_findings = [
            f
            for f in self.findings
            if "SSL/TLS" in f.get("module", "")
            and f.get("severity") in [SecurityLevel.CRITICAL.value, SecurityLevel.HIGH.value]
        ]
        if tls_findings:
            concerns.append("SSL/TLS configuration weaknesses")

        return concerns

    def _organize_findings_by_category(self) -> Dict[str, Any]:
        """Organize findings by category for detailed reporting."""
        categories = {}

        for finding in self.findings:
            module = finding.get("module", "Unknown")
            if module not in categories:
                categories[module] = {
                    "total_findings": 0,
                    "critical": [],
                    "high": [],
                    "medium": [],
                    "low": [],
                    "info": [],
                }

            categories[module]["total_findings"] += 1
            severity = finding.get("severity", SecurityLevel.INFO.value)
            categories[module][severity].append(finding)

        return categories

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate prioritized recommendations based on findings."""
        recommendations = []

        # Critical recommendations
        critical_findings = [
            f for f in self.findings if f.get("severity") == SecurityLevel.CRITICAL.value
        ]
        if critical_findings:
            recommendations.append(
                {
                    "priority": "CRITICAL",
                    "title": "Address Critical Cryptographic Vulnerabilities",
                    "description": "Immediately address critical cryptographic weaknesses that pose severe security risks",
                    "actions": [
                        f.get("remediation", f.get("message", "")) for f in critical_findings[:5]
                    ],
                    "timeline": "Immediate (within 24-48 hours)",
                }
            )

        # High priority recommendations
        high_findings = [f for f in self.findings if f.get("severity") == SecurityLevel.HIGH.value]
        if high_findings:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "title": "Strengthen Cryptographic Implementation",
                    "description": "Address high-severity cryptographic issues to improve security posture",
                    "actions": list(
                        set(
                            [f.get("remediation", f.get("message", "")) for f in high_findings[:10]]
                        )
                    ),
                    "timeline": "Short-term (within 1-2 weeks)",
                }
            )

        # General recommendations
        recommendations.extend(
            [
                {
                    "priority": "MEDIUM",
                    "title": "Implement Cryptographic Standards Compliance",
                    "description": "Ensure all cryptographic implementations meet industry standards",
                    "actions": [
                        "Review all cryptographic algorithm choices against NIST SP 800-57",
                        "Implement FIPS 140-2 validated cryptographic modules where required",
                        "Follow OWASP cryptographic storage guidelines",
                    ],
                    "timeline": "Medium-term (within 1-3 months)",
                },
                {
                    "priority": "LOW",
                    "title": "Establish Ongoing Cryptographic Monitoring",
                    "description": "Implement continuous monitoring of cryptographic health",
                    "actions": [
                        "Set up automated cryptographic algorithm scanning",
                        "Implement key rotation monitoring",
                        "Establish cryptographic incident response procedures",
                    ],
                    "timeline": "Long-term (within 3-6 months)",
                },
            ]
        )

        return recommendations

    def save_report(self, report: Dict[str, Any], filename: Optional[str] = None):
        """Save assessment report to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cryptography_assessment_report_{timestamp}.{self.output_format}"

        filepath = self.output_dir / filename

        if self.output_format == "json":
            with open(filepath, "w") as f:
                json.dump(report, f, indent=2, default=str)
        elif self.output_format == "html":
            self._save_html_report(report, filepath)
        else:
            logger.warning(f"Unsupported output format: {self.output_format}")
            # Fall back to JSON
            with open(filepath.with_suffix(".json"), "w") as f:
                json.dump(report, f, indent=2, default=str)

        logger.info(f"Assessment report saved to: {filepath}")
        return filepath

    def _save_html_report(self, report: Dict[str, Any], filepath: Path):
        """Save report in HTML format."""
        html_content = self._generate_html_report(report)
        with open(filepath, "w") as f:
            f.write(html_content)

    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report content."""
        # This would be expanded with a proper HTML template
        # For now, return a basic HTML structure
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Cryptography Assessment Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ color: #333; border-bottom: 2px solid #333; }}
                .critical {{ color: #d32f2f; }}
                .high {{ color: #f57c00; }}
                .medium {{ color: #fbc02d; }}
                .low {{ color: #388e3c; }}
                .section {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Cryptography Security Assessment Report</h1>
                <p>Platform: FreeAgentics | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="section">
                <h2>Executive Summary</h2>
                <p><strong>Overall Security Score:</strong> {report['security_score']['overall_score']}%</p>
                <p><strong>Security Posture:</strong> {report['executive_summary']['overall_security_posture']}</p>
                <p><strong>Total Findings:</strong> {report['executive_summary']['total_findings']}</p>
                <p><strong>Critical Issues:</strong> {report['executive_summary']['critical_findings']}</p>
            </div>

            <div class="section">
                <h2>Detailed Findings</h2>
                <p>See JSON report for complete technical details.</p>
            </div>
        </body>
        </html>
        """


def main():
    """Main entry point for the assessment runner."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive cryptography security assessment"
    )
    parser.add_argument(
        "--output-dir",
        default="./reports",
        help="Directory for assessment reports (default: ./reports)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "html"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--filename", help="Custom filename for the report")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize and run assessment
        runner = CryptographyAssessmentRunner(output_dir=args.output_dir, output_format=args.format)

        logger.info("Starting FreeAgentics cryptography security assessment...")
        report = runner.run_comprehensive_assessment()

        # Save report
        report_path = runner.save_report(report, args.filename)

        # Print summary
        print("\n" + "=" * 80)
        print("CRYPTOGRAPHY ASSESSMENT COMPLETED")
        print("=" * 80)
        print(f"Overall Security Score: {report['security_score']['overall_score']}%")
        print(f"Security Posture: {report['executive_summary']['overall_security_posture']}")
        print(f"Total Findings: {report['executive_summary']['total_findings']}")
        print(f"Critical Issues: {report['executive_summary']['critical_findings']}")
        print(f"High Severity Issues: {report['executive_summary']['high_severity_findings']}")
        print(f"\nReport saved to: {report_path}")

        # Exit with appropriate code
        if report["executive_summary"]["critical_findings"] > 0:
            print("\n❌ ASSESSMENT FAILED - Critical issues found")
            sys.exit(1)
        elif report["executive_summary"]["high_severity_findings"] > 5:
            print("\n⚠️  ASSESSMENT WARNING - Multiple high-severity issues found")
            sys.exit(2)
        else:
            print("\n✅ ASSESSMENT PASSED")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Assessment failed: {e}")
        traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()
