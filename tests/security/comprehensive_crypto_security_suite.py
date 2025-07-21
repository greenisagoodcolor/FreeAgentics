#!/usr/bin/env python3
"""
Comprehensive Cryptography Security Assessment Suite

This is the master script that runs all cryptographic security assessments
for the FreeAgentics platform, providing a complete security evaluation.

Components:
1. Dynamic Cryptographic Testing (test_cryptography_assessment.py)
2. Static Code Analysis (crypto_static_analysis.py)
3. Configuration Security Review
4. Compliance Validation
5. Executive Reporting

Usage:
    python comprehensive_crypto_security_suite.py [options]
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from tests.security.crypto_static_analysis import CryptographicStaticAnalyzer
from tests.security.cryptography_assessment_config import (
    ComplianceStandard,
    calculate_security_score,
)
from tests.security.run_cryptography_assessment import (
    CryptographyAssessmentRunner,
)

logger = logging.getLogger(__name__)


class ComprehensiveCryptoSecuritySuite:
    """Master class for comprehensive cryptographic security assessment."""

    def __init__(
        self, project_root: str, output_dir: str = "./crypto_security_reports"
    ):
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.assessment_results = {}
        self.consolidated_findings = []
        self.start_time = datetime.now()

        logger.info("Comprehensive Crypto Security Suite initialized")
        logger.info(f"Project root: {self.project_root}")
        logger.info(f"Output directory: {self.output_dir}")

    def run_comprehensive_assessment(self) -> Dict[str, Any]:
        """Run the complete comprehensive cryptographic security assessment."""
        logger.info("Starting comprehensive cryptographic security assessment...")

        try:
            # 1. Run dynamic cryptographic testing
            logger.info("Phase 1: Dynamic Cryptographic Testing")
            dynamic_results = self._run_dynamic_testing()
            self.assessment_results["dynamic_testing"] = dynamic_results

            # 2. Run static code analysis
            logger.info("Phase 2: Static Code Analysis")
            static_results = self._run_static_analysis()
            self.assessment_results["static_analysis"] = static_results

            # 3. Configuration security review
            logger.info("Phase 3: Configuration Security Review")
            config_results = self._run_configuration_review()
            self.assessment_results["configuration_review"] = config_results

            # 4. Compliance validation
            logger.info("Phase 4: Compliance Validation")
            compliance_results = self._run_compliance_validation()
            self.assessment_results["compliance_validation"] = compliance_results

            # 5. Generate consolidated report
            logger.info("Phase 5: Generating Consolidated Report")
            consolidated_report = self._generate_consolidated_report()

            # 6. Create executive summary
            logger.info("Phase 6: Creating Executive Summary")
            executive_summary = self._create_executive_summary(consolidated_report)

            # 7. Generate remediation roadmap
            logger.info("Phase 7: Generating Remediation Roadmap")
            remediation_roadmap = self._generate_remediation_roadmap()

            final_report = {
                "assessment_metadata": {
                    "suite_version": "1.0",
                    "platform": "FreeAgentics",
                    "assessment_type": "Comprehensive Cryptographic Security Assessment",
                    "start_time": self.start_time.isoformat(),
                    "end_time": datetime.now().isoformat(),
                    "duration_minutes": int(
                        (datetime.now() - self.start_time).total_seconds() / 60
                    ),
                    "assessor": "Comprehensive Crypto Security Suite",
                },
                "executive_summary": executive_summary,
                "consolidated_findings": consolidated_report,
                "remediation_roadmap": remediation_roadmap,
                "detailed_results": self.assessment_results,
                "compliance_status": compliance_results,
            }

            # Save comprehensive report
            self._save_comprehensive_report(final_report)

            logger.info(
                "Comprehensive cryptographic security assessment completed successfully"
            )
            return final_report

        except Exception as e:
            logger.error(f"Comprehensive assessment failed: {e}")
            raise

    def _run_dynamic_testing(self) -> Dict[str, Any]:
        """Run dynamic cryptographic testing."""
        try:
            runner = CryptographyAssessmentRunner(
                output_dir=str(self.output_dir / "dynamic"),
                output_format="json",
            )

            dynamic_report = runner.run_comprehensive_assessment()

            # Extract findings for consolidation
            if "raw_findings" in dynamic_report:
                for finding in dynamic_report["raw_findings"]:
                    self.consolidated_findings.append(
                        {
                            "source": "dynamic_testing",
                            "category": "implementation",
                            **finding,
                        }
                    )

            return {
                "status": "completed",
                "summary": dynamic_report.get("executive_summary", {}),
                "security_score": dynamic_report.get("security_score", {}),
                "total_tests": len(dynamic_report.get("raw_findings", [])),
                "report_path": str(self.output_dir / "dynamic"),
            }

        except Exception as e:
            logger.error(f"Dynamic testing failed: {e}")
            return {"status": "failed", "error": str(e), "total_tests": 0}

    def _run_static_analysis(self) -> Dict[str, Any]:
        """Run static code analysis."""
        try:
            analyzer = CryptographicStaticAnalyzer(str(self.project_root))
            vulnerabilities = analyzer.analyze_project()
            static_report = analyzer.generate_report()

            # Save static analysis report
            static_report_path = self.output_dir / "static_analysis_report.json"
            with open(static_report_path, "w") as f:
                json.dump(static_report, f, indent=2, default=str)

            # Convert vulnerabilities to consolidated format
            for vuln_data in static_report.get("vulnerabilities_by_type", {}).values():
                for vuln_list in vuln_data:
                    if isinstance(vuln_list, list):
                        for vuln in vuln_list:
                            self.consolidated_findings.append(
                                {
                                    "source": "static_analysis",
                                    "category": "code_vulnerability",
                                    "id": f"static_{len(self.consolidated_findings)}",
                                    "type": "vulnerability",
                                    "severity": vuln.get("severity", "medium"),
                                    "message": vuln.get("description", ""),
                                    "file_path": vuln.get("file_path", ""),
                                    "line_number": vuln.get("line_number", 0),
                                    "recommendation": vuln.get("recommendation", ""),
                                    "cwe_id": vuln.get("cwe_id"),
                                    "owasp_category": vuln.get("owasp_category"),
                                    "timestamp": datetime.now().isoformat(),
                                }
                            )
                    else:
                        # Handle single vulnerability objects
                        self.consolidated_findings.append(
                            {
                                "source": "static_analysis",
                                "category": "code_vulnerability",
                                "id": f"static_{len(self.consolidated_findings)}",
                                "type": "vulnerability",
                                "severity": vuln_list.get("severity", "medium"),
                                "message": vuln_list.get("description", ""),
                                "file_path": vuln_list.get("file_path", ""),
                                "line_number": vuln_list.get("line_number", 0),
                                "recommendation": vuln_list.get("recommendation", ""),
                                "cwe_id": vuln_list.get("cwe_id"),
                                "owasp_category": vuln_list.get("owasp_category"),
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

            return {
                "status": "completed",
                "summary": static_report.get("analysis_summary", {}),
                "vulnerabilities_found": len(vulnerabilities),
                "risk_score": static_report.get("analysis_summary", {}).get(
                    "risk_score", 0
                ),
                "report_path": str(static_report_path),
            }

        except Exception as e:
            logger.error(f"Static analysis failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "vulnerabilities_found": 0,
            }

    def _run_configuration_review(self) -> Dict[str, Any]:
        """Run configuration security review."""
        try:
            config_issues = []

            # Check environment configuration
            env_issues = self._check_environment_config()
            config_issues.extend(env_issues)

            # Check file permissions
            permission_issues = self._check_file_permissions()
            config_issues.extend(permission_issues)

            # Check SSL/TLS configuration
            tls_issues = self._check_tls_configuration()
            config_issues.extend(tls_issues)

            # Check key management configuration
            key_mgmt_issues = self._check_key_management_config()
            config_issues.extend(key_mgmt_issues)

            # Add to consolidated findings
            for issue in config_issues:
                self.consolidated_findings.append(
                    {
                        "source": "configuration_review",
                        "category": "configuration",
                        "id": f"config_{len(self.consolidated_findings)}",
                        **issue,
                    }
                )

            return {
                "status": "completed",
                "total_issues": len(config_issues),
                "critical_issues": len(
                    [i for i in config_issues if i.get("severity") == "critical"]
                ),
                "high_issues": len(
                    [i for i in config_issues if i.get("severity") == "high"]
                ),
                "issues_by_category": self._categorize_config_issues(config_issues),
            }

        except Exception as e:
            logger.error(f"Configuration review failed: {e}")
            return {"status": "failed", "error": str(e), "total_issues": 0}

    def _check_environment_config(self) -> List[Dict[str, Any]]:
        """Check environment configuration for security issues."""
        issues = []

        # Check for development secrets in production
        secret_key = os.getenv("SECRET_KEY", "")
        if "dev_" in secret_key or "test_" in secret_key:
            issues.append(
                {
                    "type": "environment_config",
                    "severity": "critical",
                    "message": "Development SECRET_KEY detected in environment",
                    "recommendation": "Generate and set production-specific SECRET_KEY",
                    "category": "secret_management",
                }
            )

        # Check JWT secret
        jwt_secret = os.getenv("JWT_SECRET", "")
        if "dev_" in jwt_secret or "test_" in jwt_secret:
            issues.append(
                {
                    "type": "environment_config",
                    "severity": "critical",
                    "message": "Development JWT_SECRET detected in environment",
                    "recommendation": "Generate and set production-specific JWT_SECRET",
                    "category": "secret_management",
                }
            )

        # Check if running in production mode
        production_mode = os.getenv("PRODUCTION", "false").lower()
        if production_mode != "true":
            issues.append(
                {
                    "type": "environment_config",
                    "severity": "medium",
                    "message": "PRODUCTION environment variable not set to true",
                    "recommendation": "Set PRODUCTION=true for production deployment",
                    "category": "deployment_config",
                }
            )

        return issues

    def _check_file_permissions(self) -> List[Dict[str, Any]]:
        """Check file permissions for cryptographic assets."""
        issues = []

        # Check key files
        key_files = [
            self.project_root / "auth" / "keys" / "jwt_private.pem",
            self.project_root / "auth" / "keys" / "jwt_public.pem",
        ]

        for key_file in key_files:
            if key_file.exists():
                stat_info = key_file.stat()
                permissions = oct(stat_info.st_mode)[-3:]

                if "private" in key_file.name and permissions != "600":
                    issues.append(
                        {
                            "type": "file_permissions",
                            "severity": "high",
                            "message": f"Private key file has overly permissive permissions: {permissions}",
                            "recommendation": f"Set permissions to 600 for {key_file}",
                            "category": "key_management",
                            "file_path": str(key_file),
                        }
                    )

        return issues

    def _check_tls_configuration(self) -> List[Dict[str, Any]]:
        """Check TLS/SSL configuration."""
        issues = []

        # Check docker-compose for TLS settings
        docker_compose_files = [
            self.project_root / "docker-compose.yml",
            self.project_root / "docker-compose.production.yml",
        ]

        for compose_file in docker_compose_files:
            if compose_file.exists():
                try:
                    with open(compose_file, "r") as f:
                        content = f.read().lower()

                    if "ssl_cert" not in content and "tls" not in content:
                        issues.append(
                            {
                                "type": "tls_configuration",
                                "severity": "medium",
                                "message": f"No TLS configuration found in {compose_file.name}",
                                "recommendation": "Configure TLS certificates for production deployment",
                                "category": "transport_security",
                                "file_path": str(compose_file),
                            }
                        )
                except Exception:
                    pass

        return issues

    def _check_key_management_config(self) -> List[Dict[str, Any]]:
        """Check key management configuration."""
        issues = []

        # Check for key rotation configuration
        # This would check for key rotation policies, HSM integration, etc.
        # For now, we'll check basic key storage

        auth_dir = self.project_root / "auth"
        if auth_dir.exists():
            key_dir = auth_dir / "keys"
            if not key_dir.exists():
                issues.append(
                    {
                        "type": "key_management",
                        "severity": "medium",
                        "message": "Key storage directory not found",
                        "recommendation": "Create secure key storage directory with proper permissions",
                        "category": "key_management",
                    }
                )

        return issues

    def _categorize_config_issues(
        self, issues: List[Dict[str, Any]]
    ) -> Dict[str, List]:
        """Categorize configuration issues."""
        categories = {}
        for issue in issues:
            category = issue.get("category", "other")
            if category not in categories:
                categories[category] = []
            categories[category].append(issue)
        return categories

    def _run_compliance_validation(self) -> Dict[str, Any]:
        """Run compliance validation against security standards."""
        try:
            compliance_results = {}

            # Map findings to compliance standards
            findings_by_standard = self._map_findings_to_standards()

            # Evaluate compliance for each standard
            for standard in ComplianceStandard:
                standard_findings = findings_by_standard.get(standard, [])
                compliance_status = self._evaluate_compliance_status(
                    standard, standard_findings
                )
                compliance_results[standard.value] = compliance_status

            return {
                "status": "completed",
                "compliance_by_standard": compliance_results,
                "overall_compliance_score": self._calculate_overall_compliance_score(
                    compliance_results
                ),
                "non_compliant_standards": [
                    std
                    for std, result in compliance_results.items()
                    if result["status"] == "NON_COMPLIANT"
                ],
            }

        except Exception as e:
            logger.error(f"Compliance validation failed: {e}")
            return {"status": "failed", "error": str(e)}

    def _map_findings_to_standards(
        self,
    ) -> Dict[ComplianceStandard, List[Dict]]:
        """Map findings to compliance standards."""
        findings_by_standard = {}

        for finding in self.consolidated_findings:
            # Determine which standards this finding relates to
            standards = self._determine_relevant_standards(finding)

            for standard in standards:
                if standard not in findings_by_standard:
                    findings_by_standard[standard] = []
                findings_by_standard[standard].append(finding)

        return findings_by_standard

    def _determine_relevant_standards(
        self, finding: Dict[str, Any]
    ) -> List[ComplianceStandard]:
        """Determine which compliance standards a finding relates to."""
        standards = []
        message = finding.get("message", "").lower()

        # NIST SP 800-57 - Key management and cryptographic algorithms
        if any(
            keyword in message for keyword in ["algorithm", "key", "rsa", "aes", "sha"]
        ):
            standards.append(ComplianceStandard.NIST_SP_800_57)

        # FIPS 140-2 - Cryptographic modules
        if any(keyword in message for keyword in ["module", "validation", "approved"]):
            standards.append(ComplianceStandard.FIPS_140_2)

        # OWASP Cryptographic Storage
        if any(
            keyword in message for keyword in ["password", "storage", "hash", "salt"]
        ):
            standards.append(ComplianceStandard.OWASP_CRYPTOGRAPHIC_STORAGE)

        # RFC 8446 TLS 1.3
        if any(
            keyword in message for keyword in ["tls", "ssl", "certificate", "cipher"]
        ):
            standards.append(ComplianceStandard.RFC_8446_TLS_1_3)

        return standards

    def _evaluate_compliance_status(
        self, standard: ComplianceStandard, findings: List[Dict]
    ) -> Dict[str, Any]:
        """Evaluate compliance status for a standard."""
        critical_findings = [f for f in findings if f.get("severity") == "critical"]
        high_findings = [f for f in findings if f.get("severity") == "high"]

        if critical_findings:
            status = "NON_COMPLIANT"
            score = 0
        elif high_findings:
            status = "PARTIAL_COMPLIANCE"
            score = 50
        else:
            status = "COMPLIANT"
            score = 100

        return {
            "status": status,
            "score": score,
            "total_findings": len(findings),
            "critical_findings": len(critical_findings),
            "high_findings": len(high_findings),
            "key_issues": [
                f.get("message", "") for f in critical_findings + high_findings
            ][:5],
        }

    def _calculate_overall_compliance_score(
        self, compliance_results: Dict[str, Any]
    ) -> float:
        """Calculate overall compliance score."""
        if not compliance_results:
            return 0.0

        total_score = sum(result["score"] for result in compliance_results.values())
        return total_score / len(compliance_results)

    def _generate_consolidated_report(self) -> Dict[str, Any]:
        """Generate consolidated findings report."""
        # Categorize all findings
        findings_by_source = {}
        findings_by_severity = {}
        findings_by_category = {}

        for finding in self.consolidated_findings:
            # By source
            source = finding.get("source", "unknown")
            if source not in findings_by_source:
                findings_by_source[source] = []
            findings_by_source[source].append(finding)

            # By severity
            severity = finding.get("severity", "info")
            if severity not in findings_by_severity:
                findings_by_severity[severity] = []
            findings_by_severity[severity].append(finding)

            # By category
            category = finding.get("category", "other")
            if category not in findings_by_category:
                findings_by_category[category] = []
            findings_by_category[category].append(finding)

        # Calculate consolidated scores
        security_score = calculate_security_score(self.consolidated_findings)

        return {
            "total_findings": len(self.consolidated_findings),
            "security_score": security_score,
            "findings_by_source": findings_by_source,
            "findings_by_severity": findings_by_severity,
            "findings_by_category": findings_by_category,
            "top_risks": self._identify_top_risks(),
            "critical_issues_summary": self._summarize_critical_issues(),
        }

    def _identify_top_risks(self) -> List[Dict[str, Any]]:
        """Identify top security risks."""
        critical_findings = [
            f for f in self.consolidated_findings if f.get("severity") == "critical"
        ]
        high_findings = [
            f for f in self.consolidated_findings if f.get("severity") == "high"
        ]

        top_risks = []

        # Group by message to avoid duplicates
        risk_groups = {}
        for finding in critical_findings + high_findings:
            message = finding.get("message", "")
            if message not in risk_groups:
                risk_groups[message] = {
                    "risk_description": message,
                    "severity": finding.get("severity"),
                    "count": 0,
                    "sources": set(),
                    "recommendations": set(),
                }

            risk_groups[message]["count"] += 1
            risk_groups[message]["sources"].add(finding.get("source", ""))
            risk_groups[message]["recommendations"].add(
                finding.get("recommendation", "")
            )

        # Convert to list and sort by severity and count
        for risk_data in risk_groups.values():
            risk_data["sources"] = list(risk_data["sources"])
            risk_data["recommendations"] = list(risk_data["recommendations"])
            top_risks.append(risk_data)

        top_risks.sort(
            key=lambda x: (x["severity"] == "critical", x["count"]),
            reverse=True,
        )
        return top_risks[:10]

    def _summarize_critical_issues(self) -> Dict[str, Any]:
        """Summarize critical security issues."""
        critical_findings = [
            f for f in self.consolidated_findings if f.get("severity") == "critical"
        ]

        if not critical_findings:
            return {"status": "no_critical_issues"}

        issue_categories = {}
        for finding in critical_findings:
            category = finding.get("category", "other")
            if category not in issue_categories:
                issue_categories[category] = []
            issue_categories[category].append(finding)

        return {
            "status": "critical_issues_found",
            "total_critical": len(critical_findings),
            "by_category": {
                cat: len(findings) for cat, findings in issue_categories.items()
            },
            "immediate_actions": [
                f.get("recommendation", "") for f in critical_findings[:5]
            ],
        }

    def _create_executive_summary(
        self, consolidated_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create executive summary."""
        total_findings = consolidated_report["total_findings"]
        security_score = consolidated_report["security_score"]["overall_score"]
        critical_count = len(
            consolidated_report["findings_by_severity"].get("critical", [])
        )
        high_count = len(consolidated_report["findings_by_severity"].get("high", []))

        # Determine overall risk level
        if critical_count > 0:
            risk_level = "CRITICAL"
            risk_description = "Immediate action required to address critical cryptographic vulnerabilities"
        elif high_count > 5:
            risk_level = "HIGH"
            risk_description = (
                "Multiple high-severity cryptographic issues require prompt attention"
            )
        elif high_count > 0:
            risk_level = "MEDIUM"
            risk_description = "Some cryptographic improvements needed"
        else:
            risk_level = "LOW"
            risk_description = (
                "Cryptographic implementation appears secure with minor improvements"
            )

        return {
            "overall_risk_level": risk_level,
            "risk_description": risk_description,
            "security_score": security_score,
            "total_findings": total_findings,
            "critical_findings": critical_count,
            "high_severity_findings": high_count,
            "assessment_coverage": {
                "dynamic_testing": "dynamic_testing" in self.assessment_results,
                "static_analysis": "static_analysis" in self.assessment_results,
                "configuration_review": "configuration_review"
                in self.assessment_results,
                "compliance_validation": "compliance_validation"
                in self.assessment_results,
            },
            "key_recommendations": [
                rec
                for rec in consolidated_report.get("critical_issues_summary", {}).get(
                    "immediate_actions", []
                )
            ][:3],
            "business_impact": self._assess_business_impact(
                risk_level, critical_count, high_count
            ),
        }

    def _assess_business_impact(
        self, risk_level: str, critical_count: int, high_count: int
    ) -> Dict[str, Any]:
        """Assess business impact of cryptographic vulnerabilities."""
        if risk_level == "CRITICAL":
            return {
                "impact_level": "HIGH",
                "description": "Critical cryptographic vulnerabilities pose severe security risks",
                "potential_consequences": [
                    "Data breach risk",
                    "Compliance violations",
                    "Reputation damage",
                    "Financial losses",
                ],
                "recommended_timeline": "Immediate (24-48 hours)",
            }
        elif risk_level == "HIGH":
            return {
                "impact_level": "MEDIUM",
                "description": "Multiple high-severity issues increase security risk",
                "potential_consequences": [
                    "Increased attack surface",
                    "Potential compliance issues",
                    "Security posture degradation",
                ],
                "recommended_timeline": "Short-term (1-2 weeks)",
            }
        else:
            return {
                "impact_level": "LOW",
                "description": "Security improvements will strengthen overall posture",
                "potential_consequences": [
                    "Gradual security improvements",
                    "Enhanced compliance posture",
                ],
                "recommended_timeline": "Medium-term (1-3 months)",
            }

    def _generate_remediation_roadmap(self) -> Dict[str, Any]:
        """Generate prioritized remediation roadmap."""
        critical_findings = [
            f for f in self.consolidated_findings if f.get("severity") == "critical"
        ]
        high_findings = [
            f for f in self.consolidated_findings if f.get("severity") == "high"
        ]
        medium_findings = [
            f for f in self.consolidated_findings if f.get("severity") == "medium"
        ]

        roadmap = {
            "immediate_actions": {
                "timeline": "24-48 hours",
                "priority": "CRITICAL",
                "description": "Address critical vulnerabilities immediately",
                "actions": list(
                    set([f.get("recommendation", "") for f in critical_findings])
                )[:5],
            },
            "short_term_actions": {
                "timeline": "1-2 weeks",
                "priority": "HIGH",
                "description": "Resolve high-severity issues",
                "actions": list(
                    set([f.get("recommendation", "") for f in high_findings])
                )[:10],
            },
            "medium_term_actions": {
                "timeline": "1-3 months",
                "priority": "MEDIUM",
                "description": "Implement security improvements",
                "actions": list(
                    set([f.get("recommendation", "") for f in medium_findings])
                )[:10],
            },
            "long_term_initiatives": {
                "timeline": "3-6 months",
                "priority": "LOW",
                "description": "Establish ongoing security practices",
                "actions": [
                    "Implement automated cryptographic monitoring",
                    "Establish key rotation procedures",
                    "Create cryptographic incident response plan",
                    "Conduct regular security assessments",
                    "Implement security training program",
                ],
            },
        }

        return roadmap

    def _save_comprehensive_report(self, report: Dict[str, Any]):
        """Save the comprehensive report to multiple formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save JSON report
        json_path = (
            self.output_dir / f"comprehensive_crypto_security_report_{timestamp}.json"
        )
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Save executive summary
        exec_summary_path = self.output_dir / f"executive_summary_{timestamp}.json"
        with open(exec_summary_path, "w") as f:
            json.dump(report["executive_summary"], f, indent=2, default=str)

        # Save remediation roadmap
        roadmap_path = self.output_dir / f"remediation_roadmap_{timestamp}.json"
        with open(roadmap_path, "w") as f:
            json.dump(report["remediation_roadmap"], f, indent=2, default=str)

        logger.info(f"Comprehensive report saved to: {json_path}")
        logger.info(f"Executive summary saved to: {exec_summary_path}")
        logger.info(f"Remediation roadmap saved to: {roadmap_path}")


def main():
    """Main entry point for the comprehensive crypto security suite."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive cryptographic security assessment suite"
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory (default: current directory)",
    )
    parser.add_argument(
        "--output-dir",
        default="./crypto_security_reports",
        help="Output directory for reports (default: ./crypto_security_reports)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick assessment (skip some time-intensive tests)",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        print("=" * 80)
        print("FREEAGENTICS COMPREHENSIVE CRYPTOGRAPHIC SECURITY ASSESSMENT")
        print("=" * 80)
        print(f"Project: {args.project_root}")
        print(f"Output: {args.output_dir}")
        print(f"Mode: {'Quick' if args.quick else 'Complete'}")
        print("=" * 80)

        # Initialize and run comprehensive assessment
        suite = ComprehensiveCryptoSecuritySuite(
            project_root=args.project_root, output_dir=args.output_dir
        )

        final_report = suite.run_comprehensive_assessment()

        # Print executive summary
        exec_summary = final_report["executive_summary"]
        print(f"\nOVERALL RISK LEVEL: {exec_summary['overall_risk_level']}")
        print(f"SECURITY SCORE: {exec_summary['security_score']:.1f}%")
        print(f"TOTAL FINDINGS: {exec_summary['total_findings']}")
        print(f"CRITICAL ISSUES: {exec_summary['critical_findings']}")
        print(f"HIGH SEVERITY: {exec_summary['high_severity_findings']}")
        print(f"\nRISK DESCRIPTION: {exec_summary['risk_description']}")

        if exec_summary["key_recommendations"]:
            print("\nKEY RECOMMENDATIONS:")
            for i, rec in enumerate(exec_summary["key_recommendations"], 1):
                print(f"  {i}. {rec}")

        # Exit with appropriate code
        if exec_summary["critical_findings"] > 0:
            print(
                f"\n❌ ASSESSMENT FAILED - {exec_summary['critical_findings']} critical issues found"
            )
            sys.exit(1)
        elif exec_summary["overall_risk_level"] in ["HIGH", "CRITICAL"]:
            print(
                f"\n⚠️  ASSESSMENT WARNING - {exec_summary['overall_risk_level']} risk level"
            )
            sys.exit(2)
        else:
            print(
                f"\n✅ ASSESSMENT PASSED - {exec_summary['overall_risk_level']} risk level"
            )
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n\nAssessment interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Comprehensive assessment failed: {e}")
        print(f"\n❌ ASSESSMENT ERROR: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
