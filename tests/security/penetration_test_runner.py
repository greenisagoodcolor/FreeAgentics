"""
Penetration Testing Orchestration Runner

This module provides the main orchestration and execution interface for the
comprehensive penetration testing framework. It coordinates all test modules,
generates reports, and provides both CLI and programmatic interfaces.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .api_security_tests import APISecurityTests
from .authentication_bypass_tests import AuthenticationBypassTests
from .authorization_tests import AuthorizationTests
from .business_logic_tests import BusinessLogicTests
from .penetration_testing_framework import PenetrationTestingFramework
from .session_management_tests import SessionManagementTests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PenetrationTestRunner:
    """Main penetration testing orchestration runner."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.framework = PenetrationTestingFramework()
        self.report_dir = Path("/home/green/FreeAgentics/tests/security/reports")
        self.report_dir.mkdir(exist_ok=True)

        # Register all test modules
        self._register_test_modules()

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for penetration testing."""
        return {
            "enabled_modules": [
                "authentication_bypass",
                "session_management",
                "authorization",
                "api_security",
                "business_logic",
            ],
            "output_formats": ["json", "html", "markdown"],
            "severity_threshold": "low",
            "concurrent_tests": False,
            "detailed_reporting": True,
            "include_proof_of_concept": True,
            "generate_remediation_plan": True,
        }

    def _register_test_modules(self):
        """Register all penetration test modules with the framework."""
        test_modules = {
            "authentication_bypass": AuthenticationBypassTests,
            "session_management": SessionManagementTests,
            "authorization": AuthorizationTests,
            "api_security": APISecurityTests,
            "business_logic": BusinessLogicTests,
        }

        for module_name, module_class in test_modules.items():
            if module_name in self.config["enabled_modules"]:
                try:
                    # Initialize test module
                    test_instance = module_class(self.framework.client, self.framework.auth_manager)
                    self.framework.register_test_module(test_instance)
                    logger.info(f"Registered test module: {module_name}")
                except Exception as e:
                    logger.error(f"Failed to register {module_name}: {e}")

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all registered penetration tests."""
        logger.info("Starting comprehensive penetration testing suite")
        logger.info(f"Enabled modules: {', '.join(self.config['enabled_modules'])}")

        start_time = time.time()

        try:
            # Run all tests
            results = await self.framework.run_all_tests()

            # Apply severity filtering
            filtered_results = self._filter_by_severity(results)

            # Generate reports in multiple formats
            report_files = await self._generate_reports(filtered_results)

            # Add metadata
            filtered_results["execution_summary"] = {
                "total_execution_time": time.time() - start_time,
                "tests_executed": len(self.framework.test_modules),
                "report_files": report_files,
                "configuration": self.config,
            }

            logger.info("Penetration testing completed successfully")
            logger.info(
                f"Total vulnerabilities found: {len(filtered_results.get('detailed_findings', []))}"
            )

            return filtered_results

        except Exception as e:
            logger.error(f"Penetration testing failed: {e}")
            raise

    async def run_specific_module(self, module_name: str) -> Dict[str, Any]:
        """Run a specific penetration test module."""
        if module_name not in self.config["enabled_modules"]:
            raise ValueError(f"Module {module_name} not enabled in configuration")

        logger.info(f"Running specific module: {module_name}")

        # Create temporary framework with single module
        temp_framework = PenetrationTestingFramework()

        test_modules = {
            "authentication_bypass": AuthenticationBypassTests,
            "session_management": SessionManagementTests,
            "authorization": AuthorizationTests,
            "api_security": APISecurityTests,
            "business_logic": BusinessLogicTests,
        }

        if module_name in test_modules:
            test_instance = test_modules[module_name](
                temp_framework.client, temp_framework.auth_manager
            )
            temp_framework.register_test_module(test_instance)

            results = await temp_framework.run_all_tests()
            filtered_results = self._filter_by_severity(results)

            # Generate reports
            report_files = await self._generate_reports(filtered_results, suffix=f"_{module_name}")
            filtered_results["execution_summary"] = {
                "module": module_name,
                "report_files": report_files,
            }

            return filtered_results
        else:
            raise ValueError(f"Unknown module: {module_name}")

    def _filter_by_severity(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Filter results by severity threshold."""
        threshold = self.config["severity_threshold"].lower()
        severity_order = ["info", "low", "medium", "high", "critical"]

        if threshold not in severity_order:
            return results

        threshold_index = severity_order.index(threshold)

        # Filter vulnerabilities
        filtered_findings = []
        for finding in results.get("detailed_findings", []):
            finding_severity = finding.get("severity", "info").lower()
            if finding_severity in severity_order:
                finding_index = severity_order.index(finding_severity)
                if finding_index >= threshold_index:
                    filtered_findings.append(finding)

        # Update results
        filtered_results = results.copy()
        filtered_results["detailed_findings"] = filtered_findings

        # Recalculate summary statistics
        severity_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0,
        }
        for finding in filtered_findings:
            severity = finding.get("severity", "info").lower()
            if severity in severity_counts:
                severity_counts[severity] += 1

        filtered_results["executive_summary"]["total_vulnerabilities"] = len(filtered_findings)
        filtered_results["executive_summary"]["severity_distribution"] = severity_counts

        return filtered_results

    async def _generate_reports(self, results: Dict[str, Any], suffix: str = "") -> List[str]:
        """Generate reports in multiple formats."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_files = []

        # JSON Report
        if "json" in self.config["output_formats"]:
            json_file = self.report_dir / f"pentest_report{suffix}_{timestamp}.json"
            with open(json_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            report_files.append(str(json_file))
            logger.info(f"Generated JSON report: {json_file}")

        # HTML Report
        if "html" in self.config["output_formats"]:
            html_file = self.report_dir / f"pentest_report{suffix}_{timestamp}.html"
            html_content = self._generate_html_report(results)
            with open(html_file, "w") as f:
                f.write(html_content)
            report_files.append(str(html_file))
            logger.info(f"Generated HTML report: {html_file}")

        # Markdown Report
        if "markdown" in self.config["output_formats"]:
            md_file = self.report_dir / f"pentest_report{suffix}_{timestamp}.md"
            md_content = self._generate_markdown_report(results)
            with open(md_file, "w") as f:
                f.write(md_content)
            report_files.append(str(md_file))
            logger.info(f"Generated Markdown report: {md_file}")

        return report_files

    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report."""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FreeAgentics Penetration Testing Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .header {{ border-bottom: 2px solid #e0e0e0; padding-bottom: 20px; margin-bottom: 30px; }}
        .severity-critical {{ color: #dc3545; font-weight: bold; }}
        .severity-high {{ color: #fd7e14; font-weight: bold; }}
        .severity-medium {{ color: #ffc107; font-weight: bold; }}
        .severity-low {{ color: #28a745; }}
        .severity-info {{ color: #17a2b8; }}
        .vulnerability {{ margin: 20px 0; padding: 20px; border-left: 4px solid #ddd; background: #fafafa; }}
        .vulnerability.critical {{ border-left-color: #dc3545; }}
        .vulnerability.high {{ border-left-color: #fd7e14; }}
        .vulnerability.medium {{ border-left-color: #ffc107; }}
        .vulnerability.low {{ border-left-color: #28a745; }}
        .vulnerability.info {{ border-left-color: #17a2b8; }}
        .poc {{ background: #f8f9fa; padding: 15px; border-radius: 4px; margin: 10px 0; font-family: monospace; }}
        .remediation {{ background: #d4edda; padding: 15px; border-radius: 4px; margin: 10px 0; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .summary-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
        .summary-number {{ font-size: 2em; font-weight: bold; margin-bottom: 5px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>FreeAgentics Platform - Penetration Testing Report</h1>
            <p><strong>Generated:</strong> {generated_time}</p>
            <p><strong>Target:</strong> {target}</p>
            <p><strong>Framework Version:</strong> {framework_version}</p>
        </div>

        <h2>Executive Summary</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <div class="summary-number severity-critical">{critical_count}</div>
                <div>Critical</div>
            </div>
            <div class="summary-card">
                <div class="summary-number severity-high">{high_count}</div>
                <div>High</div>
            </div>
            <div class="summary-card">
                <div class="summary-number severity-medium">{medium_count}</div>
                <div>Medium</div>
            </div>
            <div class="summary-card">
                <div class="summary-number severity-low">{low_count}</div>
                <div>Low</div>
            </div>
            <div class="summary-card">
                <div class="summary-number">{total_vulns}</div>
                <div>Total Vulnerabilities</div>
            </div>
        </div>

        <p><strong>Risk Score:</strong> {risk_score}/100</p>

        <h3>Key Recommendations</h3>
        <ul>
        {recommendations}
        </ul>

        <h2>Detailed Findings</h2>
        {vulnerabilities}

        <h2>Remediation Plan</h2>
        <h3>Immediate Actions (Critical/High)</h3>
        <ul>
        {immediate_actions}
        </ul>

        <h3>Short Term (Within Week)</h3>
        <ul>
        {short_term_actions}
        </ul>

        <h3>Medium Term (Within Month)</h3>
        <ul>
        {medium_term_actions}
        </ul>

        <h2>Test Execution Summary</h2>
        <p><strong>Tests Executed:</strong> {tests_executed}</p>
        <p><strong>Execution Time:</strong> {execution_time:.2f} seconds</p>
        <p><strong>Tests Successful:</strong> {tests_successful}</p>
    </div>
</body>
</html>
        """

        # Prepare template variables
        metadata = results.get("metadata", {})
        summary = results.get("executive_summary", {})
        findings = results.get("detailed_findings", [])
        remediation = results.get("remediation_plan", {})

        # Count vulnerabilities by severity
        severity_counts = summary.get("severity_distribution", {})

        # Generate vulnerability HTML
        vuln_html = ""
        for finding in findings:
            severity = finding.get("severity", "info").lower()
            vuln_html += f"""
            <div class="vulnerability {severity}">
                <h3 class="severity-{severity}">[{severity.upper()}] {finding.get('title', 'Unknown')}</h3>
                <p><strong>Endpoint:</strong> {finding.get('affected_endpoint', 'N/A')}</p>
                <p><strong>CWE:</strong> {finding.get('cwe_id', 'N/A')} | <strong>CVSS:</strong> {finding.get('cvss_score', 'N/A')}</p>
                <p>{finding.get('description', '')}</p>

                {f'<div class="poc"><strong>Proof of Concept:</strong><br><pre>{finding.get("proof_of_concept", "")}</pre></div>' if self.config["include_proof_of_concept"] else ''}

                <div class="remediation">
                    <strong>Remediation Steps:</strong>
                    <ol>
                    {''.join(f'<li>{step}</li>' for step in finding.get('remediation_steps', []))}
                    </ol>
                </div>
            </div>
            """

        # Generate remediation actions
        immediate_actions = ""
        short_term_actions = ""
        medium_term_actions = ""

        if "prioritization" in remediation:
            for action in remediation["prioritization"].get("immediate", []):
                immediate_actions += f"<li>{action.get('title', 'Unknown vulnerability')}</li>"
            for action in remediation["prioritization"].get("within_week", []):
                short_term_actions += f"<li>{action.get('title', 'Unknown vulnerability')}</li>"
            for action in remediation["prioritization"].get("within_month", []):
                medium_term_actions += f"<li>{action.get('title', 'Unknown vulnerability')}</li>"

        # Fill template
        return html_template.format(
            generated_time=metadata.get("generated_at", "Unknown"),
            target=metadata.get("target", "FreeAgentics Platform"),
            framework_version=metadata.get("framework_version", "1.0.0"),
            critical_count=severity_counts.get("critical", 0),
            high_count=severity_counts.get("high", 0),
            medium_count=severity_counts.get("medium", 0),
            low_count=severity_counts.get("low", 0),
            total_vulns=summary.get("total_vulnerabilities", 0),
            risk_score=summary.get("risk_score", 0),
            recommendations="".join(
                f"<li>{rec}</li>" for rec in summary.get("recommendations", [])
            ),
            vulnerabilities=vuln_html,
            immediate_actions=immediate_actions,
            short_term_actions=short_term_actions,
            medium_term_actions=medium_term_actions,
            tests_executed=metadata.get("tests_executed", 0),
            execution_time=metadata.get("total_execution_time", 0),
            tests_successful=metadata.get("tests_successful", 0),
        )

    def _generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Generate Markdown report."""
        metadata = results.get("metadata", {})
        summary = results.get("executive_summary", {})
        findings = results.get("detailed_findings", [])
        remediation = results.get("remediation_plan", {})

        md_content = f"""# FreeAgentics Platform - Penetration Testing Report

**Generated:** {metadata.get("generated_at", "Unknown")}
**Target:** {metadata.get("target", "FreeAgentics Platform")}
**Framework Version:** {metadata.get("framework_version", "1.0.0")}

## Executive Summary

| Severity | Count |
|----------|-------|
| Critical | {summary.get("severity_distribution", {}).get("critical", 0)} |
| High     | {summary.get("severity_distribution", {}).get("high", 0)} |
| Medium   | {summary.get("severity_distribution", {}).get("medium", 0)} |
| Low      | {summary.get("severity_distribution", {}).get("low", 0)} |
| Info     | {summary.get("severity_distribution", {}).get("info", 0)} |
| **Total** | **{summary.get("total_vulnerabilities", 0)}** |

**Risk Score:** {summary.get("risk_score", 0)}/100

### Key Recommendations

"""

        for rec in summary.get("recommendations", []):
            md_content += f"- {rec}\n"

        md_content += "\n## Detailed Findings\n\n"

        for i, finding in enumerate(findings, 1):
            severity = finding.get("severity", "info").upper()
            md_content += f"""### {i}. [{severity}] {finding.get('title', 'Unknown')}

**Affected Endpoint:** `{finding.get('affected_endpoint', 'N/A')}`
**CWE:** {finding.get('cwe_id', 'N/A')} | **CVSS Score:** {finding.get('cvss_score', 'N/A')}

{finding.get('description', '')}

"""

            if self.config["include_proof_of_concept"] and finding.get("proof_of_concept"):
                md_content += f"""**Proof of Concept:**
```
{finding.get('proof_of_concept', '')}
```

"""

            md_content += "**Exploitation Steps:**\n"
            for step in finding.get("exploitation_steps", []):
                md_content += f"1. {step}\n"

            md_content += "\n**Remediation Steps:**\n"
            for step in finding.get("remediation_steps", []):
                md_content += f"1. {step}\n"

            md_content += "\n---\n\n"

        md_content += """## Remediation Plan

### Immediate Actions (Critical/High Priority)

"""

        if "prioritization" in remediation:
            for action in remediation["prioritization"].get("immediate", []):
                md_content += f"- {action.get('title', 'Unknown vulnerability')}\n"

        md_content += "\n### Short Term Actions (Within Week)\n\n"

        if "prioritization" in remediation:
            for action in remediation["prioritization"].get("within_week", []):
                md_content += f"- {action.get('title', 'Unknown vulnerability')}\n"

        md_content += "\n### Medium Term Actions (Within Month)\n\n"

        if "prioritization" in remediation:
            for action in remediation["prioritization"].get("within_month", []):
                md_content += f"- {action.get('title', 'Unknown vulnerability')}\n"

        md_content += f"""

## Test Execution Summary

- **Tests Executed:** {metadata.get("tests_executed", 0)}
- **Execution Time:** {metadata.get("total_execution_time", 0):.2f} seconds
- **Tests Successful:** {metadata.get("tests_successful", 0)}

---

*Report generated by FreeAgentics Penetration Testing Framework v{metadata.get("framework_version", "1.0.0")}*
"""

        return md_content


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="FreeAgentics Penetration Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--module",
        "-m",
        choices=[
            "authentication_bypass",
            "session_management",
            "authorization",
            "api_security",
            "business_logic",
            "all",
        ],
        default="all",
        help="Test module to run (default: all)",
    )

    parser.add_argument(
        "--output",
        "-o",
        choices=["json", "html", "markdown"],
        nargs="+",
        default=["json", "html"],
        help="Output formats (default: json html)",
    )

    parser.add_argument(
        "--severity",
        "-s",
        choices=["info", "low", "medium", "high", "critical"],
        default="low",
        help="Minimum severity threshold (default: low)",
    )

    parser.add_argument("--config", "-c", type=str, help="Configuration file path (JSON)")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config = None
    if args.config:
        try:
            with open(args.config, "r") as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            sys.exit(1)

    # Override config with CLI arguments
    if not config:
        config = {}

    config["output_formats"] = args.output
    config["severity_threshold"] = args.severity

    if args.module != "all":
        config["enabled_modules"] = [args.module]

    try:
        # Initialize runner
        runner = PenetrationTestRunner(config)

        # Run tests
        if args.module == "all":
            results = asyncio.run(runner.run_all_tests())
        else:
            results = asyncio.run(runner.run_specific_module(args.module))

        # Print summary
        summary = results.get("executive_summary", {})
        print(f"\n{'='*60}")
        print("PENETRATION TESTING SUMMARY")
        print(f"{'='*60}")
        print(f"Total Vulnerabilities: {summary.get('total_vulnerabilities', 0)}")
        print(f"Risk Score: {summary.get('risk_score', 0)}/100")

        severity_dist = summary.get("severity_distribution", {})
        print(f"Critical: {severity_dist.get('critical', 0)}")
        print(f"High: {severity_dist.get('high', 0)}")
        print(f"Medium: {severity_dist.get('medium', 0)}")
        print(f"Low: {severity_dist.get('low', 0)}")
        print(f"Info: {severity_dist.get('info', 0)}")

        execution_summary = results.get("execution_summary", {})
        if "report_files" in execution_summary:
            print("\nReports generated:")
            for report_file in execution_summary["report_files"]:
                print(f"  - {report_file}")

        print(f"{'='*60}")

        # Exit with appropriate code
        critical_high_count = severity_dist.get("critical", 0) + severity_dist.get("high", 0)
        if critical_high_count > 0:
            sys.exit(1)  # Exit with error if critical/high vulnerabilities found
        else:
            sys.exit(0)

    except Exception as e:
        logger.error(f"Penetration testing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
