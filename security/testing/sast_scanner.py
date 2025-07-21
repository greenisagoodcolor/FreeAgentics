"""
Static Application Security Testing (SAST) Scanner

Integrates multiple SAST tools to scan Python code for security vulnerabilities.
Implements severity thresholds and custom Semgrep rules.
"""

import json
import logging
import subprocess  # nosec B404 # Required for SAST security tool execution
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Severity(Enum):
    """Vulnerability severity levels"""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def weight(self) -> int:
        """Get numerical weight for severity comparison"""
        weights: Dict["Severity", int] = {
            Severity.CRITICAL: 4,
            Severity.HIGH: 3,
            Severity.MEDIUM: 2,
            Severity.LOW: 1,
            Severity.INFO: 0,
        }
        return weights[self]

    def __ge__(self, other: "Severity") -> bool:
        """Compare severity levels"""
        return self.weight >= other.weight


@dataclass
class Finding:
    """Security finding from SAST scan"""

    tool: str
    rule_id: str
    severity: Severity
    file_path: str
    line_number: int
    message: str
    category: str
    confidence: str = "medium"
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None
    fix_guidance: Optional[str] = None
    code_snippet: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to dictionary"""
        return {
            "tool": self.tool,
            "rule_id": self.rule_id,
            "severity": self.severity.value,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "message": self.message,
            "category": self.category,
            "confidence": self.confidence,
            "cwe_id": self.cwe_id,
            "owasp_category": self.owasp_category,
            "fix_guidance": self.fix_guidance,
            "code_snippet": self.code_snippet,
            "metadata": self.metadata,
        }


@dataclass
class ScanConfig:
    """Configuration for SAST scanning"""

    project_root: Path
    include_paths: List[str] = field(default_factory=lambda: ["**/*.py"])
    exclude_paths: List[str] = field(
        default_factory=lambda: ["venv/", "__pycache__/", "*.pyc", ".git/"]
    )
    severity_threshold: Severity = Severity.MEDIUM
    fail_on_threshold: bool = True
    enable_bandit: bool = True
    enable_semgrep: bool = True
    enable_safety: bool = True
    custom_semgrep_rules: Optional[Path] = None
    output_format: str = "json"
    max_findings: int = 1000
    suppress_rules: Set[str] = field(default_factory=set)


class BanditScanner:
    """Bandit security scanner integration"""

    def __init__(self, config: ScanConfig):
        self.config = config

    def scan(self) -> List[Finding]:
        """Run Bandit scan"""
        logger.info("Running Bandit security scan...")
        findings: List[Finding] = []

        try:
            # Build Bandit command
            cmd = [
                "bandit",
                "-r",
                str(self.config.project_root),
                "-f",
                "json",
                "-ll",  # Only report medium severity and above
            ]

            # Add excluded paths
            if self.config.exclude_paths:
                excludes = ",".join(
                    f"{self.config.project_root}/{p}" for p in self.config.exclude_paths
                )
                cmd.extend(["-x", excludes])

            # Run Bandit
            result = subprocess.run(cmd, capture_output=True, text=True)  # nosec B603 # Safe bandit security scanner execution

            if result.returncode not in (
                0,
                1,
            ):  # Bandit returns 1 when issues found
                logger.error(f"Bandit scan failed: {result.stderr}")
                return findings

            # Parse results
            if result.stdout:
                data = json.loads(result.stdout)
                for result_item in data.get("results", []):
                    finding = Finding(
                        tool="bandit",
                        rule_id=result_item["test_id"],
                        severity=self._map_severity(result_item["issue_severity"]),
                        file_path=result_item["filename"],
                        line_number=result_item["line_number"],
                        message=result_item["issue_text"],
                        category=result_item["test_name"],
                        confidence=result_item["issue_confidence"].lower(),
                        cwe_id=result_item.get("issue_cwe", {}).get("id"),
                        code_snippet=result_item.get("code"),
                        metadata={
                            "line_range": result_item.get("line_range", []),
                            "more_info": result_item.get("more_info"),
                        },
                    )

                    # Check if rule is suppressed
                    if finding.rule_id not in self.config.suppress_rules:
                        findings.append(finding)

        except Exception as e:
            logger.error(f"Error running Bandit scan: {e}")

        logger.info(f"Bandit found {len(findings)} issues")
        return findings

    def _map_severity(self, bandit_severity: str) -> Severity:
        """Map Bandit severity to our severity enum"""
        mapping = {
            "UNDEFINED": Severity.INFO,
            "LOW": Severity.LOW,
            "MEDIUM": Severity.MEDIUM,
            "HIGH": Severity.HIGH,
        }
        return mapping.get(bandit_severity.upper(), Severity.INFO)


class SemgrepScanner:
    """Semgrep security scanner integration"""

    def __init__(self, config: ScanConfig):
        self.config = config

    def scan(self) -> List[Finding]:
        """Run Semgrep scan"""
        logger.info("Running Semgrep security scan...")
        findings: List[Finding] = []

        try:
            # Build Semgrep command
            cmd = [
                "semgrep",
                "--config=auto",  # Use default security rules
                str(self.config.project_root),
                "--json",
                "--no-error",
                "--metrics=off",
            ]

            # Add custom rules if specified
            if (
                self.config.custom_semgrep_rules
                and self.config.custom_semgrep_rules.exists()
            ):
                cmd.extend(["--config", str(self.config.custom_semgrep_rules)])

            # Add exclude patterns
            for pattern in self.config.exclude_paths:
                cmd.extend(["--exclude", pattern])

            # Run Semgrep
            result = subprocess.run(cmd, capture_output=True, text=True)  # nosec B603 # Safe semgrep security scanner execution

            if result.returncode != 0 and "No rules" not in result.stderr:
                logger.error(f"Semgrep scan failed: {result.stderr}")
                return findings

            # Parse results
            if result.stdout:
                data = json.loads(result.stdout)
                for result_item in data.get("results", []):
                    finding = Finding(
                        tool="semgrep",
                        rule_id=result_item["check_id"],
                        severity=self._map_severity(
                            result_item.get("extra", {}).get("severity", "INFO")
                        ),
                        file_path=result_item["path"],
                        line_number=result_item["start"]["line"],
                        message=result_item.get("extra", {}).get(
                            "message", result_item["check_id"]
                        ),
                        category=self._get_category(result_item),
                        confidence="high",
                        owasp_category=self._get_owasp_category(result_item),
                        fix_guidance=result_item.get("extra", {}).get("fix"),
                        code_snippet=self._get_code_snippet(result_item),
                        metadata={
                            "end_line": result_item["end"]["line"],
                            "column": result_item["start"]["col"],
                            "end_column": result_item["end"]["col"],
                            "metavars": result_item.get("extra", {}).get(
                                "metavars", {}
                            ),
                        },
                    )

                    # Check if rule is suppressed
                    if finding.rule_id not in self.config.suppress_rules:
                        findings.append(finding)

        except Exception as e:
            logger.error(f"Error running Semgrep scan: {e}")

        logger.info(f"Semgrep found {len(findings)} issues")
        return findings

    def _map_severity(self, semgrep_severity: str) -> Severity:
        """Map Semgrep severity to our severity enum"""
        mapping = {
            "ERROR": Severity.HIGH,
            "WARNING": Severity.MEDIUM,
            "INFO": Severity.LOW,
            "INVENTORY": Severity.INFO,
        }
        return mapping.get(semgrep_severity.upper(), Severity.INFO)

    def _get_category(self, result: Dict[str, Any]) -> str:
        """Extract category from Semgrep result"""
        extra = result.get("extra", {})
        metadata = extra.get("metadata", {})
        return str(metadata.get("category", "security"))

    def _get_owasp_category(self, result: Dict[str, Any]) -> Optional[str]:
        """Extract OWASP category from Semgrep result"""
        extra = result.get("extra", {})
        metadata = extra.get("metadata", {})
        owasp = metadata.get("owasp", [])
        return str(owasp[0]) if owasp else None

    def _get_code_snippet(self, result: Dict[str, Any]) -> Optional[str]:
        """Extract code snippet from Semgrep result"""
        extra = result.get("extra", {})
        lines = extra.get("lines", "")
        return str(lines) if lines else None


class SafetyScanner:
    """Safety dependency scanner integration"""

    def __init__(self, config: ScanConfig):
        self.config = config

    def scan(self) -> List[Finding]:
        """Run Safety scan for dependency vulnerabilities"""
        logger.info("Running Safety dependency scan...")
        findings: List[Finding] = []

        try:
            # Find requirements files
            req_files = list(self.config.project_root.glob("*requirements*.txt"))
            if not req_files:
                logger.warning("No requirements files found for Safety scan")
                return findings

            for req_file in req_files:
                # Run Safety check
                cmd = ["safety", "check", "--json", "--file", str(req_file)]
                result = subprocess.run(cmd, capture_output=True, text=True)  # nosec B603 # Safe safety security scanner execution

                if result.returncode != 0 and result.stdout:
                    # Parse vulnerabilities
                    data = json.loads(result.stdout)
                    for vuln in data:
                        finding = Finding(
                            tool="safety",
                            rule_id=f"CVE-{vuln.get('cve', 'UNKNOWN')}",
                            severity=self._map_severity(
                                vuln.get("severity", "unknown")
                            ),
                            file_path=str(req_file),
                            line_number=0,  # Not applicable for dependencies
                            message=vuln.get(
                                "advisory", "Vulnerable dependency detected"
                            ),
                            category="dependency-vulnerability",
                            confidence="high",
                            cwe_id=self._extract_cwe(vuln),
                            owasp_category="A06:2021 - Vulnerable and Outdated Components",
                            fix_guidance=f"Update {vuln.get('package_name')} to version {vuln.get('safe_version', 'latest')}",
                            metadata={
                                "package": vuln.get("package_name"),
                                "installed_version": vuln.get("installed_version"),
                                "vulnerable_spec": vuln.get("vulnerable_spec"),
                                "safe_version": vuln.get("safe_version"),
                            },
                        )
                        findings.append(finding)

        except Exception as e:
            logger.error(f"Error running Safety scan: {e}")

        logger.info(f"Safety found {len(findings)} vulnerable dependencies")
        return findings

    def _map_severity(self, safety_severity: str) -> Severity:
        """Map Safety severity to our severity enum"""
        # Safety doesn't provide standardized severity, so we'll estimate
        if "critical" in safety_severity.lower():
            return Severity.CRITICAL
        elif "high" in safety_severity.lower():
            return Severity.HIGH
        elif (
            "medium" in safety_severity.lower() or "moderate" in safety_severity.lower()
        ):
            return Severity.MEDIUM
        else:
            return Severity.LOW

    def _extract_cwe(self, vuln: Dict[str, Any]) -> Optional[str]:
        """Extract CWE ID from vulnerability data"""
        advisory = vuln.get("advisory", "")
        # Simple CWE extraction from advisory text
        import re

        match = re.search(r"CWE-(\d+)", advisory)
        return f"CWE-{match.group(1)}" if match else None


class CustomSemgrepRules:
    """Custom Semgrep rules for project-specific patterns"""

    @staticmethod
    def generate_custom_rules() -> Dict[str, Any]:
        """Generate custom Semgrep rules for FreeAgentics project"""
        return {
            "rules": [
                {
                    "id": "freeagentics-hardcoded-secrets",
                    "patterns": [
                        {
                            "pattern": '$KEY = "..."',
                            "metavariable-regex": {
                                "metavariable": "$KEY",
                                "regex": ".*(?:SECRET|KEY|TOKEN|PASSWORD).*",
                            },
                        },
                    ],
                    "message": "Hardcoded secret detected. Use environment variables instead.",
                    "severity": "ERROR",
                    "metadata": {
                        "category": "security",
                        "owasp": [
                            "A07:2021 - Identification and Authentication Failures"
                        ],
                    },
                },
                {
                    "id": "freeagentics-sql-injection",
                    "patterns": [
                        {"pattern": 'execute(f"... {$VAR} ...")'},
                        {"pattern": 'execute("... " + $VAR + " ...")'},
                    ],
                    "message": "Potential SQL injection. Use parameterized queries.",
                    "severity": "ERROR",
                    "metadata": {
                        "category": "security",
                        "owasp": ["A03:2021 - Injection"],
                        "cwe": ["CWE-89"],
                    },
                },
                {
                    "id": "freeagentics-insecure-random",
                    "patterns": [
                        {
                            "pattern": "random.$FUNC(...)",
                            "metavariable-regex": {
                                "metavariable": "$FUNC",
                                "regex": "^(?!SystemRandom).*",
                            },
                        }
                    ],
                    "message": "Use secrets module for cryptographic randomness",
                    "severity": "WARNING",
                    "metadata": {
                        "category": "security",
                        "owasp": ["A02:2021 - Cryptographic Failures"],
                    },
                },
                {
                    "id": "freeagentics-eval-usage",
                    "patterns": [{"pattern": "eval(...)"}],
                    "message": "Avoid using eval() as it can lead to code injection",
                    "severity": "ERROR",
                    "metadata": {
                        "category": "security",
                        "owasp": ["A03:2021 - Injection"],
                        "cwe": ["CWE-95"],
                    },
                },
                {
                    "id": "freeagentics-pickle-usage",
                    "patterns": [
                        {"pattern": "pickle.loads(...)"},
                        {"pattern": "pickle.load(...)"},
                    ],
                    "message": "Pickle deserialization can lead to arbitrary code execution",
                    "severity": "ERROR",
                    "metadata": {
                        "category": "security",
                        "owasp": ["A08:2021 - Software and Data Integrity Failures"],
                        "cwe": ["CWE-502"],
                    },
                },
                {
                    "id": "freeagentics-jwt-weak-secret",
                    "patterns": [
                        {
                            "pattern": 'jwt.encode(..., "...", ...)',
                            "metavariable-regex": {
                                "metavariable": '"..."',
                                "regex": '^".{0,32}"$',
                            },
                        },
                    ],
                    "message": "JWT secret key appears to be weak (less than 32 characters)",
                    "severity": "ERROR",
                    "metadata": {
                        "category": "security",
                        "owasp": ["A02:2021 - Cryptographic Failures"],
                    },
                },
                {
                    "id": "freeagentics-path-traversal",
                    "patterns": [
                        {
                            "pattern": "open($PATH, ...)",
                            "metavariable-regex": {
                                "metavariable": "$PATH",
                                "regex": ".*\\.\\./.*",
                            },
                        },
                        {
                            "pattern": "Path($PATH)",
                            "metavariable-regex": {
                                "metavariable": "$PATH",
                                "regex": ".*\\.\\./.*",
                            },
                        },
                    ],
                    "message": "Potential path traversal vulnerability",
                    "severity": "ERROR",
                    "metadata": {
                        "category": "security",
                        "owasp": ["A01:2021 - Broken Access Control"],
                        "cwe": ["CWE-22"],
                    },
                },
                {
                    "id": "freeagentics-xxe-vulnerability",
                    "patterns": [
                        {"pattern": "etree.parse(..., parser=None)"},
                        {"pattern": "etree.XMLParser(resolve_entities=True)"},
                    ],
                    "message": "XML parsing without disabling external entities can lead to XXE",
                    "severity": "ERROR",
                    "metadata": {
                        "category": "security",
                        "owasp": ["A03:2021 - Injection"],
                        "cwe": ["CWE-611"],
                    },
                },
            ]
        }

    @staticmethod
    def save_rules(output_path: Path) -> None:
        """Save custom rules to file"""
        rules = CustomSemgrepRules.generate_custom_rules()
        with open(output_path, "w") as f:
            yaml.dump(rules, f, default_flow_style=False)


class SASTScanner:
    """Main SAST scanner orchestrator"""

    def __init__(self, config: ScanConfig):
        self.config = config
        self.findings: List[Finding] = []

    def scan(self) -> Tuple[List[Finding], bool]:
        """
        Run all configured SAST scans

        Returns:
            Tuple of (findings, pass/fail based on threshold)
        """
        logger.info(f"Starting SAST scan of {self.config.project_root}")

        # Save custom Semgrep rules if needed
        if self.config.enable_semgrep and not self.config.custom_semgrep_rules:
            rules_path = self.config.project_root / ".semgrep-rules.yml"
            CustomSemgrepRules.save_rules(rules_path)
            self.config.custom_semgrep_rules = rules_path

        # Run Bandit scan
        if self.config.enable_bandit:
            bandit_scanner = BanditScanner(self.config)
            self.findings.extend(bandit_scanner.scan())

        # Run Semgrep scan
        if self.config.enable_semgrep:
            semgrep_scanner = SemgrepScanner(self.config)
            self.findings.extend(semgrep_scanner.scan())

        # Run Safety scan
        if self.config.enable_safety:
            safety_scanner = SafetyScanner(self.config)
            self.findings.extend(safety_scanner.scan())

        # Deduplicate findings
        self.findings = self._deduplicate_findings(self.findings)

        # Sort by severity
        self.findings.sort(key=lambda f: f.severity.weight, reverse=True)

        # Limit findings if needed
        if len(self.findings) > self.config.max_findings:
            logger.warning(
                f"Limiting findings from {len(self.findings)} to {self.config.max_findings}"
            )
            self.findings = self.findings[: self.config.max_findings]

        # Check threshold
        passed = self._check_threshold()

        # Generate summary
        self._print_summary()

        return self.findings, passed

    def _deduplicate_findings(self, findings: List[Finding]) -> List[Finding]:
        """Remove duplicate findings"""
        seen = set()
        unique_findings = []

        for finding in findings:
            # Create unique key
            key = (
                finding.tool,
                finding.rule_id,
                finding.file_path,
                finding.line_number,
            )
            if key not in seen:
                seen.add(key)
                unique_findings.append(finding)

        return unique_findings

    def _check_threshold(self) -> bool:
        """Check if findings meet severity threshold"""
        if not self.config.fail_on_threshold:
            return True

        for finding in self.findings:
            if finding.severity >= self.config.severity_threshold:
                return False

        return True

    def _print_summary(self) -> None:
        """Print scan summary"""
        severity_counts = {severity: 0 for severity in Severity}
        tool_counts: Dict[str, int] = {}

        for finding in self.findings:
            severity_counts[finding.severity] += 1
            tool_counts[finding.tool] = tool_counts.get(finding.tool, 0) + 1

        logger.info("\n" + "=" * 60)
        logger.info("SAST Scan Summary")
        logger.info("=" * 60)
        logger.info(f"Total findings: {len(self.findings)}")

        logger.info("\nBy Severity:")
        for severity in Severity:
            count = severity_counts[severity]
            if count > 0:
                logger.info(f"  {severity.value.upper()}: {count}")

        logger.info("\nBy Tool:")
        for tool, count in tool_counts.items():
            logger.info(f"  {tool}: {count}")

        logger.info("=" * 60)

    def export_findings(self, output_path: Path) -> None:
        """Export findings to file"""
        data = {
            "scan_info": {
                "project_root": str(self.config.project_root),
                "severity_threshold": self.config.severity_threshold.value,
                "total_findings": len(self.findings),
                "tools_used": list(set(f.tool for f in self.findings)),
            },
            "findings": [f.to_dict() for f in self.findings],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Findings exported to {output_path}")


def main():
    """Main entry point for SAST scanner"""
    import argparse

    parser = argparse.ArgumentParser(description="SAST Security Scanner")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory",
    )
    parser.add_argument(
        "--severity-threshold",
        choices=["critical", "high", "medium", "low", "info"],
        default="medium",
        help="Minimum severity to fail the scan",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sast-findings.json"),
        help="Output file for findings",
    )
    parser.add_argument(
        "--no-bandit", action="store_true", help="Disable Bandit scanner"
    )
    parser.add_argument(
        "--no-semgrep", action="store_true", help="Disable Semgrep scanner"
    )
    parser.add_argument(
        "--no-safety", action="store_true", help="Disable Safety scanner"
    )
    parser.add_argument("--suppress-rules", nargs="+", help="Rules to suppress")

    args = parser.parse_args()

    # Create configuration
    config = ScanConfig(
        project_root=args.project_root,
        severity_threshold=Severity(args.severity_threshold),
        enable_bandit=not args.no_bandit,
        enable_semgrep=not args.no_semgrep,
        enable_safety=not args.no_safety,
        suppress_rules=set(args.suppress_rules) if args.suppress_rules else set(),
    )

    # Run scan
    scanner = SASTScanner(config)
    findings, passed = scanner.scan()

    # Export findings
    scanner.export_findings(args.output)

    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
