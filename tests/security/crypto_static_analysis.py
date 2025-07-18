"""
Cryptographic Static Code Analysis Tool

This module performs static analysis of the codebase to identify
cryptographic vulnerabilities, weak algorithms, and implementation issues.
"""

import ast
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from tests.security.cryptography_assessment_config import SecurityLevel

logger = logging.getLogger(__name__)


class VulnerabilityType(Enum):
    """Types of cryptographic vulnerabilities."""

    WEAK_ALGORITHM = "weak_algorithm"
    HARDCODED_SECRET = "hardcoded_secret"
    WEAK_RANDOM = "weak_random"
    INSECURE_PROTOCOL = "insecure_protocol"
    KEY_MANAGEMENT = "key_management"
    IMPLEMENTATION_FLAW = "implementation_flaw"
    CONFIGURATION_ISSUE = "configuration_issue"


@dataclass
class CryptoVulnerability:
    """Represents a cryptographic vulnerability found in code."""

    vulnerability_type: VulnerabilityType
    severity: SecurityLevel
    file_path: str
    line_number: int
    line_content: str
    description: str
    recommendation: str
    cwe_id: Optional[str] = None
    owasp_category: Optional[str] = None


class CryptographicPatternMatcher:
    """Pattern matcher for cryptographic vulnerabilities."""

    def __init__(self):
        self.patterns = self._initialize_patterns()

    def _initialize_patterns(
        self,
    ) -> Dict[VulnerabilityType, List[Dict[str, Any]]]:
        """Initialize vulnerability patterns."""
        return {
            VulnerabilityType.WEAK_ALGORITHM: [
                {
                    "pattern": r"hashlib\.md5\(",
                    "severity": SecurityLevel.CRITICAL,
                    "description": "Use of MD5 hash algorithm",
                    "recommendation": "Replace MD5 with SHA-256 or stronger",
                    "cwe_id": "CWE-327",
                },
                {
                    "pattern": r"hashlib\.sha1\(",
                    "severity": SecurityLevel.CRITICAL,
                    "description": "Use of SHA-1 hash algorithm",
                    "recommendation": "Replace SHA-1 with SHA-256 or stronger",
                    "cwe_id": "CWE-327",
                },
                {
                    "pattern": r"Cipher\([^)]*DES[^)]*\)",
                    "severity": SecurityLevel.CRITICAL,
                    "description": "Use of DES encryption",
                    "recommendation": "Replace DES with AES-256",
                    "cwe_id": "CWE-327",
                },
                {
                    "pattern": r"modes\.ECB\(",
                    "severity": SecurityLevel.HIGH,
                    "description": "Use of ECB encryption mode",
                    "recommendation": "Use GCM or CBC mode with proper IV",
                    "cwe_id": "CWE-327",
                },
                {
                    "pattern": r"RC4|rc4",
                    "severity": SecurityLevel.CRITICAL,
                    "description": "Use of RC4 cipher",
                    "recommendation": "Replace RC4 with AES-GCM or ChaCha20-Poly1305",
                    "cwe_id": "CWE-327",
                },
                {
                    "pattern": r"key_size.*1024",
                    "severity": SecurityLevel.HIGH,
                    "description": "Use of 1024-bit RSA key",
                    "recommendation": "Use RSA-2048 or stronger",
                    "cwe_id": "CWE-326",
                },
            ],
            VulnerabilityType.HARDCODED_SECRET: [
                {
                    "pattern": r'(password|secret|key)\s*=\s*["\'][^"\']{8,}["\']',
                    "severity": SecurityLevel.CRITICAL,
                    "description": "Hardcoded secret detected",
                    "recommendation": "Use environment variables or secure key management",
                    "cwe_id": "CWE-798",
                },
                {
                    "pattern": r'SECRET_KEY\s*=\s*["\'](?!.*getenv)[^"\']{20,}["\']',
                    "severity": SecurityLevel.CRITICAL,
                    "description": "Hardcoded SECRET_KEY",
                    "recommendation": "Use environment variable for SECRET_KEY",
                    "cwe_id": "CWE-798",
                },
                {
                    "pattern": r'JWT_SECRET\s*=\s*["\'](?!.*getenv)[^"\']{20,}["\']',
                    "severity": SecurityLevel.CRITICAL,
                    "description": "Hardcoded JWT secret",
                    "recommendation": "Use environment variable for JWT secret",
                    "cwe_id": "CWE-798",
                },
                {
                    "pattern": r'["\'][A-Za-z0-9+/]{40,}={0,2}["\']',
                    "severity": SecurityLevel.MEDIUM,
                    "description": "Potential base64-encoded secret",
                    "recommendation": "Verify if this is a hardcoded secret",
                    "cwe_id": "CWE-798",
                },
            ],
            VulnerabilityType.WEAK_RANDOM: [
                {
                    "pattern": r"random\.random\(",
                    "severity": SecurityLevel.HIGH,
                    "description": "Use of weak random number generator",
                    "recommendation": "Use secrets module for cryptographic randomness",
                    "cwe_id": "CWE-338",
                },
                {
                    "pattern": r"random\.randint\(",
                    "severity": SecurityLevel.HIGH,
                    "description": "Use of weak random integer generator",
                    "recommendation": "Use secrets.randbelow() for cryptographic randomness",
                    "cwe_id": "CWE-338",
                },
                {
                    "pattern": r"Math\.random\(",
                    "severity": SecurityLevel.HIGH,
                    "description": "Use of JavaScript Math.random()",
                    "recommendation": "Use crypto.getRandomValues() for cryptographic randomness",
                    "cwe_id": "CWE-338",
                },
                {
                    "pattern": r"time\(\).*seed",
                    "severity": SecurityLevel.MEDIUM,
                    "description": "Time-based random seed",
                    "recommendation": "Use proper entropy source for seeding",
                    "cwe_id": "CWE-338",
                },
            ],
            VulnerabilityType.INSECURE_PROTOCOL: [
                {
                    "pattern": r"ssl\.PROTOCOL_TLS_?v1[^_]",
                    "severity": SecurityLevel.HIGH,
                    "description": "Use of TLS 1.0/1.1",
                    "recommendation": "Use TLS 1.2 or 1.3 minimum",
                    "cwe_id": "CWE-327",
                },
                {
                    "pattern": r"ssl\.PROTOCOL_SSLv[23]",
                    "severity": SecurityLevel.CRITICAL,
                    "description": "Use of SSL 2.0/3.0",
                    "recommendation": "Use TLS 1.2 or 1.3",
                    "cwe_id": "CWE-327",
                },
                {
                    "pattern": r"verify_mode\s*=\s*ssl\.CERT_NONE",
                    "severity": SecurityLevel.HIGH,
                    "description": "Certificate verification disabled",
                    "recommendation": "Enable certificate verification",
                    "cwe_id": "CWE-295",
                },
                {
                    "pattern": r"check_hostname\s*=\s*False",
                    "severity": SecurityLevel.MEDIUM,
                    "description": "Hostname verification disabled",
                    "recommendation": "Enable hostname verification",
                    "cwe_id": "CWE-295",
                },
            ],
            VulnerabilityType.KEY_MANAGEMENT: [
                {
                    "pattern": r"private_key.*\.write\(",
                    "severity": SecurityLevel.MEDIUM,
                    "description": "Private key written to file",
                    "recommendation": "Ensure proper file permissions (600)",
                    "cwe_id": "CWE-312",
                },
                {
                    "pattern": r"NoEncryption\(\)",
                    "severity": SecurityLevel.MEDIUM,
                    "description": "Unencrypted private key storage",
                    "recommendation": "Consider encrypting private keys",
                    "cwe_id": "CWE-312",
                },
                {
                    "pattern": r"password=None",
                    "severity": SecurityLevel.LOW,
                    "description": "No password for key encryption",
                    "recommendation": "Consider using password protection for keys",
                    "cwe_id": "CWE-312",
                },
            ],
            VulnerabilityType.IMPLEMENTATION_FLAW: [
                {
                    "pattern": r"==.*password",
                    "severity": SecurityLevel.MEDIUM,
                    "description": "Non-constant time password comparison",
                    "recommendation": "Use constant-time comparison (hmac.compare_digest)",
                    "cwe_id": "CWE-208",
                },
                {
                    "pattern": r"if.*token.*==",
                    "severity": SecurityLevel.MEDIUM,
                    "description": "Non-constant time token comparison",
                    "recommendation": "Use constant-time comparison",
                    "cwe_id": "CWE-208",
                },
                {
                    "pattern": r'iv\s*=\s*["\'][^"\']+["\']',
                    "severity": SecurityLevel.HIGH,
                    "description": "Hardcoded initialization vector",
                    "recommendation": "Generate random IV for each encryption",
                    "cwe_id": "CWE-329",
                },
                {
                    "pattern": r'salt\s*=\s*["\'][^"\']+["\']',
                    "severity": SecurityLevel.HIGH,
                    "description": "Hardcoded salt",
                    "recommendation": "Generate random salt for each hash",
                    "cwe_id": "CWE-329",
                },
            ],
            VulnerabilityType.CONFIGURATION_ISSUE: [
                {
                    "pattern": r"iterations\s*=\s*[0-9]{1,3}(?![0-9])",
                    "severity": SecurityLevel.MEDIUM,
                    "description": "Low iteration count for key derivation",
                    "recommendation": "Use at least 10,000 iterations for PBKDF2",
                    "cwe_id": "CWE-326",
                },
                {
                    "pattern": r"rounds\s*=\s*[0-9](?![0-9])",
                    "severity": SecurityLevel.MEDIUM,
                    "description": "Low rounds for bcrypt",
                    "recommendation": "Use at least 10 rounds for bcrypt",
                    "cwe_id": "CWE-326",
                },
            ],
        }


class CryptographicStaticAnalyzer:
    """Static analyzer for cryptographic vulnerabilities."""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.pattern_matcher = CryptographicPatternMatcher()
        self.vulnerabilities = []
        self.file_extensions = {
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".go",
            ".rs",
        }

    def analyze_project(self) -> List[CryptoVulnerability]:
        """Analyze the entire project for cryptographic vulnerabilities."""
        logger.info(
            f"Starting cryptographic static analysis of {self.project_root}"
        )

        # Get all source files
        source_files = self._get_source_files()

        logger.info(f"Analyzing {len(source_files)} source files")

        for file_path in source_files:
            try:
                self._analyze_file(file_path)
            except Exception as e:
                logger.warning(f"Error analyzing {file_path}: {e}")

        logger.info(
            f"Analysis complete. Found {len(self.vulnerabilities)} potential issues"
        )
        return self.vulnerabilities

    def _get_source_files(self) -> List[Path]:
        """Get all source files in the project."""
        source_files = []

        # Directories to exclude
        exclude_dirs = {
            ".git",
            "__pycache__",
            "node_modules",
            ".venv",
            "venv",
            "build",
            "dist",
            ".pytest_cache",
            ".mypy_cache",
        }

        for file_path in self.project_root.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix in self.file_extensions
                and not any(
                    exclude_dir in file_path.parts
                    for exclude_dir in exclude_dirs
                )
            ):
                source_files.append(file_path)

        return source_files

    def _analyze_file(self, file_path: Path):
        """Analyze a single file for cryptographic vulnerabilities."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            lines = content.split("\n")

            for line_num, line in enumerate(lines, 1):
                self._analyze_line(file_path, line_num, line)

            # Additional analysis for Python files
            if file_path.suffix == ".py":
                self._analyze_python_ast(file_path, content)

        except Exception as e:
            logger.warning(f"Error reading {file_path}: {e}")

    def _analyze_line(
        self, file_path: Path, line_number: int, line_content: str
    ):
        """Analyze a single line for cryptographic vulnerabilities."""
        line_stripped = line_content.strip()

        # Skip comments and empty lines
        if not line_stripped or line_stripped.startswith(("#", "//", "/*")):
            return

        # Check all vulnerability patterns
        for vuln_type, patterns in self.pattern_matcher.patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]

                if re.search(pattern, line_content, re.IGNORECASE):
                    vulnerability = CryptoVulnerability(
                        vulnerability_type=vuln_type,
                        severity=pattern_info["severity"],
                        file_path=str(
                            file_path.relative_to(self.project_root)
                        ),
                        line_number=line_number,
                        line_content=line_stripped,
                        description=pattern_info["description"],
                        recommendation=pattern_info["recommendation"],
                        cwe_id=pattern_info.get("cwe_id"),
                        owasp_category=self._map_to_owasp_category(vuln_type),
                    )

                    # Additional context-based filtering
                    if self._is_false_positive(vulnerability):
                        continue

                    self.vulnerabilities.append(vulnerability)

    def _analyze_python_ast(self, file_path: Path, content: str):
        """Perform AST-based analysis for Python files."""
        try:
            tree = ast.parse(content)
            visitor = CryptoASTVisitor(file_path, self.project_root)
            visitor.visit(tree)
            self.vulnerabilities.extend(visitor.vulnerabilities)
        except SyntaxError:
            # Skip files with syntax errors
            pass
        except Exception as e:
            logger.warning(f"AST analysis error for {file_path}: {e}")

    def _is_false_positive(self, vulnerability: CryptoVulnerability) -> bool:
        """Check if vulnerability is likely a false positive."""
        line_lower = vulnerability.line_content.lower()

        # Skip test files and comments
        if (
            "test_" in vulnerability.file_path
            or "/test" in vulnerability.file_path
        ):
            return True

        # Skip documentation strings
        if '"""' in line_lower or "'''" in line_lower:
            return True

        # Skip example or demo code
        if any(
            keyword in line_lower
            for keyword in ["example", "demo", "sample", "placeholder"]
        ):
            return True

        # Context-specific filters
        if (
            vulnerability.vulnerability_type
            == VulnerabilityType.HARDCODED_SECRET
        ):
            # Allow development defaults with proper warnings
            if "dev_" in line_lower and "not_for_production" in line_lower:
                return False  # This is actually a good pattern

        return False

    def _map_to_owasp_category(self, vuln_type: VulnerabilityType) -> str:
        """Map vulnerability type to OWASP category."""
        mapping = {
            VulnerabilityType.WEAK_ALGORITHM: "A2:2021 – Cryptographic Failures",
            VulnerabilityType.HARDCODED_SECRET: "A2:2021 – Cryptographic Failures",
            VulnerabilityType.WEAK_RANDOM: "A2:2021 – Cryptographic Failures",
            VulnerabilityType.INSECURE_PROTOCOL: "A2:2021 – Cryptographic Failures",
            VulnerabilityType.KEY_MANAGEMENT: "A2:2021 – Cryptographic Failures",
            VulnerabilityType.IMPLEMENTATION_FLAW: "A2:2021 – Cryptographic Failures",
            VulnerabilityType.CONFIGURATION_ISSUE: "A5:2021 – Security Misconfiguration",
        }
        return mapping.get(vuln_type, "Unknown")

    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        # Group vulnerabilities by type and severity
        vuln_by_type = {}
        vuln_by_severity = {}
        vuln_by_file = {}

        for vuln in self.vulnerabilities:
            # By type
            if vuln.vulnerability_type not in vuln_by_type:
                vuln_by_type[vuln.vulnerability_type] = []
            vuln_by_type[vuln.vulnerability_type].append(vuln)

            # By severity
            if vuln.severity not in vuln_by_severity:
                vuln_by_severity[vuln.severity] = []
            vuln_by_severity[vuln.severity].append(vuln)

            # By file
            if vuln.file_path not in vuln_by_file:
                vuln_by_file[vuln.file_path] = []
            vuln_by_file[vuln.file_path].append(vuln)

        # Calculate risk score
        risk_score = self._calculate_risk_score()

        return {
            "analysis_summary": {
                "total_vulnerabilities": len(self.vulnerabilities),
                "critical_count": len(
                    vuln_by_severity.get(SecurityLevel.CRITICAL, [])
                ),
                "high_count": len(
                    vuln_by_severity.get(SecurityLevel.HIGH, [])
                ),
                "medium_count": len(
                    vuln_by_severity.get(SecurityLevel.MEDIUM, [])
                ),
                "low_count": len(vuln_by_severity.get(SecurityLevel.LOW, [])),
                "risk_score": risk_score,
                "files_analyzed": len(self._get_source_files()),
                "files_with_issues": len(vuln_by_file),
            },
            "vulnerabilities_by_type": {
                vuln_type.value: [self._vuln_to_dict(v) for v in vulns]
                for vuln_type, vulns in vuln_by_type.items()
            },
            "vulnerabilities_by_severity": {
                severity.value: [self._vuln_to_dict(v) for v in vulns]
                for severity, vulns in vuln_by_severity.items()
            },
            "vulnerabilities_by_file": {
                file_path: [self._vuln_to_dict(v) for v in vulns]
                for file_path, vulns in vuln_by_file.items()
            },
            "recommendations": self._generate_recommendations(),
            "top_risk_files": self._identify_top_risk_files(vuln_by_file),
        }

    def _calculate_risk_score(self) -> float:
        """Calculate overall risk score (0-100)."""
        if not self.vulnerabilities:
            return 0.0

        score = 0
        for vuln in self.vulnerabilities:
            if vuln.severity == SecurityLevel.CRITICAL:
                score += 25
            elif vuln.severity == SecurityLevel.HIGH:
                score += 15
            elif vuln.severity == SecurityLevel.MEDIUM:
                score += 5
            elif vuln.severity == SecurityLevel.LOW:
                score += 1

        # Cap at 100
        return min(100.0, score)

    def _vuln_to_dict(self, vuln: CryptoVulnerability) -> Dict[str, Any]:
        """Convert vulnerability to dictionary."""
        return {
            "type": vuln.vulnerability_type.value,
            "severity": vuln.severity.value,
            "file_path": vuln.file_path,
            "line_number": vuln.line_number,
            "line_content": vuln.line_content,
            "description": vuln.description,
            "recommendation": vuln.recommendation,
            "cwe_id": vuln.cwe_id,
            "owasp_category": vuln.owasp_category,
        }

    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate prioritized recommendations."""
        recommendations = []

        # Critical issues first
        critical_vulns = [
            v
            for v in self.vulnerabilities
            if v.severity == SecurityLevel.CRITICAL
        ]
        if critical_vulns:
            recommendations.append(
                {
                    "priority": "CRITICAL",
                    "title": "Address Critical Cryptographic Vulnerabilities",
                    "description": f"Found {len(critical_vulns)} critical cryptographic issues",
                    "action_items": list(
                        set([v.recommendation for v in critical_vulns[:5]])
                    ),
                }
            )

        # Type-specific recommendations
        vuln_types = set(v.vulnerability_type for v in self.vulnerabilities)

        if VulnerabilityType.WEAK_ALGORITHM in vuln_types:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "title": "Upgrade Cryptographic Algorithms",
                    "description": "Replace deprecated cryptographic algorithms",
                    "action_items": [
                        "Replace MD5 and SHA-1 with SHA-256 or stronger",
                        "Replace DES and 3DES with AES-256",
                        "Replace RC4 with ChaCha20-Poly1305 or AES-GCM",
                    ],
                }
            )

        if VulnerabilityType.HARDCODED_SECRET in vuln_types:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "title": "Eliminate Hardcoded Secrets",
                    "description": "Remove hardcoded secrets from source code",
                    "action_items": [
                        "Use environment variables for secrets",
                        "Implement secure key management system",
                        "Audit code for additional hardcoded credentials",
                    ],
                }
            )

        return recommendations

    def _identify_top_risk_files(
        self, vuln_by_file: Dict[str, List]
    ) -> List[Dict[str, Any]]:
        """Identify files with highest risk."""
        file_risks = []

        for file_path, vulns in vuln_by_file.items():
            risk_score = sum(
                (
                    25
                    if v.severity == SecurityLevel.CRITICAL
                    else (
                        15
                        if v.severity == SecurityLevel.HIGH
                        else 5
                        if v.severity == SecurityLevel.MEDIUM
                        else 1
                    )
                )
                for v in vulns
            )

            file_risks.append(
                {
                    "file_path": file_path,
                    "risk_score": risk_score,
                    "vulnerability_count": len(vulns),
                    "critical_count": len(
                        [
                            v
                            for v in vulns
                            if v.severity == SecurityLevel.CRITICAL
                        ]
                    ),
                    "high_count": len(
                        [v for v in vulns if v.severity == SecurityLevel.HIGH]
                    ),
                }
            )

        # Sort by risk score and return top 10
        file_risks.sort(key=lambda x: x["risk_score"], reverse=True)
        return file_risks[:10]


class CryptoASTVisitor(ast.NodeVisitor):
    """AST visitor for Python-specific cryptographic analysis."""

    def __init__(self, file_path: Path, project_root: Path):
        self.file_path = file_path
        self.project_root = project_root
        self.vulnerabilities = []
        self.current_line = 1

    def visit_Call(self, node):
        """Visit function calls for cryptographic analysis."""
        try:
            # Track line number
            if hasattr(node, "lineno"):
                self.current_line = node.lineno

            # Analyze function calls
            func_name = self._get_function_name(node)

            if func_name:
                self._check_crypto_function_call(node, func_name)

        except Exception as e:
            logger.debug(f"AST visitor error: {e}")

        self.generic_visit(node)

    def visit_Assign(self, node):
        """Visit assignments for hardcoded values."""
        try:
            if hasattr(node, "lineno"):
                self.current_line = node.lineno

            # Check for hardcoded crypto values
            for target in node.targets:
                if isinstance(target, ast.Name):
                    var_name = target.id.lower()
                    if any(
                        keyword in var_name
                        for keyword in [
                            "key",
                            "secret",
                            "password",
                            "salt",
                            "iv",
                        ]
                    ):
                        if isinstance(node.value, ast.Str):
                            self._add_vulnerability(
                                VulnerabilityType.HARDCODED_SECRET,
                                SecurityLevel.HIGH,
                                f"Hardcoded value in variable '{target.id}'",
                                "Use environment variables or secure configuration",
                                "CWE-798",
                            )

        except Exception as e:
            logger.debug(f"AST assignment visitor error: {e}")

        self.generic_visit(node)

    def _get_function_name(self, node) -> Optional[str]:
        """Extract function name from call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                return f"{node.func.value.id}.{node.func.attr}"
        return None

    def _check_crypto_function_call(self, node, func_name: str):
        """Check specific cryptographic function calls."""
        func_lower = func_name.lower()

        # Check for weak hash functions
        if any(weak_hash in func_lower for weak_hash in ["md5", "sha1"]):
            self._add_vulnerability(
                VulnerabilityType.WEAK_ALGORITHM,
                SecurityLevel.CRITICAL,
                f"Use of weak hash function: {func_name}",
                "Replace with SHA-256 or stronger",
                "CWE-327",
            )

        # Check for weak random functions
        if "random.random" in func_lower or "random.randint" in func_lower:
            self._add_vulnerability(
                VulnerabilityType.WEAK_RANDOM,
                SecurityLevel.HIGH,
                f"Use of weak random function: {func_name}",
                "Use secrets module for cryptographic randomness",
                "CWE-338",
            )

        # Check for insecure comparisons
        if func_name == "==" and self._is_crypto_comparison(node):
            self._add_vulnerability(
                VulnerabilityType.IMPLEMENTATION_FLAW,
                SecurityLevel.MEDIUM,
                "Non-constant time comparison detected",
                "Use hmac.compare_digest() for secure comparison",
                "CWE-208",
            )

    def _is_crypto_comparison(self, node) -> bool:
        """Check if comparison involves cryptographic values."""
        # This would need more sophisticated analysis
        # For now, return False to avoid false positives
        return False

    def _add_vulnerability(
        self,
        vuln_type: VulnerabilityType,
        severity: SecurityLevel,
        description: str,
        recommendation: str,
        cwe_id: str,
    ):
        """Add vulnerability to the list."""
        vulnerability = CryptoVulnerability(
            vulnerability_type=vuln_type,
            severity=severity,
            file_path=str(self.file_path.relative_to(self.project_root)),
            line_number=self.current_line,
            line_content="",  # Would need to read line content
            description=description,
            recommendation=recommendation,
            cwe_id=cwe_id,
            owasp_category=self._map_to_owasp_category(vuln_type),
        )
        self.vulnerabilities.append(vulnerability)

    def _map_to_owasp_category(self, vuln_type: VulnerabilityType) -> str:
        """Map vulnerability type to OWASP category."""
        return "A2:2021 – Cryptographic Failures"


def main():
    """Main entry point for static analysis."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Cryptographic static code analysis"
    )
    parser.add_argument("project_path", help="Path to project root")
    parser.add_argument("--output", "-o", help="Output file for results")
    parser.add_argument("--format", choices=["json", "text"], default="json")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Run analysis
    analyzer = CryptographicStaticAnalyzer(args.project_path)
    vulnerabilities = analyzer.analyze_project()
    report = analyzer.generate_report()

    # Output results
    if args.output:
        if args.format == "json":
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2, default=str)
        else:
            with open(args.output, "w") as f:
                f.write("Cryptographic Static Analysis Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(
                    f"Total vulnerabilities: {report['analysis_summary']['total_vulnerabilities']}\n"
                )
                f.write(
                    f"Risk score: {report['analysis_summary']['risk_score']}\n\n"
                )

                for vuln in vulnerabilities:
                    f.write(
                        f"{vuln.severity.value.upper()}: {vuln.description}\n"
                    )
                    f.write(f"File: {vuln.file_path}:{vuln.line_number}\n")
                    f.write(f"Recommendation: {vuln.recommendation}\n\n")
    else:
        print(json.dumps(report, indent=2, default=str))

    return 0 if report["analysis_summary"]["critical_count"] == 0 else 1


if __name__ == "__main__":
    exit(main())
