"""
Tests for Security Testing Infrastructure

Tests SAST, DAST, dependency monitoring, and threat intelligence components.
"""

import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from security.testing.dast_integration import (
    Alert,
    APIScanner,
    AuthenticationTester,
    DASTConfig,
    DASTOrchestrator,
    RiskLevel,
    ZAPScanner,
)
from security.testing.dependency_monitor import (
    Dependency,
    DependencyMonitor,
    DependencyScanner,
    MonitorConfig,
    UpdateManager,
    Vulnerability,
    VulnerabilityDatabase,
    VulnerabilitySeverity,
)
from security.testing.sast_scanner import (
    BanditScanner,
    CustomSemgrepRules,
    Finding,
    SafetyScanner,
    SASTScanner,
    ScanConfig,
    SemgrepScanner,
    Severity,
)
from security.testing.threat_intelligence import (
    AbuseIPDBFeed,
    CustomFeed,
    OTXFeed,
    ThreatIndicator,
    ThreatIntelConfig,
    ThreatIntelligenceEngine,
    ThreatLevel,
    ThreatType,
)


class TestSASTScanner:
    """Test SAST scanner functionality"""

    @pytest.fixture
    def scan_config(self, tmp_path):
        """Create test scan configuration"""
        return ScanConfig(
            project_root=tmp_path,
            severity_threshold=Severity.MEDIUM,
            enable_bandit=True,
            enable_semgrep=True,
            enable_safety=True,
        )

    @pytest.fixture
    def test_python_file(self, tmp_path):
        """Create test Python file with vulnerabilities"""
        test_file = tmp_path / "vulnerable.py"
        test_file.write_text(
            """
import pickle
import random
import subprocess

# Hardcoded secret
API_KEY = "sk-1234567890abcdef"
PASSWORD = "admin123"

# SQL injection vulnerability
def get_user(user_id):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return execute(query)

# Command injection
def run_command(cmd):
    subprocess.shell(cmd, shell=True)

# Insecure random
def generate_token():
    return str(random.randint(1000, 9999))

# Pickle vulnerability
def load_data(data):
    return pickle.loads(data)

# Path traversal
def read_file(filename):
    with open(f"../uploads/{filename}") as f:
        return f.read()

# Eval usage
def calculate(expression):
    return eval(expression)
"""
        )
        return test_file

    def test_bandit_scanner(self, scan_config, test_python_file):
        """Test Bandit scanner"""
        scanner = BanditScanner(scan_config)

        with patch("subprocess.run") as mock_run:
            # Mock Bandit output
            mock_run.return_value = Mock(
                returncode=1,
                stdout=json.dumps(
                    {
                        "results": [
                            {
                                "test_id": "B301",
                                "test_name": "pickle",
                                "issue_severity": "HIGH",
                                "issue_confidence": "HIGH",
                                "issue_text": "Pickle library usage",
                                "filename": str(test_python_file),
                                "line_number": 22,
                                "code": "return pickle.loads(data)",
                            }
                        ]
                    }
                ),
                stderr="",
            )

            findings = scanner.scan()

            assert len(findings) == 1
            assert findings[0].tool == "bandit"
            assert findings[0].severity == Severity.HIGH
            assert "pickle" in findings[0].message.lower()

    def test_semgrep_scanner(self, scan_config, test_python_file):
        """Test Semgrep scanner"""
        scanner = SemgrepScanner(scan_config)

        with patch("subprocess.run") as mock_run:
            # Mock Semgrep output
            mock_run.return_value = Mock(
                returncode=0,
                stdout=json.dumps(
                    {
                        "results": [
                            {
                                "check_id": "python.lang.security.insecure-eval",
                                "path": str(test_python_file),
                                "start": {"line": 30, "col": 12},
                                "end": {"line": 30, "col": 30},
                                "extra": {
                                    "severity": "ERROR",
                                    "message": "Eval usage is dangerous",
                                    "metadata": {
                                        "category": "security",
                                        "owasp": ["A03:2021"],
                                    },
                                },
                            }
                        ]
                    }
                ),
                stderr="",
            )

            findings = scanner.scan()

            assert len(findings) == 1
            assert findings[0].tool == "semgrep"
            assert findings[0].severity == Severity.HIGH
            assert findings[0].owasp_category == "A03:2021"

    def test_safety_scanner(self, scan_config, tmp_path):
        """Test Safety scanner"""
        # Create requirements file
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("django==2.2.0\nrequests==2.20.0\n")

        scanner = SafetyScanner(scan_config)

        with patch("subprocess.run") as mock_run:
            # Mock Safety output
            mock_run.return_value = Mock(
                returncode=1,
                stdout=json.dumps(
                    [
                        {
                            "package_name": "django",
                            "installed_version": "2.2.0",
                            "vulnerable_spec": "<2.2.10",
                            "advisory": "Django SQL injection vulnerability",
                            "cve": "2020-7471",
                        }
                    ]
                ),
                stderr="",
            )

            findings = scanner.scan()

            assert len(findings) == 1
            assert findings[0].tool == "safety"
            assert findings[0].category == "dependency-vulnerability"
            assert "CVE-2020-7471" in findings[0].rule_id

    def test_custom_semgrep_rules(self, tmp_path):
        """Test custom Semgrep rules generation"""
        rules_file = tmp_path / "rules.yml"
        CustomSemgrepRules.save_rules(rules_file)

        assert rules_file.exists()

        # Load and verify rules
        import yaml

        with open(rules_file) as f:
            rules_data = yaml.safe_load(f)

        assert "rules" in rules_data
        assert len(rules_data["rules"]) > 0

        # Check for specific rules
        rule_ids = [rule["id"] for rule in rules_data["rules"]]
        assert "freeagentics-hardcoded-secrets" in rule_ids
        assert "freeagentics-sql-injection" in rule_ids
        assert "freeagentics-jwt-weak-secret" in rule_ids

    def test_sast_scanner_orchestration(self, scan_config, test_python_file):
        """Test SAST scanner orchestration"""
        scanner = SASTScanner(scan_config)

        # Mock all sub-scanners
        with (
            patch.object(BanditScanner, "scan") as mock_bandit,
            patch.object(SemgrepScanner, "scan") as mock_semgrep,
            patch.object(SafetyScanner, "scan") as mock_safety,
        ):
            mock_bandit.return_value = [
                Finding(
                    tool="bandit",
                    rule_id="B301",
                    severity=Severity.HIGH,
                    file_path=str(test_python_file),
                    line_number=22,
                    message="Pickle usage",
                    category="security",
                )
            ]

            mock_semgrep.return_value = [
                Finding(
                    tool="semgrep",
                    rule_id="insecure-eval",
                    severity=Severity.CRITICAL,
                    file_path=str(test_python_file),
                    line_number=30,
                    message="Eval usage",
                    category="security",
                )
            ]

            mock_safety.return_value = []

            findings, passed = scanner.scan()

            # Should have 2 findings
            assert len(findings) == 2

            # Should be sorted by severity (critical first)
            assert findings[0].severity == Severity.CRITICAL
            assert findings[1].severity == Severity.HIGH

            # Should fail due to critical finding
            assert not passed

    def test_severity_threshold(self, scan_config):
        """Test severity threshold checking"""
        scan_config.severity_threshold = Severity.HIGH
        scanner = SASTScanner(scan_config)

        # Add findings of different severities
        scanner.findings = [
            Finding(
                tool="test",
                rule_id="1",
                severity=Severity.LOW,
                file_path="test.py",
                line_number=1,
                message="Low severity",
                category="test",
            ),
            Finding(
                tool="test",
                rule_id="2",
                severity=Severity.MEDIUM,
                file_path="test.py",
                line_number=2,
                message="Medium severity",
                category="test",
            ),
        ]

        # Should pass - no HIGH or above
        assert scanner._check_threshold()

        # Add high severity finding
        scanner.findings.append(
            Finding(
                tool="test",
                rule_id="3",
                severity=Severity.HIGH,
                file_path="test.py",
                line_number=3,
                message="High severity",
                category="test",
            )
        )

        # Should fail - HIGH severity found
        assert not scanner._check_threshold()


class TestDependencyMonitor:
    """Test dependency monitoring functionality"""

    @pytest.fixture
    def monitor_config(self, tmp_path):
        """Create test monitor configuration"""
        return MonitorConfig(
            project_root=tmp_path,
            severity_threshold=VulnerabilitySeverity.HIGH,
            auto_update=False,
        )

    @pytest.fixture
    def test_requirements(self, tmp_path):
        """Create test requirements file"""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(
            """
django==3.2.0
requests==2.25.0
numpy==1.19.0
pandas==1.2.0
"""
        )
        return req_file

    def test_dependency_scanner_python(
        self, monitor_config, test_requirements
    ):
        """Test Python dependency scanning"""
        scanner = DependencyScanner(monitor_config)

        with patch.object(scanner, "_get_pip_version") as mock_pip:
            mock_pip.side_effect = ["3.2.0", "2.25.0", "1.19.0", "1.2.0"]

            with patch("subprocess.run") as mock_run:
                # Mock pip list output
                mock_run.return_value = Mock(
                    returncode=0,
                    stdout=json.dumps(
                        [
                            {"name": "django", "version": "3.2.0"},
                            {"name": "requests", "version": "2.25.0"},
                            {"name": "numpy", "version": "1.19.0"},
                            {"name": "pandas", "version": "1.2.0"},
                            {
                                "name": "urllib3",
                                "version": "1.26.0",
                            },  # Transitive
                        ]
                    ),
                )

                dependencies = scanner._scan_python_dependencies()

                assert len(dependencies) == 5

                # Check direct dependencies
                direct_deps = [d for d in dependencies if d.direct]
                assert len(direct_deps) == 4

                # Check transitive dependencies
                transitive_deps = [d for d in dependencies if not d.direct]
                assert len(transitive_deps) == 1
                assert transitive_deps[0].name == "urllib3"

    @pytest.mark.asyncio
    async def test_vulnerability_database_python(self, monitor_config):
        """Test vulnerability checking for Python packages"""
        vuln_db = VulnerabilityDatabase(monitor_config)

        dependencies = [
            Dependency(name="django", version="2.2.0", source="pip"),
            Dependency(name="requests", version="2.20.0", source="pip"),
        ]

        with patch.object(vuln_db, "session") as mock_session:
            # Mock safety database response
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "django": [
                        {
                            "id": "38625",
                            "specs": ["<2.2.10"],
                            "advisory": "Django SQL injection",
                            "cve": "CVE-2020-7471",
                        }
                    ],
                    "requests": [
                        {
                            "id": "38500",
                            "specs": ["<2.21.0"],
                            "advisory": "Requests insufficient verification",
                        }
                    ],
                }
            )

            mock_session.get = AsyncMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            vuln_db.session = mock_session

            await vuln_db._check_python_vulnerabilities(dependencies)

            # Check vulnerabilities were added
            assert len(dependencies[0].vulnerabilities) == 1
            assert len(dependencies[1].vulnerabilities) == 1

            # Check vulnerability details
            django_vuln = dependencies[0].vulnerabilities[0]
            assert django_vuln.package == "django"
            assert django_vuln.cve_ids == ["CVE-2020-7471"]
            assert "SQL injection" in django_vuln.title

    @pytest.mark.asyncio
    async def test_update_manager(self, monitor_config, tmp_path):
        """Test dependency update functionality"""
        update_manager = UpdateManager(monitor_config)

        # Create test requirements
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("django==2.2.0\nrequests==2.20.0\n")

        dependencies = [
            Dependency(
                name="django",
                version="2.2.0",
                source="pip",
                latest_version="4.2.0",
                update_available=True,
                vulnerabilities=[
                    Vulnerability(
                        id="CVE-2020-7471",
                        package="django",
                        installed_version="2.2.0",
                        affected_versions="<2.2.10",
                        fixed_versions=["2.2.10"],
                        severity=VulnerabilitySeverity.HIGH,
                        title="SQL injection",
                        description="",
                        published_date=datetime.now(),
                    )
                ],
            )
        ]

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            # Update dependency
            update_manager._update_pip_dependency(dependencies[0])

            # Check file was updated
            updated_content = req_file.read_text()
            assert "django==4.2.0" in updated_content
            assert "django==2.2.0" not in updated_content

    @pytest.mark.asyncio
    async def test_dependency_monitor_orchestration(self, monitor_config):
        """Test complete dependency monitoring flow"""
        monitor = DependencyMonitor(monitor_config)

        with (
            patch.object(monitor.scanner, "scan_dependencies") as mock_scan,
            patch.object(
                VulnerabilityDatabase, "check_vulnerabilities"
            ) as mock_check_vulns,
            patch.object(
                monitor.update_manager, "check_updates"
            ) as mock_check_updates,
        ):
            mock_dependencies = [
                Dependency(name="django", version="3.2.0", source="pip"),
                Dependency(name="requests", version="2.25.0", source="pip"),
            ]

            mock_scan.return_value = mock_dependencies
            mock_check_vulns.return_value = mock_dependencies
            mock_check_updates.return_value = None

            report = await monitor.check_dependencies()

            assert report["total_dependencies"] == 2
            assert report["vulnerable_dependencies"] == 0
            assert report["critical_vulnerabilities"] == 0


class TestThreatIntelligence:
    """Test threat intelligence functionality"""

    @pytest.fixture
    def threat_config(self):
        """Create test threat intelligence configuration"""
        return ThreatIntelConfig(
            redis_url="redis://localhost:6379",
            enable_otx=True,
            enable_abuseipdb=True,
            auto_block_threshold=ThreatLevel.HIGH,
        )

    def test_threat_indicator(self):
        """Test threat indicator creation"""
        indicator = ThreatIndicator(
            indicator="192.168.1.100",
            indicator_type="ip",
            threat_types=[ThreatType.BOTNET, ThreatType.MALWARE],
            threat_level=ThreatLevel.HIGH,
            source="TestFeed",
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            confidence=0.8,
        )

        assert indicator.indicator == "192.168.1.100"
        assert ThreatType.BOTNET in indicator.threat_types
        assert indicator.threat_level == ThreatLevel.HIGH

    @pytest.mark.asyncio
    async def test_otx_feed(self, threat_config):
        """Test OTX feed parsing"""
        feed = OTXFeed(threat_config)

        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(
                return_value={
                    "results": [
                        {
                            "id": "12345",
                            "name": "Malware Campaign",
                            "created": "2024-01-01T00:00:00Z",
                            "modified": "2024-01-02T00:00:00Z",
                            "tags": ["malware", "botnet"],
                            "indicators": [
                                {"type": "IPv4", "indicator": "10.0.0.1"},
                                {"type": "domain", "indicator": "evil.com"},
                            ],
                        }
                    ]
                }
            )

            mock_session.get = AsyncMock(return_value=mock_response)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            indicators = await feed.fetch_indicators()

            assert len(indicators) == 2
            assert indicators[0].source == "OTX"
            assert indicators[0].indicator == "10.0.0.1"
            assert indicators[1].indicator == "evil.com"

    @pytest.mark.asyncio
    async def test_threat_intelligence_engine(self, threat_config):
        """Test threat intelligence engine"""
        engine = ThreatIntelligenceEngine(threat_config)

        # Mock Redis client
        engine.redis_client = AsyncMock()
        engine.redis_client.pipeline = MagicMock()
        engine.redis_client.get = AsyncMock(return_value=None)

        # Test IP checking
        result = await engine.check_ip("192.168.1.1")
        assert result is None  # Not in bloom filter

        # Add to bloom filter and mock Redis response
        engine.bloom_filter.add("10.0.0.1")
        engine.redis_client.get = AsyncMock(
            return_value=json.dumps(
                {
                    "threat_types": ["malware", "botnet"],
                    "threat_level": ThreatLevel.HIGH.value,
                    "source": "OTX",
                    "confidence": 0.8,
                    "last_seen": datetime.now().isoformat(),
                }
            )
        )

        result = await engine.check_ip("10.0.0.1")
        assert result is not None
        assert result["threat_level"] == ThreatLevel.HIGH.value

    @pytest.mark.asyncio
    async def test_request_checking(self, threat_config):
        """Test request threat checking"""
        engine = ThreatIntelligenceEngine(threat_config)
        engine.redis_client = AsyncMock()

        # Test request with malicious patterns
        request_data = {
            "source_ip": "10.0.0.1",
            "user_agent": "sqlmap/1.0",
            "path": "/admin/../etc/passwd",
        }

        # Mock IP check
        engine.check_ip = AsyncMock(
            return_value={
                "threat_types": ["scanner"],
                "threat_level": ThreatLevel.MEDIUM.value,
            }
        )

        result = await engine.check_request(request_data)

        assert len(result["threats"]) >= 3  # IP, user agent, path
        assert result["should_block"]  # Should block due to multiple threats

        # Find specific threats
        threat_types = [t["type"] for t in result["threats"]]
        assert "source_ip" in threat_types
        assert "user_agent" in threat_types
        assert "path" in threat_types

    def test_malicious_patterns(self, threat_config):
        """Test malicious pattern detection"""
        engine = ThreatIntelligenceEngine(threat_config)

        # Test user agents
        assert engine._is_malicious_user_agent("sqlmap/1.0")
        assert engine._is_malicious_user_agent("nikto/2.1.5")
        assert engine._is_malicious_user_agent("")  # Empty
        assert not engine._is_malicious_user_agent("Mozilla/5.0 Firefox/91.0")

        # Test paths
        assert engine._is_suspicious_path("/../etc/passwd")
        assert engine._is_suspicious_path("/admin/phpmyadmin")
        assert engine._is_suspicious_path("/test.php?cmd=ls")
        assert engine._is_suspicious_path("/api/v1/users' OR '1'='1")
        assert not engine._is_suspicious_path("/api/v1/health")


class TestDASTIntegration:
    """Test DAST integration functionality"""

    @pytest.fixture
    def dast_config(self):
        """Create test DAST configuration"""
        return DASTConfig(
            target_url="http://localhost:8000",
            zap_proxy="http://localhost:8080",
            active_scan=True,
            risk_threshold=RiskLevel.MEDIUM,
        )

    @pytest.mark.asyncio
    async def test_authentication_tester(self, dast_config):
        """Test authentication bypass testing"""
        mock_zap = MagicMock()
        tester = AuthenticationTester(mock_zap, dast_config)

        with patch.object(tester, "_test_endpoint") as mock_test:
            # Mock successful bypass
            mock_test.return_value = {
                "status": 200,
                "headers": {},
                "body": '{"agents": [{"id": 1, "name": "Agent1"}]}',
            }

            vulnerabilities = await tester.test_authentication_bypass()

            # Should find vulnerabilities for protected endpoints
            assert len(vulnerabilities) > 0

            # Check vulnerability details
            vuln = vulnerabilities[0]
            assert vuln["type"] == "authentication_bypass"
            assert "endpoint" in vuln
            assert "technique" in vuln

    def test_api_scanner(self, dast_config):
        """Test API vulnerability scanning"""
        mock_zap = MagicMock()
        scanner = APIScanner(mock_zap, dast_config)

        # Test rate limiting check
        with patch("requests.get") as mock_get:
            mock_get.return_value = Mock(status_code=200)

            vulns = scanner._test_rate_limiting()

            # Should detect missing rate limiting
            assert len(vulns) > 0
            assert vulns[0]["type"] == "missing_rate_limiting"

    def test_zap_scanner(self, dast_config):
        """Test ZAP scanner integration"""
        scanner = ZAPScanner(dast_config)

        # Mock ZAP API
        scanner.zap = MagicMock()
        scanner.zap.spider.scan.return_value = "1"
        scanner.zap.spider.status.return_value = "100"
        scanner.zap.ascan.scan.return_value = "2"
        scanner.zap.ascan.status.return_value = "100"
        scanner.zap.core.alerts.return_value = [
            {
                "name": "SQL Injection",
                "risk": "3",
                "confidence": "High",
                "description": "SQL injection vulnerability",
                "solution": "Use parameterized queries",
                "reference": "https://owasp.org",
                "cweid": "89",
                "wascid": "19",
                "url": "http://localhost:8000/api/v1/users?id=1",
                "method": "GET",
                "param": "id",
            }
        ]

        # Run spider
        scanner.spider_target()
        scanner.zap.spider.scan.assert_called_once()

        # Get alerts
        alerts = scanner.get_alerts()
        assert len(alerts) == 1
        assert alerts[0].name == "SQL Injection"
        assert alerts[0].risk == RiskLevel.HIGH

    @pytest.mark.asyncio
    async def test_dast_orchestrator(self, dast_config):
        """Test DAST scan orchestration"""
        orchestrator = DASTOrchestrator(dast_config)

        # Mock components
        orchestrator.zap_scanner = MagicMock()
        orchestrator.api_scanner = MagicMock()
        orchestrator.auth_tester = AsyncMock()

        orchestrator.auth_tester.test_authentication_bypass.return_value = [
            {"type": "authentication_bypass", "endpoint": "/api/v1/admin"}
        ]

        orchestrator.auth_tester.test_jwt_vulnerabilities.return_value = []

        orchestrator.api_scanner.test_api_vulnerabilities.return_value = [
            {"type": "missing_rate_limiting", "endpoint": "/api/v1/auth/login"}
        ]

        orchestrator.zap_scanner.get_alerts.return_value = [
            Alert(
                name="XSS",
                risk=RiskLevel.MEDIUM,
                confidence="Medium",
                description="Cross-site scripting",
                solution="Encode output",
                reference="",
            )
        ]

        # Run scan
        results = await orchestrator.run_scan()

        assert "vulnerabilities" in results
        assert "alerts" in results
        assert "stats" in results

        # Check vulnerabilities found
        assert len(results["vulnerabilities"]) == 2

        # Check alerts
        assert len(results["alerts"]) == 1
        assert results["alerts"][0]["name"] == "XSS"

        # Check stats
        assert results["stats"]["total_alerts"] == 1
        assert results["stats"]["custom_vulnerabilities"] == 2


class TestOWASPCompliance:
    """Test OWASP Top 10 compliance checking"""

    def test_owasp_mapping(self):
        """Test mapping of vulnerabilities to OWASP categories"""
        owasp_mapping = {
            "sql_injection": "A03:2021 - Injection",
            "broken_auth": "A07:2021 - Identification and Authentication Failures",
            "sensitive_data": "A02:2021 - Cryptographic Failures",
            "xxe": "A05:2021 - Security Misconfiguration",
            "broken_access": "A01:2021 - Broken Access Control",
            "security_misconfig": "A05:2021 - Security Misconfiguration",
            "xss": "A03:2021 - Injection",
            "insecure_deserialization": "A08:2021 - Software and Data Integrity Failures",
            "vulnerable_components": "A06:2021 - Vulnerable and Outdated Components",
            "insufficient_logging": "A09:2021 - Security Logging and Monitoring Failures",
        }

        # Verify all OWASP Top 10 categories are covered
        owasp_categories = set(owasp_mapping.values())
        expected_categories = {
            "A01:2021 - Broken Access Control",
            "A02:2021 - Cryptographic Failures",
            "A03:2021 - Injection",
            "A05:2021 - Security Misconfiguration",
            "A06:2021 - Vulnerable and Outdated Components",
            "A07:2021 - Identification and Authentication Failures",
            "A08:2021 - Software and Data Integrity Failures",
            "A09:2021 - Security Logging and Monitoring Failures",
        }

        # Note: A04 (Insecure Design) and A10 (SSRF) would need additional checks
        assert len(owasp_categories.intersection(expected_categories)) >= 8
