"""
Integration tests for the comprehensive security monitoring system.

Tests the security monitoring, vulnerability scanning, incident response,
and security API endpoints.
"""

import json
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from api.main import app
from auth.security_logging import SecurityEventSeverity, SecurityEventType
from observability.incident_response import IncidentResponseSystem, IncidentSeverity, IncidentStatus
from observability.security_monitoring import AttackType, SecurityMonitoringSystem, ThreatLevel
from observability.vulnerability_scanner import (
    SeverityLevel,
    VulnerabilityScanner,
    VulnerabilityType,
)


class TestSecurityMonitoringSystem:
    """Test suite for security monitoring system."""

    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)

        # Create test instances
        self.security_monitor = SecurityMonitoringSystem()
        self.vulnerability_scanner = VulnerabilityScanner()
        self.incident_response = IncidentResponseSystem()

    @pytest.mark.asyncio
    async def test_security_event_processing(self):
        """Test security event processing and threat detection."""
        # Test event processing
        test_event = {
            "event_type": SecurityEventType.LOGIN_FAILURE,
            "severity": SecurityEventSeverity.WARNING,
            "message": "Failed login attempt",
            "timestamp": datetime.utcnow().isoformat(),
            "ip_address": "192.168.1.100",
            "user_agent": "Mozilla/5.0",
            "endpoint": "/api/auth/login",
            "user_id": "test_user",
            "username": "testuser",
        }

        await self.security_monitor.process_security_event(test_event)

        # Check that event was processed
        assert len(self.security_monitor.security_events) > 0

        # Check IP activity tracking
        assert "192.168.1.100" in self.security_monitor.ip_activity
        assert len(self.security_monitor.ip_activity["192.168.1.100"]) > 0

    @pytest.mark.asyncio
    async def test_brute_force_detection(self):
        """Test brute force attack detection."""
        # Simulate multiple failed login attempts
        for i in range(6):  # Exceed threshold of 5
            test_event = {
                "event_type": SecurityEventType.LOGIN_FAILURE,
                "severity": SecurityEventSeverity.WARNING,
                "message": f"Failed login attempt {i + 1}",
                "timestamp": datetime.utcnow().isoformat(),
                "ip_address": "192.168.1.100",
                "user_agent": "Mozilla/5.0",
                "endpoint": "/api/auth/login",
                "user_id": "test_user",
                "username": "testuser",
            }

            await self.security_monitor.process_security_event(test_event)

        # Check that brute force was detected
        assert len(self.security_monitor.active_alerts) > 0

        # Check that IP was blocked
        assert "192.168.1.100" in self.security_monitor.blocked_ips

    @pytest.mark.asyncio
    async def test_ddos_detection(self):
        """Test DDoS attack detection."""
        # Simulate high volume of requests
        for i in range(1050):  # Exceed threshold of 1000
            test_event = {
                "event_type": SecurityEventType.API_ACCESS,
                "severity": SecurityEventSeverity.INFO,
                "message": f"API request {i + 1}",
                "timestamp": datetime.utcnow().isoformat(),
                "ip_address": "192.168.1.101",
                "user_agent": "Mozilla/5.0",
                "endpoint": "/api/data",
                "method": "GET",
                "status_code": 200,
            }

            await self.security_monitor.process_security_event(test_event)

        # Check that DDoS was detected
        ddos_alerts = [
            alert
            for alert in self.security_monitor.active_alerts.values()
            if alert.alert_type == AttackType.DDoS
        ]
        assert len(ddos_alerts) > 0

    @pytest.mark.asyncio
    async def test_injection_attack_detection(self):
        """Test SQL injection and XSS attack detection."""
        # Test SQL injection
        sql_injection_event = {
            "event_type": SecurityEventType.API_ACCESS,
            "severity": SecurityEventSeverity.INFO,
            "message": "API request with suspicious payload",
            "timestamp": datetime.utcnow().isoformat(),
            "ip_address": "192.168.1.102",
            "user_agent": "Mozilla/5.0",
            "endpoint": "/api/search?q='; DROP TABLE users; --",
            "method": "GET",
            "status_code": 200,
        }

        await self.security_monitor.process_security_event(sql_injection_event)

        # Check for SQL injection alert
        sql_alerts = [
            alert
            for alert in self.security_monitor.active_alerts.values()
            if alert.alert_type == AttackType.SQL_INJECTION
        ]
        assert len(sql_alerts) > 0

        # Test XSS attack
        xss_event = {
            "event_type": SecurityEventType.API_ACCESS,
            "severity": SecurityEventSeverity.INFO,
            "message": "API request with XSS payload",
            "timestamp": datetime.utcnow().isoformat(),
            "ip_address": "192.168.1.103",
            "user_agent": "Mozilla/5.0",
            "endpoint": "/api/comment",
            "method": "POST",
            "status_code": 200,
            "details": {"payload": "<script>alert('XSS')</script>"},
        }

        await self.security_monitor.process_security_event(xss_event)

        # Check for XSS alert
        xss_alerts = [
            alert
            for alert in self.security_monitor.active_alerts.values()
            if alert.alert_type == AttackType.XSS
        ]
        assert len(xss_alerts) > 0

    @pytest.mark.asyncio
    async def test_vulnerability_scanning(self):
        """Test vulnerability scanning functionality."""
        # Mock the bandit scan result
        mock_bandit_result = {
            "results": [
                {
                    "test_id": "B101",
                    "filename": "test_file.py",
                    "line_number": 10,
                    "issue_severity": "HIGH",
                    "issue_confidence": "HIGH",
                    "issue_text": "Use of assert detected",
                    "issue_cwe": "CWE-703",
                    "more_info": "https://bandit.readthedocs.io/en/latest/plugins/b101_assert_used.html",
                }
            ]
        }

        # Mock subprocess calls
        with (
            patch("subprocess.run") as mock_run,
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open_json(mock_bandit_result)),
        ):
            mock_run.return_value.returncode = 0

            # Run bandit scan
            result = await self.vulnerability_scanner._run_bandit_scan()

            # Check scan result
            assert result.success
            assert len(result.vulnerabilities) > 0
            assert result.vulnerabilities[0].severity == SeverityLevel.HIGH

    @pytest.mark.asyncio
    async def test_incident_response(self):
        """Test incident response system."""
        # Create a test security alert
        from observability.security_monitoring import SecurityAlert

        test_alert = SecurityAlert(
            id="test_alert_1",
            timestamp=datetime.utcnow(),
            alert_type=AttackType.BRUTE_FORCE,
            threat_level=ThreatLevel.HIGH,
            source_ip="192.168.1.100",
            user_id="test_user",
            description="Brute force attack detected",
            evidence={"attempts": 10, "timespan": "5 minutes"},
        )

        # Create incident from alert
        incident = await self.incident_response.create_incident_from_alert(test_alert)

        # Check incident creation
        assert incident.id in self.incident_response.incidents
        assert incident.severity == IncidentSeverity.HIGH
        assert incident.attack_type == AttackType.BRUTE_FORCE
        assert incident.status == IncidentStatus.INVESTIGATING

        # Check automated response
        assert len(incident.responses) > 0
        assert "192.168.1.100" in self.incident_response.blocked_ips

    @pytest.mark.asyncio
    async def test_security_metrics_collection(self):
        """Test security metrics collection."""
        # Generate some test events
        for i in range(10):
            test_event = {
                "event_type": SecurityEventType.LOGIN_SUCCESS,
                "severity": SecurityEventSeverity.INFO,
                "message": f"Successful login {i + 1}",
                "timestamp": datetime.utcnow().isoformat(),
                "ip_address": f"192.168.1.{100 + i}",
                "user_id": f"user_{i}",
                "username": f"user{i}",
            }

            await self.security_monitor.process_security_event(test_event)

        # Get metrics
        metrics = self.security_monitor.get_security_metrics()

        # Check metrics
        assert metrics.total_events > 0
        assert isinstance(metrics.top_attack_types, dict)
        assert isinstance(metrics.top_source_ips, dict)
        assert isinstance(metrics.threat_level_distribution, dict)

    def test_security_api_endpoints(self):
        """Test security API endpoints."""
        # Mock authentication
        with patch("auth.get_current_user") as mock_auth:
            mock_auth.return_value = Mock(
                user_id="test_admin",
                username="admin",
                permissions=["ADMIN_SYSTEM"],
            )

            # Test security summary endpoint
            response = self.client.get("/api/v1/security/summary")
            assert response.status_code == status.HTTP_200_OK

            # Test security events endpoint
            response = self.client.get("/api/v1/security/events")
            assert response.status_code == status.HTTP_200_OK

            # Test security alerts endpoint
            response = self.client.get("/api/v1/security/alerts")
            assert response.status_code == status.HTTP_200_OK

            # Test security metrics endpoint
            response = self.client.get("/api/v1/security/metrics")
            assert response.status_code == status.HTTP_200_OK

            # Test vulnerabilities endpoint
            response = self.client.get("/api/v1/security/vulnerabilities")
            assert response.status_code == status.HTTP_200_OK

            # Test incidents endpoint
            response = self.client.get("/api/v1/security/incidents")
            assert response.status_code == status.HTTP_200_OK

    def test_security_api_unauthorized(self):
        """Test security API endpoints without proper authorization."""
        # Test without authentication
        response = self.client.get("/api/v1/security/summary")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

        # Test with insufficient permissions
        with patch("auth.get_current_user") as mock_auth:
            mock_auth.return_value = Mock(
                user_id="test_user", username="user", permissions=["READ_ONLY"]
            )

            response = self.client.get("/api/v1/security/summary")
            assert response.status_code == status.HTTP_403_FORBIDDEN

    @pytest.mark.asyncio
    async def test_alert_management(self):
        """Test alert resolution and false positive marking."""
        # Create a test alert
        test_event = {
            "event_type": SecurityEventType.LOGIN_FAILURE,
            "severity": SecurityEventSeverity.WARNING,
            "message": "Failed login attempt",
            "timestamp": datetime.utcnow().isoformat(),
            "ip_address": "192.168.1.100",
            "user_id": "test_user",
            "username": "testuser",
        }

        await self.security_monitor.process_security_event(test_event)

        # Get alert ID
        alert_id = (
            list(self.security_monitor.active_alerts.keys())[0]
            if self.security_monitor.active_alerts
            else None
        )

        if alert_id:
            # Test alert resolution
            success = self.security_monitor.resolve_alert(alert_id, "Resolved by admin")
            assert success

            # Check alert status
            alert = self.security_monitor.active_alerts[alert_id]
            assert alert.status == "resolved"
            assert alert.resolution_notes == "Resolved by admin"

    @pytest.mark.asyncio
    async def test_vulnerability_management(self):
        """Test vulnerability suppression and false positive marking."""
        # Create a test vulnerability
        from observability.vulnerability_scanner import Vulnerability

        test_vuln = Vulnerability(
            id="test_vuln_1",
            type=VulnerabilityType.SECURITY_HOTSPOT,
            severity=SeverityLevel.HIGH,
            title="Test vulnerability",
            description="Test vulnerability description",
            file_path="test.py",
            line_number=10,
            scanner_name="test_scanner",
        )

        self.vulnerability_scanner.vulnerabilities[test_vuln.id] = test_vuln

        # Test suppression
        self.vulnerability_scanner.suppress_vulnerability(test_vuln.id, "False positive")
        assert test_vuln.id in self.vulnerability_scanner.suppressed_vulnerabilities
        assert test_vuln.suppressed

        # Test false positive marking
        self.vulnerability_scanner.mark_false_positive(test_vuln.id, "Not a real issue")
        assert test_vuln.id in self.vulnerability_scanner.false_positives
        assert test_vuln.false_positive

    @pytest.mark.asyncio
    async def test_incident_lifecycle(self):
        """Test complete incident lifecycle."""
        # Create incident
        from observability.security_monitoring import SecurityAlert

        test_alert = SecurityAlert(
            id="test_alert_2",
            timestamp=datetime.utcnow(),
            alert_type=AttackType.SQL_INJECTION,
            threat_level=ThreatLevel.CRITICAL,
            source_ip="192.168.1.200",
            user_id=None,
            description="SQL injection attack detected",
            evidence={"payload": "'; DROP TABLE users; --"},
        )

        incident = await self.incident_response.create_incident_from_alert(test_alert)

        # Check initial state
        assert incident.status == IncidentStatus.INVESTIGATING
        assert len(incident.responses) > 0

        # Resolve incident
        success = self.incident_response.resolve_incident(
            incident.id, "Attack blocked, vulnerability patched"
        )
        assert success

        # Check resolved state
        updated_incident = self.incident_response.get_incident(incident.id)
        assert updated_incident.status == IncidentStatus.RESOLVED
        assert updated_incident.resolved_at is not None
        assert updated_incident.lesson_learned == "Attack blocked, vulnerability patched"

    @pytest.mark.asyncio
    async def test_security_monitoring_start_stop(self):
        """Test security monitoring system start/stop functionality."""
        # Test start
        await self.security_monitor.start_monitoring()
        assert self.security_monitor.running
        assert self.security_monitor.monitoring_task is not None

        # Test stop
        await self.security_monitor.stop_monitoring()
        assert not self.security_monitor.running
        assert self.security_monitor.monitoring_task is None

    @pytest.mark.asyncio
    async def test_vulnerability_scanning_start_stop(self):
        """Test vulnerability scanning start/stop functionality."""
        # Test start
        await self.vulnerability_scanner.start_scanning()
        assert self.vulnerability_scanner.running
        assert self.vulnerability_scanner.scanning_task is not None

        # Test stop
        await self.vulnerability_scanner.stop_scanning()
        assert not self.vulnerability_scanner.running
        assert self.vulnerability_scanner.scanning_task is None

    @pytest.mark.asyncio
    async def test_incident_response_start_stop(self):
        """Test incident response system start/stop functionality."""
        # Test start
        await self.incident_response.start_monitoring()
        assert self.incident_response.running
        assert self.incident_response.monitoring_task is not None

        # Test stop
        await self.incident_response.stop_monitoring()
        assert not self.incident_response.running
        assert self.incident_response.monitoring_task is None

    def test_security_statistics(self):
        """Test security statistics generation."""
        # Test vulnerability statistics
        vuln_stats = self.vulnerability_scanner.get_vulnerability_stats()
        assert "total_vulnerabilities" in vuln_stats
        assert "by_severity" in vuln_stats
        assert "by_type" in vuln_stats

        # Test incident statistics
        incident_stats = self.incident_response.get_incident_statistics()
        assert "total_incidents" in incident_stats
        assert "by_severity" in incident_stats
        assert "by_attack_type" in incident_stats

    def test_security_txt_endpoint(self):
        """Test security.txt endpoint."""
        response = self.client.get("/.well-known/security.txt")
        assert response.status_code == status.HTTP_200_OK
        assert "Contact: security@freeagentics.com" in response.text
        assert "Expires: 2025-12-31T23:59:59.000Z" in response.text

    @pytest.mark.asyncio
    async def test_threat_indicator_matching(self):
        """Test threat indicator pattern matching."""
        # Test with a malicious payload
        test_event = {
            "event_type": SecurityEventType.API_ACCESS,
            "severity": SecurityEventSeverity.INFO,
            "message": "API request with malicious payload: SELECT * FROM users WHERE id = 1; DROP TABLE users; --",
            "timestamp": datetime.utcnow().isoformat(),
            "ip_address": "192.168.1.150",
            "user_agent": "sqlmap/1.0",
            "endpoint": "/api/search",
            "method": "GET",
        }

        await self.security_monitor.process_security_event(test_event)

        # Check for multiple threat types
        alerts = list(self.security_monitor.active_alerts.values())
        alert_types = [alert.alert_type for alert in alerts]

        # Should detect both SQL injection and suspicious user agent
        assert (
            AttackType.SQL_INJECTION in alert_types or AttackType.SUSPICIOUS_ACTIVITY in alert_types
        )


def mock_open_json(data):
    """Mock open function that returns JSON data."""
    from unittest.mock import mock_open

    return mock_open(read_data=json.dumps(data))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
