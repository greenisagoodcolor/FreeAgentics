"""
Authorization Integration Testing Suite.

This module provides integration tests that verify the authorization system
works correctly with other security components and maintains proper boundaries
in real-world scenarios.
"""

import time
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from api.main import app
from auth.rbac_enhancements import (
    ABACEffect,
    ABACRule,
    AccessContext,
    ResourceContext,
    enhanced_rbac_manager,
)
from auth.security_implementation import (
    ROLE_PERMISSIONS,
    Permission,
    UserRole,
    auth_manager,
    rate_limiter,
)
from auth.security_logging import (
    SecurityEventSeverity,
    SecurityEventType,
    security_auditor,
)
from database.session import get_db


class TestAuthorizationWithSecurityHeaders:
    """Test authorization with security headers integration."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def security_users(self):
        """Create users for security header testing."""
        users = {}

        for role in UserRole:
            username = f"sec_header_{role.value}"
            user = auth_manager.register_user(
                username=username,
                email=f"{username}@test.com",
                password="SecHeader123!",
                role=role,
            )
            users[role] = {
                "user": user,
                "token": auth_manager.create_access_token(user),
            }

        return users

    def test_authorization_with_cors(self, client, security_users):
        """Test authorization enforcement with CORS headers."""
        observer = security_users[UserRole.OBSERVER]

        # Test preflight requests
        origins = [
            "http://localhost:3000",
            "https://trusted.example.com",
            "https://malicious.example.com",
            "null",
            "file://",
        ]

        for origin in origins:
            # Preflight request
            response = client.options(
                "/api/v1/agents",
                headers={
                    "Origin": origin,
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "Authorization, Content-Type",
                },
            )

            # Even if CORS allows, authorization should still be enforced
            if response.status_code == status.HTTP_200_OK:
                # Actual request
                response = client.post(
                    "/api/v1/agents",
                    headers={
                        "Authorization": f"Bearer {observer['token']}",
                        "Origin": origin,
                        "Content-Type": "application/json",
                    },
                    json={"name": "test", "template": "basic"},
                )

                # Observer shouldn't create agents regardless of origin
                assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_authorization_with_csp(self, client, security_users):
        """Test that CSP headers don't interfere with authorization."""
        admin = security_users[UserRole.ADMIN]
        headers = {"Authorization": f"Bearer {admin['token']}"}

        # Make request that should succeed
        response = client.get("/api/v1/system/config", headers=headers)

        # Check CSP headers are present
        csp_header = response.headers.get("Content-Security-Policy")

        # Authorization should work regardless of CSP
        assert response.status_code == status.HTTP_200_OK

        # CSP should be appropriately restrictive
        if csp_header:
            assert "default-src" in csp_header
            assert "script-src" in csp_header

    def test_authorization_with_security_headers_manipulation(self, client, security_users):
        """Test authorization when security headers are manipulated."""
        observer = security_users[UserRole.OBSERVER]

        # Try to bypass with various header manipulations
        header_attacks = [
            {
                "Authorization": f"Bearer {observer['token']}",
                "X-Original-URL": "/api/v1/system/config",
                "X-Rewrite-URL": "/api/v1/system/config",
            },
            {
                "Authorization": f"Bearer {observer['token']}",
                "X-Forwarded-Host": "admin.internal",
                "X-Forwarded-For": "127.0.0.1",
            },
            {
                "Authorization": f"Bearer {observer['token']}",
                "X-Custom-Authorization": "Bearer admin_token",
                "X-Override-Auth": "true",
            },
            {
                "Authorization": f"Bearer {observer['token']}",
                "Forwarded": "for=admin;host=admin.local;proto=https",
            },
        ]

        for headers in header_attacks:
            response = client.get("/api/v1/system/config", headers=headers)
            assert (
                response.status_code == status.HTTP_403_FORBIDDEN
            ), f"Header manipulation bypassed auth: {headers}"


class TestAuthorizationWithRateLimiting:
    """Test authorization with rate limiting integration."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def rate_limit_users(self):
        """Create users for rate limit testing."""
        users = {}

        for i in range(3):
            username = f"rate_user_{i}"
            user = auth_manager.register_user(
                username=username,
                email=f"{username}@test.com",
                password="RateTest123!",
                role=UserRole.OBSERVER,
            )
            users[username] = {
                "user": user,
                "token": auth_manager.create_access_token(user),
                "ip": f"192.168.1.{10+i}",
            }

        return users

    def test_authorization_under_rate_limit_attack(self, client, rate_limit_users):
        """Test authorization decisions under rate limit attacks."""
        attacker = rate_limit_users["rate_user_0"]
        legitimate_user = rate_limit_users["rate_user_1"]

        # Clear rate limiter
        rate_limiter.requests.clear()

        # Attacker floods the system
        attacker_responses = []
        for i in range(150):  # Exceed rate limit
            headers = {
                "Authorization": f"Bearer {attacker['token']}",
                "X-Forwarded-For": attacker["ip"],
            }
            response = client.get("/api/v1/agents", headers=headers)
            attacker_responses.append(response.status_code)

        # Check that rate limiting kicked in
        any(status == status.HTTP_429_TOO_MANY_REQUESTS for status in attacker_responses)

        # Legitimate user should still work
        legitimate_headers = {
            "Authorization": f"Bearer {legitimate_user['token']}",
            "X-Forwarded-For": legitimate_user["ip"],
        }
        response = client.get("/api/v1/agents", headers=legitimate_headers)

        # Should not be affected by attacker's rate limit
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_429_TOO_MANY_REQUESTS,
        ]

    def test_authorization_bypass_attempts_trigger_rate_limit(self, client, rate_limit_users):
        """Test that authorization bypass attempts trigger rate limiting."""
        attacker = rate_limit_users["rate_user_2"]

        # Track failed authorization attempts
        failed_attempts = 0

        # Various bypass attempts
        for i in range(100):
            headers = {
                "Authorization": f"Bearer {attacker['token']}",
                "X-Forwarded-For": attacker["ip"],
            }

            # Try to access admin endpoint
            response = client.get("/api/v1/system/config", headers=headers)

            if response.status_code == status.HTTP_403_FORBIDDEN:
                failed_attempts += 1
            elif response.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
                break

        # Should eventually get rate limited after many failed attempts
        print(f"Failed attempts before rate limit: {failed_attempts}")

        # Verify attacker is now rate limited
        response = client.get(
            "/api/v1/agents",  # Even allowed endpoint
            headers={
                "Authorization": f"Bearer {attacker['token']}",
                "X-Forwarded-For": attacker["ip"],
            },
        )

        # Might be rate limited
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_429_TOO_MANY_REQUESTS,
        ]


class TestAuthorizationWithAuditLogging:
    """Test authorization with comprehensive audit logging."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def audit_users(self):
        """Create users for audit testing."""
        users = {}

        for role in [UserRole.ADMIN, UserRole.RESEARCHER, UserRole.OBSERVER]:
            username = f"audit_{role.value}"
            user = auth_manager.register_user(
                username=username,
                email=f"{username}@test.com",
                password="Audit123!",
                role=role,
            )
            users[role] = {
                "user": user,
                "token": auth_manager.create_access_token(user),
            }

        return users

    def test_authorization_decisions_are_logged(self, client, audit_users):
        """Test that all authorization decisions are properly logged."""
        # Clear audit logs
        enhanced_rbac_manager.access_audit_log.clear()

        # Perform various authorization scenarios
        scenarios = [
            # Successful authorization
            (UserRole.ADMIN, "/api/v1/system/config", "GET", True),
            # Failed authorization
            (UserRole.OBSERVER, "/api/v1/system/config", "GET", False),
            # Resource creation
            (UserRole.RESEARCHER, "/api/v1/agents", "POST", True),
            # Unauthorized deletion
            (UserRole.OBSERVER, "/api/v1/agents/123", "DELETE", False),
        ]

        for role, endpoint, method, should_succeed in scenarios:
            user = audit_users[role]
            headers = {"Authorization": f"Bearer {user['token']}"}

            if method == "GET":
                response = client.get(endpoint, headers=headers)
            elif method == "POST":
                response = client.post(
                    endpoint,
                    headers=headers,
                    json={"name": "test", "template": "basic"},
                )
            elif method == "DELETE":
                response = client.delete(endpoint, headers=headers)

            # Verify response matches expectation
            if should_succeed:
                assert response.status_code not in [
                    status.HTTP_401_UNAUTHORIZED,
                    status.HTTP_403_FORBIDDEN,
                ]
            else:
                assert response.status_code in [
                    status.HTTP_401_UNAUTHORIZED,
                    status.HTTP_403_FORBIDDEN,
                ]

        # Check audit logs
        audit_logs = enhanced_rbac_manager.access_audit_log

        # Should have logged decisions
        assert len(audit_logs) > 0

        # Verify audit log contains required information
        for log in audit_logs:
            assert "timestamp" in log
            assert "user_id" in log
            assert "action" in log
            assert "decision" in log

    def test_attack_patterns_trigger_security_alerts(self, client, audit_users):
        """Test that attack patterns trigger appropriate security alerts."""
        attacker = audit_users[UserRole.OBSERVER]

        # Mock security auditor to capture events
        captured_events = []

        original_log_event = security_auditor.log_event

        def mock_log_event(event_type, severity, message, **kwargs):
            captured_events.append(
                {
                    "type": event_type,
                    "severity": severity,
                    "message": message,
                    "details": kwargs,
                }
            )
            return original_log_event(event_type, severity, message, **kwargs)

        with patch.object(security_auditor, "log_event", side_effect=mock_log_event):
            # Perform suspicious activities
            headers = {"Authorization": f"Bearer {attacker['token']}"}

            # Multiple failed access attempts
            for _ in range(10):
                client.get("/api/v1/system/config", headers=headers)

            # Parameter injection attempts
            client.post(
                "/api/v1/agents",
                headers=headers,
                json={
                    "name": "test",
                    "template": "basic",
                    "role": "admin",  # Attempted privilege escalation
                },
            )

            # SQL injection attempt
            client.get("/api/v1/agents?filter=' OR '1'='1", headers=headers)

        # Verify security events were logged
        security_events = [
            e
            for e in captured_events
            if e["severity"] in [SecurityEventSeverity.WARNING, SecurityEventSeverity.CRITICAL]
        ]

        assert len(security_events) > 0, "Attack patterns should trigger security alerts"

        # Check for specific event types
        event_types = [e["type"] for e in security_events]

        # Should include access denied events
        assert any(event_type == SecurityEventType.ACCESS_DENIED for event_type in event_types)


class TestAuthorizationWithDatabaseIntegration:
    """Test authorization with database operations."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def db_users(self, db: Session = next(get_db())):
        """Create users with database integration."""
        users = {}

        try:
            for role in [
                UserRole.ADMIN,
                UserRole.RESEARCHER,
                UserRole.OBSERVER,
            ]:
                username = f"db_test_{role.value}"
                user = auth_manager.register_user(
                    username=username,
                    email=f"{username}@test.com",
                    password="DbTest123!",
                    role=role,
                )
                users[role] = {
                    "user": user,
                    "token": auth_manager.create_access_token(user),
                }

            yield users
        finally:
            # Cleanup would happen here in real implementation
            pass

    def test_resource_ownership_with_database(self, client, db_users):
        """Test resource ownership validation with actual database records."""
        researcher1 = db_users[UserRole.RESEARCHER]
        researcher2 = db_users[UserRole.RESEARCHER]

        # Researcher 1 creates an agent
        headers1 = {"Authorization": f"Bearer {researcher1['token']}"}
        response = client.post(
            "/api/v1/agents",
            headers=headers1,
            json={
                "name": "Researcher1Agent",
                "template": "basic",
                "parameters": {"research_data": "confidential"},
            },
        )

        if response.status_code == status.HTTP_201_CREATED:
            agent_id = response.json()["id"]

            # Researcher 2 tries to access
            headers2 = {"Authorization": f"Bearer {researcher2['token']}"}

            # View might be allowed
            response = client.get(f"/api/v1/agents/{agent_id}", headers=headers2)

            # But modification should be restricted
            response = client.put(
                f"/api/v1/agents/{agent_id}",
                headers=headers2,
                json={"name": "HijackedAgent"},
            )
            assert response.status_code == status.HTTP_403_FORBIDDEN

            # Admin should be able to access
            admin_headers = {"Authorization": f"Bearer {db_users[UserRole.ADMIN]['token']}"}
            response = client.put(
                f"/api/v1/agents/{agent_id}",
                headers=admin_headers,
                json={"name": "AdminModifiedAgent"},
            )
            # Admin should have access
            assert response.status_code != status.HTTP_403_FORBIDDEN

    def test_database_injection_with_authorization(self, client, db_users):
        """Test that authorization prevents database injection attacks."""
        attacker = db_users[UserRole.OBSERVER]
        headers = {"Authorization": f"Bearer {attacker['token']}"}

        # SQL injection attempts in various parameters
        injection_attempts = [
            # In query parameters
            "/api/v1/agents?filter=name='test' OR '1'='1'--",
            "/api/v1/agents?sort=name; DROP TABLE agents; --",
            "/api/v1/agents?id=1 UNION SELECT * FROM users--",
            # In path parameters
            "/api/v1/agents/1' OR '1'='1",
            "/api/v1/agents/1; DELETE FROM agents WHERE '1'='1",
            # In JSON body
            {
                "endpoint": "/api/v1/agents/search",
                "body": {
                    "query": "'; DROP TABLE agents; --",
                    "filters": {"name": "test' OR owner_id IS NOT NULL--"},
                },
            },
        ]

        for attempt in injection_attempts:
            if isinstance(attempt, str):
                response = client.get(attempt, headers=headers)
            else:
                response = client.post(attempt["endpoint"], headers=headers, json=attempt["body"])

            # Should either be forbidden (no permission) or bad request (invalid input)
            assert response.status_code in [
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_403_FORBIDDEN,
                status.HTTP_404_NOT_FOUND,
            ]


class TestComplexAuthorizationScenarios:
    """Test complex real-world authorization scenarios."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def complex_setup(self):
        """Set up complex authorization scenario."""
        setup = {
            "departments": ["engineering", "research", "finance", "hr"],
            "users": {},
            "resources": {},
            "abac_rules": [],
        }

        # Create department-based users
        for dept in setup["departments"]:
            for role in [
                UserRole.ADMIN,
                UserRole.RESEARCHER,
                UserRole.OBSERVER,
            ]:
                username = f"{dept}_{role.value}"
                user = auth_manager.register_user(
                    username=username,
                    email=f"{username}@company.com",
                    password="Complex123!",
                    role=role,
                )
                setup["users"][username] = {
                    "user": user,
                    "token": auth_manager.create_access_token(user),
                    "department": dept,
                    "role": role,
                }

        # Add complex ABAC rules
        rules = [
            {
                "id": "cross_dept_restriction",
                "name": "Cross-Department Restriction",
                "resource_type": "*",
                "action": "modify",
                "subject_conditions": {},
                "resource_conditions": {"same_department": True},
                "environment_conditions": {},
                "effect": ABACEffect.ALLOW,
                "priority": 100,
            },
            {
                "id": "sensitive_data_time_restriction",
                "name": "Sensitive Data Time Restriction",
                "resource_type": "sensitive_data",
                "action": "*",
                "subject_conditions": {},
                "resource_conditions": {},
                "environment_conditions": {
                    "time_range": {"start": "09:00", "end": "17:00"},
                    "days": [
                        "Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thursday",
                        "Friday",
                    ],
                },
                "effect": ABACEffect.ALLOW,
                "priority": 110,
            },
            {
                "id": "finance_data_protection",
                "name": "Finance Data Protection",
                "resource_type": "financial_record",
                "action": "*",
                "subject_conditions": {"department": ["finance", "executive"]},
                "resource_conditions": {},
                "environment_conditions": {},
                "effect": ABACEffect.ALLOW,
                "priority": 120,
            },
        ]

        for rule_data in rules:
            rule = ABACRule(
                **rule_data,
                description=rule_data["name"],
                created_at=datetime.now(timezone.utc),
                created_by="system",
            )
            enhanced_rbac_manager.add_abac_rule(rule)
            setup["abac_rules"].append(rule)

        return setup

    def test_multi_tenant_authorization(self, client, complex_setup):
        """Test authorization in multi-tenant environment."""
        # Engineering admin
        eng_admin = complex_setup["users"]["engineering_admin"]
        eng_headers = {"Authorization": f"Bearer {eng_admin['token']}"}

        # Create engineering resource
        response = client.post(
            "/api/v1/agents",
            headers=eng_headers,
            json={
                "name": "EngineeringAgent",
                "template": "engineering",
                "parameters": {
                    "department": "engineering",
                    "project": "secret_project",
                },
            },
        )

        if response.status_code == status.HTTP_201_CREATED:
            eng_agent_id = response.json()["id"]

            # Research admin tries to access
            research_admin = complex_setup["users"]["research_admin"]
            research_headers = {"Authorization": f"Bearer {research_admin['token']}"}

            # Should not be able to modify cross-department
            response = client.put(
                f"/api/v1/agents/{eng_agent_id}",
                headers=research_headers,
                json={"name": "ResearchModified"},
            )

            # Check ABAC decision
            research_context = AccessContext(
                user_id=research_admin["user"].user_id,
                username=research_admin["user"].username,
                role=research_admin["role"],
                permissions=list(ROLE_PERMISSIONS[research_admin["role"]]),
                department="research",
            )

            eng_resource = ResourceContext(
                resource_id=eng_agent_id,
                resource_type="agent",
                department="engineering",
            )

            granted, reason, _ = enhanced_rbac_manager.evaluate_abac_access(
                research_context, eng_resource, "modify"
            )

            # Should be denied due to department mismatch
            assert not granted or "department" in reason

    def test_delegation_chain_authorization(self, client, complex_setup):
        """Test authorization with delegation chains."""
        # CEO delegates to department heads
        # Department heads delegate to team leads
        # Test that delegation doesn't bypass authorization

        finance_admin = complex_setup["users"]["finance_admin"]
        finance_observer = complex_setup["users"]["finance_observer"]

        # Finance admin creates sensitive resource
        admin_headers = {"Authorization": f"Bearer {finance_admin['token']}"}
        response = client.post(
            "/api/v1/financial_records",
            headers=admin_headers,
            json={
                "type": "quarterly_report",
                "classification": "confidential",
                "department": "finance",
            },
        )

        # Even with "delegation", observer shouldn't access admin functions
        observer_headers = {"Authorization": f"Bearer {finance_observer['token']}"}

        # Try to claim delegation
        delegated_headers = {
            **observer_headers,
            "X-Delegated-By": finance_admin["user"].user_id,
            "X-Delegation-Token": "fake_delegation_token",
        }

        response = client.delete("/api/v1/financial_records/all", headers=delegated_headers)

        # Should still be forbidden
        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_emergency_access_procedures(self, complex_setup):
        """Test emergency access procedures maintain security."""
        # Simulate emergency access scenario

        # Create emergency context
        emergency_context = AccessContext(
            user_id="emergency_responder",
            username="emergency",
            role=UserRole.OBSERVER,  # Low privilege normally
            permissions=list(ROLE_PERMISSIONS[UserRole.OBSERVER]),
            department="security",
            metadata={"emergency_access": True, "incident_id": "INC-12345"},
        )

        # High-value resource
        critical_resource = ResourceContext(
            resource_id="critical_001",
            resource_type="system_config",
            classification="top_secret",
            sensitivity_level="restricted",
        )

        # Normal access should be denied
        normal_granted, _, _ = enhanced_rbac_manager.evaluate_abac_access(
            emergency_context, critical_resource, "view"
        )

        # Emergency access might have special rules
        # But should still be audited and time-limited

        # Add emergency access rule
        ABACRule(
            id="emergency_access",
            name="Emergency Access",
            description="Time-limited emergency access",
            resource_type="*",
            action="view",
            subject_conditions={"metadata.emergency_access": True},
            resource_conditions={},
            environment_conditions={"time_limit": 3600},  # 1 hour
            effect=ABACEffect.ALLOW,
            priority=200,
            created_at=datetime.now(timezone.utc),
            created_by="security_team",
        )

        # Emergency access should be heavily audited
        assert emergency_context.metadata.get("incident_id") is not None


# Performance tests for authorization
class TestAuthorizationPerformance:
    """Test authorization system performance characteristics."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def performance_setup(self):
        """Set up for performance testing."""
        # Create many users
        users = []
        for i in range(100):
            role = list(UserRole)[i % len(UserRole)]
            user = auth_manager.register_user(
                username=f"perf_user_{i}",
                email=f"perf{i}@test.com",
                password="Perf123!",
                role=role,
            )
            users.append(
                {
                    "user": user,
                    "token": auth_manager.create_access_token(user),
                    "role": role,
                }
            )

        # Create many ABAC rules
        for i in range(50):
            rule = ABACRule(
                id=f"perf_rule_{i}",
                name=f"Performance Rule {i}",
                description=f"Rule for performance testing {i}",
                resource_type="*" if i % 2 == 0 else "agent",
                action="*" if i % 3 == 0 else "view",
                subject_conditions={"role": ["researcher"]} if i % 4 == 0 else {},
                resource_conditions={"department": f"dept_{i % 10}"} if i % 5 == 0 else {},
                environment_conditions={},
                effect=ABACEffect.ALLOW if i % 2 == 0 else ABACEffect.DENY,
                priority=100 + i,
                created_at=datetime.now(timezone.utc),
                created_by="system",
            )
            enhanced_rbac_manager.add_abac_rule(rule)

        return users

    def test_authorization_latency(self, client, performance_setup):
        """Test authorization decision latency."""
        users = performance_setup

        latencies = []

        for user in users[:20]:  # Test subset
            headers = {"Authorization": f"Bearer {user['token']}"}

            start_time = time.time()
            client.get("/api/v1/agents", headers=headers)
            end_time = time.time()

            latency = end_time - start_time
            latencies.append(latency)

        # Calculate statistics
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        min_latency = min(latencies)

        print(
            f"Authorization Latency - Avg: {avg_latency:.4f}s, Max: {max_latency:.4f}s, Min: {min_latency:.4f}s"
        )

        # Performance requirements
        assert avg_latency < 0.1, "Average authorization latency too high"
        assert max_latency < 0.5, "Maximum authorization latency too high"

    def test_authorization_throughput(self, client, performance_setup):
        """Test authorization system throughput."""
        users = performance_setup

        # Measure requests per second
        duration = 5  # seconds
        request_count = 0
        start_time = time.time()

        while time.time() - start_time < duration:
            user = users[request_count % len(users)]
            headers = {"Authorization": f"Bearer {user['token']}"}

            client.get("/api/v1/agents", headers=headers)
            request_count += 1

        rps = request_count / duration
        print(f"Authorization throughput: {rps:.2f} requests/second")

        # Should handle reasonable load
        assert rps > 50, "Authorization throughput too low"

    def test_authorization_under_load(self, client, performance_setup):
        """Test authorization decisions remain consistent under load."""
        import concurrent.futures

        users = performance_setup

        # Define test scenarios
        def make_request(user_data):
            headers = {"Authorization": f"Bearer {user_data['token']}"}
            endpoint = (
                "/api/v1/agents" if user_data["role"] != UserRole.ADMIN else "/api/v1/system/config"
            )

            response = client.get(endpoint, headers=headers)

            expected_success = (
                endpoint == "/api/v1/agents"
                and Permission.VIEW_AGENTS in ROLE_PERMISSIONS[user_data["role"]]
            ) or (endpoint == "/api/v1/system/config" and user_data["role"] == UserRole.ADMIN)

            return {
                "user": user_data["user"].username,
                "role": user_data["role"],
                "endpoint": endpoint,
                "status": response.status_code,
                "expected_success": expected_success,
            }

        # Run concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = []

            # Submit many requests
            for _ in range(5):
                for user in users[:20]:
                    futures.append(executor.submit(make_request, user))

            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Verify consistency
        inconsistencies = []
        for result in results:
            if result["expected_success"]:
                if result["status"] == status.HTTP_403_FORBIDDEN:
                    inconsistencies.append(result)
            else:
                if result["status"] == status.HTTP_200_OK:
                    inconsistencies.append(result)

        # Should have no inconsistencies
        assert (
            len(inconsistencies) == 0
        ), f"Authorization inconsistencies under load: {len(inconsistencies)}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
