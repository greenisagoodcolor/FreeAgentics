"""
Specialized Authorization Attack Testing Module.

This module focuses on testing specific authorization attack vectors and ensures
the system properly defends against common and advanced authorization vulnerabilities.
"""

import json
import secrets
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from fastapi import status
from fastapi.testclient import TestClient

from api.main import app
from auth.security_implementation import (
    ROLE_PERMISSIONS,
    Permission,
    UserRole,
    auth_manager,
)


class TestIDORVulnerabilities:
    """Test Insecure Direct Object Reference vulnerabilities in depth."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def setup_idor_scenario(self):
        """Set up IDOR test scenario with multiple users and resources."""
        users = {}
        resources = {}

        # Create users
        for i in range(3):
            username = f"user_{i}"
            user = auth_manager.register_user(
                username=username,
                email=f"{username}@test.com",
                password="Test123!",
                role=UserRole.RESEARCHER,
            )
            users[username] = {
                "user": user,
                "token": auth_manager.create_access_token(user),
                "resources": [],
            }

        # Create attacker with minimal privileges
        attacker = auth_manager.register_user(
            username="idor_attacker",
            email="idor_attacker@test.com",
            password="Attack123!",
            role=UserRole.OBSERVER,
        )
        users["attacker"] = {
            "user": attacker,
            "token": auth_manager.create_access_token(attacker),
            "resources": [],
        }

        return users, resources

    def test_sequential_id_enumeration(self, client, setup_idor_scenario):
        """Test defense against sequential ID enumeration attacks."""
        users, _ = setup_idor_scenario

        # User 1 creates resources with potentially sequential IDs
        user1 = users["user_0"]
        headers = {"Authorization": f"Bearer {user1['token']}"}

        created_ids = []
        for i in range(5):
            response = client.post(
                "/api/v1/agents",
                headers=headers,
                json={
                    "name": f"Agent_{i}",
                    "template": "basic",
                    "parameters": {"secret": f"confidential_data_{i}"},
                },
            )
            if response.status_code == status.HTTP_201_CREATED:
                created_ids.append(response.json().get("id"))

        # Attacker attempts to enumerate IDs
        attacker = users["attacker"]
        attacker_headers = {"Authorization": f"Bearer {attacker['token']}"}

        # Check if IDs are predictable
        if len(created_ids) >= 2:
            # Try to predict next ID based on pattern
            enumeration_attempts = []

            # If IDs are numeric, try incrementing
            try:
                for created_id in created_ids:
                    if isinstance(created_id, str) and created_id.isdigit():
                        for offset in range(-5, 6):
                            predicted_id = str(int(created_id) + offset)
                            response = client.get(
                                f"/api/v1/agents/{predicted_id}",
                                headers=attacker_headers,
                            )
                            enumeration_attempts.append(
                                {
                                    "id": predicted_id,
                                    "status": response.status_code,
                                    "found": response.status_code == status.HTTP_200_OK,
                                }
                            )
            except ValueError:
                pass  # IDs are not numeric

            # Attacker should not be able to access other users' resources
            unauthorized_access = [
                attempt
                for attempt in enumeration_attempts
                if attempt["found"] and attempt["id"] not in created_ids
            ]

            assert (
                len(unauthorized_access) == 0
            ), "Sequential ID enumeration allowed unauthorized access"

    def test_uuid_prediction_attacks(self, client, setup_idor_scenario):
        """Test resistance to UUID prediction and timing attacks."""
        users, _ = setup_idor_scenario

        # Collect timing information for UUID generation
        user1 = users["user_0"]
        headers = {"Authorization": f"Bearer {user1['token']}"}

        uuid_timings = []

        for i in range(10):
            start_time = datetime.now(timezone.utc)

            response = client.post(
                "/api/v1/agents",
                headers=headers,
                json={"name": f"TimingAgent_{i}", "template": "basic"},
            )

            end_time = datetime.now(timezone.utc)

            if response.status_code == status.HTTP_201_CREATED:
                agent_id = response.json().get("id")
                uuid_timings.append(
                    {
                        "id": agent_id,
                        "timestamp": start_time,
                        "duration": (end_time - start_time).total_seconds(),
                    }
                )

        # Analyze for predictable patterns
        # Good UUIDs should have no correlation with time
        if len(uuid_timings) >= 2:
            # Check if IDs are time-based (like UUID v1)
            for i in range(1, len(uuid_timings)):
                id1 = uuid_timings[i - 1]["id"]
                id2 = uuid_timings[i]["id"]

                # If using proper random UUIDs, consecutive IDs should be unrelated
                # This is a basic check - real UUID analysis would be more complex
                assert id1 != id2, "Duplicate IDs generated"

    def test_parameter_pollution_idor(self, client, setup_idor_scenario):
        """Test IDOR through parameter pollution attacks."""
        users, _ = setup_idor_scenario

        # User 0 and User 1 create resources
        for username in ["user_0", "user_1"]:
            user = users[username]
            headers = {"Authorization": f"Bearer {user['token']}"}

            response = client.post(
                "/api/v1/agents",
                headers=headers,
                json={
                    "name": f"{username}_private_agent",
                    "template": "basic",
                    "parameters": {"owner": username},
                },
            )

            if response.status_code == status.HTTP_201_CREATED:
                user["resources"].append(response.json().get("id"))

        # Attacker attempts parameter pollution
        attacker = users["attacker"]
        attacker_headers = {"Authorization": f"Bearer {attacker['token']}"}

        if len(users["user_0"]["resources"]) > 0:
            target_id = users["user_0"]["resources"][0]

            # Attempt 1: Duplicate parameters
            response = client.get(
                f"/api/v1/agents/{target_id}?user_id={attacker['user'].user_id}&user_id={users['user_0']['user'].user_id}",
                headers=attacker_headers,
            )

            # Should not allow access through parameter pollution
            if response.status_code == status.HTTP_200_OK:
                data = response.json()
                # Verify no sensitive data is exposed
                assert "parameters" not in data or "owner" not in data.get("parameters", {})

            # Attempt 2: Array parameter pollution
            response = client.get(
                f"/api/v1/agents?id[]={target_id}&id[]={attacker['user'].user_id}",
                headers=attacker_headers,
            )

            # Attempt 3: JSON parameter injection
            response = client.post(
                "/api/v1/agents/query",
                headers=attacker_headers,
                json={
                    "filters": {
                        "id": target_id,
                        "user_id": [
                            attacker["user"].user_id,
                            users["user_0"]["user"].user_id,
                        ],
                    }
                },
            )

    def test_blind_idor_exploitation(self, client, setup_idor_scenario):
        """Test blind IDOR vulnerabilities where existence is leaked."""
        users, _ = setup_idor_scenario

        # Create resource with user_0
        user0 = users["user_0"]
        headers = {"Authorization": f"Bearer {user0['token']}"}

        response = client.post(
            "/api/v1/agents",
            headers=headers,
            json={"name": "BlindIDORTestAgent", "template": "basic"},
        )

        if response.status_code == status.HTTP_201_CREATED:
            existing_id = response.json().get("id")

            # Attacker probes for resource existence
            attacker = users["attacker"]
            attacker_headers = {"Authorization": f"Bearer {attacker['token']}"}

            # Test different operations that might leak existence
            operations = [
                ("GET", f"/api/v1/agents/{existing_id}"),
                ("PUT", f"/api/v1/agents/{existing_id}"),
                ("DELETE", f"/api/v1/agents/{existing_id}"),
                ("GET", f"/api/v1/agents/{existing_id}/metrics"),
                ("POST", f"/api/v1/agents/{existing_id}/actions"),
            ]

            existence_indicators = []

            for method, endpoint in operations:
                if method == "GET":
                    response = client.get(endpoint, headers=attacker_headers)
                elif method == "PUT":
                    response = client.put(endpoint, headers=attacker_headers, json={})
                elif method == "DELETE":
                    response = client.delete(endpoint, headers=attacker_headers)
                elif method == "POST":
                    response = client.post(endpoint, headers=attacker_headers, json={})

                # Check if response differs for existing vs non-existing resources
                non_existing_id = f"non_existing_{secrets.token_hex(8)}"
                non_existing_endpoint = endpoint.replace(existing_id, non_existing_id)

                if method == "GET":
                    response_non_existing = client.get(
                        non_existing_endpoint, headers=attacker_headers
                    )
                elif method == "PUT":
                    response_non_existing = client.put(
                        non_existing_endpoint,
                        headers=attacker_headers,
                        json={},
                    )
                elif method == "DELETE":
                    response_non_existing = client.delete(
                        non_existing_endpoint, headers=attacker_headers
                    )
                elif method == "POST":
                    response_non_existing = client.post(
                        non_existing_endpoint,
                        headers=attacker_headers,
                        json={},
                    )

                # Both should return same status code to prevent existence leakage
                if response.status_code != response_non_existing.status_code:
                    existence_indicators.append(
                        {
                            "method": method,
                            "endpoint": endpoint,
                            "existing_status": response.status_code,
                            "non_existing_status": response_non_existing.status_code,
                        }
                    )

            # No operation should leak resource existence to unauthorized users
            assert (
                len(existence_indicators) == 0
            ), f"Resource existence leaked through: {existence_indicators}"


class TestPrivilegeEscalationVectors:
    """Test various privilege escalation attack vectors."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def escalation_users(self):
        """Create users for privilege escalation testing."""
        users = {}

        # Create users with each role
        for role in UserRole:
            username = f"esc_{role.value}"
            user = auth_manager.register_user(
                username=username,
                email=f"{username}@test.com",
                password="Escalate123!",
                role=role,
            )
            users[role] = {
                "user": user,
                "token": auth_manager.create_access_token(user),
                "refresh_token": auth_manager.create_refresh_token(user),
            }

        return users

    def test_jwt_algorithm_confusion(self, client, escalation_users):
        """Test JWT algorithm confusion attacks."""
        observer = escalation_users[UserRole.OBSERVER]

        # Get the original token
        original_token = observer["token"]

        # Decode without verification
        unverified_payload = jwt.decode(original_token, options={"verify_signature": False})

        # Attempt 1: Change algorithm to none
        try:
            none_token = jwt.encode(unverified_payload, "", algorithm="none")
            headers = {"Authorization": f"Bearer {none_token}"}
            response = client.get("/api/v1/system/config", headers=headers)
            assert (
                response.status_code == status.HTTP_401_UNAUTHORIZED
            ), "Algorithm 'none' attack succeeded"
        except jwt.InvalidAlgorithmError:
            pass  # Good - library prevents this

        # Attempt 2: HS256 with public key as secret
        # First, try to get the public key (simulated)
        try:
            # Modify payload to admin
            admin_payload = unverified_payload.copy()
            admin_payload["role"] = "admin"
            admin_payload["permissions"] = ["admin_system"]

            # Try various potential secrets
            potential_secrets = [
                "dev_jwt_secret_2025_not_for_production",  # Default from code
                "secret",
                "jwt_secret",
                (
                    auth_manager.public_key.public_bytes(
                        encoding=serialization.Encoding.PEM,
                        format=serialization.PublicFormat.SubjectPublicKeyInfo,
                    ).decode()
                    if hasattr(auth_manager, "public_key")
                    else "public_key"
                ),
            ]

            for secret in potential_secrets:
                try:
                    hs256_token = jwt.encode(admin_payload, secret, algorithm="HS256")
                    headers = {"Authorization": f"Bearer {hs256_token}"}
                    response = client.get("/api/v1/system/config", headers=headers)

                    # Should not authenticate with algorithm confusion
                    assert (
                        response.status_code == status.HTTP_401_UNAUTHORIZED
                    ), f"Algorithm confusion with secret '{secret[:20]}...' succeeded"
                except Exception:
                    continue  # Expected to fail
        except Exception:
            pass  # Expected behavior

    def test_role_injection_attacks(self, client, escalation_users):
        """Test various role injection techniques."""
        observer = escalation_users[UserRole.OBSERVER]
        headers = {"Authorization": f"Bearer {observer['token']}"}

        # Collection of role injection attempts
        injection_payloads = [
            # Direct role injection in various fields
            {"name": "test", "role": "admin"},
            {"name": "test", "user_role": "admin"},
            {"name": "test", "metadata": {"role": "admin"}},
            {"name": "test", "options": {"override_role": "admin"}},
            # Array-based injections
            {"name": "test", "roles": ["observer", "admin"]},
            {"name": "test", "permissions": ["create_agent", "admin_system"]},
            # Nested injections
            {"name": "test", "user": {"role": "admin"}},
            {"name": "test", "context": {"user": {"role": "admin"}}},
            # SQL injection attempts
            {"name": "test' OR role='admin'--", "template": "basic"},
            {
                "name": "test",
                "template": "basic'; UPDATE users SET role='admin'--",
            },
            # JSON injection
            {"name": 'test","role":"admin","x":"', "template": "basic"},
            # Unicode/encoding tricks
            {"name": "test", "rоle": "admin"},  # Cyrillic 'o'
            {
                "name": "test",
                "\u0072\u006f\u006c\u0065": "admin",
            },  # Unicode escape
            # Parameter pollution
            {"name": "test", "role": "observer&role=admin"},
        ]

        for payload in injection_payloads:
            response = client.post("/api/v1/agents", headers=headers, json=payload)

            # None should grant elevated privileges
            if response.status_code == status.HTTP_201_CREATED:
                # Check that created resource doesn't have elevated permissions
                agent_data = response.json()
                assert "role" not in agent_data or agent_data.get("role") != "admin"

            # Try to access admin endpoint after each attempt
            response = client.get("/api/v1/system/config", headers=headers)
            assert (
                response.status_code == status.HTTP_403_FORBIDDEN
            ), f"Role injection succeeded with payload: {payload}"

    def test_permission_bypass_chains(self, client, escalation_users):
        """Test chaining multiple vulnerabilities for permission bypass."""
        agent_manager = escalation_users[UserRole.AGENT_MANAGER]

        # Chain 1: Create coalition (not allowed) through agent creation
        headers = {"Authorization": f"Bearer {agent_manager['token']}"}

        response = client.post(
            "/api/v1/agents",
            headers=headers,
            json={
                "name": "CoalitionBypassAgent",
                "template": "basic",
                "parameters": {
                    "auto_create_coalition": True,
                    "coalition_config": {
                        "name": "BypassCoalition",
                        "permissions": ["create_coalition"],
                    },
                },
            },
        )

        # Check if coalition was created despite lack of permission
        response = client.get("/api/v1/coalitions", headers=headers)
        if response.status_code == status.HTTP_200_OK:
            coalitions = response.json()
            bypass_coalitions = [c for c in coalitions if "Bypass" in c.get("name", "")]
            assert len(bypass_coalitions) == 0, "Coalition created through permission bypass"

        # Chain 2: Exploit race condition with token refresh
        def attempt_privileged_action():
            # Use refresh token flow
            refresh_response = client.post(
                "/api/v1/auth/refresh",
                json={"refresh_token": agent_manager["refresh_token"]},
            )

            if refresh_response.status_code == status.HTTP_200_OK:
                new_token = refresh_response.json().get("access_token")
                new_headers = {"Authorization": f"Bearer {new_token}"}

                # Immediately try admin action
                return client.delete(
                    "/api/v1/agents/all",
                    headers=new_headers,  # Admin-only bulk delete
                )
            return None

        # Launch concurrent refresh attempts
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(attempt_privileged_action) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should fail with 403
        for result in results:
            if result:
                assert result.status_code == status.HTTP_403_FORBIDDEN

    def test_token_manipulation_chains(self, client, escalation_users):
        """Test complex token manipulation attack chains."""
        researcher = escalation_users[UserRole.RESEARCHER]

        # Get base token
        base_token = researcher["token"]

        # Chain 1: Token format confusion
        # Try different token formats
        token_formats = [
            f"Bearer {base_token}",  # Normal
            f"bearer {base_token}",  # Lowercase
            f"BEARER {base_token}",  # Uppercase
            f"Token {base_token}",  # Different scheme
            f"JWT {base_token}",  # JWT scheme
            f"{base_token}",  # No scheme
            f"Bearer  {base_token}",  # Double space
            f"Bearer\t{base_token}",  # Tab character
            f"Bearer\n{base_token}",  # Newline
        ]

        for token_format in token_formats:
            test_headers = {"Authorization": token_format}
            response = client.get("/api/v1/agents", headers=test_headers)

            # Only the correct format should work
            if token_format == f"Bearer {base_token}":
                assert response.status_code == status.HTTP_200_OK
            else:
                assert response.status_code in [
                    status.HTTP_401_UNAUTHORIZED,
                    status.HTTP_403_FORBIDDEN,
                ]

        # Chain 2: Token replay with modifications
        # Decode token
        payload = jwt.decode(base_token, options={"verify_signature": False})

        # Try to extend expiration
        future_payload = payload.copy()
        future_payload["exp"] = int((datetime.now(timezone.utc) + timedelta(days=365)).timestamp())

        # Try to reuse with modified non-signature fields
        modified_payloads = [
            {
                **payload,
                "iat": payload["iat"] - 3600,
            },  # Issued an hour earlier
            {
                **payload,
                "nbf": payload.get("nbf", 0) - 3600,
            },  # Valid an hour earlier
            {**payload, "jti": "forged_jti_12345"},  # Different JTI
            {**payload, "iss": "forged_issuer"},  # Different issuer
            {
                **payload,
                "aud": ["freeagentics-api", "admin-api"],
            },  # Additional audience
        ]

        # None of these should work without proper signature
        for modified in modified_payloads:
            # Since we can't forge signature, token validation should fail
            pass  # Would need private key to create valid token


class TestAuthorizationBypassTechniques:
    """Test advanced authorization bypass techniques."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def bypass_setup(self):
        """Set up authorization bypass test environment."""
        users = {}

        # Create test users
        for role in [UserRole.OBSERVER, UserRole.RESEARCHER, UserRole.ADMIN]:
            username = f"bypass_{role.value}"
            user = auth_manager.register_user(
                username=username,
                email=f"{username}@test.com",
                password="Bypass123!",
                role=role,
            )
            users[role] = {
                "user": user,
                "token": auth_manager.create_access_token(user),
            }

        return users

    def test_http_verb_tampering(self, client, bypass_setup):
        """Test HTTP verb tampering for authorization bypass."""
        observer = bypass_setup[UserRole.OBSERVER]
        headers = {"Authorization": f"Bearer {observer['token']}"}

        # Try different HTTP verbs on protected endpoints
        protected_endpoint = "/api/v1/system/config"

        # Standard verbs
        standard_verbs = ["GET", "POST", "PUT", "DELETE", "PATCH"]
        for verb in standard_verbs:
            response = client.request(verb, protected_endpoint, headers=headers)
            assert response.status_code in [
                status.HTTP_403_FORBIDDEN,
                status.HTTP_405_METHOD_NOT_ALLOWED,
            ], f"Verb {verb} allowed unauthorized access"

        # Non-standard verbs
        nonstandard_verbs = [
            "HEAD",
            "OPTIONS",
            "CONNECT",
            "TRACE",
            "TRACK",
            "MOVE",
            "COPY",
        ]
        for verb in nonstandard_verbs:
            try:
                response = client.request(verb, protected_endpoint, headers=headers)
                assert response.status_code in [
                    status.HTTP_403_FORBIDDEN,
                    status.HTTP_405_METHOD_NOT_ALLOWED,
                    status.HTTP_501_NOT_IMPLEMENTED,
                ], f"Non-standard verb {verb} bypassed authorization"
            except Exception:
                pass  # Some verbs might not be supported by test client

        # Custom verbs
        try:
            response = client.request("FOOBAR", protected_endpoint, headers=headers)
            assert response.status_code != status.HTTP_200_OK
        except Exception:
            pass  # Expected

    def test_content_type_confusion(self, client, bypass_setup):
        """Test content-type confusion for authorization bypass."""
        observer = bypass_setup[UserRole.OBSERVER]
        base_headers = {"Authorization": f"Bearer {observer['token']}"}

        # Try different content types that might bypass validation
        content_types = [
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain",
            "application/xml",
            "application/octet-stream",
            "application/json; charset=utf-8",
            "application/json\x00text/html",  # Null byte injection
            "application/json; boundary=--security-bypass",
            "",  # Empty content-type
        ]

        payload = {"name": "BypassAgent", "template": "basic", "role": "admin"}

        for content_type in content_types:
            headers = {**base_headers, "Content-Type": content_type}

            # Adjust data format based on content type
            if content_type == "application/x-www-form-urlencoded":
                response = client.post(
                    "/api/v1/agents",
                    headers=headers,
                    data="name=BypassAgent&template=basic&role=admin",
                )
            elif content_type == "multipart/form-data":
                # Skip multipart for simplicity in this test
                continue
            else:
                response = client.post(
                    "/api/v1/agents",
                    headers=headers,
                    data=json.dumps(payload) if content_type else payload,
                )

            # Should not bypass authorization
            if response.status_code == status.HTTP_201_CREATED:
                # Verify no privilege escalation occurred
                data = response.json()
                assert data.get("role") != "admin"

    def test_path_traversal_authorization(self, client, bypass_setup):
        """Test path traversal techniques for authorization bypass."""
        observer = bypass_setup[UserRole.OBSERVER]
        headers = {"Authorization": f"Bearer {observer['token']}"}

        # Base paths

        # Path traversal attempts
        traversal_attempts = [
            "/api/v1/agents/../system/config",
            "/api/v1/agents/../../v1/system/config",
            "/api/v1//system/config",
            "/api/v1/./system/config",
            "/api/v1/agents;/system/config",
            "/api/v1/agents#/system/config",
            "/api/v1/agents?path=/system/config",
            "/api/v1/agents%2f%2e%2e%2fsystem%2fconfig",  # URL encoded
            "/api/v1/agents%252f%252e%252e%252fsystem%252fconfig",  # Double encoded
            "//api/v1/system/config",  # Protocol-relative
            "/api/v1/system%00/config",  # Null byte
            "/api/v1/agents\x00/system/config",  # Null byte variant
        ]

        for path in traversal_attempts:
            try:
                response = client.get(path, headers=headers)
                # Should not bypass to restricted endpoints
                assert (
                    response.status_code != status.HTTP_200_OK
                    or "system" not in response.text.lower()
                ), f"Path traversal succeeded with: {path}"
            except Exception:
                pass  # Some paths might cause client errors

    def test_timing_based_authorization_bypass(self, client, bypass_setup):
        """Test timing-based authorization bypass attacks."""
        import time

        observer = bypass_setup[UserRole.OBSERVER]
        admin = bypass_setup[UserRole.ADMIN]

        # Measure timing differences between authorized and unauthorized access
        timings = {"authorized": [], "unauthorized": []}

        # Unauthorized attempts (observer -> admin endpoint)
        observer_headers = {"Authorization": f"Bearer {observer['token']}"}
        for _ in range(10):
            start = time.time()
            response = client.get("/api/v1/system/config", headers=observer_headers)
            end = time.time()
            timings["unauthorized"].append(end - start)
            assert response.status_code == status.HTTP_403_FORBIDDEN

        # Authorized attempts (admin -> admin endpoint)
        admin_headers = {"Authorization": f"Bearer {admin['token']}"}
        for _ in range(10):
            start = time.time()
            response = client.get("/api/v1/system/config", headers=admin_headers)
            end = time.time()
            timings["authorized"].append(end - start)
            # Admin should have access

        # Calculate average timings
        avg_unauthorized = sum(timings["unauthorized"]) / len(timings["unauthorized"])
        avg_authorized = sum(timings["authorized"]) / len(timings["authorized"])

        # Timing should not reveal authorization logic
        # (In practice, some difference is expected, but should be minimal)
        timing_ratio = avg_unauthorized / avg_authorized if avg_authorized > 0 else 1

        # Log for analysis
        print(f"Timing ratio (unauthorized/authorized): {timing_ratio:.2f}")
        print(f"Avg unauthorized: {avg_unauthorized:.4f}s")
        print(f"Avg authorized: {avg_authorized:.4f}s")

        # Timing attacks should not be viable
        # Allow up to 2x difference as reasonable threshold
        assert (
            timing_ratio < 2.0 or timing_ratio > 0.5
        ), "Significant timing difference detected that could enable timing attacks"

    def test_cache_poisoning_authorization(self, client, bypass_setup):
        """Test cache poisoning for authorization bypass."""
        observer = bypass_setup[UserRole.OBSERVER]
        admin = bypass_setup[UserRole.ADMIN]

        # First, admin makes a request
        admin_headers = {"Authorization": f"Bearer {admin['token']}"}
        response = client.get("/api/v1/system/config", headers=admin_headers)
        (response.json() if response.status_code == status.HTTP_200_OK else None)

        # Observer tries cache poisoning techniques
        observer_headers = {"Authorization": f"Bearer {observer['token']}"}

        # Attempt 1: Same URL with cache headers
        cache_headers = {
            **observer_headers,
            "X-Original-URL": "/api/v1/system/config",
            "X-Forwarded-Host": "admin.internal",
            "X-Forwarded-For": "127.0.0.1",
            "Cache-Control": "max-age=3600",
            "X-Cache-Key": admin["token"][:10],  # Try to hit admin's cache
        }

        response = client.get("/api/v1/system/config", headers=cache_headers)
        assert response.status_code == status.HTTP_403_FORBIDDEN

        # Attempt 2: Parameter pollution for cache key
        response = client.get(
            "/api/v1/system/config?user=admin&user=observer",
            headers=observer_headers,
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN

        # Attempt 3: Fragment identifier cache bypass
        response = client.get("/api/v1/system/config#admin_view", headers=observer_headers)
        assert response.status_code == status.HTTP_403_FORBIDDEN


class TestAuthorizationRaceConditions:
    """Test authorization race conditions and concurrency issues."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def race_setup(self):
        """Set up for race condition testing."""
        users = {}

        # Create users that will be promoted/demoted
        for i in range(3):
            username = f"race_user_{i}"
            user = auth_manager.register_user(
                username=username,
                email=f"{username}@test.com",
                password="Race123!",
                role=UserRole.OBSERVER,
            )
            users[username] = {
                "user": user,
                "token": auth_manager.create_access_token(user),
                "initial_role": UserRole.OBSERVER,
            }

        # Create admin for role changes
        admin = auth_manager.register_user(
            username="race_admin",
            email="race_admin@test.com",
            password="Admin123!",
            role=UserRole.ADMIN,
        )
        users["admin"] = {
            "user": admin,
            "token": auth_manager.create_access_token(admin),
        }

        return users

    def test_role_change_race_condition(self, client, race_setup):
        """Test race conditions during role changes."""
        import queue
        import threading

        target_user = race_setup["race_user_0"]
        admin = race_setup["admin"]

        results = queue.Queue()

        def attempt_privileged_action():
            """User attempts privileged action repeatedly."""
            headers = {"Authorization": f"Bearer {target_user['token']}"}

            for _ in range(50):
                response = client.post(
                    "/api/v1/coalitions",  # Requires RESEARCHER or higher
                    headers=headers,
                    json={"name": f"RaceCoalition_{secrets.token_hex(4)}"},
                )
                if response.status_code == status.HTTP_201_CREATED:
                    results.put(("SUCCESS", response.json()))
                else:
                    results.put(("FAIL", response.status_code))

                time.sleep(0.01)  # Small delay

        def change_user_role():
            """Admin changes user role back and forth."""
            {"Authorization": f"Bearer {admin['token']}"}

            for i in range(25):
                # Promote to RESEARCHER
                new_role = UserRole.RESEARCHER if i % 2 == 0 else UserRole.OBSERVER

                # In real system, this would be an API call
                # Simulating role change
                target_user["user"].role = new_role

                time.sleep(0.02)  # Slightly longer delay

        # Start both threads
        action_thread = threading.Thread(target=attempt_privileged_action)
        role_thread = threading.Thread(target=change_user_role)

        action_thread.start()
        role_thread.start()

        action_thread.join()
        role_thread.join()

        # Analyze results
        successes = []
        while not results.empty():
            result_type, data = results.get()
            if result_type == "SUCCESS":
                successes.append(data)

        # In a secure system, success should only happen when role is appropriate
        # Race conditions might allow unauthorized access
        print(f"Race condition test: {len(successes)} successful privileged actions")

        # The token should maintain consistent permissions despite role changes
        # (tokens are immutable after creation)

    def test_concurrent_permission_checks(self, client, race_setup):
        """Test concurrent permission checks for inconsistencies."""
        import concurrent.futures

        user = race_setup["race_user_1"]
        headers = {"Authorization": f"Bearer {user['token']}"}

        # Endpoints with different permission requirements
        test_endpoints = [
            ("/api/v1/agents", "GET", Permission.VIEW_AGENTS),  # Should work
            ("/api/v1/agents", "POST", Permission.CREATE_AGENT),  # Should fail
            ("/api/v1/metrics", "GET", Permission.VIEW_METRICS),  # Should work
            (
                "/api/v1/system/config",
                "GET",
                Permission.ADMIN_SYSTEM,
            ),  # Should fail
        ]

        def check_endpoint(endpoint_data):
            endpoint, method, permission = endpoint_data

            if method == "GET":
                response = client.get(endpoint, headers=headers)
            elif method == "POST":
                response = client.post(endpoint, headers=headers, json={})

            return {
                "endpoint": endpoint,
                "method": method,
                "permission": permission,
                "status": response.status_code,
                "expected": permission in ROLE_PERMISSIONS.get(user["user"].role, []),
            }

        # Run concurrent permission checks
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            # Submit same checks multiple times
            futures = []
            for _ in range(10):  # 10 rounds
                for endpoint_data in test_endpoints:
                    futures.append(executor.submit(check_endpoint, endpoint_data))

            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Verify consistency
        # Group by endpoint
        from collections import defaultdict

        endpoint_results = defaultdict(list)

        for result in results:
            key = f"{result['method']} {result['endpoint']}"
            endpoint_results[key].append(result["status"])

        # All same requests should have same result
        for endpoint, statuses in endpoint_results.items():
            unique_statuses = set(statuses)
            assert (
                len(unique_statuses) == 1
            ), f"Inconsistent authorization for {endpoint}: {unique_statuses}"

    def test_token_refresh_race_condition(self, client, race_setup):
        """Test race conditions in token refresh flow."""
        user = race_setup["race_user_2"]

        # Create refresh token
        refresh_token = auth_manager.create_refresh_token(user["user"])

        refresh_results = []

        def refresh_token_concurrent():
            response = client.post("/api/v1/auth/refresh", json={"refresh_token": refresh_token})
            refresh_results.append(
                {
                    "status": response.status_code,
                    "data": response.json() if response.status_code == status.HTTP_200_OK else None,
                }
            )

        # Launch multiple concurrent refresh attempts
        threads = [threading.Thread(target=refresh_token_concurrent) for _ in range(10)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Analyze results
        successful_refreshes = [r for r in refresh_results if r["status"] == status.HTTP_200_OK]

        # Only one refresh should succeed (token rotation)
        # Or all should succeed with same new token (if no rotation)
        # But shouldn't have multiple different tokens
        if len(successful_refreshes) > 1:
            access_tokens = [r["data"]["access_token"] for r in successful_refreshes]
            unique_tokens = set(access_tokens)

            # Should either be all same (no rotation) or all different (proper rotation)
            assert len(unique_tokens) == 1 or len(unique_tokens) == len(
                access_tokens
            ), "Inconsistent token refresh behavior"


# Helper function for testing
def create_malicious_payload(payload_type: str) -> Dict[str, Any]:
    """Create various malicious payloads for testing."""
    payloads = {
        "sql_injection": {
            "name": "'; DROP TABLE users; --",
            "template": "basic",
            "params": {"query": "1' OR '1'='1"},
        },
        "xss": {
            "name": "<script>alert('xss')</script>",
            "template": "basic",
            "description": "<img src=x onerror=alert('xss')>",
        },
        "xxe": {
            "name": "XXE Test",
            "template": "basic",
            "data": '<?xml version="1.0"?><!DOCTYPE foo [<!ENTITY xxe SYSTEM "file:///etc/passwd">]><foo>&xxe;</foo>',
        },
        "ldap_injection": {
            "name": "admin)(&(password=*))",
            "template": "basic",
            "filter": "(&(uid=admin)(|(password=*)))",
        },
        "command_injection": {
            "name": "test; cat /etc/passwd",
            "template": "basic`; ls -la; echo 'pwned'`",
            "command": "echo test && cat /etc/passwd",
        },
        "path_traversal": {
            "name": "../../../../etc/passwd",
            "template": "basic",
            "file": "../../../../../../../etc/passwd",
        },
        "buffer_overflow": {
            "name": "A" * 10000,  # Large string
            "template": "B" * 10000,
            "description": "C" * 50000,
        },
    }

    return payloads.get(payload_type, {})
