"""Comprehensive IDOR (Insecure Direct Object Reference) vulnerability tests.

This module tests all IDOR attack patterns to ensure the FreeAgentics platform
is protected against unauthorized access to resources.

Test Coverage:
- Sequential ID Enumeration
- UUID/GUID Attacks
- Parameter Manipulation
- Authorization Bypass
- Advanced IDOR Attacks
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Dict, Optional, Tuple

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from api.main import app
from auth.security_implementation import TokenManager, UserRole, create_access_token
from database.models import Agent, AgentStatus, Coalition, CoalitionStatus


class IDORTestBase:
    """Base class for IDOR vulnerability testing."""

    def __init__(self):
        self.client = TestClient(app)
        self.token_manager = TokenManager()
        self.test_users = {}
        self.test_resources = {}

    def create_test_user(
        self, username: str, role: UserRole, user_id: Optional[str] = None
    ) -> Tuple[str, Dict]:
        """Create a test user with authentication token."""
        if not user_id:
            user_id = str(uuid.uuid4())

        # Create user data
        user_data = {
            "user_id": user_id,
            "username": username,
            "email": f"{username}@test.com",
            "role": role,
        }

        # Generate token
        token = create_access_token(data=user_data)

        self.test_users[username] = {
            "user_data": user_data,
            "token": token,
            "headers": {"Authorization": f"Bearer {token}"},
        }

        return user_id, user_data

    def create_test_agent(self, db: Session, owner_id: str, name: str) -> str:
        """Create a test agent owned by a specific user."""
        agent = Agent(
            id=uuid.uuid4(),
            name=name,
            template="test-template",
            status=AgentStatus.ACTIVE,
            parameters={"owner_id": owner_id},
        )
        db.add(agent)
        db.commit()
        return str(agent.id)

    def create_test_coalition(self, db: Session, leader_id: str, name: str) -> str:
        """Create a test coalition led by a specific user."""
        coalition = Coalition(
            id=uuid.uuid4(),
            name=name,
            purpose=f"Test coalition {name}",
            status=CoalitionStatus.ACTIVE,
            leader_id=leader_id,
            parameters={"created_by": leader_id},
        )
        db.add(coalition)
        db.commit()
        return str(coalition.id)


class TestSequentialIDEnumeration(IDORTestBase):
    """Test IDOR vulnerabilities through sequential ID enumeration."""

    @pytest.fixture(autouse=True)
    def setup(self, db: Session):
        """Set up test data."""
        # Create multiple users with different roles
        self.user1_id, _ = self.create_test_user("user1", UserRole.RESEARCHER)
        self.user2_id, _ = self.create_test_user("user2", UserRole.RESEARCHER)
        self.admin_id, _ = self.create_test_user("admin", UserRole.ADMIN)

        # Create resources owned by different users
        self.user1_agents = []
        self.user2_agents = []

        for i in range(5):
            agent_id = self.create_test_agent(db, self.user1_id, f"user1_agent_{i}")
            self.user1_agents.append(agent_id)

        for i in range(5):
            agent_id = self.create_test_agent(db, self.user2_id, f"user2_agent_{i}")
            self.user2_agents.append(agent_id)

    def test_agent_id_enumeration(self):
        """Test enumeration of agent IDs to access unauthorized resources."""
        # User1 tries to access User2's agents
        user1_headers = self.test_users["user1"]["headers"]

        for agent_id in self.user2_agents:
            response = self.client.get(f"/api/v1/agents/{agent_id}", headers=user1_headers)
            # Should be forbidden or filtered
            assert response.status_code in [
                status.HTTP_403_FORBIDDEN,
                status.HTTP_404_NOT_FOUND,
            ], f"User1 should not access User2's agent {agent_id}"

    def test_sequential_id_brute_force(self):
        """Test brute force attempts on sequential IDs."""
        user1_headers = self.test_users["user1"]["headers"]

        # Try to access resources with predictable IDs
        accessed_resources = []
        for i in range(100):
            # Try different ID formats
            test_ids = [
                str(i),  # Simple numeric
                f"agent_{i}",  # Prefixed numeric
                f"{i:06d}",  # Zero-padded
                f"resource-{i}",  # Hyphenated
            ]

            for test_id in test_ids:
                response = self.client.get(f"/api/v1/agents/{test_id}", headers=user1_headers)
                if response.status_code == status.HTTP_200_OK:
                    accessed_resources.append((test_id, response.json()))

        # Should not have accessed any resources through enumeration
        assert (
            len(accessed_resources) == 0
        ), f"Accessed {len(accessed_resources)} resources through ID enumeration"

    def test_user_id_enumeration(self):
        """Test enumeration of user IDs to access profiles."""
        user1_headers = self.test_users["user1"]["headers"]

        # Try to enumerate user IDs
        for i in range(1000, 1100):
            response = self.client.get(f"/api/v1/users/{i}", headers=user1_headers)
            # Should not expose user existence
            assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_coalition_id_enumeration(self, db: Session):
        """Test coalition ID enumeration vulnerabilities."""
        # Create coalitions
        self.create_test_coalition(db, self.user1_id, "Coalition1")
        coalition2 = self.create_test_coalition(db, self.user2_id, "Coalition2")

        user1_headers = self.test_users["user1"]["headers"]

        # User1 should not access User2's coalition details
        response = self.client.get(f"/api/v1/coalitions/{coalition2}", headers=user1_headers)
        assert response.status_code in [status.HTTP_403_FORBIDDEN, status.HTTP_404_NOT_FOUND]


class TestUUIDAttacks(IDORTestBase):
    """Test IDOR vulnerabilities through UUID/GUID manipulation."""

    def test_uuid_prediction(self):
        """Test UUID prediction based on patterns."""
        admin_headers = self.test_users["admin"]["headers"]

        # Create multiple resources and analyze UUID patterns
        created_uuids = []
        for i in range(10):
            response = self.client.post(
                "/api/v1/agents",
                headers=admin_headers,
                json={
                    "name": f"test_agent_{i}",
                    "template": "basic",
                },
            )
            if response.status_code == status.HTTP_201_CREATED:
                created_uuids.append(response.json()["id"])

        # Try to predict next UUID (should fail)
        # This is a simplified test - real UUID prediction would be more complex
        user1_headers = self.test_users["user1"]["headers"]

        # Generate potential UUIDs based on patterns
        for _ in range(100):
            predicted_uuid = str(uuid.uuid4())
            response = self.client.get(f"/api/v1/agents/{predicted_uuid}", headers=user1_headers)
            # Should not find any through prediction
            assert response.status_code != status.HTTP_200_OK

    def test_uuid_version_attacks(self):
        """Test attacks exploiting different UUID versions."""
        user1_headers = self.test_users["user1"]["headers"]

        # UUID v1 (timestamp-based) - predictable if MAC address is known
        current_time = int(time.time() * 10000000)
        for i in range(10):
            # Try timestamp-based UUIDs
            test_uuid = f"{current_time + i:032x}"
            formatted_uuid = f"{test_uuid[:8]}-{test_uuid[8:12]}-{test_uuid[12:16]}-{test_uuid[16:20]}-{test_uuid[20:32]}"

            response = self.client.get(f"/api/v1/agents/{formatted_uuid}", headers=user1_headers)
            assert response.status_code != status.HTTP_200_OK

    def test_uuid_collision_attempts(self):
        """Test UUID collision attempts."""
        user1_headers = self.test_users["user1"]["headers"]

        # Try known problematic UUIDs
        problematic_uuids = [
            "00000000-0000-0000-0000-000000000000",  # Nil UUID
            "ffffffff-ffff-ffff-ffff-ffffffffffff",  # Max UUID
            "12345678-1234-1234-1234-123456789012",  # Sequential
            "11111111-2222-3333-4444-555555555555",  # Pattern
        ]

        for test_uuid in problematic_uuids:
            response = self.client.get(f"/api/v1/agents/{test_uuid}", headers=user1_headers)
            assert response.status_code != status.HTTP_200_OK


class TestParameterManipulation(IDORTestBase):
    """Test IDOR through parameter manipulation."""

    @pytest.fixture(autouse=True)
    def setup(self, db: Session):
        """Set up test data."""
        self.user1_id, _ = self.create_test_user("user1", UserRole.RESEARCHER)
        self.user2_id, _ = self.create_test_user("user2", UserRole.RESEARCHER)

        # Create test resources
        self.user1_agent = self.create_test_agent(db, self.user1_id, "user1_agent")
        self.user2_agent = self.create_test_agent(db, self.user2_id, "user2_agent")

    def test_query_parameter_idor(self):
        """Test IDOR through query parameter manipulation."""
        user1_headers = self.test_users["user1"]["headers"]

        # Try to access other user's resources through query parameters
        attack_params = [
            {"user_id": self.user2_id},
            {"owner": self.user2_id},
            {"filter": f"owner_id:{self.user2_id}"},
            {"id": self.user2_agent},
            {"agent_id": self.user2_agent},
        ]

        for params in attack_params:
            response = self.client.get("/api/v1/agents", headers=user1_headers, params=params)
            if response.status_code == status.HTTP_200_OK:
                agents = response.json()
                # Should not return other user's agents
                for agent in agents:
                    assert agent["id"] != self.user2_agent, f"Leaked agent through params: {params}"

    def test_path_parameter_idor(self):
        """Test IDOR through path parameter manipulation."""
        user1_headers = self.test_users["user1"]["headers"]

        # Direct path parameter manipulation
        response = self.client.get(f"/api/v1/agents/{self.user2_agent}", headers=user1_headers)
        assert response.status_code in [status.HTTP_403_FORBIDDEN, status.HTTP_404_NOT_FOUND]

        # Path traversal attempts
        traversal_attempts = [
            f"../agents/{self.user2_agent}",
            f"agents/../agents/{self.user2_agent}",
            f"{self.user1_agent}/../{self.user2_agent}",
            f"agents/{self.user2_agent}",
        ]

        for path in traversal_attempts:
            response = self.client.get(f"/api/v1/{path}", headers=user1_headers)
            assert response.status_code != status.HTTP_200_OK

    def test_json_payload_idor(self):
        """Test IDOR through JSON payload manipulation."""
        user1_headers = self.test_users["user1"]["headers"]

        # Try to update another user's agent
        attack_payloads = [
            {"agent_id": self.user2_agent, "status": "stopped"},
            {"id": self.user2_agent, "name": "hacked"},
            {"target": self.user2_agent, "action": "delete"},
        ]

        for payload in attack_payloads:
            response = self.client.patch(
                f"/api/v1/agents/{self.user2_agent}/status", headers=user1_headers, json=payload
            )
            assert response.status_code in [status.HTTP_403_FORBIDDEN, status.HTTP_404_NOT_FOUND]

    def test_form_data_idor(self):
        """Test IDOR through form data manipulation."""
        user1_headers = self.test_users["user1"]["headers"]

        # Try form-based attacks
        attack_data = {
            "agent_id": self.user2_agent,
            "user_id": self.user2_id,
            "owner": self.user2_id,
        }

        response = self.client.post(
            "/api/v1/agents/action", headers=user1_headers, data=attack_data
        )
        assert response.status_code != status.HTTP_200_OK


class TestAuthorizationBypass(IDORTestBase):
    """Test authorization bypass IDOR attacks."""

    @pytest.fixture(autouse=True)
    def setup(self, db: Session):
        """Set up test data with complex ownership."""
        self.user1_id, _ = self.create_test_user("user1", UserRole.RESEARCHER)
        self.user2_id, _ = self.create_test_user("user2", UserRole.RESEARCHER)
        self.observer_id, _ = self.create_test_user("observer", UserRole.OBSERVER)

        # Create resources with different access levels
        self.private_agent = self.create_test_agent(db, self.user1_id, "private_agent")
        self.shared_coalition = self.create_test_coalition(db, self.user1_id, "shared_coalition")

    def test_direct_object_access(self):
        """Test direct object access bypassing authorization."""
        observer_headers = self.test_users["observer"]["headers"]

        # Observer should have limited access
        response = self.client.delete(
            f"/api/v1/agents/{self.private_agent}", headers=observer_headers
        )
        assert response.status_code == status.HTTP_403_FORBIDDEN

        # Try to modify through different endpoints
        modify_attempts = [
            ("PATCH", f"/api/v1/agents/{self.private_agent}/status"),
            ("PUT", f"/api/v1/agents/{self.private_agent}"),
            ("POST", f"/api/v1/agents/{self.private_agent}/action"),
        ]

        for method, path in modify_attempts:
            response = self.client.request(
                method, path, headers=observer_headers, json={"status": "stopped"}
            )
            assert response.status_code in [
                status.HTTP_403_FORBIDDEN,
                status.HTTP_405_METHOD_NOT_ALLOWED,
            ]

    def test_resource_ownership_bypass(self):
        """Test bypassing resource ownership checks."""
        user2_headers = self.test_users["user2"]["headers"]

        # Try to claim ownership of another user's resource
        ownership_attacks = [
            {"owner_id": self.user2_id},
            {"owner": self.user2_id},
            {"user_id": self.user2_id},
            {"created_by": self.user2_id},
        ]

        for attack in ownership_attacks:
            response = self.client.patch(
                f"/api/v1/agents/{self.private_agent}", headers=user2_headers, json=attack
            )
            assert response.status_code in [status.HTTP_403_FORBIDDEN, status.HTTP_404_NOT_FOUND]

    def test_cross_tenant_access(self, db: Session):
        """Test cross-tenant access in multi-tenant scenarios."""
        # Create tenant-specific resources
        tenant1_agent = self.create_test_agent(db, self.user1_id, "tenant1_agent")
        tenant2_agent = self.create_test_agent(db, self.user2_id, "tenant2_agent")

        # Update agents with tenant info
        db.query(Agent).filter(Agent.id == uuid.UUID(tenant1_agent)).update(
            {"parameters": {"tenant_id": "tenant1", "owner_id": self.user1_id}}
        )
        db.query(Agent).filter(Agent.id == uuid.UUID(tenant2_agent)).update(
            {"parameters": {"tenant_id": "tenant2", "owner_id": self.user2_id}}
        )
        db.commit()

        # Try cross-tenant access
        user1_headers = self.test_users["user1"]["headers"]
        response = self.client.get(f"/api/v1/agents/{tenant2_agent}", headers=user1_headers)
        assert response.status_code in [status.HTTP_403_FORBIDDEN, status.HTTP_404_NOT_FOUND]

    def test_file_path_traversal_idor(self):
        """Test file path traversal IDOR attacks."""
        user1_headers = self.test_users["user1"]["headers"]

        # Try to access files through path traversal
        traversal_attempts = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "agents/../../../database/config",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",  # URL encoded
            "....//....//....//etc/passwd",  # Double dots
        ]

        for path in traversal_attempts:
            response = self.client.get(f"/api/v1/files/{path}", headers=user1_headers)
            assert response.status_code != status.HTTP_200_OK


class TestAdvancedIDORAttacks(IDORTestBase):
    """Test advanced IDOR attack patterns."""

    @pytest.fixture(autouse=True)
    def setup(self, db: Session):
        """Set up complex test scenario."""
        self.user1_id, _ = self.create_test_user("user1", UserRole.RESEARCHER)
        self.user2_id, _ = self.create_test_user("user2", UserRole.RESEARCHER)

        # Create interlinked resources
        self.user1_agent = self.create_test_agent(db, self.user1_id, "user1_agent")
        self.user2_agent = self.create_test_agent(db, self.user2_id, "user2_agent")
        self.shared_coalition = self.create_test_coalition(db, self.user1_id, "shared")

    def test_blind_idor_detection(self):
        """Test blind IDOR vulnerabilities through timing and response differences."""
        user1_headers = self.test_users["user1"]["headers"]

        # Measure response times for valid vs invalid resources
        valid_times = []
        invalid_times = []

        # Time valid resource access
        for _ in range(10):
            start = time.time()
            response = self.client.get(f"/api/v1/agents/{self.user1_agent}", headers=user1_headers)
            valid_times.append(time.time() - start)
            assert response.status_code == status.HTTP_200_OK

        # Time invalid resource access
        for _ in range(10):
            fake_id = str(uuid.uuid4())
            start = time.time()
            response = self.client.get(f"/api/v1/agents/{fake_id}", headers=user1_headers)
            invalid_times.append(time.time() - start)

        # Response times should be similar (no timing leaks)
        avg_valid = sum(valid_times) / len(valid_times)
        avg_invalid = sum(invalid_times) / len(invalid_times)

        # Allow 50ms difference maximum
        assert (
            abs(avg_valid - avg_invalid) < 0.05
        ), "Timing attack possible - different response times"

    def test_time_based_idor(self):
        """Test time-based IDOR attacks using timestamps."""
        user1_headers = self.test_users["user1"]["headers"]

        # Try to access resources created at specific times
        current_timestamp = int(time.time())

        # Try various timestamp-based IDs
        for offset in range(-3600, 3600, 300):  # -1 hour to +1 hour, 5 min intervals
            timestamp = current_timestamp + offset

            # Try different timestamp formats
            timestamp_ids = [
                str(timestamp),
                f"agent_{timestamp}",
                datetime.fromtimestamp(timestamp).isoformat(),
                f"{timestamp}_resource",
            ]

            for test_id in timestamp_ids:
                response = self.client.get(f"/api/v1/agents/{test_id}", headers=user1_headers)
                assert response.status_code != status.HTTP_200_OK

    def test_mass_assignment_idor(self):
        """Test IDOR through mass assignment vulnerabilities."""
        user1_headers = self.test_users["user1"]["headers"]

        # Try to assign properties that change ownership
        mass_assignment_attacks = [
            {
                "name": "Updated Agent",
                "owner_id": self.user1_id,  # Try to claim ownership
                "id": self.user2_agent,  # Try to change ID
            },
            {
                "name": "Updated Agent",
                "_id": self.user2_agent,  # Alternative ID field
                "userId": self.user1_id,
            },
            {
                "name": "Updated Agent",
                "parameters": {
                    "owner_id": self.user1_id,
                    "admin": True,
                    "bypass_auth": True,
                },
            },
        ]

        for payload in mass_assignment_attacks:
            # Try on create
            response = self.client.post("/api/v1/agents", headers=user1_headers, json=payload)
            if response.status_code == status.HTTP_201_CREATED:
                created = response.json()
                # Should not have bypassed ownership
                assert created.get("id") != self.user2_agent
                assert created.get("owner_id") != self.user2_id

            # Try on update
            response = self.client.put(
                f"/api/v1/agents/{self.user1_agent}", headers=user1_headers, json=payload
            )

    def test_indirect_object_references(self, db: Session):
        """Test IDOR through indirect references."""
        # Create linked resources
        user1_headers = self.test_users["user1"]["headers"]

        # Create agent with reference to another user's resource
        response = self.client.post(
            "/api/v1/agents",
            headers=user1_headers,
            json={
                "name": "indirect_ref_agent",
                "template": "basic",
                "parameters": {
                    "linked_agent": self.user2_agent,
                    "coalition_id": self.shared_coalition,
                },
            },
        )

        if response.status_code == status.HTTP_201_CREATED:
            new_agent_id = response.json()["id"]

            # Try to access linked resource through the reference
            response = self.client.get(
                f"/api/v1/agents/{new_agent_id}/linked", headers=user1_headers
            )
            # Should not expose linked resource details
            if response.status_code == status.HTTP_200_OK:
                data = response.json()
                assert self.user2_agent not in str(data)

    def test_race_condition_idor(self):
        """Test IDOR vulnerabilities through race conditions."""
        user1_headers = self.test_users["user1"]["headers"]
        user2_headers = self.test_users["user2"]["headers"]

        async def attempt_access(headers, agent_id):
            """Attempt to access an agent."""
            return self.client.get(f"/api/v1/agents/{agent_id}", headers=headers)

        async def attempt_delete(headers, agent_id):
            """Attempt to delete an agent."""
            return self.client.delete(f"/api/v1/agents/{agent_id}", headers=headers)

        # Create a new agent as user2
        response = self.client.post(
            "/api/v1/agents",
            headers=user2_headers,
            json={"name": "race_condition_test", "template": "basic"},
        )

        if response.status_code == status.HTTP_201_CREATED:
            race_agent_id = response.json()["id"]

            # Try to exploit race condition during deletion
            # User2 deletes while User1 tries to access
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run concurrent requests
            tasks = [
                attempt_delete(user2_headers, race_agent_id),
                attempt_access(user1_headers, race_agent_id),
                attempt_access(user1_headers, race_agent_id),
                attempt_access(user1_headers, race_agent_id),
            ]

            results = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

            # Check that user1 never got access
            for result in results[1:]:  # Skip delete result
                if hasattr(result, "status_code"):
                    assert result.status_code != status.HTTP_200_OK


class TestIDORProtectionValidation(IDORTestBase):
    """Validate that IDOR protections are properly implemented."""

    def test_proper_authorization_checks(self, db: Session):
        """Ensure all endpoints have proper authorization checks."""
        # Create test data
        user1_id, _ = self.create_test_user("user1", UserRole.RESEARCHER)
        user2_id, _ = self.create_test_user("user2", UserRole.RESEARCHER)

        self.create_test_agent(db, user1_id, "agent1")
        agent2 = self.create_test_agent(db, user2_id, "agent2")

        user1_headers = self.test_users["user1"]["headers"]

        # Test all CRUD operations
        crud_tests = [
            ("GET", f"/api/v1/agents/{agent2}", None),
            ("PUT", f"/api/v1/agents/{agent2}", {"name": "updated"}),
            ("PATCH", f"/api/v1/agents/{agent2}/status", {"status": "stopped"}),
            ("DELETE", f"/api/v1/agents/{agent2}", None),
        ]

        for method, path, json_data in crud_tests:
            response = self.client.request(method, path, headers=user1_headers, json=json_data)
            assert response.status_code in [
                status.HTTP_403_FORBIDDEN,
                status.HTTP_404_NOT_FOUND,
                status.HTTP_405_METHOD_NOT_ALLOWED,
            ], f"{method} {path} allowed unauthorized access"

    def test_consistent_error_responses(self):
        """Ensure error responses don't leak information."""
        user1_headers = self.test_users["user1"]["headers"]

        # Test non-existent vs forbidden resources
        fake_id = str(uuid.uuid4())

        # Non-existent resource
        response1 = self.client.get(f"/api/v1/agents/{fake_id}", headers=user1_headers)

        # Forbidden resource (if we can identify one)
        # Error responses should be consistent
        assert response1.status_code == status.HTTP_404_NOT_FOUND
        assert "id" not in response1.text.lower()  # Don't leak IDs
        assert "user" not in response1.text.lower()  # Don't leak user info

    def test_secure_id_generation(self):
        """Verify that IDs are generated securely."""
        admin_headers = self.test_users.get("admin", {}).get("headers")
        if not admin_headers:
            _, _ = self.create_test_user("admin", UserRole.ADMIN)
            admin_headers = self.test_users["admin"]["headers"]

        created_ids = []

        # Create multiple resources
        for i in range(20):
            response = self.client.post(
                "/api/v1/agents",
                headers=admin_headers,
                json={"name": f"secure_test_{i}", "template": "basic"},
            )

            if response.status_code == status.HTTP_201_CREATED:
                created_ids.append(response.json()["id"])

        # Verify IDs are UUIDs
        for resource_id in created_ids:
            try:
                uuid.UUID(resource_id)
            except ValueError:
                pytest.fail(f"Non-UUID ID detected: {resource_id}")

        # Verify no sequential patterns
        assert len(set(created_ids)) == len(created_ids), "Duplicate IDs detected"


# Performance impact tests
class TestIDORPerformanceImpact:
    """Test that IDOR protections don't significantly impact performance."""

    def test_authorization_performance(self, benchmark):
        """Benchmark authorization checks."""
        test_base = IDORTestBase()
        user_id, _ = test_base.create_test_user("perf_user", UserRole.RESEARCHER)
        headers = test_base.test_users["perf_user"]["headers"]

        def make_authorized_request():
            response = test_base.client.get("/api/v1/agents", headers=headers)
            assert response.status_code in [status.HTTP_200_OK, status.HTTP_404_NOT_FOUND]

        # Benchmark should complete within reasonable time
        benchmark(make_authorized_request)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
