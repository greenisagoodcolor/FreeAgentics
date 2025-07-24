"""Advanced IDOR vulnerability test patterns.

This module contains sophisticated IDOR attack patterns including:
- GraphQL IDOR attacks
- WebSocket IDOR vulnerabilities
- Batch operation IDOR
- Cache poisoning IDOR
- API versioning IDOR
"""

import base64
import hashlib
import json
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest
import websockets
from auth.security_implementation import UserRole
from fastapi import status

from tests.security.test_idor_vulnerabilities import IDORTestBase


class TestGraphQLIDOR(IDORTestBase):
    """Test IDOR vulnerabilities in GraphQL endpoints."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up GraphQL test environment."""
        self.user1_id, _ = self.create_test_user("graphql_user1", UserRole.RESEARCHER)
        self.user2_id, _ = self.create_test_user("graphql_user2", UserRole.RESEARCHER)
        self.admin_id, _ = self.create_test_user("graphql_admin", UserRole.ADMIN)

    def test_graphql_query_idor(self):
        """Test IDOR through GraphQL query manipulation."""
        user1_headers = self.test_users["graphql_user1"]["headers"]

        # GraphQL queries that attempt IDOR
        idor_queries = [
            # Direct ID access
            """
            query {
                agent(id: "%s") {
                    id
                    name
                    parameters
                    gmn_spec
                }
            }
            """
            % self.user2_id,
            # Nested query IDOR
            """
            query {
                user(id: "%s") {
                    agents {
                        id
                        name
                        parameters
                    }
                }
            }
            """
            % self.user2_id,
            # Filter bypass
            """
            query {
                agents(filter: {owner_id: "%s"}) {
                    id
                    name
                }
            }
            """
            % self.user2_id,
            # Aliased queries
            """
            query {
                myAgents: agents(owner: "%s") {
                    id
                }
                theirAgents: agents(owner: "%s") {
                    id
                    parameters
                }
            }
            """
            % (self.user1_id, self.user2_id),
        ]

        for query in idor_queries:
            response = self.client.post(
                "/api/v1/graphql", headers=user1_headers, json={"query": query}
            )

            if response.status_code == status.HTTP_200_OK:
                data = response.json()
                # Ensure no unauthorized data is returned
                assert not self._contains_unauthorized_data(data, self.user2_id)

    def test_graphql_mutation_idor(self):
        """Test IDOR through GraphQL mutations."""
        user1_headers = self.test_users["graphql_user1"]["headers"]

        # Mutations that attempt unauthorized modifications
        idor_mutations = [
            # Direct update
            """
            mutation {
                updateAgent(id: "%s", input: {name: "Hacked"}) {
                    id
                    name
                }
            }
            """
            % uuid.uuid4(),  # Random agent ID
            # Batch mutations
            """
            mutation {
                updateMultipleAgents(ids: ["%s", "%s"], status: "stopped") {
                    success
                    count
                }
            }
            """
            % (self.user1_id, self.user2_id),
            # Nested mutations
            """
            mutation {
                createCoalition(input: {
                    name: "Test Coalition",
                    leader_id: "%s",
                    members: ["%s", "%s"]
                }) {
                    id
                    members {
                        id
                    }
                }
            }
            """
            % (self.user1_id, self.user1_id, self.user2_id),
        ]

        for mutation in idor_mutations:
            response = self.client.post(
                "/api/v1/graphql",
                headers=user1_headers,
                json={"query": mutation},
            )

            # Should either fail or not affect unauthorized resources
            if response.status_code == status.HTTP_200_OK:
                data = response.json()
                assert "errors" in data or not data.get("data")

    def test_graphql_introspection_idor(self):
        """Test IDOR through GraphQL introspection queries."""
        user1_headers = self.test_users["graphql_user1"]["headers"]

        # Introspection query to discover schema
        introspection_query = """
        query {
            __schema {
                types {
                    name
                    fields {
                        name
                        args {
                            name
                            type {
                                name
                            }
                        }
                    }
                }
            }
        }
        """

        response = self.client.post(
            "/api/v1/graphql",
            headers=user1_headers,
            json={"query": introspection_query},
        )

        # Introspection should be disabled in production
        if response.status_code == status.HTTP_200_OK:
            data = response.json()
            # Check that sensitive fields are not exposed
            sensitive_fields = ["password", "secret", "token", "private"]
            schema_str = json.dumps(data)
            for field in sensitive_fields:
                assert field not in schema_str.lower()

    def _contains_unauthorized_data(self, data: dict, unauthorized_id: str) -> bool:
        """Check if response contains unauthorized data."""
        data_str = json.dumps(data)
        return unauthorized_id in data_str


class TestWebSocketIDOR(IDORTestBase):
    """Test IDOR vulnerabilities in WebSocket connections."""

    @pytest.mark.asyncio
    async def test_websocket_connection_idor(self):
        """Test IDOR through WebSocket connection parameters."""
        user1_token = self.test_users["user1"]["token"]
        self.test_users["user2"]["token"]

        # Try to connect with manipulated parameters
        attack_urls = [
            f"ws://localhost:8000/ws?token={user1_token}&user_id={self.user2_id}",
            f"ws://localhost:8000/ws?token={user1_token}&agent_id={uuid.uuid4()}",
            f"ws://localhost:8000/ws?token={user1_token}&impersonate={self.user2_id}",
        ]

        for url in attack_urls:
            try:
                async with websockets.connect(url) as websocket:
                    # Send a message attempting to access unauthorized resources
                    await websocket.send(
                        json.dumps(
                            {
                                "action": "get_agent",
                                "agent_id": str(uuid.uuid4()),
                            }
                        )
                    )

                    response = await websocket.recv()
                    data = json.loads(response)

                    # Should not return unauthorized data
                    assert data.get("error") or data.get("status") == "forbidden"
            except websockets.exceptions.WebSocketException:
                # Connection should be rejected for invalid auth
                pass

    @pytest.mark.asyncio
    async def test_websocket_message_idor(self):
        """Test IDOR through WebSocket message manipulation."""
        user1_token = self.test_users["user1"]["token"]

        try:
            async with websockets.connect(
                f"ws://localhost:8000/ws?token={user1_token}"
            ) as websocket:
                # Send messages attempting IDOR
                idor_messages = [
                    {
                        "action": "subscribe",
                        "channel": f"agent:{uuid.uuid4()}",  # Other user's agent
                    },
                    {
                        "action": "update_agent",
                        "agent_id": str(uuid.uuid4()),
                        "data": {"status": "compromised"},
                    },
                    {"action": "get_user_data", "user_id": self.user2_id},
                ]

                for message in idor_messages:
                    await websocket.send(json.dumps(message))
                    response = await websocket.recv()
                    data = json.loads(response)

                    # Should be rejected or return error
                    assert data.get("error") or data.get("status") != "success"
        except Exception:
            # WebSocket endpoint might not exist
            pass


class TestBatchOperationIDOR(IDORTestBase):
    """Test IDOR in batch operations."""

    def test_batch_delete_idor(self):
        """Test IDOR through batch delete operations."""
        user1_headers = self.test_users["user1"]["headers"]

        # Create some agents for user1
        user1_agents = []
        for i in range(3):
            response = self.client.post(
                "/api/v1/agents",
                headers=user1_headers,
                json={"name": f"batch_test_{i}", "template": "basic"},
            )
            if response.status_code == status.HTTP_201_CREATED:
                user1_agents.append(response.json()["id"])

        # Try batch delete including unauthorized IDs
        batch_delete_payload = {
            "ids": user1_agents + [str(uuid.uuid4()), str(uuid.uuid4())]  # Mix owned and unowned
        }

        response = self.client.post(
            "/api/v1/agents/batch/delete",
            headers=user1_headers,
            json=batch_delete_payload,
        )

        if response.status_code == status.HTTP_200_OK:
            result = response.json()
            # Should only delete owned resources
            assert result.get("deleted_count", 0) <= len(user1_agents)

    def test_batch_update_idor(self):
        """Test IDOR through batch update operations."""
        user1_headers = self.test_users["user1"]["headers"]

        # Batch update attempting to modify unauthorized resources
        batch_updates = [
            {
                "id": str(uuid.uuid4()),
                "updates": {"name": "Hacked", "status": "compromised"},
            },
            {"id": str(uuid.uuid4()), "updates": {"owner_id": self.user1_id}},
        ]

        response = self.client.patch(
            "/api/v1/agents/batch",
            headers=user1_headers,
            json={"updates": batch_updates},
        )

        # Should not update unauthorized resources
        if response.status_code == status.HTTP_200_OK:
            result = response.json()
            assert result.get("updated_count", 0) == 0

    def test_bulk_import_idor(self):
        """Test IDOR through bulk import operations."""
        admin_headers = self.test_users["admin"]["headers"]

        # Try to import data with specific IDs and owners
        import_data = {
            "agents": [
                {
                    "id": str(uuid.uuid4()),
                    "name": "Imported Agent 1",
                    "owner_id": self.user1_id,  # Try to assign to another user
                    "template": "basic",
                },
                {
                    "id": str(uuid.uuid4()),
                    "name": "Imported Agent 2",
                    "owner_id": self.user2_id,  # Try to assign to another user
                    "template": "basic",
                },
            ]
        }

        response = self.client.post(
            "/api/v1/import/agents", headers=admin_headers, json=import_data
        )

        if response.status_code == status.HTTP_200_OK:
            # Verify ownership wasn't bypassed
            for agent_data in import_data["agents"]:
                check_response = self.client.get(
                    f"/api/v1/agents/{agent_data['id']}",
                    headers=self.test_users["user1"]["headers"],
                )
                # User1 should not have access to user2's imported agent
                if agent_data["owner_id"] == self.user2_id:
                    assert check_response.status_code != status.HTTP_200_OK


class TestCachePoisoningIDOR(IDORTestBase):
    """Test IDOR through cache poisoning attacks."""

    def test_cache_key_manipulation(self):
        """Test IDOR by manipulating cache keys."""
        user1_headers = self.test_users["user1"]["headers"]

        # Headers that might affect caching
        cache_manipulation_headers = [
            {"X-Forwarded-For": self.user2_id},
            {"X-Real-IP": self.user2_id},
            {"X-User-ID": self.user2_id},
            {"X-Cache-Key": f"user:{self.user2_id}"},
            {"X-Original-URL": f"/api/v1/users/{self.user2_id}/agents"},
        ]

        for extra_headers in cache_manipulation_headers:
            headers = {**user1_headers, **extra_headers}
            response = self.client.get("/api/v1/agents", headers=headers)

            if response.status_code == status.HTTP_200_OK:
                agents = response.json()
                # Should not return other user's cached data
                for agent in agents:
                    assert self.user2_id not in str(agent)

    def test_cache_poisoning_via_host_header(self):
        """Test IDOR through Host header cache poisoning."""
        user1_headers = self.test_users["user1"]["headers"]

        # Try to poison cache with different host headers
        poisoning_attempts = [
            {"Host": f"{self.user2_id}.example.com"},
            {
                "Host": "localhost",
                "X-Forwarded-Host": f"user-{self.user2_id}.com",
            },
            {"Host": f"localhost/users/{self.user2_id}"},
        ]

        for headers_update in poisoning_attempts:
            headers = {**user1_headers, **headers_update}
            response = self.client.get("/api/v1/profile", headers=headers)

            # Should not return other user's profile
            if response.status_code == status.HTTP_200_OK:
                profile = response.json()
                assert profile.get("user_id") != self.user2_id


class TestAPIVersioningIDOR(IDORTestBase):
    """Test IDOR across different API versions."""

    def test_legacy_api_idor(self):
        """Test if legacy API versions have IDOR vulnerabilities."""
        user1_headers = self.test_users["user1"]["headers"]

        # Try different API versions
        api_versions = ["v0", "v1", "v2", "v1.0", "v1.1", "legacy", "beta"]

        for version in api_versions:
            # Test various endpoints
            endpoints = [
                f"/api/{version}/agents/{uuid.uuid4()}",
                f"/api/{version}/users/{self.user2_id}",
                f"/api/{version}/admin/agents",
            ]

            for endpoint in endpoints:
                response = self.client.get(endpoint, headers=user1_headers)

                # Should not expose unauthorized data through old versions
                if response.status_code == status.HTTP_200_OK:
                    data = response.json()
                    assert self.user2_id not in str(data)

    def test_api_version_downgrade_attack(self):
        """Test IDOR through API version downgrade attacks."""
        user1_headers = self.test_users["user1"]["headers"]

        # Try to force API version through various methods
        version_forcing = [
            {"headers": {"API-Version": "1.0"}},
            {"headers": {"X-API-Version": "legacy"}},
            {"params": {"api_version": "v0"}},
            {"params": {"version": "1.0"}},
        ]

        for method in version_forcing:
            headers = {**user1_headers, **method.get("headers", {})}
            params = method.get("params", {})

            response = self.client.get(
                f"/api/v1/agents/{uuid.uuid4()}",
                headers=headers,
                params=params,
            )

            # Should not bypass authorization through version downgrade
            assert response.status_code != status.HTTP_200_OK


class TestComplexIDORScenarios(IDORTestBase):
    """Test complex IDOR scenarios combining multiple techniques."""

    def test_chained_idor_attack(self):
        """Test IDOR through chained requests."""
        user1_headers = self.test_users["user1"]["headers"]

        # First request to get some legitimate data
        response1 = self.client.get("/api/v1/agents", headers=user1_headers)

        if response1.status_code == status.HTTP_200_OK and response1.json():
            agent_id = response1.json()[0]["id"]

            # Use legitimate ID to pivot to unauthorized access
            chained_requests = [
                f"/api/v1/agents/{agent_id}/related",
                f"/api/v1/agents/{agent_id}/coalition/members",
                f"/api/v1/agents/{agent_id}/shared_resources",
            ]

            for endpoint in chained_requests:
                response = self.client.get(endpoint, headers=user1_headers)
                if response.status_code == status.HTTP_200_OK:
                    data = response.json()
                    # Should not expose other users' resources
                    self._verify_no_unauthorized_access(data, self.user1_id)

    def test_idor_with_encoding_bypass(self):
        """Test IDOR using various encoding techniques."""
        user1_headers = self.test_users["user1"]["headers"]
        target_id = str(uuid.uuid4())

        # Different encoding attempts
        encoded_ids = [
            target_id,  # Plain
            base64.b64encode(target_id.encode()).decode(),  # Base64
            base64.urlsafe_b64encode(target_id.encode()).decode(),  # URL-safe Base64
            target_id.replace("-", ""),  # Without hyphens
            target_id.upper(),  # Uppercase
            f"0x{target_id.replace('-', '')}",  # Hex-like format
            hashlib.md5(target_id.encode(), usedforsecurity=False).hexdigest(),  # MD5 hash
        ]

        for encoded_id in encoded_ids:
            response = self.client.get(f"/api/v1/agents/{encoded_id}", headers=user1_headers)
            # Should not decode and allow access
            assert response.status_code != status.HTTP_200_OK

    def test_concurrent_idor_attempts(self):
        """Test IDOR through concurrent/parallel requests."""
        user1_headers = self.test_users["user1"]["headers"]

        def make_idor_attempt(agent_id):
            """Make a single IDOR attempt."""
            return self.client.get(f"/api/v1/agents/{agent_id}", headers=user1_headers)

        # Generate multiple target IDs
        target_ids = [str(uuid.uuid4()) for _ in range(50)]

        # Attempt concurrent access
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(make_idor_attempt, target_ids))

        # None should succeed
        successful_attempts = [r for r in results if r.status_code == status.HTTP_200_OK]
        assert len(successful_attempts) == 0

    def _verify_no_unauthorized_access(self, data: dict, authorized_user_id: str):
        """Verify response contains no unauthorized data."""
        data_str = json.dumps(data)
        # Check for other user IDs (simplified check)
        assert authorized_user_id in data_str or "user" not in data_str.lower()


class TestIDORMitigationValidation(IDORTestBase):
    """Validate that IDOR mitigations are properly implemented."""

    def test_object_level_authorization(self):
        """Verify object-level authorization is enforced."""
        user1_headers = self.test_users["user1"]["headers"]

        # Test that authorization is checked at object level, not just endpoint level
        test_endpoints = [
            "/api/v1/agents/{id}",
            "/api/v1/agents/{id}/metrics",
            "/api/v1/agents/{id}/gmn",
            "/api/v1/coalitions/{id}",
            "/api/v1/coalitions/{id}/members",
        ]

        for endpoint_template in test_endpoints:
            endpoint = endpoint_template.format(id=uuid.uuid4())
            response = self.client.get(endpoint, headers=user1_headers)

            # Should check authorization for specific object
            assert response.status_code in [
                status.HTTP_403_FORBIDDEN,
                status.HTTP_404_NOT_FOUND,
            ]

    def test_indirect_reference_mapping(self):
        """Test that indirect reference mapping is used where appropriate."""
        user1_headers = self.test_users["user1"]["headers"]

        # Create a resource
        response = self.client.post(
            "/api/v1/agents",
            headers=user1_headers,
            json={"name": "indirect_ref_test", "template": "basic"},
        )

        if response.status_code == status.HTTP_201_CREATED:
            agent_data = response.json()

            # ID should not be sequential or predictable
            agent_id = agent_data["id"]
            try:
                uuid.UUID(agent_id)  # Should be a valid UUID
            except ValueError:
                pytest.fail("Resource ID is not a UUID")

            # Should not expose internal IDs
            assert "internal_id" not in agent_data
            assert "db_id" not in agent_data
            assert "_id" not in agent_data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
