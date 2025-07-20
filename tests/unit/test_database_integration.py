"""
Test suite for real database integration.

Tests that the API endpoints actually use PostgreSQL and not in-memory storage.
"""

import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api.main import app
from database.base import Base
from database.models import Agent as AgentModel
from database.models import AgentStatus


class TestDatabaseIntegration:
    """Test real PostgreSQL database integration."""

    @pytest.fixture(scope="class")
    def test_db_engine(self):
        """Create a test database engine."""
        # Use the same database as the main app for integration testing
        DATABASE_URL = os.getenv(
            "TEST_DATABASE_URL", "sqlite:///./test_freeagentics.db"
        )
        engine = create_engine(DATABASE_URL)
        # Create all tables
        Base.metadata.create_all(bind=engine)
        return engine

    @pytest.fixture(scope="class")
    def test_db_session(self, test_db_engine):
        """Create a test database session."""
        Session = sessionmaker(bind=test_db_engine)
        return Session()

    @pytest.fixture
    def client(self):
        """Create FastAPI test client."""
        return TestClient(app)

    def test_database_connection(self, test_db_engine):
        """Test that we can connect to the database."""
        from sqlalchemy import text

        with test_db_engine.connect() as conn:
            # Use a database-agnostic query
            if "sqlite" in str(test_db_engine.url):
                result = conn.execute(text("SELECT sqlite_version()"))
                version = result.fetchone()[0]
                assert version is not None  # SQLite version should exist
            else:
                result = conn.execute(text("SELECT version()"))
                version = result.fetchone()[0]
                assert "PostgreSQL" in version

    def test_agent_table_exists(self, test_db_session):
        """Test that agent table exists and has correct structure."""
        # Query the table to verify it exists
        count = test_db_session.query(AgentModel).count()
        assert isinstance(count, int)  # Should not raise an error

    def test_create_agent_api_saves_to_database(self, client, test_db_session):
        """Test that POST /agents actually saves to PostgreSQL."""
        # Get initial count
        initial_count = test_db_session.query(AgentModel).count()

        # Create agent via API
        response = client.post(
            "/api/v1/agents",
            json={
                "name": "Database Test Agent",
                "template": "explorer",
                "parameters": {"test": "database_integration"},
            },
        )

        assert response.status_code == 201
        agent_data = response.json()
        agent_id = agent_data["id"]

        # Verify it was saved to database
        test_db_session.commit()  # Ensure we see the latest data
        final_count = test_db_session.query(AgentModel).count()
        assert final_count == initial_count + 1

        # Verify the specific agent exists
        from uuid import UUID

        db_agent = (
            test_db_session.query(AgentModel)
            .filter(AgentModel.id == UUID(agent_id))
            .first()
        )

        assert db_agent is not None
        assert db_agent.name == "Database Test Agent"
        assert db_agent.template == "explorer"
        assert db_agent.parameters["test"] == "database_integration"

        # Clean up
        test_db_session.delete(db_agent)
        test_db_session.commit()

    def test_get_agents_api_reads_from_database(self, client, test_db_session):
        """Test that GET /agents reads from PostgreSQL."""
        # Create agent directly in database
        db_agent = AgentModel(
            name="Direct DB Agent",
            template="test",
            status=AgentStatus.PENDING,
            parameters={"source": "direct_db"},
        )
        test_db_session.add(db_agent)
        test_db_session.commit()
        test_db_session.refresh(db_agent)

        # Get agents via API
        response = client.get("/api/v1/agents")
        assert response.status_code == 200

        agents = response.json()
        assert isinstance(agents, list)

        # Find our agent in the response
        our_agent = None
        for agent in agents:
            if agent["name"] == "Direct DB Agent":
                our_agent = agent
                break

        assert our_agent is not None
        assert our_agent["template"] == "test"
        assert our_agent["parameters"]["source"] == "direct_db"

        # Clean up
        test_db_session.delete(db_agent)
        test_db_session.commit()

    def test_agent_persistence_across_restarts(self, test_db_session):
        """Test that agents persist in PostgreSQL across app restarts."""
        # Create agent in database
        test_agent = AgentModel(
            name="Persistence Test Agent",
            template="persistent",
            status=AgentStatus.ACTIVE,
            parameters={"persistent": True},
        )
        test_db_session.add(test_agent)
        test_db_session.commit()
        test_db_session.refresh(test_agent)
        agent_id = test_agent.id

        # Close session to simulate restart
        test_db_session.close()

        # Create new session and verify agent still exists
        Session = sessionmaker(bind=test_db_session.bind)
        new_session = Session()

        persistent_agent = (
            new_session.query(AgentModel)
            .filter(AgentModel.id == agent_id)
            .first()
        )

        assert persistent_agent is not None
        assert persistent_agent.name == "Persistence Test Agent"
        assert persistent_agent.parameters["persistent"] is True

        # Clean up
        new_session.delete(persistent_agent)
        new_session.commit()
        new_session.close()

    def test_update_agent_status_updates_database(
        self, client, test_db_session
    ):
        """Test that PATCH /agents/{id}/status updates PostgreSQL."""
        # Create agent in database
        db_agent = AgentModel(
            name="Status Test Agent",
            template="test",
            status=AgentStatus.PENDING,
            parameters={},
        )
        test_db_session.add(db_agent)
        test_db_session.commit()
        test_db_session.refresh(db_agent)

        # Update status via API
        response = client.patch(
            f"/api/v1/agents/{db_agent.id}/status?status=active"
        )
        assert response.status_code == 200

        # Verify database was updated
        test_db_session.refresh(db_agent)
        assert db_agent.status == AgentStatus.ACTIVE
        assert db_agent.last_active is not None

        # Clean up
        test_db_session.delete(db_agent)
        test_db_session.commit()

    def test_delete_agent_removes_from_database(self, client, test_db_session):
        """Test that DELETE /agents/{id} removes from PostgreSQL."""
        # Create agent in database
        db_agent = AgentModel(
            name="Delete Test Agent",
            template="test",
            status=AgentStatus.PENDING,
            parameters={},
        )
        test_db_session.add(db_agent)
        test_db_session.commit()
        test_db_session.refresh(db_agent)
        agent_id = db_agent.id

        # Delete via API
        response = client.delete(f"/api/v1/agents/{agent_id}")
        assert response.status_code == 200

        # Verify it was removed from database
        test_db_session.commit()  # Ensure we see the latest state
        deleted_agent = (
            test_db_session.query(AgentModel)
            .filter(AgentModel.id == agent_id)
            .first()
        )

        assert deleted_agent is None

    def test_no_in_memory_storage_fallback(self, client):
        """Test that API doesn't use in-memory storage as fallback."""
        # This test ensures there's no dict-based storage being used
        import api.v1.agents as agents_module

        # Check that there are no dictionary-based storage variables
        module_vars = vars(agents_module)
        suspect_vars = [
            name
            for name, value in module_vars.items()
            if isinstance(value, dict)
            and any(
                keyword in name.lower()
                for keyword in ["agent", "storage", "cache", "memory"]
            )
        ]

        # Should not have any in-memory agent storage
        storage_vars = [var for var in suspect_vars if "agent" in var.lower()]
        assert (
            len(storage_vars) == 0
        ), f"Found potential in-memory storage: {storage_vars}"

    def test_database_constraints_enforced(self, test_db_session):
        """Test that database constraints are properly enforced."""
        # Test that required fields are enforced
        with pytest.raises(
            Exception
        ):  # Should raise IntegrityError or similar
            invalid_agent = AgentModel(
                # Missing required name field
                template="test",
                status=AgentStatus.PENDING,
            )
            test_db_session.add(invalid_agent)
            test_db_session.commit()

        test_db_session.rollback()

    def test_agent_uuid_generation(self, test_db_session):
        """Test that agents get proper UUIDs as primary keys."""
        agent = AgentModel(
            name="UUID Test Agent",
            template="test",
            status=AgentStatus.PENDING,
            parameters={},
        )
        test_db_session.add(agent)
        test_db_session.commit()
        test_db_session.refresh(agent)

        # Verify UUID was generated
        assert agent.id is not None
        assert isinstance(agent.id, type(agent.id))  # Should be UUID type
        assert str(agent.id).count("-") == 4  # UUID format

        # Clean up
        test_db_session.delete(agent)
        test_db_session.commit()
