"""
Test suite for Agents API endpoints with real database integration.

Tests the FastAPI agents endpoints using proper database models
and real Active Inference implementation.
"""

import os
import uuid
from datetime import datetime
from unittest.mock import Mock, patch

os.environ["REDIS_ENABLED"] = "false"
os.environ["TESTING"] = "true"
os.environ["DATABASE_URL"] = "sqlite:///./test.db"

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from api.v1.agents import router
from database.base import Base
from database.models import Agent as AgentModel
from database.models import AgentStatus
from database.session import get_db

# Import the app from the correct location
try:
    from main import app
except ImportError:
    # Fallback if main.py is not in the root
    from api.main import app

# Create test database
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=engine
)

# Create tables
Base.metadata.create_all(bind=engine)


def override_get_db():
    """Override database dependency for testing."""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


# Override dependency
app.dependency_overrides[get_db] = override_get_db


# Create a test client using compatibility wrapper
from tests.test_client_compat import TestClient

# Create global test client
client = TestClient(app)


class TestAgentsAPI:
    """Test agents API endpoints with real database."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test database before each test."""
        # Clear all data before each test
        db = TestingSessionLocal()
        db.query(AgentModel).delete()
        db.commit()
        db.close()
        yield
        # Cleanup after test
        db = TestingSessionLocal()
        db.query(AgentModel).delete()
        db.commit()
        db.close()

    def test_create_agent(self):
        """Test creating a new agent."""
        agent_data = {
            "name": "Test Explorer",
            "template": "basic-explorer",
            "parameters": {"grid_size": 10},
        }

        response = client.post("/api/v1/agents", json=agent_data)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Test Explorer"
        assert data["template"] == "basic-explorer"
        assert data["status"] == "pending"
        assert "id" in data
        assert uuid.UUID(data["id"])  # Valid UUID

    def test_list_agents_empty(self):
        """Test listing agents when none exist."""
        response = client.get("/api/v1/agents")

        assert response.status_code == 200
        data = response.json()
        assert data == []

    def test_list_agents_with_data(self):
        """Test listing agents with data."""
        # Create test agents
        db = TestingSessionLocal()
        agent1 = AgentModel(
            name="Agent 1",
            template="basic-explorer",
            status=AgentStatus.ACTIVE,
        )
        agent2 = AgentModel(
            name="Agent 2",
            template="resource-collector",
            status=AgentStatus.PENDING,
        )
        db.add(agent1)
        db.add(agent2)
        db.commit()
        db.close()

        response = client.get("/api/v1/agents")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["name"] == "Agent 1"
        assert data[1]["name"] == "Agent 2"

    def test_get_agent_by_id(self):
        """Test getting a specific agent."""
        # Create test agent
        db = TestingSessionLocal()
        agent = AgentModel(
            name="Test Agent",
            template="basic-explorer",
            status=AgentStatus.ACTIVE,
            parameters={"test": "value"},
        )
        db.add(agent)
        db.commit()
        agent_id = str(agent.id)
        db.close()

        response = client.get(f"/api/v1/agents/{agent_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == agent_id
        assert data["name"] == "Test Agent"
        assert data["parameters"] == {"test": "value"}

    def test_get_agent_not_found(self):
        """Test getting non-existent agent."""
        fake_id = str(uuid.uuid4())
        response = client.get(f"/api/v1/agents/{fake_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_update_agent_status(self):
        """Test updating agent status."""
        # Create test agent
        db = TestingSessionLocal()
        agent = AgentModel(
            name="Test Agent",
            template="basic-explorer",
            status=AgentStatus.PENDING,
        )
        db.add(agent)
        db.commit()
        agent_id = str(agent.id)
        db.close()

        response = client.patch(
            f"/api/v1/agents/{agent_id}/status", params={"status": "active"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "active"

        # Verify in database
        db = TestingSessionLocal()
        updated_agent = (
            db.query(AgentModel).filter(AgentModel.id == agent.id).first()
        )
        assert updated_agent.status == AgentStatus.ACTIVE
        assert updated_agent.last_active is not None
        db.close()

    def test_delete_agent(self):
        """Test deleting an agent."""
        # Create test agent
        db = TestingSessionLocal()
        agent = AgentModel(
            name="Test Agent",
            template="basic-explorer",
            status=AgentStatus.STOPPED,
        )
        db.add(agent)
        db.commit()
        agent_id = str(agent.id)
        db.close()

        response = client.delete(f"/api/v1/agents/{agent_id}")

        assert response.status_code == 200
        assert "deleted successfully" in response.json()["message"]

        # Verify agent is gone
        db = TestingSessionLocal()
        deleted_agent = (
            db.query(AgentModel).filter(AgentModel.id == agent.id).first()
        )
        assert deleted_agent is None
        db.close()

    def test_get_agent_metrics(self):
        """Test getting agent metrics."""
        # Create test agent with metrics
        db = TestingSessionLocal()
        agent = AgentModel(
            name="Test Agent",
            template="basic-explorer",
            status=AgentStatus.ACTIVE,
            inference_count=42,
            metrics={"avg_response_time": 123.45, "memory_usage": 67.89},
        )
        db.add(agent)
        db.commit()
        agent_id = str(agent.id)
        db.close()

        response = client.get(f"/api/v1/agents/{agent_id}/metrics")

        assert response.status_code == 200
        data = response.json()
        assert data["agent_id"] == agent_id
        assert data["total_inferences"] == 42
        assert data["avg_response_time"] == 123.45
        assert data["memory_usage"] == 67.89

    def test_filter_agents_by_status(self):
        """Test filtering agents by status."""
        # Create agents with different statuses
        db = TestingSessionLocal()
        active = AgentModel(
            name="Active", template="t1", status=AgentStatus.ACTIVE
        )
        pending = AgentModel(
            name="Pending", template="t2", status=AgentStatus.PENDING
        )
        stopped = AgentModel(
            name="Stopped", template="t3", status=AgentStatus.STOPPED
        )
        db.add_all([active, pending, stopped])
        db.commit()
        db.close()

        # Filter for active agents
        response = client.get("/api/v1/agents?status=active")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "Active"
        assert data[0]["status"] == "active"

    def test_agent_error_handling(self):
        """Test error handling in agent endpoints."""
        # Test creating agent with invalid data
        invalid_data = {
            "name": "",  # Empty name
            "template": "invalid-template",
        }

        response = client.post("/api/v1/agents", json=invalid_data)
        assert response.status_code == 422  # Validation error
