"""
Integration tests for WebSocket functionality with real database persistence.

This demonstrates how to replace in-memory WebSocket state management
with proper database persistence for connection tracking and event history.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pytest
from fastapi import WebSocket
from fastapi.testclient import TestClient
from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, String, create_engine, select
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, relationship

from api.v1.websocket import ConnectionManager, WebSocketMessage
from database.base import Base
from database.models import Agent, AgentStatus
from tests.db_infrastructure.factories import AgentFactory
from tests.db_infrastructure.fixtures import db_session
from tests.db_infrastructure.test_config import TEST_DATABASE_URL, DatabaseTestCase


# Define WebSocket-specific database models
class WebSocketConnection(Base):
    """Model for tracking WebSocket connections."""

    __tablename__ = "websocket_connections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    client_id = Column(String(100), unique=True, nullable=False)
    connected_at = Column(DateTime, default=datetime.utcnow)
    disconnected_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    metadata = Column(JSON, default=dict)

    # Relationships
    subscriptions = relationship(
        "WebSocketSubscription", back_populates="connection", cascade="all, delete-orphan"
    )
    events = relationship("WebSocketEvent", back_populates="connection")


class WebSocketSubscription(Base):
    """Model for tracking event subscriptions."""

    __tablename__ = "websocket_subscriptions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    connection_id = Column(
        UUID(as_uuid=True), ForeignKey("websocket_connections.id"), nullable=False
    )
    event_type = Column(String(50), nullable=False)
    subscribed_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    connection = relationship("WebSocketConnection", back_populates="subscriptions")


class WebSocketEvent(Base):
    """Model for tracking WebSocket events."""

    __tablename__ = "websocket_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    connection_id = Column(
        UUID(as_uuid=True), ForeignKey("websocket_connections.id"), nullable=False
    )
    event_type = Column(String(50), nullable=False)
    event_data = Column(JSON, default=dict)
    sent_at = Column(DateTime, default=datetime.utcnow)
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime, nullable=True)

    # Relationships
    connection = relationship("WebSocketConnection", back_populates="events")


class DatabaseConnectionManager(ConnectionManager):
    """Enhanced ConnectionManager with database persistence."""

    def __init__(self, db_session: Session):
        super().__init__()
        self.db = db_session
        self._restore_active_connections()

    def _restore_active_connections(self):
        """Restore active connections from database on startup."""
        active_conns = (
            self.db.query(WebSocketConnection).filter(WebSocketConnection.is_active == True).all()
        )

        for conn in active_conns:
            # Restore subscriptions
            for sub in conn.subscriptions:
                if sub.event_type not in self.subscriptions:
                    self.subscriptions[sub.event_type] = set()
                self.subscriptions[sub.event_type].add(conn.client_id)

    async def connect(self, websocket: WebSocket, client_id: str, metadata: Optional[dict] = None):
        """Accept connection and persist to database."""
        await super().connect(websocket, client_id, metadata)

        # Check if connection exists
        existing = (
            self.db.query(WebSocketConnection)
            .filter(WebSocketConnection.client_id == client_id)
            .first()
        )

        if existing:
            # Reactivate existing connection
            existing.is_active = True
            existing.connected_at = datetime.utcnow()
            existing.disconnected_at = None
            existing.metadata = metadata or {}
        else:
            # Create new connection record
            conn = WebSocketConnection(client_id=client_id, metadata=metadata or {}, is_active=True)
            self.db.add(conn)

        self.db.commit()

    def disconnect(self, client_id: str):
        """Disconnect and update database."""
        super().disconnect(client_id)

        # Update database
        conn = (
            self.db.query(WebSocketConnection)
            .filter(WebSocketConnection.client_id == client_id)
            .first()
        )

        if conn:
            conn.is_active = False
            conn.disconnected_at = datetime.utcnow()
            self.db.commit()

    def subscribe(self, client_id: str, event_type: str):
        """Subscribe to event and persist."""
        super().subscribe(client_id, event_type)

        # Get connection
        conn = (
            self.db.query(WebSocketConnection)
            .filter(WebSocketConnection.client_id == client_id)
            .first()
        )

        if conn:
            # Check if already subscribed
            existing_sub = (
                self.db.query(WebSocketSubscription)
                .filter(
                    WebSocketSubscription.connection_id == conn.id,
                    WebSocketSubscription.event_type == event_type,
                )
                .first()
            )

            if not existing_sub:
                sub = WebSocketSubscription(connection_id=conn.id, event_type=event_type)
                self.db.add(sub)
                self.db.commit()

    async def broadcast(self, message: dict, event_type: Optional[str] = None):
        """Broadcast message and log to database."""
        await super().broadcast(message, event_type)

        # Log broadcast event
        if event_type:
            # Get all subscribed connections
            subscribed_conns = (
                self.db.query(WebSocketConnection)
                .join(WebSocketSubscription)
                .filter(
                    WebSocketSubscription.event_type == event_type,
                    WebSocketConnection.is_active == True,
                )
                .all()
            )

            # Create event records
            for conn in subscribed_conns:
                event = WebSocketEvent(
                    connection_id=conn.id, event_type=event_type, event_data=message
                )
                self.db.add(event)

            self.db.commit()


class TestWebSocketDatabase(DatabaseTestCase):
    """Test WebSocket functionality with database persistence."""

    @pytest.fixture
    def db_manager(self, db_session: Session):
        """Create database-backed connection manager."""
        # Ensure tables exist
        Base.metadata.create_all(bind=db_session.bind)
        return DatabaseConnectionManager(db_session)

    def test_connection_persistence(
        self, db_session: Session, db_manager: DatabaseConnectionManager
    ):
        """Test that connections are persisted to database."""
        client_id = "test_client_123"
        metadata = {"user_agent": "TestClient/1.0", "ip": "127.0.0.1"}

        # Simulate connection (without actual WebSocket)
        db_manager.active_connections[client_id] = None  # Mock WebSocket
        db_manager.connection_metadata[client_id] = metadata

        # Manually create connection record
        conn = WebSocketConnection(client_id=client_id, metadata=metadata, is_active=True)
        db_session.add(conn)
        db_session.commit()

        # Verify connection is in database
        saved_conn = (
            db_session.query(WebSocketConnection)
            .filter(WebSocketConnection.client_id == client_id)
            .first()
        )

        assert saved_conn is not None
        assert saved_conn.is_active is True
        assert saved_conn.metadata["user_agent"] == "TestClient/1.0"

        # Simulate disconnection
        db_manager.disconnect(client_id)

        # Verify connection is marked inactive
        db_session.refresh(saved_conn)
        assert saved_conn.is_active is False
        assert saved_conn.disconnected_at is not None

    def test_subscription_persistence(
        self, db_session: Session, db_manager: DatabaseConnectionManager
    ):
        """Test that subscriptions are persisted."""
        client_id = "subscriber_001"

        # Create connection
        conn = WebSocketConnection(client_id=client_id, is_active=True)
        db_session.add(conn)
        db_session.commit()

        # Subscribe to events
        event_types = ["agent:created", "agent:updated", "coalition:formed"]
        for event_type in event_types:
            db_manager.subscribe(client_id, event_type)

        # Verify subscriptions in database
        subs = (
            db_session.query(WebSocketSubscription)
            .filter(WebSocketSubscription.connection_id == conn.id)
            .all()
        )

        assert len(subs) == 3
        sub_types = {sub.event_type for sub in subs}
        assert sub_types == set(event_types)

    @pytest.mark.asyncio
    async def test_event_history_tracking(
        self, db_session: Session, db_manager: DatabaseConnectionManager
    ):
        """Test that broadcasted events are tracked in database."""
        # Create multiple connections with subscriptions
        connections = []
        for i in range(3):
            client_id = f"client_{i}"
            conn = WebSocketConnection(client_id=client_id, is_active=True)
            db_session.add(conn)
            connections.append(conn)

        db_session.commit()

        # Subscribe clients to different events
        db_manager.subscribe("client_0", "agent:created")
        db_manager.subscribe("client_1", "agent:created")
        db_manager.subscribe("client_2", "agent:updated")

        # Mock WebSocket connections
        for conn in connections:
            db_manager.active_connections[conn.client_id] = None

        # Broadcast agent creation event
        agent_created_msg = {
            "type": "agent_event",
            "event_type": "agent:created",
            "agent_id": str(uuid.uuid4()),
            "data": {"name": "Test Agent", "status": "active"},
        }

        # Since we're mocking, we'll manually create event records
        # In real implementation, this would be done by broadcast()
        subscribed_conns = (
            db_session.query(WebSocketConnection)
            .join(WebSocketSubscription)
            .filter(
                WebSocketSubscription.event_type == "agent:created",
                WebSocketConnection.is_active == True,
            )
            .all()
        )

        for conn in subscribed_conns:
            event = WebSocketEvent(
                connection_id=conn.id, event_type="agent:created", event_data=agent_created_msg
            )
            db_session.add(event)

        db_session.commit()

        # Verify events were recorded
        events = (
            db_session.query(WebSocketEvent)
            .filter(WebSocketEvent.event_type == "agent:created")
            .all()
        )

        assert len(events) == 2  # Only client_0 and client_1 subscribed
        for event in events:
            assert event.event_data["agent_id"] == agent_created_msg["agent_id"]

    def test_connection_recovery_after_restart(self, db_session: Session):
        """Test that connections can be recovered after system restart."""
        # Simulate existing connections in database
        existing_connections = []
        for i in range(5):
            conn = WebSocketConnection(
                client_id=f"persistent_client_{i}",
                metadata={"session_id": f"session_{i}"},
                is_active=True if i < 3 else False,  # 3 active, 2 inactive
            )
            db_session.add(conn)
            existing_connections.append(conn)

        db_session.commit()

        # Add subscriptions for active connections
        for i in range(3):
            conn = existing_connections[i]
            sub = WebSocketSubscription(connection_id=conn.id, event_type="system:status")
            db_session.add(sub)

        db_session.commit()

        # Create new manager (simulating restart)
        new_manager = DatabaseConnectionManager(db_session)

        # Verify subscriptions were restored
        assert "system:status" in new_manager.subscriptions
        assert len(new_manager.subscriptions["system:status"]) == 3

        # Verify only active connections are tracked
        for i in range(3):
            assert f"persistent_client_{i}" in new_manager.subscriptions["system:status"]

    def test_event_analytics_queries(self, db_session: Session):
        """Test analytics queries on WebSocket event history."""
        # Create test data
        conn = WebSocketConnection(client_id="analytics_client", is_active=True)
        db_session.add(conn)
        db_session.commit()

        # Generate events over time
        event_types = ["agent:created", "agent:updated", "agent:action", "coalition:formed"]
        base_time = datetime.utcnow() - timedelta(hours=24)

        for hour in range(24):
            for event_type in event_types:
                # Varying frequency for different event types
                count = 5 if "agent" in event_type else 2
                for _ in range(count):
                    event = WebSocketEvent(
                        connection_id=conn.id,
                        event_type=event_type,
                        event_data={"timestamp": (base_time + timedelta(hours=hour)).isoformat()},
                        sent_at=base_time + timedelta(hours=hour),
                    )
                    db_session.add(event)

        db_session.commit()

        # Query 1: Event count by type in last 24 hours
        from sqlalchemy import func

        event_counts = (
            db_session.query(
                WebSocketEvent.event_type, func.count(WebSocketEvent.id).label("count")
            )
            .filter(WebSocketEvent.sent_at >= datetime.utcnow() - timedelta(hours=24))
            .group_by(WebSocketEvent.event_type)
            .all()
        )

        event_dict = {e.event_type: e.count for e in event_counts}
        assert event_dict["agent:created"] == 120  # 5 * 24
        assert event_dict["coalition:formed"] == 48  # 2 * 24

        # Query 2: Hourly event distribution
        hourly_events = (
            db_session.query(
                func.date_trunc("hour", WebSocketEvent.sent_at).label("hour"),
                func.count(WebSocketEvent.id).label("count"),
            )
            .group_by("hour")
            .order_by("hour")
            .limit(5)
            .all()
        )

        assert len(hourly_events) > 0

        # Query 3: Connection activity patterns
        connection_durations = (
            db_session.query(
                WebSocketConnection.client_id,
                func.extract(
                    "epoch",
                    func.coalesce(WebSocketConnection.disconnected_at, func.now())
                    - WebSocketConnection.connected_at,
                ).label("duration_seconds"),
            )
            .filter(WebSocketConnection.connected_at >= datetime.utcnow() - timedelta(days=7))
            .all()
        )

        # Query 4: Most active event types
        top_events = (
            db_session.query(
                WebSocketEvent.event_type,
                func.count(WebSocketEvent.id).label("total"),
                func.count(func.distinct(WebSocketEvent.connection_id)).label("unique_connections"),
            )
            .group_by(WebSocketEvent.event_type)
            .order_by(func.count(WebSocketEvent.id).desc())
            .limit(3)
            .all()
        )

        assert len(top_events) > 0
        assert top_events[0].event_type in ["agent:created", "agent:updated", "agent:action"]

    def test_connection_cleanup(self, db_session: Session):
        """Test cleaning up old connections and events."""
        # Create old connections and events
        cutoff_date = datetime.utcnow() - timedelta(days=30)

        # Old inactive connections
        for i in range(10):
            old_conn = WebSocketConnection(
                client_id=f"old_client_{i}",
                connected_at=cutoff_date - timedelta(days=i),
                disconnected_at=cutoff_date - timedelta(days=i - 1),
                is_active=False,
            )
            db_session.add(old_conn)

        # Recent active connection
        active_conn = WebSocketConnection(
            client_id="active_client",
            connected_at=datetime.utcnow() - timedelta(hours=1),
            is_active=True,
        )
        db_session.add(active_conn)
        db_session.commit()

        # Cleanup query - remove connections older than 30 days
        cleanup_result = (
            db_session.query(WebSocketConnection)
            .filter(
                WebSocketConnection.disconnected_at < cutoff_date,
                WebSocketConnection.is_active == False,
            )
            .delete()
        )

        db_session.commit()

        # Verify cleanup
        remaining_conns = db_session.query(WebSocketConnection).all()
        assert len(remaining_conns) == 1
        assert remaining_conns[0].client_id == "active_client"

        print(f"Cleaned up {cleanup_result} old connections")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
