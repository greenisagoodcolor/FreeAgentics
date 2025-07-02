"""
Comprehensive tests for Coalition Formation Monitoring Integration.

Tests the monitoring system for coalition formation events,
ensuring proper event handling and PyMDP/GNN alignment.
"""

from datetime import UTC, datetime, timezone
from unittest.mock import Mock

import pytest

from coalitions.formation.coalition_formation_algorithms import FormationStrategy
from coalitions.formation.monitoring_integration import (
    CoalitionFormationMonitor,
    CoalitionMonitoringEvent,
    create_coalition_monitoring_system,
)


class TestCoalitionMonitoringEvent:
    """Test CoalitionMonitoringEvent dataclass functionality."""

    def test_event_creation_minimal(self):
        """Test creating event with minimal parameters."""
        event = CoalitionMonitoringEvent(event_type="coalition_formed")

        assert event.event_type == "coalition_formed"
        assert event.coalition_id is None
        assert isinstance(event.timestamp, datetime)
        assert event.strategy_used is None
        assert event.participants == []
        assert event.business_value is None
        assert event.metadata == {}

    def test_event_creation_full(self):
        """Test creating event with all parameters."""
        timestamp = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        participants = ["agent1", "agent2", "agent3"]
        business_value = {"revenue": 1000.0, "cost": 200.0}
        metadata = {"priority": "high", "region": "west"}

        event = CoalitionMonitoringEvent(
            event_type="coalition_proposed",
            coalition_id="coalition_123",
            timestamp=timestamp,
            strategy_used=FormationStrategy.ACTIVE_INFERENCE,
            participants=participants,
            business_value=business_value,
            metadata=metadata,
        )

        assert event.event_type == "coalition_proposed"
        assert event.coalition_id == "coalition_123"
        assert event.timestamp == timestamp
        assert event.strategy_used == FormationStrategy.ACTIVE_INFERENCE
        assert event.participants == participants
        assert event.business_value == business_value
        assert event.metadata == metadata

    def test_event_timestamp_default(self):
        """Test that timestamp defaults to current UTC time."""
        # The default factory uses utcnow() which returns offset-naive datetime
        before = datetime.utcnow()
        event = CoalitionMonitoringEvent(event_type="test")
        after = datetime.utcnow()

        assert before <= event.timestamp <= after

    def test_event_mutable_fields(self):
        """Test that mutable fields can be modified."""
        event = CoalitionMonitoringEvent(event_type="test")

        # Modify participants list
        event.participants.append("agent1")
        event.participants.append("agent2")
        assert len(event.participants) == 2

        # Modify metadata dict
        event.metadata["key1"] = "value1"
        event.metadata["key2"] = "value2"
        assert len(event.metadata) == 2

    def test_event_types(self):
        """Test various event types."""
        event_types = [
            "coalition_formed",
            "coalition_proposed",
            "coalition_dissolved",
            "coalition_updated",
            "formation_started",
            "formation_completed",
            "formation_failed",
            "member_joined",
            "member_left",
            "value_updated",
        ]

        for event_type in event_types:
            event = CoalitionMonitoringEvent(event_type=event_type)
            assert event.event_type == event_type


class TestCoalitionFormationMonitor:
    """Test CoalitionFormationMonitor functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        self.monitor = CoalitionFormationMonitor(
            formation_engine=self.mock_engine)

    def test_initialization_with_engine(self):
        """Test monitor initialization with provided engine."""
        assert self.monitor.formation_engine == self.mock_engine
        assert self.monitor.event_handlers == []
        assert self.monitor.active_formations == {}

    def test_initialization_without_engine(self):
        """Test monitor initialization creates default engine."""
        monitor = CoalitionFormationMonitor()

        assert monitor.formation_engine is not None
        assert monitor.event_handlers == []
        assert monitor.active_formations == {}

    def test_register_event_handler(self):
        """Test registering event handlers."""
        handler1 = Mock()
        handler2 = Mock()
        handler3 = Mock()

        self.monitor.register_event_handler(handler1)
        self.monitor.register_event_handler(handler2)
        self.monitor.register_event_handler(handler3)

        assert len(self.monitor.event_handlers) == 3
        assert handler1 in self.monitor.event_handlers
        assert handler2 in self.monitor.event_handlers
        assert handler3 in self.monitor.event_handlers

    def test_emit_event_single_handler(self):
        """Test emitting event to single handler."""
        handler = Mock()
        self.monitor.register_event_handler(handler)

        event = CoalitionMonitoringEvent(
            event_type="test_event", coalition_id="test_123")

        self.monitor._emit_event(event)

        handler.assert_called_once_with(event)

    def test_emit_event_multiple_handlers(self):
        """Test emitting event to multiple handlers."""
        handler1 = Mock()
        handler2 = Mock()
        handler3 = Mock()

        self.monitor.register_event_handler(handler1)
        self.monitor.register_event_handler(handler2)
        self.monitor.register_event_handler(handler3)

        event = CoalitionMonitoringEvent(event_type="test_event")

        self.monitor._emit_event(event)

        handler1.assert_called_once_with(event)
        handler2.assert_called_once_with(event)
        handler3.assert_called_once_with(event)

    def test_emit_event_handler_error(self):
        """Test that errors in handlers don't stop other handlers."""
        handler1 = Mock()
        handler2 = Mock(side_effect=RuntimeError("Handler error"))
        handler3 = Mock()

        self.monitor.register_event_handler(handler1)
        self.monitor.register_event_handler(handler2)
        self.monitor.register_event_handler(handler3)

        event = CoalitionMonitoringEvent(event_type="test_event")

        # Should not raise exception
        self.monitor._emit_event(event)

        # All handlers should be called despite error
        handler1.assert_called_once_with(event)
        handler2.assert_called_once_with(event)
        handler3.assert_called_once_with(event)

    def test_emit_event_no_handlers(self):
        """Test emitting event with no registered handlers."""
        event = CoalitionMonitoringEvent(event_type="test_event")

        # Should not raise exception
        self.monitor._emit_event(event)

        # Just verify no crash
        assert len(self.monitor.event_handlers) == 0

    def test_active_formations_tracking(self):
        """Test tracking active coalition formations."""
        # Add some active formations
        self.monitor.active_formations["coalition_1"] = {
            "status": "forming",
            "participants": ["agent1", "agent2"],
            "start_time": datetime.now(UTC),
        }

        self.monitor.active_formations["coalition_2"] = {
            "status": "negotiating",
            "participants": ["agent3", "agent4", "agent5"],
            "start_time": datetime.now(UTC),
        }

        assert len(self.monitor.active_formations) == 2
        assert "coalition_1" in self.monitor.active_formations
        assert "coalition_2" in self.monitor.active_formations

        # Remove completed formation
        del self.monitor.active_formations["coalition_1"]
        assert len(self.monitor.active_formations) == 1


class TestCoalitionMonitoringWorkflow:
    """Test complete monitoring workflows."""

    def test_coalition_formation_workflow(self):
        """Test monitoring a complete coalition formation."""
        monitor = CoalitionFormationMonitor()
        events_received = []

        def event_handler(event):
            events_received.append(event)

        monitor.register_event_handler(event_handler)

        # Simulate formation workflow
        # 1. Formation started
        start_event = CoalitionMonitoringEvent(
            event_type="formation_started",
            coalition_id="coalition_test",
            participants=["agent1", "agent2", "agent3"],
            strategy_used=FormationStrategy.PREFERENCE_MATCHING,
            metadata={"initiator": "agent1"},
        )
        monitor._emit_event(start_event)

        # Track active formation
        monitor.active_formations["coalition_test"] = {
            "status": "forming",
            "start_time": start_event.timestamp,
        }

        # 2. Coalition proposed
        propose_event = CoalitionMonitoringEvent(
            event_type="coalition_proposed",
            coalition_id="coalition_test",
            participants=["agent1", "agent2", "agent3"],
            business_value={"total_value": 1500.0},
        )
        monitor._emit_event(propose_event)

        # 3. Coalition formed
        form_event = CoalitionMonitoringEvent(
            event_type="coalition_formed",
            coalition_id="coalition_test",
            participants=["agent1", "agent2", "agent3"],
            business_value={"total_value": 1500.0, "per_agent": 500.0},
        )
        monitor._emit_event(form_event)

        # Remove from active
        del monitor.active_formations["coalition_test"]

        # Verify events
        assert len(events_received) == 3
        assert events_received[0].event_type == "formation_started"
        assert events_received[1].event_type == "coalition_proposed"
        assert events_received[2].event_type == "coalition_formed"

        # Verify coalition IDs are consistent
        assert all(e.coalition_id == "coalition_test" for e in events_received)

        # Verify no active formations remain
        assert len(monitor.active_formations) == 0

    def test_multiple_handlers_different_purposes(self):
        """Test multiple handlers for different monitoring purposes."""
        monitor = CoalitionFormationMonitor()

        # Handler 1: Log all events
        all_events = []

        def log_handler(event):
            all_events.append(event)

        # Handler 2: Track only formations
        formations = []

        def formation_handler(event):
            if event.event_type == "coalition_formed":
                formations.append(event)

        # Handler 3: Calculate business value
        total_value = {"sum": 0.0}

        def value_handler(event):
            if event.business_value and "total_value" in event.business_value:
                total_value["sum"] += event.business_value["total_value"]

        monitor.register_event_handler(log_handler)
        monitor.register_event_handler(formation_handler)
        monitor.register_event_handler(value_handler)

        # Emit various events
        events = [
            CoalitionMonitoringEvent(event_type="formation_started", coalition_id="c1"),
            CoalitionMonitoringEvent(
                event_type="coalition_formed",
                coalition_id="c1",
                business_value={"total_value": 1000.0},
            ),
            CoalitionMonitoringEvent(event_type="member_joined", coalition_id="c1"),
            CoalitionMonitoringEvent(
                event_type="coalition_formed",
                coalition_id="c2",
                business_value={"total_value": 2000.0},
            ),
        ]

        for event in events:
            monitor._emit_event(event)

        # Verify handlers processed correctly
        assert len(all_events) == 4
        assert len(formations) == 2
        assert total_value["sum"] == 3000.0


class TestFactoryFunction:
    """Test the factory function."""

    def test_create_coalition_monitoring_system(self):
        """Test creating monitoring system via factory."""
        monitor = create_coalition_monitoring_system()

        assert isinstance(monitor, CoalitionFormationMonitor)
        assert monitor.formation_engine is not None
        assert monitor.event_handlers == []
        assert monitor.active_formations == {}

    def test_factory_creates_independent_instances(self):
        """Test that factory creates independent instances."""
        monitor1 = create_coalition_monitoring_system()
        monitor2 = create_coalition_monitoring_system()

        # Add handler to monitor1
        handler = Mock()
        monitor1.register_event_handler(handler)

        # Verify independence
        assert len(monitor1.event_handlers) == 1
        assert len(monitor2.event_handlers) == 0
        assert monitor1 is not monitor2
        assert monitor1.formation_engine is not monitor2.formation_engine


class TestMonitoringIntegration:
    """Test integration with PyMDP and GNN concepts."""

    def test_pymdp_aligned_monitoring(self):
        """Test monitoring events aligned with PyMDP concepts."""
        monitor = CoalitionFormationMonitor()
        events = []

        def capture_events(event):
            events.append(event)

        monitor.register_event_handler(capture_events)

        # Create event with PyMDP-aligned metadata
        event = CoalitionMonitoringEvent(
            event_type="coalition_formed",
            coalition_id="pymdp_coalition",
            participants=["belief_updater_1", "policy_selector_1"],
            metadata={
                "formation_strategy": "expected_free_energy_minimization",
                "belief_convergence": 0.95,
                "policy_alignment": 0.88,
                "collective_free_energy": -12.5,
            },
        )

        monitor._emit_event(event)

        assert len(events) == 1
        assert events[0].metadata["formation_strategy"] == "expected_free_energy_minimization"
        assert events[0].metadata["belief_convergence"] == 0.95

    def test_gnn_notation_monitoring(self):
        """Test monitoring with GNN (Generalized Notation Notation) metadata."""
        monitor = CoalitionFormationMonitor()
        events = []

        def capture_events(event):
            events.append(event)

        monitor.register_event_handler(capture_events)

        # Create event with GNN notation metadata
        event = CoalitionMonitoringEvent(
            event_type="formation_completed",
            coalition_id="gnn_coalition",
            participants=["notation_expert_1", "notation_expert_2"],
            metadata={
                "notation_system": "GNN",
                "notation_version": "1.0",
                "formalized_rules": [
                    "belief_update_notation",
                    "policy_selection_notation",
                    "free_energy_notation",
                ],
                "compatibility": "pymdp_v0.5",
            },
        )

        monitor._emit_event(event)

        assert len(events) == 1
        assert events[0].metadata["notation_system"] == "GNN"
        assert len(events[0].metadata["formalized_rules"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
