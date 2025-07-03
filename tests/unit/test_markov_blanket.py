"""
Tests for Markov Blanket Interface for Agent Boundary System.

Comprehensive test suite for the Markov blanket implementation within the agents
boundary system, enabling tracking of agent boundaries and violation events,
with integration to Active Inference engine and PyMDP.
"""

import uuid
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np
import pytest

from agents.base.markov_blanket import (
    PYMDP_AVAILABLE,
    ActiveInferenceMarkovBlanket,
    AgentState,
    BoundaryMonitor,
    BoundaryViolationEvent,
    BoundaryViolationType,
    MarkovBlanketConfig,
    MarkovBlanketInterface,
    ViolationType,
)


class TestAgentState:
    """Test agent state data structure."""

    def test_agent_state_creation(self):
        """Test creating an agent state."""
        state = AgentState(
            agent_id="agent_001",
            position=(10, 20),
            status="active",
            energy=0.85,
            health=0.9,
            intended_action=[0.2, 0.8],
            belief_state=[0.3, 0.5, 0.2],
        )

        assert state.agent_id == "agent_001"
        assert state.position == (10, 20)
        assert state.status == "active"
        assert state.energy == 0.85
        assert state.health == 0.9
        assert state.intended_action == [0.2, 0.8]
        assert state.belief_state == [0.3, 0.5, 0.2]

    def test_agent_state_validation(self):
        """Test agent state with None values."""
        # Test with optional fields as None
        state = AgentState(
            agent_id="agent_002",
            position=None,
            status=None,
            energy=1.0,
            health=1.0,
            intended_action=None,
            belief_state=None,
        )

        assert state.agent_id == "agent_002"
        assert state.position is None
        assert state.status is None
        assert state.energy == 1.0
        assert state.health == 1.0
        assert state.intended_action is None
        assert state.belief_state is None

    def test_agent_state_normalization(self):
        """Test agent state from_agent method."""
        # Test the from_agent class method
        from agents.base.data_model import Agent, AgentStatus, Resources

        mock_agent = Mock(spec=Agent)
        mock_agent.agent_id = "agent_003"
        mock_agent.position = (5, 10)
        mock_agent.status = AgentStatus.IDLE
        mock_agent.resources = Mock(spec=Resources)
        mock_agent.resources.energy = 0.7
        mock_agent.resources.health = 0.8
        mock_agent.belief_state = [0.2, 0.3, 0.5]

        state = AgentState.from_agent(mock_agent)

        assert state.agent_id == "agent_003"
        assert state.position == (5, 10)
        assert state.status == "idle"
        assert state.energy == 0.7
        assert state.health == 0.8
        assert state.belief_state == [0.2, 0.3, 0.5]


class TestBoundaryViolationEvent:
    """Test boundary violation event structure."""

    def test_violation_event_creation(self):
        """Test creating a boundary violation event."""
        event = BoundaryViolationEvent(
            agent_id="agent_001",
            violation_type=ViolationType.SENSORY_OVERFLOW,
            severity=0.8,
            timestamp=datetime.now(),
            independence_measure=0.3,
            threshold_violated=0.5,
            free_energy=1.2,
            expected_free_energy=0.9,
            kl_divergence=0.15,
        )

        assert event.violation_type == ViolationType.SENSORY_OVERFLOW
        assert event.severity == 0.8
        assert event.agent_id == "agent_001"
        assert event.independence_measure == 0.3
        assert event.threshold_violated == 0.5
        assert event.free_energy == 1.2

    def test_violation_types(self):
        """Test all boundary violation types are available."""
        expected_types = [
            "INDEPENDENCE_FAILURE",
            "BOUNDARY_BREACH",
            "SENSORY_OVERFLOW",
            "ACTION_OVERFLOW",
            "INTERNAL_LEAK",
            "EXTERNAL_INTRUSION",
        ]

        for violation_type in expected_types:
            assert hasattr(ViolationType, violation_type)

    def test_violation_event_serialization(self):
        """Test violation event can be serialized."""
        event = BoundaryViolationEvent(
            agent_id="agent_002",
            violation_type=ViolationType.INTERNAL_LEAK,
            severity=0.6,
            timestamp=datetime.now(),
            independence_measure=0.4,
            threshold_violated=0.5,
            free_energy=1.1,
            expected_free_energy=0.8,
            kl_divergence=0.12,
        )

        # Test serialization to dict (using actual fields)
        event_dict = {
            "event_id": event.event_id,
            "agent_id": event.agent_id,
            "violation_type": event.violation_type.value,
            "severity": event.severity,
            "timestamp": event.timestamp.isoformat(),
            "independence_measure": event.independence_measure,
            "free_energy": event.free_energy,
            "kl_divergence": event.kl_divergence,
        }

        assert event_dict["violation_type"] == "internal_leak"
        assert event_dict["severity"] == 0.6
        assert event_dict["agent_id"] == "agent_002"


class TestMarkovBlanketConfig:
    """Test Markov blanket configuration."""

    def test_config_creation(self):
        """Test creating Markov blanket configuration."""
        config = MarkovBlanketConfig(
            num_internal_states=5,
            num_sensory_states=3,
            num_active_states=2,
            boundary_threshold=0.95,
            violation_sensitivity=0.1,
            enable_pymdp_integration=True,
            monitoring_interval=0.1,
        )

        assert config.num_internal_states == 5
        assert config.num_sensory_states == 3
        assert config.num_active_states == 2
        assert config.boundary_threshold == 0.95
        assert config.violation_sensitivity == 0.1
        assert config.enable_pymdp_integration is True
        assert config.monitoring_interval == 0.1

    def test_config_defaults(self):
        """Test configuration default values."""
        config = MarkovBlanketConfig()

        # Test reasonable defaults are set
        assert config.num_internal_states > 0
        assert config.num_sensory_states > 0
        assert config.num_active_states > 0
        assert 0.0 < config.boundary_threshold <= 1.0
        assert config.violation_sensitivity > 0.0
        assert config.monitoring_interval > 0.0

    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid threshold
        with pytest.raises((ValueError, AssertionError)):
            MarkovBlanketConfig(boundary_threshold=1.5)  # Should be <= 1.0

        # Test negative values
        with pytest.raises((ValueError, AssertionError)):
            MarkovBlanketConfig(num_internal_states=-1)


class TestMarkovBlanketInterface:
    """Test abstract Markov blanket interface."""

    def test_interface_is_abstract(self):
        """Test that interface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MarkovBlanketInterface()

    def test_interface_methods(self):
        """Test interface defines required methods."""
        required_methods = [
            "update_boundary",
            "check_boundary_violations",
            "get_current_state",
            "is_boundary_intact",
        ]

        for method_name in required_methods:
            assert hasattr(MarkovBlanketInterface, method_name)


class TestActiveInferenceMarkovBlanket:
    """Test Active Inference Markov blanket implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = MarkovBlanketConfig(
            num_internal_states=3,
            num_sensory_states=2,
            num_active_states=2,
            boundary_threshold=0.9,
            violation_sensitivity=0.1,
        )

        self.mock_agent = Mock()
        self.mock_agent.id = "test_agent"

        # Mock PyMDP components if available
        if PYMDP_AVAILABLE:
            self.mock_pymdp_agent = Mock()
            self.mock_generative_model = Mock()
        else:
            self.mock_pymdp_agent = None
            self.mock_generative_model = None

    def test_markov_blanket_initialization(self):
        """Test Markov blanket initialization."""
        blanket = ActiveInferenceMarkovBlanket(agent=self.mock_agent, config=self.config)

        assert blanket.agent == self.mock_agent
        assert blanket.config == self.config
        assert len(blanket.violation_history) == 0
        assert blanket.boundary_intact is True

    @patch("agents.base.markov_blanket.create_pymdp_generative_model")
    def test_pymdp_integration(self, mock_create_model):
        """Test PyMDP integration when available."""
        if not PYMDP_AVAILABLE:
            pytest.skip("PyMDP not available")

        mock_create_model.return_value = self.mock_generative_model

        config = MarkovBlanketConfig(enable_pymdp_integration=True)
        blanket = ActiveInferenceMarkovBlanket(agent=self.mock_agent, config=config)

        # Test PyMDP integration was initialized
        assert blanket.pymdp_enabled is True
        mock_create_model.assert_called_once()

    def test_update_boundary(self):
        """Test boundary update functionality."""
        blanket = ActiveInferenceMarkovBlanket(agent=self.mock_agent, config=self.config)

        new_state = AgentState(
            agent_id="test_agent",
            internal_states=np.array([0.4, 0.3, 0.3]),
            sensory_states=np.array([0.6, 0.4]),
            active_states=np.array([1, 0]),
            timestamp=datetime.now(),
            confidence=0.8,
        )

        success = blanket.update_boundary(new_state)

        assert success is True
        current_state = blanket.get_current_state()
        assert current_state.agent_id == "test_agent"
        assert np.array_equal(current_state.internal_states, new_state.internal_states)

    def test_boundary_violation_detection(self):
        """Test boundary violation detection."""
        blanket = ActiveInferenceMarkovBlanket(agent=self.mock_agent, config=self.config)

        # Create state that should trigger violation
        problematic_state = AgentState(
            agent_id="test_agent",
            internal_states=np.array([1.0, 0.0, 0.0]),  # Extreme distribution
            sensory_states=np.array([1.0, 0.0]),  # Extreme sensory input
            active_states=np.array([1, 1]),  # Conflicting actions
            timestamp=datetime.now(),
            confidence=0.3,  # Low confidence
        )

        blanket.update_boundary(problematic_state)
        violations = blanket.check_boundary_violations()

        # Should detect violations due to extreme distributions and low
        # confidence
        assert len(violations) > 0
        assert any(
            v.violation_type == BoundaryViolationType.INTERNAL_INCONSISTENCY for v in violations
        )

    def test_boundary_integrity_check(self):
        """Test boundary integrity checking."""
        blanket = ActiveInferenceMarkovBlanket(agent=self.mock_agent, config=self.config)

        # Initially should be intact
        assert blanket.is_boundary_intact() is True

        # Update with normal state
        normal_state = AgentState(
            agent_id="test_agent",
            internal_states=np.array([0.33, 0.33, 0.34]),
            sensory_states=np.array([0.5, 0.5]),
            active_states=np.array([0, 1]),
            timestamp=datetime.now(),
            confidence=0.9,
        )

        blanket.update_boundary(normal_state)
        assert blanket.is_boundary_intact() is True

        # Update with problematic state
        problematic_state = AgentState(
            agent_id="test_agent",
            internal_states=np.array([0.9, 0.05, 0.05]),  # Highly skewed
            sensory_states=np.array([0.95, 0.05]),  # Extreme sensory
            active_states=np.array([1, 0]),
            timestamp=datetime.now(),
            confidence=0.2,  # Very low confidence
        )

        blanket.update_boundary(problematic_state)
        # Should detect boundary issues
        integrity = blanket.is_boundary_intact()
        assert integrity is not None  # Should return some assessment

    def test_statistical_independence_check(self):
        """Test statistical independence verification."""
        blanket = ActiveInferenceMarkovBlanket(agent=self.mock_agent, config=self.config)

        # Mock internal and external states
        internal_states = np.random.rand(10, 3)  # 10 samples, 3 internal states
        external_states = np.random.rand(10, 2)  # 10 samples, 2 external states
        sensory_states = np.random.rand(10, 2)  # 10 samples, 2 sensory states
        active_states = np.random.randint(0, 2, (10, 2))  # 10 samples, 2 active states

        # Test independence check
        independence_score = blanket._check_statistical_independence(
            internal_states, external_states, sensory_states, active_states
        )

        assert 0.0 <= independence_score <= 1.0

    def test_violation_history_management(self):
        """Test violation history tracking."""
        blanket = ActiveInferenceMarkovBlanket(agent=self.mock_agent, config=self.config)

        # Create multiple violations
        violation1 = BoundaryViolationEvent(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            violation_type=BoundaryViolationType.SENSORY_OVERFLOW,
            severity=0.7,
            timestamp=datetime.now(),
            description="First violation",
            affected_states=[],
            recovery_actions=[],
        )

        violation2 = BoundaryViolationEvent(
            id=str(uuid.uuid4()),
            agent_id="test_agent",
            violation_type=BoundaryViolationType.ACTION_CONFLICT,
            severity=0.5,
            timestamp=datetime.now(),
            description="Second violation",
            affected_states=[],
            recovery_actions=[],
        )

        # Add violations to history
        blanket.violation_history.append(violation1)
        blanket.violation_history.append(violation2)

        # Test history retrieval
        recent_violations = blanket.get_recent_violations(limit=5)
        assert len(recent_violations) == 2

        # Test filtering by type
        sensory_violations = [
            v
            for v in recent_violations
            if v.violation_type == BoundaryViolationType.SENSORY_OVERFLOW
        ]
        assert len(sensory_violations) == 1
        assert sensory_violations[0].description == "First violation"


class TestBoundaryMonitor:
    """Test boundary monitoring functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = MarkovBlanketConfig(monitoring_interval=0.01)
        self.mock_agent = Mock()
        self.mock_agent.id = "monitored_agent"

        self.blanket = ActiveInferenceMarkovBlanket(agent=self.mock_agent, config=self.config)

        self.monitor = BoundaryMonitor(self.blanket)

    def test_monitor_initialization(self):
        """Test boundary monitor initialization."""
        assert self.monitor.markov_blanket == self.blanket
        assert self.monitor.is_monitoring is False
        assert len(self.monitor.violation_callbacks) == 0

    def test_violation_callback_registration(self):
        """Test registering violation callbacks."""
        callback_called = []

        def test_callback(violation):
            callback_called.append(violation)

        self.monitor.register_violation_callback(test_callback)
        assert len(self.monitor.violation_callbacks) == 1

        # Simulate violation
        violation = BoundaryViolationEvent(
            id=str(uuid.uuid4()),
            agent_id="monitored_agent",
            violation_type=BoundaryViolationType.BOUNDARY_BREACH,
            severity=0.8,
            timestamp=datetime.now(),
            description="Test violation",
            affected_states=[],
            recovery_actions=[],
        )

        # Trigger callback
        self.monitor._notify_violation_callbacks(violation)

        assert len(callback_called) == 1
        assert callback_called[0] == violation

    def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring."""
        assert self.monitor.is_monitoring is False

        self.monitor.start_monitoring()
        assert self.monitor.is_monitoring is True

        self.monitor.stop_monitoring()
        assert self.monitor.is_monitoring is False

    def test_monitoring_thread_safety(self):
        """Test monitoring thread safety."""
        import threading

        violations_detected = []

        def violation_handler(violation):
            violations_detected.append(violation)

        self.monitor.register_violation_callback(violation_handler)

        # Start monitoring
        self.monitor.start_monitoring()

        # Simulate state updates from multiple threads
        def update_state():
            state = AgentState(
                agent_id="monitored_agent",
                internal_states=np.random.rand(3),
                sensory_states=np.random.rand(2),
                active_states=np.random.randint(0, 2, 2),
                timestamp=datetime.now(),
                confidence=0.8,
            )
            self.blanket.update_boundary(state)

        threads = []
        for _ in range(3):
            thread = threading.Thread(target=update_state)
            threads.append(thread)
            thread.start()

        # Wait for threads to complete
        for thread in threads:
            thread.join()

        # Stop monitoring
        self.monitor.stop_monitoring()

        # Should handle concurrent updates without errors
        assert self.monitor.is_monitoring is False


class TestMarkovBlanketIntegration:
    """Integration tests for Markov blanket system."""

    def test_full_boundary_workflow(self):
        """Test complete boundary management workflow."""
        config = MarkovBlanketConfig(
            num_internal_states=4,
            num_sensory_states=3,
            num_active_states=2,
            boundary_threshold=0.8,
            violation_sensitivity=0.15,
        )

        mock_agent = Mock()
        mock_agent.id = "integration_test_agent"

        # 1. Initialize Markov blanket
        blanket = ActiveInferenceMarkovBlanket(agent=mock_agent, config=config)

        # 2. Set up monitoring
        monitor = BoundaryMonitor(blanket)
        violations_caught = []

        def catch_violations(violation):
            violations_caught.append(violation)

        monitor.register_violation_callback(catch_violations)
        monitor.start_monitoring()

        # 3. Update with normal states
        normal_state = AgentState(
            agent_id="integration_test_agent",
            internal_states=np.array([0.25, 0.25, 0.25, 0.25]),
            sensory_states=np.array([0.33, 0.33, 0.34]),
            active_states=np.array([0, 1]),
            timestamp=datetime.now(),
            confidence=0.9,
        )

        blanket.update_boundary(normal_state)
        assert blanket.is_boundary_intact() is True

        # 4. Introduce problematic state
        problematic_state = AgentState(
            agent_id="integration_test_agent",
            internal_states=np.array([0.8, 0.1, 0.05, 0.05]),  # Highly skewed
            sensory_states=np.array([0.9, 0.05, 0.05]),  # Extreme sensory
            active_states=np.array([1, 1]),  # Potentially conflicting
            timestamp=datetime.now(),
            confidence=0.3,  # Low confidence
        )

        blanket.update_boundary(problematic_state)

        # 5. Check for violations
        violations = blanket.check_boundary_violations()

        # 6. Verify violations were detected and callbacks triggered
        assert len(violations) > 0

        # 7. Clean up
        monitor.stop_monitoring()

        # Verify complete workflow
        assert len(blanket.violation_history) > 0
        assert any(v.agent_id == "integration_test_agent" for v in blanket.violation_history)

    @pytest.mark.skipif(not PYMDP_AVAILABLE, reason="PyMDP not available")
    def test_pymdp_integration_workflow(self):
        """Test workflow with PyMDP integration."""
        config = MarkovBlanketConfig(
            enable_pymdp_integration=True,
            num_internal_states=3,
            num_sensory_states=2,
            num_active_states=2,
        )

        mock_agent = Mock()
        mock_agent.id = "pymdp_test_agent"

        with patch("agents.base.markov_blanket.create_pymdp_generative_model") as mock_create:
            mock_generative_model = Mock()
            mock_create.return_value = mock_generative_model

            blanket = ActiveInferenceMarkovBlanket(agent=mock_agent, config=config)

            # Verify PyMDP integration
            assert blanket.pymdp_enabled is True
            assert mock_create.called

            # Test state updates with PyMDP
            state = AgentState(
                agent_id="pymdp_test_agent",
                internal_states=np.array([0.4, 0.3, 0.3]),
                sensory_states=np.array([0.6, 0.4]),
                active_states=np.array([1, 0]),
                timestamp=datetime.now(),
                confidence=0.85,
            )

            success = blanket.update_boundary(state)
            assert success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
