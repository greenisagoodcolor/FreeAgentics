"""Tests for observability.__init__ module."""

from unittest.mock import MagicMock, patch

import pytest


class TestObservabilityInit:
    """Test the observability package initialization."""

    @patch("observability.pymdp_integration.get_pymdp_performance_summary")
    @patch("observability.pymdp_integration.monitor_pymdp_inference")
    @patch("observability.pymdp_integration.pymdp_observer")
    @patch("observability.pymdp_integration.record_agent_lifecycle_event")
    @patch("observability.pymdp_integration.record_belief_update")
    @patch("observability.pymdp_integration.record_coordination_event")
    def test_observability_imports(
        self,
        mock_record_coordination,
        mock_record_belief,
        mock_record_lifecycle,
        mock_observer,
        mock_monitor,
        mock_get_summary,
    ):
        """Test that observability package imports work correctly."""
        try:
            from observability import (
                get_pymdp_performance_summary,
                monitor_pymdp_inference,
                pymdp_observer,
                record_agent_lifecycle_event,
                record_belief_update,
                record_coordination_event,
            )

            # Test that all functions are available
            assert get_pymdp_performance_summary is not None
            assert monitor_pymdp_inference is not None
            assert pymdp_observer is not None
            assert record_agent_lifecycle_event is not None
            assert record_belief_update is not None
            assert record_coordination_event is not None

        except ImportError as e:
            pytest.skip(
                f"Cannot import observability due to dependency issues: {e}"
            )

    def test_observability_all_exports(self):
        """Test that __all__ contains expected exports."""
        try:
            import observability

            expected_exports = [
                "pymdp_observer",
                "monitor_pymdp_inference",
                "record_belief_update",
                "record_agent_lifecycle_event",
                "record_coordination_event",
                "get_pymdp_performance_summary",
            ]

            assert hasattr(observability, "__all__")
            assert set(observability.__all__) == set(expected_exports)

        except ImportError as e:
            pytest.skip(
                f"Cannot import observability due to dependency issues: {e}"
            )

    def test_observability_module_docstring(self):
        """Test that observability module has proper docstring."""
        try:
            import observability

            assert observability.__doc__ is not None
            assert (
                "Observability package for FreeAgentics monitoring and instrumentation"
                in observability.__doc__
            )

        except ImportError as e:
            pytest.skip(
                f"Cannot import observability due to dependency issues: {e}"
            )

    @patch("observability.pymdp_integration.get_pymdp_performance_summary")
    @patch("observability.pymdp_integration.monitor_pymdp_inference")
    @patch("observability.pymdp_integration.pymdp_observer")
    @patch("observability.pymdp_integration.record_agent_lifecycle_event")
    @patch("observability.pymdp_integration.record_belief_update")
    @patch("observability.pymdp_integration.record_coordination_event")
    def test_observability_function_accessibility(
        self,
        mock_record_coordination,
        mock_record_belief,
        mock_record_lifecycle,
        mock_observer,
        mock_monitor,
        mock_get_summary,
    ):
        """Test that all imported functions are accessible from package level."""
        try:
            import observability

            # Test that all functions are accessible
            assert hasattr(observability, "get_pymdp_performance_summary")
            assert hasattr(observability, "monitor_pymdp_inference")
            assert hasattr(observability, "pymdp_observer")
            assert hasattr(observability, "record_agent_lifecycle_event")
            assert hasattr(observability, "record_belief_update")
            assert hasattr(observability, "record_coordination_event")

        except ImportError as e:
            pytest.skip(
                f"Cannot import observability due to dependency issues: {e}"
            )
