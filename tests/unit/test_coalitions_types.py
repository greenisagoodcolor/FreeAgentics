"""Tests for coalitions.types module."""

from unittest.mock import MagicMock



class TestCoalitionsTypes:
    """Test the coalitions types module."""

    def test_import_formation_result(self):
        """Test that FormationResult can be imported."""
        try:
            from coalitions.types import FormationResult

            # Test that FormationResult is a class
            assert FormationResult is not None
            assert hasattr(FormationResult, "__dataclass_fields__")
        except ImportError:
            assert False, "Test bypass removed - must fix underlying issue"

    def test_formation_result_creation(self):
        """Test FormationResult creation."""
        try:
            from coalitions.types import FormationResult

            # Create mock data
            mock_coalitions = [MagicMock(), MagicMock()]
            mock_objectives = [MagicMock()]

            # Create FormationResult
            result = FormationResult(
                coalitions=mock_coalitions,
                unassigned_objectives=mock_objectives,
                formation_quality=0.8,
                objective_coverage=0.7,
                agent_utilization=0.9,
            )

            # Test attributes
            assert result.coalitions == mock_coalitions
            assert result.unassigned_objectives == mock_objectives
            assert result.formation_quality == 0.8
            assert result.objective_coverage == 0.7
            assert result.agent_utilization == 0.9
        except ImportError:
            assert False, "Test bypass removed - must fix underlying issue"

    def test_formation_result_fields(self):
        """Test FormationResult has expected fields."""
        try:
            from coalitions.types import FormationResult

            # Test dataclass fields
            expected_fields = {
                "coalitions",
                "unassigned_objectives",
                "formation_quality",
                "objective_coverage",
                "agent_utilization",
            }

            actual_fields = set(FormationResult.__dataclass_fields__.keys())
            assert actual_fields == expected_fields
        except ImportError:
            assert False, "Test bypass removed - must fix underlying issue"

    def test_formation_result_default_values(self):
        """Test FormationResult with minimal data."""
        try:
            from coalitions.types import FormationResult

            # Test with empty lists and zero values
            result = FormationResult(
                coalitions=[],
                unassigned_objectives=[],
                formation_quality=0.0,
                objective_coverage=0.0,
                agent_utilization=0.0,
            )

            assert result.coalitions == []
            assert result.unassigned_objectives == []
            assert result.formation_quality == 0.0
            assert result.objective_coverage == 0.0
            assert result.agent_utilization == 0.0
        except ImportError:
            assert False, "Test bypass removed - must fix underlying issue"
