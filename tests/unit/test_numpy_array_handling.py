"""Test numpy array handling in PyMDP operations."""

import numpy as np
import pytest

from agents.base_agent import safe_array_to_int


@pytest.mark.slow
class TestNumpyArrayHandling:
    """Test safe conversion of numpy arrays to integers."""

    def test_safe_array_to_int_scalar(self):
        """Test conversion of regular scalar."""
        assert safe_array_to_int(42) == 42
        assert safe_array_to_int(3.14) == 3

    def test_safe_array_to_int_numpy_scalar(self):
        """Test conversion of numpy scalars."""
        assert safe_array_to_int(np.int32(42)) == 42
        assert safe_array_to_int(np.int64(100)) == 100
        assert safe_array_to_int(np.float32(3.14)) == 3

    def test_safe_array_to_int_zero_d_array(self):
        """Test conversion of 0-dimensional numpy arrays."""
        assert safe_array_to_int(np.array(42)) == 42
        assert safe_array_to_int(np.array(3.14)) == 3

    def test_safe_array_to_int_single_element_array(self):
        """Test conversion of single-element arrays."""
        assert safe_array_to_int(np.array([42])) == 42
        assert safe_array_to_int(np.array([3.14])) == 3

    def test_safe_array_to_int_multi_element_array(self):
        """Test conversion of multi-element arrays (takes first element)."""
        assert safe_array_to_int(np.array([42, 99, 123])) == 42
        assert safe_array_to_int(np.array([3.14, 2.71])) == 3

    def test_safe_array_to_int_empty_array(self):
        """Test conversion of empty arrays (should raise error)."""
        with pytest.raises(ValueError, match="Empty array"):
            safe_array_to_int(np.array([]))

    def test_safe_array_to_int_list(self):
        """Test conversion of regular lists."""
        assert safe_array_to_int([42]) == 42
        assert safe_array_to_int([3.14, 2.71]) == 3

    def test_safe_array_to_int_invalid_input(self):
        """Test conversion of invalid inputs."""
        with pytest.raises(ValueError):
            safe_array_to_int("invalid")

        with pytest.raises(ValueError):
            safe_array_to_int(None)

        with pytest.raises(ValueError):
            safe_array_to_int([])


@pytest.mark.slow
class TestPyMDPArrayHandling:
    """Test PyMDP array handling with mocked PyMDP responses."""

    @pytest.fixture
    def mock_pymdp_responses(self):
        """Mock various PyMDP response types."""
        return {
            "scalar_int": 2,
            "scalar_float": 2.7,
            "numpy_scalar": np.int32(2),
            "zero_d_array": np.array(2),
            "single_element_array": np.array([2]),
            "multi_element_array": np.array([2, 0, 1]),
            "argmax_result": np.argmax(np.array([0.1, 0.8, 0.1])),  # Should be 1
        }

    def test_all_pymdp_response_types(self, mock_pymdp_responses):
        """Test that all common PyMDP response types convert correctly."""
        for name, response in mock_pymdp_responses.items():
            result = safe_array_to_int(response)
            assert isinstance(result, int), f"Failed for {name}: {response}"

            if name == "argmax_result":
                assert result == 1, f"Argmax result should be 1, got {result}"
            else:
                # Most should convert to 2
                assert result == 2, f"Failed for {name}: expected 2, got {result}"


@pytest.mark.slow
class TestAgentArrayHandling:
    """Integration tests for array handling in agent classes."""

    def test_action_mapping_with_arrays(self):
        """Test that action mapping works with numpy array indices."""
        action_map = {0: "up", 1: "down", 2: "left", 3: "right", 4: "stay"}

        # Test various numpy array types
        test_cases = [
            np.array(2),  # 0-d array
            np.array([2]),  # 1-d array
            np.int32(2),  # numpy scalar
            np.argmax(np.array([0.1, 0.1, 0.8, 0.0, 0.0])),  # argmax result
        ]

        for action_idx in test_cases:
            converted_idx = safe_array_to_int(action_idx)
            action = action_map.get(converted_idx, "stay")
            assert action == "left", (
                f"Expected 'left', got '{action}' for input {action_idx}"
            )

    def test_policy_indexing_with_arrays(self):
        """Test that policy indexing works with numpy arrays."""
        policies = ["explore", "exploit", "return", "coordinate"]

        # Simulate q_pi from PyMDP (posterior over policies)
        q_pi = np.array([0.1, 0.6, 0.2, 0.1])
        best_policy_idx = safe_array_to_int(np.argmax(q_pi))

        assert best_policy_idx == 1
        assert policies[best_policy_idx] == "exploit"


if __name__ == "__main__":
    pytest.main([__file__])
