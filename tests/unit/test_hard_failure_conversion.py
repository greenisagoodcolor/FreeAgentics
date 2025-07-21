"""Test suite for hard failure conversion - verifies exceptions are raised instead of graceful degradation."""

import numpy as np
import pytest

from agents.hard_failure_handlers import (
    ErrorHandlerHardFailure,
    HardFailureError,
    PyMDPErrorHandlerHardFailure,
    safe_array_index_hard_failure,
    safe_pymdp_operation_hard_failure,
    validate_observation_hard_failure,
    validate_pymdp_matrices_hard_failure,
    with_error_handling_hard_failure,
)


class TestErrorHandlerHardFailure:
    """Test ErrorHandlerHardFailure raises exceptions instead of returning recovery info."""

    def test_handle_error_raises_exception(self):
        """Test that handle_error raises HardFailureError instead of returning recovery dict."""
        handler = ErrorHandlerHardFailure("test_agent")
        original_error = ValueError("Test error")

        with pytest.raises(HardFailureError) as exc_info:
            handler.handle_error(original_error, "test_operation")

        assert "Hard failure in test_operation for agent test_agent" in str(
            exc_info.value
        )
        assert exc_info.value.__cause__ is original_error

    def test_reset_strategy_raises_not_implemented(self):
        """Test that reset_strategy raises NotImplementedError."""
        handler = ErrorHandlerHardFailure("test_agent")

        with pytest.raises(NotImplementedError) as exc_info:
            handler.reset_strategy("test_strategy")

        assert "Hard failure mode does not support recovery strategies" in str(
            exc_info.value
        )

    def test_reset_all_strategies_raises_not_implemented(self):
        """Test that reset_all_strategies raises NotImplementedError."""
        handler = ErrorHandlerHardFailure("test_agent")

        with pytest.raises(NotImplementedError) as exc_info:
            handler.reset_all_strategies()

        assert "Hard failure mode does not support recovery strategies" in str(
            exc_info.value
        )

    def test_get_error_statistics_minimal_info(self):
        """Test that error statistics indicate hard failure mode."""
        handler = ErrorHandlerHardFailure("test_agent")
        stats = handler.get_error_statistics()

        assert stats["mode"] == "hard_failure"
        assert "immediate failures" in stats["message"]


class TestPyMDPErrorHandlerHardFailure:
    """Test PyMDPErrorHandlerHardFailure enforces immediate failures."""

    def test_safe_execute_with_successful_operation(self):
        """Test safe_execute returns result when operation succeeds."""
        handler = PyMDPErrorHandlerHardFailure("test_agent")

        def successful_operation():
            return "success"

        success, result, error = handler.safe_execute("test_op", successful_operation)

        assert success is True
        assert result == "success"
        assert error is None

    def test_safe_execute_raises_on_failure(self):
        """Test safe_execute raises HardFailureError when operation fails."""
        handler = PyMDPErrorHandlerHardFailure("test_agent")

        def failing_operation():
            raise ValueError("Operation failed")

        with pytest.raises(HardFailureError) as exc_info:
            handler.safe_execute("test_op", failing_operation)

        assert "PyMDP operation test_op failed" in str(exc_info.value)

    def test_safe_execute_ignores_fallback_func(self):
        """Test safe_execute ignores fallback function and raises immediately."""
        handler = PyMDPErrorHandlerHardFailure("test_agent")

        def failing_operation():
            raise ValueError("Operation failed")

        def fallback_operation():
            return "fallback_result"

        # Should raise despite having a fallback function
        with pytest.raises(HardFailureError):
            handler.safe_execute("test_op", failing_operation, fallback_operation)

    def test_safe_execute_validates_callable(self):
        """Test safe_execute validates operation is callable."""
        handler = PyMDPErrorHandlerHardFailure("test_agent")

        with pytest.raises(AssertionError) as exc_info:
            handler.safe_execute("test_op", "not_callable")

        assert "must be callable" in str(exc_info.value)


class TestArrayIndexHardFailure:
    """Test safe_array_index_hard_failure enforces bounds checking with assertions."""

    def test_valid_index_returns_value(self):
        """Test valid array indexing returns correct value."""
        array = np.array([1, 2, 3, 4, 5])
        result = safe_array_index_hard_failure(array, 2)
        assert result == 3

    def test_invalid_index_raises_assertion(self):
        """Test invalid index raises AssertionError."""
        array = np.array([1, 2, 3])

        with pytest.raises(AssertionError) as exc_info:
            safe_array_index_hard_failure(array, 5)

        assert "out of bounds" in str(exc_info.value)

    def test_negative_index_raises_assertion(self):
        """Test negative index raises AssertionError."""
        array = np.array([1, 2, 3])

        with pytest.raises(AssertionError) as exc_info:
            safe_array_index_hard_failure(array, -1)

        assert "out of bounds" in str(exc_info.value)

    def test_empty_array_raises_assertion(self):
        """Test empty array raises AssertionError."""
        array = np.array([])

        with pytest.raises(AssertionError) as exc_info:
            safe_array_index_hard_failure(array, 0)

        assert "cannot be empty" in str(exc_info.value)

    def test_non_numpy_array_raises_assertion(self):
        """Test non-numpy array raises AssertionError."""
        with pytest.raises(AssertionError) as exc_info:
            safe_array_index_hard_failure([1, 2, 3], 0)

        assert "Expected numpy array" in str(exc_info.value)


class TestObservationValidationHardFailure:
    """Test validate_observation_hard_failure enforces strict validation."""

    def test_valid_observation_returns_unchanged(self):
        """Test valid observation is returned unchanged."""
        obs = np.array([1.0, 2.0, 3.0])
        result = validate_observation_hard_failure(obs)
        np.testing.assert_array_equal(result, obs)

    def test_none_observation_raises_assertion(self):
        """Test None observation raises AssertionError."""
        with pytest.raises(AssertionError) as exc_info:
            validate_observation_hard_failure(None)

        assert "cannot be None" in str(exc_info.value)

    def test_empty_array_raises_assertion(self):
        """Test empty array observation raises AssertionError."""
        obs = np.array([])

        with pytest.raises(AssertionError) as exc_info:
            validate_observation_hard_failure(obs)

        assert "cannot be empty" in str(exc_info.value)

    def test_nan_values_raise_assertion(self):
        """Test observation with NaN values raises AssertionError."""
        obs = np.array([1.0, np.nan, 3.0])

        with pytest.raises(AssertionError) as exc_info:
            validate_observation_hard_failure(obs)

        assert "NaN values" in str(exc_info.value)

    def test_infinite_values_raise_assertion(self):
        """Test observation with infinite values raises AssertionError."""
        obs = np.array([1.0, np.inf, 3.0])

        with pytest.raises(AssertionError) as exc_info:
            validate_observation_hard_failure(obs)

        assert "infinite values" in str(exc_info.value)


class TestPyMDPMatrixValidationHardFailure:
    """Test validate_pymdp_matrices_hard_failure enforces strict validation."""

    def test_valid_matrices_return_success(self):
        """Test valid matrices return success tuple."""
        A = np.array([[1, 0], [0, 1]])
        B = np.array([[1, 0], [0, 1]])
        C = np.array([[1, 0], [0, 1]])
        D = np.array([0.5, 0.5])

        success, message = validate_pymdp_matrices_hard_failure(A, B, C, D)

        assert success is True
        assert "validated successfully" in message

    def test_none_matrix_raises_assertion(self):
        """Test None matrix raises AssertionError."""
        A = np.array([[1, 0], [0, 1]])
        B = None
        C = np.array([[1, 0], [0, 1]])
        D = np.array([0.5, 0.5])

        with pytest.raises(AssertionError) as exc_info:
            validate_pymdp_matrices_hard_failure(A, B, C, D)

        assert "B matrix cannot be None" in str(exc_info.value)

    def test_invalid_type_raises_assertion(self):
        """Test invalid matrix type raises AssertionError."""
        A = np.array([[1, 0], [0, 1]])
        B = "not_a_matrix"
        C = np.array([[1, 0], [0, 1]])
        D = np.array([0.5, 0.5])

        with pytest.raises(AssertionError) as exc_info:
            validate_pymdp_matrices_hard_failure(A, B, C, D)

        assert "must be array-like" in str(exc_info.value)


class TestPyMDPOperationDecorator:
    """Test safe_pymdp_operation_hard_failure decorator enforces hard failures."""

    def test_successful_operation_returns_result(self):
        """Test decorator allows successful operations to return normally."""

        @safe_pymdp_operation_hard_failure("test_operation")
        def successful_operation(x, y):
            return x + y

        result = successful_operation(2, 3)
        assert result == 5

    def test_failing_operation_raises_hard_failure_error(self):
        """Test decorator converts exceptions to HardFailureError."""

        @safe_pymdp_operation_hard_failure("test_operation")
        def failing_operation():
            raise ValueError("Test error")

        with pytest.raises(HardFailureError) as exc_info:
            failing_operation()

        assert "PyMDP operation test_operation failed" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, ValueError)


class TestErrorHandlingDecorator:
    """Test with_error_handling_hard_failure decorator enforces hard failures."""

    def test_successful_operation_returns_result(self):
        """Test decorator allows successful operations to return normally."""

        @with_error_handling_hard_failure("test_operation")
        def successful_operation(value):
            return value * 2

        result = successful_operation(5)
        assert result == 10

    def test_failing_operation_raises_hard_failure_error(self):
        """Test decorator converts exceptions to HardFailureError."""

        @with_error_handling_hard_failure("test_operation")
        def failing_operation():
            raise RuntimeError("Test error")

        with pytest.raises(HardFailureError) as exc_info:
            failing_operation()

        assert "Operation test_operation failed" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_ignores_fallback_result_parameter(self):
        """Test decorator ignores fallback_result and raises exception."""

        @with_error_handling_hard_failure("test_operation", fallback_result="fallback")
        def failing_operation():
            raise ValueError("Test error")

        # Should raise despite having fallback_result specified
        with pytest.raises(HardFailureError):
            failing_operation()


class TestHardFailureIntegration:
    """Integration tests for hard failure patterns."""

    def test_chained_operations_fail_immediately(self):
        """Test that chained operations fail immediately when any step fails."""

        @safe_pymdp_operation_hard_failure("step1")
        def step1():
            return "step1_result"

        @safe_pymdp_operation_hard_failure("step2")
        def step2(input_data):
            raise ValueError("Step 2 failed")

        @safe_pymdp_operation_hard_failure("step3")
        def step3(input_data):
            return "step3_result"

        # Should fail at step2 and not proceed to step3
        with pytest.raises(HardFailureError) as exc_info:
            result1 = step1()
            result2 = step2(result1)
            step3(result2)  # This should never execute

        assert "step2" in str(exc_info.value)

    def test_no_silent_failures_allowed(self):
        """Test that no operation can fail silently."""
        handler = PyMDPErrorHandlerHardFailure("test_agent")

        def silent_failure_attempt():
            try:
                raise ValueError("This should not be caught")
            except ValueError:
                return "silent_failure"  # This pattern should not exist

        # Even operations that try to catch their own errors should be caught
        # when wrapped in our hard failure system
        with pytest.raises(HardFailureError):
            handler.safe_execute("silent_test", lambda: silent_failure_attempt())


if __name__ == "__main__":
    # Run the tests to verify hard failure behavior
    pytest.main([__file__, "-v"])
