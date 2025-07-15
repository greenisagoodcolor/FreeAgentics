"""Simple test to verify coverage setup works."""


def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    if a == 0 or b == 0:
        return 0
    return a * b


def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


class TestSimpleFunctions:
    """Test simple mathematical functions."""

    def test_add(self):
        """Test addition."""
        assert add(2, 3) == 5
        assert add(-1, 1) == 0
        assert add(0, 0) == 0

    def test_multiply(self):
        """Test multiplication."""
        assert multiply(2, 3) == 6
        assert multiply(5, 0) == 0
        assert multiply(0, 5) == 0
        assert multiply(-2, 3) == -6

    def test_divide(self):
        """Test division."""
        assert divide(10, 2) == 5.0
        assert divide(7, 2) == 3.5

    def test_divide_by_zero(self):
        """Test division by zero raises error."""
        import pytest

        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(10, 0)
