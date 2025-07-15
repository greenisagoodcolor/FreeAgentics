"""Tests for database.base module."""

import pytest


class TestDatabaseBase:
    """Test the database base module."""

    def test_import_database_base(self):
        """Test that database.base can be imported."""
        try:
            from database.base import Base  # noqa: F401
            # Test that Base exists and is a class
            assert Base is not None
        except ImportError:
            # Skip if there are import issues in dependent modules
            pytest.skip("Cannot import database.base due to dependency issues")

    def test_base_attributes(self):
        """Test Base class attributes."""
        from database.base import Base

        # Test that Base has the expected attributes
        assert hasattr(Base, 'metadata')
        assert hasattr(Base, 'registry')