"""Tests for database.types module."""

import uuid
from unittest.mock import MagicMock

import pytest


class TestDatabaseTypes:
    """Test the database types module."""

    def test_import_guid(self):
        """Test that GUID can be imported."""
        from database.types import GUID

        # Test that GUID is a class
        assert GUID is not None
        assert hasattr(GUID, "impl")
        assert hasattr(GUID, "cache_ok")

    def test_guid_creation(self):
        """Test GUID type creation."""
        from database.types import GUID

        # Create GUID instance
        guid_type = GUID()

        # Test basic attributes
        assert guid_type.cache_ok is True
        assert guid_type.impl is not None

    def test_guid_load_dialect_impl_postgresql(self):
        """Test load_dialect_impl with PostgreSQL dialect."""
        from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID

        from database.types import GUID

        # Create mock PostgreSQL dialect
        mock_dialect = MagicMock()
        mock_dialect.name = "postgresql"
        mock_dialect.type_descriptor = MagicMock(return_value="PostgreSQL UUID Type")

        guid_type = GUID()
        result = guid_type.load_dialect_impl(mock_dialect)

        # Should call type_descriptor with PostgreSQL UUID
        mock_dialect.type_descriptor.assert_called_once()
        args = mock_dialect.type_descriptor.call_args[0]
        assert isinstance(args[0], PostgreSQLUUID)
        assert args[0].as_uuid is True
        assert result == "PostgreSQL UUID Type"

    def test_guid_load_dialect_impl_sqlite(self):
        """Test load_dialect_impl with SQLite dialect."""
        from sqlalchemy import CHAR

        from database.types import GUID

        # Create mock SQLite dialect
        mock_dialect = MagicMock()
        mock_dialect.name = "sqlite"
        mock_dialect.type_descriptor = MagicMock(return_value="SQLite CHAR Type")

        guid_type = GUID()
        result = guid_type.load_dialect_impl(mock_dialect)

        # Should call type_descriptor with CHAR(36)
        mock_dialect.type_descriptor.assert_called_once()
        args = mock_dialect.type_descriptor.call_args[0]
        assert isinstance(args[0], CHAR)
        assert args[0].length == 36
        assert result == "SQLite CHAR Type"

    def test_guid_process_bind_param_none(self):
        """Test process_bind_param with None value."""
        from database.types import GUID

        mock_dialect = MagicMock()
        guid_type = GUID()

        result = guid_type.process_bind_param(None, mock_dialect)
        assert result is None

    def test_guid_process_bind_param_postgresql(self):
        """Test process_bind_param with PostgreSQL dialect."""
        from database.types import GUID

        mock_dialect = MagicMock()
        mock_dialect.name = "postgresql"
        test_uuid = uuid.uuid4()

        guid_type = GUID()
        result = guid_type.process_bind_param(test_uuid, mock_dialect)

        # Should return the UUID as-is for PostgreSQL
        assert result == test_uuid

    def test_guid_process_bind_param_sqlite_uuid(self):
        """Test process_bind_param with SQLite and UUID value."""
        from database.types import GUID

        mock_dialect = MagicMock()
        mock_dialect.name = "sqlite"
        test_uuid = uuid.uuid4()

        guid_type = GUID()
        result = guid_type.process_bind_param(test_uuid, mock_dialect)

        # Should return string representation for SQLite
        assert result == str(test_uuid)
        assert isinstance(result, str)

    def test_guid_process_bind_param_sqlite_string(self):
        """Test process_bind_param with SQLite and string value."""
        from database.types import GUID

        mock_dialect = MagicMock()
        mock_dialect.name = "sqlite"
        test_uuid = uuid.uuid4()
        test_string = str(test_uuid)

        guid_type = GUID()
        result = guid_type.process_bind_param(test_string, mock_dialect)

        # Should return string representation for SQLite
        assert result == test_string
        assert isinstance(result, str)

    def test_guid_process_bind_param_sqlite_invalid_string(self):
        """Test process_bind_param with SQLite and invalid UUID string."""
        from database.types import GUID

        mock_dialect = MagicMock()
        mock_dialect.name = "sqlite"
        invalid_string = "not-a-valid-uuid"

        guid_type = GUID()

        # Should raise ValueError when trying to validate UUID
        with pytest.raises(ValueError):
            guid_type.process_bind_param(invalid_string, mock_dialect)

    def test_guid_process_result_value_none(self):
        """Test process_result_value with None value."""
        from database.types import GUID

        mock_dialect = MagicMock()
        guid_type = GUID()

        result = guid_type.process_result_value(None, mock_dialect)
        assert result is None

    def test_guid_process_result_value_uuid(self):
        """Test process_result_value with UUID value."""
        from database.types import GUID

        mock_dialect = MagicMock()
        test_uuid = uuid.uuid4()

        guid_type = GUID()
        result = guid_type.process_result_value(test_uuid, mock_dialect)

        # Should return UUID as-is
        assert result == test_uuid
        assert isinstance(result, uuid.UUID)

    def test_guid_process_result_value_string(self):
        """Test process_result_value with string value."""
        from database.types import GUID

        mock_dialect = MagicMock()
        test_uuid = uuid.uuid4()
        test_string = str(test_uuid)

        guid_type = GUID()
        result = guid_type.process_result_value(test_string, mock_dialect)

        # Should convert string to UUID
        assert result == test_uuid
        assert isinstance(result, uuid.UUID)

    def test_guid_process_result_value_invalid_string(self):
        """Test process_result_value with invalid string value."""
        from database.types import GUID

        mock_dialect = MagicMock()
        invalid_string = "not-a-uuid"

        guid_type = GUID()

        # Should raise ValueError for invalid UUID string
        with pytest.raises(ValueError):
            guid_type.process_result_value(invalid_string, mock_dialect)

    def test_guid_cache_ok_attribute(self):
        """Test that GUID has cache_ok attribute set to True."""
        from database.types import GUID

        assert GUID.cache_ok is True

    def test_guid_impl_attribute(self):
        """Test that GUID has impl attribute."""
        from sqlalchemy import CHAR

        from database.types import GUID

        assert GUID.impl is CHAR
