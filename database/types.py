"""Custom database types for cross-platform compatibility."""

import uuid

from sqlalchemy import CHAR
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.types import TypeDecorator


class GUID(TypeDecorator):
    """Platform-independent GUID type.

    Uses PostgreSQL's UUID type when available,
    otherwise uses CHAR(36) for SQLite compatibility.

    This ensures that UUID columns work across different database backends.
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        """Load the dialect-specific implementation."""
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PostgreSQLUUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        """Process the value when binding to SQL."""
        if value is None:
            return value
        elif dialect.name == "postgresql":
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return str(uuid.UUID(value))
            else:
                return str(value)

    def process_result_value(self, value, dialect):
        """Process the value from SQL result."""
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                return uuid.UUID(value)
            return value
