"""Base class for SQLAlchemy models."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for all database models.

    Using DeclarativeBase for proper type inference and
    modern SQLAlchemy 2.0+ patterns.
    """

    pass
