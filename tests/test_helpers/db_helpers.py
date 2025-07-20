"""Database testing helpers and mocks."""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from database.base import Base
from database.session import get_db

# Use in-memory SQLite for testing
TEST_DATABASE_URL = "sqlite:///:memory:"


def get_test_db():
    """Create a test database session."""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # Important for in-memory databases
    )
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=engine
    )

    # Create all tables
    Base.metadata.create_all(bind=engine)

    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


def override_get_db(app):
    """Override the database dependency for testing."""
    app.dependency_overrides[get_db] = get_test_db


def reset_db_override(app):
    """Reset database dependency override."""
    app.dependency_overrides.pop(get_db, None)


class MockDatabase:
    """Mock database for unit testing without actual DB connections."""

    def __init__(self):
        self.data = {}
        self._id_counter = 1

    def add(self, obj):
        """Mock add operation."""
        obj_type = type(obj).__name__
        if obj_type not in self.data:
            self.data[obj_type] = {}

        # Assign ID if not present
        if not hasattr(obj, 'id') or obj.id is None:
            obj.id = self._id_counter
            self._id_counter += 1

        self.data[obj_type][obj.id] = obj

    def commit(self):
        """Mock commit operation."""
        pass

    def rollback(self):
        """Mock rollback operation."""
        pass

    def query(self, model):
        """Mock query operation."""
        return MockQuery(self, model)

    def close(self):
        """Mock close operation."""
        pass

    def refresh(self, obj):
        """Mock refresh operation."""
        pass


class MockQuery:
    """Mock query object for database operations."""

    def __init__(self, db, model):
        self.db = db
        self.model = model
        self.filters = []

    def filter(self, *args):
        """Mock filter operation."""
        self.filters.extend(args)
        return self

    def filter_by(self, **kwargs):
        """Mock filter_by operation."""
        self.filters.append(kwargs)
        return self

    def first(self):
        """Mock first operation."""
        model_name = self.model.__name__
        if model_name in self.db.data:
            for obj in self.db.data[model_name].values():
                # Simple filter matching
                if self._matches_filters(obj):
                    return obj
        return None

    def all(self):
        """Mock all operation."""
        model_name = self.model.__name__
        if model_name in self.db.data:
            return [
                obj
                for obj in self.db.data[model_name].values()
                if self._matches_filters(obj)
            ]
        return []

    def count(self):
        """Mock count operation."""
        return len(self.all())

    def delete(self):
        """Mock delete operation."""
        to_delete = self.all()
        model_name = self.model.__name__
        for obj in to_delete:
            if hasattr(obj, 'id') and obj.id in self.db.data.get(
                model_name, {}
            ):
                del self.db.data[model_name][obj.id]
        return len(to_delete)

    def _matches_filters(self, obj):
        """Check if object matches filters."""
        for filter_item in self.filters:
            if isinstance(filter_item, dict):
                # filter_by style
                for key, value in filter_item.items():
                    if not hasattr(obj, key) or getattr(obj, key) != value:
                        return False
        return True


def mock_db_session():
    """Create a mock database session for testing."""
    return MockDatabase()
