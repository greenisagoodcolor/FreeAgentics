#!/usr/bin/env python3
"""Create database tables directly from models."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set database URL for SQLite
os.environ["DATABASE_URL"] = "sqlite:///freeagentics.db"

from database.base import Base
from database.session import engine

# Import other models to ensure they're registered
try:
    from database.models import Agent, Coalition, KnowledgeUnit, Observation
except ImportError:
    pass


def main():
    """Create all tables."""
    print("Creating database tables...")

    # Create all tables
    Base.metadata.create_all(bind=engine)

    print("Tables created successfully!")

    # List tables
    from sqlalchemy import inspect

    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"\nCreated tables: {', '.join(tables)}")


if __name__ == "__main__":
    main()
