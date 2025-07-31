#!/usr/bin/env python3
"""Create database tables directly from models."""

import os
import sys

from sqlalchemy import inspect

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set database URL for SQLite
os.environ["DATABASE_URL"] = "sqlite:///freeagentics.db"

# Import database components after path setup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database.base import Base  # noqa: E402
from database.session import engine  # noqa: E402

# Import models to ensure they're registered with Base.metadata
try:
    import database.models  # noqa: E402 F401 - Required for model registration
except ImportError:
    pass


def main():
    """Create all tables."""
    print("Creating database tables...")

    # Create all tables
    Base.metadata.create_all(bind=engine)

    print("Tables created successfully!")

    # List tables
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"\nCreated tables: {', '.join(tables)}")


if __name__ == "__main__":
    main()
