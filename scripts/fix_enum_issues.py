#!/usr/bin/env python3
"""Fix SQLAlchemy enum issues in test database."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import create_engine, text
from database.models import AgentStatus, CoalitionStatus

# Test database URL
TEST_DATABASE_URL = "sqlite:///test.db"


def fix_enum_values():
    """Fix enum values in the database."""
    engine = create_engine(TEST_DATABASE_URL)
    
    with engine.connect() as conn:
        # Check if tables exist
        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = [row[0] for row in result]
        
        if 'agents' in tables:
            print("Fixing agent status values...")
            # Update any uppercase enum values to lowercase
            for status in AgentStatus:
                # Convert any stored enum names to values
                conn.execute(
                    text(f"UPDATE agents SET status = :value WHERE status = :name"),
                    {"value": status.value, "name": status.name}
                )
            conn.commit()
            
        if 'coalitions' in tables:
            print("Fixing coalition status values...")
            for status in CoalitionStatus:
                conn.execute(
                    text(f"UPDATE coalitions SET status = :value WHERE status = :name"),
                    {"value": status.value, "name": status.name}
                )
            conn.commit()
            
        print("Enum values fixed!")


if __name__ == "__main__":
    fix_enum_values()