#!/usr/bin/env python3
"""
Test Database Connection Script
Tests PostgreSQL connection and basic operations
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from database.connection_manager import DatabaseConnectionManager
from database.models import Agent, Coalition, KnowledgeNode


def test_basic_connection(database_url: str) -> bool:
    """Test basic database connection."""
    print("Testing basic database connection...")
    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.scalar()
            print(f"✓ Connected to PostgreSQL: {version}")
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


def test_connection_manager(database_url: str) -> bool:
    """Test connection manager with retry logic."""
    print("\nTesting connection manager...")
    try:
        manager = DatabaseConnectionManager(database_url)
        engine = manager.create_engine_with_pool_config()

        # Test session creation
        session = manager.get_db_session()
        result = session.execute(text("SELECT current_database()"))
        db_name = result.scalar()
        session.close()

        print(
            f"✓ Connection manager working - Connected to database: {db_name}"
        )
        return True
    except Exception as e:
        print(f"✗ Connection manager failed: {e}")
        return False


def test_table_existence(database_url: str) -> bool:
    """Check if required tables exist."""
    print("\nChecking database tables...")
    try:
        engine = create_engine(database_url)

        required_tables = [
            'agents',
            'coalitions',
            'agent_coalition',
            'db_knowledge_nodes',
            'db_knowledge_edges',
        ]

        with engine.connect() as conn:
            for table in required_tables:
                result = conn.execute(
                    text(
                        f"""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = '{table}'
                    )
                """
                    )
                )
                exists = result.scalar()
                if exists:
                    print(f"✓ Table '{table}' exists")
                else:
                    print(f"✗ Table '{table}' does not exist")
                    return False

        return True
    except Exception as e:
        print(f"✗ Error checking tables: {e}")
        return False


def test_crud_operations(database_url: str) -> bool:
    """Test basic CRUD operations."""
    print("\nTesting CRUD operations...")
    try:
        engine = create_engine(database_url)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Test creating an agent
        test_agent = Agent(
            name="Test Agent",
            template="grid_world",
            parameters={"test": True},
            metrics={"test_metric": 1.0},
        )
        session.add(test_agent)
        session.commit()
        print(f"✓ Created test agent with ID: {test_agent.id}")

        # Test reading the agent
        retrieved_agent = (
            session.query(Agent).filter_by(name="Test Agent").first()
        )
        if retrieved_agent:
            print(f"✓ Retrieved agent: {retrieved_agent.name}")
        else:
            print("✗ Failed to retrieve agent")
            return False

        # Test updating the agent
        retrieved_agent.metrics = {"test_metric": 2.0, "new_metric": 3.0}
        session.commit()
        print("✓ Updated agent metrics")

        # Test deleting the agent
        session.delete(retrieved_agent)
        session.commit()
        print("✓ Deleted test agent")

        session.close()
        return True
    except Exception as e:
        print(f"✗ CRUD operations failed: {e}")
        if 'session' in locals():
            session.rollback()
            session.close()
        return False


def test_connection_pooling(database_url: str) -> bool:
    """Test connection pooling."""
    print("\nTesting connection pooling...")
    try:
        manager = DatabaseConnectionManager(database_url)
        engine = manager.create_engine_with_pool_config(
            pool_size=5, max_overflow=10
        )

        # Create multiple sessions
        sessions = []
        for i in range(5):
            session = manager.get_db_session()
            sessions.append(session)
            print(f"✓ Created session {i+1}")

        # Close all sessions
        for i, session in enumerate(sessions):
            session.close()
            print(f"✓ Closed session {i+1}")

        print("✓ Connection pooling working correctly")
        return True
    except Exception as e:
        print(f"✗ Connection pooling failed: {e}")
        return False


def main():
    """Main test function."""
    # Load environment variables
    env_file = os.path.join(project_root, '.env.production')
    if os.path.exists(env_file):
        load_dotenv(env_file)
        print(f"Loaded environment from: {env_file}")
    else:
        load_dotenv()
        print("Using default .env file")

    # Get database URL
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("✗ DATABASE_URL not found in environment")
        sys.exit(1)

    print(
        f"Database URL: {database_url.replace(database_url.split('@')[0].split('//')[1].split(':')[1], '***')}"
    )
    print("=" * 50)

    # Run tests
    tests = [
        ("Basic Connection", test_basic_connection),
        ("Connection Manager", test_connection_manager),
        ("Table Existence", test_table_existence),
        ("CRUD Operations", test_crud_operations),
        ("Connection Pooling", test_connection_pooling),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func(database_url)
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} failed with unexpected error: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("=" * 50)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All database tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed. Please check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
