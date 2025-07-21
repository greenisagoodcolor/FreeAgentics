#!/usr/bin/env python3
"""Quick test to verify the database infrastructure works."""

import os
import sys

# Add parent directories to path
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)

from tests.db_infrastructure.db_reset import DatabaseReset
from tests.db_infrastructure.pool_config import DatabasePool


def test_connection():
    """Test basic database connection."""
    print("Testing database connection...")

    try:
        # Test with the existing freeagentics database
        pool = DatabasePool(
            host="localhost",
            port=5432,
            database="freeagentics",
            user="freeagentics",
            password="freeagentics123",
            min_connections=1,
            max_connections=5,
        )

        # Test query
        results = pool.execute("SELECT version()")
        print(f"✓ Connected to PostgreSQL: {results[0]['version']}")

        # Test table query
        results = pool.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """
        )

        print(f"✓ Found {len(results)} tables:")
        for row in results:
            print(f"  - {row['table_name']}")

        # Get pool stats
        stats = pool.get_pool_stats()
        print("\n✓ Connection pool stats:")
        for key, value in stats.items():
            print(f"  - {key}: {value}")

        pool.close()
        print("\n✓ Connection pool closed successfully")

        return True

    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_schema_verification():
    """Test schema verification."""
    print("\nTesting schema verification...")

    try:
        reset = DatabaseReset()
        verification = reset.verify_schema("freeagentics")

        print("✓ Schema verification results:")
        for category, items in verification.items():
            if items:
                print(f"\n  {category.upper()} ({len(items)}):")
                for name in list(items.keys())[:5]:  # Show first 5
                    print(f"    - {name}")
                if len(items) > 5:
                    print(f"    ... and {len(items) - 5} more")

        return True

    except Exception as e:
        print(f"✗ Schema verification failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_data_generation():
    """Test data generation."""
    print("\nTesting data generation...")

    try:
        from tests.db_infrastructure.data_generator import TestDataGenerator

        generator = TestDataGenerator(seed=42)

        # Generate sample data
        agent = generator.generate_agent()
        print(f"✓ Generated agent: {agent['name']} ({agent['template']})")

        coalition = generator.generate_coalition()
        print(f"✓ Generated coalition: {coalition['name']}")

        node = generator.generate_knowledge_node()
        print(f"✓ Generated knowledge node: {node['label']} ({node['type']})")

        # Generate dataset
        dataset = generator.generate_complete_dataset(
            num_agents=10,
            num_coalitions=2,
            num_knowledge_nodes=20,
            num_edges=10,
        )

        print("\n✓ Generated complete dataset:")
        for key, value in dataset.items():
            print(f"  - {key}: {len(value)} items")

        return True

    except Exception as e:
        print(f"✗ Data generation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=== FreeAgentics Database Infrastructure Test ===\n")

    tests = [
        ("Connection Test", test_connection),
        ("Schema Verification", test_schema_verification),
        ("Data Generation", test_data_generation),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        success = test_func()
        results.append((test_name, success))

    print(f"\n{'='*50}")
    print("\n=== Test Summary ===")
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(success for _, success in results)

    if all_passed:
        print("\n✓ All tests passed! Infrastructure is ready.")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
