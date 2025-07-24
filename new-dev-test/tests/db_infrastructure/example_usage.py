"""Example usage of the PostgreSQL test infrastructure."""

import logging

from tests.db_infrastructure import (
    DatabasePool,
    DatabaseReset,
    PerformanceMonitor,
    TestDataGenerator,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    """Demonstrate test infrastructure usage."""

    # 1. Reset database
    print("1. Resetting database...")
    db_reset = DatabaseReset()
    db_reset.reset_database("freeagentics_test")

    # 2. Generate test data
    print("\n2. Generating test data...")
    generator = TestDataGenerator(seed=42)  # Use seed for reproducibility
    test_data = generator.generate_complete_dataset(
        num_agents=50,
        num_coalitions=10,
        num_knowledge_nodes=500,
        num_edges=1000,
    )
    print(f"Generated {len(test_data['agents'])} agents")

    # 3. Get connection pool
    print("\n3. Setting up connection pool...")
    pool = DatabasePool(database="freeagentics_test", min_connections=2, max_connections=10)

    # 4. Insert test data
    print("\n4. Inserting test data...")
    with pool.get_connection() as (conn, cursor):
        # Insert a few agents as example
        for agent in test_data["agents"][:5]:
            cursor.execute(
                """
                INSERT INTO agents (name, template, status)
                VALUES (%s, %s, %s)
            """,
                (agent["name"], agent["template"], agent["status"]),
            )

        print(f"Inserted {cursor.rowcount} agents")

    # 5. Run performance monitoring
    print("\n5. Running performance test...")
    monitor = PerformanceMonitor("example_test")
    monitor.start_test_run({"example": True})

    # Simulate some operations
    with monitor.measure_operation("query", "select_agents"):
        results = pool.execute("SELECT * FROM agents WHERE status = 'ACTIVE'")
        print(f"Found {len(results)} active agents")

    # Get performance stats
    with monitor.monitored_connection("update_operation") as (conn, cursor):
        cursor.execute(
            """
            UPDATE agents
            SET last_active = NOW()
            WHERE status = 'ACTIVE'
        """
        )
        print(f"Updated {cursor.rowcount} agents")

    # End monitoring and get summary
    summary = monitor.end_test_run()
    print("\n6. Performance Summary:")
    print(f"Duration: {summary['duration']:.2f}s")
    print(f"Total operations: {summary['total_operations']}")

    # 7. Check pool stats
    print("\n7. Connection Pool Statistics:")
    pool_stats = pool.get_pool_stats()
    for key, value in pool_stats.items():
        print(f"  {key}: {value}")

    # 8. Verify schema
    print("\n8. Verifying schema...")
    verification = db_reset.verify_schema("freeagentics_test")
    print(f"Tables: {len(verification['tables'])}")
    print(f"Indexes: {len(verification['indexes'])}")
    print(f"Custom types: {len(verification['types'])}")

    # Clean up
    pool.close()
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
