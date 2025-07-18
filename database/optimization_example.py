"""
Example Usage of Database Query Optimization for Multi-Agent Systems

This example demonstrates how to use the optimization features:
1. Setting up optimized database connections
2. Creating and managing indexes
3. Using batch operations
4. Monitoring query performance
5. Implementing prepared statements
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from uuid import uuid4

from database.indexing_strategy import get_indexing_strategy
from database.query_optimizer import get_query_optimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_optimized_database():
    """Setup database with all optimizations enabled."""
    # Database URL - configure for your environment
    # For production with PgBouncer:
    # database_url = "postgresql://user:pass@pgbouncer:6432/dbname"

    # For direct connection:
    database_url = "postgresql://user:pass@localhost:5432/dbname"

    # Initialize query optimizer
    optimizer = get_query_optimizer(database_url, enable_pgbouncer=False)
    indexing_strategy = get_indexing_strategy()

    logger.info("Setting up optimized database...")

    async with optimizer.optimized_session() as session:
        # 1. Create multi-agent optimized indexes
        logger.info("Creating multi-agent indexes...")
        await optimizer.create_multi_agent_indexes(session)

        # 2. Apply indexing strategy recommendations
        logger.info("Analyzing and applying indexing recommendations...")
        report = await indexing_strategy.generate_indexing_report(session)

        # Log recommendations
        logger.info(f"Found {len(report['missing_indexes'])} missing indexes")
        logger.info(
            f"Found {len(report['redundant_indexes'])} redundant indexes"
        )

        # Apply recommendations (with manual approval in production)
        await indexing_strategy.apply_recommendations(
            session, auto_approve=False
        )

        # 3. Setup monitoring
        logger.info("Setting up database monitoring...")
        await optimizer.setup_monitoring(session)

        # 4. Analyze and vacuum tables
        logger.info("Running ANALYZE and VACUUM...")
        await optimizer.analyze_and_vacuum_tables(session)

    return optimizer, indexing_strategy


async def example_batch_operations(optimizer):
    """Example of using batch operations for better performance."""
    logger.info("\n=== Batch Operations Example ===")

    async with optimizer.optimized_session() as session:
        batch_manager = optimizer.batch_manager

        # Example 1: Batch insert agents
        agents_to_insert = []
        for i in range(1000):
            agents_to_insert.append(
                {
                    "id": str(uuid4()),
                    "name": f"Agent_{i}",
                    "template": "explorer",
                    "status": "active",
                    "created_at": datetime.now(),
                    "inference_count": 0,
                    "total_steps": 0,
                    "beliefs": json.dumps({"initial": True}),
                    "preferences": json.dumps({"exploration": 0.8}),
                    "metrics": json.dumps({"performance": 0.0}),
                }
            )

        # Batch insert
        inserted = await batch_manager.batch_insert(
            session, "agents", agents_to_insert
        )
        logger.info(f"Batch inserted {inserted} agents")

        # Example 2: Batch update agent activities
        updates = []
        for agent in agents_to_insert[:100]:  # Update first 100
            updates.append(
                {
                    "id": agent["id"],
                    "inference_count": 10,
                    "total_steps": 100,
                    "last_active": datetime.now(),
                }
            )

        # Batch update
        updated = await batch_manager.batch_update(session, "agents", updates)
        logger.info(f"Batch updated {updated} agents")

        # Example 3: Using pending operations buffer
        for i in range(50):
            batch_manager.add_pending_insert(
                "performance_metrics",
                {
                    "id": str(uuid4()),
                    "test_run_id": str(uuid4()),
                    "metric_type": "agent_performance",
                    "metric_name": f"metric_{i}",
                    "value": i * 0.1,
                    "timestamp": datetime.now(),
                },
            )

        # Flush all pending operations
        flush_results = await batch_manager.flush_pending_operations(session)
        logger.info(f"Flushed pending operations: {flush_results}")


async def example_prepared_statements(optimizer):
    """Example of using prepared statements for repeated queries."""
    logger.info("\n=== Prepared Statements Example ===")

    async with optimizer.optimized_session() as session:
        prep_manager = optimizer.prepared_statements

        # Register frequently used queries
        queries = {
            "find_active_agents": {
                "sql": """
                    SELECT id, name, last_active, inference_count
                    FROM agents
                    WHERE status = :status AND template = :template
                    ORDER BY last_active DESC
                    LIMIT :limit
                """,
                "params": {
                    "status": "active",
                    "template": "explorer",
                    "limit": 10,
                },
            },
            "coalition_stats": {
                "sql": """
                    SELECT
                        c.id,
                        c.name,
                        COUNT(ac.agent_id) as member_count,
                        AVG(ac.contribution_score) as avg_contribution
                    FROM coalitions c
                    LEFT JOIN agent_coalition ac ON c.id = ac.coalition_id
                    WHERE c.status = :status
                    GROUP BY c.id, c.name
                    HAVING COUNT(ac.agent_id) > :min_members
                """,
                "params": {"status": "active", "min_members": 0},
            },
        }

        # Register and prepare statements
        for name, query_info in queries.items():
            stmt_name = prep_manager.register_statement(
                name, query_info["sql"], query_info["params"]
            )

            # Prepare the statement in database
            prepare_sql = f"PREPARE {stmt_name} AS {query_info['sql']}"
            await session.execute(text(prepare_sql))
            await session.commit()

            logger.info(f"Prepared statement: {name}")

        # Use prepared statements multiple times
        for i in range(5):
            # Execute prepared statement
            stmt = prep_manager.get_statement("find_active_agents")
            if stmt:
                result = await session.execute(
                    text(
                        f"EXECUTE {stmt['name']} (:status, :template, :limit)"
                    ),
                    {"status": "active", "template": "explorer", "limit": 10},
                )
                agents = result.fetchall()
                logger.info(
                    f"Iteration {i + 1}: Found {len(agents)} active agents"
                )

        # Get performance stats
        stats = prep_manager.get_performance_stats()
        logger.info(f"Prepared statement stats: {json.dumps(stats, indent=2)}")


async def example_query_caching(optimizer):
    """Example of using query result caching."""
    logger.info("\n=== Query Caching Example ===")

    async with optimizer.optimized_session() as session:
        # Example query that benefits from caching
        query = """
            SELECT
                template,
                COUNT(*) as count,
                AVG(inference_count) as avg_inferences
            FROM agents
            WHERE status = 'active'
            GROUP BY template
        """

        # First execution (cache miss)
        start = datetime.now()
        result1 = await optimizer.execute_with_cache(
            session, query, cache_key="agent_stats", ttl=300
        )
        time1 = (datetime.now() - start).total_seconds()
        logger.info(f"First execution (cache miss): {time1:.3f}s")

        # Second execution (cache hit)
        start = datetime.now()
        result2 = await optimizer.execute_with_cache(
            session, query, cache_key="agent_stats", ttl=300
        )
        time2 = (datetime.now() - start).total_seconds()
        logger.info(f"Second execution (cache hit): {time2:.3f}s")

        # Cache performance
        logger.info(f"Cache speedup: {time1 / time2:.1f}x faster")


async def example_query_analysis(optimizer):
    """Example of analyzing query performance."""
    logger.info("\n=== Query Analysis Example ===")

    async with optimizer.optimized_session() as session:
        analyzer = optimizer.query_analyzer

        # Analyze different query patterns
        queries = [
            {
                "name": "without_index",
                "sql": """
                    SELECT * FROM agents
                    WHERE JSON_EXTRACT(beliefs, '$.confidence') > 0.5
                """,
            },
            {
                "name": "with_index",
                "sql": """
                    SELECT * FROM agents
                    WHERE status = 'active'
                    ORDER BY last_active DESC
                    LIMIT 100
                """,
            },
        ]

        for query_info in queries:
            logger.info(f"\nAnalyzing query: {query_info['name']}")

            # Analyze execution plan
            plan = await analyzer.analyze_query(session, query_info["sql"])

            logger.info(
                f"Execution time: {plan.get('execution_time', 'N/A')}ms"
            )
            logger.info(f"Sequential scans: {plan.get('seq_scans', 0)}")
            logger.info(f"Index scans: {plan.get('index_scans', 0)}")

            # Get optimization suggestions
            query_hash = hashlib.md5(query_info["sql"].encode()).hexdigest()
            suggestions = analyzer.optimization_suggestions.get(
                query_hash, set()
            )

            if suggestions:
                logger.info("Optimization suggestions:")
                for suggestion in suggestions:
                    logger.info(f"  - {suggestion}")


async def example_maintenance_scheduling(indexing_strategy):
    """Example of scheduling index maintenance."""
    logger.info("\n=== Maintenance Scheduling Example ===")

    async with indexing_strategy.maintenance_scheduler as scheduler:
        # Get maintenance schedule
        schedule = await scheduler.get_maintenance_schedule(session)

        logger.info("Maintenance schedule:")
        for task in schedule:
            logger.info(
                f"  {task['task']}: Priority={task['priority']}, Recommended time={task['recommended_time']}"
            )

        # Check for index bloat
        bloated_indexes = await scheduler.check_index_bloat(session)

        if bloated_indexes:
            logger.info(f"\nFound {len(bloated_indexes)} bloated indexes:")
            for idx in bloated_indexes[:5]:  # Show first 5
                logger.info(f"  {idx['index']}: {idx['bloat_ratio']} bloat")

        # Perform maintenance (example - only ANALYZE)
        result = await scheduler.perform_maintenance(
            session, "ANALYZE", ["agents", "coalitions"]
        )
        logger.info(
            f"\nMaintenance result: {result['task']} - Success: {result['success']}"
        )


async def generate_performance_report(optimizer, indexing_strategy):
    """Generate comprehensive performance report."""
    logger.info("\n=== Performance Report ===")

    # Get optimizer performance report
    perf_report = optimizer.get_performance_report()

    logger.info("\nQuery Statistics:")
    for query_type, stats in perf_report["query_statistics"].items():
        if stats["count"] > 0:
            avg_time = stats["total_time"] / stats["count"]
            logger.info(
                f"  {query_type}: {stats['count']} queries, avg time: {avg_time:.3f}s"
            )

    logger.info(
        f"\nCache hit rate: {perf_report['cache_statistics']['hit_rate']:.1f}%"
    )

    logger.info("\nSlow queries:")
    for query in perf_report["slow_queries"][:5]:  # Show top 5
        logger.info(f"  {query['query_name']}: {query['query_time']:.3f}s")

    # Get indexing report
    async with optimizer.optimized_session() as session:
        index_report = await indexing_strategy.generate_indexing_report(
            session
        )

        logger.info(f"\nIndex usage summary:")
        logger.info(
            f"  Total indexes: {index_report['index_usage']['total_indexes']}"
        )
        logger.info(
            f"  Unused indexes: {len(index_report['index_usage']['unused_indexes'])}"
        )
        logger.info(
            f"  Missing indexes: {len(index_report['missing_indexes'])}"
        )
        logger.info(
            f"  Redundant indexes: {len(index_report['redundant_indexes'])}"
        )


async def main():
    """Main example execution."""
    logger.info("Starting Database Optimization Examples...")

    # Setup optimized database
    optimizer, indexing_strategy = await setup_optimized_database()

    # Run examples
    await example_batch_operations(optimizer)
    await example_prepared_statements(optimizer)
    await example_query_caching(optimizer)
    await example_query_analysis(optimizer)

    # Generate performance report
    await generate_performance_report(optimizer, indexing_strategy)

    logger.info("\nDatabase optimization examples completed!")


if __name__ == "__main__":
    # Configure these for your environment
    import os

    # Example environment variables
    os.environ[
        "DATABASE_URL"
    ] = "postgresql://user:pass@localhost:5432/freeagentics"
    os.environ["ENABLE_PGBOUNCER"] = "false"

    asyncio.run(main())
