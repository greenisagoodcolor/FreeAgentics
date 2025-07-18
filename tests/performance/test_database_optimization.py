"""
Performance Tests for Database Query Optimization

Tests the effectiveness of database optimizations including:
- Index performance improvements
- Query execution time comparisons
- Connection pooling efficiency
- Batch operation performance
- Concurrent query handling
"""

import asyncio
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from database.indexing_strategy import get_indexing_strategy
from database.models import (
    Agent,
    AgentRole,
    AgentStatus,
    Coalition,
    CoalitionStatus,
)
from database.query_optimizer import (
    BatchOperationManager,
    EnhancedQueryOptimizer,
    PreparedStatementManager,
    QueryPlanAnalyzer,
    get_query_optimizer,
)

logger = logging.getLogger(__name__)


class DatabaseOptimizationBenchmark:
    """Comprehensive benchmark suite for database optimizations."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.optimizer = get_query_optimizer(database_url)
        self.indexing_strategy = get_indexing_strategy()
        self.benchmark_results = {}

        # Test data configuration
        self.num_agents = 10000
        self.num_coalitions = 1000
        self.num_relationships = 50000

    async def setup_test_data(self, session: AsyncSession):
        """Create test data for benchmarking."""
        logger.info("Setting up test data...")

        # Create agents in batches
        batch_manager = BatchOperationManager(batch_size=1000)

        agents = []
        for i in range(self.num_agents):
            agent_data = {
                "id": str(uuid4()),
                "name": f"Agent_{i}",
                "template": random.choice(
                    ["explorer", "analyzer", "coordinator", "observer"]
                ),
                "status": random.choice(list(AgentStatus)).value,
                "created_at": datetime.now()
                - timedelta(days=random.randint(0, 365)),
                "last_active": datetime.now()
                - timedelta(hours=random.randint(0, 72)),
                "inference_count": random.randint(0, 10000),
                "total_steps": random.randint(0, 100000),
                "beliefs": json.dumps(
                    {
                        "state": random.randint(0, 100),
                        "confidence": random.random(),
                    }
                ),
                "preferences": json.dumps(
                    {
                        "goals": [
                            f"goal_{j}" for j in range(random.randint(1, 5))
                        ]
                    }
                ),
                "metrics": json.dumps(
                    {
                        "performance": random.random(),
                        "efficiency": random.random(),
                    }
                ),
            }
            agents.append(agent_data)

        # Batch insert agents
        await batch_manager.batch_insert(session, "agents", agents)

        # Create coalitions
        coalitions = []
        for i in range(self.num_coalitions):
            coalition_data = {
                "id": str(uuid4()),
                "name": f"Coalition_{i}",
                "status": random.choice(list(CoalitionStatus)).value,
                "created_at": datetime.now()
                - timedelta(days=random.randint(0, 365)),
                "performance_score": random.random(),
                "cohesion_score": random.random(),
                "objectives": json.dumps(
                    {
                        "primary": f"objective_{i}",
                        "secondary": [f"sub_{j}" for j in range(3)],
                    }
                ),
                "required_capabilities": json.dumps(
                    [f"capability_{j}" for j in range(random.randint(2, 5))]
                ),
            }
            coalitions.append(coalition_data)

        # Batch insert coalitions
        await batch_manager.batch_insert(session, "coalitions", coalitions)

        # Create agent-coalition relationships
        relationships = []
        used_pairs = set()

        for _ in range(self.num_relationships):
            agent_id = random.choice(agents)["id"]
            coalition_id = random.choice(coalitions)["id"]

            # Ensure unique pairs
            pair = (agent_id, coalition_id)
            if pair not in used_pairs:
                used_pairs.add(pair)
                relationship_data = {
                    "agent_id": agent_id,
                    "coalition_id": coalition_id,
                    "role": random.choice(list(AgentRole)).value,
                    "joined_at": datetime.now()
                    - timedelta(days=random.randint(0, 30)),
                    "contribution_score": random.random(),
                    "trust_score": random.random(),
                }
                relationships.append(relationship_data)

        # Batch insert relationships
        await batch_manager.batch_insert(
            session, "agent_coalition", relationships
        )

        logger.info(
            f"Created {len(agents)} agents, {len(coalitions)} coalitions, {len(relationships)} relationships"
        )

    async def benchmark_query_performance(
        self, session: AsyncSession
    ) -> Dict[str, Any]:
        """Benchmark various query patterns before and after optimization."""
        results = {
            "before_optimization": {},
            "after_optimization": {},
            "improvements": {},
        }

        # Test queries
        test_queries = [
            {
                "name": "active_agents_lookup",
                "query": """
                    SELECT id, name, status, last_active, inference_count
                    FROM agents
                    WHERE status = 'active'
                    ORDER BY last_active DESC
                    LIMIT 100
                """,
            },
            {
                "name": "coalition_members",
                "query": """
                    SELECT a.id, a.name, ac.role, ac.contribution_score
                    FROM agents a
                    JOIN agent_coalition ac ON a.id = ac.agent_id
                    WHERE ac.coalition_id = :coalition_id
                    ORDER BY ac.contribution_score DESC
                """,
                "params": {
                    "coalition_id": str(uuid4())
                },  # Will be replaced with actual ID
            },
            {
                "name": "agent_search",
                "query": """
                    SELECT id, name, template, status
                    FROM agents
                    WHERE name LIKE :search_term
                    ORDER BY created_at DESC
                    LIMIT 50
                """,
                "params": {"search_term": "Agent_1%"},
            },
            {
                "name": "performance_aggregation",
                "query": """
                    SELECT
                        c.id,
                        c.name,
                        COUNT(ac.agent_id) as member_count,
                        AVG(ac.contribution_score) as avg_contribution,
                        MAX(ac.contribution_score) as max_contribution
                    FROM coalitions c
                    JOIN agent_coalition ac ON c.id = ac.coalition_id
                    WHERE c.status = 'active'
                    GROUP BY c.id, c.name
                    ORDER BY avg_contribution DESC
                    LIMIT 20
                """,
            },
            {
                "name": "time_series_query",
                "query": """
                    SELECT
                        DATE_TRUNC('hour', last_active) as hour,
                        COUNT(*) as active_agents,
                        AVG(inference_count) as avg_inferences
                    FROM agents
                    WHERE last_active > NOW() - INTERVAL '24 hours'
                    GROUP BY hour
                    ORDER BY hour DESC
                """,
            },
        ]

        # Get a real coalition ID for testing
        coalition_result = await session.execute(
            text("SELECT id FROM coalitions LIMIT 1")
        )
        coalition_id = coalition_result.scalar()

        # Run benchmarks before optimization
        logger.info("Running queries before optimization...")
        for test in test_queries:
            query = test["query"]
            params = test.get("params", {})

            # Update coalition_id if needed
            if "coalition_id" in params:
                params["coalition_id"] = str(coalition_id)

            # Measure execution time
            start_time = time.time()
            for _ in range(5):  # Run 5 times for average
                await session.execute(text(query), params)
            execution_time = (time.time() - start_time) / 5

            results["before_optimization"][test["name"]] = execution_time

            # Analyze query plan
            analyzer = QueryPlanAnalyzer()
            plan = await analyzer.analyze_query(session, query, params)
            results["before_optimization"][f"{test['name']}_plan"] = plan

        # Apply optimizations
        logger.info("Applying database optimizations...")
        await self.apply_optimizations(session)

        # Run benchmarks after optimization
        logger.info("Running queries after optimization...")
        for test in test_queries:
            query = test["query"]
            params = test.get("params", {})

            # Update coalition_id if needed
            if "coalition_id" in params:
                params["coalition_id"] = str(coalition_id)

            # Measure execution time
            start_time = time.time()
            for _ in range(5):  # Run 5 times for average
                await session.execute(text(query), params)
            execution_time = (time.time() - start_time) / 5

            results["after_optimization"][test["name"]] = execution_time

            # Analyze query plan
            analyzer = QueryPlanAnalyzer()
            plan = await analyzer.analyze_query(session, query, params)
            results["after_optimization"][f"{test['name']}_plan"] = plan

        # Calculate improvements
        for query_name in results["before_optimization"]:
            if "_plan" not in query_name:
                before = results["before_optimization"][query_name]
                after = results["after_optimization"][query_name]
                improvement = (
                    ((before - after) / before) * 100 if before > 0 else 0
                )
                results["improvements"][query_name] = {
                    "before_ms": before * 1000,
                    "after_ms": after * 1000,
                    "improvement_percent": improvement,
                }

        return results

    async def apply_optimizations(self, session: AsyncSession):
        """Apply all database optimizations."""
        # Create multi-agent indexes
        await self.optimizer.create_multi_agent_indexes(session)

        # Apply indexing strategy recommendations
        await self.indexing_strategy.apply_recommendations(
            session, auto_approve=True
        )

        # Analyze and vacuum tables
        await self.optimizer.analyze_and_vacuum_tables(session)

    async def benchmark_connection_pooling(self) -> Dict[str, Any]:
        """Benchmark connection pooling performance."""
        results = {
            "without_pooling": {},
            "with_pooling": {},
            "pgbouncer_pooling": {},
        }

        # Test configurations
        configs = [
            {
                "name": "without_pooling",
                "engine": create_async_engine(
                    self.database_url.replace(
                        "postgresql://", "postgresql+asyncpg://"
                    ),
                    pool_size=1,
                    max_overflow=0,
                ),
            },
            {
                "name": "with_pooling",
                "engine": create_async_engine(
                    self.database_url.replace(
                        "postgresql://", "postgresql+asyncpg://"
                    ),
                    pool_size=20,
                    max_overflow=10,
                ),
            },
        ]

        # Test concurrent connections
        async def run_concurrent_queries(engine, num_queries=100):
            async def single_query():
                async with AsyncSession(engine) as session:
                    result = await session.execute(
                        text("SELECT COUNT(*) FROM agents")
                    )
                    return result.scalar()

            start_time = time.time()
            tasks = [single_query() for _ in range(num_queries)]
            await asyncio.gather(*tasks)
            return time.time() - start_time

        for config in configs:
            logger.info(f"Testing {config['name']}...")
            execution_time = await run_concurrent_queries(config["engine"])
            results[config["name"]] = {
                "total_time": execution_time,
                "queries_per_second": 100 / execution_time,
            }

            # Cleanup
            await config["engine"].dispose()

        return results

    async def benchmark_batch_operations(
        self, session: AsyncSession
    ) -> Dict[str, Any]:
        """Benchmark batch operation performance."""
        results = {
            "individual_inserts": {},
            "batch_inserts": {},
            "individual_updates": {},
            "batch_updates": {},
        }

        batch_manager = BatchOperationManager()

        # Test data
        test_agents = []
        for i in range(1000):
            test_agents.append(
                {
                    "id": str(uuid4()),
                    "name": f"BatchTest_{i}",
                    "template": "test_template",
                    "status": "active",
                    "created_at": datetime.now(),
                    "inference_count": 0,
                    "total_steps": 0,
                }
            )

        # Benchmark individual inserts
        start_time = time.time()
        for agent in test_agents[:100]:  # Test with 100 records
            await session.execute(
                text(
                    """
                    INSERT INTO agents (id, name, template, status, created_at, inference_count, total_steps)
                    VALUES (:id, :name, :template, :status, :created_at, :inference_count, :total_steps)
                """
                ),
                agent,
            )
        await session.commit()
        results["individual_inserts"]["time"] = time.time() - start_time
        results["individual_inserts"]["records_per_second"] = (
            100 / results["individual_inserts"]["time"]
        )

        # Benchmark batch inserts
        start_time = time.time()
        await batch_manager.batch_insert(
            session, "agents", test_agents[100:1000]
        )  # 900 records
        results["batch_inserts"]["time"] = time.time() - start_time
        results["batch_inserts"]["records_per_second"] = (
            900 / results["batch_inserts"]["time"]
        )

        # Prepare update data
        update_data = []
        for agent in test_agents[:100]:
            update_data.append(
                {
                    "id": agent["id"],
                    "inference_count": random.randint(1, 100),
                    "total_steps": random.randint(100, 1000),
                }
            )

        # Benchmark individual updates
        start_time = time.time()
        for update in update_data:
            await session.execute(
                text(
                    """
                    UPDATE agents
                    SET inference_count = :inference_count, total_steps = :total_steps, updated_at = NOW()
                    WHERE id = :id
                """
                ),
                update,
            )
        await session.commit()
        results["individual_updates"]["time"] = time.time() - start_time
        results["individual_updates"]["records_per_second"] = (
            100 / results["individual_updates"]["time"]
        )

        # Benchmark batch updates
        batch_update_data = []
        for agent in test_agents[100:1000]:
            batch_update_data.append(
                {
                    "id": agent["id"],
                    "inference_count": random.randint(1, 100),
                    "total_steps": random.randint(100, 1000),
                }
            )

        start_time = time.time()
        await batch_manager.batch_update(session, "agents", batch_update_data)
        results["batch_updates"]["time"] = time.time() - start_time
        results["batch_updates"]["records_per_second"] = (
            900 / results["batch_updates"]["time"]
        )

        # Calculate improvements
        results["improvements"] = {
            "insert_speedup": results["batch_inserts"]["records_per_second"]
            / results["individual_inserts"]["records_per_second"],
            "update_speedup": results["batch_updates"]["records_per_second"]
            / results["individual_updates"]["records_per_second"],
        }

        return results

    async def benchmark_prepared_statements(
        self, session: AsyncSession
    ) -> Dict[str, Any]:
        """Benchmark prepared statement performance."""
        results = {"without_prepared": {}, "with_prepared": {}}

        prep_manager = PreparedStatementManager()

        # Test query
        query = """
            SELECT id, name, status, inference_count
            FROM agents
            WHERE template = :template AND status = :status
            ORDER BY inference_count DESC
            LIMIT 10
        """

        params = {"template": "explorer", "status": "active"}

        # Benchmark without prepared statements
        start_time = time.time()
        for _ in range(100):
            await session.execute(text(query), params)
        results["without_prepared"]["time"] = time.time() - start_time
        results["without_prepared"]["queries_per_second"] = (
            100 / results["without_prepared"]["time"]
        )

        # Register prepared statement
        prep_name = prep_manager.register_statement(
            "agent_lookup", query, params
        )

        # Prepare the statement
        await session.execute(text(f"PREPARE {prep_name} AS {query}"))
        await session.commit()

        # Benchmark with prepared statements
        start_time = time.time()
        for _ in range(100):
            await session.execute(
                text(f"EXECUTE {prep_name} (:template, :status)"), params
            )
        results["with_prepared"]["time"] = time.time() - start_time
        results["with_prepared"]["queries_per_second"] = (
            100 / results["with_prepared"]["time"]
        )

        # Calculate improvement
        results["improvement_percent"] = (
            (
                results["with_prepared"]["queries_per_second"]
                - results["without_prepared"]["queries_per_second"]
            )
            / results["without_prepared"]["queries_per_second"]
            * 100
        )

        return results

    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run all benchmarks and generate comprehensive report."""
        logger.info(
            "Starting comprehensive database optimization benchmark..."
        )

        all_results = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "num_agents": self.num_agents,
                "num_coalitions": self.num_coalitions,
                "num_relationships": self.num_relationships,
            },
            "benchmarks": {},
        }

        async with self.optimizer.optimized_session() as session:
            # Setup test data
            await self.setup_test_data(session)

            # Run query performance benchmark
            logger.info("Running query performance benchmark...")
            all_results["benchmarks"][
                "query_performance"
            ] = await self.benchmark_query_performance(session)

            # Run batch operations benchmark
            logger.info("Running batch operations benchmark...")
            all_results["benchmarks"][
                "batch_operations"
            ] = await self.benchmark_batch_operations(session)

            # Run prepared statements benchmark
            logger.info("Running prepared statements benchmark...")
            all_results["benchmarks"][
                "prepared_statements"
            ] = await self.benchmark_prepared_statements(session)

        # Run connection pooling benchmark
        logger.info("Running connection pooling benchmark...")
        all_results["benchmarks"][
            "connection_pooling"
        ] = await self.benchmark_connection_pooling()

        # Generate performance report
        all_results[
            "optimizer_report"
        ] = self.optimizer.get_performance_report()

        # Calculate overall improvements
        all_results["summary"] = self._generate_summary(
            all_results["benchmarks"]
        )

        return all_results

    def _generate_summary(self, benchmarks: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of benchmark results."""
        summary = {
            "key_improvements": [],
            "recommendations": [],
            "performance_gains": {},
        }

        # Analyze query performance improvements
        if "query_performance" in benchmarks:
            query_improvements = benchmarks["query_performance"][
                "improvements"
            ]
            avg_improvement = sum(
                imp["improvement_percent"]
                for imp in query_improvements.values()
            ) / len(query_improvements)

            summary["performance_gains"][
                "average_query_improvement"
            ] = f"{avg_improvement:.1f}%"

            for query, improvement in query_improvements.items():
                if improvement["improvement_percent"] > 20:
                    summary["key_improvements"].append(
                        f"{query}: {improvement['improvement_percent']:.1f}% faster"
                    )

        # Analyze batch operation improvements
        if "batch_operations" in benchmarks:
            batch_improvements = benchmarks["batch_operations"]["improvements"]
            summary["performance_gains"][
                "insert_speedup"
            ] = f"{batch_improvements['insert_speedup']:.1f}x"
            summary["performance_gains"][
                "update_speedup"
            ] = f"{batch_improvements['update_speedup']:.1f}x"

        # Generate recommendations
        if avg_improvement < 10:
            summary["recommendations"].append(
                "Consider additional indexes or query restructuring for better performance"
            )

        if batch_improvements["insert_speedup"] < 5:
            summary["recommendations"].append(
                "Increase batch size for insert operations to improve throughput"
            )

        return summary


@pytest.mark.asyncio
async def test_database_optimization_comprehensive():
    """Run comprehensive database optimization tests."""
    database_url = "postgresql://test_user:test_pass@localhost:5432/test_db"

    benchmark = DatabaseOptimizationBenchmark(database_url)
    results = await benchmark.run_comprehensive_benchmark()

    # Assert improvements
    assert results["benchmarks"]["query_performance"]["improvements"]
    assert (
        results["benchmarks"]["batch_operations"]["improvements"][
            "insert_speedup"
        ]
        > 1
    )
    assert (
        results["benchmarks"]["batch_operations"]["improvements"][
            "update_speedup"
        ]
        > 1
    )

    # Log results
    logger.info(
        f"Benchmark completed: {json.dumps(results['summary'], indent=2)}"
    )

    return results


@pytest.mark.asyncio
async def test_index_effectiveness():
    """Test the effectiveness of created indexes."""
    database_url = "postgresql://test_user:test_pass@localhost:5432/test_db"
    optimizer = get_query_optimizer(database_url)

    async with optimizer.optimized_session() as session:
        # Create indexes
        await optimizer.create_multi_agent_indexes(session)

        # Test index usage
        test_queries = [
            "SELECT * FROM agents WHERE status = 'active' ORDER BY last_active DESC LIMIT 10",
            "SELECT * FROM coalitions WHERE performance_score > 0.8",
            "SELECT * FROM agent_coalition WHERE coalition_id = '123' ORDER BY contribution_score DESC",
        ]

        for query in test_queries:
            # Analyze query plan
            analyzer = QueryPlanAnalyzer()
            plan = await analyzer.analyze_query(session, query)

            # Assert index usage
            assert plan["index_scans"] > 0, f"Query not using indexes: {query}"
            assert (
                plan["seq_scans"] == 0
            ), f"Query using sequential scan: {query}"


@pytest.mark.asyncio
async def test_concurrent_query_handling():
    """Test database performance under concurrent load."""
    database_url = "postgresql://test_user:test_pass@localhost:5432/test_db"
    optimizer = get_query_optimizer(database_url)

    async def concurrent_query_task(query_id: int):
        async with optimizer.optimized_session() as session:
            # Run various queries
            queries = [
                "SELECT COUNT(*) FROM agents WHERE status = 'active'",
                "SELECT AVG(performance_score) FROM coalitions",
                "SELECT COUNT(*) FROM agent_coalition WHERE role = 'leader'",
            ]

            query = queries[query_id % len(queries)]
            start_time = time.time()

            result = await session.execute(text(query))
            value = result.scalar()

            return {
                "query_id": query_id,
                "query": query,
                "result": value,
                "execution_time": time.time() - start_time,
            }

    # Run 100 concurrent queries
    tasks = [concurrent_query_task(i) for i in range(100)]
    results = await asyncio.gather(*tasks)

    # Analyze results
    execution_times = [r["execution_time"] for r in results]
    avg_time = sum(execution_times) / len(execution_times)
    max_time = max(execution_times)

    # Assert performance criteria
    assert avg_time < 0.1, f"Average query time too high: {avg_time:.3f}s"
    assert max_time < 0.5, f"Maximum query time too high: {max_time:.3f}s"

    logger.info(
        f"Concurrent query test: avg={avg_time:.3f}s, max={max_time:.3f}s"
    )


if __name__ == "__main__":
    # Run benchmarks directly
    import asyncio

    async def main():
        database_url = (
            "postgresql://test_user:test_pass@localhost:5432/test_db"
        )
        benchmark = DatabaseOptimizationBenchmark(database_url)
        results = await benchmark.run_comprehensive_benchmark()

        # Save results to file
        with open("database_optimization_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print(
            f"Benchmark completed. Results saved to database_optimization_results.json"
        )
        print(f"\nSummary: {json.dumps(results['summary'], indent=2)}")

    asyncio.run(main())
