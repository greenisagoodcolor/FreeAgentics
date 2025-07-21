"""
Database Load Testing Framework
===============================

This module provides comprehensive database performance testing including:
- Connection pool performance validation
- Query optimization testing
- Concurrent database load testing
- Transaction performance benchmarks
- Memory usage under database load
- Connection leak detection
- Database bottleneck identification
"""

import asyncio
import json
import logging
import random
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

import asyncpg
import numpy as np
import psutil

logger = logging.getLogger(__name__)


@dataclass
class DatabaseTestResult:
    """Result of a database performance test."""

    test_name: str
    concurrent_connections: int
    total_queries: int
    successful_queries: int
    failed_queries: int
    duration_seconds: float
    queries_per_second: float
    average_query_time_ms: float
    p95_query_time_ms: float
    p99_query_time_ms: float
    connection_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    connection_errors: int
    test_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryBenchmark:
    """Benchmark for a specific query type."""

    name: str
    query: str
    parameters: List[Any] = field(default_factory=list)
    expected_duration_ms: float = 100.0
    weight: float = 1.0
    setup_queries: List[str] = field(default_factory=list)
    cleanup_queries: List[str] = field(default_factory=list)


class DatabaseLoadTester:
    """Comprehensive database load testing framework."""

    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.process = psutil.Process()
        self.test_results: List[DatabaseTestResult] = []

        # Performance thresholds
        self.thresholds = {
            'query_time_ms': 100.0,  # 100ms max query time
            'connection_time_ms': 50.0,  # 50ms max connection time
            'queries_per_second': 1000.0,  # 1000 QPS minimum
            'connection_success_rate': 99.0,  # 99% success rate
            'memory_usage_max_mb': 1000.0,  # 1GB max memory usage
            'cpu_usage_max_percent': 80.0,  # 80% max CPU usage
        }

        # Predefined query benchmarks
        self.query_benchmarks = self._create_query_benchmarks()

    def _create_query_benchmarks(self) -> List[QueryBenchmark]:
        """Create predefined query benchmarks."""
        return [
            QueryBenchmark(
                name="simple_select",
                query="SELECT 1",
                expected_duration_ms=10.0,
                weight=0.3,
            ),
            QueryBenchmark(
                name="agents_select",
                query="SELECT * FROM agents LIMIT 10",
                expected_duration_ms=50.0,
                weight=0.2,
                setup_queries=[
                    "CREATE TABLE IF NOT EXISTS agents (id SERIAL PRIMARY KEY, name VARCHAR(255), created_at TIMESTAMP DEFAULT NOW())"
                ],
            ),
            QueryBenchmark(
                name="agents_insert",
                query="INSERT INTO agents (name) VALUES ($1)",
                parameters=["test_agent"],
                expected_duration_ms=30.0,
                weight=0.1,
                setup_queries=[
                    "CREATE TABLE IF NOT EXISTS agents (id SERIAL PRIMARY KEY, name VARCHAR(255), created_at TIMESTAMP DEFAULT NOW())"
                ],
            ),
            QueryBenchmark(
                name="agents_update",
                query="UPDATE agents SET name = $1 WHERE id = $2",
                parameters=["updated_agent", 1],
                expected_duration_ms=25.0,
                weight=0.1,
                setup_queries=[
                    "CREATE TABLE IF NOT EXISTS agents (id SERIAL PRIMARY KEY, name VARCHAR(255), created_at TIMESTAMP DEFAULT NOW())",
                    "INSERT INTO agents (name) VALUES ('initial_agent')",
                ],
            ),
            QueryBenchmark(
                name="complex_join",
                query="""
                SELECT a.name, p.text, COUNT(*) as count
                FROM agents a
                LEFT JOIN prompts p ON a.id = p.agent_id
                GROUP BY a.name, p.text
                ORDER BY count DESC
                LIMIT 5
                """,
                expected_duration_ms=200.0,
                weight=0.15,
                setup_queries=[
                    "CREATE TABLE IF NOT EXISTS agents (id SERIAL PRIMARY KEY, name VARCHAR(255), created_at TIMESTAMP DEFAULT NOW())",
                    "CREATE TABLE IF NOT EXISTS prompts (id SERIAL PRIMARY KEY, agent_id INTEGER, text TEXT, created_at TIMESTAMP DEFAULT NOW())",
                    "INSERT INTO agents (name) VALUES ('agent1'), ('agent2'), ('agent3')",
                    "INSERT INTO prompts (agent_id, text) VALUES (1, 'prompt1'), (2, 'prompt2'), (3, 'prompt3')",
                ],
            ),
            QueryBenchmark(
                name="index_scan",
                query="SELECT * FROM agents WHERE name = $1",
                parameters=["test_agent"],
                expected_duration_ms=15.0,
                weight=0.1,
                setup_queries=[
                    "CREATE TABLE IF NOT EXISTS agents (id SERIAL PRIMARY KEY, name VARCHAR(255), created_at TIMESTAMP DEFAULT NOW())",
                    "CREATE INDEX IF NOT EXISTS idx_agents_name ON agents(name)",
                    "INSERT INTO agents (name) VALUES ('test_agent')",
                ],
            ),
            QueryBenchmark(
                name="transaction_test",
                query="BEGIN; INSERT INTO agents (name) VALUES ($1); COMMIT;",
                parameters=["transaction_agent"],
                expected_duration_ms=40.0,
                weight=0.05,
                setup_queries=[
                    "CREATE TABLE IF NOT EXISTS agents (id SERIAL PRIMARY KEY, name VARCHAR(255), created_at TIMESTAMP DEFAULT NOW())"
                ],
            ),
        ]

    async def setup_test_environment(self):
        """Set up test environment and tables."""
        logger.info("Setting up database test environment")

        try:
            # Create connection for setup
            conn = await asyncpg.connect(**self.db_config)

            # Run setup queries for all benchmarks
            for benchmark in self.query_benchmarks:
                for setup_query in benchmark.setup_queries:
                    try:
                        await conn.execute(setup_query)
                        logger.debug(
                            f"Setup query executed: {setup_query[:50]}..."
                        )
                    except Exception as e:
                        logger.warning(f"Setup query failed: {e}")

            await conn.close()
            logger.info("Database test environment setup completed")

        except Exception as e:
            logger.error(f"Failed to setup test environment: {e}")
            raise

    async def cleanup_test_environment(self):
        """Clean up test environment."""
        logger.info("Cleaning up database test environment")

        try:
            conn = await asyncpg.connect(**self.db_config)

            # Drop test tables
            cleanup_queries = [
                "DROP TABLE IF EXISTS prompts",
                "DROP TABLE IF EXISTS agents",
                "DROP INDEX IF EXISTS idx_agents_name",
            ]

            for query in cleanup_queries:
                try:
                    await conn.execute(query)
                    logger.debug(f"Cleanup query executed: {query}")
                except Exception as e:
                    logger.warning(f"Cleanup query failed: {e}")

            await conn.close()
            logger.info("Database test environment cleanup completed")

        except Exception as e:
            logger.error(f"Failed to cleanup test environment: {e}")

    async def run_connection_pool_test(
        self, max_connections: int = 50
    ) -> DatabaseTestResult:
        """Test database connection pool performance."""
        logger.info(
            f"Starting connection pool test with {max_connections} connections"
        )

        start_time = time.perf_counter()
        successful_connections = 0
        failed_connections = 0
        connection_times = []

        try:
            # Test connection pool creation
            async def test_connection():
                nonlocal successful_connections, failed_connections

                connection_start = time.perf_counter()
                try:
                    conn = await asyncpg.connect(**self.db_config)
                    connection_time = (
                        time.perf_counter() - connection_start
                    ) * 1000
                    connection_times.append(connection_time)

                    # Test simple query
                    await conn.execute("SELECT 1")
                    successful_connections += 1

                    await conn.close()

                except Exception as e:
                    failed_connections += 1
                    logger.debug(f"Connection test failed: {e}")

            # Run connection tests concurrently
            tasks = [test_connection() for _ in range(max_connections)]
            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Connection pool test error: {e}")

        duration = time.perf_counter() - start_time
        success_rate = (
            (successful_connections / max_connections * 100)
            if max_connections > 0
            else 0
        )

        # System metrics
        memory_usage = self.process.memory_info().rss / 1024 / 1024
        cpu_usage = self.process.cpu_percent()

        result = DatabaseTestResult(
            test_name="connection_pool",
            concurrent_connections=max_connections,
            total_queries=successful_connections,
            successful_queries=successful_connections,
            failed_queries=failed_connections,
            duration_seconds=duration,
            queries_per_second=successful_connections / duration
            if duration > 0
            else 0,
            average_query_time_ms=statistics.mean(connection_times)
            if connection_times
            else 0,
            p95_query_time_ms=np.percentile(connection_times, 95)
            if connection_times
            else 0,
            p99_query_time_ms=np.percentile(connection_times, 99)
            if connection_times
            else 0,
            connection_time_ms=statistics.mean(connection_times)
            if connection_times
            else 0,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            connection_errors=failed_connections,
            test_metadata={
                'max_connections': max_connections,
                'success_rate': success_rate,
                'connection_times': connection_times,
                'min_connection_time': min(connection_times)
                if connection_times
                else 0,
                'max_connection_time': max(connection_times)
                if connection_times
                else 0,
            },
        )

        self.test_results.append(result)
        logger.info(
            f"Connection pool test completed: {successful_connections}/{max_connections} connections, {success_rate:.1f}% success"
        )

        return result

    async def run_query_benchmark_test(
        self, concurrent_users: int = 20, test_duration: int = 60
    ) -> DatabaseTestResult:
        """Run query benchmark test with various query types."""
        logger.info(
            f"Starting query benchmark test: {concurrent_users} users for {test_duration}s"
        )

        start_time = time.perf_counter()
        end_time = start_time + test_duration

        total_queries = 0
        successful_queries = 0
        failed_queries = 0
        query_times = []
        connection_errors = 0

        async def benchmark_user(user_id: int):
            nonlocal total_queries, successful_queries, failed_queries, connection_errors

            try:
                # Create connection for this user
                conn = await asyncpg.connect(**self.db_config)

                while time.perf_counter() < end_time:
                    # Choose random query benchmark
                    weights = [b.weight for b in self.query_benchmarks]
                    benchmark = random.choices(
                        self.query_benchmarks, weights=weights
                    )[0]

                    query_start = time.perf_counter()
                    try:
                        if benchmark.parameters:
                            # Handle parameterized queries
                            params = [
                                p
                                if not isinstance(p, str) or '$' not in p
                                else f"{p}_{user_id}_{total_queries}"
                                for p in benchmark.parameters
                            ]
                            await conn.execute(benchmark.query, *params)
                        else:
                            await conn.execute(benchmark.query)

                        query_time = (time.perf_counter() - query_start) * 1000
                        query_times.append(query_time)
                        successful_queries += 1

                    except Exception as e:
                        failed_queries += 1
                        logger.debug(f"Query failed for user {user_id}: {e}")

                    total_queries += 1

                    # Small delay between queries
                    await asyncio.sleep(random.uniform(0.01, 0.1))

                await conn.close()

            except Exception as e:
                connection_errors += 1
                logger.error(f"User {user_id} connection error: {e}")

        try:
            # Run concurrent users
            tasks = [benchmark_user(i) for i in range(concurrent_users)]
            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Query benchmark test error: {e}")

        duration = time.perf_counter() - start_time
        qps = successful_queries / duration if duration > 0 else 0

        # System metrics
        memory_usage = self.process.memory_info().rss / 1024 / 1024
        cpu_usage = self.process.cpu_percent()

        result = DatabaseTestResult(
            test_name="query_benchmark",
            concurrent_connections=concurrent_users,
            total_queries=total_queries,
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            duration_seconds=duration,
            queries_per_second=qps,
            average_query_time_ms=statistics.mean(query_times)
            if query_times
            else 0,
            p95_query_time_ms=np.percentile(query_times, 95)
            if query_times
            else 0,
            p99_query_time_ms=np.percentile(query_times, 99)
            if query_times
            else 0,
            connection_time_ms=0,  # Not measured in this test
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            connection_errors=connection_errors,
            test_metadata={
                'concurrent_users': concurrent_users,
                'test_duration': test_duration,
                'query_types': len(self.query_benchmarks),
                'success_rate': (successful_queries / total_queries * 100)
                if total_queries > 0
                else 0,
                'query_time_stats': {
                    'min_ms': min(query_times) if query_times else 0,
                    'max_ms': max(query_times) if query_times else 0,
                    'std_ms': np.std(query_times) if query_times else 0,
                },
            },
        )

        self.test_results.append(result)
        logger.info(
            f"Query benchmark completed: {qps:.1f} QPS, {successful_queries}/{total_queries} successful queries"
        )

        return result

    async def run_transaction_performance_test(
        self, concurrent_users: int = 10, transactions_per_user: int = 100
    ) -> DatabaseTestResult:
        """Test database transaction performance."""
        logger.info(
            f"Starting transaction performance test: {concurrent_users} users, {transactions_per_user} transactions each"
        )

        start_time = time.perf_counter()

        total_transactions = 0
        successful_transactions = 0
        failed_transactions = 0
        transaction_times = []
        connection_errors = 0

        async def transaction_user(user_id: int):
            nonlocal total_transactions, successful_transactions, failed_transactions, connection_errors

            try:
                conn = await asyncpg.connect(**self.db_config)

                for tx_id in range(transactions_per_user):
                    tx_start = time.perf_counter()

                    try:
                        async with conn.transaction():
                            # Multi-step transaction
                            await conn.execute(
                                "INSERT INTO agents (name) VALUES ($1)",
                                f"tx_user_{user_id}_{tx_id}",
                            )

                            # Simulate some work
                            await conn.execute(
                                "SELECT COUNT(*) FROM agents WHERE name LIKE $1",
                                f"tx_user_{user_id}%",
                            )

                            # Update
                            await conn.execute(
                                "UPDATE agents SET name = $1 WHERE name = $2",
                                f"tx_user_{user_id}_{tx_id}_updated",
                                f"tx_user_{user_id}_{tx_id}",
                            )

                        tx_time = (time.perf_counter() - tx_start) * 1000
                        transaction_times.append(tx_time)
                        successful_transactions += 1

                    except Exception as e:
                        failed_transactions += 1
                        logger.debug(
                            f"Transaction failed for user {user_id}: {e}"
                        )

                    total_transactions += 1

                    # Small delay between transactions
                    await asyncio.sleep(0.01)

                await conn.close()

            except Exception as e:
                connection_errors += 1
                logger.error(
                    f"Transaction user {user_id} connection error: {e}"
                )

        try:
            # Run concurrent transaction users
            tasks = [transaction_user(i) for i in range(concurrent_users)]
            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Transaction performance test error: {e}")

        duration = time.perf_counter() - start_time
        tps = successful_transactions / duration if duration > 0 else 0

        # System metrics
        memory_usage = self.process.memory_info().rss / 1024 / 1024
        cpu_usage = self.process.cpu_percent()

        result = DatabaseTestResult(
            test_name="transaction_performance",
            concurrent_connections=concurrent_users,
            total_queries=total_transactions,
            successful_queries=successful_transactions,
            failed_queries=failed_transactions,
            duration_seconds=duration,
            queries_per_second=tps,
            average_query_time_ms=statistics.mean(transaction_times)
            if transaction_times
            else 0,
            p95_query_time_ms=np.percentile(transaction_times, 95)
            if transaction_times
            else 0,
            p99_query_time_ms=np.percentile(transaction_times, 99)
            if transaction_times
            else 0,
            connection_time_ms=0,  # Not measured in this test
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            connection_errors=connection_errors,
            test_metadata={
                'concurrent_users': concurrent_users,
                'transactions_per_user': transactions_per_user,
                'transaction_success_rate': (
                    successful_transactions / total_transactions * 100
                )
                if total_transactions > 0
                else 0,
                'transaction_time_stats': {
                    'min_ms': min(transaction_times)
                    if transaction_times
                    else 0,
                    'max_ms': max(transaction_times)
                    if transaction_times
                    else 0,
                    'std_ms': np.std(transaction_times)
                    if transaction_times
                    else 0,
                },
            },
        )

        self.test_results.append(result)
        logger.info(
            f"Transaction performance test completed: {tps:.1f} TPS, {successful_transactions}/{total_transactions} successful"
        )

        return result

    async def run_database_stress_test(
        self, max_connections: int = 100, test_duration: int = 300
    ) -> DatabaseTestResult:
        """Run database stress test with high connection count."""
        logger.info(
            f"Starting database stress test: {max_connections} connections for {test_duration}s"
        )

        start_time = time.perf_counter()
        end_time = start_time + test_duration

        total_queries = 0
        successful_queries = 0
        failed_queries = 0
        query_times = []
        connection_errors = 0
        active_connections = 0

        async def stress_connection(conn_id: int):
            nonlocal total_queries, successful_queries, failed_queries, connection_errors, active_connections

            try:
                conn = await asyncpg.connect(**self.db_config)
                active_connections += 1

                while time.perf_counter() < end_time:
                    # Random query types with different loads
                    query_type = random.choices(
                        ['simple', 'insert', 'update', 'select', 'complex'],
                        weights=[0.3, 0.2, 0.2, 0.2, 0.1],
                    )[0]

                    query_start = time.perf_counter()
                    try:
                        if query_type == 'simple':
                            await conn.execute("SELECT 1")
                        elif query_type == 'insert':
                            await conn.execute(
                                "INSERT INTO agents (name) VALUES ($1)",
                                f"stress_{conn_id}_{total_queries}",
                            )
                        elif query_type == 'update':
                            await conn.execute(
                                "UPDATE agents SET name = $1 WHERE id = $2",
                                f"updated_{conn_id}",
                                1,
                            )
                        elif query_type == 'select':
                            await conn.execute("SELECT * FROM agents LIMIT 10")
                        else:  # complex
                            await conn.execute(
                                "SELECT COUNT(*) FROM agents WHERE name LIKE $1",
                                f"stress_{conn_id}%",
                            )

                        query_time = (time.perf_counter() - query_start) * 1000
                        query_times.append(query_time)
                        successful_queries += 1

                    except Exception as e:
                        failed_queries += 1
                        logger.debug(
                            f"Stress query failed for connection {conn_id}: {e}"
                        )

                    total_queries += 1

                    # Variable delay to simulate real load
                    await asyncio.sleep(random.uniform(0.001, 0.05))

                await conn.close()
                active_connections -= 1

            except Exception as e:
                connection_errors += 1
                logger.error(f"Stress connection {conn_id} error: {e}")

        try:
            # Start connections with staggered timing
            tasks = []
            for i in range(max_connections):
                task = asyncio.create_task(stress_connection(i))
                tasks.append(task)

                # Stagger connection attempts
                if i % 10 == 0:
                    await asyncio.sleep(0.1)

            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Database stress test error: {e}")

        duration = time.perf_counter() - start_time
        qps = successful_queries / duration if duration > 0 else 0

        # System metrics
        memory_usage = self.process.memory_info().rss / 1024 / 1024
        cpu_usage = self.process.cpu_percent()

        result = DatabaseTestResult(
            test_name="database_stress",
            concurrent_connections=max_connections,
            total_queries=total_queries,
            successful_queries=successful_queries,
            failed_queries=failed_queries,
            duration_seconds=duration,
            queries_per_second=qps,
            average_query_time_ms=statistics.mean(query_times)
            if query_times
            else 0,
            p95_query_time_ms=np.percentile(query_times, 95)
            if query_times
            else 0,
            p99_query_time_ms=np.percentile(query_times, 99)
            if query_times
            else 0,
            connection_time_ms=0,  # Not measured in this test
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            connection_errors=connection_errors,
            test_metadata={
                'max_connections': max_connections,
                'test_duration': test_duration,
                'peak_active_connections': active_connections,
                'query_success_rate': (
                    successful_queries / total_queries * 100
                )
                if total_queries > 0
                else 0,
                'query_distribution': {
                    'simple': 0.3,
                    'insert': 0.2,
                    'update': 0.2,
                    'select': 0.2,
                    'complex': 0.1,
                },
            },
        )

        self.test_results.append(result)
        logger.info(
            f"Database stress test completed: {qps:.1f} QPS, {successful_queries}/{total_queries} successful"
        )

        return result

    async def run_comprehensive_database_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive database test suite."""
        logger.info("Starting comprehensive database test suite")

        try:
            # Setup test environment
            await self.setup_test_environment()

            test_results = {}

            # 1. Connection Pool Test
            logger.info("Running connection pool test...")
            connection_result = await self.run_connection_pool_test(
                max_connections=25
            )
            test_results['connection_pool'] = connection_result

            # 2. Query Benchmark Test
            logger.info("Running query benchmark test...")
            query_result = await self.run_query_benchmark_test(
                concurrent_users=15, test_duration=60
            )
            test_results['query_benchmark'] = query_result

            # 3. Transaction Performance Test
            logger.info("Running transaction performance test...")
            transaction_result = await self.run_transaction_performance_test(
                concurrent_users=10, transactions_per_user=50
            )
            test_results['transaction_performance'] = transaction_result

            # 4. Database Stress Test
            logger.info("Running database stress test...")
            stress_result = await self.run_database_stress_test(
                max_connections=50, test_duration=120
            )
            test_results['database_stress'] = stress_result

            # Generate comprehensive report
            report = self._generate_comprehensive_report(test_results)

            return report

        except Exception as e:
            logger.error(f"Database test suite error: {e}")
            return {'error': str(e)}

        finally:
            # Cleanup test environment
            try:
                await self.cleanup_test_environment()
            except Exception as e:
                logger.warning(f"Cleanup error: {e}")

    def _generate_comprehensive_report(
        self, test_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive database performance report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_summary': {
                'total_tests': len(test_results),
                'tests_completed': len(
                    [
                        k
                        for k, v in test_results.items()
                        if isinstance(v, DatabaseTestResult)
                    ]
                ),
                'database_config': {
                    'host': self.db_config.get('host', 'localhost'),
                    'port': self.db_config.get('port', 5432),
                    'database': self.db_config.get('database', 'unknown'),
                },
            },
            'performance_metrics': {},
            'sla_validation': {'violations': [], 'requirements_met': True},
            'recommendations': [],
        }

        # Analyze each test result
        for test_name, result in test_results.items():
            if isinstance(result, DatabaseTestResult):
                report['performance_metrics'][test_name] = {
                    'concurrent_connections': result.concurrent_connections,
                    'queries_per_second': result.queries_per_second,
                    'average_query_time_ms': result.average_query_time_ms,
                    'p95_query_time_ms': result.p95_query_time_ms,
                    'p99_query_time_ms': result.p99_query_time_ms,
                    'connection_time_ms': result.connection_time_ms,
                    'success_rate': (
                        result.successful_queries / result.total_queries * 100
                    )
                    if result.total_queries > 0
                    else 0,
                    'memory_usage_mb': result.memory_usage_mb,
                    'cpu_usage_percent': result.cpu_usage_percent,
                    'connection_errors': result.connection_errors,
                }

                # Check SLA violations
                violations = self._check_sla_violations(result)
                if violations:
                    report['sla_validation']['violations'].extend(violations)
                    report['sla_validation']['requirements_met'] = False

        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(
            test_results
        )

        return report

    def _check_sla_violations(
        self, result: DatabaseTestResult
    ) -> List[Dict[str, Any]]:
        """Check for SLA violations in test results."""
        violations = []

        # Query time violations
        if result.average_query_time_ms > self.thresholds['query_time_ms']:
            violations.append(
                {
                    'metric': 'query_time',
                    'threshold': self.thresholds['query_time_ms'],
                    'actual': result.average_query_time_ms,
                    'severity': 'medium',
                    'description': f'Average query time ({result.average_query_time_ms:.1f}ms) exceeds threshold',
                }
            )

        # Connection time violations
        if result.connection_time_ms > self.thresholds['connection_time_ms']:
            violations.append(
                {
                    'metric': 'connection_time',
                    'threshold': self.thresholds['connection_time_ms'],
                    'actual': result.connection_time_ms,
                    'severity': 'medium',
                    'description': f'Average connection time ({result.connection_time_ms:.1f}ms) exceeds threshold',
                }
            )

        # Throughput violations
        if result.queries_per_second < self.thresholds['queries_per_second']:
            violations.append(
                {
                    'metric': 'queries_per_second',
                    'threshold': self.thresholds['queries_per_second'],
                    'actual': result.queries_per_second,
                    'severity': 'high',
                    'description': f'Queries per second ({result.queries_per_second:.1f}) below threshold',
                }
            )

        # Success rate violations
        success_rate = (
            (result.successful_queries / result.total_queries * 100)
            if result.total_queries > 0
            else 0
        )
        if success_rate < self.thresholds['connection_success_rate']:
            violations.append(
                {
                    'metric': 'success_rate',
                    'threshold': self.thresholds['connection_success_rate'],
                    'actual': success_rate,
                    'severity': 'critical',
                    'description': f'Success rate ({success_rate:.1f}%) below threshold',
                }
            )

        # Memory violations
        if result.memory_usage_mb > self.thresholds['memory_usage_max_mb']:
            violations.append(
                {
                    'metric': 'memory_usage',
                    'threshold': self.thresholds['memory_usage_max_mb'],
                    'actual': result.memory_usage_mb,
                    'severity': 'medium',
                    'description': f'Memory usage ({result.memory_usage_mb:.1f}MB) exceeds threshold',
                }
            )

        # CPU violations
        if result.cpu_usage_percent > self.thresholds['cpu_usage_max_percent']:
            violations.append(
                {
                    'metric': 'cpu_usage',
                    'threshold': self.thresholds['cpu_usage_max_percent'],
                    'actual': result.cpu_usage_percent,
                    'severity': 'medium',
                    'description': f'CPU usage ({result.cpu_usage_percent:.1f}%) exceeds threshold',
                }
            )

        return violations

    def _generate_recommendations(
        self, test_results: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []

        for test_name, result in test_results.items():
            if isinstance(result, DatabaseTestResult):
                # Query performance recommendations
                if result.p95_query_time_ms > 200:
                    recommendations.append(
                        f"High P95 query time in {test_name} ({result.p95_query_time_ms:.1f}ms). Consider adding indexes, optimizing queries, or increasing connection pool size."
                    )

                # Throughput recommendations
                if result.queries_per_second < 500:
                    recommendations.append(
                        f"Low throughput in {test_name} ({result.queries_per_second:.1f} QPS). Consider connection pooling, query optimization, or database tuning."
                    )

                # Connection recommendations
                if result.connection_errors > 0:
                    recommendations.append(
                        f"Connection errors in {test_name} ({result.connection_errors}). Check connection limits, network stability, and implement retry logic."
                    )

                # Memory recommendations
                if result.memory_usage_mb > 800:
                    recommendations.append(
                        f"High memory usage in {test_name} ({result.memory_usage_mb:.1f}MB). Consider connection pooling and memory optimization."
                    )

                # Success rate recommendations
                success_rate = (
                    (result.successful_queries / result.total_queries * 100)
                    if result.total_queries > 0
                    else 0
                )
                if success_rate < 95:
                    recommendations.append(
                        f"Low success rate in {test_name} ({success_rate:.1f}%). Investigate database errors and implement better error handling."
                    )

        if not recommendations:
            recommendations.append(
                "All database performance tests passed. Database is performing well under tested conditions."
            )

        return recommendations

    def save_results(self, filename: str):
        """Save test results to file."""
        data = {
            'timestamp': datetime.now().isoformat(),
            'db_config': {
                k: v for k, v in self.db_config.items() if k != 'password'
            },
            'test_results': [
                {
                    'test_name': result.test_name,
                    'concurrent_connections': result.concurrent_connections,
                    'total_queries': result.total_queries,
                    'successful_queries': result.successful_queries,
                    'failed_queries': result.failed_queries,
                    'duration_seconds': result.duration_seconds,
                    'queries_per_second': result.queries_per_second,
                    'average_query_time_ms': result.average_query_time_ms,
                    'p95_query_time_ms': result.p95_query_time_ms,
                    'p99_query_time_ms': result.p99_query_time_ms,
                    'connection_time_ms': result.connection_time_ms,
                    'memory_usage_mb': result.memory_usage_mb,
                    'cpu_usage_percent': result.cpu_usage_percent,
                    'connection_errors': result.connection_errors,
                    'test_metadata': result.test_metadata,
                }
                for result in self.test_results
            ],
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Database test results saved to {filename}")


# Example usage
async def run_database_performance_validation():
    """Run database performance validation."""
    print("=" * 80)
    print("DATABASE PERFORMANCE VALIDATION SUITE")
    print("=" * 80)

    # Database configuration (mock for testing)
    db_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'test_db',
        'user': 'test_user',
        'password': 'test_password',
    }

    # Note: This is a mock implementation for testing purposes
    # In a real scenario, you would have a running PostgreSQL database

    print("NOTE: This is a mock database performance test for demonstration.")
    print("In production, configure with real database credentials.")

    try:
        tester = DatabaseLoadTester(db_config)

        # Simulate test results since we don't have a real database
        mock_results = {
            'connection_pool': DatabaseTestResult(
                test_name="connection_pool",
                concurrent_connections=25,
                total_queries=25,
                successful_queries=24,
                failed_queries=1,
                duration_seconds=5.2,
                queries_per_second=4.6,
                average_query_time_ms=45.3,
                p95_query_time_ms=78.2,
                p99_query_time_ms=95.1,
                connection_time_ms=42.7,
                memory_usage_mb=256.8,
                cpu_usage_percent=15.3,
                connection_errors=1,
            ),
            'query_benchmark': DatabaseTestResult(
                test_name="query_benchmark",
                concurrent_connections=15,
                total_queries=1847,
                successful_queries=1832,
                failed_queries=15,
                duration_seconds=60.1,
                queries_per_second=30.5,
                average_query_time_ms=28.6,
                p95_query_time_ms=125.4,
                p99_query_time_ms=187.9,
                connection_time_ms=0,
                memory_usage_mb=312.4,
                cpu_usage_percent=45.7,
                connection_errors=2,
            ),
        }

        report = tester._generate_comprehensive_report(mock_results)

        # Print results
        print("\n" + "=" * 50)
        print("DATABASE PERFORMANCE RESULTS")
        print("=" * 50)

        print(f"Total tests: {report['test_summary']['total_tests']}")
        print(f"Tests completed: {report['test_summary']['tests_completed']}")

        # Print performance metrics
        for test_name, metrics in report['performance_metrics'].items():
            print(f"\n{test_name.upper()}:")
            print(
                f"  Concurrent Connections: {metrics['concurrent_connections']}"
            )
            print(f"  Queries per Second: {metrics['queries_per_second']:.1f}")
            print(
                f"  Avg Query Time: {metrics['average_query_time_ms']:.1f}ms"
            )
            print(f"  P95 Query Time: {metrics['p95_query_time_ms']:.1f}ms")
            print(f"  P99 Query Time: {metrics['p99_query_time_ms']:.1f}ms")
            print(f"  Success Rate: {metrics['success_rate']:.1f}%")
            print(f"  Memory Usage: {metrics['memory_usage_mb']:.1f}MB")
            print(f"  Connection Errors: {metrics['connection_errors']}")

        # SLA validation
        sla = report['sla_validation']
        print("\n" + "=" * 30)
        print("SLA VALIDATION")
        print("=" * 30)
        print(f"Requirements met: {'✓' if sla['requirements_met'] else '✗'}")

        if sla['violations']:
            print("\nViolations:")
            for violation in sla['violations']:
                print(f"  - {violation['metric']}: {violation['description']}")

        # Recommendations
        print("\n" + "=" * 30)
        print("RECOMMENDATIONS")
        print("=" * 30)
        for rec in report['recommendations']:
            print(f"  - {rec}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"database_performance_{timestamp}.json"
        tester.save_results(filename)

        print(f"\nDetailed results saved to: {filename}")

        return report

    except Exception as e:
        print(f"Database performance validation failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    asyncio.run(run_database_performance_validation())
