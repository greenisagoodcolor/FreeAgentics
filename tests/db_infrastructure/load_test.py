"""Comprehensive load testing script for FreeAgentics database."""

import argparse
import concurrent.futures
import json
import logging
import os
import random
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from tests.db_infrastructure.data_generator import TestDataGenerator
from tests.db_infrastructure.db_reset import DatabaseReset
from tests.db_infrastructure.performance_monitor import (
    LoadTestRunner,
    PerformanceMonitor,
)
from tests.db_infrastructure.pool_config import close_all_pools, get_pool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DatabaseLoadTest:
    """Comprehensive database load testing suite."""

    def __init__(self, db_name: str = "freeagentics_test"):
        """Initialize load test with database name."""
        self.db_name = db_name
        self.data_generator = TestDataGenerator()
        self.db_reset = DatabaseReset()
        self.pool = None
        self.test_data = None

    def setup(self, reset_db: bool = True, populate_data: bool = True):
        """Set up test environment."""
        logger.info("Setting up test environment...")

        if reset_db:
            logger.info("Resetting database...")
            if not self.db_reset.reset_database(self.db_name):
                raise RuntimeError("Failed to reset database")

        # Get connection pool
        self.pool = get_pool(
            "load_test",
            min_connections=10,
            max_connections=100,
            database=self.db_name,
        )

        if populate_data:
            logger.info("Populating test data...")
            self._populate_test_data()

    def teardown(self):
        """Clean up test environment."""
        logger.info("Cleaning up test environment...")
        close_all_pools()

    def _populate_test_data(self):
        """Populate database with test data."""
        # Generate test data
        self.test_data = self.data_generator.generate_complete_dataset(
            num_agents=100,
            num_coalitions=20,
            num_knowledge_nodes=1000,
            num_edges=2000,
        )

        with self.pool.get_connection() as (conn, cursor):
            # Insert agents
            for agent in self.test_data["agents"]:
                cursor.execute(
                    """
                    INSERT INTO agents (id, name, template, status, gmn_spec, pymdp_config,
                                      beliefs, preferences, position, metrics, parameters,
                                      inference_count, total_steps, created_at, last_active)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                    (
                        agent["id"],
                        agent["name"],
                        agent["template"],
                        agent["status"],
                        agent["gmn_spec"],
                        agent["pymdp_config"],
                        agent["beliefs"],
                        agent["preferences"],
                        agent["position"],
                        agent["metrics"],
                        agent["parameters"],
                        agent["inference_count"],
                        agent["total_steps"],
                        agent["created_at"],
                        agent["last_active"],
                    ),
                )

            # Insert coalitions
            for coalition in self.test_data["coalitions"]:
                cursor.execute(
                    """
                    INSERT INTO coalitions (id, name, description, status, objectives,
                                          required_capabilities, achieved_objectives,
                                          performance_score, cohesion_score, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                    (
                        coalition["id"],
                        coalition["name"],
                        coalition["description"],
                        coalition["status"],
                        coalition["objectives"],
                        coalition["required_capabilities"],
                        coalition["achieved_objectives"],
                        coalition["performance_score"],
                        coalition["cohesion_score"],
                        coalition["created_at"],
                    ),
                )

            # Insert memberships
            for membership in self.test_data["memberships"]:
                cursor.execute(
                    """
                    INSERT INTO agent_coalition (agent_id, coalition_id, role, joined_at, contribution_score)
                    VALUES (%s, %s, %s, %s, %s)
                """,
                    (
                        membership["agent_id"],
                        membership["coalition_id"],
                        membership["role"],
                        membership["joined_at"],
                        membership["contribution_score"],
                    ),
                )

            # Insert knowledge nodes
            for node in self.test_data["knowledge_nodes"]:
                cursor.execute(
                    """
                    INSERT INTO knowledge_nodes (id, type, label, properties, version,
                                               is_current, confidence, source, creator_agent_id, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                    (
                        node["id"],
                        node["type"],
                        node["label"],
                        node["properties"],
                        node["version"],
                        node["is_current"],
                        node["confidence"],
                        node["source"],
                        node["creator_agent_id"],
                        node["created_at"],
                    ),
                )

            # Insert edges
            for edge in self.test_data["knowledge_edges"]:
                cursor.execute(
                    """
                    INSERT INTO knowledge_edges (id, source_id, target_id, type, properties, confidence, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                    (
                        edge["id"],
                        edge["source_id"],
                        edge["target_id"],
                        edge["type"],
                        edge["properties"],
                        edge["confidence"],
                        edge["created_at"],
                    ),
                )

        logger.info(
            f"Populated database with {len(self.test_data['agents'])} agents, "
            f"{len(self.test_data['coalitions'])} coalitions, "
            f"{len(self.test_data['knowledge_nodes'])} knowledge nodes"
        )

    def test_agent_queries(self):
        """Test various agent query patterns."""
        with self.pool.get_connection() as (conn, cursor):
            # Get active agents
            cursor.execute("SELECT * FROM agents WHERE status = 'ACTIVE'")
            active_agents = cursor.fetchall()

            # Get agent with coalitions
            cursor.execute(
                """
                SELECT a.*, array_agg(c.name) as coalitions
                FROM agents a
                LEFT JOIN agent_coalition ac ON a.id = ac.agent_id
                LEFT JOIN coalitions c ON ac.coalition_id = c.id
                WHERE a.id = %s
                GROUP BY a.id
            """,
                (random.choice(active_agents)["id"],),
            )

            # Update agent metrics
            agent_id = random.choice(active_agents)["id"]
            cursor.execute(
                """
                UPDATE agents
                SET metrics = metrics || %s,
                    inference_count = inference_count + 1,
                    last_active = NOW()
                WHERE id = %s
            """,
                (
                    json.dumps(
                        {"last_inference_time": random.uniform(0.1, 2.0)}
                    ),
                    agent_id,
                ),
            )

    def test_coalition_operations(self):
        """Test coalition-related operations."""
        with self.pool.get_connection() as (conn, cursor):
            # Find coalitions needing members
            cursor.execute(
                """
                SELECT c.*, COUNT(ac.agent_id) as member_count
                FROM coalitions c
                LEFT JOIN agent_coalition ac ON c.id = ac.coalition_id
                WHERE c.status = 'FORMING'
                GROUP BY c.id
                HAVING COUNT(ac.agent_id) < 5
            """
            )

            # Update coalition performance
            cursor.execute(
                """
                UPDATE coalitions
                SET performance_score = (
                    SELECT AVG(ac.contribution_score)
                    FROM agent_coalition ac
                    WHERE ac.coalition_id = coalitions.id
                )
                WHERE status = 'ACTIVE'
            """
            )

    def test_knowledge_graph_queries(self):
        """Test knowledge graph operations."""
        with self.pool.get_connection() as (conn, cursor):
            # Find connected nodes
            cursor.execute(
                """
                WITH RECURSIVE connected_nodes AS (
                    SELECT id, type, label, 0 as depth
                    FROM knowledge_nodes
                    WHERE id = %s

                    UNION ALL

                    SELECT kn.id, kn.type, kn.label, cn.depth + 1
                    FROM knowledge_nodes kn
                    JOIN knowledge_edges ke ON kn.id = ke.target_id
                    JOIN connected_nodes cn ON ke.source_id = cn.id
                    WHERE cn.depth < 3
                )
                SELECT DISTINCT * FROM connected_nodes
            """,
                (random.choice(self.test_data["knowledge_nodes"])["id"],),
            )

            # Insert new knowledge
            new_node = self.data_generator.generate_knowledge_node()
            cursor.execute(
                """
                INSERT INTO knowledge_nodes (type, label, properties, confidence, source)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id
            """,
                (
                    new_node["type"],
                    new_node["label"],
                    new_node["properties"],
                    new_node["confidence"],
                    new_node["source"],
                ),
            )

    def test_complex_analytics(self):
        """Test complex analytical queries."""
        with self.pool.get_connection() as (conn, cursor):
            # Agent performance ranking
            cursor.execute(
                """
                SELECT
                    a.name,
                    a.template,
                    a.inference_count,
                    COUNT(DISTINCT ac.coalition_id) as coalition_count,
                    AVG(ac.contribution_score) as avg_contribution,
                    COUNT(DISTINCT kn.id) as knowledge_contributions
                FROM agents a
                LEFT JOIN agent_coalition ac ON a.id = ac.agent_id
                LEFT JOIN knowledge_nodes kn ON a.id = kn.creator_agent_id
                GROUP BY a.id, a.name, a.template, a.inference_count
                ORDER BY a.inference_count DESC
                LIMIT 10
            """
            )

            # Coalition effectiveness analysis
            cursor.execute(
                """
                SELECT
                    c.name,
                    c.status,
                    c.performance_score,
                    COUNT(DISTINCT ac.agent_id) as member_count,
                    jsonb_array_length(c.objectives::jsonb) as objective_count,
                    jsonb_array_length(c.achieved_objectives::jsonb) as achieved_count
                FROM coalitions c
                LEFT JOIN agent_coalition ac ON c.id = ac.coalition_id
                GROUP BY c.id, c.name, c.status, c.performance_score, c.objectives, c.achieved_objectives
                HAVING COUNT(DISTINCT ac.agent_id) > 0
            """
            )

    def run_load_test(
        self,
        test_duration: int = 60,
        num_threads: int = 10,
        operations_per_second: int = 100,
    ):
        """Run the main load test."""
        logger.info(
            f"Starting load test: {num_threads} threads, {test_duration}s duration"
        )

        monitor = PerformanceMonitor(
            f"load_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        monitor.start_test_run(
            {
                "num_threads": num_threads,
                "duration": test_duration,
                "target_ops_per_second": operations_per_second,
            }
        )

        # Define test operations
        operations = [
            ("agent_queries", self.test_agent_queries, 0.4),
            ("coalition_ops", self.test_coalition_operations, 0.2),
            ("knowledge_queries", self.test_knowledge_graph_queries, 0.3),
            ("analytics", self.test_complex_analytics, 0.1),
        ]

        # Calculate operations per thread
        ops_per_thread_per_second = operations_per_second / num_threads
        sleep_between_ops = 1.0 / ops_per_thread_per_second

        stop_event = threading.Event()
        operation_counts = defaultdict(int)
        error_counts = defaultdict(int)

        def worker(thread_id: int):
            """Worker thread for load testing."""
            while not stop_event.is_set():
                # Select operation based on weights
                rand = random.random()
                cumulative = 0

                for op_name, op_func, weight in operations:
                    cumulative += weight
                    if rand <= cumulative:
                        try:
                            with monitor.measure_operation(
                                "database_operation", op_name
                            ):
                                op_func()
                            operation_counts[op_name] += 1
                        except Exception as e:
                            error_counts[op_name] += 1
                            logger.error(f"Operation {op_name} failed: {e}")
                        break

                time.sleep(sleep_between_ops)

        # Start worker threads
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_threads
        ) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]

            # Run for specified duration
            time.sleep(test_duration)
            stop_event.set()

            # Wait for threads to complete
            concurrent.futures.wait(futures)

        # End monitoring and generate report
        summary = monitor.end_test_run()

        # Add operation statistics
        summary["operation_counts"] = dict(operation_counts)
        summary["error_counts"] = dict(error_counts)
        summary["total_operations"] = sum(operation_counts.values())
        summary["total_errors"] = sum(error_counts.values())
        summary["error_rate"] = (
            summary["total_errors"] / summary["total_operations"] * 100
            if summary["total_operations"] > 0
            else 0
        )

        # Generate report
        report_path = (
            f"load_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        _report = monitor.generate_report(report_path)

        logger.info(f"Load test completed. Report saved to: {report_path}")
        logger.info(f"Total operations: {summary['total_operations']}")
        logger.info(f"Error rate: {summary['error_rate']:.2f}%")

        return summary

    def run_stress_test(
        self, max_connections: int = 200, ramp_up_time: int = 30
    ):
        """Run a stress test to find breaking points."""
        logger.info(
            f"Starting stress test: ramping up to {max_connections} connections"
        )

        runner = LoadTestRunner(
            f"stress_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        def stress_operation():
            """Single stress test operation."""
            operation = random.choice(
                [
                    self.test_agent_queries,
                    self.test_coalition_operations,
                    self.test_knowledge_graph_queries,
                    self.test_complex_analytics,
                ]
            )
            operation()

        summary = runner.run_concurrent_test(
            stress_operation,
            num_threads=max_connections,
            duration_seconds=60,
            ramp_up_seconds=ramp_up_time,
        )

        return summary


def main():
    """Command-line interface for load testing."""
    parser = argparse.ArgumentParser(description="Database load testing")
    parser.add_argument(
        "--test",
        choices=["load", "stress", "quick"],
        default="quick",
        help="Type of test to run",
    )
    parser.add_argument(
        "--duration", type=int, default=60, help="Test duration in seconds"
    )
    parser.add_argument(
        "--threads", type=int, default=10, help="Number of concurrent threads"
    )
    parser.add_argument(
        "--ops-per-second",
        type=int,
        default=100,
        help="Target operations per second",
    )
    parser.add_argument(
        "--no-reset", action="store_true", help="Skip database reset"
    )
    parser.add_argument(
        "--no-populate", action="store_true", help="Skip data population"
    )

    args = parser.parse_args()

    # Create and run test
    test = DatabaseLoadTest()

    try:
        test.setup(
            reset_db=not args.no_reset, populate_data=not args.no_populate
        )

        if args.test == "load":
            summary = test.run_load_test(
                test_duration=args.duration,
                num_threads=args.threads,
                operations_per_second=args.ops_per_second,
            )
        elif args.test == "stress":
            summary = test.run_stress_test(max_connections=args.threads)
        else:  # quick
            summary = test.run_load_test(
                test_duration=10, num_threads=5, operations_per_second=50
            )

        print("\nTest completed successfully!")
        print(f"Total operations: {summary.get('total_operations', 0)}")
        print(f"Error rate: {summary.get('error_rate', 0):.2f}%")

    finally:
        test.teardown()


if __name__ == "__main__":
    main()
