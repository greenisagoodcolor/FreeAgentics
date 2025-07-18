"""Real database load testing with PostgreSQL.

Replaces mocked database operations with actual PostgreSQL queries,
providing realistic performance metrics and load testing capabilities.
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import pytest

from database.models import Agent, AgentStatus, KnowledgeEdge, KnowledgeNode
from tests.db_infrastructure.factories import (
    AgentFactory,
    KnowledgeGraphFactory,
    TestDataGenerator,
)
from tests.db_infrastructure.fixtures import (
    PerformanceTestCase,
    isolated_db_test,
)
from tests.db_infrastructure.test_config import (
    create_test_engine,
    setup_test_database,
    teardown_test_database,
)

logger = logging.getLogger(__name__)


class RealDatabaseLoadTester(PerformanceTestCase):
    """Real database load testing with actual PostgreSQL operations."""

    def __init__(self, use_sqlite: bool = False):
        super().__init__()
        self.use_sqlite = use_sqlite
        self.engine = create_test_engine(use_sqlite=use_sqlite)
        self.performance_metrics = {}
        self.operation_counts = {
            "agent_creates": 0,
            "agent_reads": 0,
            "agent_updates": 0,
            "knowledge_creates": 0,
            "complex_queries": 0,
        }

    def setup(self):
        """Setup test database."""
        setup_test_database(self.engine)

    def teardown(self):
        """Teardown test database."""
        teardown_test_database(self.engine)

    async def test_create_agents_batch(
        self, num_agents: int = 100
    ) -> List[str]:
        """Test batch agent creation with real database."""
        from sqlalchemy.orm import sessionmaker

        Session = sessionmaker(bind=self.engine)

        with self.time_operation("create_agents_batch"):
            with isolated_db_test(self.engine) as session:
                agents = AgentFactory.create_batch(
                    session,
                    count=num_agents,
                    agent_type="resource_collector",
                    status=AgentStatus.ACTIVE,
                )
                agent_ids = [agent.agent_id for agent in agents]

                # Force flush to database
                session.flush()

                # Verify they're in the database
                count = (
                    session.query(Agent)
                    .filter(Agent.agent_id.in_(agent_ids))
                    .count()
                )

                assert (
                    count == num_agents
                ), f"Expected {num_agents} agents, found {count}"

                self.operation_counts["agent_creates"] += num_agents
                logger.info(f"✅ Created {num_agents} agents in database")

                # Commit for persistent test (normally rolled back)
                session.commit()

                return agent_ids

    async def test_concurrent_agent_reads(
        self, agent_ids: List[str], num_threads: int = 10
    ) -> Dict[str, Any]:
        """Test concurrent agent reading with real database connections."""
        from sqlalchemy.orm import sessionmaker

        Session = sessionmaker(bind=self.engine)

        results = {"successful_reads": 0, "failed_reads": 0, "agents_found": 0}

        def read_agent_batch(batch_ids):
            """Read agents from database in a thread."""
            session = Session()
            local_results = {"successful": 0, "failed": 0, "found": 0}

            try:
                for agent_id in batch_ids:
                    try:
                        agent = (
                            session.query(Agent)
                            .filter(Agent.agent_id == agent_id)
                            .first()
                        )
                        if agent:
                            local_results["found"] += 1
                            # Simulate reading data
                            _ = agent.belief_state
                            _ = agent.position
                        local_results["successful"] += 1
                        self.operation_counts["agent_reads"] += 1
                    except Exception as e:
                        local_results["failed"] += 1
                        logger.error(f"Read failed for {agent_id}: {e}")
            finally:
                session.close()

            return local_results

        with self.time_operation("concurrent_agent_reads"):
            # Split agent IDs into batches
            batch_size = max(1, len(agent_ids) // num_threads)
            batches = [
                agent_ids[i : i + batch_size]
                for i in range(0, len(agent_ids), batch_size)
            ]

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(read_agent_batch, batch)
                    for batch in batches
                ]

                for future in as_completed(futures):
                    batch_result = future.result()
                    results["successful_reads"] += batch_result["successful"]
                    results["failed_reads"] += batch_result["failed"]
                    results["agents_found"] += batch_result["found"]

        logger.info(
            f"✅ Concurrent reads: {results['successful_reads']} successful, "
            f"{results['agents_found']} agents found"
        )

        return results

    async def test_concurrent_agent_updates(
        self, agent_ids: List[str], num_threads: int = 5
    ) -> Dict[str, Any]:
        """Test concurrent agent updates with real database transactions."""
        from sqlalchemy.orm import sessionmaker

        Session = sessionmaker(bind=self.engine)

        results = {
            "successful_updates": 0,
            "failed_updates": 0,
            "conflicts": 0,
        }

        def update_agent_batch(batch_ids):
            """Update agents in database with conflict handling."""
            session = Session()
            local_results = {"successful": 0, "failed": 0, "conflicts": 0}

            for agent_id in batch_ids:
                try:
                    # Start transaction
                    agent = (
                        session.query(Agent)
                        .filter(Agent.agent_id == agent_id)
                        .with_for_update()
                        .first()
                    )  # Lock row for update

                    if agent:
                        # Update belief state with timestamp
                        agent.belief_state = {
                            "timestamp": datetime.utcnow().isoformat(),
                            "updated": True,
                            "thread_id": id(session),
                            "iteration": hash(agent_id) % 1000,
                            "confidence": 0.85,
                            "observations": 42,
                        }
                        agent.updated_at = datetime.utcnow()
                        agent.inference_count = (
                            agent.inference_count or 0
                        ) + 1

                        session.commit()
                        local_results["successful"] += 1
                        self.operation_counts["agent_updates"] += 1
                    else:
                        local_results["failed"] += 1

                except Exception as e:
                    session.rollback()
                    if (
                        "deadlock" in str(e).lower()
                        or "concurrent" in str(e).lower()
                    ):
                        local_results["conflicts"] += 1
                    else:
                        local_results["failed"] += 1
                    logger.warning(f"Update failed for {agent_id}: {e}")

            session.close()
            return local_results

        with self.time_operation("concurrent_agent_updates"):
            # Smaller batches for updates to reduce conflicts
            batch_size = max(1, len(agent_ids) // (num_threads * 2))
            batches = [
                agent_ids[i : i + batch_size]
                for i in range(0, len(agent_ids), batch_size)
            ]

            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(update_agent_batch, batch)
                    for batch in batches
                ]

                for future in as_completed(futures):
                    batch_result = future.result()
                    results["successful_updates"] += batch_result["successful"]
                    results["failed_updates"] += batch_result["failed"]
                    results["conflicts"] += batch_result.get("conflicts", 0)

        logger.info(
            f"✅ Concurrent updates: {results['successful_updates']} successful, "
            f"{results['failed_updates']} failed, {results['conflicts']} conflicts"
        )

        return results

    async def test_knowledge_graph_operations(
        self, num_nodes: int = 500, num_edges: int = 1000
    ) -> bool:
        """Test knowledge graph database operations with real data."""
        from sqlalchemy.orm import sessionmaker

        Session = sessionmaker(bind=self.engine)

        with self.time_operation("knowledge_graph_operations"):
            with isolated_db_test(self.engine) as session:
                # Create knowledge graph
                graph_data = KnowledgeGraphFactory.create_connected_graph(
                    session,
                    num_nodes=num_nodes,
                    connectivity=num_edges
                    / (
                        num_nodes * (num_nodes - 1) / 2
                    ),  # Convert to probability
                )

                # Verify data was created
                node_count = session.query(KnowledgeNode).count()
                edge_count = session.query(KnowledgeEdge).count()

                logger.info(
                    f"✅ Created knowledge graph: {node_count} nodes, {edge_count} edges"
                )

                # Test graph traversal query
                start_time = time.time()

                # Find connected components
                connected_nodes = (
                    session.query(KnowledgeNode)
                    .join(
                        KnowledgeEdge,
                        KnowledgeNode.node_id == KnowledgeEdge.source_node_id,
                    )
                    .distinct()
                    .limit(100)
                    .all()
                )

                traversal_time = time.time() - start_time
                logger.info(
                    f"Graph traversal took {traversal_time:.3f}s for {len(connected_nodes)} nodes"
                )

                self.operation_counts["knowledge_creates"] += (
                    node_count + edge_count
                )

                # Commit for persistent test
                session.commit()

                return True

    async def test_complex_queries(self) -> Dict[str, Any]:
        """Test complex database queries with real PostgreSQL."""
        from sqlalchemy import and_, func, or_, text
        from sqlalchemy.orm import sessionmaker

        Session = sessionmaker(bind=self.engine)
        session = Session()
        results = {}

        try:
            # Query 1: Active agents with recent updates (with aggregation)
            with self.time_operation("active_agents_query"):
                recent_time = datetime.utcnow() - timedelta(minutes=30)

                active_agents = (
                    session.query(
                        Agent.agent_type,
                        func.count(Agent.id).label("count"),
                        func.avg(Agent.inference_count).label(
                            "avg_inferences"
                        ),
                    )
                    .filter(
                        and_(
                            Agent.status == AgentStatus.ACTIVE,
                            or_(
                                Agent.updated_at >= recent_time,
                                Agent.created_at >= recent_time,
                            ),
                        )
                    )
                    .group_by(Agent.agent_type)
                    .all()
                )

                results["active_agents_query"] = {
                    "groups": len(active_agents),
                    "data": [
                        {
                            "type": row.agent_type,
                            "count": row.count,
                            "avg_inferences": float(row.avg_inferences or 0),
                        }
                        for row in active_agents
                    ],
                }

            # Query 2: Coalition formation candidates with capability matching
            with self.time_operation("coalition_candidates_query"):
                # Complex query with JSON operations (if PostgreSQL)
                if not self.use_sqlite:
                    candidates = session.execute(
                        text(
                            """
                        SELECT a1.agent_id, a2.agent_id,
                               a1.capabilities, a2.capabilities,
                               a1.position, a2.position
                        FROM agents a1
                        CROSS JOIN agents a2
                        WHERE a1.id < a2.id
                          AND a1.status = 'active'
                          AND a2.status = 'active'
                          AND a1.agent_type IN ('resource_collector', 'explorer')
                          AND a2.agent_type IN ('resource_collector', 'explorer')
                          AND jsonb_typeof(a1.capabilities) = 'array'
                          AND jsonb_typeof(a2.capabilities) = 'array'
                        LIMIT 50
                    """
                        )
                    ).fetchall()
                else:
                    # Simplified version for SQLite
                    candidates = (
                        session.query(Agent)
                        .filter(
                            Agent.status == AgentStatus.ACTIVE,
                            Agent.agent_type.in_(
                                ["resource_collector", "explorer"]
                            ),
                        )
                        .limit(50)
                        .all()
                    )

                results["coalition_candidates_query"] = {
                    "count": len(candidates)
                }

            # Query 3: Knowledge graph analytics
            with self.time_operation("knowledge_analytics_query"):
                # Analyze knowledge graph structure
                node_stats = (
                    session.query(
                        KnowledgeNode.node_type,
                        func.count(KnowledgeNode.node_id).label("count"),
                        func.min(KnowledgeNode.created_at).label("oldest"),
                        func.max(KnowledgeNode.created_at).label("newest"),
                    )
                    .group_by(KnowledgeNode.node_type)
                    .all()
                )

                edge_stats = (
                    session.query(
                        KnowledgeEdge.edge_type,
                        func.count(KnowledgeEdge.edge_id).label("count"),
                        func.avg(KnowledgeEdge.weight).label("avg_weight"),
                    )
                    .group_by(KnowledgeEdge.edge_type)
                    .all()
                )

                results["knowledge_analytics_query"] = {
                    "node_types": len(node_stats),
                    "edge_types": len(edge_stats),
                    "total_nodes": sum(row.count for row in node_stats),
                    "total_edges": sum(row.count for row in edge_stats),
                }

            # Query 4: Performance statistics
            with self.time_operation("performance_stats_query"):
                agent_performance = (
                    session.query(
                        Agent.agent_type,
                        func.count(Agent.id).label("agent_count"),
                        func.sum(Agent.inference_count).label(
                            "total_inferences"
                        ),
                        func.avg(
                            func.cast(
                                func.json_extract(
                                    Agent.metrics, "$.success_rate"
                                ),
                                func.Float,
                            )
                        ).label("avg_success_rate"),
                    )
                    .filter(Agent.metrics.isnot(None))
                    .group_by(Agent.agent_type)
                    .all()
                )

                results["performance_stats_query"] = {
                    "agent_types_analyzed": len(agent_performance)
                }

            self.operation_counts["complex_queries"] += 4
            logger.info(
                f"✅ Complex queries completed: {len(results)} query types"
            )

            return results

        except Exception as e:
            logger.error(f"Complex query failed: {e}")
            raise
        finally:
            session.close()

    def analyze_performance_results(self) -> Dict[str, Any]:
        """Analyze real performance test results."""
        analysis = {}

        # Analyze timing data
        for operation in self.timings:
            stats = self.get_timing_stats(operation)
            if stats:
                analysis[operation] = stats

        # Add operation counts
        analysis["operation_summary"] = self.operation_counts

        # Calculate throughput
        if (
            "create_agents_batch" in analysis
            and self.operation_counts["agent_creates"] > 0
        ):
            create_stats = analysis["create_agents_batch"]
            analysis["create_agents_batch"]["throughput"] = (
                self.operation_counts["agent_creates"] / create_stats["total"]
            )

        return analysis


@pytest.mark.db_test
class TestRealDatabaseLoad:
    """Test suite for real database load testing."""

    @pytest.mark.parametrize(
        "num_agents,expected_time",
        [
            (10, 2.0),  # 10 agents should complete in < 2 seconds
            (100, 10.0),  # 100 agents should complete in < 10 seconds
            (500, 30.0),  # 500 agents should complete in < 30 seconds
        ],
    )
    async def test_agent_creation_scaling(self, num_agents, expected_time):
        """Test agent creation performance at different scales."""
        tester = RealDatabaseLoadTester()
        tester.setup()

        try:
            start_time = time.time()
            agent_ids = await tester.test_create_agents_batch(num_agents)
            duration = time.time() - start_time

            assert len(agent_ids) == num_agents
            assert (
                duration < expected_time
            ), f"Creation took {duration:.2f}s, expected < {expected_time}s"

            # Verify persistence
            from sqlalchemy.orm import sessionmaker

            Session = sessionmaker(bind=tester.engine)
            session = Session()
            count = (
                session.query(Agent)
                .filter(Agent.agent_id.in_(agent_ids))
                .count()
            )
            session.close()

            assert count == num_agents

        finally:
            tester.teardown()

    @pytest.mark.slow_db_test
    async def test_concurrent_operations_stress(self):
        """Stress test concurrent database operations."""
        tester = RealDatabaseLoadTester()
        tester.setup()

        try:
            # Create initial agents
            agent_ids = await tester.test_create_agents_batch(200)

            # Run concurrent operations
            tasks = []

            # Multiple read threads
            for _ in range(3):
                tasks.append(
                    tester.test_concurrent_agent_reads(
                        agent_ids, num_threads=10
                    )
                )

            # Multiple update threads
            for _ in range(2):
                tasks.append(
                    tester.test_concurrent_agent_updates(
                        agent_ids, num_threads=5
                    )
                )

            # Execute all concurrently
            results = await asyncio.gather(*tasks)

            # Analyze results
            total_reads = sum(r["successful_reads"] for r in results[:3])
            total_updates = sum(r["successful_updates"] for r in results[3:])
            total_conflicts = sum(r.get("conflicts", 0) for r in results[3:])

            logger.info(
                f"Stress test completed: {total_reads} reads, {total_updates} updates, {total_conflicts} conflicts"
            )

            # Assert reasonable success rates
            assert (
                total_reads > len(agent_ids) * 2
            )  # At least 2x reads per agent
            assert (
                total_updates > len(agent_ids) * 0.8
            )  # At least 80% update success

        finally:
            tester.teardown()

    @pytest.mark.postgres_only
    async def test_knowledge_graph_performance(self):
        """Test knowledge graph operations at scale."""
        tester = RealDatabaseLoadTester()
        tester.setup()

        try:
            # Test increasing graph sizes
            sizes = [(100, 200), (500, 1000), (1000, 3000)]

            for num_nodes, num_edges in sizes:
                start_time = time.time()
                success = await tester.test_knowledge_graph_operations(
                    num_nodes, num_edges
                )
                duration = time.time() - start_time

                assert success
                logger.info(
                    f"Graph with {num_nodes} nodes, {num_edges} edges created in {duration:.2f}s"
                )

                # Performance assertion: should scale roughly linearly
                expected_time = (
                    num_nodes + num_edges
                ) * 0.01  # 10ms per element
                assert duration < expected_time

            # Test complex queries on the populated graph
            query_results = await tester.test_complex_queries()
            assert len(query_results) >= 4

        finally:
            tester.teardown()

    async def test_complete_load_scenario(self):
        """Run a complete load test scenario."""
        tester = RealDatabaseLoadTester()
        tester.setup()

        try:
            logger.info("🚀 Starting complete load test scenario")

            # Phase 1: Agent creation
            logger.info("Phase 1: Creating agents...")
            agent_ids = await tester.test_create_agents_batch(500)

            # Phase 2: Knowledge graph
            logger.info("Phase 2: Building knowledge graph...")
            await tester.test_knowledge_graph_operations(1000, 2000)

            # Phase 3: Concurrent operations
            logger.info("Phase 3: Concurrent read/write operations...")
            read_task = tester.test_concurrent_agent_reads(
                agent_ids[:250], num_threads=20
            )
            update_task = tester.test_concurrent_agent_updates(
                agent_ids[250:], num_threads=10
            )

            read_results, update_results = await asyncio.gather(
                read_task, update_task
            )

            # Phase 4: Complex queries
            logger.info("Phase 4: Complex analytical queries...")
            query_results = await tester.test_complex_queries()

            # Analyze overall performance
            analysis = tester.analyze_performance_results()

            # Print summary
            print("\n" + "=" * 60)
            print("LOAD TEST SUMMARY")
            print("=" * 60)
            print(
                f"Total agents created: {analysis['operation_summary']['agent_creates']}"
            )
            print(
                f"Total agent reads: {analysis['operation_summary']['agent_reads']}"
            )
            print(
                f"Total agent updates: {analysis['operation_summary']['agent_updates']}"
            )
            print(
                f"Total knowledge operations: {analysis['operation_summary']['knowledge_creates']}"
            )
            print(
                f"Complex queries executed: {analysis['operation_summary']['complex_queries']}"
            )

            print("\nPerformance Metrics:")
            for operation, stats in analysis.items():
                if operation != "operation_summary" and isinstance(
                    stats, dict
                ):
                    print(f"\n{operation}:")
                    print(f"  Total time: {stats.get('total', 0):.3f}s")
                    print(f"  Average: {stats.get('mean', 0):.3f}s")
                    if "throughput" in stats:
                        print(
                            f"  Throughput: {stats['throughput']:.1f} ops/sec"
                        )

            # Assertions
            assert read_results["successful_reads"] > 0
            assert update_results["successful_updates"] > 0
            assert len(query_results) >= 4

            logger.info("✅ Complete load test scenario passed!")

        finally:
            tester.teardown()


if __name__ == "__main__":
    import asyncio

    async def run_manual_tests():
        """Run load tests manually."""
        print("Running real database load tests...\n")

        # Test with PostgreSQL
        print("Testing with PostgreSQL:")
        pg_tester = RealDatabaseLoadTester(use_sqlite=False)
        pg_tester.setup()

        try:
            await pg_tester.test_create_agents_batch(100)
            agent_ids = await pg_tester.test_create_agents_batch(50)
            await pg_tester.test_concurrent_agent_reads(
                agent_ids, num_threads=5
            )
            await pg_tester.test_concurrent_agent_updates(
                agent_ids, num_threads=3
            )
            await pg_tester.test_knowledge_graph_operations(100, 200)
            await pg_tester.test_complex_queries()

            analysis = pg_tester.analyze_performance_results()
            print("\nPostgreSQL Performance Analysis:")
            print(json.dumps(analysis, indent=2, default=str))

        finally:
            pg_tester.teardown()

        # Test with SQLite for comparison
        print("\n\nTesting with SQLite:")
        sqlite_tester = RealDatabaseLoadTester(use_sqlite=True)
        sqlite_tester.setup()

        try:
            await sqlite_tester.test_create_agents_batch(100)
            analysis = sqlite_tester.analyze_performance_results()
            print("\nSQLite Performance Analysis:")
            print(json.dumps(analysis, indent=2, default=str))

        finally:
            sqlite_tester.teardown()

    asyncio.run(run_manual_tests())
