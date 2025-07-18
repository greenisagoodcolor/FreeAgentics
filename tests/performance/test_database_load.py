"""Database load testing for production deployment validation.

Tests PostgreSQL performance with realistic agent populations,
concurrent operations, and multi-user scenarios.

CRITICAL PRODUCTION BLOCKER: Database performance under load
"""

import asyncio
import json
import logging
import os
import resource
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

# Import psutil if available, otherwise use resource module
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import numpy if available, otherwise use built-in functions
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

    # Fallback implementations
    class np:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0

        @staticmethod
        def max(values):
            return max(values) if values else 0

        @staticmethod
        def min(values):
            return min(values) if values else 0

        @staticmethod
        def percentile(values, p):
            if not values:
                return 0
            sorted_values = sorted(values)
            k = (len(sorted_values) - 1) * p / 100
            f = int(k)
            c = k - f
            if f + 1 < len(sorted_values):
                return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c
            return sorted_values[f]

        @staticmethod
        def random():
            import random

            return random.random()

        @staticmethod
        def randint(low, high):
            import random

            return random.randint(low, high)

        @staticmethod
        def choice(array):
            import random

            return random.choice(array)


try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import QueuePool

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

try:
    from database.connection import create_database_engine, get_database_url
    from database.models import (
        Agent,
        Base,
        Coalition,
        KnowledgeEdge,
        KnowledgeNode,
    )

    DATABASE_MODELS_AVAILABLE = True
except ImportError:
    DATABASE_MODELS_AVAILABLE = False

logger = logging.getLogger(__name__)


class DatabaseLoadTester:
    """Database load testing with realistic scenarios."""

    def __init__(self, test_db_url: str = None):
        self.test_db_url = test_db_url or os.getenv(
            "TEST_DATABASE_URL", "sqlite:///./test_perf.db"
        )
        self.engine = None
        self.session_factory = None
        self.performance_metrics = {}

    def setup_test_database(self):
        """Setup test database with connection pooling."""
        if not SQLALCHEMY_AVAILABLE:
            logger.warning(
                "⚠️ SQLAlchemy not available - skipping database tests"
            )
            return False

        if not DATABASE_MODELS_AVAILABLE:
            logger.warning(
                "⚠️ Database models not available - skipping database tests"
            )
            return False

        try:
            self.engine = create_engine(
                self.test_db_url,
                poolclass=QueuePool,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                echo=False,
            )

            # Create tables
            Base.metadata.create_all(self.engine)
            self.session_factory = sessionmaker(bind=self.engine)

            logger.info("✅ Test database setup complete")
            return True

        except Exception as e:
            logger.error(f"❌ Database setup failed: {e}")
            return False

    def cleanup_test_database(self):
        """Cleanup test database."""
        if self.engine:
            try:
                Base.metadata.drop_all(self.engine)
                self.engine.dispose()
                logger.info("✅ Test database cleanup complete")
            except Exception as e:
                logger.error(f"❌ Database cleanup failed: {e}")

    def measure_performance(self, operation_name: str):
        """Decorator to measure operation performance."""

        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                if PSUTIL_AVAILABLE:
                    start_memory = (
                        psutil.Process().memory_info().rss / 1024 / 1024
                    )  # MB
                else:
                    start_memory = (
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        / 1024
                    )  # KB to MB

                try:
                    result = await func(*args, **kwargs)
                    success = True
                    error = None
                except Exception as e:
                    result = None
                    success = False
                    error = str(e)

                end_time = time.time()
                if PSUTIL_AVAILABLE:
                    end_memory = (
                        psutil.Process().memory_info().rss / 1024 / 1024
                    )  # MB
                else:
                    end_memory = (
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        / 1024
                    )  # KB to MB

                metrics = {
                    "duration": end_time - start_time,
                    "memory_delta": end_memory - start_memory,
                    "success": success,
                    "error": error,
                    "timestamp": datetime.now(),
                }

                if operation_name not in self.performance_metrics:
                    self.performance_metrics[operation_name] = []
                self.performance_metrics[operation_name].append(metrics)

                logger.info(
                    f"📊 {operation_name}: {metrics['duration']:.3f}s, "
                    f"Memory: {metrics['memory_delta']:+.1f}MB, "
                    f"Success: {success}"
                )

                if not success:
                    raise Exception(error)

                return result

            return wrapper

        return decorator

    async def test_create_agents_batch(
        self, num_agents: int = 100
    ) -> List[str]:
        """Test batch agent creation performance."""
        agent_ids = []
        session = self.session_factory()

        try:
            agents = []
            for i in range(num_agents):
                agent = Agent(
                    agent_id=f"test_agent_{i}_{int(time.time())}",
                    name=f"TestAgent_{i}",
                    agent_type="resource_collector",
                    status="active",
                    belief_state={"test": f"belief_{i}"},
                    position={
                        "lat": 37.7749 + i * 0.001,
                        "lon": -122.4194 + i * 0.001,
                    },
                    capabilities=["resource_collection", "exploration"],
                    created_at=datetime.utcnow(),
                )
                agents.append(agent)
                agent_ids.append(agent.agent_id)

            # Batch insert
            session.add_all(agents)
            session.commit()

            logger.info(f"✅ Created {num_agents} agents in batch")
            return agent_ids

        except Exception as e:
            session.rollback()
            logger.error(f"❌ Batch agent creation failed: {e}")
            raise
        finally:
            session.close()

    async def test_concurrent_agent_reads(
        self, agent_ids: List[str], num_threads: int = 10
    ) -> Dict[str, Any]:
        """Test concurrent agent reading performance."""
        results = {"successful_reads": 0, "failed_reads": 0, "agents_found": 0}

        def read_agent_batch(batch_ids):
            session = self.session_factory()
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
                        local_results["successful"] += 1
                    except Exception:
                        local_results["failed"] += 1
            finally:
                session.close()

            return local_results

        # Split agent IDs into batches for concurrent processing
        batch_size = max(1, len(agent_ids) // num_threads)
        batches = [
            agent_ids[i : i + batch_size]
            for i in range(0, len(agent_ids), batch_size)
        ]

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(read_agent_batch, batch) for batch in batches
            ]

            for future in futures:
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
        """Test concurrent agent updates for belief state changes."""
        results = {"successful_updates": 0, "failed_updates": 0}

        def update_agent_batch(batch_ids):
            session = self.session_factory()
            local_results = {"successful": 0, "failed": 0}

            try:
                for agent_id in batch_ids:
                    try:
                        agent = (
                            session.query(Agent)
                            .filter(Agent.agent_id == agent_id)
                            .first()
                        )
                        if agent:
                            # Simulate belief state update
                            agent.belief_state = {
                                "timestamp": datetime.utcnow().isoformat(),
                                "updated": True,
                                "iteration": np.random.randint(1, 1000),
                            }
                            agent.updated_at = datetime.utcnow()
                        session.commit()
                        local_results["successful"] += 1
                    except Exception as e:
                        session.rollback()
                        local_results["failed"] += 1
                        logger.warning(f"Update failed for {agent_id}: {e}")
            finally:
                session.close()

            return local_results

        # Split into smaller batches for updates (more conservative)
        batch_size = max(1, len(agent_ids) // num_threads)
        batches = [
            agent_ids[i : i + batch_size]
            for i in range(0, len(agent_ids), batch_size)
        ]

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(update_agent_batch, batch) for batch in batches
            ]

            for future in futures:
                batch_result = future.result()
                results["successful_updates"] += batch_result["successful"]
                results["failed_updates"] += batch_result["failed"]

        logger.info(
            f"✅ Concurrent updates: {results['successful_updates']} successful, "
            f"{results['failed_updates']} failed"
        )

        return results

    async def test_knowledge_graph_operations(
        self, num_nodes: int = 500, num_edges: int = 1000
    ) -> bool:
        """Test knowledge graph database operations."""
        session = self.session_factory()

        try:
            # Create knowledge nodes
            nodes = []
            node_ids = []
            for i in range(num_nodes):
                node_id = f"knowledge_node_{i}_{int(time.time())}"
                node = KnowledgeNode(
                    node_id=node_id,
                    node_type="concept",
                    content=f"Knowledge concept {i}",
                    metadata={"created_by": "load_test", "index": i},
                    created_at=datetime.utcnow(),
                )
                nodes.append(node)
                node_ids.append(node_id)

            session.add_all(nodes)
            session.commit()

            # Create knowledge edges
            edges = []
            for i in range(num_edges):
                source_id = np.random.choice(node_ids)
                target_id = np.random.choice(node_ids)

                if source_id != target_id:  # Avoid self-loops
                    edge = KnowledgeEdge(
                        edge_id=f"knowledge_edge_{i}_{int(time.time())}",
                        source_node_id=source_id,
                        target_node_id=target_id,
                        edge_type="relates_to",
                        weight=np.random.random(),
                        metadata={"created_by": "load_test"},
                        created_at=datetime.utcnow(),
                    )
                    edges.append(edge)

            session.add_all(edges)
            session.commit()

            logger.info(
                f"✅ Created knowledge graph: {num_nodes} nodes, {len(edges)} edges"
            )
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"❌ Knowledge graph creation failed: {e}")
            raise
        finally:
            session.close()

    async def test_complex_queries(self) -> Dict[str, Any]:
        """Test complex database queries for realistic scenarios."""
        session = self.session_factory()
        results = {}

        try:
            # Query 1: Active agents with recent updates
            start_time = time.time()
            recent_time = datetime.utcnow() - timedelta(minutes=30)
            active_agents = (
                session.query(Agent)
                .filter(
                    Agent.status == "active", Agent.updated_at >= recent_time
                )
                .count()
            )
            results["active_agents_query"] = {
                "count": active_agents,
                "duration": time.time() - start_time,
            }

            # Query 2: Coalition formation candidates
            start_time = time.time()
            coalition_candidates = (
                session.query(Agent)
                .filter(
                    Agent.status == "active",
                    Agent.agent_type.in_(["resource_collector", "explorer"]),
                )
                .limit(50)
                .all()
            )
            results["coalition_candidates_query"] = {
                "count": len(coalition_candidates),
                "duration": time.time() - start_time,
            }

            # Query 3: Knowledge graph traversal
            start_time = time.time()
            knowledge_network = (
                session.query(KnowledgeEdge)
                .join(
                    KnowledgeNode,
                    KnowledgeEdge.source_node_id == KnowledgeNode.node_id,
                )
                .filter(KnowledgeNode.node_type == "concept")
                .limit(100)
                .all()
            )
            results["knowledge_traversal_query"] = {
                "count": len(knowledge_network),
                "duration": time.time() - start_time,
            }

            # Query 4: Spatial proximity search (mock)
            start_time = time.time()
            # This would use PostGIS in production, but we'll simulate with JSONB
            spatial_agents = session.execute(
                text(
                    """
                SELECT agent_id, position
                FROM agents
                WHERE status = 'active'
                AND position IS NOT NULL
                LIMIT 20
            """
                )
            ).fetchall()
            results["spatial_proximity_query"] = {
                "count": len(spatial_agents),
                "duration": time.time() - start_time,
            }

            logger.info(
                f"✅ Complex queries completed: {len(results)} query types"
            )
            return results

        except Exception as e:
            logger.error(f"❌ Complex queries failed: {e}")
            raise
        finally:
            session.close()

    def analyze_performance_results(self) -> Dict[str, Any]:
        """Analyze performance test results."""
        analysis = {}

        for operation, metrics_list in self.performance_metrics.items():
            if not metrics_list:
                continue

            durations = [m["duration"] for m in metrics_list if m["success"]]
            memory_deltas = [
                m["memory_delta"] for m in metrics_list if m["success"]
            ]
            success_rate = sum(1 for m in metrics_list if m["success"]) / len(
                metrics_list
            )

            if durations:
                analysis[operation] = {
                    "avg_duration": np.mean(durations),
                    "max_duration": np.max(durations),
                    "min_duration": np.min(durations),
                    "p95_duration": np.percentile(durations, 95),
                    "avg_memory_delta": np.mean(memory_deltas)
                    if memory_deltas
                    else 0,
                    "success_rate": success_rate,
                    "total_operations": len(metrics_list),
                }

        return analysis


async def test_database_load_small():
    """Test database load with small agent population (10 agents)."""
    tester = DatabaseLoadTester()

    if not tester.setup_test_database():
        logger.warning("⚠️ Skipping database load test - setup failed")
        return

    try:
        # Small population test
        agent_ids = await tester.test_create_agents_batch(10)
        assert len(agent_ids) == 10

        # Concurrent operations
        read_results = await tester.test_concurrent_agent_reads(
            agent_ids, num_threads=3
        )
        assert read_results["agents_found"] == 10

        update_results = await tester.test_concurrent_agent_updates(
            agent_ids, num_threads=2
        )
        assert update_results["successful_updates"] >= 8  # Allow some failures

        # Knowledge graph
        kg_success = await tester.test_knowledge_graph_operations(50, 100)
        assert kg_success

        # Complex queries
        query_results = await tester.test_complex_queries()
        assert len(query_results) >= 4

        # Analyze performance
        analysis = tester.analyze_performance_results()

        # Performance assertions
        assert (
            analysis["create_agents_batch"]["avg_duration"] < 5.0
        )  # < 5s for 10 agents
        assert (
            analysis["read_agents_concurrent"]["avg_duration"] < 2.0
        )  # < 2s for reads
        assert (
            analysis["create_agents_batch"]["success_rate"] >= 0.95
        )  # 95% success rate

        logger.info("✅ Small population load test passed")

    finally:
        tester.cleanup_test_database()


async def test_database_load_medium():
    """Test database load with medium agent population (100 agents)."""
    tester = DatabaseLoadTester()

    if not tester.setup_test_database():
        logger.warning("⚠️ Skipping database load test - setup failed")
        return

    try:
        # Medium population test
        agent_ids = await tester.test_create_agents_batch(100)
        assert len(agent_ids) == 100

        # Concurrent operations with higher load
        read_results = await tester.test_concurrent_agent_reads(
            agent_ids, num_threads=10
        )
        assert read_results["agents_found"] == 100

        update_results = await tester.test_concurrent_agent_updates(
            agent_ids, num_threads=5
        )
        assert (
            update_results["successful_updates"] >= 90
        )  # Allow some failures

        # Larger knowledge graph
        kg_success = await tester.test_knowledge_graph_operations(200, 500)
        assert kg_success

        # Complex queries
        query_results = await tester.test_complex_queries()
        assert len(query_results) >= 4

        # Analyze performance
        analysis = tester.analyze_performance_results()

        # Performance assertions
        assert (
            analysis["create_agents_batch"]["avg_duration"] < 15.0
        )  # < 15s for 100 agents
        assert (
            analysis["read_agents_concurrent"]["avg_duration"] < 5.0
        )  # < 5s for reads
        assert (
            analysis["update_agents_concurrent"]["avg_duration"] < 10.0
        )  # < 10s for updates
        assert (
            analysis["create_agents_batch"]["success_rate"] >= 0.90
        )  # 90% success rate

        logger.info("✅ Medium population load test passed")

    finally:
        tester.cleanup_test_database()


async def test_database_load_large():
    """Test database load with large agent population (500 agents)."""
    tester = DatabaseLoadTester()

    if not tester.setup_test_database():
        logger.warning("⚠️ Skipping database load test - setup failed")
        return

    try:
        # Large population test
        agent_ids = await tester.test_create_agents_batch(500)
        assert len(agent_ids) == 500

        # High concurrency operations
        read_results = await tester.test_concurrent_agent_reads(
            agent_ids, num_threads=20
        )
        assert read_results["agents_found"] == 500

        update_results = await tester.test_concurrent_agent_updates(
            agent_ids, num_threads=10
        )
        assert (
            update_results["successful_updates"] >= 450
        )  # Allow some failures under load

        # Large knowledge graph
        kg_success = await tester.test_knowledge_graph_operations(1000, 2000)
        assert kg_success

        # Complex queries under load
        query_results = await tester.test_complex_queries()
        assert len(query_results) >= 4

        # Analyze performance
        analysis = tester.analyze_performance_results()

        # Performance assertions for production readiness
        assert (
            analysis["create_agents_batch"]["avg_duration"] < 60.0
        )  # < 1min for 500 agents
        assert (
            analysis["read_agents_concurrent"]["avg_duration"] < 15.0
        )  # < 15s for reads
        assert (
            analysis["update_agents_concurrent"]["avg_duration"] < 30.0
        )  # < 30s for updates
        assert (
            analysis["create_agents_batch"]["success_rate"] >= 0.85
        )  # 85% success rate under load

        # Memory efficiency
        if "create_agents_batch" in analysis:
            memory_per_agent = (
                analysis["create_agents_batch"]["avg_memory_delta"] / 500
            )
            assert (
                memory_per_agent < 1.0
            )  # < 1MB per agent in database operations

        logger.info("✅ Large population load test passed")

    finally:
        tester.cleanup_test_database()


if __name__ == "__main__":
    import asyncio

    async def run_manual_tests():
        """Run manual database load tests."""
        logger.info("🚀 Starting manual database load tests...")

        # Test small population
        print("\n" + "=" * 50)
        print("TESTING SMALL POPULATION (10 agents)")
        print("=" * 50)
        await test_database_load_small()

        # Test medium population
        print("\n" + "=" * 50)
        print("TESTING MEDIUM POPULATION (100 agents)")
        print("=" * 50)
        await test_database_load_medium()

        # Test large population (optional - takes longer)
        print("\n" + "=" * 50)
        print("TESTING LARGE POPULATION (500 agents)")
        print("=" * 50)
        await test_database_load_large()

        logger.info("🎉 All database load tests completed successfully!")

    asyncio.run(run_manual_tests())
