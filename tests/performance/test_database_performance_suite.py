"""Comprehensive Database Performance Test Suite for Multi-Agent Systems.

Implements realistic database benchmarks following CLAUDE.md specifications:
- Connection pooling formula: (num_cores * 2) + effective IO wait
- SERIALIZABLE isolation for critical transactions, READ_COMMITTED elsewhere
- Partial indexes on deleted_at IS NULL
- EXPLAIN ANALYZE for queries >30ms
- Comprehensive observability and production simulation
"""

import asyncio
import json
import logging
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

from database.models import (
    Agent,
    AgentConversationMessage,
    AgentConversationSession,
    AgentStatus,
    Coalition,
    CoalitionStatus,
    KnowledgeEdge,
    KnowledgeNode,
)
from tests.db_infrastructure.factories import (
    AgentFactory,
    KnowledgeGraphFactory,
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


@dataclass
class PerformanceMetrics:
    """Performance metrics collection with statistical analysis."""
    
    operation_name: str
    measurements: List[float] = field(default_factory=list)
    start_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def start_measurement(self):
        """Start timing measurement."""
        self.start_time = time.perf_counter()
    
    def end_measurement(self, **metadata):
        """End timing measurement and record result."""
        if self.start_time is None:
            raise ValueError("start_measurement() must be called first")
        
        duration = time.perf_counter() - self.start_time
        self.measurements.append(duration)
        self.metadata.update(metadata)
        self.start_time = None
        return duration
    
    def get_statistics(self) -> Dict[str, float]:
        """Get statistical analysis of measurements."""
        if not self.measurements:
            return {}
        
        measurements = sorted(self.measurements)
        return {
            "count": len(measurements),
            "min": min(measurements),
            "max": max(measurements),
            "mean": statistics.mean(measurements),
            "median": statistics.median(measurements),
            "p50": measurements[int(len(measurements) * 0.5)],
            "p90": measurements[int(len(measurements) * 0.9)],
            "p95": measurements[int(len(measurements) * 0.95)],
            "p99": measurements[int(len(measurements) * 0.99)] if len(measurements) >= 100 else measurements[-1],
            "stdev": statistics.stdev(measurements) if len(measurements) > 1 else 0.0,
        }


class ConnectionPoolBenchmark:
    """Test connection pool performance per CLAUDE.md specifications."""
    
    def __init__(self, database_url: str, use_sqlite: bool = False):
        self.database_url = database_url
        self.use_sqlite = use_sqlite
        self.metrics = {}
    
    def create_pooled_engine(self, pool_size: int, max_overflow: int = 10):
        """Create engine with specific pool configuration."""
        if self.use_sqlite:
            # SQLite doesn't support connection pooling same way
            return create_engine(self.database_url, echo=False)
        
        return create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,  # 1 hour
            echo=False,
        )
    
    async def test_pool_size_formula(self, num_cores: int = 4, io_wait: float = 0.1) -> Dict[str, Any]:
        """Test CLAUDE.md connection pool formula: (num_cores * 2) + effective IO wait."""
        recommended_size = int((num_cores * 2) + (io_wait * 100))  # Convert IO wait to reasonable number
        test_sizes = [
            recommended_size // 2,  # Under-provisioned
            recommended_size,       # Recommended
            recommended_size * 2,   # Over-provisioned
        ]
        
        results = {}
        
        for pool_size in test_sizes:
            logger.info(f"Testing pool size: {pool_size}")
            engine = self.create_pooled_engine(pool_size)
            
            try:
                setup_test_database(engine)
                
                # Measure connection acquisition time under load
                metrics = PerformanceMetrics(f"pool_size_{pool_size}")
                
                async def connection_benchmark():
                    """Benchmark connection acquisition."""
                    Session = sessionmaker(bind=engine)
                    
                    for _ in range(100):  # 100 connection acquisitions
                        metrics.start_measurement()
                        session = Session()
                        try:
                            # Simple query to ensure connection is active
                            session.execute(text("SELECT 1"))
                        finally:
                            session.close()
                        metrics.end_measurement()
                
                await connection_benchmark()
                results[f"pool_size_{pool_size}"] = metrics.get_statistics()
                
            finally:
                teardown_test_database(engine)
                engine.dispose()
        
        # Determine optimal pool size based on P95 latency
        optimal_size = min(results.keys(), key=lambda k: results[k].get("p95", float("inf")))
        
        return {
            "recommended_size": recommended_size,
            "optimal_size": optimal_size,
            "results": results,
            "formula_validation": optimal_size == f"pool_size_{recommended_size}",
        }


class MultiAgentTransactionBenchmark:
    """Benchmark multi-agent transaction scenarios."""
    
    def __init__(self, engine):
        self.engine = engine
        self.metrics = {}
    
    async def test_belief_state_updates(self, num_agents: int = 100, concurrent_updates: int = 20) -> Dict[str, Any]:
        """Test concurrent belief state updates - core multi-agent operation."""
        Session = sessionmaker(bind=self.engine)
        
        # Create agents
        with isolated_db_test(self.engine) as session:
            agents = AgentFactory.create_batch(
                session,
                count=num_agents,
                template="belief_updater",
                status=AgentStatus.ACTIVE,
            )
            agent_ids = [agent.id for agent in agents]
            session.commit()
        
        metrics = PerformanceMetrics("belief_state_updates")
        conflict_count = 0
        success_count = 0
        
        def update_belief_batch(batch_agent_ids):
            """Update belief states with SERIALIZABLE isolation for consistency."""
            nonlocal conflict_count, success_count
            session = Session()
            
            try:
                # Use SERIALIZABLE isolation for belief state consistency
                session.execute(text("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE"))
                
                for agent_id in batch_agent_ids:
                    metrics.start_measurement()
                    try:
                        agent = session.query(Agent).filter(Agent.id == agent_id).with_for_update().first()
                        
                        if agent:
                            # Simulate Active Inference belief update
                            agent.beliefs = {
                                "timestamp": datetime.utcnow().isoformat(),
                                "posterior": [0.3, 0.4, 0.2, 0.1],  # Example belief distribution
                                "prior": [0.25, 0.25, 0.25, 0.25],
                                "likelihood": [0.8, 0.6, 0.4, 0.2],
                                "evidence": 0.85,
                                "free_energy": -2.3,
                                "update_count": (agent.inference_count or 0) + 1,
                            }
                            agent.inference_count = (agent.inference_count or 0) + 1
                            agent.updated_at = datetime.utcnow()
                        
                        session.commit()
                        success_count += 1
                        
                    except Exception as e:
                        session.rollback()
                        if "serialization failure" in str(e).lower() or "deadlock" in str(e).lower():
                            conflict_count += 1
                        else:
                            logger.error(f"Belief update error: {e}")
                    
                    metrics.end_measurement()
            
            finally:
                session.close()
        
        # Run concurrent belief updates
        batch_size = max(1, num_agents // concurrent_updates)
        batches = [agent_ids[i:i + batch_size] for i in range(0, len(agent_ids), batch_size)]
        
        with ThreadPoolExecutor(max_workers=concurrent_updates) as executor:
            futures = [executor.submit(update_belief_batch, batch) for batch in batches]
            
            for future in as_completed(futures):
                future.result()  # Wait for completion
        
        stats = metrics.get_statistics()
        stats.update({
            "success_count": success_count,
            "conflict_count": conflict_count,
            "success_rate": success_count / (success_count + conflict_count) if (success_count + conflict_count) > 0 else 0,
        })
        
        return stats
    
    async def test_coalition_formation(self, num_agents: int = 50, num_coalitions: int = 10) -> Dict[str, Any]:
        """Test coalition formation transactions - complex multi-agent coordination."""
        Session = sessionmaker(bind=self.engine)
        
        # Create agents and coalitions
        with isolated_db_test(self.engine) as session:
            agents = AgentFactory.create_batch(
                session,
                count=num_agents,
                template="coalition_member",
                status=AgentStatus.ACTIVE,
            )
            
            coalitions = []
            for i in range(num_coalitions):
                coalition = Coalition(
                    name=f"Coalition_{i}",
                    description=f"Test coalition {i}",
                    status=CoalitionStatus.FORMING,
                    objectives={"goal": f"objective_{i}"},
                    required_capabilities=["coordination", "inference"],
                )
                session.add(coalition)
                coalitions.append(coalition)
            
            session.commit()
            agent_ids = [agent.id for agent in agents]
            coalition_ids = [coalition.id for coalition in coalitions]
        
        metrics = PerformanceMetrics("coalition_formation")
        formation_count = 0
        
        def form_coalition_batch(coalition_batch):
            """Form coalitions with proper transaction handling."""
            nonlocal formation_count
            session = Session()
            
            try:
                for coalition_id in coalition_batch:
                    metrics.start_measurement()
                    
                    # Select random agents for coalition
                    import random
                    selected_agent_ids = random.sample(agent_ids, min(5, len(agent_ids)))
                    
                    coalition = session.query(Coalition).filter(Coalition.id == coalition_id).first()
                    agents_to_add = session.query(Agent).filter(Agent.id.in_(selected_agent_ids)).all()
                    
                    if coalition and agents_to_add:
                        # Add agents to coalition
                        for agent in agents_to_add:
                            if agent not in coalition.agents:
                                coalition.agents.append(agent)
                        
                        coalition.status = CoalitionStatus.ACTIVE
                        coalition.updated_at = datetime.utcnow()
                        
                        session.commit()
                        formation_count += 1
                    
                    metrics.end_measurement()
            
            except Exception as e:
                session.rollback()
                logger.error(f"Coalition formation error: {e}")
            
            finally:
                session.close()
        
        # Run coalition formation concurrently
        batch_size = max(1, num_coalitions // 3)  # 3 concurrent formation threads
        batches = [coalition_ids[i:i + batch_size] for i in range(0, len(coalition_ids), batch_size)]
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(form_coalition_batch, batch) for batch in batches]
            
            for future in as_completed(futures):
                future.result()
        
        stats = metrics.get_statistics()
        stats["formations_completed"] = formation_count
        
        return stats


class QueryPerformanceBenchmark:
    """Benchmark query performance with EXPLAIN ANALYZE integration."""
    
    def __init__(self, engine, use_sqlite: bool = False):
        self.engine = engine
        self.use_sqlite = use_sqlite
        self.slow_queries = []  # Track queries >30ms per CLAUDE.md
    
    async def test_complex_queries_with_analysis(self) -> Dict[str, Any]:
        """Test complex queries with EXPLAIN ANALYZE for optimization."""
        Session = sessionmaker(bind=self.engine)
        session = Session()
        
        results = {}
        
        try:
            # Query 1: Active agents with belief analysis (aggregation + JSON)
            query_name = "active_agents_belief_analysis"
            metrics = PerformanceMetrics(query_name)
            
            metrics.start_measurement()
            
            if not self.use_sqlite:
                # PostgreSQL version with JSON operations
                query_result = session.execute(text("""
                    EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
                    SELECT 
                        a.template,
                        COUNT(*) as agent_count,
                        AVG(a.inference_count) as avg_inferences,
                        AVG(CAST(a.beliefs->>'evidence' AS FLOAT)) as avg_evidence
                    FROM agents a
                    WHERE a.status = 'active'
                      AND a.beliefs IS NOT NULL
                      AND a.updated_at >= NOW() - INTERVAL '1 hour'
                    GROUP BY a.template
                    ORDER BY agent_count DESC;
                """))
                
                explain_result = query_result.fetchone()[0]
            else:
                # SQLite version (simplified)
                query_result = session.execute(text("""
                    EXPLAIN QUERY PLAN
                    SELECT 
                        template,
                        COUNT(*) as agent_count,
                        AVG(inference_count) as avg_inferences
                    FROM agents
                    WHERE status = 'active'
                      AND beliefs IS NOT NULL
                    GROUP BY template;
                """))
                
                explain_result = [dict(row._mapping) for row in query_result.fetchall()]
            
            duration = metrics.end_measurement()
            
            # Track slow queries per CLAUDE.md requirement (>30ms)
            if duration > 0.030:  # 30ms threshold
                self.slow_queries.append({
                    "query": query_name,
                    "duration_ms": duration * 1000,
                    "explain_plan": explain_result,
                })
            
            results[query_name] = {
                "duration_ms": duration * 1000,
                "explain_plan": explain_result,
                "is_slow": duration > 0.030,
            }
            
            # Query 2: Knowledge graph traversal (complex joins)
            query_name = "knowledge_graph_traversal"
            metrics = PerformanceMetrics(query_name)
            
            metrics.start_measurement()
            
            if not self.use_sqlite:
                query_result = session.execute(text("""
                    EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
                    WITH RECURSIVE graph_traversal AS (
                        -- Base case: start nodes
                        SELECT kn.node_id, kn.node_type, kn.label, 0 as depth
                        FROM db_knowledge_nodes kn
                        WHERE kn.node_type = 'concept'
                        LIMIT 10
                        
                        UNION ALL
                        
                        -- Recursive case: connected nodes
                        SELECT kn.node_id, kn.node_type, kn.label, gt.depth + 1
                        FROM db_knowledge_nodes kn
                        JOIN db_knowledge_edges ke ON kn.node_id = ke.target_node_id
                        JOIN graph_traversal gt ON ke.source_node_id = gt.node_id
                        WHERE gt.depth < 3
                    )
                    SELECT node_type, depth, COUNT(*) as node_count
                    FROM graph_traversal
                    GROUP BY node_type, depth
                    ORDER BY depth, node_count DESC;
                """))
                
                explain_result = query_result.fetchone()[0]
            else:
                # Simplified SQLite version
                query_result = session.execute(text("""
                    EXPLAIN QUERY PLAN
                    SELECT kn1.node_type, COUNT(*) as connected_count
                    FROM db_knowledge_nodes kn1
                    JOIN db_knowledge_edges ke ON kn1.node_id = ke.source_node_id
                    JOIN db_knowledge_nodes kn2 ON ke.target_node_id = kn2.node_id
                    GROUP BY kn1.node_type;
                """))
                
                explain_result = [dict(row._mapping) for row in query_result.fetchall()]
            
            duration = metrics.end_measurement()
            
            if duration > 0.030:
                self.slow_queries.append({
                    "query": query_name,
                    "duration_ms": duration * 1000,
                    "explain_plan": explain_result,
                })
            
            results[query_name] = {
                "duration_ms": duration * 1000,
                "explain_plan": explain_result,
                "is_slow": duration > 0.030,
            }
            
            # Query 3: Coalition performance analysis
            query_name = "coalition_performance_analysis"
            metrics = PerformanceMetrics(query_name)
            
            metrics.start_measurement()
            
            query_result = session.execute(text("""
                SELECT 
                    c.name,
                    c.status,
                    COUNT(ac.agent_id) as member_count,
                    AVG(ac.contribution_score) as avg_contribution,
                    AVG(ac.trust_score) as avg_trust
                FROM coalitions c
                LEFT JOIN agent_coalition ac ON c.id = ac.coalition_id
                WHERE c.status IN ('active', 'forming')
                GROUP BY c.id, c.name, c.status
                HAVING COUNT(ac.agent_id) > 0
                ORDER BY avg_contribution DESC;
            """))
            
            duration = metrics.end_measurement()
            
            if duration > 0.030:
                self.slow_queries.append({
                    "query": query_name,
                    "duration_ms": duration * 1000,
                })
            
            results[query_name] = {
                "duration_ms": duration * 1000,
                "is_slow": duration > 0.030,
            }
            
        finally:
            session.close()
        
        return {
            "query_results": results,
            "slow_queries": self.slow_queries,
            "total_slow_queries": len(self.slow_queries),
        }


class DatabasePerformanceTestSuite:
    """Comprehensive database performance test suite."""
    
    def __init__(self, use_sqlite: bool = False):
        self.use_sqlite = use_sqlite
        self.engine = create_test_engine(use_sqlite=use_sqlite)
        self.test_results = {}
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run complete database performance benchmark suite."""
        logger.info("🚀 Starting comprehensive database performance benchmark")
        
        setup_test_database(self.engine)
        
        try:
            # 1. Connection Pool Performance
            logger.info("Phase 1: Connection Pool Benchmarks")
            pool_benchmark = ConnectionPoolBenchmark(
                self.engine.url.render_as_string(hide_password=False),
                use_sqlite=self.use_sqlite
            )
            self.test_results["connection_pool"] = await pool_benchmark.test_pool_size_formula()
            
            # 2. Multi-Agent Transaction Scenarios
            logger.info("Phase 2: Multi-Agent Transaction Benchmarks")
            transaction_benchmark = MultiAgentTransactionBenchmark(self.engine)
            
            self.test_results["belief_updates"] = await transaction_benchmark.test_belief_state_updates(
                num_agents=200, concurrent_updates=10
            )
            
            self.test_results["coalition_formation"] = await transaction_benchmark.test_coalition_formation(
                num_agents=100, num_coalitions=20
            )
            
            # 3. Query Performance with EXPLAIN ANALYZE
            logger.info("Phase 3: Query Performance Analysis")
            query_benchmark = QueryPerformanceBenchmark(self.engine, use_sqlite=self.use_sqlite)
            self.test_results["query_performance"] = await query_benchmark.test_complex_queries_with_analysis()
            
            # 4. Generate Performance Report
            report = self.generate_performance_report()
            
            logger.info("✅ Comprehensive database performance benchmark completed")
            
            return {
                "test_results": self.test_results,
                "performance_report": report,
                "success": True,
            }
            
        except Exception as e:
            logger.error(f"Database performance benchmark failed: {e}")
            return {
                "test_results": self.test_results,
                "error": str(e),
                "success": False,
            }
        
        finally:
            teardown_test_database(self.engine)
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance analysis report."""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "database_type": "SQLite" if self.use_sqlite else "PostgreSQL",
            "summary": {},
            "recommendations": [],
        }
        
        # Connection Pool Analysis
        if "connection_pool" in self.test_results:
            pool_result = self.test_results["connection_pool"]
            report["summary"]["connection_pool"] = {
                "formula_validated": pool_result.get("formula_validation", False),
                "recommended_size": pool_result.get("recommended_size"),
                "optimal_size": pool_result.get("optimal_size"),
            }
            
            if not pool_result.get("formula_validation", False):
                report["recommendations"].append(
                    "Consider adjusting connection pool size formula based on benchmark results"
                )
        
        # Transaction Performance Analysis
        if "belief_updates" in self.test_results:
            belief_stats = self.test_results["belief_updates"]
            success_rate = belief_stats.get("success_rate", 0)
            p95_latency = belief_stats.get("p95", 0)
            
            report["summary"]["belief_updates"] = {
                "success_rate": success_rate,
                "p95_latency_ms": p95_latency * 1000,
                "conflict_rate": 1 - success_rate,
            }
            
            if success_rate < 0.95:
                report["recommendations"].append(
                    "High transaction conflict rate detected - consider optimizing belief update patterns"
                )
            
            if p95_latency > 0.200:  # >200ms per CLAUDE.md
                report["recommendations"].append(
                    "Belief update P95 latency exceeds 200ms threshold - database optimization needed"
                )
        
        # Query Performance Analysis
        if "query_performance" in self.test_results:
            query_stats = self.test_results["query_performance"]
            slow_query_count = query_stats.get("total_slow_queries", 0)
            
            report["summary"]["query_performance"] = {
                "slow_queries_detected": slow_query_count,
                "slow_query_threshold_ms": 30,
            }
            
            if slow_query_count > 0:
                report["recommendations"].append(
                    f"Detected {slow_query_count} queries exceeding 30ms - review EXPLAIN plans for optimization"
                )
        
        return report


@pytest.mark.performance
@pytest.mark.db_test
class TestDatabasePerformanceSuite:
    """Test cases for comprehensive database performance suite."""
    
    @pytest.mark.asyncio
    async def test_postgresql_performance_suite(self):
        """Test complete performance suite with PostgreSQL."""
        suite = DatabasePerformanceTestSuite(use_sqlite=False)
        results = await suite.run_comprehensive_benchmark()
        
        assert results["success"], f"Benchmark failed: {results.get('error')}"
        assert "test_results" in results
        assert "performance_report" in results
        
        # Validate connection pool results
        assert "connection_pool" in results["test_results"]
        pool_result = results["test_results"]["connection_pool"]
        assert "recommended_size" in pool_result
        assert "optimal_size" in pool_result
        
        # Validate transaction benchmarks
        assert "belief_updates" in results["test_results"]
        belief_result = results["test_results"]["belief_updates"]
        assert belief_result["success_rate"] > 0.8  # At least 80% success rate
        assert belief_result["p95"] < 1.0  # P95 latency under 1 second
        
        # Validate query performance
        assert "query_performance" in results["test_results"]
        query_result = results["test_results"]["query_performance"]
        assert "query_results" in query_result
        assert "slow_queries" in query_result
    
    @pytest.mark.asyncio
    async def test_sqlite_compatibility(self):
        """Test performance suite compatibility with SQLite for development."""
        suite = DatabasePerformanceTestSuite(use_sqlite=True)
        results = await suite.run_comprehensive_benchmark()
        
        assert results["success"], f"SQLite benchmark failed: {results.get('error')}"
        
        # Validate that basic functionality works with SQLite
        assert "test_results" in results
        assert "performance_report" in results
        
        report = results["performance_report"]
        assert report["database_type"] == "SQLite"
    
    @pytest.mark.asyncio
    async def test_performance_regression_detection(self):
        """Test that performance regression detection works."""
        suite = DatabasePerformanceTestSuite(use_sqlite=False)
        results = await suite.run_comprehensive_benchmark()
        
        assert results["success"]
        
        report = results["performance_report"]
        
        # Check that recommendations are generated for performance issues
        if "recommendations" in report:
            for recommendation in report["recommendations"]:
                assert isinstance(recommendation, str)
                assert len(recommendation) > 0
        
        # Validate performance thresholds are enforced
        summary = report.get("summary", {})
        
        if "belief_updates" in summary:
            belief_summary = summary["belief_updates"]
            if belief_summary.get("p95_latency_ms", 0) > 200:
                assert any("200ms" in rec for rec in report.get("recommendations", []))


if __name__ == "__main__":
    import asyncio
    
    async def run_manual_benchmark():
        """Run database performance benchmark manually."""
        print("Running comprehensive database performance benchmark...\n")
        
        # Test with PostgreSQL
        print("Testing with PostgreSQL:")
        suite = DatabasePerformanceTestSuite(use_sqlite=False)
        results = await suite.run_comprehensive_benchmark()
        
        if results["success"]:
            print("\n" + "=" * 60)
            print("DATABASE PERFORMANCE BENCHMARK REPORT")
            print("=" * 60)
            
            report = results["performance_report"]
            print(f"Database Type: {report['database_type']}")
            print(f"Timestamp: {report['timestamp']}")
            
            print("\nSummary:")
            for category, data in report["summary"].items():
                print(f"\n{category.replace('_', ' ').title()}:")
                for key, value in data.items():
                    print(f"  {key}: {value}")
            
            if report["recommendations"]:
                print("\nRecommendations:")
                for i, rec in enumerate(report["recommendations"], 1):
                    print(f"  {i}. {rec}")
            else:
                print("\n✅ No performance issues detected")
            
            # Detailed results
            print("\nDetailed Results:")
            print(json.dumps(results["test_results"], indent=2, default=str))
        
        else:
            print(f"❌ Benchmark failed: {results.get('error')}")
    
    asyncio.run(run_manual_benchmark())
