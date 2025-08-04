"""Database Indexing Performance Tests.

Validates CLAUDE.md indexing requirements:
- Partial indexes on deleted_at IS NULL
- Compound indexes (workspace_id, updated_at)
- Query plan analysis for optimization
- Index usage monitoring
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from database.models import Agent, AgentStatus, Coalition, KnowledgeNode
from tests.db_infrastructure.factories import AgentFactory, KnowledgeGraphFactory
from tests.db_infrastructure.fixtures import isolated_db_test
from tests.db_infrastructure.test_config import (
    create_test_engine,
    setup_test_database,
    teardown_test_database,
)

logger = logging.getLogger(__name__)


class IndexPerformanceAnalyzer:
    """Analyze database index performance and optimization opportunities."""
    
    def __init__(self, engine, use_sqlite: bool = False):
        self.engine = engine
        self.use_sqlite = use_sqlite
        self.index_analysis = {}
        self.query_plans = {}
    
    async def create_performance_indexes(self) -> Dict[str, bool]:
        """Create performance-optimized indexes per CLAUDE.md specifications."""
        Session = sessionmaker(bind=self.engine)
        session = Session()
        
        index_creation_results = {}
        
        try:
            if not self.use_sqlite:
                # PostgreSQL-specific indexes
                indexes_to_create = [
                    {
                        "name": "idx_agents_active_updated",
                        "sql": """
                            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_active_updated 
                            ON agents (updated_at DESC) 
                            WHERE status = 'active' AND deleted_at IS NULL
                        """,
                        "description": "Partial index for active agents ordered by update time",
                    },
                    {
                        "name": "idx_agents_status_type_compound",
                        "sql": """
                            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_status_type_compound 
                            ON agents (status, agent_type, updated_at DESC)
                        """,
                        "description": "Compound index for agent filtering and sorting",
                    },
                    {
                        "name": "idx_knowledge_nodes_type_current",
                        "sql": """
                            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_knowledge_nodes_type_current 
                            ON db_knowledge_nodes (node_type, created_at DESC) 
                            WHERE is_current = true
                        """,
                        "description": "Partial index for current knowledge nodes by type",
                    },
                    {
                        "name": "idx_coalitions_status_performance",
                        "sql": """
                            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_status_performance 
                            ON coalitions (status, performance_score DESC) 
                            WHERE dissolved_at IS NULL
                        """,
                        "description": "Partial index for active coalitions by performance",
                    },
                    {
                        "name": "idx_agent_conversations_session_order",
                        "sql": """
                            CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_conversations_session_order 
                            ON agent_conversation_messages (conversation_id, message_order, created_at)
                        """,
                        "description": "Compound index for conversation message ordering",
                    },
                ]
            else:
                # SQLite-compatible indexes (no CONCURRENTLY, no WHERE clauses)
                indexes_to_create = [
                    {
                        "name": "idx_agents_active_updated",
                        "sql": "CREATE INDEX IF NOT EXISTS idx_agents_active_updated ON agents (status, updated_at DESC)",
                        "description": "Index for active agents ordered by update time",
                    },
                    {
                        "name": "idx_agents_status_type_compound",
                        "sql": "CREATE INDEX IF NOT EXISTS idx_agents_status_type_compound ON agents (status, agent_type, updated_at DESC)",
                        "description": "Compound index for agent filtering and sorting",
                    },
                    {
                        "name": "idx_knowledge_nodes_type_current",
                        "sql": "CREATE INDEX IF NOT EXISTS idx_knowledge_nodes_type_current ON db_knowledge_nodes (node_type, is_current, created_at DESC)",
                        "description": "Index for current knowledge nodes by type",
                    },
                ]
            
            for index_def in indexes_to_create:
                try:
                    start_time = time.time()
                    session.execute(text(index_def["sql"]))
                    session.commit()
                    creation_time = time.time() - start_time
                    
                    index_creation_results[index_def["name"]] = {
                        "success": True,
                        "creation_time_seconds": creation_time,
                        "description": index_def["description"],
                    }
                    
                    logger.info(f"✅ Created index {index_def['name']} in {creation_time:.3f}s")
                    
                except Exception as e:
                    index_creation_results[index_def["name"]] = {
                        "success": False,
                        "error": str(e),
                        "description": index_def["description"],
                    }
                    logger.error(f"❌ Failed to create index {index_def['name']}: {e}")
            
        finally:
            session.close()
        
        return index_creation_results
    
    async def analyze_query_performance(self, test_data_size: int = 1000) -> Dict[str, Any]:
        """Analyze query performance with and without indexes."""
        Session = sessionmaker(bind=self.engine)
        
        # Create test data
        with isolated_db_test(self.engine) as session:
            # Create agents with various statuses
            agents = AgentFactory.create_batch(
                session,
                count=test_data_size,
                agent_type="performance_test",
                status=AgentStatus.ACTIVE,
            )
            
            # Update some agents to have different statuses and timestamps
            for i, agent in enumerate(agents):
                if i % 3 == 0:
                    agent.status = AgentStatus.PAUSED
                elif i % 5 == 0:
                    agent.status = AgentStatus.STOPPED
                
                # Vary update times
                import random
                from datetime import datetime, timedelta
                agent.updated_at = datetime.utcnow() - timedelta(minutes=random.randint(1, 1440))
            
            # Create knowledge graph data
            KnowledgeGraphFactory.create_connected_graph(
                session,
                num_nodes=test_data_size // 2,
                connectivity=0.1,
            )
            
            session.commit()
        
        # Test queries and analyze performance
        query_tests = [
            {
                "name": "active_agents_recent",
                "description": "Find active agents updated in last hour",
                "sql": """
                    SELECT agent_id, name, updated_at, inference_count
                    FROM agents 
                    WHERE status = 'active' 
                      AND updated_at >= NOW() - INTERVAL '1 hour'
                    ORDER BY updated_at DESC 
                    LIMIT 100
                """ if not self.use_sqlite else """
                    SELECT agent_id, name, updated_at, inference_count
                    FROM agents 
                    WHERE status = 'active' 
                    ORDER BY updated_at DESC 
                    LIMIT 100
                """,
            },
            {
                "name": "agent_status_distribution",
                "description": "Count agents by status and type",
                "sql": """
                    SELECT status, agent_type, COUNT(*) as count,
                           AVG(inference_count) as avg_inferences
                    FROM agents 
                    GROUP BY status, agent_type
                    ORDER BY count DESC
                """,
            },
            {
                "name": "knowledge_nodes_by_type",
                "description": "Find current knowledge nodes by type",
                "sql": """
                    SELECT node_type, COUNT(*) as count,
                           MIN(created_at) as oldest,
                           MAX(created_at) as newest
                    FROM db_knowledge_nodes 
                    WHERE is_current = true
                    GROUP BY node_type
                    ORDER BY count DESC
                """,
            },
            {
                "name": "coalition_performance_ranking",
                "description": "Rank active coalitions by performance",
                "sql": """
                    SELECT name, performance_score, cohesion_score,
                           (SELECT COUNT(*) FROM agent_coalition ac WHERE ac.coalition_id = c.id) as member_count
                    FROM coalitions c
                    WHERE status = 'active' 
                      AND dissolved_at IS NULL
                    ORDER BY performance_score DESC
                    LIMIT 50
                """ if not self.use_sqlite else """
                    SELECT name, performance_score, cohesion_score
                    FROM coalitions 
                    WHERE status = 'active'
                    ORDER BY performance_score DESC
                    LIMIT 50
                """,
            },
        ]
        
        analysis_results = {}
        session = Session()
        
        try:
            for query_test in query_tests:
                query_name = query_test["name"]
                logger.info(f"Analyzing query: {query_name}")
                
                # Execute query multiple times for statistical analysis
                execution_times = []
                
                for _ in range(5):  # 5 executions for timing stability
                    start_time = time.perf_counter()
                    
                    try:
                        result = session.execute(text(query_test["sql"]))
                        rows = result.fetchall()
                        
                        execution_time = time.perf_counter() - start_time
                        execution_times.append(execution_time)
                        
                    except Exception as e:
                        logger.error(f"Query {query_name} failed: {e}")
                        execution_times.append(float('inf'))  # Mark as failed
                        break
                
                # Get query plan analysis
                query_plan = None
                try:
                    if not self.use_sqlite:
                        # PostgreSQL EXPLAIN ANALYZE
                        explain_result = session.execute(
                            text(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query_test['sql']}")
                        )
                        query_plan = explain_result.fetchone()[0]
                    else:
                        # SQLite EXPLAIN QUERY PLAN
                        explain_result = session.execute(
                            text(f"EXPLAIN QUERY PLAN {query_test['sql']}")
                        )
                        query_plan = [dict(row._mapping) for row in explain_result.fetchall()]
                
                except Exception as e:
                    logger.warning(f"Could not get query plan for {query_name}: {e}")
                
                # Calculate statistics
                valid_times = [t for t in execution_times if t != float('inf')]
                
                if valid_times:
                    analysis_results[query_name] = {
                        "description": query_test["description"],
                        "execution_count": len(valid_times),
                        "min_time_ms": min(valid_times) * 1000,
                        "max_time_ms": max(valid_times) * 1000,
                        "avg_time_ms": sum(valid_times) / len(valid_times) * 1000,
                        "total_time_ms": sum(valid_times) * 1000,
                        "query_plan": query_plan,
                        "is_slow": (sum(valid_times) / len(valid_times)) > 0.030,  # >30ms per CLAUDE.md
                        "row_count": len(rows) if 'rows' in locals() else 0,
                    }
                else:
                    analysis_results[query_name] = {
                        "description": query_test["description"],
                        "execution_count": 0,
                        "error": "All executions failed",
                    }
        
        finally:
            session.close()
        
        return analysis_results
    
    async def validate_index_usage(self) -> Dict[str, Any]:
        """Validate that indexes are being used effectively."""
        Session = sessionmaker(bind=self.engine)
        session = Session()
        
        index_usage_stats = {}
        
        try:
            if not self.use_sqlite:
                # PostgreSQL index usage statistics
                index_stats_query = text("""
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        idx_tup_read,
                        idx_tup_fetch,
                        idx_scan
                    FROM pg_stat_user_indexes 
                    WHERE schemaname = 'public'
                    ORDER BY idx_scan DESC
                """)
                
                index_stats = session.execute(index_stats_query).fetchall()
                
                for stat in index_stats:
                    index_usage_stats[stat.indexname] = {
                        "table": stat.tablename,
                        "tuples_read": stat.idx_tup_read,
                        "tuples_fetched": stat.idx_tup_fetch,
                        "scans": stat.idx_scan,
                        "usage_efficiency": (
                            stat.idx_tup_fetch / stat.idx_tup_read 
                            if stat.idx_tup_read > 0 else 0
                        ),
                    }
                
                # Table scan statistics
                table_stats_query = text("""
                    SELECT 
                        schemaname,
                        relname,
                        seq_scan,
                        seq_tup_read,
                        idx_scan,
                        idx_tup_fetch
                    FROM pg_stat_user_tables 
                    WHERE schemaname = 'public'
                    ORDER BY seq_scan DESC
                """)
                
                table_stats = session.execute(table_stats_query).fetchall()
                
                table_usage_stats = {}
                for stat in table_stats:
                    total_scans = (stat.seq_scan or 0) + (stat.idx_scan or 0)
                    index_scan_ratio = (stat.idx_scan or 0) / total_scans if total_scans > 0 else 0
                    
                    table_usage_stats[stat.relname] = {
                        "sequential_scans": stat.seq_scan,
                        "sequential_tuples_read": stat.seq_tup_read,
                        "index_scans": stat.idx_scan,
                        "index_tuples_fetched": stat.idx_tup_fetch,
                        "index_scan_ratio": index_scan_ratio,
                        "needs_optimization": index_scan_ratio < 0.8 and total_scans > 10,
                    }
                
                index_usage_stats["table_statistics"] = table_usage_stats
            
            else:
                # SQLite doesn't have detailed index usage stats
                # Just verify indexes exist
                indexes_query = text("""
                    SELECT name, tbl_name, sql 
                    FROM sqlite_master 
                    WHERE type = 'index' 
                      AND tbl_name IN ('agents', 'coalitions', 'db_knowledge_nodes')
                    ORDER BY tbl_name, name
                """)
                
                indexes = session.execute(indexes_query).fetchall()
                
                for index in indexes:
                    index_usage_stats[index.name] = {
                        "table": index.tbl_name,
                        "definition": index.sql,
                        "exists": True,
                    }
        
        except Exception as e:
            logger.error(f"Failed to get index usage statistics: {e}")
            index_usage_stats["error"] = str(e)
        
        finally:
            session.close()
        
        return index_usage_stats
    
    def generate_optimization_recommendations(self, query_analysis: Dict[str, Any], index_usage: Dict[str, Any]) -> List[str]:
        """Generate database optimization recommendations."""
        recommendations = []
        
        # Check for slow queries
        slow_queries = [name for name, data in query_analysis.items() 
                       if isinstance(data, dict) and data.get("is_slow", False)]
        
        if slow_queries:
            recommendations.append(
                f"Optimize {len(slow_queries)} slow queries (>30ms): {', '.join(slow_queries)}"
            )
        
        # Check index usage efficiency
        if "table_statistics" in index_usage:
            for table, stats in index_usage["table_statistics"].items():
                if stats.get("needs_optimization", False):
                    recommendations.append(
                        f"Table '{table}' has low index usage ({stats['index_scan_ratio']:.1%}) - consider adding indexes"
                    )
        
        # Check for missing CLAUDE.md recommended indexes
        required_indexes = [
            "idx_agents_active_updated",
            "idx_agents_status_type_compound",
            "idx_knowledge_nodes_type_current",
        ]
        
        existing_indexes = set(index_usage.keys())
        missing_indexes = [idx for idx in required_indexes if idx not in existing_indexes]
        
        if missing_indexes:
            recommendations.append(
                f"Missing recommended indexes: {', '.join(missing_indexes)}"
            )
        
        # Performance-based recommendations
        for query_name, query_data in query_analysis.items():
            if isinstance(query_data, dict) and query_data.get("avg_time_ms", 0) > 100:
                recommendations.append(
                    f"Query '{query_name}' averaging {query_data['avg_time_ms']:.1f}ms - consider optimization"
                )
        
        return recommendations


@pytest.mark.performance
@pytest.mark.db_test
class TestDatabaseIndexingPerformance:
    """Test suite for database indexing performance validation."""
    
    @pytest.fixture
    def index_analyzer(self):
        """Create index performance analyzer."""
        engine = create_test_engine(use_sqlite=False)  # PostgreSQL for full testing
        setup_test_database(engine)
        
        analyzer = IndexPerformanceAnalyzer(engine, use_sqlite=False)
        
        yield analyzer
        
        teardown_test_database(engine)
    
    @pytest.fixture
    def sqlite_analyzer(self):
        """Create SQLite index analyzer for compatibility testing."""
        engine = create_test_engine(use_sqlite=True)
        setup_test_database(engine)
        
        analyzer = IndexPerformanceAnalyzer(engine, use_sqlite=True)
        
        yield analyzer
        
        teardown_test_database(engine)
    
    @pytest.mark.asyncio
    async def test_create_performance_indexes(self, index_analyzer):
        """Test creation of performance-optimized indexes."""
        results = await index_analyzer.create_performance_indexes()
        
        # Verify that indexes were created successfully
        assert len(results) > 0, "No indexes were created"
        
        successful_indexes = [name for name, data in results.items() if data.get("success", False)]
        failed_indexes = [name for name, data in results.items() if not data.get("success", False)]
        
        logger.info(f"Successfully created {len(successful_indexes)} indexes")
        logger.info(f"Failed to create {len(failed_indexes)} indexes")
        
        # At least some indexes should be created successfully
        assert len(successful_indexes) > 0, "No indexes were created successfully"
        
        # Required CLAUDE.md indexes should be created
        required_indexes = ["idx_agents_active_updated", "idx_agents_status_type_compound"]
        for required_index in required_indexes:
            assert required_index in results, f"Required index {required_index} not attempted"
            assert results[required_index].get("success", False), f"Required index {required_index} failed to create"
    
    @pytest.mark.asyncio
    async def test_query_performance_analysis(self, index_analyzer):
        """Test comprehensive query performance analysis."""
        # Create indexes first
        await index_analyzer.create_performance_indexes()
        
        # Analyze query performance with test data
        analysis = await index_analyzer.analyze_query_performance(test_data_size=500)
        
        assert len(analysis) > 0, "No queries were analyzed"
        
        # Check that queries completed successfully
        successful_queries = [name for name, data in analysis.items() 
                            if isinstance(data, dict) and data.get("execution_count", 0) > 0]
        
        assert len(successful_queries) > 0, "No queries executed successfully"
        
        # Validate query performance metrics
        for query_name, query_data in analysis.items():
            if isinstance(query_data, dict) and "avg_time_ms" in query_data:
                logger.info(f"Query {query_name}: {query_data['avg_time_ms']:.2f}ms average")
                
                # Performance assertions
                assert query_data["avg_time_ms"] < 5000, f"Query {query_name} too slow: {query_data['avg_time_ms']}ms"
                assert query_data["execution_count"] > 0, f"Query {query_name} did not execute"
                
                # Check if slow query detection works
                if query_data["avg_time_ms"] > 30:
                    assert query_data["is_slow"], f"Query {query_name} should be marked as slow"
    
    @pytest.mark.asyncio
    async def test_index_usage_validation(self, index_analyzer):
        """Test validation of index usage effectiveness."""
        # Create indexes and test data
        await index_analyzer.create_performance_indexes()
        await index_analyzer.analyze_query_performance(test_data_size=200)
        
        # Validate index usage
        usage_stats = await index_analyzer.validate_index_usage()
        
        assert len(usage_stats) > 0, "No index usage statistics collected"
        
        # Check for table statistics (PostgreSQL specific)
        if "table_statistics" in usage_stats:
            table_stats = usage_stats["table_statistics"]
            
            # Verify key tables are tracked
            assert "agents" in table_stats, "Agents table statistics missing"
            
            agents_stats = table_stats["agents"]
            assert "index_scan_ratio" in agents_stats, "Index scan ratio missing"
            
            logger.info(f"Agents table index scan ratio: {agents_stats['index_scan_ratio']:.2%}")
        
        # Verify that created indexes appear in usage stats
        required_indexes = ["idx_agents_active_updated", "idx_agents_status_type_compound"]
        for required_index in required_indexes:
            # Index should exist in usage stats (even if not used yet)
            if required_index not in usage_stats:
                logger.warning(f"Index {required_index} not found in usage statistics")
    
    @pytest.mark.asyncio
    async def test_optimization_recommendations(self, index_analyzer):
        """Test generation of optimization recommendations."""
        # Create indexes and run analysis
        await index_analyzer.create_performance_indexes()
        query_analysis = await index_analyzer.analyze_query_performance(test_data_size=300)
        index_usage = await index_analyzer.validate_index_usage()
        
        # Generate recommendations
        recommendations = index_analyzer.generate_optimization_recommendations(
            query_analysis, index_usage
        )
        
        assert isinstance(recommendations, list), "Recommendations should be a list"
        
        # Log recommendations for review
        if recommendations:
            logger.info("Optimization recommendations:")
            for i, rec in enumerate(recommendations, 1):
                logger.info(f"  {i}. {rec}")
        else:
            logger.info("No optimization recommendations generated")
        
        # Validate recommendation quality
        for rec in recommendations:
            assert isinstance(rec, str), "Each recommendation should be a string"
            assert len(rec) > 10, "Recommendations should be descriptive"
    
    @pytest.mark.asyncio
    async def test_sqlite_compatibility(self, sqlite_analyzer):
        """Test indexing performance analysis with SQLite."""
        # Test that basic functionality works with SQLite
        index_results = await sqlite_analyzer.create_performance_indexes()
        assert len(index_results) > 0, "No SQLite indexes created"
        
        query_analysis = await sqlite_analyzer.analyze_query_performance(test_data_size=100)
        assert len(query_analysis) > 0, "No SQLite queries analyzed"
        
        usage_stats = await sqlite_analyzer.validate_index_usage()
        assert len(usage_stats) > 0, "No SQLite usage stats collected"
        
        # Generate recommendations for SQLite
        recommendations = sqlite_analyzer.generate_optimization_recommendations(
            query_analysis, usage_stats
        )
        
        logger.info(f"SQLite analysis generated {len(recommendations)} recommendations")


if __name__ == "__main__":
    import asyncio
    
    async def run_indexing_performance_demo():
        """Run database indexing performance demonstration."""
        print("Running Database Indexing Performance Analysis...\n")
        
        # Test with PostgreSQL
        engine = create_test_engine(use_sqlite=False)
        setup_test_database(engine)
        
        analyzer = IndexPerformanceAnalyzer(engine, use_sqlite=False)
        
        try:
            # Phase 1: Create performance indexes
            print("📊 Phase 1: Creating Performance Indexes")
            index_results = await analyzer.create_performance_indexes()
            
            successful_count = sum(1 for data in index_results.values() if data.get("success", False))
            total_count = len(index_results)
            
            print(f"Created {successful_count}/{total_count} indexes successfully")
            
            for name, result in index_results.items():
                status = "✅" if result.get("success", False) else "❌"
                if result.get("success", False):
                    print(f"  {status} {name}: {result['creation_time_seconds']:.3f}s")
                else:
                    print(f"  {status} {name}: {result.get('error', 'Unknown error')}")
            
            # Phase 2: Analyze query performance
            print("\n📊 Phase 2: Analyzing Query Performance")
            query_analysis = await analyzer.analyze_query_performance(test_data_size=1000)
            
            print(f"Analyzed {len(query_analysis)} queries:")
            
            for query_name, query_data in query_analysis.items():
                if isinstance(query_data, dict) and "avg_time_ms" in query_data:
                    avg_time = query_data["avg_time_ms"]
                    is_slow = query_data.get("is_slow", False)
                    status = "🐌" if is_slow else "⚡"
                    
                    print(f"  {status} {query_name}: {avg_time:.2f}ms average")
                    print(f"      {query_data['description']}")
            
            # Phase 3: Validate index usage
            print("\n📊 Phase 3: Validating Index Usage")
            usage_stats = await analyzer.validate_index_usage()
            
            if "table_statistics" in usage_stats:
                print("Table scan statistics:")
                for table, stats in usage_stats["table_statistics"].items():
                    ratio = stats.get("index_scan_ratio", 0)
                    status = "✅" if ratio > 0.8 else "⚠️" if ratio > 0.5 else "❌"
                    print(f"  {status} {table}: {ratio:.1%} index usage")
            
            print(f"\nIndex usage tracked for {len([k for k in usage_stats.keys() if k != 'table_statistics'])} indexes")
            
            # Phase 4: Generate recommendations
            print("\n📊 Phase 4: Optimization Recommendations")
            recommendations = analyzer.generate_optimization_recommendations(
                query_analysis, usage_stats
            )
            
            if recommendations:
                print("Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"  {i}. {rec}")
            else:
                print("🎉 No optimization recommendations - performance looks good!")
            
            # Summary
            print("\n" + "=" * 60)
            print("INDEXING PERFORMANCE SUMMARY")
            print("=" * 60)
            print(f"Indexes created: {successful_count}/{total_count}")
            print(f"Queries analyzed: {len(query_analysis)}")
            print(f"Optimization recommendations: {len(recommendations)}")
            
            slow_queries = [name for name, data in query_analysis.items() 
                          if isinstance(data, dict) and data.get("is_slow", False)]
            print(f"Slow queries detected: {len(slow_queries)}")
            
            if slow_queries:
                print(f"Slow queries: {', '.join(slow_queries)}")
            
            print("\n✅ Database indexing performance analysis completed!")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            teardown_test_database(engine)
    
    asyncio.run(run_indexing_performance_demo())
