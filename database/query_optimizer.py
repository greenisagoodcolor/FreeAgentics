"""
Enhanced Database Query Optimizer for Multi-Agent Systems.

Implements advanced PostgreSQL optimization techniques including:
- Query plan analysis with EXPLAIN ANALYZE
- Prepared statement management
- Connection pooling with pgbouncer support
- Batch operations for bulk inserts/updates
- Query result caching with TTL
- Automatic slow query detection and optimization
"""

import hashlib
import json
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from sqlalchemy import (
    create_engine,
    event,
    text,
)
from sqlalchemy.engine import Result
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool, QueuePool

logger = logging.getLogger(__name__)


class PreparedStatementManager:
    """Manages prepared statements for frequently used queries."""

    def __init__(self):
        """Initialize the prepared statement manager."""
        self.prepared_statements: Dict[str, Dict[str, Any]] = {}
        self.statement_usage: Dict[str, int] = defaultdict(int)
        self.statement_performance: Dict[str, List[float]] = defaultdict(list)

    def register_statement(
        self, name: str, sql: str, params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Register a prepared statement."""
        if name not in self.prepared_statements:
            # Create prepared statement name with hash for uniqueness
            stmt_hash = hashlib.md5(sql.encode(), usedforsecurity=False).hexdigest()[:8]
            prepared_name = f"stmt_{name}_{stmt_hash}"

            # Store the prepared statement
            self.prepared_statements[name] = {
                "name": prepared_name,
                "sql": sql,
                "params": params or {},
                "created_at": datetime.now(),
            }

            logger.info(f"Registered prepared statement: {name} -> {prepared_name}")

        return str(self.prepared_statements[name]["name"])

    def get_statement(self, name: str) -> Optional[Dict[str, Any]]:
        """Get prepared statement details."""
        self.statement_usage[name] += 1
        return self.prepared_statements.get(name)

    def track_performance(self, name: str, execution_time: float):
        """Track statement execution performance."""
        self.statement_performance[name].append(execution_time)

        # Keep only last 100 measurements
        if len(self.statement_performance[name]) > 100:
            self.statement_performance[name].pop(0)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all statements."""
        stats = {}
        for name, times in self.statement_performance.items():
            if times:
                stats[name] = {
                    "usage_count": self.statement_usage[name],
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "p95_time": (
                        sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else times[0]
                    ),
                }
        return stats


class QueryPlanAnalyzer:
    """Analyzes query execution plans for optimization opportunities."""

    def __init__(self):
        """Initialize the query plan analyzer."""
        self.slow_query_threshold = 0.1  # 100ms
        self.query_plans: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.optimization_suggestions: Dict[str, Set[str]] = defaultdict(set)

    async def analyze_query(
        self,
        session: AsyncSession,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze query execution plan using EXPLAIN ANALYZE."""
        try:
            # Prepare EXPLAIN ANALYZE query
            explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"

            # Execute with parameters if provided
            if params:
                result = await session.execute(text(explain_query), params)
            else:
                result = await session.execute(text(explain_query))

            # Get the execution plan
            plan_data = result.scalar()
            if isinstance(plan_data, str):
                plan_data = json.loads(plan_data)

            # Extract key metrics
            plan_summary = self._extract_plan_metrics(
                plan_data[0] if isinstance(plan_data, list) else plan_data
            )

            # Store the plan for history
            query_hash = hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()
            self.query_plans[query_hash].append(
                {
                    "timestamp": datetime.now(),
                    "plan": plan_summary,
                    "query": query[:100],  # Store first 100 chars for reference
                }
            )

            # Analyze for optimization opportunities
            self._analyze_for_optimizations(query_hash, plan_summary)

            return plan_summary

        except Exception as e:
            logger.error(f"Failed to analyze query plan: {e}")
            return {"error": str(e)}

    def _extract_plan_metrics(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from execution plan."""
        metrics = {
            "total_cost": plan.get("Total Cost", 0),
            "execution_time": plan.get("Execution Time", 0),
            "planning_time": plan.get("Planning Time", 0),
            "shared_buffers_hit": plan.get("Shared Buffers Hit", 0),
            "shared_buffers_read": plan.get("Shared Buffers Read", 0),
            "node_types": [],
            "index_scans": 0,
            "seq_scans": 0,
            "nested_loops": 0,
            "hash_joins": 0,
            "sort_operations": 0,
        }

        # Recursively analyze plan nodes
        self._analyze_plan_node(plan.get("Plan", {}), metrics)

        return metrics

    def _analyze_plan_node(self, node: Dict[str, Any], metrics: Dict[str, Any]):
        """Recursively analyze plan nodes."""
        node_type = node.get("Node Type", "")
        metrics["node_types"].append(node_type)

        # Count specific operations
        if "Index Scan" in node_type:
            metrics["index_scans"] += 1
        elif "Seq Scan" in node_type:
            metrics["seq_scans"] += 1
        elif "Nested Loop" in node_type:
            metrics["nested_loops"] += 1
        elif "Hash Join" in node_type:
            metrics["hash_joins"] += 1
        elif "Sort" in node_type:
            metrics["sort_operations"] += 1

        # Analyze child nodes
        for child in node.get("Plans", []):
            self._analyze_plan_node(child, metrics)

    def _analyze_for_optimizations(self, query_hash: str, metrics: Dict[str, Any]):
        """Analyze metrics for optimization opportunities."""
        suggestions = self.optimization_suggestions[query_hash]

        # Check for missing indexes
        if metrics["seq_scans"] > 0 and metrics["index_scans"] == 0:
            suggestions.add("Consider adding indexes - query uses sequential scans only")

        # Check for excessive nested loops
        if metrics["nested_loops"] > 3:
            suggestions.add("High nested loop count - consider query restructuring")

        # Check for sort operations that could use indexes
        if metrics["sort_operations"] > 0:
            suggestions.add("Sort operations detected - consider adding sorted indexes")

        # Check buffer hit ratio
        total_buffers = metrics["shared_buffers_hit"] + metrics["shared_buffers_read"]
        if total_buffers > 0:
            hit_ratio = metrics["shared_buffers_hit"] / total_buffers
            if hit_ratio < 0.9:
                suggestions.add(
                    f"Low buffer hit ratio ({hit_ratio:.2%}) - consider increasing shared_buffers"
                )

        # Check execution time
        if metrics["execution_time"] > self.slow_query_threshold * 1000:  # Convert to ms
            suggestions.add(f"Slow query detected ({metrics['execution_time']:.1f}ms)")


class BatchOperationManager:
    """Manages batch operations for efficient bulk inserts and updates."""

    def __init__(self, batch_size: int = 1000):
        """Initialize the batch operation manager.

        Args:
            batch_size: Maximum number of operations per batch.
        """
        self.batch_size = batch_size
        self.pending_inserts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.pending_updates: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.batch_performance: Dict[str, List[float]] = defaultdict(list)

    def _escape_identifier(self, identifier: str) -> str:
        """Safely escape SQL identifiers to prevent injection.

        Args:
            identifier: The table or column name to escape

        Returns:
            Safely quoted identifier
        """
        # Remove any potentially dangerous characters and quote the identifier
        # Only allow alphanumeric characters, underscores, and dots for schema.table notation
        import re

        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)?$", identifier):
            raise ValueError(f"Invalid identifier: {identifier}")
        return f'"{identifier}"'

    async def batch_insert(
        self,
        session: AsyncSession,
        table_name: str,
        records: List[Dict[str, Any]],
    ) -> int:
        """Perform batch insert with optimal chunk size."""
        if not records:
            return 0

        start_time = time.time()
        inserted_count = 0

        try:
            # Process in chunks
            for i in range(0, len(records), self.batch_size):
                chunk = records[i : i + self.batch_size]

                # Build INSERT statement with ON CONFLICT DO NOTHING
                columns = list(chunk[0].keys())

                # Safely escape table name and column names
                escaped_table = self._escape_identifier(table_name)
                escaped_columns = [self._escape_identifier(col) for col in columns]

                values_clause = ", ".join(
                    [
                        "(" + ", ".join([f":{col}_{j}" for col in columns]) + ")"
                        for j in range(len(chunk))
                    ]
                )

                insert_stmt = text(
                    f"""
                    INSERT INTO {escaped_table} ({", ".join(escaped_columns)})
                    VALUES {values_clause}
                    ON CONFLICT DO NOTHING
                """  # nosec B608
                )

                # Prepare parameters
                params = {}
                for j, record in enumerate(chunk):
                    for col, value in record.items():
                        params[f"{col}_{j}"] = value

                # Execute batch insert
                result: Result[Any] = await session.execute(insert_stmt, params)
                inserted_count += getattr(result, "rowcount", 0)

            await session.commit()

            # Track performance
            execution_time = time.time() - start_time
            self.batch_performance[f"insert_{table_name}"].append(execution_time)

            logger.info(
                f"Batch inserted {inserted_count} records into {table_name} in {execution_time:.3f}s"
            )
            return inserted_count

        except Exception as e:
            await session.rollback()
            logger.error(f"Batch insert failed: {e}")
            raise

    async def batch_update(
        self,
        session: AsyncSession,
        table_name: str,
        updates: List[Dict[str, Any]],
        key_column: str = "id",
    ) -> int:
        """Perform batch update using CASE statements for efficiency."""
        if not updates:
            return 0

        start_time = time.time()
        updated_count = 0

        try:
            # Group updates by columns being updated
            update_groups = defaultdict(list)
            for update in updates:
                key_value = update.pop(key_column)
                update_cols = frozenset(update.keys())
                update_groups[update_cols].append({key_column: key_value, **update})

            # Process each group
            for columns, group_updates in update_groups.items():
                if not group_updates:
                    continue

                # Safely escape table name and column names
                escaped_table = self._escape_identifier(table_name)
                escaped_key_column = self._escape_identifier(key_column)

                # Build UPDATE statement with CASE
                set_clauses = []
                for col in columns:
                    escaped_col = self._escape_identifier(col)
                    case_clause = f"{escaped_col} = CASE {escaped_key_column}"
                    for update in group_updates:
                        case_clause += f" WHEN :{key_column}_{id(update)} THEN :{col}_{id(update)}"
                    case_clause += f" ELSE {escaped_col} END"
                    set_clauses.append(case_clause)

                # Build WHERE clause
                where_values = [f":{key_column}_{id(u)}" for u in group_updates]
                where_clause = f"{escaped_key_column} IN ({', '.join(where_values)})"

                update_stmt = text(
                    f"""
                    UPDATE {escaped_table}
                    SET {", ".join(set_clauses)}, updated_at = NOW()
                    WHERE {where_clause}
                """  # nosec B608
                )

                # Prepare parameters
                params = {}
                for update in group_updates:
                    update_id = id(update)
                    for col, value in update.items():
                        params[f"{col}_{update_id}"] = value

                # Execute batch update
                result: Result[Any] = await session.execute(update_stmt, params)
                updated_count += getattr(result, "rowcount", 0)

            await session.commit()

            # Track performance
            execution_time = time.time() - start_time
            self.batch_performance[f"update_{table_name}"].append(execution_time)

            logger.info(
                f"Batch updated {updated_count} records in {table_name} in {execution_time:.3f}s"
            )
            return updated_count

        except Exception as e:
            await session.rollback()
            logger.error(f"Batch update failed: {e}")
            raise

    def add_pending_insert(self, table_name: str, record: Dict[str, Any]):
        """Add record to pending inserts buffer."""
        self.pending_inserts[table_name].append(record)

    def add_pending_update(self, table_name: str, update: Dict[str, Any]):
        """Add update to pending updates buffer."""
        self.pending_updates[table_name].append(update)

    async def flush_pending_operations(self, session: AsyncSession) -> Dict[str, int]:
        """Flush all pending operations."""
        results = {}

        # Flush inserts
        for table_name, records in self.pending_inserts.items():
            if records:
                count = await self.batch_insert(session, table_name, records)
                results[f"insert_{table_name}"] = count
                self.pending_inserts[table_name].clear()

        # Flush updates
        for table_name, updates in self.pending_updates.items():
            if updates:
                count = await self.batch_update(session, table_name, updates)
                results[f"update_{table_name}"] = count
                self.pending_updates[table_name].clear()

        return results


class EnhancedQueryOptimizer:
    """Enhanced query optimizer with all advanced features."""

    def __init__(self, database_url: str, enable_pgbouncer: bool = True):
        """Initialize the query optimizer.

        Args:
            database_url: PostgreSQL connection URL.
            enable_pgbouncer: Whether to enable PgBouncer connection pooling.
        """
        self.database_url = database_url
        self.enable_pgbouncer = enable_pgbouncer

        # Initialize components
        self.prepared_statements = PreparedStatementManager()
        self.query_analyzer = QueryPlanAnalyzer()
        self.batch_manager = BatchOperationManager()

        # Query cache with TTL
        self.query_cache: Dict[str, Tuple[Any, float]] = {}
        self.cache_ttl = 300  # 5 minutes default

        # Performance tracking
        self.query_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {
                "count": 0,
                "total_time": 0,
                "errors": 0,
                "cache_hits": 0,
                "slow_queries": [],
            }
        )

        # Connection pool configuration
        self.pool_config = self._get_pool_config()

        # Create engines
        self._engine = None
        self._async_engine = None

    def _get_pool_config(self) -> Dict[str, Any]:
        """Get optimized connection pool configuration."""
        if self.enable_pgbouncer:
            # PgBouncer compatible settings (no pool config for NullPool)
            return {
                "poolclass": NullPool,
                "pool_pre_ping": True,
                "pool_recycle": 900,  # 15 minutes
                "connect_args": {
                    "server_settings": {"jit": "off"},  # Disable JIT for connection pooling
                    "command_timeout": 60,
                    "prepared_statement_cache_size": 0,  # Disable with PgBouncer
                },
            }
        else:
            # Direct connection settings
            return {
                "poolclass": QueuePool,
                "pool_size": 10,
                "max_overflow": 20,
                "pool_pre_ping": True,
                "pool_recycle": 3600,  # 1 hour
                "pool_timeout": 30,
                "connect_args": {
                    "server_settings": {
                        "jit": "on",
                        "random_page_cost": "1.1",  # SSD optimized
                        "effective_cache_size": "4GB",
                        "shared_buffers": "256MB",
                        "work_mem": "16MB",
                    },
                    "prepared_statement_cache_size": 256,
                },
            }

    @property
    def engine(self):
        """Get or create synchronous engine."""
        if self._engine is None:
            config = self.pool_config.copy()
            poolclass = config.pop("poolclass", QueuePool)

            self._engine = create_engine(self.database_url, poolclass=poolclass, **config)

            # Add event listeners for monitoring
            self._setup_engine_monitoring(self._engine)

        return self._engine

    @property
    def async_engine(self):
        """Get or create asynchronous engine."""
        if self._async_engine is None:
            # Convert to async URL
            async_url = self.database_url.replace("postgresql://", "postgresql+asyncpg://")

            config = self.pool_config.copy()
            poolclass = config.pop("poolclass", QueuePool)

            self._async_engine = create_async_engine(async_url, poolclass=poolclass, **config)

        return self._async_engine

    def _setup_engine_monitoring(self, engine):
        """Set up event listeners for query monitoring."""

        @event.listens_for(engine, "before_cursor_execute")
        def before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            conn.info.setdefault("query_start_time", []).append(time.time())
            conn.info.setdefault("current_query", []).append(statement[:100])

        @event.listens_for(engine, "after_cursor_execute")
        def after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            total_time = time.time() - conn.info["query_start_time"].pop(-1)
            query_snippet = conn.info["current_query"].pop(-1)

            # Track query performance
            self._track_query_performance(query_snippet, total_time)

    def _track_query_performance(self, query: str, execution_time: float):
        """Track query performance metrics."""
        query_type = query.split()[0].upper() if query else "UNKNOWN"

        stats: Dict[str, Any] = self.query_stats[query_type]
        stats["count"] += 1
        stats["total_time"] += execution_time

        # Track slow queries
        if execution_time > self.query_analyzer.slow_query_threshold:
            stats["slow_queries"].append(
                {
                    "query": query,
                    "time": execution_time,
                    "timestamp": datetime.now(),
                }
            )

            # Keep only last 10 slow queries per type
            if len(stats["slow_queries"]) > 10:
                stats["slow_queries"].pop(0)

    @asynccontextmanager
    async def optimized_session(self):
        """Get an optimized database session."""
        async with AsyncSession(self.async_engine) as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                logger.error(f"Session error: {e}")
                raise
            finally:
                await session.close()

    async def execute_with_cache(
        self,
        session: AsyncSession,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        cache_key: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> Any:
        """Execute query with caching support."""
        # Generate cache key if not provided
        if cache_key is None:
            param_str = json.dumps(params, sort_keys=True) if params else ""
            cache_key = hashlib.md5(
                f"{query}{param_str}".encode(), usedforsecurity=False
            ).hexdigest()

        # Check cache
        if cache_key in self.query_cache:
            cached_result, cached_time = self.query_cache[cache_key]
            if time.time() - cached_time < (ttl or self.cache_ttl):
                self.query_stats["CACHE"]["cache_hits"] += 1
                return cached_result

        # Execute query
        start_time = time.time()
        try:
            result: Result[Any]
            if params:
                result = await session.execute(text(query), params)
            else:
                result = await session.execute(text(query))

            # Process result based on type
            if query.strip().upper().startswith("SELECT"):
                data: Any = result.fetchall()
            else:
                data = getattr(result, "rowcount", 0)

            # Cache result
            self.query_cache[cache_key] = (data, time.time())

            # Clean old cache entries periodically
            if len(self.query_cache) > 1000:
                self._clean_cache()

            return data

        except Exception as e:
            self.query_stats["ERROR"]["errors"] += 1
            logger.error(f"Query execution failed: {e}")
            raise
        finally:
            execution_time = time.time() - start_time
            self._track_query_performance(query[:100], execution_time)

    def _clean_cache(self):
        """Clean expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key
            for key, (_, cached_time) in self.query_cache.items()
            if current_time - cached_time > self.cache_ttl
        ]

        for key in expired_keys:
            del self.query_cache[key]

    async def create_multi_agent_indexes(self, session: AsyncSession):
        """Create optimized indexes for multi-agent scenarios."""
        indexes = [
            # Agent performance indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_active_inference ON agents (status, inference_count DESC) WHERE status = 'active'",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_coalition_lookup ON agents USING btree (id) INCLUDE (name, status, last_active)",
            # Coalition optimization indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_active_perf ON coalitions (status, performance_score DESC, cohesion_score DESC) WHERE status = 'active'",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_coalitions_date_range ON coalitions (created_at, dissolved_at) WHERE dissolved_at IS NULL",
            # Agent-Coalition relationship indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_coalition_compound ON agent_coalition (coalition_id, agent_id, joined_at DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_coalition_performance ON agent_coalition (contribution_score DESC, trust_score DESC)",
            # Time-series optimization for monitoring
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_time_series ON agents (last_active) WHERE last_active IS NOT NULL",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_metrics_time ON performance_metrics (timestamp DESC, metric_type)",
            # JSON field specific indexes for PostgreSQL
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_beliefs_path ON agents USING gin ((beliefs -> 'state'))",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agents_metrics_value ON agents USING gin ((metrics -> 'performance'))",
            # Partial indexes for common queries
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_recent_active_agents ON agents (last_active DESC) WHERE last_active > NOW() - INTERVAL '1 hour'",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_high_performing_coalitions ON coalitions (performance_score DESC) WHERE performance_score > 0.8",
        ]

        logger.info("Creating multi-agent optimized indexes...")

        for index_sql in indexes:
            try:
                await session.execute(text(index_sql))
                await session.commit()
                logger.info(f"Created index: {index_sql.split('idx_')[1].split(' ')[0]}")
            except Exception as e:
                await session.rollback()
                logger.warning(f"Index creation failed (may already exist): {e}")

    async def analyze_and_vacuum_tables(
        self, session: AsyncSession, tables: Optional[List[str]] = None
    ):
        """Run ANALYZE and VACUUM on tables for optimization."""
        if tables is None:
            tables = [
                "agents",
                "coalitions",
                "agent_coalition",
                "knowledge_nodes",
                "knowledge_edges",
            ]

        for table in tables:
            try:
                # Run ANALYZE to update statistics
                await session.execute(text(f"ANALYZE {table}"))
                logger.info(f"Analyzed table: {table}")

                # Run VACUUM to reclaim space (non-blocking)
                await session.execute(text(f"VACUUM (ANALYZE) {table}"))
                logger.info(f"Vacuumed table: {table}")

                await session.commit()
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to analyze/vacuum {table}: {e}")

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        return {
            "query_statistics": dict(self.query_stats),
            "prepared_statements": self.prepared_statements.get_performance_stats(),
            "batch_operations": {
                table: {
                    "avg_time": sum(times) / len(times) if times else 0,
                    "operations": len(times),
                }
                for table, times in self.batch_manager.batch_performance.items()
            },
            "cache_statistics": {
                "size": len(self.query_cache),
                "hit_rate": self.query_stats["CACHE"]["cache_hits"]
                / max(sum(s["count"] for s in self.query_stats.values()), 1)
                * 100,
            },
            "optimization_suggestions": dict(self.query_analyzer.optimization_suggestions),
            "slow_queries": [
                query
                for stats in self.query_stats.values()
                for query in stats.get("slow_queries", [])
            ][
                :20
            ],  # Top 20 slow queries
        }

    async def setup_monitoring(self, session: AsyncSession):
        """Set up database monitoring and slow query logging."""
        monitoring_configs = [
            # Enable query logging for slow queries
            "ALTER SYSTEM SET log_min_duration_statement = 100",  # Log queries > 100ms
            "ALTER SYSTEM SET log_checkpoints = on",
            "ALTER SYSTEM SET log_connections = on",
            "ALTER SYSTEM SET log_disconnections = on",
            "ALTER SYSTEM SET log_lock_waits = on",
            "ALTER SYSTEM SET log_temp_files = 0",
            # Enable statistics collection
            "ALTER SYSTEM SET track_activities = on",
            "ALTER SYSTEM SET track_counts = on",
            "ALTER SYSTEM SET track_io_timing = on",
            "ALTER SYSTEM SET track_functions = 'all'",
            # Auto-explain for slow queries
            "ALTER SYSTEM SET auto_explain.log_min_duration = 100",
            "ALTER SYSTEM SET auto_explain.log_analyze = on",
            "ALTER SYSTEM SET auto_explain.log_buffers = on",
            "ALTER SYSTEM SET auto_explain.log_timing = on",
        ]

        logger.info("Setting up database monitoring...")

        for config in monitoring_configs:
            try:
                await session.execute(text(config))
                await session.commit()
                logger.info(f"Applied config: {config}")
            except Exception as e:
                await session.rollback()
                logger.warning(f"Failed to apply config (may require superuser): {e}")

        # Reload configuration
        try:
            await session.execute(text("SELECT pg_reload_conf()"))
            await session.commit()
            logger.info("PostgreSQL configuration reloaded")
        except Exception as e:
            logger.warning(f"Failed to reload config: {e}")


# Global instance
_optimizer: Optional[EnhancedQueryOptimizer] = None


def get_query_optimizer(
    database_url: Optional[str] = None, enable_pgbouncer: bool = True
) -> EnhancedQueryOptimizer:
    """Get or create global query optimizer instance."""
    global _optimizer

    if _optimizer is None:
        if database_url is None:
            raise ValueError("Database URL required for first initialization")
        _optimizer = EnhancedQueryOptimizer(database_url, enable_pgbouncer)

    return _optimizer
