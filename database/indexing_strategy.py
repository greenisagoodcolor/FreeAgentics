"""
Database Indexing Strategy for Multi-Agent Systems.

Implements intelligent indexing strategies including:
- Automatic index recommendation based on query patterns
- Index usage monitoring and optimization
- Partitioning strategies for time-series data
- Composite index design for multi-column queries
- Index maintenance scheduling
"""

import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class IndexUsageMonitor:
    """Monitors index usage and recommends optimizations."""

    def __init__(self):
        """Initialize the index usage monitor."""
        self.index_stats: Dict[str, Dict[str, Any]] = {}
        self.missing_index_recommendations: List[Dict[str, Any]] = []
        self.redundant_indexes: List[str] = []
        self.index_maintenance_schedule: Dict[str, datetime] = {}

    async def analyze_index_usage(self, session: AsyncSession) -> Dict[str, Any]:
        """Analyze current index usage patterns."""
        # Query to get index usage statistics
        index_usage_query = text(
            """
            SELECT
                schemaname,
                tablename,
                indexname,
                idx_scan,
                idx_tup_read,
                idx_tup_fetch,
                pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
                CASE
                    WHEN idx_scan = 0 THEN 'UNUSED'
                    WHEN idx_scan < 10 THEN 'RARELY_USED'
                    WHEN idx_scan < 100 THEN 'OCCASIONALLY_USED'
                    ELSE 'FREQUENTLY_USED'
                END as usage_category
            FROM pg_stat_user_indexes
            WHERE schemaname = 'public'
            ORDER BY idx_scan DESC
        """
        )

        result = await session.execute(index_usage_query)
        indexes = result.fetchall()

        usage_report: Dict[str, Any] = {
            "total_indexes": len(indexes),
            "unused_indexes": [],
            "rarely_used_indexes": [],
            "frequently_used_indexes": [],
            "index_details": [],
        }

        for idx in indexes:
            index_info = {
                "schema": idx.schemaname,
                "table": idx.tablename,
                "index": idx.indexname,
                "scans": idx.idx_scan,
                "tuples_read": idx.idx_tup_read,
                "tuples_fetched": idx.idx_tup_fetch,
                "size": idx.index_size,
                "usage_category": idx.usage_category,
            }

            usage_report["index_details"].append(index_info)

            if idx.usage_category == "UNUSED":
                usage_report["unused_indexes"].append(idx.indexname)
            elif idx.usage_category == "RARELY_USED":
                usage_report["rarely_used_indexes"].append(idx.indexname)
            elif idx.usage_category == "FREQUENTLY_USED":
                usage_report["frequently_used_indexes"].append(idx.indexname)

        # Store stats for trend analysis
        for idx in usage_report["index_details"]:
            self.index_stats[idx["index"]] = idx

        return usage_report

    async def find_missing_indexes(self, session: AsyncSession) -> List[Dict[str, Any]]:
        """Identify missing indexes based on query patterns."""
        # Query to find tables with sequential scans
        seq_scan_query = text(
            """
            SELECT
                schemaname,
                tablename,
                seq_scan,
                seq_tup_read,
                idx_scan,
                n_tup_ins + n_tup_upd + n_tup_del as write_activity,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as table_size
            FROM pg_stat_user_tables
            WHERE schemaname = 'public'
                AND seq_scan > 0
                AND seq_scan > COALESCE(idx_scan, 0) * 2  -- More seq scans than index scans
            ORDER BY seq_scan DESC
        """
        )

        result = await session.execute(seq_scan_query)
        tables = result.fetchall()

        recommendations = []

        for table in tables:
            # Analyze which columns are frequently used in WHERE clauses
            column_usage = await self._analyze_column_usage(session, table.tablename)

            if column_usage:
                recommendation = {
                    "table": table.tablename,
                    "reason": f"High sequential scan count ({table.seq_scan}) vs index scans ({table.idx_scan})",
                    "table_size": table.table_size,
                    "suggested_columns": column_usage,
                    "priority": "HIGH" if table.seq_scan > 1000 else "MEDIUM",
                    "estimated_improvement": f"{(table.seq_scan / (table.seq_scan + table.idx_scan)) * 100:.1f}% reduction in seq scans",
                }
                recommendations.append(recommendation)

        self.missing_index_recommendations = recommendations
        return recommendations

    async def _analyze_column_usage(
        self, session: AsyncSession, table_name: str
    ) -> List[str]:
        """Analyze which columns are frequently used in queries."""
        # This is a simplified version - in production, you'd analyze pg_stat_statements
        # For now, return common patterns based on table structure
        column_patterns = {
            "agents": ["status", "template", "last_active", "created_at"],
            "coalitions": ["status", "performance_score", "created_at"],
            "agent_coalition": ["agent_id", "coalition_id", "joined_at"],
            "knowledge_nodes": ["type", "creator_agent_id", "created_at"],
            "knowledge_edges": ["source_id", "target_id", "type"],
        }

        return column_patterns.get(table_name, [])

    async def find_redundant_indexes(
        self, session: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Identify redundant or duplicate indexes."""
        redundant_query = text(
            """
            WITH index_columns AS (
                SELECT
                    n.nspname as schema_name,
                    t.relname as table_name,
                    i.relname as index_name,
                    array_agg(a.attname ORDER BY array_position(ix.indkey, a.attnum)) as columns,
                    ix.indisunique as is_unique,
                    ix.indisprimary as is_primary,
                    pg_size_pretty(pg_relation_size(i.oid)) as index_size
                FROM pg_index ix
                JOIN pg_class t ON t.oid = ix.indrelid
                JOIN pg_class i ON i.oid = ix.indexrelid
                JOIN pg_namespace n ON n.oid = t.relnamespace
                JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey)
                WHERE n.nspname = 'public'
                    AND NOT ix.indisprimary  -- Exclude primary keys
                GROUP BY n.nspname, t.relname, i.relname, ix.indisunique, ix.indisprimary, i.oid
            )
            SELECT
                a.table_name,
                a.index_name as redundant_index,
                b.index_name as covering_index,
                a.columns as redundant_columns,
                b.columns as covering_columns,
                a.index_size
            FROM index_columns a
            JOIN index_columns b ON a.table_name = b.table_name
                AND a.index_name != b.index_name
                AND a.columns <@ b.columns  -- a's columns are subset of b's
            ORDER BY a.table_name, a.index_name
        """
        )

        result = await session.execute(redundant_query)
        redundant = result.fetchall()

        redundant_list = []
        for idx in redundant:
            redundant_info = {
                "table": idx.table_name,
                "redundant_index": idx.redundant_index,
                "covered_by": idx.covering_index,
                "columns": idx.redundant_columns,
                "size": idx.index_size,
                "recommendation": f"DROP INDEX {idx.redundant_index} -- covered by {idx.covering_index}",
            }
            redundant_list.append(redundant_info)
            self.redundant_indexes.append(idx.redundant_index)

        return redundant_list


class PartitioningStrategy:
    """Implements partitioning strategies for time-series data."""

    def __init__(self):
        """Initialize the partitioning strategy."""
        self.partition_config = {
            "agents": {
                "partition_column": "created_at",
                "partition_interval": "monthly",
                "retention_period": 365,  # days
            },
            "performance_metrics": {
                "partition_column": "timestamp",
                "partition_interval": "daily",
                "retention_period": 90,  # days
            },
            "agent_coalition": {
                "partition_column": "joined_at",
                "partition_interval": "monthly",
                "retention_period": 365,  # days
            },
        }

    async def create_partitioned_table(
        self, session: AsyncSession, table_name: str
    ) -> bool:
        """Create a partitioned version of a table."""
        if table_name not in self.partition_config:
            logger.warning(f"No partition config for table: {table_name}")
            return False

        config = self.partition_config[table_name]
        partition_col = config["partition_column"]

        try:
            # Create partitioned table
            create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS {table_name}_partitioned (
                    LIKE {table_name} INCLUDING ALL
                ) PARTITION BY RANGE ({partition_col})
            """

            await session.execute(text(create_table_sql))
            await session.commit()

            # Create initial partitions
            await self._create_partitions(session, table_name, config)

            logger.info(f"Created partitioned table: {table_name}_partitioned")
            return True

        except Exception as e:
            await session.rollback()
            logger.error(f"Failed to create partitioned table: {e}")
            return False

    async def _create_partitions(
        self, session: AsyncSession, table_name: str, config: Dict[str, Any]
    ):
        """Create partitions based on configuration."""
        interval = config["partition_interval"]

        if interval == "daily":
            await self._create_daily_partitions(session, table_name, config)
        elif interval == "monthly":
            await self._create_monthly_partitions(session, table_name, config)
        elif interval == "yearly":
            await self._create_yearly_partitions(session, table_name, config)

    async def _create_monthly_partitions(
        self, session: AsyncSession, table_name: str, config: Dict[str, Any]
    ):
        """Create monthly partitions."""
        current_date = datetime.now()

        # Create partitions for past 3 months and next 3 months
        for i in range(-3, 4):
            partition_date = current_date + timedelta(days=i * 30)
            partition_name = (
                f"{table_name}_partitioned_{partition_date.strftime('%Y_%m')}"
            )

            start_date = partition_date.replace(day=1)
            if partition_date.month == 12:
                end_date = start_date.replace(year=start_date.year + 1, month=1)
            else:
                end_date = start_date.replace(month=start_date.month + 1)

            create_partition_sql = f"""
                CREATE TABLE IF NOT EXISTS {partition_name}
                PARTITION OF {table_name}_partitioned
                FOR VALUES FROM ('{start_date.strftime("%Y-%m-%d")}')
                TO ('{end_date.strftime("%Y-%m-%d")}')
            """

            try:
                await session.execute(text(create_partition_sql))
                await session.commit()
                logger.info(f"Created partition: {partition_name}")
            except Exception as e:
                await session.rollback()
                logger.warning(f"Partition may already exist: {e}")

    async def _create_daily_partitions(
        self, session: AsyncSession, table_name: str, config: Dict[str, Any]
    ):
        """Create daily partitions."""
        current_date = datetime.now()

        # Create partitions for past 7 days and next 7 days
        for i in range(-7, 8):
            partition_date = current_date + timedelta(days=i)
            partition_name = (
                f"{table_name}_partitioned_{partition_date.strftime('%Y_%m_%d')}"
            )

            start_date = partition_date.replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            end_date = start_date + timedelta(days=1)

            create_partition_sql = f"""
                CREATE TABLE IF NOT EXISTS {partition_name}
                PARTITION OF {table_name}_partitioned
                FOR VALUES FROM ('{start_date.strftime("%Y-%m-%d")}')
                TO ('{end_date.strftime("%Y-%m-%d")}')
            """

            try:
                await session.execute(text(create_partition_sql))
                await session.commit()
                logger.info(f"Created daily partition: {partition_name}")
            except Exception as e:
                await session.rollback()
                logger.warning(f"Daily partition may already exist: {e}")

    async def _create_yearly_partitions(
        self, session: AsyncSession, table_name: str, config: Dict[str, Any]
    ):
        """Create yearly partitions."""
        current_date = datetime.now()

        # Create partitions for past year, current year, and next year
        for i in range(-1, 2):
            year = current_date.year + i
            partition_name = f"{table_name}_partitioned_{year}"

            start_date = datetime(year, 1, 1)
            end_date = datetime(year + 1, 1, 1)

            create_partition_sql = f"""
                CREATE TABLE IF NOT EXISTS {partition_name}
                PARTITION OF {table_name}_partitioned
                FOR VALUES FROM ('{start_date.strftime("%Y-%m-%d")}')
                TO ('{end_date.strftime("%Y-%m-%d")}')
            """

            try:
                await session.execute(text(create_partition_sql))
                await session.commit()
                logger.info(f"Created yearly partition: {partition_name}")
            except Exception as e:
                await session.rollback()
                logger.warning(f"Yearly partition may already exist: {e}")

    async def maintain_partitions(self, session: AsyncSession) -> Dict[str, Any]:
        """Maintain partitions by creating new ones and dropping old ones."""
        maintenance_report: Dict[str, Any] = {
            "created_partitions": [],
            "dropped_partitions": [],
            "errors": [],
        }

        for table_name, config in self.partition_config.items():
            try:
                # Create future partitions
                created = await self._create_future_partitions(
                    session, table_name, config
                )
                maintenance_report["created_partitions"].extend(created)

                # Drop old partitions based on retention
                dropped = await self._drop_old_partitions(session, table_name, config)
                maintenance_report["dropped_partitions"].extend(dropped)

            except Exception as e:
                maintenance_report["errors"].append(
                    {"table": table_name, "error": str(e)}
                )

        return maintenance_report

    async def _create_future_partitions(
        self, session: AsyncSession, table_name: str, config: Dict[str, Any]
    ) -> List[str]:
        """Create partitions for future dates."""
        created: List[str] = []
        # Implementation would create partitions for next period
        # This is a placeholder for the actual implementation
        return created

    async def _drop_old_partitions(
        self, session: AsyncSession, table_name: str, config: Dict[str, Any]
    ) -> List[str]:
        """Drop partitions older than retention period."""
        dropped = []
        retention_days = config["retention_period"]
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        # Query to find old partitions
        find_partitions_sql = text(
            """
            SELECT
                child.relname as partition_name,
                pg_get_expr(child.relpartbound, child.oid) as partition_constraint
            FROM pg_inherits
            JOIN pg_class parent ON pg_inherits.inhparent = parent.oid
            JOIN pg_class child ON pg_inherits.inhrelid = child.oid
            WHERE parent.relname = :table_name
        """
        )

        result = await session.execute(
            find_partitions_sql, {"table_name": f"{table_name}_partitioned"}
        )
        partitions = result.fetchall()

        for partition in partitions:
            # Parse partition constraint to determine date range
            # This is simplified - actual implementation would parse the constraint properly
            if self._is_partition_old(partition.partition_constraint, cutoff_date):
                try:
                    await session.execute(
                        text(f"DROP TABLE {partition.partition_name}")
                    )
                    await session.commit()
                    dropped.append(partition.partition_name)
                    logger.info(f"Dropped old partition: {partition.partition_name}")
                except Exception as e:
                    await session.rollback()
                    logger.error(
                        f"Failed to drop partition {partition.partition_name}: {e}"
                    )

        return dropped

    def _is_partition_old(self, constraint: str, cutoff_date: datetime) -> bool:
        """Check if partition is older than cutoff date."""
        # This is a simplified check - actual implementation would parse the constraint
        # and extract the date range to compare with cutoff_date
        return False  # Placeholder


class CompositeIndexDesigner:
    """Designs optimal composite indexes based on query patterns."""

    def __init__(self):
        """Initialize the composite index designer."""
        self.query_patterns: Dict[str, List[List[str]]] = defaultdict(list)
        self.composite_recommendations: List[Dict[str, Any]] = []

    def analyze_query_pattern(
        self, table: str, columns: List[str], query_type: str = "SELECT"
    ):
        """Analyze query patterns to recommend composite indexes."""
        # Store query pattern
        self.query_patterns[table].append(columns)

        # Analyze for composite index opportunities
        if len(columns) > 1:
            self._analyze_composite_opportunity(table, columns, query_type)

    def _analyze_composite_opportunity(
        self, table: str, columns: List[str], query_type: str
    ):
        """Analyze if a composite index would be beneficial."""
        # Count how often this column combination appears
        pattern_count = sum(
            1 for pattern in self.query_patterns[table] if set(pattern) == set(columns)
        )

        if pattern_count >= 3:  # Threshold for recommendation
            recommendation = {
                "table": table,
                "columns": columns,
                "query_type": query_type,
                "frequency": pattern_count,
                "index_name": f"idx_{table}_{'_'.join(columns)}",
                "recommendation": self._get_column_order_recommendation(table, columns),
            }

            # Avoid duplicate recommendations
            if not any(r["columns"] == columns for r in self.composite_recommendations):
                self.composite_recommendations.append(recommendation)

    def _get_column_order_recommendation(self, table: str, columns: List[str]) -> str:
        """Recommend optimal column order for composite index."""
        # General rules for column ordering:
        # 1. Equality conditions before range conditions
        # 2. Most selective columns first
        # 3. Sort columns last

        column_selectivity = {
            # Estimated selectivity (lower is more selective)
            "id": 1,
            "status": 10,
            "type": 20,
            "template": 30,
            "created_at": 100,
            "updated_at": 100,
            "score": 50,
            "name": 40,
        }

        # Sort columns by selectivity
        ordered_columns = sorted(columns, key=lambda c: column_selectivity.get(c, 50))

        return f"CREATE INDEX CONCURRENTLY idx_{table}_{'_'.join(ordered_columns)} ON {table} ({', '.join(ordered_columns)})"

    def get_composite_index_recommendations(self) -> List[Dict[str, Any]]:
        """Get all composite index recommendations."""
        # Sort by frequency
        return sorted(
            self.composite_recommendations,
            key=lambda x: x["frequency"],
            reverse=True,
        )


class IndexMaintenanceScheduler:
    """Schedules and manages index maintenance operations."""

    def __init__(self):
        """Initialize the index maintenance scheduler."""
        self.maintenance_tasks = {
            "REINDEX": {"interval_days": 30, "last_run": None},
            "ANALYZE": {"interval_days": 1, "last_run": None},
            "VACUUM": {"interval_days": 7, "last_run": None},
            "CHECK_BLOAT": {"interval_days": 7, "last_run": None},
        }

    async def get_maintenance_schedule(
        self, session: AsyncSession
    ) -> List[Dict[str, Any]]:
        """Get upcoming maintenance tasks."""
        schedule = []
        current_time = datetime.now()

        for task_name, task_config in self.maintenance_tasks.items():
            last_run = task_config.get("last_run")
            interval = timedelta(days=task_config["interval_days"])

            if last_run is None or current_time - last_run > interval:
                schedule.append(
                    {
                        "task": task_name,
                        "priority": "HIGH" if last_run is None else "NORMAL",
                        "overdue_by": (
                            (current_time - (last_run + interval)).days
                            if last_run
                            else 0
                        ),
                        "recommended_time": self._get_recommended_time(task_name),
                    }
                )

        return sorted(schedule, key=lambda x: x["priority"] == "HIGH", reverse=True)

    def _get_recommended_time(self, task_name: str) -> str:
        """Get recommended time for maintenance task."""
        # Schedule heavy operations during off-peak hours
        heavy_tasks = ["REINDEX", "VACUUM"]

        if task_name in heavy_tasks:
            return "02:00-04:00 UTC (off-peak)"
        else:
            return "Any time (low impact)"

    async def check_index_bloat(self, session: AsyncSession) -> List[Dict[str, Any]]:
        """Check for index bloat that requires maintenance."""
        bloat_query = text(
            """
            SELECT
                schemaname,
                tablename,
                indexname,
                pg_size_pretty(real_size) as real_size,
                pg_size_pretty(extra_size) as bloat_size,
                round(100 * extra_ratio)::text || '%' as bloat_ratio
            FROM (
                SELECT
                    schemaname, tablename, indexname,
                    pg_relation_size(indexrelid) as real_size,
                    pg_relation_size(indexrelid) -
                    (pg_relation_size(indexrelid) * (100 - avg_leaf_density) / 100) as extra_size,
                    (100 - avg_leaf_density) as extra_ratio
                FROM (
                    SELECT
                        schemaname,
                        tablename,
                        indexname,
                        indexrelid,
                        100 * (1 - avg_leaf_density / 100.0) as avg_leaf_density
                    FROM pg_stat_user_indexes
                    JOIN pgstatindex(indexrelid) ON true
                    WHERE schemaname = 'public'
                ) index_density
            ) index_bloat
            WHERE extra_ratio > 20  -- More than 20% bloat
            ORDER BY extra_size DESC
        """
        )

        try:
            result = await session.execute(bloat_query)
            bloated_indexes = []

            for idx in result.fetchall():
                bloated_indexes.append(
                    {
                        "schema": idx.schemaname,
                        "table": idx.tablename,
                        "index": idx.indexname,
                        "size": idx.real_size,
                        "bloat_size": idx.bloat_size,
                        "bloat_ratio": idx.bloat_ratio,
                        "recommendation": f"REINDEX INDEX CONCURRENTLY {idx.indexname}",
                    }
                )

            return bloated_indexes

        except Exception as e:
            logger.error(f"Failed to check index bloat: {e}")
            return []

    async def perform_maintenance(
        self,
        session: AsyncSession,
        task_name: str,
        target_tables: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Perform a maintenance task."""
        result = {
            "task": task_name,
            "started_at": datetime.now(),
            "completed_at": None,
            "success": False,
            "details": [],
        }

        try:
            if task_name == "ANALYZE":
                await self._perform_analyze(session, target_tables)
            elif task_name == "VACUUM":
                await self._perform_vacuum(session, target_tables)
            elif task_name == "REINDEX":
                await self._perform_reindex(session, target_tables)
            elif task_name == "CHECK_BLOAT":
                bloat_info = await self.check_index_bloat(session)
                result["details"] = bloat_info

            result["success"] = True
            self.maintenance_tasks[task_name]["last_run"] = datetime.now()

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Maintenance task {task_name} failed: {e}")

        result["completed_at"] = datetime.now()
        return result

    async def _perform_analyze(
        self, session: AsyncSession, tables: Optional[List[str]] = None
    ):
        """Perform ANALYZE on tables."""
        if tables is None:
            tables = [
                "agents",
                "coalitions",
                "agent_coalition",
                "knowledge_nodes",
                "knowledge_edges",
            ]

        for table in tables:
            await session.execute(text(f"ANALYZE {table}"))
            await session.commit()
            logger.info(f"Analyzed table: {table}")

    async def _perform_vacuum(
        self, session: AsyncSession, tables: Optional[List[str]] = None
    ):
        """Perform VACUUM on tables."""
        if tables is None:
            tables = ["agents", "coalitions", "agent_coalition"]

        for table in tables:
            await session.execute(text(f"VACUUM (ANALYZE) {table}"))
            await session.commit()
            logger.info(f"Vacuumed table: {table}")

    async def _perform_reindex(
        self, session: AsyncSession, tables: Optional[List[str]] = None
    ):
        """Perform REINDEX on tables."""
        if tables is None:
            # Get all indexes that need reindexing
            bloated = await self.check_index_bloat(session)
            indexes = [idx["index"] for idx in bloated]
        else:
            # Reindex all indexes on specified tables
            indexes = []
            for table in tables:
                idx_query = text(
                    """
                    SELECT indexname
                    FROM pg_indexes
                    WHERE schemaname = 'public' AND tablename = :table
                """
                )
                result = await session.execute(idx_query, {"table": table})
                indexes.extend([row.indexname for row in result])

        for index in indexes:
            try:
                await session.execute(text(f"REINDEX INDEX CONCURRENTLY {index}"))
                await session.commit()
                logger.info(f"Reindexed: {index}")
            except Exception as e:
                await session.rollback()
                logger.error(f"Failed to reindex {index}: {e}")


class IndexingStrategy:
    """Main class coordinating all indexing strategies."""

    def __init__(self):
        """Initialize the indexing strategy coordinator."""
        self.usage_monitor = IndexUsageMonitor()
        self.partitioning = PartitioningStrategy()
        self.composite_designer = CompositeIndexDesigner()
        self.maintenance_scheduler = IndexMaintenanceScheduler()

    async def generate_indexing_report(self, session: AsyncSession) -> Dict[str, Any]:
        """Generate comprehensive indexing strategy report."""
        logger.info("Generating comprehensive indexing report...")

        report = {
            "timestamp": datetime.now().isoformat(),
            "index_usage": await self.usage_monitor.analyze_index_usage(session),
            "missing_indexes": await self.usage_monitor.find_missing_indexes(session),
            "redundant_indexes": await self.usage_monitor.find_redundant_indexes(
                session
            ),
            "composite_recommendations": self.composite_designer.get_composite_index_recommendations(),
            "maintenance_schedule": await self.maintenance_scheduler.get_maintenance_schedule(
                session
            ),
            "index_bloat": await self.maintenance_scheduler.check_index_bloat(session),
        }

        # Generate SQL script for recommendations
        report["optimization_script"] = self._generate_optimization_script(report)

        return report

    def _generate_optimization_script(self, report: Dict[str, Any]) -> str:
        """Generate SQL script for all recommendations."""
        script_lines = [
            "-- Auto-generated Index Optimization Script",
            f"-- Generated at: {datetime.now().isoformat()}",
            "-- Review each recommendation before executing",
            "",
            "-- Missing Indexes",
        ]

        # Add missing index recommendations
        for rec in report["missing_indexes"]:
            script_lines.append(f"-- Table: {rec['table']} - {rec['reason']}")
            for col in rec["suggested_columns"]:
                script_lines.append(
                    f"CREATE INDEX CONCURRENTLY idx_{rec['table']}_{col} ON {rec['table']} ({col});"
                )
            script_lines.append("")

        # Add composite index recommendations
        script_lines.append("-- Composite Indexes")
        for rec in report["composite_recommendations"]:
            script_lines.append(f"-- Frequency: {rec['frequency']} queries")
            script_lines.append(rec["recommendation"] + ";")
            script_lines.append("")

        # Add redundant index removals
        script_lines.append("-- Redundant Indexes to Remove")
        for idx in report["redundant_indexes"]:
            script_lines.append(f"{idx['recommendation']};")

        # Add maintenance recommendations
        script_lines.append("")
        script_lines.append("-- Maintenance Operations")
        for idx in report["index_bloat"]:
            script_lines.append(f"{idx['recommendation']};")

        return "\n".join(script_lines)

    async def apply_recommendations(
        self, session: AsyncSession, auto_approve: bool = False
    ) -> Dict[str, Any]:
        """Apply indexing recommendations with optional auto-approval."""
        results: Dict[str, Any] = {
            "created_indexes": [],
            "dropped_indexes": [],
            "errors": [],
            "skipped": [],
        }

        report = await self.generate_indexing_report(session)

        # Apply missing indexes
        for rec in report["missing_indexes"]:
            if not auto_approve and rec["priority"] != "HIGH":
                results["skipped"].append(
                    f"Skipped {rec['table']} index (requires approval)"
                )
                continue

            for col in rec["suggested_columns"]:
                index_name = f"idx_{rec['table']}_{col}"
                try:
                    await session.execute(
                        text(
                            f"CREATE INDEX CONCURRENTLY {index_name} ON {rec['table']} ({col})"
                        )
                    )
                    await session.commit()
                    results["created_indexes"].append(index_name)
                    logger.info(f"Created index: {index_name}")
                except Exception as e:
                    await session.rollback()
                    results["errors"].append(f"Failed to create {index_name}: {str(e)}")

        # Drop redundant indexes
        for idx in report["redundant_indexes"]:
            if not auto_approve:
                results["skipped"].append(
                    f"Skipped dropping {idx['redundant_index']} (requires approval)"
                )
                continue

            try:
                await session.execute(text(f"DROP INDEX {idx['redundant_index']}"))
                await session.commit()
                results["dropped_indexes"].append(idx["redundant_index"])
                logger.info(f"Dropped redundant index: {idx['redundant_index']}")
            except Exception as e:
                await session.rollback()
                results["errors"].append(
                    f"Failed to drop {idx['redundant_index']}: {str(e)}"
                )

        return results


# Global instance
_indexing_strategy: Optional[IndexingStrategy] = None


def get_indexing_strategy() -> IndexingStrategy:
    """Get or create global indexing strategy instance."""
    global _indexing_strategy

    if _indexing_strategy is None:
        _indexing_strategy = IndexingStrategy()

    return _indexing_strategy
